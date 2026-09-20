#!/usr/bin/env python3
"""Collect leader-arm gravity identification data by autonomous MIT replay.

    *** WITH --execute THE LEADER (right/secondary) ARM MOVES ON ITS OWN. ***

Nothing is identified here; this only records. The fit is a separate offline
step that regresses MEASURED motor torque (current x torque constant) against
MEASURED joint state, so the MIT tracking torque is simply part of what the
motor is seen to supply and the coarse DEFAULT-firmware t_ff encoding does not
bias the data.

Sequence
--------
1. Enable into a position hold at the current pose, then home at
   --start-home-speed (the validated single-arm hold/home routine, kp/kd as
   used on the leader by the paired MIT home).
2. Static poses (``--poses``, default the 18 leader calibration poses already
   visited by identify_gravity.py): forward order, then reverse order, straight
   joint-space quintic moves between them, settle, then a measurement window.
3. Dataset replay (``--episodes N``; default 0 = skipped): N episodes of
   tight_insertion_whole_v1 chosen by farthest-point sampling. Each is entered
   from home, played forward, held, played backward, held, and left to home.
   The replayed path is the recorded ``action`` (the leader's own teleop pose).
   Time is scaled by --time-scale and any step faster than --replay-speed-cap
   is stretched further.
4. Home, then keep holding in MIT.

Read-only unless --execute: the dry run connects, reads feedback and the
firmware version, builds the whole plan and prints its duration, speeds and
limit margins.

Every 100 Hz tick of every segment is written to ``<out>/segments/*.npz``;
``manifest.jsonl`` indexes them, ``static_summary.csv`` holds the per-pose
averages and ``metadata.json`` the run configuration.

    python tools/piper/leader_gravity_id_collect.py                   # dry run
    python tools/piper/leader_gravity_id_collect.py --execute         # static only
    python tools/piper/leader_gravity_id_collect.py --execute --no-static --episodes 1
    python tools/piper/leader_gravity_id_collect.py --execute --episodes 50
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ruri.client.controllers.single_piper import discovery
from ruri.client.controllers.single_piper.calibration_ranges import measured_for_arm
from ruri.client.controllers.single_piper.hardware_registry import (
    find_arm_by_hardware_id,
    find_arm_registration,
)
from ruri.client.controllers.single_piper.mit import leader_follower as mit
from ruri.client.controllers.single_piper.mit import policy_controller as pc
from ruri.client.controllers.single_piper.normalization import (
    CALIBRATION_RANGES,
    JOINT_NAMES,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_POSES = Path("/home/omen/junshan/piper_teleop_agx/factr/data/leader_poses.csv")
DEFAULT_DATASET = Path("/mnt/data/datasets/tight_insertion_whole_v1")
DEFAULT_OUT_ROOT = Path("/mnt/data/datasets/leader_gravity_id")
LEADER_CALIBRATION = mit.ASSET_ROOT / "calibration_leader.json"
DATASET_FPS = 30.0
R2D = 180.0 / math.pi

# Mirrors pyAgxArm piper/default move_mit so the log carries the integer that
# actually went on the wire alongside the float we asked for.
PIPER_TORQUE_B = np.array([4.0, 4.0, 4.0, 1.0, 1.0, 1.0])

_running = True


def _stop(_sig, _frame):
    global _running
    _running = False
    pc._running = False  # also interrupts the borrowed hold/home routine


def default_tff_code(tau):
    """Integer t_ff code the DEFAULT (<= S-V1.8-2) encoder produces."""
    t = np.clip(np.asarray(tau, float), -8.0 * PIPER_TORQUE_B, 8.0 * PIPER_TORQUE_B)
    return np.array(
        [int((v / b + 8.0) * 255.0 / 16.0) for v, b in zip(t, PIPER_TORQUE_B)],
        dtype=np.int16,
    )


# --------------------------------------------------------------------- plan


@dataclass
class Segment:
    name: str
    kind: str  # transit | settle | measure | replay
    q: np.ndarray  # (n, 6) reference
    qd: np.ndarray  # (n, 6) reference velocity
    info: dict = field(default_factory=dict)


def quintic_segment(name, start, end, speed, rate, info=None, min_seconds=1.0):
    start, end = np.asarray(start, float), np.asarray(end, float)
    duration = max(min_seconds, 1.875 * float(np.max(np.abs(end - start))) / speed)
    n = max(1, int(math.ceil(duration * rate)))
    q = np.empty((n, 6))
    qd = np.empty((n, 6))
    for i in range(n):
        alpha, alpha_dot = mit.quintic_blend((i + 1) / rate, duration)
        q[i] = start + alpha * (end - start)
        qd[i] = alpha_dot * (end - start)
    return Segment(name, "transit", q, qd, dict(info or {}))


def hold_segment(name, kind, pose, seconds, rate, info=None):
    n = max(1, int(round(seconds * rate)))
    return Segment(
        name,
        kind,
        np.repeat(np.asarray(pose, float)[None], n, 0),
        np.zeros((n, 6)),
        dict(info or {}),
    )


def retime(frames, time_scale, speed_cap, rate):
    """Resample 30 Hz frames at ``rate``; slow down, never speed up."""
    frames = np.asarray(frames, float)
    base = (1.0 / DATASET_FPS) / time_scale
    steps = np.max(np.abs(np.diff(frames, axis=0)), axis=1)
    durations = np.maximum(base, steps / speed_cap)
    t = np.concatenate(([0.0], np.cumsum(durations)))
    ticks = np.arange(1.0 / rate, t[-1] + 1e-9, 1.0 / rate)
    q = np.stack([np.interp(ticks, t, frames[:, j]) for j in range(6)], axis=1)
    qd = np.gradient(q, 1.0 / rate, axis=0) if len(q) > 1 else np.zeros_like(q)
    return q, qd


def load_poses(path):
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return np.array([[float(r[f"q{j}"]) for j in range(1, 7)] for r in rows])


def load_dataset_actions(root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted((Path(root) / "data").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}/data")
    table = pa.concat_tables(
        [pq.read_table(f, columns=["action", "episode_index"]) for f in files]
    )
    action = np.array(table["action"].to_pylist(), dtype=float)
    episode = np.array(table["episode_index"].to_pylist())
    # whole_v1 is normalized with the current NOMINAL table; checked against
    # follower gravity torque in all four source parts before this was written.
    lo = np.array([CALIBRATION_RANGES[n][0] for n in JOINT_NAMES])
    hi = np.array([CALIBRATION_RANGES[n][1] for n in JOINT_NAMES])
    q = np.radians((lo + (action[:, :6] + 100.0) / 200.0 * (hi - lo)) / 1000.0)
    return {int(e): q[episode == e] for e in np.unique(episode)}


def select_episodes(episodes, count):
    """Deterministic farthest-point sampling on per-episode mean/std pose."""
    ids = sorted(episodes)
    if count >= len(ids):
        return ids
    feats = np.array(
        [np.r_[episodes[i].mean(0), episodes[i].std(0)] for i in ids]
    )
    z = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    chosen = [int(np.argmax(np.linalg.norm(z, axis=1)))]
    dist = np.linalg.norm(z - z[chosen[0]], axis=1)
    while len(chosen) < count:
        k = int(np.argmax(dist))
        chosen.append(k)
        dist = np.minimum(dist, np.linalg.norm(z - z[k], axis=1))
    return [ids[k] for k in chosen]


def build_plan(args, poses, episodes, episode_ids):
    """Everything after the startup home. The arm starts this plan at zero."""
    rate = args.rate
    home = np.zeros(6)
    plan = []
    here = home
    if poses is not None:
        order = list(range(len(poses)))
        for direction, seq in (("forward", order), ("reverse", order[::-1])):
            for n in seq:
                info = {"pose_index": n, "pass": direction}
                tag = f"static_{direction}_{n:02d}"
                plan.append(
                    quintic_segment(tag + "_transit", here, poses[n], args.transit_speed, rate, info)
                )
                plan.append(hold_segment(tag + "_settle", "settle", poses[n], args.settle, rate, info))
                plan.append(hold_segment(tag + "_measure", "measure", poses[n], args.measure, rate, info))
                here = poses[n]
    for episode_id in episode_ids:
        q, qd = retime(episodes[episode_id], args.time_scale, args.replay_speed_cap, rate)
        tag = f"episode_{episode_id:04d}"
        info = {"episode_index": episode_id}
        plan.append(quintic_segment(tag + "_enter", here, q[0], args.transit_speed, rate, info))
        plan.append(hold_segment(tag + "_start_hold", "settle", q[0], args.replay_hold, rate, info))
        plan.append(Segment(tag + "_forward", "replay", q, qd, {**info, "direction": "forward"}))
        plan.append(hold_segment(tag + "_end_hold", "settle", q[-1], args.replay_hold, rate, info))
        plan.append(
            Segment(tag + "_reverse", "replay", q[::-1].copy(), -qd[::-1], {**info, "direction": "reverse"})
        )
        plan.append(hold_segment(tag + "_return_hold", "settle", q[0], args.replay_hold, rate, info))
        plan.append(quintic_segment(tag + "_leave", q[0], home, args.transit_speed, rate, info))
        here = home
    if np.any(here != home):
        plan.append(quintic_segment("final_home", here, home, args.transit_speed, rate))
    return plan


def describe_plan(plan, rate, arm_name):
    lower = np.maximum(mit.JOINT_LIMIT_LOWER, _measured_bounds(arm_name)[0])
    upper = np.minimum(mit.JOINT_LIMIT_UPPER, _measured_bounds(arm_name)[1])
    total = sum(len(s.q) for s in plan) / rate
    by_kind = {}
    for s in plan:
        by_kind[s.kind] = by_kind.get(s.kind, 0.0) + len(s.q) / rate
    allq = np.concatenate([s.q for s in plan]) if plan else np.zeros((1, 6))
    allqd = np.concatenate([s.qd for s in plan]) if plan else np.zeros((1, 6))
    jumps = [
        float(np.max(np.abs(b.q[0] - a.q[-1]))) for a, b in zip(plan[:-1], plan[1:])
    ]
    print(f"\nplan: {len(plan)} segments, {total / 60:.1f} min")
    for kind, seconds in sorted(by_kind.items()):
        print(f"  {kind:8s} {seconds / 60:6.1f} min")
    print("  reference range (deg) min: " + " ".join(f"{v * R2D:+7.1f}" for v in allq.min(0)))
    print("  reference range (deg) max: " + " ".join(f"{v * R2D:+7.1f}" for v in allq.max(0)))
    print("  |reference speed| max (rad/s): " + " ".join(f"{v:5.2f}" for v in np.abs(allqd).max(0)))
    print(f"  largest step between segments: {max(jumps, default=0.0):.4f} rad")
    below = allq < lower
    above = allq > upper
    clipped = int(np.sum(np.any(below | above, axis=1)))
    print(
        f"  ticks outside nominal/{arm_name} limits (will be clamped): {clipped}"
    )
    return {"total_s": total, "by_kind_s": by_kind, "clamped_ticks": clipped}


def _measured_bounds(arm_name):
    ranges = measured_for_arm(arm_name)
    lo = np.radians([ranges[f"joint{i}"][0] / 1000.0 for i in range(1, 7)])
    hi = np.radians([ranges[f"joint{i}"][1] / 1000.0 for i in range(1, 7)])
    return lo, hi


# ------------------------------------------------------------------ logging


class Recorder:
    FIELDS = (
        "t_ns", "q", "qd", "torque", "current", "q_ref", "qd_ref",
        "tff", "tff_code", "gripper_width", "gripper_force",
    )

    def __init__(self, out):
        self.out = Path(out)
        (self.out / "segments").mkdir(parents=True, exist_ok=False)
        self.manifest = open(self.out / "manifest.jsonl", "a")
        self.static_rows = []
        self.index = 0

    def begin(self):
        self.rows = {k: [] for k in self.FIELDS}
        self.health = []

    def add(self, **values):
        for key in self.FIELDS:
            self.rows[key].append(values[key])

    def end(self, segment, outcome, kp, kd):
        n = len(self.rows["t_ns"])
        if n == 0:
            return
        path = self.out / "segments" / f"{self.index:04d}_{segment.name}.npz"
        arrays = {k: np.asarray(v) for k, v in self.rows.items()}
        arrays["health"] = np.asarray(self.health, dtype=float).reshape(-1, 1 + 6 * 3)
        np.savez_compressed(path, **arrays)
        record = {
            "index": self.index,
            "file": str(path.relative_to(self.out)),
            "segment": segment.name,
            "kind": segment.kind,
            "ticks": n,
            "planned_ticks": len(segment.q),
            "outcome": outcome,
            "kp": kp,
            "kd": kd,
            **segment.info,
        }
        self.manifest.write(json.dumps(record) + "\n")
        self.manifest.flush()
        if segment.kind == "measure" and outcome == "complete":
            torque = arrays["torque"]
            self.static_rows.append(
                {
                    "pose_index": segment.info["pose_index"],
                    "pass": segment.info["pass"],
                    **{f"q{j + 1}": float(v) for j, v in enumerate(arrays["q"].mean(0))},
                    **{f"tau{j + 1}": float(v) for j, v in enumerate(torque.mean(0))},
                    **{f"tau_sd{j + 1}": float(v) for j, v in enumerate(torque.std(0))},
                    "n": n,
                }
            )
        self.index += 1

    def close(self):
        self.manifest.close()
        if self.static_rows:
            with open(self.out / "static_summary.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.static_rows[0]))
                writer.writeheader()
                writer.writerows(self.static_rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def firmware_version(arm, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = arm.get_firmware(timeout=0.5)
        if info is not None:
            return {k: str(v).strip("\x00 ").strip() for k, v in info.items()}
        time.sleep(0.1)
    return None


def require_default_firmware(info):
    """build_arm encodes t_ff for <= S-V1.8-2; anything newer decodes garbage."""
    if info is None:
        raise mit.TeleopAbort("could not read firmware version; refusing MIT")
    match = re.search(r"S-V(\d+)\.(\d+)-(\d+)", info.get("software_version", ""))
    if not match:
        raise mit.TeleopAbort(f"unrecognised firmware {info}; refusing MIT")
    if tuple(int(v) for v in match.groups()) > (1, 8, 2):
        raise mit.TeleopAbort(
            f"firmware {info['software_version']} needs a newer PiperFW than "
            "build_arm uses; refusing MIT"
        )


# --------------------------------------------------------------------- run


def resolve_leader(args):
    registration = find_arm_registration(args.leader_side, args.leader_role)
    device = discovery.discover_piper_can(
        candidates=None if args.leader_can is None else [args.leader_can],
        configure=args.configure_can,
        hardware_id=registration.can_hardware_id,
    )
    resolved = find_arm_by_hardware_id(device.hardware_id)
    if resolved != registration:
        raise RuntimeError(f"{device.interface} is {resolved.name}, not {registration.name}")
    return device.interface, registration


def execute_plan(arm, gravity, gripper, grip_width, plan, recorder, args):
    """Run every segment; return None on completion or an abort reason."""
    global _running
    dt_s = 1.0 / args.rate
    watchdog = mit.FeedbackWatchdog(args.feedback_timeout)
    next_tick = time.perf_counter()
    next_status = next_health = next_print = next_tick
    overruns = 0
    for number, segment in enumerate(plan):
        recorder.begin()
        outcome = "complete"
        try:
            for i in range(len(segment.q)):
                if not _running:
                    outcome = "interrupted"
                    break
                now = time.perf_counter()
                sample = mit.read_sample(arm, "leader")
                watchdog.observe(sample.stamps, now)
                current = [arm.get_motor_states(j).msg.current for j in mit.JOINTS]
                grip = mit.read_gripper(gripper)
                if grip is None:
                    raise mit.TeleopAbort("lost leader gripper feedback")
                q_des, qd_des = mit.clamp_joint_target(segment.q[i], segment.qd[i])
                tff = mit.send_position_impedance(
                    arm, sample, q_des, qd_des, gravity,
                    args.kp, args.kd, args.max_reference_speed, "leader",
                )
                gripper.move_gripper_m(grip_width, args.grip_base)
                track = float(np.max(np.abs(sample.q - q_des)))
                speed = float(np.max(np.abs(sample.qd)))
                if track > args.max_track_error:
                    joint = int(np.argmax(np.abs(sample.q - q_des))) + 1
                    raise mit.TeleopAbort(f"J{joint} tracking error {track:.3f} rad")
                if speed > args.max_joint_speed:
                    joint = int(np.argmax(np.abs(sample.qd))) + 1
                    raise mit.TeleopAbort(f"J{joint} speed {speed:.3f} rad/s")
                if now >= next_status:
                    mit.require_can_mit_enabled(arm, "leader")
                    next_status = now + 0.1
                recorder.add(
                    t_ns=time.monotonic_ns(), q=sample.q, qd=sample.qd,
                    torque=sample.joint_effort, current=current,
                    q_ref=q_des, qd_ref=qd_des, tff=tff,
                    tff_code=default_tff_code(tff),
                    gripper_width=grip.width, gripper_force=grip.force,
                )
                if now >= next_health:
                    row = [time.monotonic_ns()]
                    for j in mit.JOINTS:
                        state = arm.get_driver_states(j)
                        msg = None if state is None else state.msg
                        row += [
                            getattr(msg, "motor_temp", math.nan),
                            getattr(msg, "foc_temp", math.nan),
                            getattr(msg, "vol", math.nan),
                        ]
                    recorder.health.append(row)
                    next_health = now + 1.0
                if now >= next_print:
                    print(
                        f"  [{number + 1}/{len(plan)}] {segment.name:34s} "
                        f"track={track * R2D:5.2f}deg |qd|={speed:4.2f} overruns={overruns}"
                    )
                    next_print = now + 2.0
                next_tick += dt_s
                slack = next_tick - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    overruns += 1
                    next_tick = time.perf_counter()
        except (mit.TeleopAbort, ValueError, OSError) as exc:
            outcome = f"aborted: {exc}"
            recorder.end(segment, outcome, args.kp, args.kd)
            return str(exc)
        recorder.end(segment, outcome, args.kp, args.kd)
        if outcome != "complete":
            return None
    print(f"plan complete, overruns={overruns}")
    return None


def run(args):
    global _running
    gravity = mit.GravityCompensator(args.leader_calibration)
    poses = None if not args.static else load_poses(args.poses)
    episodes, episode_ids = {}, []
    if args.episodes:
        episodes = load_dataset_actions(args.dataset)
        episode_ids = select_episodes(episodes, args.episodes)

    interface, registration = resolve_leader(args)
    print(f"leader: {registration.name} on {interface}")
    plan = build_plan(args, poses, episodes, episode_ids)
    if episode_ids:
        print(f"episodes ({len(episode_ids)}): {episode_ids}")
    summary = describe_plan(plan, args.rate, registration.name)

    arm = mit.build_arm(interface)
    started = False
    aborted = None
    recorder = None
    try:
        mit.reject_teaching_mode(arm, "leader")
        firmware = firmware_version(arm)
        print(f"  firmware: {firmware}")
        require_default_firmware(firmware)
        sample = mit.read_sample(arm, "leader")
        print(f"  ctrl/mode/status: {mit.arm_status_text(arm)}")
        print("  q(deg): " + " ".join(f"{v * R2D:+7.1f}" for v in sample.q))
        mit.bounded_torque(gravity.torque(sample.q), "leader")
        gripper = arm.init_effector("agx_gripper")
        time.sleep(0.5)
        grip = mit.read_gripper(gripper)
        if grip is None:
            raise mit.TeleopAbort("missing leader gripper feedback")
        print(f"  gripper: {grip.width * 1000:.1f} mm (held at this width)")
        if plan:
            print(
                f"  first planned pose after home: max |q| "
                f"{np.max(np.abs(plan[0].q[-1])) * R2D:.1f} deg"
            )
        if not args.execute:
            print("\nDRY RUN: nothing enabled, nothing sent.")
            return 0

        out = args.out or DEFAULT_OUT_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        recorder = Recorder(out)
        metadata = {
            "created": dt.datetime.now().isoformat(),
            "arm": registration.name,
            "can_interface": interface,
            "can_hardware_id": registration.can_hardware_id,
            "firmware": firmware,
            "piper_fw_encoding": "DEFAULT",
            "args": {k: str(v) for k, v in vars(args).items()},
            "ruri_commit": git_head(REPO),
            "pyagxarm_commit": git_head(REPO / "vendor" / "pyAgxArm"),
            "leader_calibration": str(args.leader_calibration),
            "leader_calibration_sha256": sha256(args.leader_calibration),
            "urdf": str(gravity.urdf),
            "poses_file": None if poses is None else str(args.poses),
            "poses_sha256": None if poses is None else sha256(args.poses),
            "dataset": None if not episode_ids else str(args.dataset),
            "dataset_table": "NOMINAL (calibration/piper_range.json)",
            "dataset_path_source": "action[:6]",
            "episode_ids": episode_ids,
            "plan": summary,
            "units": {
                "q": "rad", "qd": "rad/s", "torque": "N*m (SDK current*k*b)",
                "current": "A", "tff": "N*m requested", "tff_code": "DEFAULT 8-bit code",
                "gripper_width": "m", "health": "t_ns, then per joint motor_temp, foc_temp, vol",
            },
        }
        (Path(out) / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"\nrecording to {out}")

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
        print("\n*** THE LEADER ARM WILL MOVE. WORKSPACE CLEAR, E-STOP IN HAND. ***")
        for i in range(5, 0, -1):
            if not _running:
                return 0
            print(f"    starting in {i} ...")
            time.sleep(1.0)

        arm.set_follower_mode()
        time.sleep(0.3)
        pc.enable_follower_with_hold(
            arm, gravity, args.kp, args.kd, args.max_reference_speed
        )
        started = True
        home_args = argparse.Namespace(
            rate=args.rate,
            feedback_timeout=args.feedback_timeout,
            follower_kp=args.kp,
            follower_kd=args.kd,
            max_reference_speed=args.max_reference_speed,
            max_joint_speed=args.max_joint_speed,
            max_track_error=args.max_track_error,
            home_speed=args.start_home_speed,
            grip_force=args.grip_base,
        )
        if not pc.hold_then_home_follower(
            arm, gravity, gripper, home_args, 1.0, "LEADER STARTUP HOME",
            skip_home_below=mit.HOME_TOLERANCE,
        ):
            return 0
        aborted = execute_plan(arm, gravity, gripper, grip.width, plan, recorder, args)
        if aborted is None and _running:
            pc.hold_then_home_follower(
                arm, gravity, gripper, home_args, 0.5, "LEADER FINAL HOME",
                skip_home_below=mit.HOME_TOLERANCE,
            )
    except (mit.TeleopAbort, ValueError, OSError) as exc:
        aborted = str(exc)
    finally:
        if aborted:
            print(f"\nABORTED: {aborted}")
        if started:
            recovered = False
            if aborted and _running:
                try:
                    home_args.home_speed = args.abort_home_speed
                    recovered = pc.hold_then_home_follower(
                        arm, gravity, gripper, home_args,
                        args.abort_hold_seconds, "ABORT RECOVERY",
                    )
                except Exception as exc:  # noqa: BLE001 - best-effort recovery
                    print(f"ABORT RECOVERY SKIPPED/FAILED: {exc}")
            if not recovered:
                try:
                    hold = mit.read_sample(arm, "final hold").q.copy()
                    end = time.perf_counter() + 0.5
                    while time.perf_counter() < end:
                        sample = mit.read_sample(arm, "final hold")
                        mit.send_position_impedance(
                            arm, sample, hold, np.zeros(6), gravity,
                            args.kp, args.kd, args.max_reference_speed, "final hold",
                        )
                        time.sleep(1.0 / args.rate)
                except Exception as exc:  # noqa: BLE001 - best-effort final hold
                    print(f"final hold warning: {exc}")
            print("LEADER REMAINS ENERGIZED IN ITS FINAL MIT COMMAND; support it before disabling torque.")
        if recorder is not None:
            recorder.close()
            print(f"data: {recorder.out}")
        try:
            arm.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"disconnect warning: {exc}")
    return 1 if aborted else 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true", help="enable the leader and move it")
    p.add_argument("--leader-can", default=None, help="default: discover by registered hardware ID")
    p.add_argument("--leader-side", default="right")
    p.add_argument("--leader-role", default="secondary")
    p.add_argument("--configure-can", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--leader-calibration", type=Path, default=LEADER_CALIBRATION)
    p.add_argument("--static", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--poses", type=Path, default=DEFAULT_POSES)
    p.add_argument("--settle", type=float, default=1.5)
    p.add_argument("--measure", type=float, default=2.0)
    p.add_argument("--episodes", type=int, default=0, help="dataset episodes to replay (0 = none)")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--time-scale", type=float, default=0.5, help="0.5 = half speed")
    p.add_argument("--replay-speed-cap", type=float, default=0.5, help="rad/s")
    p.add_argument("--replay-hold", type=float, default=0.5, help="seconds at path ends")
    p.add_argument("--transit-speed", type=float, default=0.3, help="peak rad/s")
    p.add_argument("--start-home-speed", type=float, default=0.2)
    p.add_argument("--abort-home-speed", type=float, default=0.2)
    p.add_argument("--abort-hold-seconds", type=float, default=10.0)
    p.add_argument("--rate", type=float, default=100.0)
    p.add_argument("--kp", type=float, default=mit.DEFAULT_FOLLOWER_KP,
                   help="default: the gain the paired MIT home already uses on the leader")
    p.add_argument("--kd", type=float, default=mit.DEFAULT_FOLLOWER_KD)
    p.add_argument("--max-reference-speed", type=float, default=1.0)
    p.add_argument("--max-track-error", type=float, default=0.35)
    p.add_argument("--max-joint-speed", type=float, default=2.0)
    p.add_argument("--feedback-timeout", type=float, default=0.2)
    p.add_argument("--grip-base", type=float, default=0.1)
    p.add_argument("--out", type=Path, default=None)
    return p


def validate(args):
    checks = (
        ("time_scale", 0.05, 1.0), ("replay_speed_cap", 0.05, 1.0),
        ("transit_speed", 0.05, 1.0), ("start_home_speed", 0.05, 0.5),
        ("abort_home_speed", 0.05, 0.5), ("kp", 1.0, 20.0), ("kd", 0.1, 2.0),
        ("max_track_error", 0.05, 0.5), ("max_joint_speed", 0.5, 4.0),
        ("rate", 50.0, 200.0), ("settle", 0.5, 10.0), ("measure", 0.5, 10.0),
        ("replay_hold", 0.1, 5.0), ("max_reference_speed", 0.1, 3.0),
    )
    for name, low, high in checks:
        value = getattr(args, name)
        if not (math.isfinite(value) and low <= value <= high):
            raise SystemExit(f"--{name.replace('_', '-')} must be in [{low}, {high}]")
    if args.episodes < 0:
        raise SystemExit("--episodes must be >= 0")
    if max(args.replay_speed_cap, args.transit_speed) > args.max_reference_speed:
        raise SystemExit("--max-reference-speed must cover the replay and transit speeds")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
