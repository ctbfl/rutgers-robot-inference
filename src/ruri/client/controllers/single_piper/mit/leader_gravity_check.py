"""Bounded, operator-supervised right-leader gravity trial; offline by default.

Use --execute only after the operator is supporting the right arm and the
position-control worker has stopped. There is no automatic homing, follower,
camera, gripper command, or friction feed-forward. A completed/interrupted
trial attempts a fresh measured-position hold and leaves torque enabled.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

from . import leader_follower as mit
from . import policy_controller as pc
from ..discovery import discover_piper_can
from ..hardware_registry import find_arm_by_name

CALIBRATION = mit.ASSET_ROOT / "calibration_leader.json"
FEEDBACK_TIMEOUT = 0.2
RATE = mit.DEFAULT_RATE_HZ


def torque_contract():
    """Reject a coefficient migration before connecting or enabling hardware."""
    from pyAgxArm import ArmModel, PiperFW, create_agx_arm_config

    config = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.V189)
    k = np.array([1.18125] * 3 + [0.95844] * 3)
    b = np.array([4.0] * 3 + [1.0] * 3)
    for name, expected in (("joint_torque_k", k), ("joint_torque_b", b)):
        if not np.array_equal(config.get(name), expected):
            raise mit.TeleopAbort(f"SDK {name} differs from the audited leader calibration units")
    if not np.array_equal(config.get("joint_torque_c", np.ones(6)), np.ones(6)):
        raise mit.TeleopAbort("SDK c coefficients require a separate calibration migration")
    return {"driver": "v189", "effort_units": "legacy SDK current*k*b, not independent torque sensing",
            "k": k.tolist(), "b": b.tolist(), "calibration": str(CALIBRATION),
            "calibration_sha256": hashlib.sha256(CALIBRATION.read_bytes()).hexdigest()}


@contextmanager
def hardware_owner():
    # Share the workstation control tool's writer lock. Also catch known RURI
    # writers that do not participate in that lock; never terminate them here.
    with Path(f"/tmp/astra-piper-{os.getuid()}.hardware.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        markers = ("single_piper.mit.assisted", "single_piper.mit.policy_controller", "single_piper.mit.leader_follower",
                   "single_piper.mit.leader_gravity_check", "leader_gravity_id_collect.py",
                   "leader_gravity_id_collect_v182_deprecated.py",
                   "client_single_piper", "single_piper_leader_follower_teleop.py",
                   "set_joint_zero.py", "lerobot-teleoperate")
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            except OSError:
                continue
            if not argv or not any(x in Path(argv[0]).name for x in ("python", "lerobot")):
                continue
            writer = any(any(marker in arg for marker in markers) for arg in argv)
            writer |= any("probe_arm_range.py" in arg for arg in argv) and "--disable-torque" in argv
            if writer:
                raise mit.TeleopAbort(f"Existing controller PID {proc.name}; stop it explicitly first")
        yield


def fresh_sample(arm):
    sample = mit.read_sample(arm, "right leader")
    status = arm.get_arm_status()
    stamps = (*sample.stamps, float(status.timestamp) if status is not None else float("nan"))
    # pyAgxArm stamps feedback with host Unix time. Check every joint, not just
    # whether any member of a tuple advanced while another remained cached.
    age = time.time() - np.asarray(stamps)
    if not np.isfinite(age).all() or np.any(age < -0.05) or np.any(age > FEEDBACK_TIMEOUT):
        raise mit.TeleopAbort("Stale joint/motor/status feedback")
    if status.msg.err_code != 0:
        raise mit.TeleopAbort(f"Arm error code {status.msg.err_code}")
    if np.max(np.abs(sample.qd)) > mit.DEFAULT_MAX_JOINT_SPEED:
        raise mit.TeleopAbort("Measured speed exceeds existing MIT speed limit")
    if np.any(sample.q < arm._lower) or np.any(sample.q > arm._upper):
        raise mit.TeleopAbort("Measured pose outside this arm's measured limits")
    return sample


def require_stationary(sample):
    if np.max(np.abs(sample.qd)) > 0.03:
        raise mit.TeleopAbort("Trial requires a stationary start; hold the arm still before release")


def send_release(arm, sample, gravity, hold, fraction):
    """Smoothly remove position stiffness before using the real leader path."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("release fraction must be in [0, 1]")
    if fraction == 1.0:
        return mit.send_leader(arm, sample, gravity, mit.DEFAULT_LEADER_KD)
    torque = mit.bounded_torque(
        gravity.torque(sample.q) - fraction * mit.DEFAULT_LEADER_KD * sample.qd, "right leader release")
    for i, joint in enumerate(mit.JOINTS):
        arm.move_mit(joint_index=joint, p_des=float(hold[i]), v_des=0.0,
                     kp=(1-fraction)*mit.DEFAULT_FOLLOWER_KP,
                     kd=(1-fraction)*mit.DEFAULT_FOLLOWER_KD, t_ff=float(torque[i]))
    return torque


def trial(arm, gravity, seconds, stop, record):
    start = time.perf_counter()
    initial = fresh_sample(arm)
    require_stationary(initial)
    hold = initial.q.copy()
    next_tick = start
    last_phase = None
    checked_release = False
    next_countdown = 0.0
    gravity_started = None
    while not stop.is_set():
        now = time.perf_counter()
        elapsed = now-start
        if gravity_started is not None and now-gravity_started >= seconds:
            break
        sample = fresh_sample(arm)
        mit.require_can_mit_enabled(arm, "right leader trial")
        if elapsed < 2.0:
            phase, fraction = "position_hold", 0.0
        else:
            if not checked_release:
                require_stationary(sample)
                checked_release = True
            phase = "release" if elapsed < 4.0 else "gravity_only"
            # The quintic can overshoot 1 by floating-point roundoff near its
            # endpoint; this is a gain blend, never a clipped joint target.
            fraction = float(np.clip(mit.quintic_blend(elapsed-2.0, 2.0)[0], 0.0, 1.0))
        torque = send_release(arm, sample, gravity, hold, fraction)
        if phase == "gravity_only" and gravity_started is None:
            gravity_started = time.perf_counter()
        record({"kind": "sample", "wall_time_ns": time.time_ns(), "phase": phase,
                "q": sample.q.tolist(), "qd": sample.qd.tolist(), "t_ff": torque.tolist(),
                "effort_legacy": sample.joint_effort.tolist(),
                "current_a": [float(arm.get_motor_states(j).msg.current) for j in mit.JOINTS]})
        if phase != last_phase:
            print(f"PHASE: {phase}", flush=True)
            last_phase = phase
        if phase == "gravity_only" and elapsed >= next_countdown:
            remaining = max(0., seconds-(time.perf_counter()-gravity_started))
            print(f"可以拖动 / GRAVITY ACTIVE: remaining {remaining:.0f}s; Ctrl+C = position hold", flush=True)
            next_countdown = elapsed+10.0
        next_tick = max(next_tick+1.0/RATE, time.perf_counter())
        time.sleep(max(0.0, next_tick-time.perf_counter()))


def final_position_hold(arm, gravity):
    """No home/re-enable: capture fresh pose, then support it before disconnect."""
    hold = fresh_sample(arm).q.copy()
    deadline = time.perf_counter()+0.5
    while time.perf_counter() < deadline:
        sample = fresh_sample(arm)
        mit.require_can_mit_enabled(arm, "right leader final hold")
        mit.send_position_impedance(arm, sample, hold, np.zeros(6), gravity,
                                    mit.DEFAULT_FOLLOWER_KP, mit.DEFAULT_FOLLOWER_KD,
                                    0.15, "right leader final hold")
        time.sleep(1.0/RATE)
    return hold.tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-for-enter", action="store_true",
                        help="Wait for the operator's Enter key before hardware access")
    parser.add_argument("--seconds", type=float, default=30.0, help="gravity-only duration, 1–120 s")
    parser.add_argument("--log", type=Path, default=Path("logs")/f"leader-gravity-{time.time_ns()}.jsonl")
    args = parser.parse_args(argv)
    if not np.isfinite(args.seconds) or not 1.0 <= args.seconds <= 120.0:
        parser.error("--seconds must be between 1 and 120")
    contract = torque_contract()
    gravity = mit.GravityCompensator(CALIBRATION)
    mit.bounded_torque(gravity.torque(np.zeros(6)), "offline home")
    print(json.dumps({**contract, "arm": "right_arm", "hardware_opened": False,
                      "home_t_ff": gravity.torque(np.zeros(6)).tolist(),
                      "gravity_seconds": args.seconds}, indent=2))
    if not args.execute:
        print("OFFLINE CHECK COMPLETE; --execute requires the supported right arm and no existing writer.")
        return 0
    if args.wait_for_enter:
        if not sys.stdin.isatty():
            parser.error("--wait-for-enter requires an interactive terminal")
        try:
            input(f"扶住右臂并保持静止；按回车开始，Ctrl+C 取消。\n"
                  f"看到 GRAVITY ACTIVE 后，开始计时 {args.seconds:g} 秒。\n")
        except (EOFError, KeyboardInterrupt):
            print("Cancelled before hardware access.")
            return 130
    stop = threading.Event()
    pc._running = True
    def stopping(_sig, _frame):
        stop.set()
        pc._running = False
    old_signals = {sig: signal.signal(sig, stopping) for sig in (signal.SIGINT, signal.SIGTERM)}
    arm = None
    attempted_enable = False
    outcome = "completed"
    exit_code = 0
    final_hold = None
    try:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with hardware_owner(), args.log.open("x") as log:
            def record(row):
                log.write(json.dumps(row, allow_nan=False)+"\n")
                if row.get("kind") != "sample":
                    log.flush()
            try:
                registration = find_arm_by_name("right_arm")
                device = discover_piper_can(hardware_id=registration.can_hardware_id, configure=False)
                arm = mit.build_arm(device.interface)
                if arm.arm_name != "right_arm":
                    raise mit.TeleopAbort("Refusing a non-right arm")
                sample = fresh_sample(arm)
                require_stationary(sample)
                mit.bounded_torque(gravity.torque(sample.q), "right leader startup")
                mit.reject_teaching_mode(arm, "right leader")
                if stop.is_set():
                    raise mit.TeleopAbort("Startup cancelled before mode/enable")
                record({"kind": "header", **contract, "firmware": arm.firmware_info,
                        "hardware_id": registration.can_hardware_id, "rate_hz": RATE,
                        "damping": mit.DEFAULT_LEADER_KD, "torque_ceiling": mit.TAU_CLAMP.tolist()})
                arm.set_follower_mode()
                time.sleep(0.3)
                sample = fresh_sample(arm)
                require_stationary(sample)
                attempted_enable = True
                pc.enable_follower_with_hold(arm, gravity, mit.DEFAULT_FOLLOWER_KP,
                                             mit.DEFAULT_FOLLOWER_KD, 0.15, hold_target=sample.q)
                trial(arm, gravity, args.seconds, stop, record)
                if stop.is_set():
                    outcome = "interrupted"
            except Exception as exc:
                outcome, exit_code = f"aborted: {type(exc).__name__}: {exc}", 1
                print(outcome, flush=True)
            finally:
                if attempted_enable:
                    try:
                        final_hold = final_position_hold(arm, gravity)
                    except Exception as exc:
                        outcome += f"; final hold unavailable: {exc}"
                        exit_code = 1
                        print("FINAL HOLD UNCONFIRMED: support the arm; no blind commands or re-enable.", flush=True)
                if arm is not None:
                    arm.disconnect()
                record({"kind": "outcome", "outcome": outcome, "final_hold_q": final_hold,
                        "torque_disabled": False})
    finally:
        for sig, previous in old_signals.items():
            signal.signal(sig, previous)
    print(f"{outcome}; log={args.log}", flush=True)
    if attempted_enable:
        print("ARM REMAINS ENERGIZED; process exit is not torque disable.", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
