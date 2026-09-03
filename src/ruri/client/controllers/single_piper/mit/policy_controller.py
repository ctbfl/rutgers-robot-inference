#!/usr/bin/env python3
"""Follower-only 100 Hz MIT controller for RURI policy targets.

The process is read-only unless ``--execute`` is supplied.  It is the sole CAN
owner; the RURI Controller communicates over localhost UDP and never opens CAN
itself.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .telemetry import MITTelemetryPublisher, parse_udp_endpoint
from .leader_follower import (
    DEFAULT_ABORT_HOLD_SECONDS,
    DEFAULT_ABORT_HOME_SPEED,
    DEFAULT_FOLLOWER_KD,
    DEFAULT_FOLLOWER_KP,
    DEFAULT_MAX_JOINT_SPEED,
    DEFAULT_RATE_HZ,
    DEFAULT_START_HOME_SPEED,
    FOLLOWER_CAN,
    HOME_SETTLE_SECONDS,
    HOME_TOLERANCE,
    N_JOINTS,
    R2D,
    ArmSample,
    FeedbackWatchdog,
    GravityCompensator,
    TeleopAbort,
    arm_status_text,
    build_arm,
    home_reference,
    read_gripper,
    read_sample,
    recovery_home_duration,
    reject_teaching_mode,
    require_can_mit_enabled,
    send_position_impedance,
)

ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT / "assets"
COMMAND_PROTOCOL = "piper_mit_policy_command"
COMMAND_VERSION = 1
DEFAULT_COMMAND_ADDRESS = "udp://127.0.0.1:6671"
DEFAULT_TELEMETRY_ADDRESS = "udp://127.0.0.1:6670"
DEFAULT_COMMAND_TIMEOUT = 0.30
MAX_PACKET_BYTES = 8192
DIAGNOSTIC_HISTORY_SECONDS = 5.0

# Same physical domain used to normalize the demonstrations.
POLICY_Q_MIN = np.deg2rad(np.array([-150.0, 0.0, -170.0, -100.0, -65.0, -100.0]))
POLICY_Q_MAX = np.deg2rad(np.array([150.0, 180.0, 0.0, 100.0, 65.0, 130.0]))
POLICY_GRIPPER_MIN = 0.0
POLICY_GRIPPER_MAX = 0.068

_running = True


def _stop(_sig, _frame) -> None:
    global _running
    _running = False


@dataclass(frozen=True)
class PolicyTarget:
    session: str
    sequence: int
    q: np.ndarray
    gripper_width: float


def startup_home_is_within_threshold(
    q: Sequence[float], threshold_rad: float
) -> bool:
    """Return whether every joint is close enough to skip startup homing."""
    joints = np.asarray(q, dtype=float)
    if joints.shape != (N_JOINTS,) or not np.all(np.isfinite(joints)):
        raise ValueError(f"startup home joints must be finite shape ({N_JOINTS},)")
    if not math.isfinite(threshold_rad) or threshold_rad < 0:
        raise ValueError("startup home skip threshold must be finite and non-negative")
    return float(np.max(np.abs(joints))) <= threshold_rad


class PolicyCommandReceiver:
    """Non-blocking, stale- and replay-resistant localhost UDP receiver."""

    def __init__(self, address: str, max_age_s: float):
        endpoint = parse_udp_endpoint(address)
        self.address = address
        self.max_age_s = float(max_age_s)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((endpoint.host, endpoint.port))
        self._socket.setblocking(False)
        self.session: str | None = None
        self.sequence = -1
        self.last_received_at: float | None = None
        self.accepted = 0
        self.rejected = 0

    def _decode(
        self, payload: bytes, source: tuple[str, int], now: float
    ) -> PolicyTarget:
        if source[0] != "127.0.0.1":
            raise ValueError("policy command source is not localhost")
        packet = json.loads(payload)
        if not isinstance(packet, dict):
            raise TypeError("policy command must be a JSON object")
        if (
            packet.get("protocol") != COMMAND_PROTOCOL
            or packet.get("version") != COMMAND_VERSION
        ):
            raise ValueError("policy command protocol/version mismatch")
        session = packet.get("session")
        sequence = packet.get("sequence")
        sent_ns = packet.get("monotonic_ns")
        if not isinstance(session, str) or not session or len(session) > 128:
            raise ValueError("invalid policy command session")
        if not isinstance(sequence, int) or sequence < 0:
            raise ValueError("invalid policy command sequence")
        if not isinstance(sent_ns, int):
            raise TypeError("invalid policy command timestamp")
        age_s = now - sent_ns / 1_000_000_000.0
        if age_s < -0.01 or age_s > self.max_age_s:
            raise ValueError(f"stale policy command age={age_s:.3f}s")
        q = np.asarray(packet.get("q_rad"), dtype=float)
        gripper = float(packet.get("gripper_width_m"))
        if (
            q.shape != (N_JOINTS,)
            or not np.all(np.isfinite(q))
            or not math.isfinite(gripper)
        ):
            raise ValueError("policy target must contain finite 6-D q and gripper")
        if np.any(q < POLICY_Q_MIN) or np.any(q > POLICY_Q_MAX):
            raise ValueError(
                "policy joint target is outside the dataset calibration domain"
            )
        if not POLICY_GRIPPER_MIN <= gripper <= POLICY_GRIPPER_MAX:
            raise ValueError("policy gripper target is outside [0, 0.068] m")
        return PolicyTarget(
            session=session, sequence=sequence, q=q, gripper_width=gripper
        )

    def receive_latest(self) -> PolicyTarget | None:
        latest = None
        while True:
            try:
                payload, source = self._socket.recvfrom(MAX_PACKET_BYTES)
            except BlockingIOError:
                break
            now = time.monotonic()
            try:
                target = self._decode(payload, source, now)
                if self.session is None:
                    self.session = target.session
                if target.session != self.session or target.sequence <= self.sequence:
                    raise ValueError("wrong session or non-increasing sequence")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.rejected += 1
                continue
            self.sequence = target.sequence
            self.last_received_at = now
            self.accepted += 1
            latest = target
        return latest

    def close(self) -> None:
        self._socket.close()


def write_diagnostic_log(
    path: Path,
    samples: Sequence[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    """Write the pre-abort controller ring buffer without affecting control timing."""
    output = path.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "trace_header",
                    "schema_version": 1,
                    "wall_time_ns": time.time_ns(),
                    **metadata,
                },
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        for sample in samples:
            stream.write(
                json.dumps(sample, allow_nan=False, separators=(",", ":")) + "\n"
            )


def enable_follower_with_hold(
    arm, gravity, kp, kd, max_reference_speed, timeout=5.0
) -> ArmSample:
    deadline = time.perf_counter() + timeout
    hold = None
    last = None
    while time.perf_counter() < deadline:
        enabled = arm.get_joints_enable_status_list()
        if not (enabled and all(enabled)):
            arm.enable()
        last = read_sample(arm, "follower")
        if hold is None:
            hold = last.q.copy()
        enabled = arm.get_joints_enable_status_list()
        if enabled and all(enabled):
            send_position_impedance(
                arm,
                last,
                hold,
                np.zeros(N_JOINTS),
                gravity,
                kp,
                kd,
                max_reference_speed,
                "follower startup hold",
            )
            ctrl, mode, _ = arm_status_text(arm)
            if "CAN" in ctrl and "MIT" in mode:
                arm.set_auto_set_motion_mode_enabled(False)
                return last
        time.sleep(0.02)
    raise TeleopAbort(
        "failed to enable follower and confirm CAN/MIT: "
        f"enabled={arm.get_joints_enable_status_list()} mode={arm_status_text(arm)[:2]}"
    )


def hold_then_home_follower(
    arm,
    gravity,
    gripper,
    args,
    hold_seconds: float,
    label: str,
    *,
    skip_home_below: float | None = None,
) -> bool:
    require_can_mit_enabled(arm, label)
    sample = read_sample(arm, label)
    hold = sample.q.copy()
    grip = read_gripper(gripper)
    if grip is None:
        raise TeleopAbort(f"{label} missing gripper feedback")
    watchdog = FeedbackWatchdog(args.feedback_timeout)
    dt = 1.0 / args.rate
    start = time.perf_counter()
    next_tick = start
    while _running and time.perf_counter() - start < hold_seconds:
        now = time.perf_counter()
        sample = read_sample(arm, label)
        watchdog.observe(sample.stamps, now)
        send_position_impedance(
            arm,
            sample,
            hold,
            np.zeros(N_JOINTS),
            gravity,
            args.follower_kp,
            args.follower_kd,
            args.max_reference_speed,
            label,
        )
        gripper.move_gripper_m(grip.width, args.grip_force)
        if (
            now - start > 1.0
            and float(np.max(np.abs(sample.qd))) > args.max_joint_speed
        ):
            raise TeleopAbort(f"{label} could not stop motion")
        next_tick += dt
        time.sleep(max(0.0, next_tick - time.perf_counter()))
    if not _running:
        return False

    sample = read_sample(arm, label)
    home_start_q = sample.q.copy()
    home_error = float(np.max(np.abs(home_start_q)))
    skip_home = skip_home_below is not None and startup_home_is_within_threshold(
        home_start_q, skip_home_below
    )
    if skip_home:
        print(
            f"{label}: already near home; skipping homing trajectory "
            f"(max error={home_error:.3f} rad <= {skip_home_below:.3f} rad)"
        )
    else:
        duration = recovery_home_duration(home_start_q, home_start_q, args.home_speed)
        print(
            f"{label}: homing follower over {duration:.1f}s "
            f"at <= {args.home_speed:.2f} rad/s"
        )
        start = time.perf_counter()
        next_tick = start
        next_status = start
        while _running and time.perf_counter() - start < duration:
            now = time.perf_counter()
            sample = read_sample(arm, label)
            watchdog.observe(sample.stamps, now)
            q_des, qd_des = home_reference(home_start_q, now - start, duration)
            send_position_impedance(
                arm,
                sample,
                q_des,
                qd_des,
                gravity,
                args.follower_kp,
                args.follower_kd,
                args.max_reference_speed,
                label,
            )
            gripper.move_gripper_m(grip.width, args.grip_force)
            track = float(np.max(np.abs(sample.q - q_des)))
            speed = float(np.max(np.abs(sample.qd)))
            if track > args.max_track_error:
                raise TeleopAbort(f"{label} tracking error {track:.3f} rad")
            if speed > args.max_joint_speed:
                raise TeleopAbort(f"{label} speed {speed:.3f} rad/s")
            if now >= next_status:
                require_can_mit_enabled(arm, label)
                print(
                    f"  home {now - start:5.1f}/{duration:.1f}s "
                    f"track={track * R2D:4.1f}deg"
                )
                next_status = now + 1.0
            next_tick += dt
            time.sleep(max(0.0, next_tick - time.perf_counter()))
    if not _running:
        return False

    settle_start = time.perf_counter()
    next_tick = settle_start
    while time.perf_counter() - settle_start < HOME_SETTLE_SECONDS:
        sample = read_sample(arm, label)
        send_position_impedance(
            arm,
            sample,
            np.zeros(N_JOINTS),
            np.zeros(N_JOINTS),
            gravity,
            args.follower_kp,
            args.follower_kd,
            args.max_reference_speed,
            label,
        )
        gripper.move_gripper_m(grip.width, args.grip_force)
        next_tick += dt
        time.sleep(max(0.0, next_tick - time.perf_counter()))
    sample = read_sample(arm, label)
    residual = float(np.max(np.abs(sample.q)))
    if residual > HOME_TOLERANCE:
        raise TeleopAbort(f"{label} home residual {residual:.3f} rad")
    print(f"{label}: home complete, residual={residual:.3f} rad")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-can", default=FOLLOWER_CAN)
    parser.add_argument(
        "--follower-calibration",
        type=Path,
        default=ASSET_ROOT / "calibration_follower.json",
    )
    parser.add_argument("--command-address", default=DEFAULT_COMMAND_ADDRESS)
    parser.add_argument("--telemetry-address", default=DEFAULT_TELEMETRY_ADDRESS)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--follower-kp", type=float, default=DEFAULT_FOLLOWER_KP)
    parser.add_argument("--follower-kd", type=float, default=DEFAULT_FOLLOWER_KD)
    parser.add_argument("--max-reference-speed", type=float, default=3.0)
    parser.add_argument(
        "--max-joint-speed", type=float, default=DEFAULT_MAX_JOINT_SPEED
    )
    parser.add_argument("--max-track-error", type=float, default=0.35)
    parser.add_argument("--feedback-timeout", type=float, default=0.20)
    parser.add_argument(
        "--command-timeout", type=float, default=DEFAULT_COMMAND_TIMEOUT
    )
    parser.add_argument(
        "--diagnostic-log",
        type=Path,
        default=None,
        help="optional JSONL path for the final 5 seconds of 100 Hz controller state",
    )
    parser.add_argument("--home-speed", type=float, default=DEFAULT_START_HOME_SPEED)
    parser.add_argument(
        "--startup-home-skip-threshold-rad",
        type=float,
        default=HOME_TOLERANCE,
        help=(
            "skip the startup homing trajectory when every joint is this close "
            "to zero home; must not exceed the final home tolerance"
        ),
    )
    parser.add_argument(
        "--abort-hold-seconds", type=float, default=DEFAULT_ABORT_HOLD_SECONDS
    )
    parser.add_argument(
        "--abort-home-speed", type=float, default=DEFAULT_ABORT_HOME_SPEED
    )
    parser.add_argument("--grip-force", type=float, default=1.0)
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="seconds after first policy command; 0 is unlimited",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="ENERGIZE follower and accept policy motion",
    )
    return parser


def validate_args(args) -> None:
    parse_udp_endpoint(args.command_address)
    parse_udp_endpoint(args.telemetry_address)
    if not 10 <= args.rate <= 200:
        raise ValueError("--rate must be in [10, 200]")
    if not 0 < args.follower_kp <= 50 or not 0 < args.follower_kd <= 2:
        raise ValueError("invalid follower gains")
    if not 0 < args.max_joint_speed <= 5 or not 0 < args.max_track_error <= 0.7:
        raise ValueError("invalid speed/tracking guard")
    if not 0.05 <= args.feedback_timeout <= 1 or not 0.1 <= args.command_timeout <= 2:
        raise ValueError("invalid feedback/command timeout")
    if not 0.05 <= args.home_speed <= 0.5 or not 0.05 <= args.abort_home_speed <= 0.5:
        raise ValueError("invalid home speed")
    if not 0.0 <= args.startup_home_skip_threshold_rad <= HOME_TOLERANCE:
        raise ValueError(
            "--startup-home-skip-threshold-rad must be between 0 and "
            f"{HOME_TOLERANCE:.3f}"
        )
    if not 0.5 <= args.abort_hold_seconds <= 60 or not 0 < args.grip_force <= 5:
        raise ValueError("invalid abort hold/gripper force")
    if args.seconds < 0:
        raise ValueError("--seconds cannot be negative")


def run(args) -> int:
    global _running
    _running = True
    validate_args(args)
    gravity = GravityCompensator(args.follower_calibration)
    print(
        f"controller: follower-only MIT {args.rate:.0f} Hz, kp={args.follower_kp}, kd={args.follower_kd}"
    )
    print("policy joint target: latest 30 Hz pose, tracked directly with v_des=0")
    print(f"measured joint-speed abort: {args.max_joint_speed:.1f} rad/s")
    print("policy gripper target: direct/full speed (no software slew limit)")
    print(f"connecting follower={args.follower_can}")

    arm = gripper = receiver = telemetry = None
    started = False
    command_started = False
    aborted = None
    recovered_home = False
    overruns = commands = 0
    diagnostic_samples: deque[dict[str, object]] | None = None
    if args.diagnostic_log is not None:
        diagnostic_samples = deque(
            maxlen=max(1, int(round(args.rate * DIAGNOSTIC_HISTORY_SECONDS)))
        )

    def dump_diagnostics(outcome: str) -> None:
        if args.diagnostic_log is None or diagnostic_samples is None:
            return
        try:
            write_diagnostic_log(
                args.diagnostic_log,
                list(diagnostic_samples),
                {
                    "outcome": outcome,
                    "rate_hz": args.rate,
                    "follower_kp": args.follower_kp,
                    "follower_kd": args.follower_kd,
                    "max_joint_speed": args.max_joint_speed,
                    "max_track_error": args.max_track_error,
                    "commands": commands,
                    "overruns": overruns,
                },
            )
            print(
                f"diagnostic trace: {args.diagnostic_log} "
                f"({len(diagnostic_samples)} controller samples)"
            )
        except (OSError, TypeError, ValueError) as exc:
            print(f"diagnostic trace warning: {exc}")

    try:
        arm = build_arm(args.follower_can)
        reject_teaching_mode(arm, "follower")
        sample = read_sample(arm, "follower")
        print(f"  ctrl={arm_status_text(arm)[0]} mode={arm_status_text(arm)[1]}")
        print("  q(deg)=" + " ".join(f"{value * R2D:+6.1f}" for value in sample.q))
        gravity.torque(sample.q)
        gripper = arm.init_effector("agx_gripper")
        time.sleep(0.5)
        grip = read_gripper(gripper)
        if grip is None:
            raise TeleopAbort("missing follower gripper feedback")
        print(f"  gripper={grip.width * 1000:.1f} mm")
        if not args.execute:
            print("READ-ONLY CHECK COMPLETE: nothing enabled and no command sent.")
            return 0

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
        print(
            "\n*** FOLLOWER WILL ENTER MIT; HOME MOTION IS SKIPPED WHEN "
            "ALREADY WITHIN THRESHOLD. E-STOP IN HAND. ***"
        )

        receiver = PolicyCommandReceiver(args.command_address, args.command_timeout)
        telemetry = MITTelemetryPublisher(args.telemetry_address)
        arm.set_follower_mode()
        time.sleep(0.3)
        sample = enable_follower_with_hold(
            arm, gravity, args.follower_kp, args.follower_kd, args.max_reference_speed
        )
        started = True
        print(
            f"  follower enabled: ctrl={arm_status_text(arm)[0]} "
            f"mode={arm_status_text(arm)[1]}"
        )
        if not hold_then_home_follower(
            arm,
            gravity,
            gripper,
            args,
            0.0,
            "STARTUP HOME",
            skip_home_below=args.startup_home_skip_threshold_rad,
        ):
            raise TeleopAbort("startup home cancelled")

        sample = read_sample(arm, "follower")
        grip = read_gripper(gripper)
        if grip is None:
            raise TeleopAbort("missing gripper after home")
        latest_goal = sample.q.copy()
        grip_goal = grip.width
        first_command_time = None
        watchdog = FeedbackWatchdog(args.feedback_timeout)
        dt = 1.0 / args.rate
        next_tick = time.perf_counter()
        next_report = next_tick
        next_status = next_tick
        print(
            f"READY: holding home; waiting for policy commands at {args.command_address}"
        )

        while _running:
            now = time.perf_counter()
            target = receiver.receive_latest()
            target_delta = np.zeros(N_JOINTS)
            target_received = target is not None
            if target is not None:
                target_delta = target.q - latest_goal
                latest_goal = target.q.copy()
                grip_goal = target.gripper_width
                if not command_started:
                    command_started = True
                    first_command_time = now
                    print(f"POLICY ENGAGED: session={target.session[:12]}")
            if command_started:
                if (
                    receiver.last_received_at is None
                    or now - receiver.last_received_at > args.command_timeout
                ):
                    raise TeleopAbort(
                        f"policy command timeout {now - receiver.last_received_at:.3f}s exceeds {args.command_timeout:.3f}s"
                    )
                if args.seconds > 0 and now - first_command_time >= args.seconds:
                    print(f"policy duration {args.seconds:.1f}s complete")
                    break

            sample = read_sample(arm, "follower")
            watchdog.observe(sample.stamps, now)
            # LeRobot schedules one policy pose every 33.3 ms.  MIT tracks that
            # latest pose directly at 100 Hz; it is not a trajectory planner.
            q_cmd = latest_goal
            qd_cmd = np.zeros(N_JOINTS)
            track = float(np.max(np.abs(sample.q - q_cmd)))
            speed = float(np.max(np.abs(sample.qd)))
            if diagnostic_samples is not None:
                diagnostic_samples.append(
                    {
                        "kind": "controller_sample",
                        "wall_time_ns": time.time_ns(),
                        "monotonic_ns": time.monotonic_ns(),
                        "target_received": target_received,
                        "target_sequence": receiver.sequence,
                        "q": sample.q.tolist(),
                        "qd": sample.qd.tolist(),
                        "q_target": q_cmd.tolist(),
                        "target_delta": target_delta.tolist(),
                        "tracking_error": track,
                        "max_measured_speed": speed,
                    }
                )
            if track > args.max_track_error:
                raise TeleopAbort(
                    f"follower tracking error {track:.3f} rad exceeds {args.max_track_error:.3f}"
                )
            if speed > args.max_joint_speed:
                joint = int(np.argmax(np.abs(sample.qd))) + 1
                raise TeleopAbort(
                    f"measured J{joint} speed {speed:.3f} rad/s exceeds "
                    f"{args.max_joint_speed:.3f}"
                )
            send_position_impedance(
                arm,
                sample,
                q_cmd,
                qd_cmd,
                gravity,
                args.follower_kp,
                args.follower_kd,
                args.max_reference_speed,
                "follower policy",
            )
            # Explicit user choice: direct target, no software gripper slew limit.
            gripper.move_gripper_m(grip_goal, args.grip_force)
            grip = read_gripper(gripper)
            if grip is None:
                raise TeleopAbort("lost gripper feedback")

            if now >= next_status:
                require_can_mit_enabled(arm, "follower policy")
                next_status = now + 0.5
            telemetry.publish(
                phase="engaged",
                follower_q=sample.q,
                follower_qd=sample.qd,
                follower_joint_effort=sample.joint_effort,
                follower_target_q=q_cmd,
                follower_target_qd=qd_cmd,
                follower_gripper_width=grip.width,
                follower_gripper_target=grip_goal,
                follower_gripper_force=grip.force,
                overruns=overruns,
            )
            commands += 1
            if now >= next_report:
                age = (
                    None
                    if receiver.last_received_at is None
                    else now - receiver.last_received_at
                )
                age_text = "waiting" if age is None else f"{age * 1000:4.0f}ms"
                print(
                    f"  policy={'ON' if command_started else 'WAIT'} track={track * R2D:4.1f}deg "
                    f"|qd|max={speed:4.2f}rad/s command_age={age_text} overruns={overruns}"
                )
                next_report = now + 0.5
            next_tick += dt
            slack = next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                overruns += 1
                next_tick = time.perf_counter()
    except (TeleopAbort, ValueError, FileNotFoundError, OSError) as exc:
        aborted = str(exc)
        print(f"\nABORTED: {aborted}")
        dump_diagnostics(f"aborted: {aborted}")
        if telemetry is not None:
            telemetry.publish_status("aborted", aborted)
    finally:
        if started and arm is not None and gripper is not None:
            if aborted and command_started:
                try:
                    args.home_speed = args.abort_home_speed
                    recovered_home = hold_then_home_follower(
                        arm,
                        gravity,
                        gripper,
                        args,
                        args.abort_hold_seconds,
                        "ABORT RECOVERY",
                    )
                except Exception as exc:  # noqa: BLE001 - best-effort safety recovery
                    print(f"ABORT RECOVERY SKIPPED/FAILED: {exc}")
            if not recovered_home:
                try:
                    sample = read_sample(arm, "final hold")
                    hold = sample.q.copy()
                    grip = read_gripper(gripper)
                    end = time.perf_counter() + 0.5
                    while time.perf_counter() < end:
                        sample = read_sample(arm, "final hold")
                        send_position_impedance(
                            arm,
                            sample,
                            hold,
                            np.zeros(N_JOINTS),
                            gravity,
                            args.follower_kp,
                            args.follower_kd,
                            args.max_reference_speed,
                            "final hold",
                        )
                        if grip is not None:
                            gripper.move_gripper_m(grip.width, args.grip_force)
                        time.sleep(1.0 / args.rate)
                except Exception as exc:  # noqa: BLE001 - best-effort final hold
                    print(f"final hold warning: {exc}")
        if telemetry is not None:
            if aborted is None:
                telemetry.publish_status("stopped", "policy controller ended")
            telemetry.close()
        if receiver is not None:
            receiver.close()
        if aborted is None:
            dump_diagnostics("stopped")
        if arm is not None:
            try:
                arm.disconnect()
            except Exception as exc:  # noqa: BLE001 - disconnect must not hide result
                print(f"disconnect warning: {exc}")
    if started:
        print(
            f"commands={commands}, overruns={overruns}, recovered_home={recovered_home}"
        )
        print(
            "FOLLOWER REMAINS ENERGIZED IN ITS FINAL MIT COMMAND; support before disabling torque."
        )
    return 1 if aborted else 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
