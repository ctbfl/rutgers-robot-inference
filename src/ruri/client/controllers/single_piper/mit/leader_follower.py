#!/usr/bin/env python3
"""Planner-free dual-Piper joint teleoperation with MIT control on both arms.

Control topology
----------------

    leader q --------------------------> follower MIT position target

    leader torque   = gravity_leader(q) - leader_damping * qdot
    follower torque = kp * (q_leader - q_follower)
                    + kd * (qdot_leader - qdot_follower)
                    + gravity_follower(q_follower)

There is deliberately no arm-joint force reflection, NEXT model, contact
estimator, or ``move_j``/``move_js`` call in this program. The follower receives
a new MIT impedance target every tick, so no trajectory planner is repeatedly
restarted.

The gripper remains a separate channel. Its already validated direct feedback
logic is retained: the follower tracks the leader width, while the leader tracks
the follower's measured width with a low base force plus the follower's measured
grip force. This makes the leader gripper backdrivable in free space and renders
contact without reintroducing arm-joint force feedback.

The leader and follower use separate gravity calibrations because they are
different physical arms with different end payloads. The leader has kp=0 and is
therefore free to move by hand; only calibrated gravity and damping are applied.

SAFETY AND VALIDATION STATUS
----------------------------

The leader gravity loop is hardware validated. Streaming MIT position control
on the follower is NEW and has not yet been validated as a complete teleop loop.
Consequently this script is read-only unless ``--execute`` is supplied. Start
with the default low follower gains, both arms supported, clear workspace, and
the E-stop in hand.

Before coupling, both arms independently return to all-zero home through slow
quintic MIT position-impedance streams with their own gravity feed-forward.
This removes the initial leader/follower gap without reintroducing ``move_j`` or
its planner. Only after both arms are home does the leader become gravity-only
and the follower begin its quintic transition to the live leader target.

Examples
--------

    # Read-only: connect, validate feedback/calibrations, print the command.
    python -m ruri.client.controllers.single_piper.mit.leader_follower

    # First short hardware test. BOTH ARMS WILL MOVE.
    python -m ruri.client.controllers.single_piper.mit.leader_follower \
        --execute --seconds 20 --no-gripper

    # Normal run after the short test has been judged stable.
    python -m ruri.client.controllers.single_piper.mit.leader_follower --execute

    # Opt-in read-only stream for the LeRobot v3 recorder plugin.
    python -m ruri.client.controllers.single_piper.mit.leader_follower --execute \
        --telemetry-address udp://127.0.0.1:6670
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import signal
import time
from typing import Optional, Sequence

import numpy as np

from .telemetry import MITTelemetryPublisher, parse_udp_endpoint


ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT / "assets"

from .gravity import PiperGravity, URDF_DEFAULT


LEADER_CAN = "can_leader"       # RIGHT arm on this rig
FOLLOWER_CAN = "can_follower"   # LEFT arm on this rig
JOINTS = tuple(range(1, 7))
N_JOINTS = len(JOINTS)
R2D = 180.0 / np.pi

DEFAULT_RATE_HZ = 100.0
DEFAULT_LEADER_KD = 0.2
DEFAULT_FOLLOWER_KP = 10.0
DEFAULT_FOLLOWER_KD = 0.8
DEFAULT_MAX_JOINT_SPEED = 4.0
DEFAULT_START_HOME_SPEED = 0.20
DEFAULT_ABORT_HOLD_SECONDS = 10.0
DEFAULT_ABORT_HOME_SPEED = 0.20
HOME_MIN_SECONDS = 5.0
HOME_SETTLE_SECONDS = 2.0
HOME_TOLERANCE = 0.05

# These are feed-forward ceilings, not total MIT torque ceilings. They are well
# below what the SDK accepts on J1-3 and match the already validated leader
# gravity scripts. Exceeding one means the model/pose deserves inspection.
TAU_CLAMP = np.array([10.0, 12.0, 10.0, 3.0, 3.0, 3.0])

_running = True


class TeleopAbort(RuntimeError):
    """A controlled stop requested by a runtime safety check."""


def _stop(_sig, _frame) -> None:
    global _running
    _running = False


@dataclass(frozen=True)
class ArmSample:
    q: np.ndarray
    qd: np.ndarray
    joint_effort: np.ndarray
    stamps: tuple[float, ...]


@dataclass(frozen=True)
class GripperSample:
    width: float
    force: float
    timestamp: float


class FeedbackWatchdog:
    """Detect a silent CAN source even when the SDK keeps returning its cache."""

    def __init__(self, timeout: float):
        self.timeout = float(timeout)
        self._last_stamps: Optional[tuple[float, ...]] = None
        self._last_change: Optional[float] = None

    def observe(self, stamps: tuple[float, ...], now: float) -> None:
        if stamps != self._last_stamps:
            self._last_stamps = stamps
            self._last_change = now
            return
        if self._last_change is None:
            self._last_change = now
            return
        if now - self._last_change > self.timeout:
            raise TeleopAbort(
                f"feedback timestamps did not advance for "
                f"{now - self._last_change:.3f}s"
            )


class GravityCompensator:
    """URDF gravity plus the measured per-arm sign and joint corrections."""

    def __init__(self, calibration_path: Path):
        self.path = calibration_path.resolve()
        with self.path.open() as fh:
            calibration = json.load(fh)

        required = {
            "torque_sign_vs_urdf",
            "payload_mass_kg",
            "payload_com_m",
            "per_joint_scale",
        }
        missing = required.difference(calibration)
        if missing:
            raise ValueError(
                f"{self.path} is missing calibration fields: "
                + ", ".join(sorted(missing))
            )

        configured_urdf = calibration.get("urdf")
        urdf = Path(configured_urdf or URDF_DEFAULT).expanduser()
        if not urdf.is_absolute():
            urdf = self.path.parent / urdf
        urdf = urdf.resolve()
        if not urdf.is_file():
            raise FileNotFoundError(f"gravity URDF does not exist: {urdf}")

        payload_com = np.asarray(calibration["payload_com_m"], dtype=float)
        scale = np.asarray(calibration["per_joint_scale"], dtype=float)
        if payload_com.shape != (3,):
            raise ValueError(f"{self.path}: payload_com_m must contain 3 values")
        if scale.shape != (N_JOINTS,):
            raise ValueError(f"{self.path}: per_joint_scale must contain 6 values")
        if not np.all(np.isfinite(payload_com)) or not np.all(np.isfinite(scale)):
            raise ValueError(f"{self.path}: calibration contains non-finite values")

        self.urdf = urdf.resolve()
        self.payload_mass = float(calibration["payload_mass_kg"])
        self.payload_com = payload_com
        self.scale = scale
        self.sign = float(calibration["torque_sign_vs_urdf"])
        if self.sign not in (-1.0, 1.0):
            raise ValueError(f"{self.path}: torque_sign_vs_urdf must be +1 or -1")

        self.model = PiperGravity(
            str(self.urdf),
            payload_mass=self.payload_mass,
            payload_com=tuple(float(v) for v in self.payload_com),
        )

    def torque(self, q: Sequence[float]) -> np.ndarray:
        q_array = joint_vector(q, "gravity position")
        return self.sign * self.model.torque(q_array) * self.scale


def joint_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (N_JOINTS,):
        raise ValueError(f"{name} must contain exactly 6 values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def quintic_blend(elapsed: float, duration: float) -> tuple[float, float]:
    """Return C2-continuous blend alpha and d(alpha)/dt in [0, 1]."""
    if duration <= 0.0 or elapsed >= duration:
        return 1.0, 0.0
    if elapsed <= 0.0:
        return 0.0, 0.0
    u = elapsed / duration
    alpha = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    derivative = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    return alpha, derivative


def engagement_reference(
    follower_start: Sequence[float],
    leader_q: Sequence[float],
    leader_qd: Sequence[float],
    elapsed: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothly join the follower's start pose to the live leader stream."""
    q0 = joint_vector(follower_start, "follower start")
    ql = joint_vector(leader_q, "leader position")
    qdl = joint_vector(leader_qd, "leader velocity")
    alpha, alpha_dot = quintic_blend(elapsed, duration)
    delta = ql - q0
    return q0 + alpha * delta, alpha * qdl + alpha_dot * delta


def home_reference(
    start_q: Sequence[float],
    elapsed: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Quintic position/velocity reference from ``start_q`` to all-zero home."""
    start = joint_vector(start_q, "home start")
    alpha, alpha_dot = quintic_blend(elapsed, duration)
    return (1.0 - alpha) * start, -alpha_dot * start


def recovery_home_duration(
    leader_start: Sequence[float],
    follower_start: Sequence[float],
    max_speed: float,
) -> float:
    """Duration that bounds a quintic home's peak target speed.

    The maximum derivative of the 10u^3-15u^4+6u^5 blend is 1.875, so
    ``duration >= 1.875 * distance / max_speed`` enforces the requested bound.
    """
    farthest = max(
        float(np.max(np.abs(joint_vector(leader_start, "leader home start")))),
        float(np.max(np.abs(joint_vector(follower_start, "follower home start")))),
    )
    return max(HOME_MIN_SECONDS, 1.875 * farthest / float(max_speed))


def bounded_torque(raw: Sequence[float], label: str) -> np.ndarray:
    torque = joint_vector(raw, f"{label} torque")
    if np.any(np.abs(torque) > TAU_CLAMP):
        joint = int(np.argmax(np.abs(torque) - TAU_CLAMP)) + 1
        raise TeleopAbort(
            f"{label} gravity command J{joint}={torque[joint - 1]:+.3f} N*m "
            f"exceeds the {TAU_CLAMP[joint - 1]:.3f} N*m ceiling"
        )
    return torque


def build_arm(channel: str):
    try:
        from pyAgxArm import (
            AgxArmFactory,
            ArmModel,
            PiperFW,
            create_agx_arm_config,
        )
    except ImportError as exc:
        raise ImportError(
            "Piper controllers require the pyAgxArm submodule and the RURI "
            "piper extra; see the Piper installation instructions"
        ) from exc
    config = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.DEFAULT,
        channel=channel,
    )
    arm = AgxArmFactory.create_arm(config)
    arm.connect()
    time.sleep(0.5)
    return arm


def read_sample(arm, label: str) -> ArmSample:
    angles = arm.get_joint_angles()
    motors = [arm.get_motor_states(joint) for joint in JOINTS]
    if angles is None or any(motor is None for motor in motors):
        raise TeleopAbort(f"{label} is missing joint-angle or motor feedback")
    q = joint_vector(list(angles.msg), f"{label} position")
    qd = joint_vector([motor.msg.velocity for motor in motors], f"{label} velocity")
    joint_effort = joint_vector(
        [motor.msg.torque for motor in motors],
        f"{label} joint effort",
    )
    stamps = (float(angles.timestamp),) + tuple(
        float(motor.timestamp) for motor in motors
    )
    return ArmSample(
        q=q,
        qd=qd,
        joint_effort=joint_effort,
        stamps=stamps,
    )


def arm_status_text(arm) -> tuple[str, str, str]:
    status = arm.get_arm_status()
    if status is None:
        return "?", "?", "?"
    msg = status.msg
    return str(msg.ctrl_mode), str(msg.mode_feedback), str(msg.arm_status)


def reject_teaching_mode(arm, label: str) -> None:
    ctrl, _, _ = arm_status_text(arm)
    if "TEACH" in ctrl:
        raise TeleopAbort(
            f"{label} ctrl_mode={ctrl}; release the teach button and run "
            f"factr/to_can_mode.py before commanding MIT"
        )


def send_leader(
    arm,
    sample: ArmSample,
    gravity: GravityCompensator,
    damping: float,
) -> np.ndarray:
    torque = bounded_torque(
        gravity.torque(sample.q) - float(damping) * sample.qd,
        "leader",
    )
    for index, joint in enumerate(JOINTS):
        arm.move_mit(
            joint_index=joint,
            p_des=float(sample.q[index]),
            v_des=0.0,
            kp=0.0,
            kd=0.0,
            t_ff=float(torque[index]),
        )
    return torque


def send_position_impedance(
    arm,
    sample: ArmSample,
    q_des: Sequence[float],
    qd_des: Sequence[float],
    gravity: GravityCompensator,
    kp: float,
    kd: float,
    max_reference_speed: float,
    label: str,
) -> np.ndarray:
    target = joint_vector(q_des, f"{label} target")
    target_velocity = np.clip(
        joint_vector(qd_des, f"{label} target velocity"),
        -max_reference_speed,
        max_reference_speed,
    )
    torque = bounded_torque(gravity.torque(sample.q), label)
    for index, joint in enumerate(JOINTS):
        arm.move_mit(
            joint_index=joint,
            p_des=float(target[index]),
            v_des=float(target_velocity[index]),
            kp=float(kp),
            kd=float(kd),
            t_ff=float(torque[index]),
        )
    return torque


def send_follower(
    arm,
    sample: ArmSample,
    q_des: Sequence[float],
    qd_des: Sequence[float],
    gravity: GravityCompensator,
    kp: float,
    kd: float,
    max_reference_speed: float,
) -> np.ndarray:
    return send_position_impedance(
        arm,
        sample,
        q_des,
        qd_des,
        gravity,
        kp,
        kd,
        max_reference_speed,
        "follower",
    )


def enable_both_with_safe_commands(
    leader,
    follower,
    leader_gravity: GravityCompensator,
    follower_gravity: GravityCompensator,
    leader_kd: float,
    follower_kp: float,
    follower_kd: float,
    max_reference_speed: float,
    timeout: float = 5.0,
) -> tuple[ArmSample, ArmSample]:
    """Enable and enter MIT while immediately supporting every enabled arm.

    ``move_mit`` normally sends a mode-control frame before every joint frame.
    Once both controllers report CAN/MIT, automatic mode setting is disabled so
    the steady-state loop sends only the six MIT joint frames per arm and tick.
    """
    deadline = time.perf_counter() + timeout
    follower_hold: Optional[np.ndarray] = None
    last_leader: Optional[ArmSample] = None
    last_follower: Optional[ArmSample] = None

    while time.perf_counter() < deadline:
        leader_enabled = leader.get_joints_enable_status_list()
        follower_enabled = follower.get_joints_enable_status_list()
        if not (leader_enabled and all(leader_enabled)):
            leader.enable()
        if not (follower_enabled and all(follower_enabled)):
            follower.enable()

        last_leader = read_sample(leader, "leader")
        last_follower = read_sample(follower, "follower")
        if follower_hold is None:
            follower_hold = last_follower.q.copy()

        leader_enabled = leader.get_joints_enable_status_list()
        follower_enabled = follower.get_joints_enable_status_list()
        if leader_enabled and all(leader_enabled):
            send_leader(leader, last_leader, leader_gravity, leader_kd)
        if follower_enabled and all(follower_enabled):
            send_follower(
                follower,
                last_follower,
                follower_hold,
                np.zeros(N_JOINTS),
                follower_gravity,
                follower_kp,
                follower_kd,
                max_reference_speed,
            )

        if (leader_enabled and all(leader_enabled)
                and follower_enabled and all(follower_enabled)):
            leader_ctrl, leader_mode, _ = arm_status_text(leader)
            follower_ctrl, follower_mode, _ = arm_status_text(follower)
            if ("CAN" in leader_ctrl and "MIT" in leader_mode
                    and "CAN" in follower_ctrl and "MIT" in follower_mode):
                leader.set_auto_set_motion_mode_enabled(False)
                follower.set_auto_set_motion_mode_enabled(False)
                return last_leader, last_follower
        time.sleep(0.02)

    raise TeleopAbort(
        "failed to enable both arms and confirm CAN/MIT: "
        f"leader={leader.get_joints_enable_status_list()} "
        f"follower={follower.get_joints_enable_status_list()} "
        f"leader_mode={arm_status_text(leader)[:2]} "
        f"follower_mode={arm_status_text(follower)[:2]}"
    )


def read_gripper(gripper) -> Optional[GripperSample]:
    try:
        status = gripper.get_gripper_status()
        if status is None:
            return None
        return GripperSample(
            width=max(0.0, float(status.msg.value)),
            force=float(status.msg.force),
            timestamp=float(status.timestamp),
        )
    except Exception:
        return None


def send_grippers(
    leader_gripper,
    follower_gripper,
    leader_sample: GripperSample,
    follower_sample: GripperSample,
    follower_force: float,
    leader_base_force: float,
    feedback_gain: float,
    leader_max_force: float,
) -> tuple[float, float, float]:
    """Run the validated direct gripper feedback channel.

    Returns ``(width_difference, measured_force, rendered_force)`` for runtime
    diagnostics. The follower force measurement is made unsigned because its
    sign convention depends on closing direction; direction is supplied by the
    follower's actual width being used as the leader position target.
    """
    measured_force = abs(float(follower_sample.force))
    rendered_force = min(
        float(leader_max_force),
        float(leader_base_force) + float(feedback_gain) * measured_force,
    )
    follower_gripper.move_gripper_m(leader_sample.width, follower_force)
    leader_gripper.move_gripper_m(follower_sample.width, rendered_force)
    return (
        leader_sample.width - follower_sample.width,
        measured_force,
        rendered_force,
    )


def send_gripper_hold(
    leader_gripper,
    follower_gripper,
    follower_force: float,
    leader_base_force: float,
) -> None:
    """Remove reflected grip force and hold each gripper at its current width."""
    if leader_gripper is None or follower_gripper is None:
        return
    leader_sample = read_gripper(leader_gripper)
    follower_sample = read_gripper(follower_gripper)
    if leader_sample is None or follower_sample is None:
        return
    follower_gripper.move_gripper_m(follower_sample.width, follower_force)
    leader_gripper.move_gripper_m(leader_sample.width, leader_base_force)


def require_can_mit_enabled(arm, label: str) -> None:
    enabled = arm.get_joints_enable_status_list()
    ctrl, mode, status = arm_status_text(arm)
    if not (enabled and all(enabled)):
        raise TeleopAbort(f"{label} de-energized: {enabled}")
    if "CAN" not in ctrl or "MIT" not in mode:
        raise TeleopAbort(f"{label} left CAN/MIT: ctrl={ctrl}, mode={mode}")
    if "NORMAL" not in status:
        raise TeleopAbort(f"{label} arm_status={status}")


def home_both_with_mit(
    leader,
    follower,
    leader_gravity: GravityCompensator,
    follower_gravity: GravityCompensator,
    leader_gripper,
    follower_gripper,
    args: argparse.Namespace,
    *,
    hold_seconds: float,
    home_speed: float,
    phase_label: str,
) -> bool:
    """Optionally hold, then home both arms with planner-free MIT commands.

    Motion continues only while feedback, enable state, and CAN/MIT mode remain
    healthy. This is used both before coupling and after a healthy runtime abort.
    """
    command_label = phase_label.lower()
    leader_label = f"leader {command_label}"
    follower_label = f"follower {command_label}"
    require_can_mit_enabled(leader, leader_label)
    require_can_mit_enabled(follower, follower_label)
    leader_sample = read_sample(leader, leader_label)
    follower_sample = read_sample(follower, follower_label)
    leader_hold = leader_sample.q.copy()
    follower_hold = follower_sample.q.copy()
    leader_fresh = FeedbackWatchdog(args.feedback_timeout)
    follower_fresh = FeedbackWatchdog(args.feedback_timeout)
    dt = 1.0 / args.rate

    if hold_seconds > 0.0:
        print(
            f"\n{phase_label}: holding both arms at their current poses for "
            f"{hold_seconds:.1f}s ..."
        )
    hold_start = time.perf_counter()
    next_tick = hold_start
    next_report = hold_start
    while _running and time.perf_counter() - hold_start < hold_seconds:
        now = time.perf_counter()
        leader_sample = read_sample(leader, leader_label)
        follower_sample = read_sample(follower, follower_label)
        leader_fresh.observe(leader_sample.stamps, now)
        follower_fresh.observe(follower_sample.stamps, now)
        send_position_impedance(
            leader,
            leader_sample,
            leader_hold,
            np.zeros(N_JOINTS),
            leader_gravity,
            args.follower_kp,
            args.follower_kd,
            args.max_reference_speed,
            leader_label,
        )
        send_position_impedance(
            follower,
            follower_sample,
            follower_hold,
            np.zeros(N_JOINTS),
            follower_gravity,
            args.follower_kp,
            args.follower_kd,
            args.max_reference_speed,
            follower_label,
        )
        send_gripper_hold(
            leader_gripper,
            follower_gripper,
            args.grip_force,
            args.grip_base,
        )
        hold_speed = max(
            float(np.max(np.abs(leader_sample.qd))),
            float(np.max(np.abs(follower_sample.qd))),
        )
        # Give the position hold one second to brake any motion already in
        # progress. Continued high speed after that is not recoverable.
        if now - hold_start > 1.0 and hold_speed > args.max_joint_speed:
            raise TeleopAbort(
                f"{command_label} could not stop motion: {hold_speed:.3f} rad/s"
            )
        if now >= next_report:
            remaining = max(0.0, hold_seconds - (now - hold_start))
            print(f"  {command_label} hold: {remaining:4.1f}s remaining")
            require_can_mit_enabled(leader, leader_label)
            require_can_mit_enabled(follower, follower_label)
            next_report = now + 1.0
        next_tick += dt
        time.sleep(max(0.0, next_tick - time.perf_counter()))

    if not _running:
        print(f"  {command_label} cancelled by signal; staying at current pose")
        return False

    # Re-anchor after the hold, then send each arm along its own C2-continuous
    # path to zero. No move_j or firmware trajectory planner is involved.
    leader_sample = read_sample(leader, leader_label)
    follower_sample = read_sample(follower, follower_label)
    leader_start = leader_sample.q.copy()
    follower_start = follower_sample.q.copy()
    duration = recovery_home_duration(
        leader_start,
        follower_start,
        home_speed,
    )
    print(
        f"{phase_label}: homing both arms with MIT over {duration:.1f}s "
        f"(target peak speed <= {home_speed:.2f} rad/s) ..."
    )
    home_start = time.perf_counter()
    next_tick = home_start
    next_report = home_start
    peak_track = 0.0
    while _running and time.perf_counter() - home_start < duration:
        now = time.perf_counter()
        elapsed = now - home_start
        leader_sample = read_sample(leader, leader_label)
        follower_sample = read_sample(follower, follower_label)
        leader_fresh.observe(leader_sample.stamps, now)
        follower_fresh.observe(follower_sample.stamps, now)
        leader_des, leader_vel = home_reference(leader_start, elapsed, duration)
        follower_des, follower_vel = home_reference(follower_start, elapsed, duration)
        send_position_impedance(
            leader,
            leader_sample,
            leader_des,
            leader_vel,
            leader_gravity,
            args.follower_kp,
            args.follower_kd,
            args.max_reference_speed,
            leader_label,
        )
        send_position_impedance(
            follower,
            follower_sample,
            follower_des,
            follower_vel,
            follower_gravity,
            args.follower_kp,
            args.follower_kd,
            args.max_reference_speed,
            follower_label,
        )
        send_gripper_hold(
            leader_gripper,
            follower_gripper,
            args.grip_force,
            args.grip_base,
        )
        track = max(
            float(np.max(np.abs(leader_sample.q - leader_des))),
            float(np.max(np.abs(follower_sample.q - follower_des))),
        )
        recovery_speed = max(
            float(np.max(np.abs(leader_sample.qd))),
            float(np.max(np.abs(follower_sample.qd))),
        )
        peak_track = max(peak_track, track)
        if track > args.max_track_error:
            raise TeleopAbort(
                f"{command_label} tracking error {track:.3f} rad exceeds "
                f"{args.max_track_error:.3f} rad"
            )
        if recovery_speed > args.max_joint_speed:
            raise TeleopAbort(
                f"{command_label} speed {recovery_speed:.3f} rad/s exceeds "
                f"{args.max_joint_speed:.3f} rad/s"
            )
        if now >= next_report:
            require_can_mit_enabled(leader, leader_label)
            require_can_mit_enabled(follower, follower_label)
            print(
                f"  {command_label}: {elapsed:5.1f}/{duration:.1f}s "
                f"track={track * R2D:4.1f}deg"
            )
            next_report = now + 1.0
        next_tick += dt
        time.sleep(max(0.0, next_tick - time.perf_counter()))

    if not _running:
        print(f"  {command_label} cancelled by signal; staying at current pose")
        return False

    # Give the low-gain impedance loop time to settle at exactly zero.
    settle_start = time.perf_counter()
    next_tick = settle_start
    while _running and time.perf_counter() - settle_start < HOME_SETTLE_SECONDS:
        leader_sample = read_sample(leader, leader_label)
        follower_sample = read_sample(follower, follower_label)
        send_position_impedance(
            leader,
            leader_sample,
            np.zeros(N_JOINTS),
            np.zeros(N_JOINTS),
            leader_gravity,
            args.follower_kp,
            args.follower_kd,
            args.max_reference_speed,
            leader_label,
        )
        send_position_impedance(
            follower,
            follower_sample,
            np.zeros(N_JOINTS),
            np.zeros(N_JOINTS),
            follower_gravity,
            args.follower_kp,
            args.follower_kd,
            args.max_reference_speed,
            follower_label,
        )
        next_tick += dt
        time.sleep(max(0.0, next_tick - time.perf_counter()))

    if not _running:
        print(f"  {command_label} cancelled by signal; staying at current pose")
        return False

    require_can_mit_enabled(leader, leader_label)
    require_can_mit_enabled(follower, follower_label)
    leader_sample = read_sample(leader, leader_label)
    follower_sample = read_sample(follower, follower_label)
    residual = max(
        float(np.max(np.abs(leader_sample.q))),
        float(np.max(np.abs(follower_sample.q))),
    )
    if residual > HOME_TOLERANCE:
        raise TeleopAbort(
            f"{command_label} did not reach home: residual {residual:.3f} rad "
            f"> {HOME_TOLERANCE:.3f} rad"
        )
    print(
        f"{phase_label} COMPLETE: both arms home, residual "
        f"{residual:.3f} rad, peak tracking error {peak_track * R2D:.1f} deg"
    )
    return True


def recover_home_after_abort(
    leader,
    follower,
    leader_gravity: GravityCompensator,
    follower_gravity: GravityCompensator,
    leader_gripper,
    follower_gripper,
    args: argparse.Namespace,
) -> bool:
    """Hold for the configured delay, then slowly home after a runtime abort."""
    return home_both_with_mit(
        leader,
        follower,
        leader_gravity,
        follower_gravity,
        leader_gripper,
        follower_gripper,
        args,
        hold_seconds=args.abort_hold_seconds,
        home_speed=args.abort_home_speed,
        phase_label="ABORT RECOVERY",
    )


def print_calibration(label: str, gravity: GravityCompensator) -> None:
    print(f"  {label:<8} calibration={gravity.path}")
    print(
        f"           urdf={gravity.urdf}\n"
        f"           payload={gravity.payload_mass:.3f} kg  sign={gravity.sign:+.0f}  "
        f"scale=" + ", ".join(f"{value:.3f}" for value in gravity.scale)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--leader-can", default=LEADER_CAN)
    parser.add_argument("--follower-can", default=FOLLOWER_CAN)
    parser.add_argument(
        "--leader-calibration",
        type=Path,
        default=ASSET_ROOT / "calibration_leader.json",
    )
    parser.add_argument(
        "--follower-calibration",
        type=Path,
        default=ASSET_ROOT / "calibration_follower.json",
    )
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--leader-kd", type=float, default=DEFAULT_LEADER_KD)
    parser.add_argument("--follower-kp", type=float, default=DEFAULT_FOLLOWER_KP)
    parser.add_argument("--follower-kd", type=float, default=DEFAULT_FOLLOWER_KD)
    parser.add_argument(
        "--engage-seconds",
        type=float,
        default=2.0,
        help="quintic follower coupling ramp duration (default 2.0)",
    )
    parser.add_argument(
        "--start-home-speed",
        type=float,
        default=DEFAULT_START_HOME_SPEED,
        help="maximum quintic target speed while homing both arms before coupling",
    )
    parser.add_argument(
        "--max-start-gap",
        type=float,
        default=0.35,
        help="refuse coupling if the joint gap still exceeds this after homing",
    )
    parser.add_argument(
        "--max-track-error",
        type=float,
        default=0.35,
        help="abort if follower differs from its active MIT target by this many rad",
    )
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=DEFAULT_MAX_JOINT_SPEED,
        help="abort if either arm reports a faster joint, rad/s (default 4.0)",
    )
    parser.add_argument(
        "--max-reference-speed",
        type=float,
        default=3.0,
        help="clamp follower MIT v_des to this magnitude, rad/s",
    )
    parser.add_argument(
        "--feedback-timeout",
        type=float,
        default=0.20,
        help="abort when cached feedback timestamps stop advancing",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="0 runs until Ctrl-C; use a short duration for the first test",
    )
    parser.add_argument(
        "--abort-hold-seconds",
        type=float,
        default=DEFAULT_ABORT_HOLD_SECONDS,
        help="hold both current poses this long before abort recovery home",
    )
    parser.add_argument(
        "--abort-home-speed",
        type=float,
        default=DEFAULT_ABORT_HOME_SPEED,
        help="maximum quintic target speed during abort recovery, rad/s",
    )
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument(
        "--grip-force",
        type=float,
        default=1.0,
        help="force used by the follower gripper, N (default 1.0)",
    )
    parser.add_argument(
        "--grip-base",
        type=float,
        default=0.1,
        help="leader gripper free-space base force, N (default 0.1)",
    )
    parser.add_argument(
        "--grip-gain",
        type=float,
        default=1.0,
        help="gain on follower measured grip force reflected to leader",
    )
    parser.add_argument(
        "--grip-max-force",
        type=float,
        default=5.0,
        help="maximum force rendered by the leader gripper, N (default 5.0)",
    )
    parser.add_argument(
        "--telemetry-address",
        default="",
        help=(
            "opt-in localhost UDP stream for a read-only LeRobot recorder, "
            "for example udp://127.0.0.1:6670 (default off)"
        ),
    )
    parser.add_argument(
        "--quiet-status",
        action="store_true",
        help="suppress the periodic 0.5 s status line while retaining lifecycle output",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="ENERGIZE AND MOVE both arms; without this flag the script is read-only",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 10.0 <= args.rate <= 200.0:
        raise ValueError("--rate must be between 10 and 200 Hz")
    if not 0.0 <= args.leader_kd <= 5.0:
        raise ValueError("--leader-kd must be between 0 and 5")
    if not 0.0 < args.follower_kp <= 50.0:
        raise ValueError("--follower-kp must be in (0, 50]; start at the default 10")
    if not 0.0 < args.follower_kd <= 2.0:
        raise ValueError("--follower-kd must be in (0, 2]; start at the default 0.8")
    if args.engage_seconds < 0.5:
        raise ValueError("--engage-seconds must be at least 0.5")
    if not 0.05 <= args.start_home_speed <= 0.5:
        raise ValueError("--start-home-speed must be between 0.05 and 0.5 rad/s")
    if not 0.0 < args.max_start_gap <= 0.5:
        raise ValueError("--max-start-gap must be in (0, 0.5]")
    if not 0.0 < args.max_track_error <= 0.7:
        raise ValueError("--max-track-error must be in (0, 0.7]")
    if not 0.0 < args.max_joint_speed <= 5.0:
        raise ValueError("--max-joint-speed must be in (0, 5]")
    if not 0.0 < args.max_reference_speed <= 5.0:
        raise ValueError("--max-reference-speed must be in (0, 5]")
    if not 0.05 <= args.feedback_timeout <= 1.0:
        raise ValueError("--feedback-timeout must be between 0.05 and 1.0 s")
    if args.seconds < 0.0:
        raise ValueError("--seconds cannot be negative")
    if not 0.5 <= args.abort_hold_seconds <= 60.0:
        raise ValueError("--abort-hold-seconds must be between 0.5 and 60 s")
    if not 0.05 <= args.abort_home_speed <= 0.5:
        raise ValueError("--abort-home-speed must be between 0.05 and 0.5 rad/s")
    if not 0.0 < args.grip_force <= 5.0:
        raise ValueError("--grip-force must be in (0, 5] N")
    if not 0.0 < args.grip_base <= 1.0:
        raise ValueError("--grip-base must be in (0, 1] N")
    if not 0.0 <= args.grip_gain <= 5.0:
        raise ValueError("--grip-gain must be in [0, 5]")
    if not args.grip_base <= args.grip_max_force <= 10.0:
        raise ValueError("--grip-max-force must be >= --grip-base and <= 10 N")
    if args.telemetry_address:
        parse_udp_endpoint(args.telemetry_address)


def run(args: argparse.Namespace) -> int:
    global _running
    _running = True
    validate_args(args)

    leader_gravity = GravityCompensator(args.leader_calibration)
    follower_gravity = GravityCompensator(args.follower_calibration)
    print("gravity models:")
    print_calibration("leader", leader_gravity)
    print_calibration("follower", follower_gravity)
    print(
        f"\ncontroller: {args.rate:.0f} Hz, leader kp=0 kd={args.leader_kd}, "
        f"follower kp={args.follower_kp} kd={args.follower_kd}"
    )
    print(
        f"  follower startup coupling: quintic {args.engage_seconds:.1f}s, "
        f"max post-home gap {args.max_start_gap:.3f} rad"
    )
    print(
        f"  both-arm startup home: MIT quintic "
        f"<= {args.start_home_speed:.2f} rad/s"
    )
    print("  planner calls: none; arm force reflection/NEXT: none; gripper feedback: direct")

    print(f"\nconnecting leader={args.leader_can}, follower={args.follower_can}")
    leader = follower = None
    leader_gripper = follower_gripper = None
    leader_grip_sample: Optional[GripperSample] = None
    follower_grip_sample: Optional[GripperSample] = None
    last_leader: Optional[ArmSample] = None
    last_follower: Optional[ArmSample] = None
    telemetry: Optional[MITTelemetryPublisher] = None
    aborted: Optional[str] = None
    commands = overruns = 0
    peak_track = 0.0
    started = False
    teleop_started = False
    recovered_home = False

    if args.execute and args.telemetry_address:
        telemetry = MITTelemetryPublisher(args.telemetry_address)
        print(
            f"  read-only telemetry: {args.telemetry_address} "
            "(best effort, no CAN access by recorder)"
        )

    try:
        leader = build_arm(args.leader_can)
        follower = build_arm(args.follower_can)
        reject_teaching_mode(leader, "leader")
        reject_teaching_mode(follower, "follower")

        last_leader = read_sample(leader, "leader")
        last_follower = read_sample(follower, "follower")
        start_gap = float(np.max(np.abs(last_leader.q - last_follower.q)))
        leader_mode = arm_status_text(leader)
        follower_mode = arm_status_text(follower)
        print(
            f"  leader   ctrl={leader_mode[0]} mode={leader_mode[1]} "
            f"q(deg)=" + " ".join(f"{v * R2D:+6.1f}" for v in last_leader.q)
        )
        print(
            f"  follower ctrl={follower_mode[0]} mode={follower_mode[1]} "
            f"q(deg)=" + " ".join(f"{v * R2D:+6.1f}" for v in last_follower.q)
        )
        print(f"  initial max joint gap={start_gap:.3f} rad ({start_gap * R2D:.1f} deg)")

        leader_tau = leader_gravity.torque(last_leader.q) - args.leader_kd * last_leader.qd
        follower_tau = follower_gravity.torque(last_follower.q)
        bounded_torque(leader_tau, "leader")
        bounded_torque(follower_tau, "follower")
        print("  leader gravity+damping now (N*m):   "
              + " ".join(f"{v:+6.2f}" for v in leader_tau))
        print("  follower gravity now (N*m):         "
              + " ".join(f"{v:+6.2f}" for v in follower_tau))

        if not args.no_gripper:
            try:
                leader_gripper = leader.init_effector("agx_gripper")
                follower_gripper = follower.init_effector("agx_gripper")
                time.sleep(0.5)
                leader_grip_sample = read_gripper(leader_gripper)
                follower_grip_sample = read_gripper(follower_gripper)
                print(
                    f"  gripper leader={getattr(leader_grip_sample, 'width', None)} "
                    f"follower={getattr(follower_grip_sample, 'width', None)}"
                )
            except Exception as exc:
                print(f"  gripper init failed ({exc}); continuing without gripper")
                leader_gripper = follower_gripper = None

        if not args.execute:
            print("\nREAD-ONLY CHECK COMPLETE: nothing enabled and no command was sent.")
            print("The reported initial gap is allowed: --execute homes both arms first.")
            print("Add --execute only with both arms supported and the E-stop in hand.")
            return 0

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
        print(
            "\n*** BOTH ARMS WILL ENTER MIT CONTROL AND GO HOME. "
            "SUPPORT THEM; E-STOP IN HAND. ***"
        )
        for remaining in range(5, 0, -1):
            print(f"    starting in {remaining} ...")
            time.sleep(1.0)
            if not _running:
                raise TeleopAbort("startup cancelled by signal")

        # Keep both devices in the broadcasting/PC-controlled linkage setting.
        # Do not call set_motion_mode(): move_mit performs the transition and an
        # explicit set_motion_mode has de-energized this hardware before.
        leader.set_follower_mode()
        follower.set_follower_mode()
        time.sleep(0.3)

        last_leader, last_follower = enable_both_with_safe_commands(
            leader,
            follower,
            leader_gravity,
            follower_gravity,
            args.leader_kd,
            args.follower_kp,
            args.follower_kd,
            args.max_reference_speed,
        )
        started = True
        print("  both arms enabled in CAN/MIT; repeated auto mode frames disabled")
        print("  both arms are receiving position impedance + gravity support")

        homed = home_both_with_mit(
            leader,
            follower,
            leader_gravity,
            follower_gravity,
            leader_gripper,
            follower_gripper,
            args,
            hold_seconds=0.0,
            home_speed=args.start_home_speed,
            phase_label="STARTUP HOME",
        )
        if not homed:
            raise TeleopAbort("startup home cancelled before coupling")

        last_leader = read_sample(leader, "leader post-home")
        last_follower = read_sample(follower, "follower post-home")
        post_home_gap = float(np.max(np.abs(last_leader.q - last_follower.q)))
        if post_home_gap > args.max_start_gap:
            raise TeleopAbort(
                f"post-home gap {post_home_gap:.3f} rad exceeds "
                f"--max-start-gap {args.max_start_gap:.3f}"
            )
        follower_start = last_follower.q.copy()
        teleop_started = True

        print(f"  post-home leader/follower gap={post_home_gap:.3f} rad")
        print("  startup home complete; leader now changes to gravity-only control")
        print(
            f"  engaging follower over {args.engage_seconds:.1f}s; "
            "hold the leader steady until ENGAGED"
        )

        leader_fresh = FeedbackWatchdog(args.feedback_timeout)
        follower_fresh = FeedbackWatchdog(args.feedback_timeout)

        dt = 1.0 / args.rate
        start = time.perf_counter()
        next_tick = start
        last_report = start
        last_status_check = start
        engaged_reported = False
        grip_difference = 0.0
        grip_measured_force = 0.0
        grip_rendered_force = 0.0

        while _running:
            now = time.perf_counter()
            elapsed = now - start
            if args.seconds > 0.0 and elapsed >= args.seconds:
                break

            last_leader = read_sample(leader, "leader")
            last_follower = read_sample(follower, "follower")
            leader_fresh.observe(last_leader.stamps, now)
            follower_fresh.observe(last_follower.stamps, now)

            q_des, qd_des = engagement_reference(
                follower_start,
                last_leader.q,
                last_leader.qd,
                elapsed,
                args.engage_seconds,
            )
            send_leader(
                leader,
                last_leader,
                leader_gravity,
                args.leader_kd,
            )
            send_follower(
                follower,
                last_follower,
                q_des,
                qd_des,
                follower_gravity,
                args.follower_kp,
                args.follower_kd,
                args.max_reference_speed,
            )

            leader_grip_sample = None
            follower_grip_sample = None
            if leader_gripper is not None and follower_gripper is not None:
                leader_grip_sample = read_gripper(leader_gripper)
                follower_grip_sample = read_gripper(follower_gripper)
                if leader_grip_sample is not None and follower_grip_sample is not None:
                    (grip_difference,
                     grip_measured_force,
                     grip_rendered_force) = send_grippers(
                        leader_gripper,
                        follower_gripper,
                        leader_grip_sample,
                        follower_grip_sample,
                        args.grip_force,
                        args.grip_base,
                        args.grip_gain,
                        args.grip_max_force,
                    )

            track = float(np.max(np.abs(last_follower.q - q_des)))
            peak_track = max(peak_track, track)
            if track > args.max_track_error:
                joint = int(np.argmax(np.abs(last_follower.q - q_des))) + 1
                raise TeleopAbort(
                    f"follower J{joint} tracking error {track:.3f} rad exceeds "
                    f"{args.max_track_error:.3f} rad"
                )

            speed = max(
                float(np.max(np.abs(last_leader.qd))),
                float(np.max(np.abs(last_follower.qd))),
            )
            if speed > args.max_joint_speed:
                raise TeleopAbort(
                    f"measured joint speed {speed:.3f} rad/s exceeds "
                    f"{args.max_joint_speed:.3f} rad/s"
                )

            if telemetry is not None:
                telemetry.publish(
                    phase=(
                        "engaged"
                        if elapsed >= args.engage_seconds
                        else "engaging"
                    ),
                    leader_q=last_leader.q,
                    leader_qd=last_leader.qd,
                    follower_q=last_follower.q,
                    follower_qd=last_follower.qd,
                    follower_joint_effort=last_follower.joint_effort,
                    follower_target_q=q_des,
                    follower_target_qd=qd_des,
                    leader_gripper_width=(
                        None
                        if leader_grip_sample is None
                        else leader_grip_sample.width
                    ),
                    follower_gripper_width=(
                        None
                        if follower_grip_sample is None
                        else follower_grip_sample.width
                    ),
                    follower_gripper_target=(
                        None
                        if leader_grip_sample is None
                        else leader_grip_sample.width
                    ),
                    follower_gripper_force=(
                        None
                        if follower_grip_sample is None
                        else follower_grip_sample.force
                    ),
                    overruns=overruns,
                )

            commands += 1
            if not engaged_reported and elapsed >= args.engage_seconds:
                engaged_reported = True
                print("  ENGAGED: follower target is now the live leader joint pose")

            if now - last_status_check >= 0.5:
                last_status_check = now
                require_can_mit_enabled(leader, "leader")
                require_can_mit_enabled(follower, "follower")

            if not args.quiet_status and now - last_report >= 0.5:
                last_report = now
                phase = "ENGAGED" if elapsed >= args.engage_seconds else "engaging"
                grip_report = ""
                if leader_gripper is not None and follower_gripper is not None:
                    grip_report = (
                        f" grip_diff={grip_difference * 1000:+5.1f}mm"
                        f" measured={grip_measured_force:.2f}N"
                        f" rendered={grip_rendered_force:.2f}N"
                    )
                print(
                    f"  t={elapsed:6.1f}s {phase:8s} "
                    f"track={track * R2D:5.1f}deg "
                    f"|qd|max={speed:4.2f}rad/s overruns={overruns}"
                    f"{grip_report}"
                )

            next_tick += dt
            slack = next_tick - time.perf_counter()
            if slack > 0.0:
                time.sleep(slack)
            else:
                overruns += 1
                next_tick = time.perf_counter()

    except (TeleopAbort, ValueError, FileNotFoundError) as exc:
        aborted = str(exc)
        print(f"\nABORTED: {aborted}")
        if telemetry is not None:
            telemetry.publish_status("aborted", aborted)
    finally:
        if telemetry is not None and aborted is None:
            telemetry.publish_status("stopped", "teleoperation ended")

        # A runtime abort first holds both current poses for the configured
        # delay, then homes both through a slow MIT quintic. Recovery is skipped
        # when the abort itself means closed-loop motion is no longer trustworthy.
        if started and leader is not None and follower is not None:
            if aborted and teleop_started:
                try:
                    recovered_home = recover_home_after_abort(
                        leader,
                        follower,
                        leader_gravity,
                        follower_gravity,
                        leader_gripper,
                        follower_gripper,
                        args,
                    )
                except Exception as exc:
                    print(f"ABORT RECOVERY SKIPPED/FAILED: {exc}")
            elif aborted:
                print(
                    "ABORT RECOVERY NOT STARTED: startup coupling never began; "
                    "holding the current pose"
                )

            if not recovered_home:
                # Never disable either arm here: that would make it fall. If
                # abort recovery was impossible, freeze both at their current
                # poses. A normal Ctrl-C keeps the original behaviour: leader
                # gravity-only, follower holding its current pose.
                print("\nsettling into final MIT hold (no force feedback) ...")
                try:
                    hold_start = time.perf_counter()
                    leader_hold = read_sample(leader, "leader").q.copy()
                    follower_hold = read_sample(follower, "follower").q.copy()
                    while time.perf_counter() - hold_start < 0.5:
                        last_leader = read_sample(leader, "leader")
                        last_follower = read_sample(follower, "follower")
                        if aborted:
                            send_position_impedance(
                                leader,
                                last_leader,
                                leader_hold,
                                np.zeros(N_JOINTS),
                                leader_gravity,
                                args.follower_kp,
                                args.follower_kd,
                                args.max_reference_speed,
                                "leader final hold",
                            )
                        else:
                            send_leader(
                                leader,
                                last_leader,
                                leader_gravity,
                                args.leader_kd,
                            )
                        send_position_impedance(
                            follower,
                            last_follower,
                            follower_hold,
                            np.zeros(N_JOINTS),
                            follower_gravity,
                            args.follower_kp,
                            args.follower_kd,
                            args.max_reference_speed,
                            "follower final hold",
                        )
                        send_gripper_hold(
                            leader_gripper,
                            follower_gripper,
                            args.grip_force,
                            args.grip_base,
                        )
                        time.sleep(1.0 / args.rate)
                except Exception as exc:
                    print(f"  final hold warning: {exc}")

        for arm in (leader, follower):
            if arm is not None:
                try:
                    arm.disconnect()
                except Exception as exc:
                    print(f"disconnect warning: {exc}")

        if telemetry is not None:
            telemetry.close()

    if started:
        print(
            f"\ncommands={commands}, overruns={overruns}, "
            f"peak follower target error={peak_track * R2D:.1f} deg"
        )
        if recovered_home:
            print("BOTH ARMS ARE HOME AND HOLDING IN MIT.")
        elif aborted:
            print("BOTH ARMS HAVE A FROZEN FINAL MIT POSITION HOLD.")
        else:
            print("FOLLOWER IS HOLDING. LEADER HAS A FROZEN FINAL GRAVITY COMMAND.")
        print("Support both arms before disabling torque.")
    if telemetry is not None:
        telemetry_error = (
            "" if telemetry.last_error is None else f", last_error={telemetry.last_error}"
        )
        print(
            f"telemetry packets sent={telemetry.sent}, dropped={telemetry.dropped}"
            f"{telemetry_error}"
        )
    return 1 if aborted else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
