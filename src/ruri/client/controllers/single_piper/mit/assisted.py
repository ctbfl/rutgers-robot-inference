"""Persistent assisted MIT worker; --simulate exercises the same protocol/core.

Startup enables into HOLD and optionally homes the connected arms before READY.
Timeout and normal exit hold; runtime aborts use teleop's MIT hold/home recovery.
Only this process opens CAN; leader_only opens only the registered right arm. The optional simulation path opens
neither CAN nor cameras and is suitable for client/server integration tests.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import time

import numpy as np

from ...single_piper_assisted.config import SinglePiperAssistedConfig
from ...single_piper_assisted.control import (
    AssistedControl,
    clamp_pose,
    physical_bounds,
    pose,
)
from ..calibration_ranges import measured_for_arm
from ..discovery import read_can_hardware_id
from ..hardware_registry import find_arm_by_hardware_id, find_arm_registration
from . import leader_follower as mit
from .telemetry import MITTelemetryPublisher, parse_udp_endpoint


class SimulatedRig:
    def __init__(self, config):
        self.config = config
        self.leader_name, self.follower_name = "right_arm", "left_arm"
        self.leader = np.asarray([0.0, 0.4, -0.4, 0.0, 0.0, 0.0, 0.04])
        self.follower = self.leader.copy()
        if config.leader_only:
            self.follower_name = "right_arm"
        self.previous = self.leader.copy(), self.follower.copy()
        self.samples = None
        self.enabled = False
        self.human_until = 0.0

    def read(self):
        return self.leader.copy(), (self.leader if self.config.leader_only else self.follower).copy()

    def enable(self, should_stop=lambda: False):
        self.enabled = not should_stop()

    def home(self, should_stop=lambda: False):
        starts = self.read()
        duration = mit.recovery_home_duration(
            starts[0][:6], starts[1][:6], self.config.start_home_speed
        )
        began = time.monotonic()
        print(
            f"STARTUP HOME: simulated MIT home over {duration:.1f}s", flush=True
        )
        while True:
            if should_stop():
                raise mit.TeleopAbort("startup home cancelled")
            elapsed = time.monotonic() - began
            for arm, start in zip((self.leader, self.follower), starts):
                arm[:6], _ = mit.home_reference(start[:6], elapsed, duration)
            if elapsed >= duration:
                break
            time.sleep(1.0 / self.config.mit_rate_hz)
        self.initial_hold = self.read()
        print("STARTUP HOME COMPLETE: right simulated leader home" if self.config.leader_only
              else "STARTUP HOME COMPLETE: both simulated arms home", flush=True)

    def emergency_stop(self):
        self.enabled = False

    def send(self, control, dt):
        self.previous = self.leader.copy(), self.follower.copy()
        # Kinematic simulation only: zero leader stiffness must not secretly
        # keep a position servo running during the gravity-only phase.
        if time.monotonic() >= self.human_until:
            scale = 1.-control.reference.alpha if self.config.uses_gravity_handoff else 1.
            self.leader[:6] += np.clip(control.leader_target[:6]-self.leader[:6], -dt*scale, dt*scale)
            self.leader[6] += np.clip(control.leader_target[6]-self.leader[6], -dt, dt)
        if not self.config.leader_only:
            self.follower += np.clip(control.target - self.follower, -dt, dt)

    def human(self, target, duration_s=0.0):
        if not np.isfinite(duration_s) or not 0 <= duration_s <= 10.0:
            raise ValueError("simulated human duration must be in [0,10] seconds")
        self.leader = pose(target)
        self.human_until = time.monotonic() + duration_s

    def telemetry(self):
        return {
            "leader_q": self.leader[:6],
            "leader_qd": np.zeros(6),
            "follower_q": None if self.config.leader_only else self.follower[:6],
            "follower_qd": None if self.config.leader_only else np.zeros(6),
            "follower_joint_effort": None if self.config.leader_only else np.zeros(6),
            "leader_gripper_width": self.leader[6],
            "follower_gripper_width": None if self.config.leader_only else self.follower[6],
            "follower_gripper_force": None if self.config.leader_only else 0.0,
        }

    def control_telemetry(self):
        return {"leader_joint_effort": [0.] * 6, "leader_gripper_force": 0.}

    def close(self):
        pass


class NominalLimitedArm:
    """Apply the teleop nominal clamp before the arm's hardware clamp, on every MIT write."""

    def __init__(self, arm):
        self._arm = arm

    def __getattr__(self, name):
        return getattr(self._arm, name)

    def move_mit(self, joint_index, p_des, **kwargs):
        i = joint_index - 1
        target = float(
            np.clip(p_des, mit.JOINT_LIMIT_LOWER[i], mit.JOINT_LIMIT_UPPER[i])
        )
        return self._arm.move_mit(joint_index=joint_index, p_des=target, **kwargs)


class LimitedGripper:
    """Nominal then measured limits for every gripper write, including shared homing."""

    def __init__(self, gripper, arm_name, gripper_max_m=None):
        self._gripper = gripper
        lower, upper = physical_bounds(measured_for_arm(arm_name))
        self._lower, self._upper = lower[6], upper[6]
        self._nominal_upper = mit.GRIPPER_LIMIT_UPPER
        if gripper_max_m is not None:
            self._lower, self._upper = 0., gripper_max_m
            self._nominal_upper = gripper_max_m

    def __getattr__(self, name):
        return getattr(self._gripper, name)

    def move_gripper_m(self, width, force):
        nominal = np.clip(width, mit.GRIPPER_LIMIT_LOWER, self._nominal_upper)
        self._gripper.move_gripper_m(
            float(np.clip(nominal, self._lower, self._upper)), force
        )


class PiperRig:
    def __init__(self, config, leader_can, follower_can):
        self.config = config
        self.leader = self.follower = None
        self.leader_gripper = self.follower_gripper = None
        self.samples = self.grips = None
        self.energized = self.enabled = False
        self.startup_phase = "opening feedback"
        try:
            registrations = []
            for channel, side, role in (
                (leader_can, config.leader_side, config.leader_role),
                (follower_can, config.follower_side, config.follower_role),
            ):
                actual = find_arm_by_hardware_id(read_can_hardware_id(channel))
                expected = find_arm_registration(side, role)
                if actual != expected:
                    raise ValueError(
                        f"{channel} resolves to {actual.name}; expected {expected.name}"
                    )
                registrations.append(actual)
            if leader_can == follower_can:
                raise ValueError("leader and follower cannot share a CAN interface")
            self.leader_name, self.follower_name = (r.name for r in registrations)
            self.leader = NominalLimitedArm(mit.build_arm(leader_can))
            self.follower = NominalLimitedArm(mit.build_arm(follower_can))
            mit.reject_teaching_mode(self.leader, "leader")
            mit.reject_teaching_mode(self.follower, "follower")
            self.leader_gravity = mit.GravityCompensator(
                mit.ASSET_ROOT / "calibration_leader.json"
            )
            self.follower_gravity = mit.GravityCompensator(
                mit.ASSET_ROOT / "calibration_follower.json"
            )
            self.leader_gripper = LimitedGripper(
                self.leader.init_effector("agx_gripper"), self.leader_name, config.gripper_max_m
            )
            self.follower_gripper = LimitedGripper(
                self.follower.init_effector("agx_gripper"), self.follower_name, config.gripper_max_m
            )
            time.sleep(0.5)
            self.fresh = [
                [mit.FeedbackWatchdog(config.feedback_timeout_s) for _ in range(8)]
                for _ in range(2)
            ]
            self.last_status = 0.0
            self.enabled = False
        except BaseException:
            self.close()
            raise

    def read(self, *, check_speed=True):
        devices = [(self.leader, self.leader_gripper, "leader")]
        if self.follower is not None:
            devices.append((self.follower, self.follower_gripper, "follower"))
        self.samples = tuple(mit.read_sample(arm, label) for arm, _, label in devices)
        self.grips = tuple(mit.read_gripper(grip) for _, grip, _ in devices)
        if any(g is None for g in self.grips):
            raise mit.TeleopAbort("all connected gripper feedback streams are required")
        now = time.perf_counter()
        for label, sample, grip, watches in zip(
            ("leader", "follower"), self.samples, self.grips, self.fresh
        ):
            if not np.all(
                np.isfinite((*sample.stamps, grip.timestamp, grip.width, grip.force))
            ):
                raise mit.TeleopAbort(
                    "non-finite feedback timestamps or gripper feedback"
                )
            # A live gripper must not hide a stalled joint feedback stream.
            for stream, stamp, watch in zip(
                (
                    "joint angles",
                    "motor 1",
                    "motor 2",
                    "motor 3",
                    "motor 4",
                    "motor 5",
                    "motor 6",
                    "gripper",
                ),
                (*sample.stamps, grip.timestamp),
                watches,
                strict=True,
            ):
                try:
                    watch.observe((stamp,), now)
                except mit.TeleopAbort as exc:
                    raise mit.TeleopAbort(f"{label} {stream}: {exc}") from exc
            if check_speed and np.max(np.abs(sample.qd)) > self.config.max_joint_speed:
                joint = int(np.argmax(np.abs(sample.qd)))
                raise mit.TeleopAbort(
                    f"{label} J{joint + 1} measured speed {sample.qd[joint]:.3f} rad/s "
                    f"exceeded {self.config.max_joint_speed:.3f} rad/s limit"
                )
        values = tuple(np.r_[s.q, g.width] for s, g in zip(self.samples, self.grips))
        # Core accepts leader and controlled observations; single-arm uses the
        # same measurement, with no second driver or second set of commands.
        return values if len(values) == 2 else (values[0], values[0].copy())

    def _prepare_stops(self, should_stop):
        """Reset only latched stops, once, when this startup explicitly requests it."""
        for arm, label in ((self.leader, "right leader"), (self.follower, "left follower")):
            if arm is None:
                continue
            if should_stop():
                raise mit.TeleopAbort("startup cancelled")
            status = mit.arm_status_text(arm)[2]
            if "EMERGENCY_STOP" not in status:
                continue
            if not self.config.reset_latched_stop:
                raise mit.TeleopAbort(
                    f"{label} has a latched electronic stop: {status}; "
                    "explicit --reset-stop is required to reset it before enable/home"
                )
            stamp = float(arm.get_arm_status().timestamp)
            if not np.isfinite(stamp):
                raise mit.TeleopAbort(f"{label} stop status timestamp is non-finite")
            print(f"Explicit startup reset: {label} electronic stop", flush=True)
            self.energized = True  # A failed reset must still leave the arm stopped.
            arm.reset()
            deadline = time.monotonic()+2.
            while time.monotonic() < deadline:
                if should_stop():
                    raise mit.TeleopAbort("startup cancelled during stop reset")
                feedback = arm.get_arm_status()
                if feedback is not None:
                    current_stamp = float(feedback.timestamp)
                    if (np.isfinite(current_stamp) and current_stamp > stamp
                            and "NORMAL" in str(feedback.msg.arm_status)
                            and feedback.msg.err_code == 0):
                        break
                time.sleep(.01)
            else:
                raise mit.TeleopAbort(f"{label} stop reset lacked fresh NORMAL feedback")

    def enable(self, should_stop=lambda: False):
        if should_stop():
            raise mit.TeleopAbort("startup cancelled")
        # Use the same role/enable/move_mit sequence as persistent teleop.
        # Ordinary startup never clears a latched stop. A manual --reset-stop
        # applies only here, never to runtime faults or a background retry.
        self._prepare_stops(should_stop)
        self.startup_phase = "role setup"
        self.leader.set_follower_mode()
        self.follower.set_follower_mode()
        time.sleep(0.3)
        self.fresh = [
            [mit.FeedbackWatchdog(self.config.feedback_timeout_s) for _ in range(8)]
            for _ in range(2)
        ]
        # Capture AFTER role setup: the old pre-settle pose can already differ
        # from the arm we are about to support. Reuse these exact references in
        # the first persistent HOLD tick.
        self.initial_hold = tuple(clamp_pose(value, self.config.gripper_max_m) for value in self.read())

        def read_feedback():
            self.read()
            return self.samples

        def before_enable():
            self.energized = True

        c = self.config
        self.startup_phase = "CAN/MIT enable"
        mit.enable_both_with_safe_commands(
            self.leader,
            self.follower,
            self.leader_gravity,
            self.follower_gravity,
            c.leader_kd,
            c.follower_kp,
            c.follower_kd,
            c.max_reference_speed,
            leader_hold=self.initial_hold[0][:6],
            leader_hold_kp=c.leader_policy_kp,
            leader_hold_kd=c.leader_policy_kd,
            follower_hold=self.initial_hold[1][:6],
            read_feedback=read_feedback,
            should_stop=should_stop,
            before_enable=before_enable,
            rate_hz=c.mit_rate_hz,
        )
        # A cached CAN/MIT bit is insufficient. Keep streaming support after
        # auto-mode frames are disabled, and require healthy, advancing status
        # from BOTH arms before advertising READY.
        hold = AssistedControl(
            c,
            *self.initial_hold,
            leader_name=self.leader_name,
            follower_name=self.follower_name,
        )
        hold.leader_gain = 1.0
        self.startup_phase = "supported hold verification"
        self.enabled = True
        deadline = time.monotonic() + c.feedback_timeout_s
        status_stamps = [
            arm.get_arm_status().timestamp for arm in (self.leader, self.follower)
        ]
        advanced = [False, False]
        while time.monotonic() < deadline:
            if should_stop():
                raise mit.TeleopAbort("startup cancelled")
            self.read()
            self.send(hold, 1.0 / c.mit_rate_hz)
            for i, (arm, label) in enumerate(
                ((self.leader, "leader"), (self.follower, "follower"))
            ):
                mit.require_can_mit_enabled(arm, label)
                stamp = arm.get_arm_status().timestamp
                if not np.isfinite(stamp):
                    raise mit.TeleopAbort(f"{label} arm status timestamp is non-finite")
                advanced[i] |= stamp != status_stamps[i]
            time.sleep(1.0 / c.mit_rate_hz)
        if not all(advanced):
            raise mit.TeleopAbort(
                "startup arm status timestamps did not advance on both arms"
            )
        self.startup_phase = "running"

    def _home_args(self):
        c = self.config
        return argparse.Namespace(
            rate=c.mit_rate_hz,
            feedback_timeout=c.feedback_timeout_s,
            follower_kp=c.follower_kp,
            follower_kd=c.follower_kd,
            leader_home_kp=c.leader_policy_kp if c.uses_gravity_handoff else c.follower_kp,
            leader_home_kd=c.leader_policy_kd if c.uses_gravity_handoff else c.follower_kd,
            max_reference_speed=c.max_reference_speed,
            max_track_error=c.max_track_error,
            max_joint_speed=c.max_joint_speed,
            grip_force=c.grip_force,
            grip_base=c.grip_base,
            abort_hold_seconds=c.abort_hold_seconds,
            abort_home_speed=c.abort_home_speed,
        )

    def home(self, should_stop=lambda: False):
        self.startup_phase = "startup home"
        c = self.config

        def read_feedback():
            self.read()  # Keep every joint/motor/gripper watchdog active during home.
            return self.samples

        if not mit.home_both_with_mit(
            self.leader,
            self.follower,
            self.leader_gravity,
            self.follower_gravity,
            self.leader_gripper,
            self.follower_gripper,
            self._home_args(),
            hold_seconds=0.0,
            home_speed=c.start_home_speed,
            phase_label="STARTUP HOME",
            read_feedback=read_feedback,
            should_stop=should_stop,
        ):
            raise mit.TeleopAbort("startup home cancelled")
        # Preserve zero references when entering the persistent loop. Reusing
        # the pre-home holds here would command both arms back to their starts.
        leader, follower = self.read()
        self.initial_hold = (
            np.r_[np.zeros(6), leader[6]],
            np.r_[np.zeros(6), follower[6]],
        )
        self.startup_phase = "running"

    def recover_home_after_abort(self, *, should_stop, on_progress):
        self.startup_phase = "abort recovery"

        def read_feedback():
            # The proven recovery loop allows one second to brake in HOLD,
            # then enforces speed/tracking limits throughout the home path.
            # Keep every feedback stream's watchdog active here as well.
            self.read(check_speed=False)
            on_progress()
            return self.samples

        return mit.recover_home_after_abort(
            self.leader,
            self.follower,
            self.leader_gravity,
            self.follower_gravity,
            self.leader_gripper,
            self.follower_gripper,
            self._home_args(),
            read_feedback=read_feedback,
            should_stop=should_stop,
        )

    def _send_leader_handoff(self, control):
        """The same validated gravity endpoint for one- and two-arm handoff."""
        c = self.config
        alpha = control.reference.alpha if control.mode in ("POLICY", "HUMAN") else 0.
        if not np.isfinite(alpha) or not 0. <= alpha <= 1.:
            raise mit.TeleopAbort("right leader invalid handoff weight")
        velocity = np.asarray(c.leader_sign)*control.reference_velocity
        if alpha == 1.:
            mit.send_leader(self.leader, self.samples[0], self.leader_gravity, c.leader_kd)
        elif alpha == 0.:
            mit.send_position_impedance(
                self.leader, self.samples[0], control.leader_target[:6],
                velocity, self.leader_gravity,
                c.leader_policy_kp, c.leader_policy_kd, c.max_reference_speed, "right leader",
            )
        else:
            torque = mit.bounded_torque(
                self.leader_gravity.torque(self.samples[0].q)-alpha*c.leader_kd*self.samples[0].qd,
                "right leader handoff",
            )
            target = mit.joint_vector(control.leader_target[:6], "right leader target")
            velocity = np.clip(mit.joint_vector(velocity, "right leader velocity"),
                               -c.max_reference_speed, c.max_reference_speed)
            for i, joint in enumerate(mit.JOINTS):
                self.leader.move_mit(joint_index=joint, p_des=float(target[i]), v_des=float(velocity[i]),
                                     kp=(1.-alpha)*c.leader_policy_kp,
                                     kd=(1.-alpha)*c.leader_policy_kd, t_ff=float(torque[i]))

    def send(self, control, dt):
        del dt
        c = self.config
        # The legacy shared-reference mode retains its existing check. Gravity
        # handoff permits intentional pose gaps; do not reintroduce its removed
        # raw-error abort. Feedback, speed and feed-forward checks remain active.
        if not c.uses_gravity_handoff:
            for sample, target, label in ((self.samples[0], control.leader_target, "leader"),
                                          (self.samples[1], control.target, "follower")):
                error = float(np.max(np.abs(sample.q-target[:6])))
                if error > c.max_track_error:
                    raise mit.TeleopAbort(f"{label} reference tracking error {error:.3f} rad")
        start_ns = time.monotonic_ns()
        if c.uses_gravity_handoff:
            self._send_leader_handoff(control)
        else:
            mit.send_position_impedance(
                self.leader, self.samples[0], control.leader_target[:6],
                np.asarray(c.leader_sign)*control.reference_velocity,
                self.leader_gravity, c.leader_policy_kp, c.leader_policy_kd,
                c.max_reference_speed, "leader",
            )
        mit.send_position_impedance(
            self.follower,
            self.samples[1],
            control.target[:6],
            control.follower_velocity if c.uses_gravity_handoff else control.reference_velocity,
            self.follower_gravity,
            c.follower_kp,
            c.follower_kd,
            c.max_reference_speed,
            "follower",
        )
        self._send_gripper(
            self.follower_gripper, self.follower_name, control.target[6], c.grip_force
        )
        gripper_phase = control.gripper_handoff.phase if control.gripper_handoff else None
        if (control.mode == "HUMAN" or control.correcting) and gripper_phase != "aligning":
            force = min(
                c.grip_max_force, c.grip_base + c.grip_gain * abs(self.grips[1].force)
            )
            # Gravity HUMAN follows the absolute leader width. Reflect at the
            # same coordinate: retaining a 100mm leader / 40mm follower entry
            # offset would keep pushing the leader 60mm wider on every tick.
            # Legacy relative teleop keeps its original anchor by default.
            width = (
                self.grips[1].width
                if control.correcting or (c.uses_gravity_handoff and c.human_gripper_absolute_teleop)
                else (
                    control.anchor_leader[6]
                    + self.grips[1].width
                    - control.anchor_target[6]
                )
            )
            self._send_gripper(self.leader_gripper, self.leader_name, width, force)
        else:
            force = (control.gripper_handoff.opening_force(self.grips[0].width)
                     if gripper_phase in ("opening", "armed")
                     else c.grip_force if gripper_phase == "aligning" else c.grip_base)
            self._send_gripper(
                self.leader_gripper,
                self.leader_name,
                control.leader_target[6],
                force,
            )
        self.send_span_ns = time.monotonic_ns() - start_ns
        # Match the single-arm worker: check mode after this tick's MIT writes,
        # not while reading feedback before the first supported control tick.
        now = time.perf_counter()
        if self.enabled and now - self.last_status >= 0.1:
            for arm, label in ((self.leader, "leader"), (self.follower, "follower")):
                try:
                    mit.require_can_mit_enabled(arm, label)
                except mit.TeleopAbort as exc:
                    # Capture the controller's reason before emergency_stop()
                    # changes its feedback to our own stop condition.
                    msg = arm.get_arm_status().msg
                    raise mit.TeleopAbort(
                        f"{exc}; status={msg.arm_status}, "
                        f"teach={getattr(msg, 'teach_status', 'unavailable')}, "
                        f"err_code={getattr(msg, 'err_code', 'unavailable')}"
                    ) from exc
            self.last_status = now

    def _send_gripper(self, gripper, arm_name, width, force):
        low, high = physical_bounds(measured_for_arm(arm_name))
        if self.config.gripper_max_m is not None:
            gripper.move_gripper_m(float(np.clip(width, 0., self.config.gripper_max_m)), force)
            return
        nominal = np.clip(width, mit.GRIPPER_LIMIT_LOWER, mit.GRIPPER_LIMIT_UPPER)
        gripper.move_gripper_m(float(np.clip(nominal, low[6], high[6])), force)

    def control_telemetry(self):
        return {"leader_joint_effort": self.samples[0].joint_effort.tolist(),
                "leader_gripper_force": float(self.grips[0].force)}

    def telemetry(self):
        a, b = self.samples
        ga, gb = self.grips
        return {
            "leader_q": a.q,
            "leader_qd": a.qd,
            "follower_q": b.q,
            "follower_qd": b.qd,
            "follower_joint_effort": b.joint_effort,
            "leader_gripper_width": ga.width,
            "follower_gripper_width": gb.width,
            "follower_gripper_force": gb.force,
        }

    def close(self):
        for arm in (self.leader, self.follower):
            if arm is not None:
                try:
                    arm.disconnect()
                except Exception as exc:
                    print(f"CAN disconnect failed: {exc}", flush=True)

    def emergency_stop(self):
        if not self.energized:
            return
        # SDK's damped electronic stop does not require a valid measured pose.
        # Delivery cannot be guaranteed on a failed CAN bus; no automatic reset.
        for arm in (self.leader, self.follower):
            if arm is not None:
                try:
                    arm.electronic_emergency_stop()
                except Exception as exc:
                    print(f"Electronic stop delivery failed: {exc}", flush=True)


class LeaderRig(PiperRig):
    """One real right arm, using the shared-reference core and V189 MIT path."""

    def __init__(self, config, leader_can):
        self.config = config
        self.leader = self.follower = None
        self.leader_gripper = self.follower_gripper = None
        self.samples = self.grips = None
        self.energized = self.enabled = False
        self.startup_phase = "opening right leader feedback"
        try:
            actual = find_arm_by_hardware_id(read_can_hardware_id(leader_can))
            expected = find_arm_registration(config.leader_side, config.leader_role)
            if actual != expected or actual.name != "right_arm":
                raise ValueError(f"{leader_can} must resolve to the registered right leader")
            self.leader_name = self.follower_name = actual.name
            self.leader = NominalLimitedArm(mit.build_arm(leader_can))
            mit.reject_teaching_mode(self.leader, "right leader")
            self.leader_gravity = mit.GravityCompensator(mit.ASSET_ROOT / "calibration_leader.json")
            self.leader_gripper = LimitedGripper(self.leader.init_effector("agx_gripper"), actual.name,
                                                  config.gripper_max_m)
            self.fresh = [[mit.FeedbackWatchdog(config.feedback_timeout_s) for _ in range(8)]]
            self.status_fresh = mit.FeedbackWatchdog(config.feedback_timeout_s)
            self.last_status = 0.0
            time.sleep(0.5)
        except BaseException:
            self.close()
            raise

    def enable(self, should_stop=lambda: False):
        if should_stop():
            raise mit.TeleopAbort("startup cancelled")
        self._prepare_stops(should_stop)
        self.startup_phase = "right leader CAN/MIT enable"
        self.leader.set_follower_mode()
        time.sleep(0.3)
        self.initial_hold = tuple(clamp_pose(v, self.config.gripper_max_m) for v in self.read())
        hold = AssistedControl(self.config, *self.initial_hold)
        confirmation = mit.MITStartupConfirmation(self.leader)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if should_stop():
                raise mit.TeleopAbort("startup cancelled")
            self.read()
            if confirmation.auto_mode_disabled:
                mit.require_can_mit_enabled(self.leader, "right leader startup")
            flags = self.leader.get_joints_enable_status_list()
            if not flags or len(flags) != 6 or not all(flags):
                self.energized = True
                self.leader.enable()
            flags = self.leader.get_joints_enable_status_list()
            if flags and len(flags) == 6 and all(flags):
                self.energized = True
                self.send(hold, 1.0/self.config.mit_rate_hz)
                if confirmation.observe():
                    self.enabled = True
                    self.startup_phase = "supported right hold verification"
                    verify_until = time.monotonic()+self.config.feedback_timeout_s
                    while time.monotonic() < verify_until:
                        if should_stop():
                            raise mit.TeleopAbort("startup cancelled")
                        self.read()
                        self.send(hold, 1.0/self.config.mit_rate_hz)
                        time.sleep(1.0/self.config.mit_rate_hz)
                    self.startup_phase = "running"
                    return
            time.sleep(1.0/self.config.mit_rate_hz)
        raise mit.TeleopAbort("right leader startup lacked advancing healthy CAN/MIT feedback")

    def send(self, control, dt):
        del dt
        c = self.config
        start = time.monotonic_ns()
        self._send_leader_handoff(control)
        self._send_gripper(self.leader_gripper, self.leader_name, control.leader_target[6], c.grip_base)
        self.send_span_ns = time.monotonic_ns()-start
        if self.enabled:
            now = time.perf_counter()
            status = self.leader.get_arm_status()
            stamp = float(status.timestamp) if status is not None else float("nan")
            if not np.isfinite(stamp):
                raise mit.TeleopAbort("right leader status timestamp is non-finite")
            self.status_fresh.observe((stamp,), now)
            if now-self.last_status >= 0.1:
                mit.require_can_mit_enabled(self.leader, "right leader")
                if status.msg.err_code != 0:
                    raise mit.TeleopAbort(f"right leader error code {status.msg.err_code}")
                self.last_status = now

    def _home(self, should_stop, on_progress, hold_seconds, speed):
        self.startup_phase = "right leader MIT home"
        c = self.config
        mit.require_can_mit_enabled(self.leader, "right leader before home")
        leader, observed = self.read(check_speed=False)
        control = AssistedControl(c, leader, observed)
        began = time.monotonic()
        while time.monotonic()-began < hold_seconds:
            if should_stop():
                return False
            self.read(check_speed=time.monotonic()-began > 1.0)
            self.send(control, 1.0/c.mit_rate_hz)
            on_progress()
            time.sleep(1.0/c.mit_rate_hz)
        start = clamp_pose(self.read()[0], self.config.gripper_max_m)
        duration = mit.recovery_home_duration(start[:6], start[:6], speed)
        print(f"RIGHT LEADER HOME: {duration:.1f}s, reference <= {speed:.2f} rad/s", flush=True)
        began = time.monotonic()
        settled_since = None
        while True:
            if should_stop():
                return False
            elapsed = time.monotonic()-began
            self.read()
            q, velocity = mit.home_reference(start[:6], elapsed, duration)
            control.leader_target = np.r_[q, start[6]]
            control.reference_velocity = velocity
            self.send(control, 1.0/c.mit_rate_hz)
            on_progress()
            if elapsed >= duration:
                settled = np.max(np.abs(self.samples[0].q)) <= .05 and np.max(np.abs(self.samples[0].qd)) <= .05
                settled_since = (settled_since or time.monotonic()) if settled else None
                if settled_since is not None and time.monotonic()-settled_since >= .3:
                    break
                if elapsed > duration+5.:
                    raise mit.TeleopAbort("right leader did not settle at home")
            time.sleep(1.0/c.mit_rate_hz)
        self.initial_hold = (np.r_[np.zeros(6), start[6]], np.r_[np.zeros(6), start[6]])
        self.startup_phase = "running"
        return True

    def home(self, should_stop=lambda: False):
        if not self._home(should_stop, lambda: None, 0., self.config.start_home_speed):
            raise mit.TeleopAbort("startup home cancelled")

    def recover_home_after_abort(self, *, should_stop, on_progress):
        return self._home(should_stop, on_progress, self.config.abort_hold_seconds, self.config.abort_home_speed)

    def telemetry(self):
        return {"leader_q": self.samples[0].q, "leader_qd": self.samples[0].qd,
                "leader_gripper_width": self.grips[0].width}


def run(
    config,
    worker_session,
    leader_can=None,
    follower_can=None,
    *,
    execute=False,
    duration_s=0.0,
):
    if not config.simulate and not execute:
        raise ValueError("real MIT worker requires --execute")
    endpoint = parse_udp_endpoint(config.command_address)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind((endpoint.host, endpoint.port))
    receiver.setblocking(False)
    telemetry = MITTelemetryPublisher(config.telemetry_address)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    previous_handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    rig = control = None
    failed = False
    ready_reported = False
    try:
        rig = (
            SimulatedRig(config)
            if config.simulate
            else LeaderRig(config, leader_can) if config.leader_only
            else PiperRig(config, leader_can, follower_can)
        )
        rig.enable(should_stop=lambda: stopped)
        if config.startup_home:
            rig.home(should_stop=lambda: stopped)
        leader, follower = getattr(rig, "initial_hold", rig.read())
        control = AssistedControl(
            config,
            leader,
            follower,
            worker_session=worker_session,
            leader_name=rig.leader_name,
            follower_name=rig.follower_name,
        )
        control.leader_gain = 1.0
        dt = 1.0 / config.mit_rate_hz
        started = next_tick = time.monotonic()
        while not stopped and (
            not duration_s or time.monotonic() - started < duration_s
        ):
            leader, follower = rig.read()
            control.leader, control.follower = leader, follower
            for _ in range(128):
                try:
                    data, address = receiver.recvfrom(8192)
                except BlockingIOError:
                    break
                if address[0] != "127.0.0.1":
                    continue
                try:
                    packet = json.loads(data)
                    if not isinstance(packet, dict):
                        continue
                    # Simulation injection is never accepted by PiperRig. It
                    # still goes through session/sequence/age admission first.
                    if packet.get("kind") == "simulation_leader" and config.simulate:
                        target = packet.get("target")
                        admitted = control.receive({**packet, "kind": "heartbeat"})
                        if admitted and control.mode in ("HUMAN", "POLICY"):
                            rig.human(target, packet.get("duration_s", 0.0))
                            leader, follower = rig.read()
                            control.leader, control.follower = leader, follower
                    else:
                        control.receive(packet)
                except (ValueError, TypeError):
                    continue
            control.tick(leader, follower)
            rig.send(control, dt)
            if not ready_reported:
                subject = "right leader" if config.leader_only else "both arms"
                location = "homed" if config.startup_home else "holding nominal-clamped current poses"
                print(f"ASSISTED READY: {subject} {location}; ready for policy", flush=True)
                ready_reported = True
            telemetry.publish(
                phase="engaged",
                **rig.telemetry(),
                follower_target_q=control.target[:6],
                follower_target_qd=control.follower_velocity,
                follower_gripper_target=control.target[6],
                control={
                    **control.status(),
                    **rig.control_telemetry(),
                    "send_span_ns": getattr(rig, "send_span_ns", 0),
                    "leader_target": control.leader_target.tolist(),
                },
            )
            next_tick += dt
            time.sleep(max(0.0, next_tick - time.monotonic()))
            next_tick = max(next_tick, time.monotonic() - dt)
    except Exception as exc:
        failed = True
        fault_reason = str(exc)
        if isinstance(rig, PiperRig):
            print(f"ASSISTED failure phase: {rig.startup_phase}", flush=True)
            # Record controller state before recovery or emergency stop.
            for arm, label in ((rig.leader, "leader"), (rig.follower, "follower")):
                if arm is None:
                    continue
                try:
                    print(
                        f"{label} before recovery: {mit.arm_status_text(arm)}; "
                        f"enabled={arm.get_joints_enable_status_list()}",
                        flush=True,
                    )
                except Exception as diagnostic_error:
                    print(f"{label} status unavailable: {diagnostic_error}", flush=True)
        if control is not None:
            control.fault(fault_reason)
        print(f"ASSISTED FAULT: {fault_reason}", flush=True)
        recovered_home = False
        if isinstance(rig, PiperRig) and rig.enabled and ready_reported:
            print("ASSISTED RECOVERY: holding current poses, then MIT home", flush=True)

            def recovery_progress():
                telemetry.publish(
                    phase="recovering",
                    reason=fault_reason,
                    control=control.status(),
                    **rig.telemetry(),
                )

            try:
                recovery_progress()
                recovered_home = rig.recover_home_after_abort(
                    should_stop=lambda: stopped,
                    on_progress=recovery_progress,
                )
            except Exception as recovery_error:
                print(f"ABORT RECOVERY SKIPPED/FAILED: {recovery_error}", flush=True)
        if rig is not None and not recovered_home:
            rig.emergency_stop()
        telemetry.publish(
            phase="aborted",
            reason=fault_reason,
            control={
                **({} if control is None else control.status()),
                "worker_session": worker_session,
                "mode": "FAULT",
                "reason": fault_reason,
                "recovered_home": recovered_home,
            },
        )
    finally:
        # Normal shutdown holds. The exception path has already completed
        # recovery or stopped because healthy closed-loop motion was unavailable.
        if rig is not None and rig.enabled and control is not None and not failed:
            try:
                control.leader, control.follower = rig.read()
                control._revoke("HOLD", "shutdown")
                for _ in range(max(1, int(config.mit_rate_hz * 0.3))):
                    leader, follower = rig.read()
                    control.tick(leader, follower)
                    rig.send(control, 1.0 / config.mit_rate_hz)
                    time.sleep(1.0 / config.mit_rate_hz)
            except Exception as exc:
                failed = True
                rig.emergency_stop()
                print(f"Final hold unavailable: {exc}", flush=True)
        if control is not None and not failed:
            telemetry.publish(
                phase="stopped", reason="worker stopped", control=control.status()
            )
        if rig is not None:
            rig.close()
        receiver.close()
        telemetry.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--worker-session", required=True)
    parser.add_argument("--leader-can")
    parser.add_argument("--follower-can")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration-s", type=float, default=0.0)
    args = parser.parse_args()
    config = SinglePiperAssistedConfig.from_args(json.loads(args.config_json))
    return run(
        config,
        args.worker_session,
        args.leader_can,
        args.follower_can,
        execute=args.execute,
        duration_s=args.duration_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
