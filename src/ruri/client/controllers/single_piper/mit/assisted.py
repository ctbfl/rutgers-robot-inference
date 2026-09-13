"""Persistent assisted MIT worker; --simulate exercises the same protocol/core.

Startup captures current poses. No automatic home on startup, timeout or exit.
Only this process opens the two CAN devices. The optional simulation path opens
neither CAN nor cameras and is suitable for client/server integration tests.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import time

import numpy as np

from ..calibration_ranges import measured_for_arm
from ..hardware_registry import find_arm_by_hardware_id, find_arm_registration
from ..discovery import read_can_hardware_id
from ...single_piper_assisted.config import SinglePiperAssistedConfig
from ...single_piper_assisted.control import AssistedControl, physical_bounds, pose
from . import leader_follower as mit
from .telemetry import MITTelemetryPublisher, parse_udp_endpoint


class SimulatedRig:
    def __init__(self, config):
        self.config = config
        self.leader_name, self.follower_name = "right_arm", "left_arm"
        self.leader = np.asarray([0., 0.4, -0.4, 0., 0., 0., 0.04])
        self.follower = self.leader.copy()
        self.previous = self.leader.copy(), self.follower.copy()
        self.samples = None
        self.enabled = False

    def read(self):
        return self.leader.copy(), self.follower.copy()

    def enable(self, should_stop=lambda: False):
        self.enabled = not should_stop()

    def emergency_stop(self):
        self.enabled = False

    def send(self, control, dt):
        self.previous = self.leader.copy(), self.follower.copy()
        if control.mode != "HUMAN":
            self.leader += np.clip(control.leader_target - self.leader, -dt, dt)
        self.follower += np.clip(control.target - self.follower, -dt, dt)

    def human(self, target):
        self.leader = pose(target)

    def telemetry(self):
        return {"leader_q": self.leader[:6], "leader_qd": np.zeros(6),
                "follower_q": self.follower[:6], "follower_qd": np.zeros(6),
                "follower_joint_effort": np.zeros(6), "leader_gripper_width": self.leader[6],
                "follower_gripper_width": self.follower[6], "follower_gripper_force": 0.}

    def close(self):
        pass


class PiperRig:
    def __init__(self, config, leader_can, follower_can):
        self.config = config
        self.leader = self.follower = None
        self.leader_gripper = self.follower_gripper = None
        self.samples = self.grips = None
        self.energized = self.enabled = False
        try:
            registrations = []
            for channel, side, role in ((leader_can, config.leader_side, config.leader_role),
                                        (follower_can, config.follower_side, config.follower_role)):
                actual = find_arm_by_hardware_id(read_can_hardware_id(channel))
                expected = find_arm_registration(side, role)
                if actual != expected:
                    raise ValueError(f"{channel} resolves to {actual.name}; expected {expected.name}")
                registrations.append(actual)
            if leader_can == follower_can:
                raise ValueError("leader and follower cannot share a CAN interface")
            self.leader_name, self.follower_name = (r.name for r in registrations)
            self.leader = mit.build_arm(leader_can)
            self.follower = mit.build_arm(follower_can)
            mit.reject_teaching_mode(self.leader, "leader")
            mit.reject_teaching_mode(self.follower, "follower")
            self.leader_gravity = mit.GravityCompensator(mit.ASSET_ROOT / "calibration_leader.json")
            self.follower_gravity = mit.GravityCompensator(mit.ASSET_ROOT / "calibration_follower.json")
            self.leader_gripper = self.leader.init_effector("agx_gripper")
            self.follower_gripper = self.follower.init_effector("agx_gripper")
            time.sleep(0.5)
            self.fresh = [[mit.FeedbackWatchdog(config.feedback_timeout_s) for _ in range(8)] for _ in range(2)]
            self.last_status = 0.
            self.enabled = False
        except BaseException:
            self.close()
            raise

    def read(self):
        self.samples = (mit.read_sample(self.leader, "leader"), mit.read_sample(self.follower, "follower"))
        self.grips = (mit.read_gripper(self.leader_gripper), mit.read_gripper(self.follower_gripper))
        if any(g is None for g in self.grips):
            raise mit.TeleopAbort("both gripper feedback streams are required")
        now = time.perf_counter()
        for sample, grip, watches in zip(self.samples, self.grips, self.fresh):
            # A live gripper must not hide a stalled joint feedback stream.
            for stamp, watch in zip((*sample.stamps, grip.timestamp), watches, strict=True):
                watch.observe((stamp,), now)
            if np.max(np.abs(sample.qd)) > self.config.max_joint_speed:
                raise mit.TeleopAbort("measured joint speed exceeded limit")
        if self.enabled and now - self.last_status >= 0.1:
            mit.require_can_mit_enabled(self.leader, "leader")
            mit.require_can_mit_enabled(self.follower, "follower")
            self.last_status = now
        return tuple(np.r_[s.q, g.width] for s, g in zip(self.samples, self.grips))

    def enable(self, should_stop=lambda: False):
        # Both arms receive holds during enable; never leave a supported leader
        # waiting in gravity-only mode while the follower initializes.
        initial = self.read()
        self.leader.set_follower_mode()
        self.follower.set_follower_mode()
        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline:
            if should_stop():
                raise mit.TeleopAbort("startup cancelled")
            self.read()
            ready = []
            for arm, sample, gravity, target, label in (
                (self.leader, self.samples[0], self.leader_gravity, initial[0], "leader"),
                (self.follower, self.samples[1], self.follower_gravity, initial[1], "follower"),
            ):
                enabled = arm.get_joints_enable_status_list()
                if not (enabled and all(enabled)):
                    self.energized = True
                    arm.enable()
                else:
                    self.energized = True
                    mit.send_position_impedance(arm, sample, target[:6], np.zeros(6), gravity,
                                                self.config.leader_policy_kp if label == "leader" else self.config.follower_kp,
                                                self.config.leader_policy_kd if label == "leader" else self.config.follower_kd,
                                                self.config.max_reference_speed, label)
                ctrl, mode, _ = mit.arm_status_text(arm)
                ready.append(bool(enabled and all(enabled) and "CAN" in ctrl and "MIT" in mode))
            if all(ready):
                for arm in (self.leader, self.follower):
                    arm.set_auto_set_motion_mode_enabled(False)
                self.enabled = True
                return
            time.sleep(0.01)
        raise mit.TeleopAbort("could not enable both arms in CAN/MIT")

    def send(self, control, dt):
        del dt
        c = self.config
        for sample, target, label in ((self.samples[1], control.target, "follower"),):
            if np.max(np.abs(sample.q - target[:6])) > c.max_track_error:
                raise mit.TeleopAbort(f"{label} tracking error exceeded limit")
        start_ns = time.monotonic_ns()
        if control.mode == "HUMAN":
            mit.send_leader(self.leader, self.samples[0], self.leader_gravity, c.leader_kd)
        else:
            if np.max(np.abs(self.samples[0].q - control.leader_target[:6])) > c.max_track_error:
                raise mit.TeleopAbort("leader tracking error exceeded limit")
            mit.send_position_impedance(self.leader, self.samples[0], control.leader_target[:6],
                                        np.zeros(6), self.leader_gravity,
                                        c.leader_policy_kp * control.leader_gain, c.leader_policy_kd,
                                        c.max_reference_speed, "leader")
        mit.send_position_impedance(self.follower, self.samples[1], control.target[:6], np.zeros(6),
                                    self.follower_gravity, c.follower_kp, c.follower_kd,
                                    c.max_reference_speed, "follower")
        self.follower_gripper.move_gripper_m(float(control.target[6]), c.grip_force)
        if control.mode == "HUMAN":
            force = min(c.grip_max_force, c.grip_base + c.grip_gain * abs(self.grips[1].force))
            # Reflect force at the offset-corrected leader coordinate, retaining
            # the relative gripper anchor instead of reverting to absolute teleop.
            width = control.anchor_leader[6] + self.grips[1].width - control.anchor_target[6]
            low, high = physical_bounds(measured_for_arm(self.leader_name))
            self.leader_gripper.move_gripper_m(float(np.clip(width, low[6], high[6])), force)
        else:
            self.leader_gripper.move_gripper_m(float(control.leader_target[6]), c.grip_base)
        self.send_span_ns = time.monotonic_ns() - start_ns

    def telemetry(self):
        a, b = self.samples
        ga, gb = self.grips
        return {"leader_q": a.q, "leader_qd": a.qd, "follower_q": b.q,
                "follower_qd": b.qd, "follower_joint_effort": b.joint_effort,
                "leader_gripper_width": ga.width, "follower_gripper_width": gb.width,
                "follower_gripper_force": gb.force}

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


def run(config, worker_session, leader_can=None, follower_can=None, *, execute=False, duration_s=0.):
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
    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    rig = control = None
    failed = False
    try:
        rig = SimulatedRig(config) if config.simulate else PiperRig(config, leader_can, follower_can)
        leader, follower = rig.read()
        control = AssistedControl(config, leader, follower, worker_session=worker_session,
                                  leader_name=rig.leader_name, follower_name=rig.follower_name)
        rig.enable(should_stop=lambda: stopped)
        control.leader_gain = 1.
        dt = 1. / config.mit_rate_hz
        started = next_tick = time.monotonic()
        print("ASSISTED READY: HOLD at current poses; no automatic homing", flush=True)
        while not stopped and (not duration_s or time.monotonic() - started < duration_s):
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
                        if admitted and control.mode == "HUMAN":
                            rig.human(target)
                    else:
                        control.receive(packet)
                except (ValueError, TypeError):
                    continue
            control.tick(leader, follower)
            rig.send(control, dt)
            telemetry.publish(phase="engaged", **rig.telemetry(),
                              follower_target_q=control.target[:6], follower_target_qd=np.zeros(6),
                              follower_gripper_target=control.target[6],
                              control={**control.status(), "send_span_ns": getattr(rig, "send_span_ns", 0),
                                       "leader_target": control.leader_target.tolist()})
            next_tick += dt
            time.sleep(max(0., next_tick - time.monotonic()))
            next_tick = max(next_tick, time.monotonic() - dt)
    except Exception as exc:
        failed = True
        if rig is not None:
            rig.emergency_stop()
        if control is not None:
            control.fault(str(exc))
        print(f"ASSISTED FAULT: {exc}", flush=True)
        telemetry.publish(phase="aborted", reason=str(exc), control=None if control is None else control.status())
    finally:
        # Normal shutdown: a short supported hold. Faults do not attempt home
        # or further closed-loop motion with feedback that may be invalid.
        if rig is not None and rig.enabled and control is not None and not failed:
            try:
                control.leader, control.follower = rig.read()
                control._revoke("HOLD", "shutdown")
                for _ in range(max(1, int(config.mit_rate_hz * 0.3))):
                    leader, follower = rig.read()
                    control.tick(leader, follower)
                    rig.send(control, 1. / config.mit_rate_hz)
                    time.sleep(1. / config.mit_rate_hz)
            except Exception as exc:
                failed = True
                rig.emergency_stop()
                print(f"Final hold unavailable: {exc}", flush=True)
        if control is not None and not failed:
            telemetry.publish(phase="stopped", reason="worker stopped", control=control.status())
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
    parser.add_argument("--duration-s", type=float, default=0.)
    args = parser.parse_args()
    config = SinglePiperAssistedConfig.from_args(json.loads(args.config_json))
    return run(config, args.worker_session, args.leader_can, args.follower_can,
               execute=args.execute, duration_s=args.duration_s)


if __name__ == "__main__":
    raise SystemExit(main())
