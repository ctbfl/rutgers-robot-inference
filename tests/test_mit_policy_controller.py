from __future__ import annotations

import json
import socket
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from ruri.client.controllers.single_piper.mit import leader_follower as mit
from ruri.client.controllers.single_piper.mit import policy_controller as pc
from ruri.client.controllers.single_piper.mit_io import (
    MITCommandSender as PolicyCommandSender,
)

from ruri.client.controllers.single_piper.mit.policy_controller import (
    PolicyCommandReceiver,
    startup_home_is_within_threshold,
    write_diagnostic_log,
)


def free_udp_address() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"udp://127.0.0.1:{port}"


class CommandProtocolTests(unittest.TestCase):
    def test_startup_home_skip_uses_max_absolute_joint_error(self):
        self.assertTrue(
            startup_home_is_within_threshold(
                [0.049, -0.05, 0.0, 0.01, -0.02, 0.03], 0.05
            )
        )
        self.assertFalse(
            startup_home_is_within_threshold(
                [0.0, 0.0, 0.051, 0.0, 0.0, 0.0], 0.05
            )
        )

    def test_sender_receiver_round_trip_and_sequence(self):
        address = free_udp_address()
        receiver = PolicyCommandReceiver(address, max_age_s=0.3)
        sender = PolicyCommandSender(address)
        try:
            sender.send([0.1, 0.2, -0.3, 0.4, -0.5, 0.6], 0.04)
            time.sleep(0.01)
            target = receiver.receive_latest()
            self.assertIsNotNone(target)
            self.assertEqual(target.sequence, 1)
            np.testing.assert_allclose(target.q, [0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
            self.assertAlmostEqual(target.gripper_width, 0.04)
        finally:
            sender.close()
            receiver.close()


    def test_diagnostic_log_contains_header_and_samples(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.jsonl"
            write_diagnostic_log(
                path,
                [{"kind": "controller_sample", "tracking_error": 0.1}],
                {"outcome": "aborted: test", "rate_hz": 100.0},
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(records[0]["kind"], "trace_header")
        self.assertEqual(records[0]["outcome"], "aborted: test")
        self.assertEqual(records[1]["tracking_error"], 0.1)

    def test_receiver_coalesces_pending_commands_to_latest(self):
        address = free_udp_address()
        receiver = PolicyCommandReceiver(address, max_age_s=0.3)
        sender = PolicyCommandSender(address)
        try:
            base = np.array([0.01, 0.02, -0.03, 0.04, -0.05, 0.06])
            sender.send(base, 0.01)
            sender.send(2.0 * base, 0.02)
            sender.send(3.0 * base, 0.03)
            time.sleep(0.01)
            target = receiver.receive_latest()
            self.assertIsNotNone(target)
            self.assertEqual(target.sequence, 3)
            np.testing.assert_allclose(target.q, 3.0 * base)
            self.assertAlmostEqual(target.gripper_width, 0.03)
        finally:
            sender.close()
            receiver.close()

class SimulatedArm:
    """Stationary feedback source; no CAN implementation or sockets."""

    def __init__(self):
        # Includes the small J2/J3 offsets seen in the real, disabled arms.
        self.q = np.array([-0.04, -0.037, 0.03, 0.08, 0.36, -0.19])
        self.stamp = 1.0
        self.enabled = False
        self.freeze = False
        self.frames = []
        self.status = SimpleNamespace(
            ctrl_mode="STANDBY", mode_feedback="MOVE_P", arm_status="NORMAL", err_code=0,
        )
        self.set_auto_set_motion_mode_enabled = MagicMock()
        self.set_follower_mode = MagicMock()
        self.disconnect = MagicMock()
        self.gripper = MagicMock()
        self.gripper.get_gripper_status.side_effect = lambda: SimpleNamespace(
            timestamp=self.stamp, msg=SimpleNamespace(value=0.04, force=0.0),
        )
        self.firmware_info = {"software_version": "S-V1.9-1"}
        self.firmware_driver = "v189"

    def get_arm_status(self):
        return SimpleNamespace(timestamp=self.stamp, msg=self.status)

    def get_joints_enable_status_list(self):
        return [self.enabled] * 6

    def enable(self):
        self.enabled = True

    def get_joint_angles(self):
        if not self.freeze:
            self.stamp += 0.01
        return SimpleNamespace(timestamp=self.stamp, msg=self.q.tolist())

    def get_motor_states(self, joint):
        return SimpleNamespace(
            timestamp=self.stamp, msg=SimpleNamespace(velocity=0.0, current=0.0, torque=0.0),
        )

    def move_mit(self, **command):
        self.frames.append(command)
        self.status.ctrl_mode = "CAN_CTRL"
        self.status.mode_feedback = "MOVE_MIT"

    def init_effector(self, kind):
        assert kind == "agx_gripper"
        return self.gripper


class StartupAndHoldTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.now = 1.0
        self.stack.enter_context(patch.object(pc.time, "perf_counter", side_effect=lambda: self.now))
        self.stack.enter_context(patch.object(pc.time, "monotonic", side_effect=lambda: self.now))
        self.stack.enter_context(patch.object(pc.time, "sleep", side_effect=self.advance))
        self.stack.enter_context(patch.object(pc.signal, "signal"))
        self.stack.enter_context(patch("builtins.print"))
        self.stack.enter_context(patch.object(pc, "_running", True))
        self.arm = SimulatedArm()
        self.gravity = SimpleNamespace(torque=lambda q: np.zeros(6))

    def advance(self, duration):
        self.now += duration

    def start(self):
        return pc.enable_follower_with_hold(self.arm, self.gravity, 10, 0.8, 3, timeout=0.2)

    def test_startup_requires_fresh_feedback_after_auto_mode_frames_stop(self):
        self.start()
        self.arm.set_auto_set_motion_mode_enabled.assert_called_once_with(False)
        self.assertGreaterEqual(len(self.arm.frames), 4 * 6)
        for command in self.arm.frames:
            self.assertEqual(command["p_des"], self.arm.q[command["joint_index"] - 1])
        self.assertGreaterEqual(self.now, 1.06)

    def test_stale_mit_status_cannot_complete_startup(self):
        self.arm.freeze = True
        self.arm.enabled = True
        self.arm.status.ctrl_mode = "CAN_CTRL"
        self.arm.status.mode_feedback = "MOVE_MIT"
        with self.assertRaisesRegex(mit.TeleopAbort, "status timestamps"):
            self.start()
        self.arm.set_auto_set_motion_mode_enabled.assert_not_called()

    def test_status_must_keep_advancing_after_auto_mode_is_disabled(self):
        self.arm.set_auto_set_motion_mode_enabled.side_effect = lambda _: setattr(self.arm, "freeze", True)
        with self.assertRaisesRegex(mit.TeleopAbort, "status timestamps"):
            self.start()

    def test_mode_loss_after_auto_mode_is_disabled_aborts(self):
        self.arm.set_auto_set_motion_mode_enabled.side_effect = lambda _: setattr(
            self.arm.status, "ctrl_mode", "STANDBY"
        )
        with self.assertRaisesRegex(mit.TeleopAbort, "left CAN/MIT"):
            self.start()

    def test_nonzero_error_code_cannot_complete_startup(self):
        self.arm.status.err_code = 1
        with self.assertRaises(mit.TeleopAbort):
            self.start()
        self.arm.set_auto_set_motion_mode_enabled.assert_not_called()

    def test_bad_gravity_is_rejected_before_enable(self):
        self.gravity.torque = lambda q: np.full(6, 100)
        with self.assertRaisesRegex(mit.TeleopAbort, "ceiling"):
            self.start()
        self.assertFalse(self.arm.enabled)
        self.assertEqual(self.arm.frames, [])

    def test_dual_startup_requires_fresh_status_from_both_arms(self):
        follower = SimulatedArm()
        follower.freeze = True
        with self.assertRaisesRegex(mit.TeleopAbort, "status timestamps"):
            mit.enable_both_with_safe_commands(
                self.arm, follower, self.gravity, self.gravity,
                0.2, 10, 0.8, 3, timeout=0.2, leader_hold=self.arm.q,
            )
        for arm in (self.arm, follower):
            arm.set_auto_set_motion_mode_enabled.assert_not_called()

    def run_worker(self, flags):
        self.stack.enter_context(patch.object(pc, "build_arm", return_value=self.arm))
        self.stack.enter_context(patch.object(pc, "GravityCompensator", return_value=self.gravity))
        self.telemetry = self.stack.enter_context(patch.object(pc, "MITTelemetryPublisher"))
        self.receiver = self.stack.enter_context(patch.object(pc, "PolicyCommandReceiver"))
        self.home = self.stack.enter_context(patch.object(pc, "hold_then_home_follower", return_value=True))
        return pc.run(pc.build_parser().parse_args(flags))

    def test_hold_only_runs_without_policy_or_home_or_gripper_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hold.jsonl"
            result = self.run_worker([
                "--execute", "--hold-only", "--seconds", "0.05",
                "--diagnostic-log", str(path),
            ])
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(result, 0)
        self.home.assert_not_called()
        self.receiver.assert_not_called()
        self.arm.gripper.move_gripper_m.assert_not_called()
        self.arm.disconnect.assert_called_once()
        self.assertTrue(self.arm.enabled)  # Exit does not disable torque.
        self.assertTrue(self.arm.frames)
        for command in self.arm.frames:
            self.assertEqual(command["p_des"], self.arm.q[command["joint_index"] - 1])
            self.assertEqual(command["v_des"], 0)
        self.assertEqual(records[0]["firmware_driver"], "v189")
        self.assertEqual(records[0]["firmware"]["software_version"], "S-V1.9-1")
        self.assertTrue(records[0]["hold_only"])
        self.assertGreater(len(records), 1)
        for sample in records[1:]:
            self.assertEqual(sample["q_target"], self.arm.q.tolist())
            self.assertEqual(len(sample["t_ff"]), 6)
            self.assertEqual(len(sample["current_a"]), 6)
            self.assertIsNone(sample["target_sequence"])

    def test_hold_only_without_execute_does_not_enable(self):
        self.assertEqual(self.run_worker(["--hold-only"]), 0)
        self.assertFalse(self.arm.enabled)
        self.assertEqual(self.arm.frames, [])
        self.home.assert_not_called()
        self.receiver.assert_not_called()
        self.arm.set_follower_mode.assert_not_called()

    def test_hold_duration_is_checked_without_waiting_for_policy_commands(self):
        began = self.now
        self.assertEqual(self.run_worker(["--execute", "--hold-only", "--seconds", "0.2"]), 0)
        self.assertLess(self.now - began, 2.0)

    def test_normal_worker_still_homes_before_policy_loop(self):
        def finish_home(*args, **kwargs):
            pc._running = False
            return True
        self.stack.enter_context(patch.object(pc, "build_arm", return_value=self.arm))
        self.stack.enter_context(patch.object(pc, "GravityCompensator", return_value=self.gravity))
        self.stack.enter_context(patch.object(pc, "MITTelemetryPublisher"))
        receiver = self.stack.enter_context(patch.object(pc, "PolicyCommandReceiver"))
        home = self.stack.enter_context(patch.object(pc, "hold_then_home_follower", side_effect=finish_home))
        self.assertEqual(pc.run(pc.build_parser().parse_args(["--execute"])), 0)
        receiver.assert_called_once()
        home.assert_called_once()



if __name__ == "__main__":
    unittest.main()
