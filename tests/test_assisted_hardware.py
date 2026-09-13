"""Exercise the actual PiperRig SDK boundary without opening physical devices."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from ruri.client.controllers.single_piper.mit import assisted
from ruri.client.controllers.single_piper_assisted.config import (
    SinglePiperAssistedConfig,
)
from ruri.client.controllers.single_piper_assisted.control import (
    PROTOCOL,
    AssistedControl,
)


class AssistedHardwareTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.now = 1.0
        self.stamp = 1.0
        self.q = np.array([0.0, 0.4, -0.4, 0.0, 0.0, 0.0])
        self.arms = [self.arm(), self.arm()]
        self.stack.enter_context(
            patch.object(
                assisted, "read_can_hardware_id", side_effect=lambda channel: channel
            )
        )
        self.stack.enter_context(
            patch.object(
                assisted,
                "find_arm_by_hardware_id",
                side_effect=lambda channel: SimpleNamespace(
                    name="right_arm" if channel == "canL" else "left_arm"
                ),
            )
        )
        self.stack.enter_context(
            patch.object(
                assisted,
                "find_arm_registration",
                side_effect=lambda side, role: SimpleNamespace(name=f"{side}_arm"),
            )
        )
        self.stack.enter_context(
            patch.object(assisted.mit, "build_arm", side_effect=self.arms)
        )
        self.stack.enter_context(
            patch.object(
                assisted.mit,
                "GravityCompensator",
                side_effect=lambda path: SimpleNamespace(torque=lambda q: np.zeros(6)),
            )
        )
        self.stack.enter_context(
            patch.object(assisted.time, "sleep", side_effect=self.advance)
        )
        self.stack.enter_context(
            patch.object(assisted.time, "monotonic", side_effect=lambda: self.now)
        )
        self.stack.enter_context(
            patch.object(assisted.time, "perf_counter", side_effect=lambda: self.now)
        )
        self.rig = assisted.PiperRig(SinglePiperAssistedConfig(), "canL", "canF")
        self.addCleanup(self.rig.close)

    def advance(self, seconds):
        self.now += seconds

    def arm(self):
        arm = MagicMock()
        enabled = [False]
        arm.get_joints_enable_status_list.side_effect = lambda: [enabled[0]] * 6
        arm.enable.side_effect = lambda: enabled.__setitem__(0, True)
        arm.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(
                ctrl_mode="CAN_CTRL", mode_feedback="MOVE_MIT", arm_status="NORMAL"
            )
        )

        def angles():
            self.stamp += 0.01
            return SimpleNamespace(msg=self.q.tolist(), timestamp=self.stamp)

        arm.get_joint_angles.side_effect = angles
        arm.get_motor_states.side_effect = lambda joint: SimpleNamespace(
            msg=SimpleNamespace(velocity=0.0, torque=0.0), timestamp=self.stamp
        )
        arm.init_effector.return_value.get_gripper_status.side_effect = lambda: (
            SimpleNamespace(
                msg=SimpleNamespace(value=0.04, force=0.2), timestamp=self.stamp
            )
        )
        return arm

    def test_enable_holds_both_arms_then_policy_and_human_commands(self):
        self.rig.enable()
        self.assertTrue(self.rig.enabled)
        for arm, kp in zip(self.arms, (3.0, 10.0)):
            arm.set_follower_mode.assert_called_once()
            self.assertEqual(len(arm.move_mit.call_args_list), 6)
            self.assertTrue(
                all(call.kwargs["kp"] == kp for call in arm.move_mit.call_args_list)
            )
            arm.set_auto_set_motion_mode_enabled.assert_called_once_with(False)
        leader, follower = self.rig.read()
        control = AssistedControl(
            self.rig.config, leader, follower, clock=lambda: self.now
        )
        control.leader_gain = 1.0
        self.rig.send(control, 0.01)
        np.testing.assert_allclose(
            [
                call.kwargs["p_des"]
                for call in self.arms[1].move_mit.call_args_list[-6:]
            ],
            self.q,
        )
        request = {
            "protocol": PROTOCOL,
            "worker_session": control.session,
            "client_session": "test",
            "sequence": 1,
            "monotonic_ns": int(self.now * 1e9),
            "kind": "mode",
            "mode": "HUMAN",
            "request_id": "takeover",
            "epoch": control.epoch,
        }
        self.assertTrue(control.receive(request))
        self.rig.send(control, 0.01)
        self.assertTrue(
            all(
                call.kwargs["kp"] == 0
                for call in self.arms[0].move_mit.call_args_list[-6:]
            )
        )
        self.assertTrue(
            all(
                call.kwargs["kp"] == 10
                for call in self.arms[1].move_mit.call_args_list[-6:]
            )
        )
        self.rig.emergency_stop()
        for arm in self.arms:
            arm.electronic_emergency_stop.assert_called_once()
            arm.disable.assert_not_called()

    def test_one_stale_motor_is_not_masked_by_other_live_feedback(self):
        original = self.arms[0].get_motor_states.side_effect
        self.arms[0].get_motor_states.side_effect = lambda joint: (
            SimpleNamespace(
                msg=SimpleNamespace(velocity=0.0, torque=0.0), timestamp=1.0
            )
            if joint == 2
            else original(joint)
        )
        self.rig.read()
        self.advance(0.3)
        with self.assertRaisesRegex(
            assisted.mit.TeleopAbort, "timestamps did not advance"
        ):
            self.rig.read()

    def test_startup_cancel_and_tracking_error_do_not_issue_home(self):
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, "cancelled"):
            self.rig.enable(should_stop=lambda: True)
        for arm in self.arms:
            arm.enable.assert_not_called()
            arm.move_mit.assert_not_called()
        self.rig.enable()
        leader, follower = self.rig.read()
        control = AssistedControl(
            self.rig.config, leader, follower, clock=lambda: self.now
        )
        control.target[0] += 0.4
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, "tracking error"):
            self.rig.send(control, 0.01)

    def test_nonfinite_feedback_and_invalid_config_are_rejected(self):
        self.arms[0].init_effector.return_value.get_gripper_status.side_effect = (
            lambda: SimpleNamespace(
                msg=SimpleNamespace(value=0.04, force=float("nan")),
                timestamp=self.stamp,
            )
        )
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, "non-finite"):
            self.rig.read()
        for value in (
            {"simulate": "false"},
            {"trace_queue_size": 100.5},
            {"command_timeout_s": float("nan")},
        ):
            with self.assertRaises(ValueError):
                SinglePiperAssistedConfig.from_args(value)


if __name__ == "__main__":
    unittest.main()
