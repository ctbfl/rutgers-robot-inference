"""Exercise the actual PiperRig SDK boundary without opening physical devices."""

import unittest
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

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
            timestamp=1.0,
            msg=SimpleNamespace(
                ctrl_mode="CAN_CTRL",
                mode_feedback="MOVE_MIT",
                arm_status="NORMAL",
                teach_status="STOP",
                err_code=0,
            ),
        )

        def angles():
            self.stamp += 0.01
            arm.get_arm_status.return_value.timestamp = self.stamp
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
            self.assertGreaterEqual(len(arm.move_mit.call_args_list), 6)
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

    def test_enable_does_not_use_leader_status_from_before_follower_writes(self):
        status = self.arms[0].get_arm_status.return_value.msg
        self.arms[0].move_mit.side_effect = lambda **_: setattr(
            status, "ctrl_mode", "CAN_CTRL"
        )
        self.arms[1].move_mit.side_effect = lambda **_: setattr(
            status, "ctrl_mode", "STANDBY"
        )
        with self.assertRaisesRegex(
            assisted.mit.TeleopAbort, "failed to enable.*STANDBY"
        ):
            self.rig.enable()
        self.assertFalse(self.rig.enabled)
        for arm in self.arms:
            arm.set_auto_set_motion_mode_enabled.assert_not_called()

    def test_persistent_standby_after_mit_still_aborts(self):
        self.rig.enable()
        leader, follower = self.rig.read()
        control = AssistedControl(
            self.rig.config, leader, follower, clock=lambda: self.now
        )
        self.arms[0].get_arm_status.return_value.msg.ctrl_mode = "STANDBY"
        self.advance(0.11)
        with self.assertRaisesRegex(
            assisted.mit.TeleopAbort,
            "leader left CAN/MIT.*status=NORMAL, teach=STOP, err_code=0",
        ):
            self.rig.send(control, 0.01)

    def test_startup_cancel_does_not_enable_or_move(self):
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, "cancelled"):
            self.rig.enable(should_stop=lambda: True)
        for arm in self.arms:
            arm.enable.assert_not_called()
            arm.move_mit.assert_not_called()

    def test_distant_policy_target_is_sent_to_both_arms_without_gap_rejection(self):
        self.rig.enable()
        leader, follower = self.rig.read()
        control = AssistedControl(
            self.rig.config, leader, follower, clock=lambda: self.now
        )
        target = np.r_[self.q.copy(), 0.04]
        target[0] += 0.8  # Valid nominal target, beyond the former 0.35-rad gate.
        control._mode("POLICY", control.epoch)
        self.assertTrue(
            control.receive(
                {
                    "protocol": PROTOCOL,
                    "worker_session": control.session,
                    "client_session": "test",
                    "sequence": 1,
                    "kind": "target",
                    "monotonic_ns": int(self.now * 1e9),
                    "epoch": control.epoch,
                    "target": target.tolist(),
                }
            )
        )
        self.rig.send(control, 0.01)
        for arm in self.arms:
            sent = [call.kwargs["p_des"] for call in arm.move_mit.call_args_list[-6:]]
            np.testing.assert_allclose(sent, target[:6])
            arm.electronic_emergency_stop.assert_not_called()

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

    def test_role_settling_pose_is_the_initial_hold_and_no_mode_reset_is_sent(self):
        self.arms[1].set_follower_mode.side_effect = lambda: self.q.__setitem__(0, 0.05)
        self.rig.enable()
        self.assertAlmostEqual(self.rig.initial_hold[0][0], 0.05)
        for arm in self.arms:
            self.assertAlmostEqual(arm.move_mit.call_args_list[0].kwargs["p_des"], 0.05)
            arm.set_motion_mode.assert_not_called()
            arm.reset.assert_not_called()

    def test_startup_and_gravity_mode_apply_nominal_clamp_without_clipping_measurement(
        self,
    ):
        self.q[1:3] = [-0.0346, 0.033]
        self.rig.enable()
        np.testing.assert_allclose(self.rig.read()[0][1:3], self.q[1:3])
        np.testing.assert_allclose(self.rig.initial_hold[0][1:3], [0.0, 0.0])
        control = AssistedControl(
            self.rig.config, *self.rig.read(), clock=lambda: self.now
        )
        control.mode = "HUMAN"
        self.rig.send(control, 0.01)
        for arm in self.arms:
            for call in arm.move_mit.call_args_list:
                i = call.kwargs["joint_index"] - 1
                self.assertGreaterEqual(
                    call.kwargs["p_des"], assisted.mit.JOINT_LIMIT_LOWER[i]
                )
                self.assertLessEqual(
                    call.kwargs["p_des"], assisted.mit.JOINT_LIMIT_UPPER[i]
                )

    def test_nominal_then_per_arm_hardware_clamp(self):
        raw = MagicMock()
        lower = assisted.mit.JOINT_LIMIT_LOWER.copy()
        upper = assisted.mit.JOINT_LIMIT_UPPER.copy()
        upper[5] -= 0.02
        hardware = assisted.mit.HardwareLimitedArm(raw, lower, upper, "test")
        arm = assisted.NominalLimitedArm(hardware)
        arm.move_mit(joint_index=6, p_des=9.0)
        self.assertAlmostEqual(raw.move_mit.call_args.kwargs["p_des"], upper[5])

    def test_public_action_boundary_clamps_before_normalized_validation(self):
        from ruri.client.controllers.single_piper_assisted.controller import (
            SinglePiperAssistedController,
        )

        robot = SinglePiperAssistedController({"simulate": True})
        robot._io = MagicMock()
        robot._policy_epoch = 1
        robot.send_action([1000, -1000, 1000, 0, 0, 1000, 1000], reference_revision=0)
        target = np.asarray(robot._io.send.call_args.kwargs["target"])
        np.testing.assert_allclose(target, assisted.clamp_pose(target))
        self.assertAlmostEqual(target[1], 0.0)
        self.assertAlmostEqual(target[5], assisted.mit.JOINT_LIMIT_UPPER[5])
        self.assertAlmostEqual(target[6], 0.08)

    def test_startup_home_uses_teleop_trajectory_and_keeps_zero_hold_references(self):
        positions = [self.q.copy(), self.q.copy()]
        positions[0][0], positions[1][0] = 0.2, -0.1
        for arm, position in zip(self.arms, positions):

            def angles(arm=arm, position=position):
                self.stamp += 0.01
                arm.get_arm_status.return_value.timestamp = self.stamp
                return SimpleNamespace(msg=position.tolist(), timestamp=self.stamp)

            arm.get_joint_angles.side_effect = angles
            arm.move_mit.side_effect = lambda position=position, **kw: (
                position.__setitem__(kw["joint_index"] - 1, kw["p_des"])
            )
        self.rig.enable()
        with patch.object(
            assisted.mit, "home_both_with_mit", wraps=assisted.mit.home_both_with_mit
        ) as home:
            self.rig.home()
        home.assert_called_once()
        self.assertEqual(
            home.call_args.kwargs["home_speed"], self.rig.config.start_home_speed
        )
        for position, held in zip(positions, self.rig.initial_hold):
            np.testing.assert_allclose(position, np.zeros(6), atol=1e-6)
            np.testing.assert_allclose(held[:6], np.zeros(6), atol=1e-6)
            self.assertAlmostEqual(held[6], 0.04)
        for arm in self.arms:
            arm.move_j.assert_not_called()
            arm.set_motion_mode.assert_not_called()
            for call in arm.move_mit.call_args_list:
                i = call.kwargs["joint_index"] - 1
                self.assertGreaterEqual(
                    call.kwargs["p_des"], assisted.mit.JOINT_LIMIT_LOWER[i]
                )
                self.assertLessEqual(
                    call.kwargs["p_des"], assisted.mit.JOINT_LIMIT_UPPER[i]
                )

    def test_startup_home_cancel_does_not_report_home_complete(self):
        self.rig.enable()
        before = [p.copy() for p in self.rig.initial_hold]
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, "startup home cancelled"):
            self.rig.home(should_stop=lambda: True)
        for actual, expected in zip(self.rig.initial_hold, before):
            np.testing.assert_allclose(actual, expected)

    def run_runtime_abort(self, *, unhealthy=False):
        positions = [self.q.copy(), self.q.copy()]
        positions[0][0], positions[1][0] = 0.2, -0.1
        for arm, position in zip(self.arms, positions):

            def angles(arm=arm, position=position):
                self.stamp += 0.01
                arm.get_arm_status.return_value.timestamp = self.stamp
                return SimpleNamespace(msg=position.tolist(), timestamp=self.stamp)

            arm.get_joint_angles.side_effect = angles
            arm.move_mit.side_effect = lambda position=position, **kw: (
                position.__setitem__(kw["joint_index"] - 1, kw["p_des"])
            )
        self.rig.config = replace(self.rig.config, abort_hold_seconds=0.5)
        sends = 0

        class AbortingRig(assisted.PiperRig):
            def send(rig, control, dt):
                nonlocal sends
                if rig.startup_phase == "running":
                    sends += 1
                    if sends == 2:
                        if unhealthy:
                            self.arms[
                                0
                            ].get_arm_status.return_value.msg.arm_status = (
                                "COMMUNICATION_ERR"
                            )
                        raise RuntimeError("injected runtime abort")
                return super().send(control, dt)

        receiver = MagicMock()
        receiver.recvfrom.side_effect = BlockingIOError
        publisher = create_autospec(assisted.MITTelemetryPublisher, instance=True)
        with ExitStack() as stack:
            stack.enter_context(patch.object(assisted, "PiperRig", AbortingRig))
            stack.enter_context(
                patch.object(assisted.mit, "build_arm", side_effect=self.arms)
            )
            stack.enter_context(
                patch.object(assisted.socket, "socket", return_value=receiver)
            )
            stack.enter_context(
                patch.object(assisted, "MITTelemetryPublisher", return_value=publisher)
            )
            result = assisted.run(self.rig.config, "test", "canL", "canF", execute=True)
        self.assertEqual(result, 1)
        self.assertEqual(publisher.publish.call_args.kwargs["phase"], "aborted")
        self.assertEqual(
            publisher.publish.call_args.kwargs["reason"], "injected runtime abort"
        )
        return positions, publisher

    def test_runtime_abort_finishes_real_teleop_recovery_before_disconnect(self):
        positions, publisher = self.run_runtime_abort()
        self.assertTrue(publisher.publish.call_args.kwargs["control"]["recovered_home"])
        phases = [call.kwargs["phase"] for call in publisher.publish.call_args_list]
        self.assertEqual(phases[0], "engaged")
        self.assertIn("recovering", phases)
        for arm, position in zip(self.arms, positions):
            np.testing.assert_allclose(position, np.zeros(6), atol=1e-6)
            arm.electronic_emergency_stop.assert_not_called()
            arm.move_j.assert_not_called()
            arm.disable.assert_not_called()

    def test_unhealthy_controller_prevents_abort_home_and_uses_stop(self):
        positions, publisher = self.run_runtime_abort(unhealthy=True)
        self.assertFalse(publisher.publish.call_args.kwargs["control"]["recovered_home"])
        for arm, position in zip(self.arms, positions):
            self.assertGreater(np.max(np.abs(position)), 0.1)
            arm.electronic_emergency_stop.assert_called_once()

    def test_abort_recovery_keeps_per_motor_feedback_watchdogs(self):
        original = self.arms[0].get_motor_states.side_effect
        self.arms[0].get_motor_states.side_effect = lambda joint: (
            SimpleNamespace(
                msg=SimpleNamespace(velocity=0.0, torque=0.0), timestamp=1.0
            )
            if joint == 2
            else original(joint)
        )
        self.rig.read()
        for arm in self.arms:
            arm.enable()
        self.advance(0.3)
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, "motor 2:.*timestamps"):
            self.rig.recover_home_after_abort(
                should_stop=lambda: False, on_progress=lambda: None
            )
        for arm in self.arms:
            arm.move_mit.assert_not_called()

    def test_cached_ready_status_does_not_declare_startup_ready(self):
        status = self.arms[0].get_arm_status.return_value
        # Joint feedback advances; the arm-status stream remains frozen.
        self.arms[0].get_arm_status.side_effect = lambda: SimpleNamespace(
            msg=status.msg, timestamp=1.0
        )
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, "status timestamps"):
            self.rig.enable()

    def test_slow_follower_enable_does_not_interrupt_leader_support(self):
        follower_ready_at = self.now + 0.38  # role settling + delayed motor enable
        self.arms[1].enable.side_effect = lambda: None
        self.arms[1].get_joints_enable_status_list.side_effect = lambda: (
            [self.now >= follower_ready_at] * 6
        )
        leader_ticks = []
        self.arms[0].move_mit.side_effect = lambda **kw: (
            leader_ticks.append(self.now) if kw["joint_index"] == 1 else None
        )
        self.rig.enable()
        self.assertTrue(self.rig.enabled)
        self.assertLess(leader_ticks[0], follower_ready_at)
        self.assertGreaterEqual(len(leader_ticks), 8)
        self.assertLessEqual(max(np.diff(leader_ticks)), 0.010001)

    def test_latched_stop_is_reported_before_enable_or_role_commands(self):
        self.arms[0].get_arm_status.return_value.msg.arm_status = "EMERGENCY_STOP(0x1)"
        with self.assertRaisesRegex(
            assisted.mit.TeleopAbort, "latched electronic stop"
        ):
            self.rig.enable()
        for arm in self.arms:
            arm.set_follower_mode.assert_not_called()
            arm.enable.assert_not_called()
            arm.reset.assert_not_called()

    def test_automatic_correction_uses_gravity_only_leader_and_supported_follower(self):
        self.rig.enable()
        control = AssistedControl(
            self.rig.config, *self.rig.read(), clock=lambda: self.now
        )
        control.mode = "POLICY"
        control.correcting = True
        self.rig.send(control, 0.01)
        self.assertTrue(
            all(c.kwargs["kp"] == 0 for c in self.arms[0].move_mit.call_args_list[-6:])
        )
        self.assertTrue(
            all(c.kwargs["kp"] == 10 for c in self.arms[1].move_mit.call_args_list[-6:])
        )


class AssistedRecoveryLifecycleTests(unittest.TestCase):
    def test_disconnect_waits_for_active_recovery_without_requesting_hold(self):
        from ruri.client.controllers.single_piper_assisted.controller import (
            SinglePiperAssistedController,
        )

        robot = SinglePiperAssistedController({"simulate": True})
        robot._io, robot._worker = io, worker = MagicMock(), MagicMock()
        io.snapshot.return_value = {"phase": "recovering"}
        robot.disconnect()
        io.request_mode.assert_not_called()
        worker.stop.assert_called_once_with(allow_watchdog_recovery=True)
        io.close.assert_called_once()

    def test_fault_while_waiting_for_hold_still_allows_recovery_to_finish(self):
        from ruri.client.controllers.single_piper_assisted.controller import (
            SinglePiperAssistedController,
        )

        robot = SinglePiperAssistedController({"simulate": True})
        robot._io, robot._worker = io, worker = MagicMock(), MagicMock()
        io.snapshot.side_effect = [{"phase": "engaged"}, {"phase": "recovering"}]
        io.request_mode.side_effect = TimeoutError("worker started recovery")
        robot.disconnect()
        worker.stop.assert_called_once_with(allow_watchdog_recovery=True)

    def test_process_manager_waits_for_recovery_instead_of_sending_sigint(self):
        from ruri.client.controllers.single_piper_assisted.controller import (
            ManagedAssistedProcess,
        )

        for from_telemetry in (True, False):
            with self.subTest(from_telemetry=from_telemetry):
                worker = ManagedAssistedProcess(SinglePiperAssistedConfig(), "test")
                worker.process = process = MagicMock()
                process.poll.return_value = None
                process.wait.return_value = 1
                if not from_telemetry:
                    worker.logs.append(
                        "ASSISTED RECOVERY: holding current poses, then MIT home"
                    )
                with patch(
                    "ruri.client.controllers.single_piper.mit_process.os.killpg"
                ) as kill:
                    worker.stop(allow_watchdog_recovery=from_telemetry)
                kill.assert_not_called()
                process.wait.assert_called_once_with(
                    timeout=worker.config.worker_shutdown_timeout_s
                )
                process.stdout.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
