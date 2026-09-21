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


class MockPiperRigTestCase(unittest.TestCase):
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
        self.rig = self.make_rig()
        self.addCleanup(self.rig.close)

    def make_rig(self):
        return assisted.PiperRig(SinglePiperAssistedConfig(), "canL", "canF")

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

class AssistedHardwareTests(MockPiperRigTestCase):
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
                call.kwargs["kp"] == 3
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

    def test_distant_policy_proposal_moves_shared_reference_at_bounded_speed(self):
        self.rig.enable()
        leader, follower = self.rig.read()
        control = AssistedControl(
            self.rig.config, leader, follower, clock=lambda: self.now
        )
        target = np.r_[self.q.copy(), 0.04]
        target[0] += 0.8  # A distant proposal must not jump the motor reference.
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
        self.advance(.01)
        control.tick(leader, follower)
        self.rig.send(control, 0.01)
        for arm in self.arms:
            sent = [call.kwargs["p_des"] for call in arm.move_mit.call_args_list[-6:]]
            np.testing.assert_allclose(sent, control.target[:6])
            self.assertGreater(sent[0], 0.)
            self.assertLessEqual(sent[0], .00150001)
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

    def test_automatic_correction_retains_fixed_impedance_on_both_arms(self):
        self.rig.enable()
        control = AssistedControl(
            self.rig.config, *self.rig.read(), clock=lambda: self.now
        )
        control.mode = "POLICY"
        control.correcting = True
        self.rig.send(control, 0.01)
        self.assertTrue(
            all(c.kwargs["kp"] == 3 for c in self.arms[0].move_mit.call_args_list[-6:])
        )
        self.assertTrue(
            all(c.kwargs["kp"] == 10 for c in self.arms[1].move_mit.call_args_list[-6:])
        )


class DualHandoffHardwareTests(MockPiperRigTestCase):
    def make_rig(self):
        return assisted.PiperRig(SinglePiperAssistedConfig(gravity_handoff=True,
            reference_speed=4., max_reference_speed=4., policy_reference_speed=.5), "canL", "canF")

    def test_leader_gravity_blend_and_follower_fixed_gains_use_separate_velocities(self):
        self.rig.enable()
        control = AssistedControl(self.rig.config, *self.rig.read(), clock=lambda: self.now)
        control.mode = 'POLICY'
        qd = np.array([.1, -.2, .3, -.4, .2, -.1])
        self.rig.samples = (replace(self.rig.samples[0], qd=qd), self.rig.samples[1])
        self.rig.leader_gravity.torque = lambda q: np.full(6, .3)
        self.rig.follower_gravity.torque = lambda q: np.full(6, .6)
        self.assertEqual(assisted.mit.GravityCompensator.call_count, 2)
        assisted.mit.GravityCompensator.assert_any_call(assisted.mit.ASSET_ROOT / "calibration_follower.json")
        control.reference_velocity = np.full(6, .05)
        control.follower_velocity = np.array([4., -4., 2., -.5, .1, 0.])
        control.target[0] += .6  # Intentional gap is no longer a raw-error abort.
        control.leader_target[0] += .6
        for alpha in (0., .5, 1., .5, 0.):
            control.reference.alpha = alpha
            self.rig.send(control, .01)
            leader = [x.kwargs for x in self.arms[0].move_mit.call_args_list[-6:]]
            follower = [x.kwargs for x in self.arms[1].move_mit.call_args_list[-6:]]
            np.testing.assert_allclose([x['kp'] for x in leader], (1-alpha)*3.)
            np.testing.assert_allclose([x['kd'] for x in leader], (1-alpha)*.5)
            np.testing.assert_allclose([x['t_ff'] for x in leader], .3-alpha*.2*qd)
            np.testing.assert_allclose([x['v_des'] for x in leader], 0. if alpha == 1. else .05)
            np.testing.assert_allclose([x['kp'] for x in follower], 10.)
            np.testing.assert_allclose([x['kd'] for x in follower], .8)
            np.testing.assert_allclose([x['t_ff'] for x in follower], .6)
            np.testing.assert_allclose([x['p_des'] for x in follower], control.target[:6])
            np.testing.assert_allclose([x['v_des'] for x in follower], control.follower_velocity)

    def test_live_interpolation_reaches_left_sdk_and_leader_remains_gravity_only(self):
        self.rig.enable()
        control = AssistedControl(self.rig.config, *self.rig.read(), clock=lambda: self.now)
        control.mode = 'POLICY'
        leader, follower = self.rig.read()
        leader[0] += .3
        saw_human = False
        for k in range(140):
            self.advance(.01)
            control.last_heartbeat = control.last_command = self.now
            if control.reference.phase == 'human':
                leader[0] += .001
                saw_human = True
            control.tick(leader, follower)
            self.rig.send(control, .01)
            left = [x.kwargs for x in self.arms[1].move_mit.call_args_list[-6:]]
            np.testing.assert_allclose([x['p_des'] for x in left], control.target[:6])
            np.testing.assert_allclose([x['v_des'] for x in left], control.follower_velocity)
        self.assertTrue(saw_human)
        self.assertEqual(control.reference.alpha, 1.)
        np.testing.assert_allclose(control.target, leader)

    def test_original_left_gravity_also_applies_to_enable_home_and_hold(self):
        self.rig.follower_gravity.torque = lambda q: np.full(6, .6)
        self.rig.enable()
        for arm in self.arms:
            arm.move_mit.side_effect = lambda **kw: self.q.__setitem__(kw['joint_index']-1, kw['p_des'])
        self.rig.home()
        control = AssistedControl(self.rig.config, *self.rig.read(), clock=lambda: self.now)
        self.rig.send(control, .01)
        self.assertTrue(control.status()['follower_gravity_compensation'])
        for call in self.arms[1].move_mit.call_args_list:
            self.assertEqual(call.kwargs['t_ff'], .6)
            self.assertEqual(call.kwargs['kp'], 10.)
            self.assertEqual(call.kwargs['kd'], .8)
        for call in self.arms[0].move_mit.call_args_list:
            self.assertEqual(call.kwargs['kp'], 3.)
            self.assertEqual(call.kwargs['kd'], .5)

    def test_left_mit_writes_match_single_piper_for_identical_reference_and_feedback(self):
        from ruri.client.controllers.single_piper.mit import policy_controller as original
        self.rig.enable()
        control = AssistedControl(self.rig.config, *self.rig.read(), clock=lambda: self.now)
        control.mode = 'POLICY'
        control.target[:6] = [.3, .6, -.8, .2, -.3, .1]
        control.follower_velocity = np.array([.1, -.2, .3, -.4, .2, -.1])
        self.rig.follower_gravity.torque = lambda q: np.arange(6)*.17
        expected = MagicMock()
        original.send_position_impedance(expected, self.rig.samples[1], control.target[:6],
            control.follower_velocity, self.rig.follower_gravity,
            original.DEFAULT_FOLLOWER_KP, original.DEFAULT_FOLLOWER_KD,
            self.rig.config.max_reference_speed, 'original single Piper')
        for alpha in (0., .5, 1.):
            control.reference.alpha = alpha
            self.rig.send(control, .01)
            self.assertEqual(self.arms[1].move_mit.call_args_list[-6:], expected.move_mit.call_args_list)

    def test_explicit_reset_only_resets_latched_arm_before_enable(self):
        self.rig.config = replace(self.rig.config, reset_latched_stop=True)
        status = self.arms[0].get_arm_status.return_value
        status.msg.arm_status = 'EMERGENCY_STOP(0x1)'
        def reset():
            status.msg.arm_status = 'NORMAL'
            status.timestamp += .01
        self.arms[0].reset.side_effect = reset
        self.rig.enable()
        self.arms[0].reset.assert_called_once()
        self.arms[1].reset.assert_not_called()
        self.assertTrue(self.rig.enabled)

    def test_stop_reset_requires_fresh_normal_feedback_before_any_enable(self):
        self.rig.config = replace(self.rig.config, reset_latched_stop=True)
        status = self.arms[0].get_arm_status.return_value
        status.msg.arm_status = 'EMERGENCY_STOP(0x1)'
        self.arms[0].reset.side_effect = lambda: setattr(status.msg, 'arm_status', 'NORMAL')
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, 'reset lacked fresh NORMAL'):
            self.rig.enable()
        self.arms[0].reset.assert_called_once()
        for arm in self.arms:
            arm.enable.assert_not_called()
            arm.set_follower_mode.assert_not_called()

    def test_explicit_reset_does_not_reset_healthy_arms(self):
        self.rig.config = replace(self.rig.config, reset_latched_stop=True)
        self.rig.enable()
        for arm in self.arms:
            arm.reset.assert_not_called()


class LeaderOnlyHardwareTests(MockPiperRigTestCase):
    def make_rig(self):
        return assisted.LeaderRig(SinglePiperAssistedConfig(leader_only=True), "canL")

    def test_four_rad_reference_reaches_sdk_without_old_three_rad_clip(self):
        self.rig.config = replace(self.rig.config, reference_speed=4.,
                                  policy_reference_speed=.5, max_reference_speed=4.)
        self.rig.enable()
        control = AssistedControl(self.rig.config, *self.rig.read(), clock=lambda: self.now)
        control.reference_velocity = np.array([4., -4., 3.5, -3.5, .5, 0.])
        self.rig.send(control, .01)
        calls = self.arms[0].move_mit.call_args_list[-6:]
        np.testing.assert_allclose([call.kwargs['v_des'] for call in calls], control.reference_velocity)
        self.assertEqual(self.arms[1].mock_calls, [])

    def test_single_driver_blends_impedance_into_exact_gravity_teleop(self):
        assisted.mit.build_arm.assert_called_once_with("canL")
        self.rig.enable()
        self.assertIsNone(self.rig.follower)
        self.arms[1].assert_not_called()
        self.assertEqual(self.arms[1].mock_calls, [])
        control = AssistedControl(self.rig.config, *self.rig.read(), clock=lambda: self.now)
        control.mode = 'POLICY'
        qd = np.array([.1, -.2, .3, -.4, .2, -.1])
        self.rig.samples = (replace(self.rig.samples[0], qd=qd),)
        gravity = np.arange(6)*.1
        self.rig.leader_gravity.torque = lambda q: gravity.copy()
        control.leader_target[:6] += .01
        for alpha in (0., .25, .5, .75, 1., .75, .5, .25, 0.):
            control.reference.alpha = alpha
            control.reference_velocity = np.array([.03, -.04, .02, .01, -.01, .0])
            self.rig.send(control, .01)
            calls = self.arms[0].move_mit.call_args_list[-6:]
            np.testing.assert_allclose([c.kwargs["v_des"] for c in calls],
                                       0. if alpha == 1. else control.reference_velocity)
            np.testing.assert_allclose([c.kwargs["t_ff"] for c in calls], gravity-alpha*.2*qd)
            self.assertTrue(all(c.kwargs["kp"] == (1-alpha)*3. and
                                c.kwargs["kd"] == (1-alpha)*.5 for c in calls))
            actual_torque = np.array([x.kwargs['t_ff']+x.kwargs['kp']*(x.kwargs['p_des']-self.q[i])
                                      +x.kwargs['kd']*(x.kwargs['v_des']-qd[i])
                                      for i,x in enumerate(calls)])
            np.testing.assert_allclose(actual_torque,
                gravity+(1-alpha)*(3.*.01+.5*(control.reference_velocity-qd))-alpha*.2*qd)
        self.assertNotIn("follower_q", self.rig.telemetry())
        self.rig.emergency_stop()
        self.rig.close()
        self.arms[0].electronic_emergency_stop.assert_called_once()
        self.assertEqual(self.arms[1].mock_calls, [])

    def test_gravity_phase_ignores_position_target_but_preserves_torque_guard(self):
        self.rig.enable()
        control = AssistedControl(self.rig.config, *self.rig.read(), clock=lambda: self.now)
        control.mode = 'POLICY'
        control.reference.alpha = 1.
        control.leader_target[0] += .8
        self.rig.send(control, .01)
        calls = self.arms[0].move_mit.call_args_list[-6:]
        np.testing.assert_allclose([x.kwargs['p_des'] for x in calls], self.q)
        self.assertTrue(all(x.kwargs['kp'] == 0. and x.kwargs['kd'] == 0. for x in calls))
        self.rig.leader_gravity.torque = lambda q: np.ones(6)*100.
        with self.assertRaises(assisted.mit.TeleopAbort):
            self.rig.send(control, .01)
        control.reference.alpha = float('nan')
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, 'weight'):
            self.rig.send(control, .01)

    def test_single_home_reaches_zero_and_preserves_fixed_gains(self):
        self.rig.enable()
        self.arms[0].move_mit.side_effect = lambda **kw: self.q.__setitem__(kw["joint_index"]-1, kw["p_des"])
        self.rig.home()
        np.testing.assert_allclose(self.q, 0., atol=1e-8)
        np.testing.assert_allclose(self.rig.initial_hold[0][:6], 0.)
        self.assertTrue(all(c.kwargs["kp"] == 3. and c.kwargs["kd"] == .5 for c in self.arms[0].move_mit.call_args_list))
        self.assertEqual(self.arms[1].mock_calls, [])

    def test_single_cancel_and_cached_status_cannot_enable_ready(self):
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, "cancelled"):
            self.rig.enable(should_stop=lambda: True)
        self.arms[0].enable.assert_not_called()
        status = self.arms[0].get_arm_status.return_value
        self.arms[0].get_arm_status.side_effect = lambda: SimpleNamespace(msg=status.msg, timestamp=1.)
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, "advancing healthy"):
            self.rig.enable()
        self.assertFalse(self.rig.enabled)
        self.assertEqual(self.arms[1].mock_calls, [])

    def test_single_position_error_does_not_abort_any_handoff_phase(self):
        self.rig.enable()
        control = AssistedControl(self.rig.config, *self.rig.read(), clock=lambda: self.now)
        for error in (.351, .6):
            control.leader_target[:6] = self.q
            control.leader_target[0] += error
            for mode, phase, alpha in (("POLICY", "policy", 0.), ("POLICY", "blend", .73),
                                       ("POLICY", "human", 1.), ("POLICY", "recover", .5),
                                       ("HUMAN", "blend", .5), ("HOLD", "policy", 0.)):
                with self.subTest(error=error, mode=mode, phase=phase):
                    control.mode, control.reference.phase = mode, phase
                    control.reference.alpha = alpha
                    before = self.arms[0].move_mit.call_count
                    self.rig.send(control, .01)
                    self.assertEqual(self.arms[0].move_mit.call_count, before+6)

    def test_single_stale_status_still_stops_motion(self):
        self.rig.enable()
        control = AssistedControl(self.rig.config, *self.rig.read(), clock=lambda: self.now)
        self.rig.send(control, .01)
        self.advance(.3)
        with self.assertRaisesRegex(assisted.mit.TeleopAbort, "timestamps"):
            self.rig.send(control, .01)

    def test_controller_connect_discovers_only_right_and_opens_no_camera(self):
        from ruri.client.controllers.single_piper_assisted import controller as boundary
        robot = boundary.SinglePiperAssistedController({"leader_only": True, "camera_enabled": False})
        with patch.object(boundary, "SinglePiperLeaderFollowerTeleopController") as discovery, patch.object(boundary, "SinglePiperController") as camera:
            robot.connect()
            self.assertTrue(robot.is_connected)
            d = discovery.return_value
            d.connect.assert_not_called()
            d._discover_registered_arm.assert_called_once()
            self.assertEqual(d._discover_registered_arm.call_args.kwargs["side"], "right")
            camera.assert_not_called()
            self.assertEqual(robot._images(), ({}, {}))
            robot.disconnect()


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
