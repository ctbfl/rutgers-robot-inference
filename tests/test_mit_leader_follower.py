import argparse
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ruri.client.controllers.single_piper.mit import leader_follower as mit_teleop


class FakeGravity:
    def __init__(self, torque):
        self._torque = np.asarray(torque, dtype=float)

    def torque(self, _q):
        return self._torque.copy()


class FakeArm:
    def __init__(self):
        self.commands = []

    def move_mit(self, **command):
        self.commands.append(command)


class FakeGripper:
    def __init__(self):
        self.commands = []

    def move_gripper_m(self, width, force):
        self.commands.append((width, force))


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


class FakeControlArm:
    def __init__(self, q):
        self.q = np.asarray(q, dtype=float)
        self.qd = np.zeros(6)
        self.stamp = 0.0

    def get_joints_enable_status_list(self):
        return [True] * 6

    def get_arm_status(self):
        return SimpleNamespace(msg=SimpleNamespace(
            ctrl_mode="CAN_CTRL",
            mode_feedback="MOVE_MIT",
            arm_status="NORMAL",
        ))

    def get_joint_angles(self):
        self.stamp += 0.001
        return SimpleNamespace(msg=self.q.tolist(), timestamp=self.stamp)

    def get_motor_states(self, joint):
        self.stamp += 0.001
        return SimpleNamespace(
            msg=SimpleNamespace(
                velocity=float(self.qd[joint - 1]),
                current=float(joint),
                torque=float(joint) / 10.0,
            ),
            timestamp=self.stamp,
        )

    def move_mit(self, joint_index, p_des, v_des, **_):
        self.q[joint_index - 1] = p_des
        self.qd[joint_index - 1] = v_des


class QuinticBlendTests(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(mit_teleop.quintic_blend(-1.0, 2.0), (0.0, 0.0))
        self.assertEqual(mit_teleop.quintic_blend(2.0, 2.0), (1.0, 0.0))
        self.assertEqual(mit_teleop.quintic_blend(3.0, 2.0), (1.0, 0.0))

    def test_midpoint(self):
        alpha, alpha_dot = mit_teleop.quintic_blend(1.0, 2.0)
        self.assertAlmostEqual(alpha, 0.5)
        self.assertAlmostEqual(alpha_dot, 0.9375)

    def test_monotonic(self):
        values = [mit_teleop.quintic_blend(t, 2.0)[0]
                  for t in np.linspace(0.0, 2.0, 101)]
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))


class ArmSampleTests(unittest.TestCase):
    def test_one_motor_feedback_read_provides_velocity_and_effort(self):
        arm = FakeControlArm(np.arange(6, dtype=float) * 0.1)

        sample = mit_teleop.read_sample(arm, "test")

        np.testing.assert_allclose(sample.q, arm.q)
        np.testing.assert_allclose(sample.qd, 0.0)
        np.testing.assert_allclose(sample.joint_effort, np.arange(1.0, 7.0) / 10.0)


class EngagementReferenceTests(unittest.TestCase):
    def test_starts_at_follower_and_ends_at_leader(self):
        qf = np.arange(6, dtype=float) * 0.1
        ql = qf + 0.2
        qdl = np.full(6, 0.3)

        q0, qd0 = mit_teleop.engagement_reference(qf, ql, qdl, 0.0, 2.0)
        np.testing.assert_allclose(q0, qf)
        np.testing.assert_allclose(qd0, 0.0)

        q1, qd1 = mit_teleop.engagement_reference(qf, ql, qdl, 2.0, 2.0)
        np.testing.assert_allclose(q1, ql)
        np.testing.assert_allclose(qd1, qdl)

    def test_midpoint_includes_blend_velocity(self):
        qf = np.zeros(6)
        ql = np.full(6, 0.2)
        qdl = np.zeros(6)
        q, qd = mit_teleop.engagement_reference(qf, ql, qdl, 1.0, 2.0)
        np.testing.assert_allclose(q, 0.1)
        np.testing.assert_allclose(qd, 0.1875)


class HomeReferenceTests(unittest.TestCase):
    def test_home_reference_starts_at_pose_and_ends_at_zero(self):
        start = np.array([0.2, 1.0, -0.8, 0.1, -0.2, 0.3])
        q0, qd0 = mit_teleop.home_reference(start, 0.0, 10.0)
        np.testing.assert_allclose(q0, start)
        np.testing.assert_allclose(qd0, 0.0)

        q1, qd1 = mit_teleop.home_reference(start, 10.0, 10.0)
        np.testing.assert_allclose(q1, 0.0)
        np.testing.assert_allclose(qd1, 0.0)

    def test_duration_bounds_quintic_peak_speed(self):
        leader = np.array([0.0, 1.0, -0.5, 0.0, 0.0, 0.0])
        follower = np.array([0.0, 0.8, -0.4, 0.0, 0.0, 0.0])
        duration = mit_teleop.recovery_home_duration(leader, follower, 0.2)
        self.assertAlmostEqual(duration, 1.875 / 0.2)
        _, velocity = mit_teleop.home_reference(leader, duration / 2, duration)
        self.assertLessEqual(float(np.max(np.abs(velocity))), 0.2 + 1e-12)

    def test_abort_recovery_holds_then_homes_both_arms(self):
        leader = FakeControlArm([0.2, 0.4, -0.3, 0.1, -0.2, 0.1])
        follower = FakeControlArm([0.1, 0.3, -0.2, 0.0, -0.1, 0.2])
        gravity = FakeGravity(np.zeros(6))
        args = argparse.Namespace(
            feedback_timeout=0.2,
            rate=100.0,
            abort_hold_seconds=0.5,
            abort_home_speed=0.5,
            follower_kp=10.0,
            follower_kd=0.8,
            max_reference_speed=3.0,
            grip_force=1.0,
            grip_base=0.1,
            max_joint_speed=4.0,
            max_track_error=0.35,
        )
        clock = FakeClock()
        mit_teleop._running = True
        with patch.object(mit_teleop.time, "perf_counter", clock.perf_counter), \
                patch.object(mit_teleop.time, "sleep", clock.sleep):
            recovered = mit_teleop.recover_home_after_abort(
                leader,
                follower,
                gravity,
                gravity,
                None,
                None,
                args,
            )

        self.assertTrue(recovered)
        np.testing.assert_allclose(leader.q, 0.0, atol=1e-12)
        np.testing.assert_allclose(follower.q, 0.0, atol=1e-12)

    def test_startup_home_independently_homes_both_arms_without_hold(self):
        leader = FakeControlArm([0.4, -0.2, 0.3, 0.1, -0.1, 0.2])
        follower = FakeControlArm([-0.3, 0.1, -0.2, 0.2, 0.1, -0.1])
        gravity = FakeGravity(np.zeros(6))
        args = argparse.Namespace(
            feedback_timeout=0.2,
            rate=100.0,
            follower_kp=10.0,
            follower_kd=0.8,
            max_reference_speed=3.0,
            grip_force=1.0,
            grip_base=0.1,
            max_joint_speed=4.0,
            max_track_error=0.35,
        )
        clock = FakeClock()
        mit_teleop._running = True
        with patch.object(mit_teleop.time, "perf_counter", clock.perf_counter), \
                patch.object(mit_teleop.time, "sleep", clock.sleep):
            homed = mit_teleop.home_both_with_mit(
                leader,
                follower,
                gravity,
                gravity,
                None,
                None,
                args,
                hold_seconds=0.0,
                home_speed=0.5,
                phase_label="STARTUP HOME",
            )

        self.assertTrue(homed)
        np.testing.assert_allclose(leader.q, 0.0, atol=1e-12)
        np.testing.assert_allclose(follower.q, 0.0, atol=1e-12)


class GravityCalibrationTests(unittest.TestCase):
    def test_both_calibrations_load_and_produce_finite_torque(self):
        for filename in (
            "calibration_leader.json",
            "calibration_follower.json",
        ):
            gravity = mit_teleop.GravityCompensator(
                mit_teleop.ASSET_ROOT / filename
            )
            torque = gravity.torque(np.zeros(6))
            self.assertEqual(torque.shape, (6,))
            self.assertTrue(np.all(np.isfinite(torque)))


class SafetyTests(unittest.TestCase):
    def test_execute_is_opt_in(self):
        args = mit_teleop.build_parser().parse_args([])
        self.assertFalse(args.execute)
        self.assertEqual(args.max_joint_speed, 4.0)
        self.assertEqual(args.start_home_speed, 0.2)
        self.assertEqual(args.abort_hold_seconds, 10.0)
        self.assertEqual(args.abort_home_speed, 0.2)
        self.assertEqual(args.telemetry_address, "")

    def test_torque_ceiling_aborts(self):
        with self.assertRaises(mit_teleop.TeleopAbort):
            mit_teleop.bounded_torque([0.0, 13.0, 0.0, 0.0, 0.0, 0.0], "test")

    def test_new_entrypoint_has_no_planner_or_force_estimator(self):
        source = Path(mit_teleop.__file__).read_text()
        self.assertNotIn(".move_j(", source)
        self.assertNotIn(".move_js(", source)
        self.assertNotIn("ExternalTorqueEstimator", source)

    def test_rejects_aggressive_default_overrides(self):
        args = argparse.Namespace(
            rate=100.0,
            leader_kd=0.2,
            follower_kp=80.0,
            follower_kd=0.8,
            engage_seconds=2.0,
            start_home_speed=0.2,
            max_start_gap=0.35,
            max_track_error=0.35,
            max_joint_speed=2.5,
            max_reference_speed=3.0,
            feedback_timeout=0.2,
            seconds=0.0,
            abort_hold_seconds=10.0,
            abort_home_speed=0.2,
            grip_force=1.0,
            grip_base=0.1,
            grip_gain=1.0,
            grip_max_force=5.0,
        )
        with self.assertRaises(ValueError):
            mit_teleop.validate_args(args)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.sample = mit_teleop.ArmSample(
            q=np.arange(6, dtype=float) * 0.1,
            qd=np.arange(6, dtype=float) * 0.01,
            joint_effort=np.arange(6, dtype=float) * 0.1,
            stamps=(1.0,) * 7,
        )

    def test_leader_is_gravity_plus_damping_with_zero_impedance(self):
        arm = FakeArm()
        gravity = FakeGravity([0.0, 2.0, -3.0, 0.4, -0.5, 0.1])
        torque = mit_teleop.send_leader(arm, self.sample, gravity, damping=0.2)

        np.testing.assert_allclose(
            torque,
            gravity.torque(self.sample.q) - 0.2 * self.sample.qd,
        )
        self.assertEqual(len(arm.commands), 6)
        self.assertTrue(all(command["kp"] == 0.0 for command in arm.commands))
        self.assertTrue(all(command["kd"] == 0.0 for command in arm.commands))

    def test_follower_streams_impedance_and_own_gravity(self):
        arm = FakeArm()
        gravity = FakeGravity([0.0, 2.0, -3.0, 0.4, -0.5, 0.1])
        q_des = self.sample.q + 0.05
        qd_des = np.full(6, 0.2)
        torque = mit_teleop.send_follower(
            arm,
            self.sample,
            q_des,
            qd_des,
            gravity,
            kp=10.0,
            kd=0.8,
            max_reference_speed=3.0,
        )

        np.testing.assert_allclose(torque, gravity.torque(self.sample.q))
        self.assertEqual(len(arm.commands), 6)
        for index, command in enumerate(arm.commands):
            self.assertEqual(command["joint_index"], index + 1)
            self.assertAlmostEqual(command["p_des"], q_des[index])
            self.assertAlmostEqual(command["v_des"], qd_des[index])
            self.assertEqual(command["kp"], 10.0)
            self.assertEqual(command["kd"], 0.8)
            self.assertAlmostEqual(command["t_ff"], torque[index])


class FeedbackWatchdogTests(unittest.TestCase):
    def test_repeated_timestamps_abort_after_timeout(self):
        watchdog = mit_teleop.FeedbackWatchdog(timeout=0.2)
        stamps = (1.0,) * 7
        watchdog.observe(stamps, 10.0)
        watchdog.observe(stamps, 10.1)
        with self.assertRaises(mit_teleop.TeleopAbort):
            watchdog.observe(stamps, 10.21)

    def test_new_timestamp_resets_timeout(self):
        watchdog = mit_teleop.FeedbackWatchdog(timeout=0.2)
        watchdog.observe((1.0,) * 7, 10.0)
        watchdog.observe((2.0,) * 7, 10.19)
        watchdog.observe((2.0,) * 7, 10.30)


class GripperCommandTests(unittest.TestCase):
    def test_free_space_uses_low_base_force(self):
        leader = FakeGripper()
        follower = FakeGripper()
        leader_sample = mit_teleop.GripperSample(0.030, 0.0, 1.0)
        follower_sample = mit_teleop.GripperSample(0.030, 0.0, 1.0)

        difference, measured, rendered = mit_teleop.send_grippers(
            leader,
            follower,
            leader_sample,
            follower_sample,
            follower_force=1.0,
            leader_base_force=0.1,
            feedback_gain=1.0,
            leader_max_force=5.0,
        )

        self.assertAlmostEqual(difference, 0.0)
        self.assertAlmostEqual(measured, 0.0)
        self.assertAlmostEqual(rendered, 0.1)
        self.assertEqual(follower.commands, [(0.030, 1.0)])
        self.assertEqual(leader.commands, [(0.030, 0.1)])

    def test_contact_targets_follower_width_and_reflects_measured_force(self):
        leader = FakeGripper()
        follower = FakeGripper()
        leader_sample = mit_teleop.GripperSample(0.020, 0.0, 1.0)
        follower_sample = mit_teleop.GripperSample(0.030, -1.4, 1.0)

        difference, measured, rendered = mit_teleop.send_grippers(
            leader,
            follower,
            leader_sample,
            follower_sample,
            follower_force=1.0,
            leader_base_force=0.1,
            feedback_gain=1.0,
            leader_max_force=5.0,
        )

        self.assertAlmostEqual(difference, -0.010)
        self.assertAlmostEqual(measured, 1.4)
        self.assertAlmostEqual(rendered, 1.5)
        self.assertEqual(follower.commands, [(0.020, 1.0)])
        self.assertEqual(leader.commands, [(0.030, 1.5)])

    def test_rendered_force_is_clamped(self):
        leader = FakeGripper()
        follower = FakeGripper()
        sample = mit_teleop.GripperSample(0.025, 20.0, 1.0)
        _, _, rendered = mit_teleop.send_grippers(
            leader,
            follower,
            sample,
            sample,
            follower_force=1.0,
            leader_base_force=0.1,
            feedback_gain=1.0,
            leader_max_force=5.0,
        )
        self.assertAlmostEqual(rendered, 5.0)
        self.assertEqual(leader.commands, [(0.025, 5.0)])


class JointLimitTests(unittest.TestCase):
    def test_limits_match_the_recording_normalizer(self):
        # The MIT loop duplicates these numbers so it never imports the dataset
        # layer. If the two disagree, a demonstration can hold a target that
        # inference cannot command, which is what the clamp exists to prevent.
        from ruri.client.controllers.single_piper.normalization import (
            CALIBRATION_RANGES,
            JOINT_NAMES,
        )

        for index, name in enumerate(JOINT_NAMES):
            raw_min, raw_max = CALIBRATION_RANGES[name]
            self.assertAlmostEqual(
                mit_teleop.JOINT_LIMIT_LOWER[index], np.radians(raw_min / 1000.0),
                places=12, msg=f"{name} lower limit drifted",
            )
            self.assertAlmostEqual(
                mit_teleop.JOINT_LIMIT_UPPER[index], np.radians(raw_max / 1000.0),
                places=12, msg=f"{name} upper limit drifted",
            )

    def test_in_range_target_passes_through(self):
        qd = np.full(6, 0.3)
        clamped, velocity = mit_teleop.clamp_joint_target(np.zeros(6), qd)

        np.testing.assert_allclose(clamped, np.zeros(6))
        np.testing.assert_allclose(velocity, qd)

    def test_out_of_range_target_is_clamped(self):
        clamped, _ = mit_teleop.clamp_joint_target(
            mit_teleop.JOINT_LIMIT_LOWER - 0.2, np.zeros(6)
        )
        np.testing.assert_allclose(clamped, mit_teleop.JOINT_LIMIT_LOWER)

    def test_clamped_velocity_may_return_but_not_leave(self):
        outside = mit_teleop.JOINT_LIMIT_UPPER + 0.1

        _, held = mit_teleop.clamp_joint_target(outside, np.full(6, 0.5))
        _, released = mit_teleop.clamp_joint_target(outside, np.full(6, -0.5))

        np.testing.assert_allclose(held, np.zeros(6))
        np.testing.assert_allclose(released, np.full(6, -0.5))


class LeaderIsUnconstrainedTests(unittest.TestCase):
    """The box is enforced only on the follower target.

    The operator closes the loop on the follower, which stops at the boundary,
    so no wall torque is rendered on the leader and it stays a free handle.
    """

    def test_leader_command_is_gravity_and_damping_only_outside_the_box(self):
        gravity = FakeGravity([0.0, 2.0, -3.0, 0.4, -0.5, 0.1])
        sample = mit_teleop.ArmSample(
            q=mit_teleop.JOINT_LIMIT_UPPER + 1.0,
            qd=np.full(6, 0.1),
            joint_effort=np.zeros(6),
            stamps=(1.0,) * 7,
        )

        torque = mit_teleop.send_leader(FakeArm(), sample, gravity, 0.2)

        np.testing.assert_allclose(
            torque, gravity.torque(sample.q) - 0.2 * sample.qd
        )

    def test_no_wall_is_exposed(self):
        for name in ("joint_limit_wall", "LIMIT_WALL_KP", "LIMIT_WALL_TORQUE"):
            self.assertFalse(
                hasattr(mit_teleop, name),
                f"{name} should have been removed with the leader wall",
            )


if __name__ == "__main__":
    unittest.main()
