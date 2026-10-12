"""Explicit gripper input across policy, alignment, gravity and re-arming."""

from dataclasses import replace
import unittest

import numpy as np

from ruri.client.controllers.single_piper_assisted.config import SinglePiperAssistedConfig
from ruri.client.controllers.single_piper_assisted.control import AssistedControl
from ruri.client.controllers.single_piper_assisted.follower_handoff import FollowerHandoff


class GripperTakeoverTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.config = SinglePiperAssistedConfig(gravity_handoff=True, human_trigger="gripper",
            gripper_max_m=.1,
            reference_speed=4., max_reference_speed=4., policy_reference_speed=.5)
        self.leader = np.array([0., .4, -.4, 0., 0., 0., .04])
        self.follower = self.leader.copy()
        self.control = AssistedControl(self.config, self.leader, self.follower, clock=lambda: self.now)
        self.control.mode = "POLICY"

    def tick(self, proposal=None):
        c = self.control
        self.now += .01
        c.last_heartbeat = c.last_command = self.now
        c.policy_reference = (c.target if proposal is None else proposal).copy()
        c.policy_valid, c.policy_observation_ns = True, int(self.now*1e9)
        before = c.target.copy()
        c.tick(self.leader, self.follower, self.now)
        self.assertLessEqual(max(abs(c.target[:6]-before[:6])), .04+1e-12)
        return c

    def arm(self):
        self.leader[6] = self.control.gripper_handoff.open_width
        for _ in range(5):
            self.tick()
        self.assertEqual(self.control.gripper_handoff.phase, "armed")

    def trigger(self):
        self.arm()
        self.leader[6] -= .01  # No need to reach fully closed.
        self.tick()
        self.assertEqual(self.control.reference.phase, "gripper_align")

    def enter_human(self):
        self.trigger()
        for _ in range(250):
            self.leader[6] = self.control.leader_target[6]
            c = self.tick()
            self.assertNotEqual(c.reference.phase, "blend")
            if c.reference.phase == "human":
                return
        self.fail("alignment did not enter gravity")

    def test_joint_deflection_never_triggers_and_left_gripper_still_tracks_policy(self):
        self.arm()
        self.leader[0] += .6
        proposal = self.follower.copy(); proposal[6] = .01
        for _ in range(200):
            c = self.tick(proposal)
            self.assertEqual(c.reference.phase, "policy")
            self.assertEqual(c.reference.alpha, 0.)
            self.assertEqual(c.leader_target[6], c.gripper_handoff.open_width)
        self.assertAlmostEqual(c.target[6], .01)

    def test_startup_closed_or_partial_open_does_not_trigger(self):
        for width in (.0, .04, .065):
            self.leader[6] = width
            for _ in range(30):
                self.tick()
                self.assertEqual(self.control.reference.phase, "policy")
                self.assertEqual(self.control.gripper_handoff.phase, "opening")

    def test_sub_centimetre_closure_is_ignored_then_threshold_triggers_immediately(self):
        self.arm()
        self.leader[6] -= .009
        for _ in range(30):
            self.tick()
        self.assertEqual(self.control.reference.phase, "policy")
        self.assertEqual(self.control.gripper_handoff.phase, "armed")
        self.leader[6] -= .001
        self.tick()
        self.assertEqual(self.control.reference.phase, "gripper_align")

    def test_slow_squeeze_over_ten_seconds_uses_fixed_threshold_and_weak_spring(self):
        self.arm()
        baseline = self.leader[6]
        for k in range(1, 1000):
            self.leader[6] = baseline-k*.00001
            c = self.tick()
            self.assertEqual(c.reference.phase, "policy")
            self.assertEqual(c.gripper_handoff.phase, "armed")
            self.assertAlmostEqual(c.leader_target[6], .1)
            self.assertLessEqual(c.gripper_handoff.opening_force(self.leader[6]), .05)
            self.assertAlmostEqual(c.gripper_handoff.closure, baseline-self.leader[6])
            self.assertEqual(c.target[6], .04)
        self.leader[6] = baseline-.01
        self.tick()
        self.assertEqual(c.reference.phase, "gripper_align")

    def test_absolute_ninety_mm_threshold_independent_of_previous_measurements(self):
        # Armed at 98.5 mm (within the arrival tolerance). Always trigger at
        # 90 mm, not 88.5 mm, and do not accumulate a per-tick distance.
        self.leader[6] = .0985
        for _ in range(5):
            self.tick()
        self.assertEqual(self.control.gripper_handoff.phase, "armed")
        for width in (.095, .1048, .094, .09001):
            self.leader[6] = width
            self.tick()
            self.assertEqual(self.control.reference.phase, "policy")
        self.leader[6] = .090
        self.tick()
        self.assertEqual(self.control.reference.phase, "gripper_align")

    def test_large_closure_triggers_in_one_sample(self):
        self.arm()
        self.leader[6] = .04
        self.tick()
        self.assertEqual(self.control.reference.phase, "gripper_align")

    def test_alignment_freezes_measured_left_grip_and_rejects_old_policy(self):
        # The left servo has not yet reached the policy's old gripper command.
        self.control.target[6] = .02
        revision = self.control.reference_revision
        self.trigger()
        c = self.control
        self.assertEqual(c.target[6], self.follower[6])
        self.assertEqual(c.reference_revision, revision+1)
        self.assertFalse(c.policy_valid)
        self.assertTrue(c.status()["intervention_active"])
        goal = self.follower.copy(); goal[0], goal[6] = 1., .075
        for _ in range(180):
            self.tick(goal)
            self.assertEqual(c.target[6], .04)
            self.assertEqual(c.reference.phase, "gripper_align")
            self.assertEqual(c.reference.alpha, 0.)
            np.testing.assert_allclose(c.leader_target[:6], c.target[:6])
        # Commanded alignment alone is insufficient: measured right is .07.
        self.assertAlmostEqual(c.leader_target[6], .04)
        self.assertAlmostEqual(c.target[0], 0.)

    def test_fast_alignment_at_recorded_gaps_waits_for_measured_arrival(self):
        from ruri.client.controllers.single_piper_assisted.gripper_handoff import GripperHandoff
        # Three handovers in the accepted hardware trial took 1.3/1.3/2.1 s.
        # Use their follower openings; retain the feedback handshake at speed.
        for target in (.0428, .0433, .0124):
            with self.subTest(target=target):
                h = GripperHandoff(self.config, .1, 0., .1)
                for _ in range(5):
                    h.observe(.1, target, .01)
                triggered, ready = h.observe(.09, target, .01)
                self.assertTrue(triggered)
                elapsed = 0.
                while abs(h.leader_target-target) > 1e-12:
                    before = h.leader_target
                    _, ready = h.observe(.09, target, .01)
                    elapsed += .01
                    self.assertFalse(ready)  # A command is not measured arrival.
                    self.assertLessEqual(abs(h.leader_target-before), .002+1e-12)
                self.assertLessEqual(elapsed, .4+1e-12)
                for _ in range(4):
                    _, ready = h.observe(target, target, .01)
                    self.assertFalse(ready)
                _, ready = h.observe(target, target, .01)
                self.assertTrue(ready)
        self.assertEqual(self.config.gripper_reference_speed, .04)

    def test_measured_alignment_directly_enters_pure_gravity_and_retains_left_reference(self):
        self.leader[0] += .2
        self.enter_human()
        c = self.control
        self.assertEqual(c.reference.alpha, 1.)
        self.assertEqual(c.leader_gain, 0.)
        self.assertAlmostEqual(c.target[0], 0.)
        self.assertAlmostEqual(c.leader_target[0], .2)
        self.assertEqual(c.follower_handoff.progress, 0.)
        self.assertEqual(c.target[6], .04)
        self.assertLessEqual(abs(self.leader[6]-.04), .002)
        self.leader[6] = .02
        c = self.tick()
        self.assertEqual(c.target[6], .02)

    def test_release_reopens_only_right_then_rearms_for_second_gesture(self):
        self.enter_human()
        c = self.control
        alphas = []
        for _ in range(150):
            self.follower = c.target.copy()
            c = self.tick()
            if c.reference.phase == "recover":
                alphas.append(c.reference.alpha)
            if c.reference.phase == "policy":
                break
        self.assertTrue(any(0. < a < 1. for a in alphas))
        self.assertEqual(c.reference.phase, "policy")
        self.assertEqual(c.gripper_handoff.phase, "opening")
        self.assertEqual(c.leader_target[6], c.gripper_handoff.open_width)
        left_grip = c.target[6]
        for width in np.linspace(self.leader[6], c.gripper_handoff.open_width, 40):
            self.leader[6] = width
            self.tick()
            self.assertEqual(c.target[6], left_grip)
            self.assertEqual(c.reference.phase, "policy")
        self.trigger()
        self.assertEqual(c.intervention_id, 2)

    def test_pause_during_alignment_cancels_pending_takeover(self):
        self.trigger()
        c = self.control
        c._revoke("HOLD", "paused")
        self.leader[6] = .04
        self.tick()
        self.assertEqual(c.mode, "HOLD")
        self.assertEqual(c.status()["gripper_handoff_phase"], "inactive")
        self.assertAlmostEqual(c.leader_target[6], .09)
        c.mode = "POLICY"
        self.tick()
        self.assertEqual(c.gripper_handoff.phase, "opening")
        self.assertEqual(c.reference.phase, "policy")

    def test_moving_shared_reference_brakes_during_alignment(self):
        self.arm()
        goal = self.follower.copy(); goal[0] = 1.
        for _ in range(10):
            self.tick(goal)
        self.leader[6] -= .01
        velocity = self.control.reference_velocity.copy()
        self.tick(goal)
        np.testing.assert_allclose(self.control.reference_velocity, velocity)
        speeds = []
        for _ in range(10):
            c = self.tick(goal)
            np.testing.assert_array_equal(c.target[:6], c.leader_target[:6])
            speeds.append(c.reference_velocity[0])
        self.assertTrue(all(a >= b-1e-12 for a, b in zip(speeds, speeds[1:])))
        self.assertAlmostEqual(speeds[-1], 0.)

    def test_direct_handoff_preserves_initial_follower_velocity(self):
        h = FollowerHandoff(self.config, np.zeros(6), -np.ones(6), np.ones(6))
        v = np.full(6, .3)
        h.sync(np.zeros(6), v)
        r, actual_v = h.follow(np.full(6, .2), .01)
        np.testing.assert_array_equal(r, 0.)
        np.testing.assert_array_equal(actual_v, v)
        r, actual_v = h.follow(np.full(6, .2), .01)
        np.testing.assert_allclose(actual_v, .3, atol=.003)
        for _ in range(49):
            r, _ = h.follow(np.full(6, .2), .01)
        np.testing.assert_allclose(r, .2)

    def test_gripper_trigger_requires_dual_gravity_path(self):
        for changes in ({"leader_only": True}, {"gravity_handoff": False}, {"human_trigger": "bad"}):
            with self.assertRaises(ValueError):
                replace(self.config, **changes)

    def test_hundred_mm_range_survives_old_nominal_and_measured_clamps(self):
        c = self.control
        self.assertEqual(c.upper[6], .1)
        self.assertEqual(c.gripper_handoff.open_width, .1)
        self.assertAlmostEqual(c.status()["gripper_trigger_width_m"], .09)
        self.assertEqual(c.status()["action_upper"][6], 100.)
        self.arm()
        self.leader[6] = .095
        for _ in range(100):
            proposal = self.follower.copy(); proposal[6] = .1
            self.tick(proposal)
        self.assertGreater(c.target[6], .07994)
        self.assertEqual(c.leader_target[6], .1)

    def test_circle_controller_converts_hundred_percent_to_hundred_mm(self):
        from unittest.mock import MagicMock
        from ruri.client.controllers.single_piper_assisted.controller import SinglePiperAssistedController
        from ruri.client.controllers.single_piper.normalization import normalize_pose
        for maximum in (None, .1):
            with self.subTest(gripper_max_m=maximum):
                robot = SinglePiperAssistedController(replace(self.config, gripper_max_m=maximum))
                self.assertEqual(robot.action_bounds[1][6], 100.)
                robot._io = MagicMock()
                robot._policy_epoch = 0
                for normalized, metres in ((0., 0.), (10., .01), (90., .09), (100., .1)):
                    action = normalize_pose(self.leader[:6], 0.)
                    action[6] = normalized
                    robot.send_action(action, reference_revision=0)
                    self.assertAlmostEqual(robot._io.send.call_args.kwargs["target"][6], metres)
                    state = robot._normalize_gripper(normalize_pose(self.leader[:6], metres))
                    self.assertAlmostEqual(state[6], normalized)

    def test_follower_source_tracks_target_generator_not_leader_alpha(self):
        self.enter_human()
        c = self.control
        # First interpolation tick preserves the old goal exactly.
        self.assertEqual(c.follower_handoff.progress, 0.)
        self.assertEqual(c.status()['follower_control_source'], 'policy')
        self.tick()
        self.assertEqual(c.status()['follower_control_source'], 'human')
        self.assertTrue(c.status()['follower_transition'])
        for _ in range(200):
            self.tick()
            if c.reference.phase == 'recover':
                break
        self.assertEqual(c.reference.phase, 'recover')
        self.assertEqual(c.status()['follower_control_source'], 'hold')
        self.tick()
        self.assertGreater(c.reference.alpha, .9)
        self.assertEqual(c.status()['follower_control_source'], 'policy')
        self.assertEqual(c.status()['follower_gripper_source'], 'human')


if __name__ == "__main__":
    unittest.main()
