"""Two actuator streams through takeover, live handover, release and re-entry."""
from dataclasses import replace
import unittest

import numpy as np

from ruri.client.controllers.single_piper_assisted.config import SinglePiperAssistedConfig
from ruri.client.controllers.single_piper_assisted.control import AssistedControl
from ruri.client.controllers.single_piper_assisted.follower_handoff import FollowerHandoff


class DualGravityHandoffTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.config = SinglePiperAssistedConfig(gravity_handoff=True,
            reference_speed=4., max_reference_speed=4., policy_reference_speed=.5)
        self.q = np.array([0., .4, -.4, 0., 0., 0., .04])
        self.control = AssistedControl(self.config, self.q, self.q, clock=lambda: self.now)
        self.control.mode = 'POLICY'

    def tick(self, q=None, policy=None):
        c = self.control
        old = c.target.copy()
        self.now += .01
        c.last_command = c.last_heartbeat = self.now
        c.policy_reference = old.copy() if policy is None else policy.copy()
        c.policy_observation_ns = int(self.now*1e9)
        c.policy_valid = True
        c.tick(old if q is None else q, old, self.now)
        self.assertLessEqual(max(abs(c.target[:6]-old[:6])), self.config.reference_speed*.01+1e-10)
        return c

    def enter_human(self):
        moved = self.q.copy(); moved[0] += .3
        for _ in range(200):
            old = self.control.target.copy()
            c = self.tick(moved)
            if c.reference.phase == 'human':
                np.testing.assert_allclose(c.target, old)
                return moved
            np.testing.assert_array_equal(c.target, c.leader_target)
            np.testing.assert_array_equal(c.reference_velocity, c.follower_velocity)
        self.fail('no human takeover')

    def test_policy_and_unloading_share_reference_then_left_starts_without_jump(self):
        proposal = self.q.copy(); proposal[0] = .5
        for _ in range(20):
            c = self.tick(policy=proposal)
            np.testing.assert_array_equal(c.target, c.leader_target)
        q = self.enter_human()
        c = self.control
        self.assertEqual(c.reference.alpha, 1.)
        self.assertGreater(abs(c.target[0]-q[0]), .1)
        np.testing.assert_allclose(c.leader_target, q)
        np.testing.assert_array_equal(c.follower_velocity, 0.)

    def test_half_second_handover_uses_live_leader_and_consistent_velocity(self):
        q = self.enter_human()
        start = self.control.target[:6].copy()
        for k in range(1, 51):
            q[0] += .002  # Move during interpolation; do not chase an old snapshot.
            before = self.control.target[:6].copy()
            c = self.tick(q)
            self.assertEqual(c.reference.phase, 'human')
            u = k/50.
            s = u**3*(10-15*u+6*u*u)
            np.testing.assert_allclose(c.target[:6], (1-s)*start+s*q[:6], atol=1e-12)
            np.testing.assert_allclose(c.follower_velocity, (c.target[:6]-before)/.01, atol=1e-10)
            np.testing.assert_array_equal(c.reference_velocity, 0.)
        np.testing.assert_allclose(c.target, q)
        self.assertFalse(c.follower_handoff.limited)
        q[0] += .002
        c = self.tick(q)
        np.testing.assert_allclose(c.target, q)

    def test_quiet_release_reuses_last_sent_left_goal_and_requeries_policy(self):
        q = self.enter_human()
        for _ in range(55):
            q[0] += .002
            self.tick(q)
        q[0] += .0001
        for _ in range(70):
            old = self.control.target.copy()
            revision = self.control.reference_revision
            c = self.tick(q)
            if c.reference.phase == 'recover':
                break
        self.assertEqual(c.reference.phase, 'recover')
        self.assertEqual(c.reference.alpha, 1.)
        self.assertFalse(c.policy_valid)
        self.assertEqual(c.reference_revision, revision+1)
        np.testing.assert_allclose(c.target, old)
        np.testing.assert_allclose(c.leader_target, c.target)
        revision = c.reference_revision
        alphas = []
        for _ in range(50):
            proposal = c.target.copy(); proposal[0] += .05
            c = self.tick(policy=proposal)
            alphas.append(c.reference.alpha)
            np.testing.assert_array_equal(c.target, c.leader_target)
            np.testing.assert_array_equal(c.reference_velocity, c.follower_velocity)
        self.assertTrue(any(0. < a < 1. for a in alphas))
        self.assertEqual(c.reference.phase, 'policy')
        self.assertEqual(c.reference_revision, revision)

    def test_large_gap_keeps_gravity_until_rate_limited_follower_catches_up(self):
        self.config = replace(self.config, reference_speed=.15, policy_reference_speed=.15)
        self.control = AssistedControl(self.config, self.q, self.q, clock=lambda: self.now)
        self.control.mode = 'POLICY'
        q = self.enter_human()
        for _ in range(100):
            c = self.tick(q)
        self.assertEqual(c.reference.phase, 'human')
        self.assertTrue(c.follower_handoff.limited)
        self.assertEqual(c.reference.alpha, 1.)
        for _ in range(180):
            c = self.tick(q)
            if c.reference.phase == 'recover':
                break
        self.assertEqual(c.reference.phase, 'recover')
        np.testing.assert_allclose(c.target, q, atol=np.deg2rad(.1))

    def test_reintervention_and_hold_reset_follower_handover(self):
        q = self.enter_human()
        for _ in range(80):
            c = self.tick(q)
        self.assertEqual(c.reference.phase, 'recover')
        q[0] += .3
        for _ in range(100):
            c = self.tick(q)
            if c.reference.phase == 'human':
                break
            np.testing.assert_allclose(c.target, c.leader_target)
        self.assertEqual(c.follower_handoff.progress, 0.)
        self.assertEqual(c.intervention_id, 1)  # Same uninterrupted intervention.
        c._revoke('HOLD', 'test')
        self.assertFalse(c.follower_handoff.active)
        np.testing.assert_array_equal(c.follower_velocity, 0.)

    def test_default_confirm_is_point_one_seconds(self):
        self.assertEqual(self.config.human_confirm_s, .1)
        moved = self.q.copy(); moved[0] = .3
        for _ in range(5):
            self.assertEqual(self.tick(moved).reference.phase, 'policy')
        for _ in range(20):
            c = self.tick(moved)
        self.assertEqual(c.reference.phase, 'blend')

    def test_interpolation_projects_to_common_bounds_without_overshoot(self):
        h = FollowerHandoff(self.config, np.zeros(6), -np.ones(6), np.ones(6))
        for _ in range(60):
            r, v = h.follow(np.full(6, 10.), .01)
            self.assertTrue(np.all((r >= 0.) & (r <= 1.)))
            self.assertTrue(np.all(abs(v) <= 4.))
        np.testing.assert_allclose(r, 1.)


if __name__ == '__main__':
    unittest.main()
