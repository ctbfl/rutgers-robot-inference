"""Right-arm handoff mechanics; no CAN or camera access."""
from dataclasses import replace
import unittest

import numpy as np

from ruri.client.controllers.single_piper_assisted.config import SinglePiperAssistedConfig
from ruri.client.controllers.single_piper_assisted.control import AssistedControl
from ruri.client.controllers.single_piper_assisted.gravity_handoff import GravityHandoff


class GravityHandoffTests(unittest.TestCase):
    def reference(self, **kwargs):
        config = replace(SinglePiperAssistedConfig(leader_only=True), **kwargs)
        return GravityHandoff(config, np.zeros(6), -np.ones(6), np.ones(6))

    def human(self, ref):
        q = np.array([.2, 0., 0., 0., 0., 0.])
        for _ in range(200):
            ref.step(q, np.full(6, -.5), .01)
            if ref.phase == 'human':
                break
        self.assertEqual(ref.phase, 'human')
        self.assertEqual(ref.alpha, 1.)
        return q

    def test_entry_needs_large_per_joint_error_for_point_one_second(self):
        r = self.reference(interaction_filter_s=.001)
        for _ in range(9):
            r.step(np.full(6, .1), None, .01)
            self.assertEqual(r.alpha, 0.)
            self.assertFalse(r.intervening)
        r.step(np.full(6, .1), None, .01)
        self.assertTrue(r.intervening)
        self.assertEqual(r.alpha, 0.)
        for k in range(1, 51):
            # The position spring vanishes during the handoff: finish it even
            # when the error disappears instead of re-detecting a lost spring.
            r.step(r.r.copy(), None, .01)
            u = k/50
            self.assertAlmostEqual(r.alpha, u*u*(3-2*u))
        self.assertEqual(r.phase, 'human')

    def test_small_multijoint_errors_and_brief_spikes_do_not_trigger(self):
        r = self.reference()
        for _ in range(200):
            r.step(r.r+np.deg2rad(2.6), np.ones(6), .01)
            self.assertEqual(r.alpha, 0.)
        for _ in range(3):
            for _ in range(3):
                r.step(r.r+.12, None, .01)
                self.assertEqual(r.alpha, 0.)
            for _ in range(20):
                r.step(r.r.copy(), None, .01)
            self.assertEqual(r.confirm_elapsed, 0.)

    def test_policy_motion_with_small_lag_is_not_human_intervention(self):
        r = self.reference()
        for _ in range(300):
            r.step(r.r-.005, np.ones(6), .01)
            self.assertFalse(r.intervening)

    def test_handoff_keeps_reference_velocity_continuous_at_entry(self):
        r = self.reference(interaction_filter_s=.001)
        r.step(r.r, np.ones(6), .01)
        for _ in range(9):
            r.step(r.r+.2, np.ones(6), .01)
        velocity = r.velocity.copy()
        r.step(r.r+.2, -np.ones(6), .01)
        self.assertEqual(r.phase, 'blend')
        np.testing.assert_allclose(r.velocity, velocity)

    def test_human_has_no_admittance_and_shadows_large_hand_motion(self):
        r = self.reference(reference_speed=4., max_reference_speed=4., policy_reference_speed=.5)
        self.human(r)
        for k in range(100):
            q = np.array([.2+.003*k, 0., 0., 0., 0., 0.])
            actual, velocity = r.step(q, -np.ones(6), .01)
            self.assertEqual(r.alpha, 1.)
            self.assertEqual(r.phase, 'human')
            np.testing.assert_allclose(actual, q)
            np.testing.assert_array_equal(velocity, 0.)
            np.testing.assert_array_equal(r.human_velocity, 0.)

    def test_slow_cumulative_movement_and_tremor_prevent_release(self):
        for oscillatory in (False, True):
            r = self.reference()
            q = self.human(r)
            for k in range(200):
                q[0] = .2+(.002*np.sin(k*.4) if oscillatory else .0001*k)
                r.step(q, None, .01)
                self.assertEqual(r.phase, 'human')
                self.assertEqual(r.alpha, 1.)

    def test_quiet_half_second_then_smooth_half_second_recovery_from_current_pose(self):
        r = self.reference()
        q = self.human(r)
        # Deliberately move far away from the pre-intervention reference.
        q[0] = .7
        r.step(q, np.full(6, -.9), .01)
        for _ in range(49):
            r.step(q, np.full(6, -.9), .01)
            self.assertEqual(r.alpha, 1.)
            self.assertEqual(r.phase, 'human')
        r.step(q, np.full(6, -.9), .01)
        self.assertEqual(r.phase, 'recover')
        self.assertTrue(r.rebased)
        self.assertEqual(r.alpha, 1.)
        np.testing.assert_allclose(r.r, q)
        np.testing.assert_array_equal(r.velocity, 0.)
        np.testing.assert_array_equal(r.policy_velocity, 0.)
        for k in range(1, 51):
            before = r.r.copy()
            r.step(r.r.copy(), r.r+.1, .01)
            u = k/50
            self.assertAlmostEqual(r.alpha, 1-u*u*(3-2*u))
            self.assertLessEqual(max(abs(r.r-before)), .15*.01+1e-12)
        self.assertEqual(r.phase, 'policy')
        self.assertFalse(r.intervening)

    def test_stationary_hand_with_small_noise_counts_as_quiet(self):
        r = self.reference()
        q = self.human(r)
        for k in range(60):
            r.step(q+np.deg2rad(.02)*np.sin(k), None, .01)
        self.assertEqual(r.phase, 'recover')
        self.assertLess(r.alpha, 1.)

    def test_recovery_motion_alone_does_not_reenter_human(self):
        r = self.reference()
        q = self.human(r)
        for _ in range(60):
            r.step(q, None, .01)
        for _ in range(80):
            r.step(r.r.copy(), np.ones(6), .01)
        self.assertEqual(r.phase, 'policy')
        self.assertEqual(r.alpha, 0.)

    def test_new_deflection_during_recovery_requires_confirmation(self):
        r = self.reference(interaction_filter_s=.001)
        q = self.human(r)
        while r.phase != 'recover':
            r.step(q, None, .01)
        for _ in range(9):
            r.step(r.r+.15, None, .01)
            self.assertNotEqual(r.phase, 'blend')
        before = r.alpha
        r.step(r.r+.15, None, .01)
        self.assertEqual(r.phase, 'blend')
        self.assertEqual(r.alpha, before)

    def test_explicit_human_unloads_smoothly_and_does_not_auto_release(self):
        r = self.reference()
        for _ in range(300):
            before = r.alpha
            r.step(np.zeros(6), None, .01, manual=True)
            self.assertGreaterEqual(r.alpha, before)
            self.assertLessEqual(r.alpha-before, .031)
        self.assertEqual(r.phase, 'human')
        r.hold(np.full(6, .1))
        self.assertEqual(r.alpha, 0.)
        self.assertFalse(r.intervening)
        self.assertEqual(len(r._motion), 0)

    def test_config_quiet_range_validation(self):
        for values in ((0.,)*6, (float('nan'),)*6, (.1,)*5, (True,)*6):
            with self.assertRaises(ValueError):
                self.reference(human_quiet_range_deg=values)

    def test_worker_rebases_and_revokes_old_policy_at_release(self):
        now = 10.
        q = np.array([0., .4, -.4, 0., 0., 0., .04])
        control = AssistedControl(SinglePiperAssistedConfig(leader_only=True), q, q, clock=lambda: now)
        control.mode = 'POLICY'
        moved = q.copy(); moved[0] = .2
        for _ in range(160):
            now += .01
            control.last_heartbeat = control.last_command = now
            control.policy_reference = q.copy()
            control.policy_valid = True
            control.policy_observation_ns = int(now*1e9)
            previous = control.reference.phase
            revision = control.reference_revision
            control.tick(moved, moved, now)
            if previous == 'human' and control.reference.phase == 'recover':
                break
        self.assertEqual(control.reference.phase, 'recover')
        self.assertEqual(control.reference_revision, revision+1)
        self.assertFalse(control.policy_valid)
        self.assertEqual(control.reference.alpha, 1.)
        np.testing.assert_allclose(control.leader_target, moved)
        np.testing.assert_array_equal(control.reference_velocity, 0.)
        revision = control.reference_revision
        for _ in range(50):
            now += .01
            control.last_heartbeat = control.last_command = now
            control.policy_reference = control.target.copy()
            control.policy_reference[0] += .1
            control.policy_valid = True
            control.policy_observation_ns = int(now*1e9)
            control.tick(control.target, control.target, now)
        self.assertEqual(control.reference.phase, 'policy')
        self.assertTrue(control.policy_valid)
        self.assertEqual(control.reference_revision, revision)
        self.assertGreater(control.reference_velocity[0], 0.)


if __name__ == '__main__':
    unittest.main()
