"""Numerical control law checks; no transport, hardware, or camera dependencies."""
import math
import copy
import unittest
from dataclasses import replace
import numpy as np
from ruri.client.controllers.single_piper_assisted.config import SinglePiperAssistedConfig
from ruri.client.controllers.single_piper_assisted.reference import SharedReference
from ruri.client.controllers.single_piper_assisted.control import synchronized_bounds, physical_bounds
from ruri.client.controllers.single_piper.calibration_ranges import NOMINAL, measured_for_arm


class SharedReferenceTests(unittest.TestCase):
    def reference(self, **kwargs):
        c = replace(SinglePiperAssistedConfig(leader_only=True, human_confirm_s=.5), **kwargs)
        return SharedReference(c, np.zeros(6), np.full(6, -1.), np.ones(6))

    def test_exact_one_step_equations_and_old_reference_estimator(self):
        r = self.reference()
        q = np.array([.1, -.05, 0., 0., 0., 0.])
        target = np.array([-.1, .01, 0., 0., 0., 0.])
        interaction = 3*q
        effective = np.array([.22, -.07, 0., 0., 0., 0.])
        alpha = 0.  # First sample cannot satisfy the sustained entry gate.
        vh = effective/2.
        vp = np.array([-.15, .02, 0., 0., 0., 0.])
        v = (1-alpha)*vp+alpha*vh
        actual, velocity = r.step(q, target, .01)
        np.testing.assert_allclose(r.interaction, interaction)
        np.testing.assert_allclose(r.effective, effective)
        self.assertAlmostEqual(r.alpha, alpha)
        np.testing.assert_allclose(velocity, v)
        np.testing.assert_allclose(actual, v*.01)

    def test_bias_deadzone_and_no_fictitious_human_motion(self):
        r = self.reference(interaction_bias=(.3,)*6)
        r.step(np.full(6, .1), None, .01)
        np.testing.assert_allclose(r.interaction, 0., atol=1e-15)
        np.testing.assert_array_equal(r.velocity, np.zeros(6))
        self.assertEqual(r.alpha, 0.)
        r = self.reference()
        r.step(np.full(6, .08/3), None, .01)
        np.testing.assert_allclose(r.effective, 0., atol=1e-15)

    def test_confirmation_then_separate_half_second_rise(self):
        r = self.reference(interaction_filter_s=.001)
        for _ in range(49):
            r.step(r.r+.1, None, .01)
            self.assertEqual(r.alpha, 0.)
            self.assertFalse(r.intervening)
        r.step(r.r+.1, None, .01)
        self.assertTrue(r.intervening)
        self.assertEqual(r.alpha, 0.)
        for k in range(1, 51):
            r.step(r.r+.06, None, .01)  # Lower maintenance threshold after confirmation.
            self.assertAlmostEqual(r.alpha, k/50.)
        self.assertTrue(r.human_latched)

    def test_rate_and_joint_projection_use_actually_applied_velocity(self):
        r = self.reference(admittance_damping=(.01,)*6)
        r.r[:] = .9999
        before = r.r.copy()
        r.step(np.full(6, 10.), np.full(6, 100.), .01, manual=True)
        np.testing.assert_allclose(r.r, 1.)
        np.testing.assert_allclose(r.velocity, (r.r-before)/.01)
        self.assertLessEqual(np.max(abs(r.velocity)), .15)
        r.step(np.full(6, 10.), None, .01, manual=True)
        np.testing.assert_allclose(r.velocity, 0.)
        self.assertLessEqual(r.alpha, 1.)

    def test_policy_stays_at_half_rad_while_human_can_use_four(self):
        r = self.reference(reference_speed=4., max_reference_speed=4.,
                           policy_reference_speed=.5, admittance_damping=(.05,)*6)
        r.step(np.zeros(6), np.full(6, 10.), .01)
        self.assertEqual(r.alpha, 0.)
        np.testing.assert_allclose(r.policy_velocity, .5)
        np.testing.assert_allclose(r.velocity, .5)
        r.alpha = 1.
        before = r.r.copy()
        r.step(before+.2, np.full(6, -10.), .01, manual=True)
        np.testing.assert_allclose(r.policy_velocity, -.5)
        np.testing.assert_allclose(r.velocity, 4.)
        np.testing.assert_allclose(r.r-before, .04)
        r.step(r.r-.2, np.full(6, 10.), .01, manual=True)
        np.testing.assert_allclose(r.velocity, -4.)

    def test_split_speed_limits_reject_inconsistent_configuration(self):
        with self.assertRaisesRegex(ValueError, "max_reference_speed"):
            self.reference(reference_speed=4., policy_reference_speed=.5)
        with self.assertRaisesRegex(ValueError, "policy_reference_speed"):
            self.reference(reference_speed=.5, policy_reference_speed=1.)

    def test_fast_admittance_reaches_four_and_settles_without_overshooting_hand(self):
        r = self.reference(reference_speed=4., max_reference_speed=4.,
                           policy_reference_speed=.5, admittance_damping=(.1,)*6)
        # With full human authority, 0.16 rad of deflection gives (3*.16-.08)/.1 = 4.
        q = np.array([.16, -.16, 0., 0., 0., 0.])
        r.alpha = 1.
        r.step(q, np.zeros(6), .01, manual=True)
        np.testing.assert_allclose(r.velocity, [4., -4., 0., 0., 0., 0.])
        # A stationary constrained hand: varying worker dt up to its 20 ms cap.
        # The reference should slow toward the hand, never force 4 rad/s near it.
        previous_error = np.abs(q-r.r)
        for dt in [.01, .02, .005]*40:
            r.step(q, None, dt, manual=True)
            error = np.abs(q-r.r)
            self.assertTrue(np.all(error <= previous_error+1e-12))
            self.assertGreaterEqual(r.r[0], 0.)
            self.assertLessEqual(r.r[0], q[0])
            self.assertLessEqual(r.r[1], 0.)
            self.assertGreaterEqual(r.r[1], q[1])
            self.assertLessEqual(np.max(np.abs(r.velocity)), 4.+1e-12)
            previous_error = error
        self.assertLess(np.max(np.abs(r.velocity)), .001)
        self.assertLessEqual(np.max(np.abs(q-r.r)), .08/3+1e-6)

    def test_fast_admittance_follows_sustained_drag_with_less_spring_deflection(self):
        errors = []
        for damping in (2., .1):
            r = self.reference(reference_speed=4., max_reference_speed=4.,
                               policy_reference_speed=.5, admittance_damping=(damping,)*6)
            # Synthetic 0.3 rad/s human motion, automatic authority, stationary policy.
            for k in range(1, 151):
                q = np.array([k*.01*.3, 0., 0., 0., 0., 0.])
                r.step(q, np.zeros(6), .01)
            errors.append(float(q[0]-r.r[0]))
        self.assertLess(errors[1], errors[0]/2)

    def confirm_human(self, reference, steps=110):
        for _ in range(steps):
            q = reference.r.copy()
            q[1] += .1  # Above 5 degrees, including the entry filter settling time.
            reference.step(q, np.full(6, -1.), .01)

    def test_default_filter_and_sustained_interaction_latch_full_authority(self):
        r = self.reference(admittance_damping=(.1,)*6)
        self.confirm_human(r, 100)
        self.assertFalse(r.human_latched)
        self.assertLess(r.alpha, 1.)
        self.confirm_human(r, 10)
        self.assertTrue(r.human_latched)
        self.assertEqual(r.alpha, 1.)
        self.assertEqual(r.phase, 'human')

    def test_opposing_policies_have_identical_output_during_exclusive_human_control(self):
        r = self.reference(admittance_damping=(.1,)*6)
        self.confirm_human(r)
        a, b = copy.deepcopy(r), copy.deepcopy(r)
        for k in range(60):
            q = r.r.copy()
            q[1] += .05+k*.001
            qa, va = a.step(q, np.ones(6), .01)
            qb, vb = b.step(q, -np.ones(6), .01)
            self.assertEqual(a.alpha, 1.)
            self.assertEqual(b.alpha, 1.)
            np.testing.assert_allclose(qa, qb)
            np.testing.assert_allclose(va, vb)

    def test_half_second_hold_then_half_second_fade_and_fresh_policy(self):
        r = self.reference()
        self.confirm_human(r)
        q = r.r.copy()
        for _ in range(100):
            r.step(q, None, .01)
            if r.quiet_elapsed > 0:
                break
        self.assertAlmostEqual(r.quiet_elapsed, .01)
        for k in range(2, 101):
            # An old opposing proposal is presented exactly when policy first
            # regains weight. It must not execute on that transition tick.
            target = np.ones(6) if k == 51 else None
            r.step(q, target, .01)
            expected = 1. if k <= 50 else (100-k)/50.
            self.assertAlmostEqual(r.alpha, expected)
            self.assertEqual(r.human_latched, k <= 50)
            if k == 51:
                np.testing.assert_allclose(r.policy_velocity, 0.)
        self.assertFalse(r.intervening)
        self.assertEqual(r.phase, 'policy')
        self.assertEqual(r.alpha, 0.)

    def test_policy_motion_during_fade_does_not_restart_human_timer(self):
        r = self.reference()
        self.confirm_human(r)
        for _ in range(200):
            r.step(r.r.copy(), None, .01)
            if r.phase == 'recover':
                break
        self.assertEqual(r.phase, 'recover')
        for _ in range(60):
            r.step(r.r.copy(), np.ones(6), .01)
        self.assertEqual(r.phase, 'policy')
        self.assertEqual(r.alpha, 0.)

    def test_renewed_interaction_must_confirm_again_before_continuous_rise(self):
        r = self.reference(interaction_filter_s=.001)
        self.confirm_human(r)
        for _ in range(200):
            r.step(r.r.copy(), None, .01)
            if r.phase == 'recover' and r.alpha <= .5:
                break
        self.assertAlmostEqual(r.alpha, .5)
        for _ in range(49):
            before = r.alpha
            r.step(r.r+.1, None, .01)
            self.assertLessEqual(r.alpha, before)
        r.step(r.r+.1, None, .01)
        self.assertEqual(r.phase, 'blend')
        start = r.alpha
        for k in range(1, 51):
            q = r.r.copy(); q[1] += .1
            r.step(q, None, .01)
            self.assertAlmostEqual(r.alpha, start+(1-start)*k/50)
            self.assertEqual(r.quiet_elapsed, 0.)
        self.assertTrue(r.human_latched)

    def test_motion_without_spring_evidence_cannot_keep_human_authority(self):
        r = self.reference()
        self.confirm_human(r)
        for k in range(150):
            # All joints physically move while staying inside the spring deadzone.
            r.step(r.r+.01*np.sin(k*.1), np.ones(6), .01)
        self.assertFalse(r.intervening)
        self.assertEqual(r.alpha, 0.)

    def test_missing_evidence_freezes_partial_weight_then_fades_without_jump(self):
        r = self.reference(interaction_filter_s=.001)
        self.confirm_human(r, 70)
        self.assertAlmostEqual(r.alpha, .4)
        for _ in range(50):
            r.step(r.r.copy(), None, .01)
            self.assertAlmostEqual(r.alpha, .4)
        for k in range(1, 51):
            r.step(r.r.copy(), None, .01)
            self.assertAlmostEqual(r.alpha, .4*(1-k/50))
        self.assertFalse(r.intervening)
        self.confirm_human(r)
        r.hold()
        self.assertFalse(r.intervening)
        self.assertFalse(r.human_latched)
        self.assertEqual(r.confirm_elapsed, 0.)
        self.assertEqual(r.quiet_elapsed, 0.)
        self.assertEqual(r.alpha, 0.)
        np.testing.assert_array_equal(r.filtered_error, 0.)

    def test_brief_spikes_and_interrupted_confirmation_never_accumulate(self):
        r = self.reference()
        for _ in range(5):
            for _ in range(20):
                r.step(r.r+.12, None, .01)
                self.assertEqual(r.alpha, 0.)
            for _ in range(20):
                r.step(r.r.copy(), None, .01)
                self.assertEqual(r.alpha, 0.)
            self.assertEqual(r.confirm_elapsed, 0.)
        self.assertFalse(r.intervening)

    def test_no_contact_error_envelope_and_multiple_small_errors_do_not_trigger(self):
        r = self.reference(admittance_damping=(.1,)*6, reference_speed=4.,
                           max_reference_speed=4., policy_reference_speed=.5)
        # Exceeds the recorded 2.35-degree peak; vector norm also exceeds 5 deg.
        for k in range(500):
            q = r.r + np.deg2rad(2.6)*np.cos(k*.01)
            r.step(q, np.ones(6), .01)
            self.assertEqual(r.alpha, 0.)
            self.assertEqual(r.confirm_elapsed, 0.)
        self.assertFalse(r.intervening)

    def test_short_single_sample_spike_is_filtered_below_entry_gate(self):
        r = self.reference()
        r.step(np.deg2rad(np.full(6, 15.)), None, .01)
        self.assertLess(np.max(abs(r.filtered_error)), np.deg2rad(5.))
        self.assertEqual(r.confirm_elapsed, 0.)

    def test_policy_motion_alone_cannot_start_confirmation(self):
        r = self.reference()
        for _ in range(200):
            r.step(r.r.copy(), np.ones(6), .01)
            self.assertEqual(r.confirm_elapsed, 0.)
            self.assertFalse(r.human_latched)
            self.assertEqual(r.alpha, 0.)

    def test_hold_clears_state_and_invalid_inputs_are_rejected(self):
        r = self.reference()
        r.step(np.ones(6), None, .01)
        r.hold()
        self.assertEqual(r.alpha, 0.)
        np.testing.assert_allclose(r.velocity, 0.)
        for dt in (-.01, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                r.step(np.zeros(6), None, dt)
        with self.assertRaises(ValueError):
            r.step(np.zeros(7), None, .01)
        with self.assertRaises(ValueError):
            self.reference(admittance_damping=(0.,)*6)
        with self.assertRaises(ValueError):
            self.reference(human_release_delay_s=1., human_release_s=1.)
        for values in ({'human_confirm_s': 0.}, {'interaction_filter_s': 0.},
                       {'human_trigger_angle_deg': float('nan')},
                       {'human_trigger_angle_deg': 1.}):
            with self.assertRaises(ValueError):
                self.reference(**values)

    def test_right_only_bounds_are_not_restricted_by_absent_left_arm(self):
        actual = synchronized_bounds(SinglePiperAssistedConfig(leader_only=True))
        nlo, nhi = physical_bounds(NOMINAL)
        rlo, rhi = physical_bounds(measured_for_arm('right_arm'))
        np.testing.assert_allclose(actual[0], np.maximum(nlo, rlo))
        np.testing.assert_allclose(actual[1], np.minimum(nhi, rhi))

    def test_toy_inertial_plant_releases_back_to_policy_with_fixed_gains(self):
        # A smoke check for this explicit ideal plant, not a hardware stability proof.
        r = self.reference()
        q, qd = np.zeros(6), np.zeros(6)
        peak_alpha = 0.
        dt = .002
        for k in range(5000):
            target = np.array([.1, 0., 0., 0., 0., 0.])
            pos, vel = r.step(q, target, dt)
            external = np.array([.7 if 2 < k*dt < 4 else 0., 0., 0., 0., 0., 0.])
            qdd = (3*(pos-q)+.5*(vel-qd)+external-.1*qd)/.3
            qd += qdd*dt
            q += qd*dt
            peak_alpha = max(peak_alpha, r.alpha)
            self.assertLessEqual(np.max(abs(r.velocity)), .150001)
        self.assertGreater(peak_alpha, .5)
        self.assertLess(r.alpha, .01)
        self.assertLess(abs(q[0]-.1), .015)


if __name__ == '__main__':
    unittest.main()
