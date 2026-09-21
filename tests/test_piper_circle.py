"""Offline circle FK/IK and actual-state replanning. No CAN or cameras."""
import importlib.util
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'))
SCIPY_AVAILABLE = importlib.util.find_spec('scipy') is not None
if SCIPY_AVAILABLE:
    from piper_circle_policy import CirclePolicy, PiperTCP, IKConvergenceError, RUNTIME_IK_TOLERANCE_M
    from client_single_piper_shared_circle import parse_args
from ruri.client.controllers.single_piper.normalization import normalize_pose, denormalize_action


@unittest.skipUnless(SCIPY_AVAILABLE, 'circle example requires the piper-circle extra')
class CirclePolicyTests(unittest.TestCase):
    def test_nominal_tcp_and_requested_circle_center(self):
        policy = CirclePolicy()
        np.testing.assert_allclose(policy.center-policy.home_tcp, [.05, 0., .15])
        np.testing.assert_allclose(policy.home_tcp, [.19142645, -.00000144, .22502035], atol=1e-7)
        a, up = policy.goal(policy.period/4)
        b, down = policy.goal(3*policy.period/4)
        np.testing.assert_allclose(a-policy.center, [0., 0., .10], atol=1e-12)
        np.testing.assert_allclose(b-policy.center, [0., 0., -.10], atol=1e-12)
        self.assertGreater(up[2], 0.)
        self.assertLess(down[2], 0.)

    def test_full_approach_and_loop_have_continuous_bounded_solutions(self):
        audit = CirclePolicy().validate_path()
        self.assertEqual(audit['samples'], 181)
        self.assertLess(audit['max_ik_error_mm'], .05)
        self.assertLess(audit['max_neighbor_joint_step_rad'], .04)
        self.assertLess(audit['loop_joint_closure_rad'], .001)
        self.assertLess(max(audit['loop_peak_joint_speed_rad_s']), .5)
        self.assertAlmostEqual(audit['angular_speed_rad_s'], .5)
        self.assertEqual(audit['policy_speed_limit_rad_s'], .5)
        self.assertEqual(set(audit['peak_gravity_legacy']), {'leader'})
        self.assertFalse(audit['collision_checked'])

    def test_geometry_jacobian_matches_finite_difference(self):
        kin = PiperTCP()
        q = np.array([.2, .5, -.6, .1, -.3, .2])
        p, _, origins, axes = kin.frames(q)
        jac = np.cross(axes, p-origins).T
        for i in range(6):
            dq = np.eye(6)[i]*1e-6
            diff = (kin.fk(q+dq)[0]-kin.fk(q-dq)[0])/(2e-6)
            np.testing.assert_allclose(diff, jac[:, i], atol=1e-7)

    def test_replan_from_dragged_observation_while_circle_phase_continues(self):
        clock = [0.]
        p = CirclePolicy(clock=lambda: clock[0])
        q = p.kin.ik(*[p.goal(0)[0], np.array([0., .5, -.4, 0., -.2, 0.]), p.goal(0)[1]])
        p.infer({'observation.state': normalize_pose(q, .04)})
        old_goal = np.asarray(p.last['goal_tcp'])
        q[0] += .15
        clock[0] = 5.
        result = p.infer({'observation.state': normalize_pose(q, .04)})
        current = p.kin.fk(q)[0]
        goal = p.goal(5.)[0]
        delta = goal-current
        expected = current+delta*min(1., p.tcp_speed*p.lookahead/np.linalg.norm(delta))
        actual_q, width = denormalize_action(result['action_chunk'][0])
        np.testing.assert_allclose(p.last['observed_tcp'], current, atol=1e-7)
        np.testing.assert_allclose(p.kin.fk(actual_q)[0], expected, atol=5e-5)
        self.assertFalse(np.allclose(goal, old_goal))
        self.assertAlmostEqual(width, .04)
        self.assertEqual(p.last['phase_time_s'], 5.)

    def test_observation_overshoot_is_preserved_but_action_stays_in_limits(self):
        p = CirclePolicy(clock=lambda: 0.)
        q = np.array([0., -.005, .004, 0., -.02, 0.])
        result = p.infer({'observation.state': normalize_pose(q, .04)})
        np.testing.assert_allclose(p.last['observed_tcp'], p.kin.fk(q)[0], atol=1e-7)
        target, _ = denormalize_action(result['action_chunk'][0])
        self.assertGreaterEqual(target[1], -1e-7)
        self.assertLessEqual(target[2], 1e-7)

    def test_dragged_pose_from_abort_log_is_accepted_at_runtime_precision(self):
        clock = [0.]
        p = CirclePolicy(clock=lambda: clock[0])
        clock[0] = 17.33367285109125
        q = np.array([-.5455899241734274, 1.7208073760038092, -.816569743838067,
                      -.043563418129778465, .5784544739884806, -.5720491156336616])
        result = p.infer({'observation.state': normalize_pose(q, .034)})
        self.assertEqual(p.last['ik_status'], 'ok')
        self.assertGreater(p.last['ik_error_mm'], .05)
        self.assertLess(p.last['ik_error_mm'], 1000*RUNTIME_IK_TOLERANCE_M)
        target, _ = denormalize_action(result['action_chunk'][0])
        np.testing.assert_allclose(p.kin.fk(target)[0], p.last['proposed_tcp'], atol=RUNTIME_IK_TOLERANCE_M)
        # The strict geometric audit must still reject this same residual.
        with self.assertRaises(IKConvergenceError):
            p.kin.ik(p.last['proposed_tcp'], q, p.preferred_axis(p.last['proposed_tcp'], clock[0]))

    def test_retry_shortens_cartesian_step_from_current_observation(self):
        p = CirclePolicy(clock=lambda: 0.)
        q = np.array([.2, .5, -.6, .1, -.3, .2])
        real_ik = p.kin.ik
        calls = []
        def solve(xyz, seed, axis, **kwargs):
            calls.append(np.array(xyz))
            if len(calls) == 1:
                raise IKConvergenceError('temporary local solve failure')
            return real_ik(xyz, seed, axis, **kwargs)
        with patch.object(p.kin, 'ik', side_effect=solve):
            p.infer({'observation.state': normalize_pose(q, .04)})
        self.assertEqual(p.last['ik_status'], 'reduced_step')
        self.assertEqual(p.last['ik_step_scale'], .5)
        actual = np.array(p.last['observed_tcp'])
        np.testing.assert_allclose(calls[1]-actual, .5*(calls[0]-actual))

    def test_unsolved_ik_proposes_current_pose_and_recovers_on_next_query(self):
        clock = [0.]
        p = CirclePolicy(clock=lambda: clock[0])
        q = np.array([.2, .5, -.6, .1, -.3, .2])
        observation = {'observation.state': normalize_pose(q, .04)}
        p.infer(observation)
        q[0] += .15
        observation = {'observation.state': normalize_pose(q, .04)}
        with patch.object(p.kin, 'ik', side_effect=IKConvergenceError('unreachable step')) as solve:
            result = p.infer(observation)
        self.assertEqual(solve.call_count, 4)
        self.assertEqual(p.last['ik_status'], 'waiting_ik')
        self.assertEqual(p.last['ik_failure'], 'unreachable step')
        np.testing.assert_allclose(result['action_chunk'][0], observation['observation.state'], atol=1e-5)
        old_goal = p.last['goal_tcp']
        clock[0] = 2.
        p.infer(observation)
        self.assertEqual(p.last['ik_status'], 'ok')
        self.assertFalse(np.allclose(p.last['goal_tcp'], old_goal))

    def test_invalid_observation_and_unexpected_errors_are_not_ik_fallbacks(self):
        p = CirclePolicy()
        with self.assertRaises(ValueError):
            p.infer({'observation.state': np.full(7, np.nan)})
        with patch.object(p.kin, 'ik', side_effect=RuntimeError('unexpected bug')):
            with self.assertRaisesRegex(RuntimeError, 'unexpected bug'):
                p.infer({'observation.state': normalize_pose(np.zeros(6), .04)})

    def test_ik_fallback_keeps_scheduler_running_during_human_control(self):
        from ruri.client.schedulers.assisted import AssistedScheduler
        p = CirclePolicy()
        observation = normalize_pose(np.array([.2, .5, -.6, .1, -.3, .2]), .04)
        state = {'mode': 'HOLD', 'epoch': 1, 'armed': False, 'reference_revision': 2,
                 'policy_reference': observation.copy(), 'control_schema': 'ruri_gravity_handoff_v1',
                 'human_exclusive': True, 'intervention_active': True, 'human_weight': 1.}
        robot = Mock(action_bounds=None, last_command_id=0)
        robot.status.side_effect = lambda: state.copy()
        scheduler = AssistedScheduler(robot, p, policy_version='circle-test', prompt='circle', execution_horizon=1)
        try:
            scheduler.resume()
            state['mode'] = 'POLICY'  # Automatic HUMAN retains the POLICY session.
            with patch.object(p.kin, 'ik', side_effect=IKConvergenceError('no solution')):
                deadline = time.monotonic()+2.
                count = 0
                while time.monotonic() < deadline and count < 3:
                    stamp = time.monotonic_ns()
                    scheduler.step({'observation_id': str(stamp), 'monotonic_ns': stamp,
                        'observation': {'observation.state': observation.copy()},
                        'telemetry': {'control': state.copy()}})
                    count = sum(x.kwargs.get('model_action', True) for x in robot.send_action.call_args_list)
                    time.sleep(.005)
                self.assertGreaterEqual(count, 3)
                self.assertEqual(p.last['ik_status'], 'waiting_ik')
                robot.request_mode.assert_not_called()  # No involuntary HOLD.
                for call in robot.send_action.call_args_list:
                    if call.kwargs.get('model_action', True):
                        np.testing.assert_allclose(call.args[0], observation, atol=1e-5)
        finally:
            scheduler.close()

    def test_bad_parameters_unreachable_targets_and_no_implicit_hardware(self):
        args = parse_args([])
        self.assertFalse(args.execute or args.simulate)
        self.assertEqual(args.radius, .10)
        self.assertEqual(args.reference_speed, 4.)
        self.assertEqual(args.policy_speed, .5)
        self.assertEqual(args.quiet_angle, .1)
        self.assertAlmostEqual(2*np.pi/args.period, .5)
        with self.assertRaises(ValueError):
            CirclePolicy(radius=-.01)
        with self.assertRaises(ValueError):
            PiperTCP().ik([5., 0., 5.], np.zeros(6))
        with self.assertRaises(ValueError):
            CirclePolicy(period=1.).validate_path()
        with patch('client_single_piper_shared_circle.SinglePiperAssistedController') as robot:
            from client_single_piper_shared_circle import main
            with patch('builtins.print'):
                main([])
            robot.assert_not_called()


if __name__ == '__main__':
    unittest.main()
