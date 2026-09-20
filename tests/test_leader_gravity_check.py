from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ruri.client.controllers.single_piper.calibration_ranges import measured_for_arm
from ruri.client.controllers.single_piper.mit import leader_gravity_check as gc
from ruri.client.controllers.single_piper.mit import leader_follower as mit


class Clock:
    now = 0.0
    def perf_counter(self):
        return self.now
    def time(self):
        return 1000.0+self.now
    def sleep(self, seconds):
        self.now += seconds


class Arm:
    def __init__(self, clock):
        self.clock = clock
        self.q = np.array([.00279, 0., -.00489, 0., 0., 0.])
        self.qd = np.zeros(6)
        self.frames = []
        self.enabled = True
        self.arm_name = "right_arm"
        bounds = measured_for_arm(self.arm_name)
        self._lower = np.deg2rad([bounds[f"joint{j}"][0]/1000 for j in range(1, 7)])
        self._upper = np.deg2rad([bounds[f"joint{j}"][1]/1000 for j in range(1, 7)])
        self.status = SimpleNamespace(ctrl_mode="CAN_CTRL", mode_feedback="MOVE_MIT", arm_status="NORMAL", err_code=0)
        self.firmware_info = {"software_version": "S-V1.9-1"}
        self.set_follower_mode = MagicMock()
        self.set_auto_set_motion_mode_enabled = MagicMock()
        self.disconnect = MagicMock()
        self.stale_joint = None
    def get_joint_angles(self):
        return SimpleNamespace(timestamp=self.clock.time(), msg=self.q.tolist())
    def get_motor_states(self, joint):
        stamp = self.clock.time() - (1. if joint == self.stale_joint else 0.)
        return SimpleNamespace(timestamp=stamp, msg=SimpleNamespace(velocity=self.qd[joint-1], current=0., torque=0.))
    def get_arm_status(self):
        return SimpleNamespace(timestamp=self.clock.time(), msg=self.status)
    def get_joints_enable_status_list(self):
        return [self.enabled]*6
    def enable(self):
        self.enabled = True
    def move_mit(self, **frame):
        self.frames.append(frame)


class GravityTrialTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.arm = Arm(self.clock)
        self.gravity = mit.GravityCompensator(gc.CALIBRATION)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(gc.time, "time", self.clock.time))
        self.stack.enter_context(patch.object(gc.time, "perf_counter", self.clock.perf_counter))
        self.stack.enter_context(patch.object(gc.time, "sleep", self.clock.sleep))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch("socket.socket", side_effect=AssertionError("no physical sockets")))

    def test_default_is_offline_and_uses_leader_calibration(self):
        with patch.object(gc, "discover_piper_can") as discover, patch.object(mit, "build_arm") as build:
            self.assertEqual(gc.main([]), 0)
        discover.assert_not_called()
        build.assert_not_called()
        self.assertEqual(self.gravity.path.name, "calibration_leader.json")
        self.assertEqual(self.gravity.urdf.name, "piper_no_gripper_description.urdf")

    def test_changed_torque_coefficients_fail_before_hardware(self):
        import pyAgxArm
        config = pyAgxArm.create_agx_arm_config(robot="piper", firmeware_version="v189")
        config["joint_torque_c"] = [1., 1., 1., .813252, .813252, .813252]
        with patch.object(pyAgxArm, "create_agx_arm_config", return_value=config):
            with self.assertRaisesRegex(mit.TeleopAbort, "calibration migration"):
                gc.main([])

    def test_release_ramp_then_actual_leader_law_and_no_gripper(self):
        rows = []
        gc.trial(self.arm, self.gravity, 1., threading.Event(), rows.append)
        self.assertEqual({r['phase'] for r in rows}, {"position_hold", "release", "gravity_only"})
        first, last = self.arm.frames[:6], self.arm.frames[-6:]
        self.assertTrue(all(f['kp']==10. and f['kd']==.8 for f in first))
        self.assertTrue(all(f['kp']==0. and f['kd']==0. for f in last))
        for row in rows:
            self.assertEqual(len(row['current_a']), 6)
            self.assertEqual(len(row['effort_legacy']), 6)
        self.assertTrue(any(0. < f['kp'] < 10. for f in self.arm.frames))

    def test_final_hold_captures_current_pose_without_home(self):
        self.arm.q = np.array([.1, .2, -.3, .15, .1, -.1])
        hold = gc.final_position_hold(self.arm, self.gravity)
        np.testing.assert_array_equal(hold, self.arm.q)
        self.assertTrue(all(f['kp']==10. and f['kd']==.8 for f in self.arm.frames))
        for frame in self.arm.frames:
            self.assertEqual(frame['p_des'], self.arm.q[frame['joint_index']-1])
        self.arm.set_follower_mode.assert_not_called()

    def test_one_stale_motor_aborts_even_when_other_feedback_is_fresh(self):
        self.arm.stale_joint = 3
        with self.assertRaisesRegex(mit.TeleopAbort, "Stale"):
            gc.fresh_sample(self.arm)
        with self.assertRaisesRegex(mit.TeleopAbort, "Stale"):
            gc.final_position_hold(self.arm, self.gravity)
        self.assertEqual(self.arm.frames, [])

    def test_error_code_rejected_even_with_normal_status_text(self):
        self.arm.status.err_code = 1
        with self.assertRaisesRegex(mit.TeleopAbort, "error code"):
            gc.fresh_sample(self.arm)

    def test_initial_motion_rejected_without_commands(self):
        self.arm.qd[0] = .031
        with self.assertRaisesRegex(mit.TeleopAbort, "stationary"):
            gc.trial(self.arm, self.gravity, 1., threading.Event(), lambda r: None)
        self.assertEqual(self.arm.frames, [])

    def test_manual_drag_beyond_twenty_degrees_remains_active_within_arm_limits(self):
        def move_hand(row):
            if row['phase'] == "gravity_only":
                self.arm.q[2] = -.6
        gc.trial(self.arm, self.gravity, 1., threading.Event(), move_hand)
        self.assertEqual(self.arm.frames[-4]['p_des'], -.6)
        self.assertTrue(all(f['kp']==0. for f in self.arm.frames[-6:]))

    def test_current_stationary_pose_can_start_without_homing(self):
        self.arm.q = np.array([0., .0947, -.356, 0., .028, -.0104])
        initial = self.arm.q.copy()
        gc.trial(self.arm, self.gravity, 1., threading.Event(), lambda r: None)
        np.testing.assert_array_equal([f['p_des'] for f in self.arm.frames[:6]], initial)

    def test_requested_duration_starts_at_actual_gravity_phase(self):
        free_times = []
        def record(row):
            if row['phase'] == 'gravity_only':
                free_times.append(self.clock.now)
        gc.trial(self.arm, self.gravity, 1., threading.Event(), record)
        self.assertGreaterEqual(free_times[0], 4.)
        self.assertGreaterEqual(self.clock.now-free_times[0], 1.)
        self.assertLess(self.clock.now-free_times[0], 1.02)

    def test_measured_joint_limit_still_rejects(self):
        self.arm.q[2] = self.arm._lower[2]-.001
        with self.assertRaisesRegex(mit.TeleopAbort, "measured limits"):
            gc.fresh_sample(self.arm)
        self.assertEqual(self.arm.frames, [])

    def test_enter_gate_happens_before_hardware_access(self):
        with patch.object(gc.sys.stdin, 'isatty', return_value=True), \
             patch('builtins.input', side_effect=KeyboardInterrupt), \
             patch.object(gc, 'discover_piper_can') as discover, \
             patch.object(mit, 'build_arm') as build:
            self.assertEqual(gc.main(['--execute', '--wait-for-enter']), 130)
        discover.assert_not_called()
        build.assert_not_called()

    def test_torque_ceiling_rejected_before_any_frame(self):
        sample = gc.fresh_sample(self.arm)
        huge = SimpleNamespace(torque=lambda q: np.array([0., 13., 0., 0., 0., 0.]))
        with self.assertRaises(mit.TeleopAbort):
            gc.send_release(self.arm, sample, huge, sample.q, 1.)
        self.assertEqual(self.arm.frames, [])

    def test_cancelled_trial_does_not_release_position_hold(self):
        stop = threading.Event()
        stop.set()
        gc.trial(self.arm, self.gravity, 1., stop, lambda r: None)
        self.assertEqual(self.arm.frames, [])

    def mocked_execute(self, folder):
        with patch.object(gc, 'hardware_owner', return_value=nullcontext()), \
             patch.object(gc, 'discover_piper_can', return_value=SimpleNamespace(interface='offline-right')) as discover, \
             patch.object(mit, 'build_arm', return_value=self.arm):
            code = gc.main(['--execute', '--seconds', '1', '--log', str(Path(folder)/'trial.jsonl')])
        self.assertFalse(discover.call_args.kwargs['configure'])
        self.assertEqual(discover.call_args.kwargs['hardware_id'], 'usb:1d50:606f:002B00464759530920353131')
        rows = [json.loads(line) for line in (Path(folder)/'trial.jsonl').read_text().splitlines()]
        return code, rows

    def test_complete_execute_with_real_handshake_and_final_hold_in_simulation(self):
        with tempfile.TemporaryDirectory() as folder:
            code, rows = self.mocked_execute(folder)
        self.assertEqual(code, 0, rows[-1])
        self.assertEqual(rows[0]['firmware']['software_version'], 'S-V1.9-1')
        self.assertEqual(rows[-1]['outcome'], 'completed')
        self.assertIsNotNone(rows[-1]['final_hold_q'])
        self.assertTrue(all(f['kp']==10. for f in self.arm.frames[-6:]))
        self.arm.disconnect.assert_called_once()

    def test_wrong_arm_fails_before_mode_or_enable(self):
        self.arm.arm_name = 'left_arm'
        with tempfile.TemporaryDirectory() as folder:
            code, rows = self.mocked_execute(folder)
        self.assertEqual(code, 1)
        self.assertIn('non-right', rows[-1]['outcome'])
        self.assertEqual(self.arm.frames, [])
        self.arm.set_follower_mode.assert_not_called()

    def test_trial_abort_restores_fresh_position_hold(self):
        def abort_trial(*args):
            self.arm.q[0] = .2
            raise mit.TeleopAbort('operator trial stopped')
        with tempfile.TemporaryDirectory() as folder, patch.object(gc, 'trial', side_effect=abort_trial):
            code, rows = self.mocked_execute(folder)
        self.assertEqual(code, 1)
        self.assertIn('operator trial stopped', rows[-1]['outcome'])
        self.assertEqual(rows[-1]['final_hold_q'][0], .2)
        self.assertEqual(self.arm.frames[-6]['p_des'], .2)
        self.assertTrue(all(f['kp']==10. for f in self.arm.frames[-6:]))


if __name__ == '__main__':
    unittest.main()
