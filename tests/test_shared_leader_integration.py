"""Real single-leader worker/UDP/scheduler in simulation, with slow single-row policy."""
import socket
import time
import unittest
import numpy as np
from ruri.client.controllers.single_piper_assisted import SinglePiperAssistedController
from ruri.client.controllers.single_piper.normalization import normalize_pose
from ruri.client.schedulers.assisted import AssistedScheduler


def address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return f'udp://127.0.0.1:{sock.getsockname()[1]}'


class SharedLeaderIntegrationTests(unittest.TestCase):
    def test_one_arm_latest_state_inference_intervention_release_and_hold(self):
        robot = SinglePiperAssistedController(dict(leader_only=True, simulate=True, camera_enabled=False,
            command_address=address(), telemetry_address=address(), arm_connect_timeout_s=5.))
        self.addCleanup(robot.stop)
        robot.start()
        inputs_seen, records = [], []

        class Policy:
            output_chunk_size = 1
            def infer(self, inputs):
                inputs_seen.append(inputs)
                time.sleep(.05)  # Longer than a scheduler tick: old anchors would expire.
                action = inputs['observation.state'].copy()
                action[0] += .7
                return {'action_chunk': action[None, :]}

        scheduler = AssistedScheduler(robot, Policy(), policy_version='single-row', prompt='test', execution_horizon=1)
        self.addCleanup(scheduler.close)
        scheduler.resume()

        def drive(seconds):
            deadline = time.monotonic()+seconds
            while time.monotonic() < deadline:
                sample = robot.get_sample()
                scheduler.step(sample)
                records.append(sample['telemetry'])
                self.assertIsNone(sample['telemetry']['follower']['q'])
                self.assertFalse(sample['telemetry']['control']['follower_present'])
                self.assertEqual(sample['cameras'], {})
                self.assertNotIn('observation.images.top', sample['observation'])
                leader = sample['telemetry']['leader']
                np.testing.assert_allclose(sample['observation']['observation.state'], normalize_pose(leader['q'], leader['gripper_width']))
                time.sleep(1/30)

        drive(1.7)
        self.assertGreater(len(inputs_seen), 12)
        self.assertGreater(robot.status()['target'][0], .7)
        self.assertEqual(robot.status()['reference_revision'], 0)
        epoch = robot.status()['epoch']
        packet = records[-1]
        q = np.array(packet['leader']['q']); q[0] += .2
        robot.simulate_leader(normalize_pose(q, packet['leader']['gripper_width']), duration_s=2.)
        before = len(inputs_seen)
        drive(.05)
        self.assertEqual(robot.status()['human_weight'], 0.)
        self.assertFalse(robot.status()['intervention_active'])
        drive(.7)
        self.assertTrue(robot.status()['human_exclusive'])
        self.assertEqual(robot.status()['human_weight'], 1.)
        np.testing.assert_allclose(robot.status()['applied_policy_velocity'], 0.)
        self.assertGreater(len(inputs_seen), before+2)
        drive(2.3)
        self.assertLess(robot.status()['human_weight'], .03)
        self.assertFalse(robot.status()['intervention_active'])
        self.assertEqual(robot.status()['mode'], 'POLICY')
        self.assertEqual(robot.status()['epoch'], epoch)
        # Initial blend, gravity-only entry, and fresh recovery. Finishing the
        # gain ramp preserves the fresh proposal instead of braking it again.
        self.assertEqual(robot.status()['reference_revision'], 3)
        exclusive = [p for p in records if p['control']['human_exclusive']]
        self.assertTrue(exclusive)
        for packet in exclusive:
            self.assertEqual(packet['control']['human_weight'], 1.)
            np.testing.assert_allclose(packet['control']['applied_policy_velocity'], 0.)
        self.assertTrue(all(not any(k.startswith('context.rtc.') for k in inputs) for inputs in inputs_seen))
        for packet in records:
            self.assertLessEqual(max(abs(v) for v in packet['control']['reference_qd']), .150001)
        scheduler.pause()
        state = robot.status()
        self.assertEqual(state['mode'], 'HOLD')
        np.testing.assert_allclose(state['reference_q'], state['leader_target'][:6])
        np.testing.assert_allclose(state['reference_qd'], 0.)


if __name__ == '__main__':
    unittest.main()
