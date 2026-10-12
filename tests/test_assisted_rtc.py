"""Compare the assisted client to the existing RTC client on a fixed replay."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from ruri.client.schedulers.assisted import AssistedScheduler
from ruri.client.schedulers.rtc import RTCScheduler


class Robot:
    def __init__(self, args=None):
        self.actions = []
        self.mode_requests = []
        self.eight = threading.Event()
        self.mode, self.epoch, self.revision = "HOLD", 0, 0
        self.last_command_id = 0
        self.reference = np.zeros(7, dtype=np.float32)

    def start(self):
        pass

    def stop(self):
        pass

    def get_observation(self):
        return {"observation.state": np.zeros(7, dtype=np.float32)}

    def status(self):
        return {
            "mode": self.mode,
            "epoch": self.epoch,
            "aligned": False,
            "aligning": False,
            "armed": False,
            "reference_revision": self.revision,
            "policy_reference": self.reference.copy(),
        }

    def request_mode(self, mode):
        self.mode_requests.append(mode)
        self.mode = mode
        self.epoch += 1
        return self.status()

    def send_action(self, target, **metadata):
        self.last_command_id += 1
        if metadata.get("model_action", True):
            self.actions.append(target.copy())
            self.reference = target.copy()
            if len(self.actions) == 8:
                self.eight.set()

    def sample(self):
        return {
            "observation_id": str(time.monotonic_ns()),
            "monotonic_ns": time.monotonic_ns(),
            "observation": self.get_observation(),
            "telemetry": {"control": self.status()},
        }


class ReplayPolicy:
    output_chunk_size = 10

    def __init__(self, robot, apply_rtc=True):
        self.robot, self.apply_rtc = robot, apply_rtc
        self.inputs = []
        self.release = threading.Event()

    def infer(self, inputs):
        self.inputs.append(inputs)
        call = len(self.inputs)
        if call == 2:
            if not self.robot.eight.wait(2):
                raise AssertionError("RTC did not keep executing during inference")
            time.sleep(0.002)  # action 7 must commit before this result returns
        elif call > 2:
            self.release.wait(2)
        chunk = np.zeros((10, 7), dtype=np.float32)
        chunk[:, 0] = np.arange(10) + (100 if call > 1 else 0)
        return {
            "action_chunk": chunk,
            "rtc.applied": self.apply_rtc,
            "rtc.method": "training-time",
        }


class AssistedRTCEquivalenceTests(unittest.TestCase):
    robot_type = Robot
    scheduler_options = {}

    def test_no_intervention_matches_existing_rtc_actions_prefix_and_delay(self):
        args = SimpleNamespace(
            control_hz=50.0,
            execution_horizon=5,
            max_chunks=2,
            prompt="test",
            scheduler_log_enabled=False,
        )
        original_robot = Robot()
        original_policy = ReplayPolicy(original_robot)
        RTCScheduler().run(lambda args: original_robot, original_policy, args=args)

        robot = self.robot_type()
        policy = ReplayPolicy(robot)
        scheduler = AssistedScheduler(
            robot, policy, policy_version="v0", prompt="test", control_hz=50.0,
            **self.scheduler_options,
        )
        self.addCleanup(scheduler.close)
        self.addCleanup(policy.release.set)
        scheduler.resume()
        deadline = time.monotonic() + 3
        audits = []
        while len(robot.actions) < 15 and time.monotonic() < deadline:
            audits.extend(scheduler.step(robot.sample())["inference_results"])
            time.sleep(0.02)
        np.testing.assert_array_equal(robot.actions, original_robot.actions)
        self.assertEqual(
            [a[0] for a in robot.actions],
            [0, 1, 2, 3, 4, 5, 6, 7, 103, 104, 105, 106, 107, 108, 109],
        )
        for key in (
            "context.rtc.prev_chunk_left_over",
            "context.rtc.consumed_steps",
            "context.rtc.estimated_inference_delay_steps",
        ):
            np.testing.assert_array_equal(
                policy.inputs[1][key], original_policy.inputs[1][key]
            )
        self.assertEqual(audits[1]["install"]["expired"], 3)
        self.assertEqual(robot.mode_requests, ["POLICY"])

    def test_prefix_refusal_holds_instead_of_silent_plain_sampling(self):
        robot = self.robot_type()
        policy = ReplayPolicy(robot, apply_rtc=False)
        scheduler = AssistedScheduler(
            robot, policy, policy_version="v0", prompt="test", control_hz=50.0,
            **self.scheduler_options,
        )
        self.addCleanup(scheduler.close)
        self.addCleanup(policy.release.set)
        scheduler.resume()
        with self.assertRaisesRegex(RuntimeError, "policy inference failed"):
            for _ in range(40):
                scheduler.step(robot.sample())
                time.sleep(0.02)
        self.assertEqual(robot.mode, "HOLD")
        self.assertTrue(all(action[0] < 100 for action in robot.actions))

    def test_rebase_drops_rtc_history_and_bootstraps_fresh_chunk_from_zero(self):
        robot = self.robot_type()
        entered, release = threading.Event(), threading.Event()
        inputs_seen = []

        class Policy:
            output_chunk_size = 10

            def infer(self, inputs):
                inputs_seen.append(inputs)
                if len(inputs_seen) == 2:
                    entered.set()
                    release.wait(2)
                target = np.zeros((10, 7), dtype=np.float32)
                target[:, 0] = 10 * len(inputs_seen) + np.arange(10)
                return {"action_chunk": target, "rtc.applied": True}

        scheduler = AssistedScheduler(
            robot, Policy(), policy_version="v0", prompt="test", **self.scheduler_options,
        )
        self.addCleanup(scheduler.close)
        self.addCleanup(release.set)
        scheduler.resume()
        for _ in range(30):
            scheduler.step(robot.sample())
            time.sleep(0.01)
            if entered.is_set():
                break
        self.assertTrue(entered.is_set())
        before = len(robot.actions)
        epoch = robot.epoch
        robot.revision += 1
        for _ in range(8):
            scheduler.step(robot.sample())
        self.assertEqual(len(robot.actions), before)
        release.set()
        for _ in range(30):
            scheduler.step(robot.sample())
            time.sleep(0.01)
            if len(robot.actions) > before:
                break
        self.assertEqual(robot.epoch, epoch)
        self.assertEqual(robot.actions[before][0], 30)
        self.assertNotIn("context.rtc.prev_chunk_left_over", inputs_seen[2])
        self.assertGreaterEqual(scheduler.discarded_results, 1)


class GravityRobot(Robot):
    authority_phase = 'policy'
    human_weight = 0.

    def status(self):
        return {**super().status(), 'control_schema': 'ruri_gravity_handoff_v1',
                'authority_phase': self.authority_phase, 'human_weight': self.human_weight,
                'intervention_active': self.authority_phase != 'policy'}


class GravityRTCEquivalenceTests(AssistedRTCEquivalenceTests):
    robot_type = GravityRobot
    scheduler_options = {'rtc_during_policy': True}

    def test_takeover_and_recovery_discard_prefix_including_inflight_results(self):
        robot = GravityRobot()
        seen = []
        entered, release = threading.Event(), threading.Event()

        class Policy:
            output_chunk_size = 10

            def infer(self, inputs):
                seen.append((robot.authority_phase, inputs))
                call = len(seen)
                if call == 2:
                    entered.set()
                    release.wait(2.)
                chunk = np.zeros((10, 7), np.float32)
                chunk[:, 0] = 20. if call == 2 else 30.
                return {'action_chunk': chunk, 'rtc.applied': True}

        scheduler = AssistedScheduler(robot, Policy(), policy_version='v0', prompt='test',
                                      execution_horizon=2, rtc_during_policy=True)
        self.addCleanup(scheduler.close)
        self.addCleanup(release.set)
        scheduler.resume()

        def drive_until(predicate):
            deadline = time.monotonic()+2.
            while not predicate() and time.monotonic() < deadline:
                scheduler.step(robot.sample())
                time.sleep(.003)
            self.assertTrue(predicate())

        drive_until(entered.is_set)
        self.assertIn('context.rtc.prev_chunk_left_over', seen[1][1])
        robot.authority_phase, robot.human_weight = 'blend', .2
        robot.revision += 1
        scheduler.step(robot.sample())
        release.set()
        for phase, alpha in (('blend', .2), ('human', 1.), ('recover', .5)):
            robot.authority_phase, robot.human_weight = phase, alpha
            robot.revision += 1
            before = len(seen)
            drive_until(lambda: len(seen) >= before+2)
            for observed_phase, inputs in seen[before:]:
                self.assertFalse(any(k.startswith('context.rtc.') for k in inputs))
        before = len(seen)
        # The hardware deliberately preserves revision at recovery completion.
        # Scheduler must still discard the unexecuted recovery proposals.
        robot.authority_phase, robot.human_weight = 'policy', 0.
        drive_until(lambda: len(seen) >= before+3)
        resumed = [inputs for phase, inputs in seen[before:] if phase == 'policy']
        self.assertNotIn('context.rtc.prev_chunk_left_over', resumed[0])
        self.assertTrue(any('context.rtc.prev_chunk_left_over' in x for x in resumed[1:]))
        self.assertFalse(any(a[0] == 20. for a in robot.actions))
        self.assertGreater(scheduler.discarded_results, 0)


class AssistedSchedulerShutdownTests(unittest.TestCase):
    def scheduler(self, state):
        robot = Mock()
        robot.status.return_value = state
        scheduler = AssistedScheduler(robot, SimpleNamespace(output_chunk_size=1),
                                      policy_version='test', prompt='test')
        self.addCleanup(scheduler.close)
        return scheduler, robot

    def test_close_leaves_fault_recovery_in_charge(self):
        for phase, mode in (('recovering', 'FAULT'), ('aborted', 'FAULT'),
                            ('engaged', 'FAULT'), ('engaged', 'STOPPED')):
            with self.subTest(phase=phase, mode=mode):
                scheduler, robot = self.scheduler({'phase': phase, 'mode': mode})
                scheduler.close()
                robot.request_mode.assert_not_called()
                self.assertTrue(scheduler._closed.is_set())
                self.assertFalse(scheduler._thread.is_alive())

    def test_healthy_close_still_requests_hold(self):
        scheduler, robot = self.scheduler({'phase': 'engaged', 'mode': 'POLICY'})
        scheduler.close()
        scheduler.close()
        robot.request_mode.assert_called_once_with('HOLD')

    def test_recovery_race_does_not_mask_original_fault(self):
        scheduler, robot = self.scheduler({'phase': 'engaged', 'mode': 'POLICY'})
        robot.request_mode.side_effect = TimeoutError('worker already recovering')
        with self.assertLogs('ruri.client.schedulers.assisted', level='WARNING'):
            with self.assertRaisesRegex(RuntimeError, 'original hardware fault'):
                try:
                    raise RuntimeError('original hardware fault')
                finally:
                    scheduler.close()
        self.assertFalse(scheduler._thread.is_alive())

    def test_unavailable_status_during_cleanup_does_not_raise(self):
        scheduler, robot = self.scheduler({})
        robot.status.side_effect = TimeoutError('feedback closed')
        with self.assertLogs('ruri.client.schedulers.assisted', level='WARNING'):
            scheduler.close()
        robot.request_mode.assert_not_called()
        self.assertFalse(scheduler._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
