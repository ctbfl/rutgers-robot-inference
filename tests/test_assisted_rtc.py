"""Compare the assisted client to the existing RTC client on a fixed replay."""

import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from ruri.client.schedulers.assisted import AssistedScheduler
from ruri.client.schedulers.rtc import RTCScheduler


class Robot:
    def __init__(self, args=None):
        self.actions = []
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
            "aligned": True,
            "armed": False,
            "reference_revision": self.revision,
            "policy_reference": self.reference.copy(),
        }

    def request_mode(self, mode):
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

        robot = Robot()
        policy = ReplayPolicy(robot)
        scheduler = AssistedScheduler(
            robot, policy, policy_version="v0", prompt="test", control_hz=50.0
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

    def test_prefix_refusal_holds_instead_of_silent_plain_sampling(self):
        robot = Robot()
        policy = ReplayPolicy(robot, apply_rtc=False)
        scheduler = AssistedScheduler(
            robot, policy, policy_version="v0", prompt="test", control_hz=50.0
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
        robot = Robot()
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
            robot, Policy(), policy_version="v0", prompt="test"
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


if __name__ == "__main__":
    unittest.main()
