"""Request cadence and protocol adaptation without asynchronous timing races."""

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ruri.client.schedulers.assisted import AssistedScheduler
from tests.test_assisted_rtc import Robot


class SchedulerHookTests(unittest.TestCase):
    def scheduler(self, **kwargs):
        robot = Robot()
        policy = SimpleNamespace(output_chunk_size=10)
        # Inspect requests at the worker queue boundary, before inference.
        with patch("ruri.client.schedulers.assisted.threading.Thread"):
            scheduler = AssistedScheduler(robot, policy, policy_version="v1", prompt="test", **kwargs)
        self.addCleanup(scheduler.close)
        return scheduler, robot

    def request(self, scheduler, robot, step):
        scheduler._sample_step = step
        scheduler._request(robot.sample(), robot.status())
        try:
            return scheduler._requests.get_nowait()
        except queue.Empty:
            return None

    def test_late_result_skips_grid_point_without_shifting_cadence(self):
        scheduler, robot = self.scheduler(request_interval_steps=3)
        self.assertIsNotNone(self.request(scheduler, robot, 1))
        self.assertIsNone(self.request(scheduler, robot, 4))  # Still pending.
        scheduler._pending = None
        self.assertIsNone(self.request(scheduler, robot, 5))
        self.assertIsNone(self.request(scheduler, robot, 6))
        self.assertIsNotNone(self.request(scheduler, robot, 7))
        scheduler._pending = None
        self.assertIsNone(self.request(scheduler, robot, 7))  # No duplicate on a tick.
        robot.revision += 1
        self.assertIsNotNone(self.request(scheduler, robot, 8))  # New reference resets grid.

    def test_adapter_changes_worker_input_without_replacing_timeline(self):
        adapted_policy = SimpleNamespace(output_chunk_size=10)
        adapted_inputs = {"custom": "input"}
        adapter = Mock()
        adapter.prepare.return_value = adapted_policy, adapted_inputs
        scheduler, robot = self.scheduler(request_adapter=adapter)
        token, policy, inputs, horizon, timeline = self.request(scheduler, robot, 1)
        self.assertIs(policy, adapted_policy)
        self.assertIs(inputs, adapted_inputs)
        self.assertEqual(horizon, 10)
        self.assertIs(timeline, scheduler._timeline)
        self.assertIs(adapter.prepare.call_args.args[1], scheduler.policy)
        self.assertEqual(adapter.prepare.call_args.args[2]["prompt"], "test")
        self.assertEqual(token.scheduler_instance_id, scheduler.scheduler_instance_id)
        other, _ = self.scheduler()
        self.assertNotEqual(token.scheduler_instance_id, other.scheduler_instance_id)

    def test_human_policy_replacement_is_explicit_and_rejects_active_execution(self):
        scheduler, robot = self.scheduler()
        robot.mode = "HUMAN"
        policy = SimpleNamespace(output_chunk_size=10)
        with self.assertRaises(RuntimeError):
            scheduler.replace_policy(policy, "v2")
        scheduler.replace_policy(policy, "v2", allow_human=True)
        self.assertIs(scheduler.policy, policy)
        self.assertEqual(robot.mode, "HUMAN")
        scheduler._want_policy = True
        with self.assertRaises(RuntimeError):
            scheduler.replace_policy(policy, "v3", allow_human=True)

    def test_request_interval_must_fit_strictly_inside_horizon(self):
        for interval in (0, -1, 10, 11, True, 1.5):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.scheduler(request_interval_steps=interval)


if __name__ == "__main__":
    unittest.main()
