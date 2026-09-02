from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ruri.client.schedulers import BlockingScheduler


class BlockingSchedulerTests(unittest.TestCase):
    def test_logs_inference_and_every_executed_action(self):
        with tempfile.TemporaryDirectory() as log_dir:
            args = SimpleNamespace(
                control_hz=30.0,
                max_chunks=1,
                execute_actions_per_chunk=1,
                scheduler_log_dir=log_dir,
            )

            class FakeController:
                action_bounds = (
                    np.asarray([-100.0], dtype=np.float32),
                    np.asarray([100.0], dtype=np.float32),
                )

                def __init__(self, received_args):
                    pass

                def start(self):
                    pass

                def get_observation(self):
                    return {"observation.state": np.zeros(1, dtype=np.float32)}

                def send_action(self, action):
                    pass

                def stop(self):
                    pass

            class FakePolicy:
                output_chunk_size = 2

                def __init__(self, received_args):
                    pass

                def infer(self, inputs):
                    return {
                        "action_chunk": np.asarray([[101.0], [2.0]], dtype=np.float32)
                    }

            scheduler = BlockingScheduler()
            policy = FakePolicy(args)
            with patch("ruri.client.schedulers.blocking.time.sleep"):
                scheduler.run(FakeController, policy, args=args)

            self.assertIsNotNone(scheduler.last_log_path)
            records = [
                json.loads(line)
                for line in Path(scheduler.last_log_path).read_text().splitlines()
            ]
            self.assertTrue(any(r["event"] == "inference_returned" for r in records))
            actions = [r for r in records if r["event"] == "action"]
            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0]["action"], [100.0])
            self.assertEqual(actions[0]["pre_clip_action"], [101.0])
            self.assertTrue(actions[0]["clipped"])
            stop = next(r for r in records if r["event"] == "run_stop")
            self.assertEqual(stop["stats"]["actions_sent"], 1)

    def test_runs_complete_chunks_with_one_shared_args_object(self):
        events = []
        instances = {}
        args = SimpleNamespace(
            control_hz=30.0,
            max_chunks=2,
            prompt="test task",
            scheduler_log_enabled=False,
        )

        class FakeController:
            def __init__(self, received_args):
                self.args = received_args
                self.observations = 0
                self.actions = []
                instances["controller"] = self
                events.append("controller.init")

            def start(self):
                events.append("controller.start")

            def get_observation(self):
                self.observations += 1
                events.append(f"controller.observe.{self.observations}")
                return {"observation.state": np.asarray([self.observations])}

            def send_action(self, action):
                value = int(action[0])
                self.actions.append(value)
                events.append(f"controller.action.{value}")

            def stop(self):
                events.append("controller.stop")

        class FakePolicy:
            output_chunk_size = 2

            def __init__(self, received_args):
                self.args = received_args
                self.calls = 0
                instances["policy"] = self
                events.append("policy.init")

            def infer(self, observation):
                self.calls += 1
                self.assert_shared_args()
                if observation["prompt"] != args.prompt:
                    raise AssertionError("Scheduler did not inject args.prompt")
                if "context.actions_per_chunk" in observation:
                    raise AssertionError("Scheduler injected obsolete chunk context")
                events.append(f"policy.infer.{self.calls}")
                base = self.calls * 10
                return {
                    "action_chunk": np.asarray(
                        [[base + 1], [base + 2]], dtype=np.float32
                    )
                }

            def assert_shared_args(self):
                if self.args is not args:
                    raise AssertionError("Policy did not receive the global args object")

        policy = FakePolicy(args)
        with patch("ruri.client.schedulers.blocking.time.sleep"):
            BlockingScheduler().run(FakeController, policy, args=args)

        self.assertIs(instances["controller"].args, args)
        self.assertIs(instances["policy"].args, args)
        self.assertEqual(instances["controller"].actions, [11, 12, 21, 22])
        self.assertEqual(
            events,
            [
                "policy.init",
                "controller.init",
                "controller.start",
                "controller.observe.1",
                "policy.infer.1",
                "controller.action.11",
                "controller.action.12",
                "controller.observe.2",
                "policy.infer.2",
                "controller.action.21",
                "controller.action.22",
                "controller.stop",
            ],
        )

    def test_stops_controller_when_controller_start_fails(self):
        events = []
        args = SimpleNamespace(
            control_hz=30.0,
            max_chunks=1,
            scheduler_log_enabled=False,
        )

        class FailingController:
            def __init__(self, received_args):
                self.args = received_args

            def start(self):
                events.append("controller.start")
                raise RuntimeError("hardware failed")

            def stop(self):
                events.append("controller.stop")

        class FakePolicy:
            output_chunk_size = 1

            def __init__(self, received_args):
                self.args = received_args

        policy = FakePolicy(args)
        with self.assertRaisesRegex(RuntimeError, "hardware failed"):
            BlockingScheduler().run(FailingController, policy, args=args)

        self.assertEqual(
            events,
            ["controller.start", "controller.stop"],
        )

    def test_rejects_execution_prefix_before_starting_hardware(self):
        events = []
        args = SimpleNamespace(
            control_hz=30.0,
            max_chunks=1,
            execute_actions_per_chunk=3,
            scheduler_log_enabled=False,
        )

        class FakeController:
            def __init__(self, received_args):
                pass

            def start(self):
                events.append("controller.start")

            def stop(self):
                events.append("controller.stop")

        class FakePolicy:
            output_chunk_size = 2

            def __init__(self, received_args):
                pass

        policy = FakePolicy(args)
        with self.assertRaisesRegex(ValueError, "output_chunk_size"):
            BlockingScheduler().run(FakeController, policy, args=args)

        self.assertEqual(events, ["controller.stop"])

    def test_predicts_ten_but_executes_only_first_five(self):
        instances = {}
        args = SimpleNamespace(
            control_hz=30.0,
            max_chunks=2,
            execute_actions_per_chunk=5,
            scheduler_log_enabled=False,
        )

        class FakeController:
            def __init__(self, received_args):
                self.actions = []
                instances["controller"] = self

            def start(self):
                pass

            def get_observation(self):
                return {"observation.state": np.asarray([0.0])}

            def send_action(self, action):
                self.actions.append(int(action[0]))

            def stop(self):
                pass

        class FakePolicy:
            output_chunk_size = 10

            def __init__(self, received_args):
                self.calls = 0
                instances["policy"] = self

            def start(self):
                pass

            def infer(self, observation):
                self.calls += 1
                if "context.actions_per_chunk" in observation:
                    raise AssertionError("Scheduler injected obsolete chunk context")
                base = self.calls * 100
                return {
                    "action_chunk": np.arange(
                        base, base + 10, dtype=np.float32
                    )[:, None]
                }

        policy = FakePolicy(args)
        with patch("ruri.client.schedulers.blocking.time.sleep"):
            BlockingScheduler().run(FakeController, policy, args=args)

        self.assertEqual(instances["policy"].calls, 2)
        self.assertEqual(
            instances["controller"].actions,
            [100, 101, 102, 103, 104, 200, 201, 202, 203, 204],
        )

    def test_rejects_invalid_action_chunk(self):
        with self.assertRaisesRegex(ValueError, "non-empty shape"):
            BlockingScheduler._action_chunk({"action_chunk": np.asarray([])})

        with self.assertRaisesRegex(ValueError, "NaN"):
            BlockingScheduler._action_chunk(
                {"action_chunk": np.asarray([[float("nan")]])}
            )

    def test_minimum_jerk_keeps_path_duration_and_eases_both_ends(self):
        start = np.asarray([0.0], dtype=np.float32)
        chunk = np.arange(1.0, 11.0, dtype=np.float32)[:, None]

        profiled = BlockingScheduler._minimum_jerk_chunk(
            start_action=start,
            action_chunk=chunk,
            control_hz=30.0,
        )

        self.assertEqual(profiled.shape, chunk.shape)
        np.testing.assert_array_equal(profiled[-1], chunk[-1])
        steps = np.diff(np.concatenate((start, profiled[:, 0])))
        self.assertGreater(steps[len(steps) // 2], steps[0] * 10.0)
        self.assertGreater(steps[len(steps) // 2], steps[-1] * 10.0)
        self.assertTrue(np.all(np.diff(profiled[:, 0]) > 0.0))

    def test_minimum_jerk_rejects_infeasible_fixed_duration_limit(self):
        with self.assertRaisesRegex(ValueError, "infeasible in the original duration"):
            BlockingScheduler._minimum_jerk_chunk(
                start_action=np.asarray([0.0]),
                action_chunk=np.arange(1.0, 11.0)[:, None],
                control_hz=30.0,
                max_acceleration=1.0,
            )


if __name__ == "__main__":
    unittest.main()
