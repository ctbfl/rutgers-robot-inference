from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from ruri.client.schedulers import RollingScheduler


class RollingSchedulerTests(unittest.TestCase):
    def test_merges_by_timestep_and_drops_already_executed_actions(self):
        current = {
            5: np.asarray([50.0], dtype=np.float32),
            6: np.asarray([60.0], dtype=np.float32),
        }
        incoming = np.asarray([[30.0], [40.0], [5.0], [6.0], [7.0]])

        merged, stale, overlap = RollingScheduler._merge_chunks(
            current=current,
            incoming=incoming,
            first_timestep=3,
            latest_timestep=4,
            old_weight=0.3,
            new_weight=0.7,
        )

        self.assertEqual(list(merged), [5, 6, 7])
        np.testing.assert_allclose(merged[5], [18.5])
        np.testing.assert_allclose(merged[6], [22.2])
        np.testing.assert_allclose(merged[7], [7.0])
        self.assertEqual(stale, 2)
        self.assertEqual(overlap, 2)

    def test_runs_prefetch_in_background_with_shared_global_args(self):
        events: list[str] = []
        instances = {}
        args = SimpleNamespace(
            control_hz=100.0,
            max_chunks=2,
            chunk_size_threshold=0.5,
            aggregate_fn_name="weighted_average",
            prompt="pick the object",
            scheduler_log_enabled=False,
        )

        class FakeController:
            def __init__(self, received_args):
                self.args = received_args
                self.actions: list[np.ndarray] = []
                self.observation_threads: list[str] = []
                instances["controller"] = self

            def start(self):
                events.append("controller.start")

            def get_observation(self):
                self.observation_threads.append(threading.current_thread().name)
                return {"observation.state": np.asarray([len(self.actions)])}

            def send_action(self, action):
                self.actions.append(np.asarray(action).copy())

            def stop(self):
                events.append("controller.stop")

        class FakePolicy:
            output_chunk_size = 6

            def __init__(self, received_args):
                self.args = received_args
                self.calls = 0
                self.inputs = []
                instances["policy"] = self

            def start(self):
                events.append("policy.start")

            def infer(self, inputs):
                self.calls += 1
                self.inputs.append(inputs)
                base = self.calls * 10
                return {
                    "action_chunk": np.arange(base, base + 6, dtype=np.float32)[
                        :, None
                    ]
                }

            def stop(self):
                events.append("policy.stop")

        scheduler = RollingScheduler()
        scheduler.run(FakeController, FakePolicy, args=args)

        robot = instances["controller"]
        policy = instances["policy"]
        self.assertIs(robot.args, args)
        self.assertIs(policy.args, args)
        self.assertEqual(policy.calls, 2)
        self.assertEqual(policy.inputs[0]["prompt"], args.prompt)
        self.assertNotIn("context.actions_per_chunk", policy.inputs[1])
        self.assertEqual(
            robot.observation_threads,
            ["ruri-rolling-inference", "ruri-rolling-inference"],
        )
        self.assertEqual(events[0:2], ["policy.start", "controller.start"])
        self.assertEqual(events[-2:], ["controller.stop", "policy.stop"])
        self.assertEqual(scheduler.last_run_stats["chunks_received"], 2)
        self.assertGreater(scheduler.last_run_stats["overlap_actions"], 0)

    def test_holds_last_target_at_control_rate_when_inference_is_late(self):
        instances = {}
        args = SimpleNamespace(
            control_hz=100.0,
            max_chunks=2,
            chunk_size_threshold=0.5,
            scheduler_log_enabled=False,
        )

        class TimedController:
            def __init__(self, received_args):
                self.send_times: list[float] = []
                instances["controller"] = self

            def start(self):
                pass

            def get_observation(self):
                return {"observation.state": np.zeros(1)}

            def send_action(self, action):
                self.send_times.append(time.monotonic())

            def stop(self):
                pass

        class SlowPolicy:
            output_chunk_size = 2

            def __init__(self, received_args):
                self.calls = 0

            def start(self):
                pass

            def infer(self, inputs):
                self.calls += 1
                if self.calls == 2:
                    time.sleep(0.055)
                return {"action_chunk": np.asarray([[1.0], [2.0]])}

            def stop(self):
                pass

        scheduler = RollingScheduler()
        scheduler.run(TimedController, SlowPolicy, args=args)

        send_times = instances["controller"].send_times
        intervals = np.diff(send_times)
        self.assertGreaterEqual(scheduler.last_run_stats["hold_ticks"], 2)
        self.assertGreaterEqual(len(send_times), 4)
        self.assertLess(float(intervals.max()), 0.03)

    def test_logs_every_sent_and_rejected_action_with_fatal_error(self):
        with tempfile.TemporaryDirectory() as log_dir:
            args = SimpleNamespace(
                control_hz=200.0,
                max_chunks=1,
                scheduler_log_dir=log_dir,
            )

            class RejectingController:
                def __init__(self, received_args):
                    pass

                def start(self):
                    pass

                def get_observation(self):
                    return {"observation.state": np.zeros(1)}

                def send_action(self, action):
                    if float(action[0]) > 1.0:
                        raise ValueError("test action rejected")

                def stop(self):
                    pass

            class FakePolicy:
                output_chunk_size = 2

                def __init__(self, received_args):
                    pass

                def start(self):
                    pass

                def infer(self, inputs):
                    return {"action_chunk": np.asarray([[1.0], [2.0]])}

                def stop(self):
                    pass

            scheduler = RollingScheduler()
            with self.assertRaisesRegex(ValueError, "test action rejected"):
                scheduler.run(RejectingController, FakePolicy, args=args)

            self.assertIsNotNone(scheduler.last_log_path)
            log_path = Path(scheduler.last_log_path)
            self.assertTrue(log_path.is_file())
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            actions = [record for record in records if record["event"] == "action"]
            self.assertEqual([record["outcome"] for record in actions], ["sent", "rejected"])
            self.assertEqual(actions[1]["action"], [2.0])
            self.assertEqual(actions[1]["error_type"], "ValueError")
            inference = next(
                record
                for record in records
                if record["event"] == "inference_result"
            )
            self.assertEqual(inference["observation_timestep"], -1)
            self.assertEqual(inference["first_action_timestep"], 0)
            self.assertTrue(
                any(record["event"] == "scheduler_error" for record in records)
            )

    def test_rejects_unknown_aggregation(self):
        args = SimpleNamespace(aggregate_fn_name="mystery")

        with self.assertRaisesRegex(ValueError, "Unknown aggregate_fn_name"):
            RollingScheduler().run(lambda _: object(), lambda _: object(), args=args)


if __name__ == "__main__":
    unittest.main()
