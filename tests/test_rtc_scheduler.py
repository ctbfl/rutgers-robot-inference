from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest

import numpy as np

from ruri.client.schedulers import RTCScheduler


class RTCSchedulerTests(unittest.TestCase):
    def test_discards_actions_sent_during_inference_and_replaces_tail(self):
        instances: dict[str, object] = {}
        eight_actions_sent = threading.Event()
        args = SimpleNamespace(
            control_hz=100.0,
            execution_horizon=5,
            max_chunks=2,
            prompt="test task",
            scheduler_log_enabled=False,
        )

        class FakeController:
            def __init__(self, received_args):
                self.args = received_args
                self.actions: list[int] = []
                instances["controller"] = self

            def start(self):
                pass

            def get_observation(self):
                return {"observation.state": np.asarray([len(self.actions)])}

            def send_action(self, action):
                self.actions.append(int(action[0]))
                if len(self.actions) == 8:
                    eight_actions_sent.set()
                return action

            def stop(self):
                pass

        class DelayedPolicy:
            output_chunk_size = 10

            def __init__(self, received_args):
                self.args = received_args
                self.calls = 0
                self.inputs: list[dict] = []
                instances["policy"] = self

            def start(self):
                pass

            def infer(self, inputs):
                self.calls += 1
                self.inputs.append(inputs)
                if self.calls == 1:
                    return {
                        "action_chunk": np.arange(10, dtype=np.float32)[:, None]
                    }
                if not eight_actions_sent.wait(timeout=1.0):
                    raise AssertionError("controller did not keep executing")
                # Let send_action return so timestep 7 is committed before the
                # policy response is timestamped.
                time.sleep(0.001)
                return {
                    "action_chunk": np.arange(100, 110, dtype=np.float32)[:, None],
                    "rtc.applied": True,
                }

            def stop(self):
                pass

        scheduler = RTCScheduler()
        scheduler.run(FakeController, DelayedPolicy, args=args)

        controller = instances["controller"]
        policy = instances["policy"]
        self.assertIs(controller.args, args)
        self.assertIs(policy.args, args)
        self.assertEqual(policy.calls, 2)
        self.assertEqual(
            controller.actions,
            [0, 1, 2, 3, 4, 5, 6, 7, 103, 104, 105, 106, 107, 108, 109],
        )
        second_inputs = policy.inputs[1]
        self.assertEqual(second_inputs["context.rtc.consumed_steps"], 5)
        self.assertEqual(
            second_inputs["context.rtc.estimated_inference_delay_steps"], 4
        )
        np.testing.assert_array_equal(
            second_inputs["context.rtc.prev_chunk_left_over"][:, 0],
            [5, 6, 7, 8, 9],
        )
        self.assertNotIn("context.rtc.execution_horizon", second_inputs)
        self.assertNotIn("context.rtc.inference_delay", second_inputs)
        self.assertEqual(scheduler.last_run_stats["expired_actions"], 3)
        self.assertEqual(scheduler.last_run_stats["max_actual_delay_steps"], 3)
        self.assertEqual(scheduler.last_run_stats["hold_ticks"], 0)

    def test_holds_last_action_if_inference_outlives_old_chunk(self):
        instances: dict[str, object] = {}
        five_actions_sent = threading.Event()
        args = SimpleNamespace(
            control_hz=100.0,
            execution_horizon=2,
            max_chunks=2,
            scheduler_log_enabled=False,
        )

        class FakeController:
            def __init__(self, received_args):
                self.actions: list[int] = []
                instances["controller"] = self

            def start(self):
                pass

            def get_observation(self):
                return {"observation.state": np.zeros(1)}

            def send_action(self, action):
                self.actions.append(int(action[0]))
                if len(self.actions) == 5:
                    five_actions_sent.set()

            def stop(self):
                pass

        class SlowPolicy:
            output_chunk_size = 3

            def __init__(self, received_args):
                self.calls = 0

            def start(self):
                pass

            def infer(self, inputs):
                self.calls += 1
                if self.calls == 2:
                    if not five_actions_sent.wait(timeout=1.0):
                        raise AssertionError("expected held command was not sent")
                    time.sleep(0.001)
                    return {
                        "action_chunk": np.asarray([[10.0], [11.0], [12.0]]),
                        "rtc.applied": True,
                    }
                return {"action_chunk": np.asarray([[0.0], [1.0], [2.0]])}

            def stop(self):
                pass

        scheduler = RTCScheduler()
        scheduler.run(FakeController, SlowPolicy, args=args)

        self.assertEqual(instances["controller"].actions, [0, 1, 2, 2, 2])
        self.assertEqual(scheduler.last_run_stats["hold_ticks"], 2)
        self.assertEqual(scheduler.last_run_stats["stale_chunks"], 1)

    def test_trace_records_request_return_install_and_every_action(self):
        with tempfile.TemporaryDirectory() as log_dir:
            send_times: list[float] = []
            args = SimpleNamespace(
                control_hz=200.0,
                execution_horizon=1,
                max_chunks=1,
                scheduler_log_dir=log_dir,
            )

            class FakeController:
                def __init__(self, received_args):
                    pass

                def start(self):
                    pass

                def get_observation(self):
                    return {"observation.state": np.zeros(1)}

                def send_action(self, action):
                    send_times.append(time.monotonic())
                    return action

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

            scheduler = RTCScheduler()
            scheduler.run(FakeController, FakePolicy, args=args)

            log_path = Path(scheduler.last_log_path)
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            events = [record["event"] for record in records]
            self.assertIn("inference_triggered", events)
            self.assertIn("inference_started", events)
            self.assertIn("inference_returned", events)
            self.assertIn("chunk_installed", events)
            actions = [record for record in records if record["event"] == "action"]
            self.assertEqual([record["timestep"] for record in actions], [0, 1])
            self.assertGreaterEqual(send_times[1] - send_times[0], 0.003)

    def test_rejects_policy_horizon_mismatch(self):
        args = SimpleNamespace(
            control_hz=100.0,
            execution_horizon=1,
            max_chunks=1,
            scheduler_log_enabled=False,
        )

        class FakeController:
            def __init__(self, received_args):
                pass

            def start(self):
                pass

            def get_observation(self):
                return {"observation.state": np.zeros(1)}

            def stop(self):
                pass

        class ShortPolicy:
            output_chunk_size = 3

            def __init__(self, received_args):
                pass

            def start(self):
                pass

            def infer(self, inputs):
                return {"action_chunk": np.zeros((2, 1), dtype=np.float32)}

            def stop(self):
                pass

        with self.assertRaisesRegex(ValueError, "does not match"):
            RTCScheduler().run(FakeController, ShortPolicy, args=args)

    def test_rejects_execution_horizon_before_starting_hardware(self):
        events = []
        args = SimpleNamespace(
            control_hz=100.0,
            execution_horizon=4,
            max_chunks=0,
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
            output_chunk_size = 3

            def __init__(self, received_args):
                pass

            def start(self):
                events.append("policy.start")

            def stop(self):
                events.append("policy.stop")

        with self.assertRaisesRegex(ValueError, "output_chunk_size"):
            RTCScheduler().run(FakeController, FakePolicy, args=args)

        self.assertNotIn("controller.start", events)
        self.assertEqual(events[0], "policy.start")
        self.assertIn("controller.stop", events)
        self.assertIn("policy.stop", events)

    def test_rejects_silent_rtc_server_fallback(self):
        args = SimpleNamespace(
            control_hz=200.0,
            execution_horizon=2,
            max_chunks=2,
            scheduler_log_enabled=False,
        )

        class FakeController:
            def __init__(self, received_args):
                pass

            def start(self):
                pass

            def get_observation(self):
                return {"observation.state": np.zeros(1)}

            def send_action(self, action):
                pass

            def stop(self):
                pass

        class FallbackPolicy:
            output_chunk_size = 4

            def __init__(self, received_args):
                self.calls = 0

            def start(self):
                pass

            def infer(self, inputs):
                self.calls += 1
                return {
                    "action_chunk": np.zeros((4, 1), dtype=np.float32),
                    "rtc.applied": False,
                    "rtc.reason": "test fallback",
                }

            def stop(self):
                pass

        with self.assertRaisesRegex(RuntimeError, "test fallback"):
            RTCScheduler().run(FakeController, FallbackPolicy, args=args)


if __name__ == "__main__":
    unittest.main()
