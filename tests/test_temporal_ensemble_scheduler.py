from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest

import numpy as np

from ruri.client.schedulers import TemporalEnsembleScheduler
from ruri.client.schedulers.temporal_ensemble import _ActionEnsemble


class TemporalEnsembleSchedulerTests(unittest.TestCase):
    def test_online_weights_match_lerobot_exponential_formula(self):
        coefficient = 0.01
        ensemble = _ActionEnsemble(coefficient)
        ensemble.add_chunk(np.asarray([[0.0], [10.0], [20.0]]), 0)
        ensemble.add_chunk(np.asarray([[100.0], [110.0], [120.0]]), 1)
        ensemble.add_chunk(np.asarray([[200.0], [210.0], [220.0]]), 2)

        first = ensemble.select_next()
        self.assertIsNotNone(first)
        np.testing.assert_allclose(first.value, [0.0])
        ensemble.commit(first, first.value)

        second = ensemble.select_next()
        weight_1 = np.exp(-coefficient)
        expected_second = (10.0 + weight_1 * 100.0) / (1.0 + weight_1)
        np.testing.assert_allclose(second.value, [expected_second], rtol=1e-6)
        ensemble.commit(second, second.value)

        third = ensemble.select_next()
        weight_2 = np.exp(-2.0 * coefficient)
        expected_third = (
            20.0 + weight_1 * 110.0 + weight_2 * 200.0
        ) / (1.0 + weight_1 + weight_2)
        np.testing.assert_allclose(third.value, [expected_third], rtol=1e-6)
        self.assertEqual(third.contributors, 3)

    def test_queries_each_step_and_ensembles_every_overlapping_prediction(self):
        instances = {}
        args = SimpleNamespace(
            control_hz=100.0,
            temporal_ensemble_coeff=0.0,
            max_chunks=3,
            scheduler_log_enabled=False,
        )

        class FakeController:
            def __init__(self, received_args):
                self.actions: list[float] = []
                instances["controller"] = self

            def start(self):
                pass

            def get_observation(self):
                return {"observation.state": np.asarray([len(self.actions)])}

            def send_action(self, action):
                self.actions.append(float(action[0]))
                return action

            def stop(self):
                pass

        class FakePolicy:
            output_chunk_size = 4

            def __init__(self, received_args):
                self.calls = 0
                instances["policy"] = self

            def start(self):
                pass

            def infer(self, inputs):
                self.calls += 1
                base = (self.calls - 1) * 100.0
                return {
                    "action_chunk": np.arange(
                        base, base + 40.0, 10.0, dtype=np.float32
                    )[:, None]
                }

            def stop(self):
                pass

        scheduler = TemporalEnsembleScheduler()
        scheduler.run(FakeController, FakePolicy, args=args)

        self.assertEqual(instances["policy"].calls, 3)
        np.testing.assert_allclose(
            instances["controller"].actions,
            [0.0, 55.0, 110.0, 120.0, 175.0, 230.0],
        )
        self.assertEqual(scheduler.last_run_stats["hold_ticks"], 0)
        self.assertEqual(scheduler.last_run_stats["max_contributors"], 3)

    def test_late_result_discards_only_expired_predictions(self):
        instances = {}
        three_actions_sent = threading.Event()
        args = SimpleNamespace(
            control_hz=100.0,
            temporal_ensemble_coeff=0.0,
            max_chunks=2,
            scheduler_log_enabled=False,
        )

        class FakeController:
            def __init__(self, received_args):
                self.actions: list[float] = []
                instances["controller"] = self

            def start(self):
                pass

            def get_observation(self):
                return {"observation.state": np.asarray([len(self.actions)])}

            def send_action(self, action):
                self.actions.append(float(action[0]))
                if len(self.actions) == 3:
                    three_actions_sent.set()
                return action

            def stop(self):
                pass

        class DelayedPolicy:
            output_chunk_size = 4

            def __init__(self, received_args):
                self.calls = 0

            def start(self):
                pass

            def infer(self, inputs):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "action_chunk": np.arange(4, dtype=np.float32)[:, None]
                    }
                if not three_actions_sent.wait(timeout=1.0):
                    raise AssertionError("control loop did not continue")
                time.sleep(0.001)
                return {
                    "action_chunk": np.arange(100, 104, dtype=np.float32)[:, None]
                }

            def stop(self):
                pass

        scheduler = TemporalEnsembleScheduler()
        scheduler.run(FakeController, DelayedPolicy, args=args)

        np.testing.assert_allclose(
            instances["controller"].actions,
            [0.0, 1.0, 2.0, 52.5, 103.0],
        )
        self.assertEqual(scheduler.last_run_stats["stale_predictions"], 2)
        self.assertEqual(scheduler.last_run_stats["hold_ticks"], 0)

    def test_trace_records_raw_overlap_and_ensembled_action(self):
        with tempfile.TemporaryDirectory() as log_dir:
            args = SimpleNamespace(
                control_hz=100.0,
                temporal_ensemble_coeff=0.01,
                max_chunks=2,
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
                    return action

                def stop(self):
                    pass

            class FakePolicy:
                output_chunk_size = 2

                def __init__(self, received_args):
                    self.calls = 0

                def start(self):
                    pass

                def infer(self, inputs):
                    self.calls += 1
                    base = float(self.calls * 10)
                    return {
                        "action_chunk": np.asarray(
                            [[base], [base + 1.0]], dtype=np.float32
                        )
                    }

                def stop(self):
                    pass

            scheduler = TemporalEnsembleScheduler()
            scheduler.run(FakeController, FakePolicy, args=args)

            records = [
                json.loads(line)
                for line in Path(scheduler.last_log_path).read_text().splitlines()
            ]
            updates = [r for r in records if r["event"] == "chunk_ensembled"]
            self.assertEqual(len(updates), 2)
            overlap = updates[1]["first_overlap"]
            self.assertEqual(overlap["old_action"], [11.0])
            self.assertEqual(overlap["new_action"], [20.0])
            self.assertEqual(overlap["contributors"], 2)
            self.assertTrue(any(r["event"] == "action" for r in records))


if __name__ == "__main__":
    unittest.main()
