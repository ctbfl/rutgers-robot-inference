"""Torque-history wrapper contract, without OpenPI, GPUs, or a robot."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ruri.common import zmq as codec
from ruri.server.serve import handle_request
from ruri.server.wrappers.pi05 import pi05_torque as mod
from ruri.server.wrappers.pi05.pi05_torque import (
    EFFORT_HISTORY, EFFORT_TIMES, HISTORY_OFFSETS_S, Pi05TrainRTCTorqueWrapper,
    select_effort_history,
)


class FakePolicy:
    def __init__(self):
        self._model = SimpleNamespace(action_horizon=30, delay_probs=[1] * 8)
        self.calls = []

    def infer(self, observation):
        self.calls.append(observation)
        return {"actions": np.zeros((30, 7), dtype=np.float32)}


def buffer(rows=60, hz=30.0, start=1000.0):
    """Row i holds the value i in every channel, so a selected row reads as its index."""
    history = np.repeat(np.arange(rows, dtype=np.float32)[:, None], 7, axis=1)
    times = start + np.arange(rows, dtype=np.float64) / hz
    return history, times


class SelectionTests(unittest.TestCase):
    def test_uniform_30hz_buffer_reproduces_dataset_offsets(self):
        history, times = buffer(60)
        selected, info = select_effort_history(history, times)
        self.assertEqual(selected.shape, (10, 7))
        self.assertEqual(selected.dtype, np.float32)
        np.testing.assert_array_equal(selected[:, 0], 59 + np.arange(-54, 1, 6))
        self.assertEqual(info["torque.slots_from_oldest_row"], 0)
        self.assertLess(info["torque.max_hold_ms"], 1.0)

    def test_short_buffer_repeats_oldest_row_like_episode_start(self):
        history, times = buffer(4)  # 0.1 s into an episode
        selected, info = select_effort_history(history, times)
        # dataset frame 3: offsets -54..-6 clamp to frame 0, offset 0 is frame 3
        np.testing.assert_array_equal(selected[:, 0], [0] * 9 + [3])
        self.assertEqual(info["torque.slots_from_oldest_row"], 9)

    def test_single_row(self):
        history, times = buffer(1)
        selected, info = select_effort_history(history, times)
        np.testing.assert_array_equal(selected[:, 0], np.zeros(10))
        self.assertEqual(info["torque.slots_from_oldest_row"], 9)

    def test_faster_jittered_buffer_holds_the_latest_row_at_each_slot(self):
        rng = np.random.default_rng(0)
        times = np.cumsum(rng.uniform(0.008, 0.012, 400)) + 5.0  # ~100 Hz, jittery
        # Encode the row index at 0.25 per row so values stay inside the unit guard.
        history = np.repeat((np.arange(400, dtype=np.float32) * 0.25)[:, None], 7, axis=1)
        selected, info = select_effort_history(history, times)
        for slot, offset in enumerate(HISTORY_OFFSETS_S):
            target = times[-1] + offset
            expected = np.flatnonzero(times <= target + 1e-3)[-1]
            self.assertEqual(int(round(selected[slot, 0] / 0.25)), expected)
        self.assertLess(info["torque.max_hold_ms"], 13.0)

    def test_sparse_buffer_is_reported(self):
        # 4 Hz does not line up with the 0.2 s grid, so slots hold rows up to 150 ms old.
        history, times = buffer(10, hz=4.0)
        _, info = select_effort_history(history, times)
        self.assertGreater(info["torque.max_hold_ms"], 100.0)
        # 5 Hz lands exactly on the grid: nothing is stale.
        history, times = buffer(10, hz=5.0)
        _, info = select_effort_history(history, times)
        self.assertLess(info["torque.max_hold_ms"], 1.0)

    def test_rejections(self):
        history, times = buffer(20)
        cases = {
            "single frame": (np.zeros(7, np.float32), times[:1], "buffered history"),
            "wrong width": (np.zeros((20, 6), np.float32), times, r"\(N, 7\)"),
            "empty": (np.zeros((0, 7), np.float32), times[:0], r"N >= 1"),
            "nan": (np.where(history == 3, np.nan, history), times, "NaN"),
            "units": (history * 1000, times, "N\\*m"),
            "int history": (history.astype(np.int64), times, "float array"),
            "list": (history.tolist(), times, "np.ndarray"),
            "length mismatch": (history, times[:-1], "one entry per history row"),
            "not increasing": (history, np.concatenate([times[:5], times[4:19]]), "strictly increasing"),
            "int ns": (history, (times * 1e9).astype(np.int64), "float64"),
            "float32 s": (history, times.astype(np.float32), "float64"),
        }
        for name, (h, t, pattern) in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex((ValueError, TypeError), pattern):
                    select_effort_history(h, t)


class WrapperTests(unittest.TestCase):
    def make_wrapper(self, **kwargs):
        with patch.object(Pi05TrainRTCTorqueWrapper, "_load_policy", return_value=FakePolicy()):
            return Pi05TrainRTCTorqueWrapper("unused", warmup=False, **kwargs)

    def request(self, **overrides):
        history, times = buffer(60)
        request = {
            "observation.state": np.arange(7, dtype=np.float32),
            "observation.images.top": np.zeros((16, 20, 3), dtype=np.uint8),
            "observation.images.wrist": np.zeros((16, 20, 3), dtype=np.uint8),
            EFFORT_HISTORY: history,
            EFFORT_TIMES: times,
            "prompt": "C3",
        }
        request.update(overrides)
        return request

    def test_lazy_import(self):
        subprocess.run([sys.executable, "-c", (
            "import sys; import ruri.server.wrappers.pi05.pi05_torque; "
            "assert 'jax' not in sys.modules; assert 'pi05_torque' not in sys.modules"
        )], check=True)

    def test_offsets_match_the_model_package_when_it_is_importable(self):
        package = Path(mod.TORQUE_PACKAGE)
        if not (package / "pi05_torque/spec.py").is_file():
            self.skipTest("pi05-torque not on this machine")
        source = (package / "pi05_torque/spec.py").read_text()
        namespace: dict = {}
        exec(compile(source, "spec.py", "exec"), namespace)
        expected = tuple(o / namespace["FPS"] for o in namespace["HISTORY_OFFSETS"])
        np.testing.assert_allclose(expected, HISTORY_OFFSETS_S)

    def test_over_the_wire(self):
        wrapper = self.make_wrapper()
        response = codec.unpack(codec.pack(handle_request(wrapper, codec.unpack(codec.pack(self.request())))))
        self.assertEqual(response["action_chunk"].shape, (30, 7))
        self.assertEqual(response["prompt"], "C3")
        self.assertEqual(response["torque.slots_from_oldest_row"], 0)
        self.assertEqual(response["torque.history_rows"], 60)
        sent = wrapper.policy.calls[-1]
        self.assertEqual(sent["observation/joint_effort"].shape, (10, 7))
        np.testing.assert_array_equal(sent["observation/joint_effort"][:, 0], 59 + np.arange(-54, 1, 6))
        self.assertNotIn("observation/joint_effort_history", sent)
        self.assertEqual(sent["observation/state"].shape, (7,))

    def test_metadata_is_the_client_contract(self):
        wrapper = self.make_wrapper()
        description = codec.unpack(codec.pack(handle_request(wrapper, {"type": "metadata"})))
        json.dumps(description)  # printable by list_policy_servers.py
        inputs = description["inputs"]
        for key in ("observation.state", "observation.images.top", "observation.images.wrist",
                    "prompt", EFFORT_HISTORY, EFFORT_TIMES,
                    "context.rtc.prev_chunk_left_over", "context.rtc.consumed_steps",
                    "context.rtc.estimated_inference_delay_steps"):
            self.assertIn(key, inputs)
        for key in ("observation/state", "observation/joint_effort_history", "observation/joint_effort"):
            self.assertNotIn(key, inputs)
        self.assertEqual(inputs[EFFORT_HISTORY]["shape"], ["N", 7])
        self.assertEqual(inputs[EFFORT_TIMES]["dtype"], "float64")
        self.assertEqual(len(inputs["prompt"]["allowed_values"]), 25)
        outputs = description["outputs"]
        self.assertEqual(outputs["output_chunk_size"], 30)
        self.assertEqual(outputs["action_dim"], 7)
        self.assertIn("torque.slots_from_oldest_row", outputs["response_keys"])
        policy = description["policy"]
        self.assertEqual(policy["name"], "pi05_train_rtc_torque")
        self.assertTrue(policy["supports_rtc"])
        self.assertEqual(policy["torque"]["history_offsets_frames_at_30hz"], list(range(-54, 1, 6)))
        self.assertGreaterEqual(len(policy["client_guide"]), 8)

    def test_prompt_is_required_and_must_be_a_trained_string(self):
        wrapper = self.make_wrapper()
        request = self.request()
        del request["prompt"]
        with self.assertRaisesRegex(KeyError, "target hole"):
            wrapper.infer(request)
        for bad in ("c3", "C5", "insert into C3", "F0"):
            with self.subTest(bad):
                with self.assertRaisesRegex(ValueError, "not a trained task string"):
                    wrapper.infer(self.request(prompt=bad))
        self.assertEqual(wrapper.infer(self.request(prompt=" E4 "))["prompt"], "E4")
        self.assertEqual(wrapper.policy.calls[-1]["prompt"], "E4")
        with self.assertRaisesRegex(ValueError, "no default prompt"):
            self.make_wrapper(default_prompt="A0")

    def test_missing_or_single_frame_effort(self):
        wrapper = self.make_wrapper()
        request = self.request()
        del request[EFFORT_HISTORY], request[EFFORT_TIMES]
        with self.assertRaisesRegex(KeyError, EFFORT_HISTORY):
            wrapper.infer(request)
        with self.assertRaisesRegex(KeyError, "single 'observation.joint_effort'"):
            wrapper.infer({**request, "observation.joint_effort": np.zeros(7, np.float32)})
        request = self.request()
        del request[EFFORT_TIMES]
        with self.assertRaisesRegex(KeyError, EFFORT_TIMES):
            wrapper.infer(request)
        self.assertEqual(wrapper.policy.calls, [])

    def test_bad_state(self):
        wrapper = self.make_wrapper()
        for state in (np.zeros(8, np.float32), np.array([0] * 6 + [np.nan], np.float32)):
            with self.subTest(state=state):
                with self.assertRaisesRegex(ValueError, "seven finite"):
                    wrapper.infer(self.request(**{"observation.state": state}))

    def test_clipped_values_are_reported(self):
        wrapper = self.make_wrapper()
        wrapper._effort_q01 = np.full(7, -1.0)
        wrapper._effort_q99 = np.full(7, 1.0)
        history, times = buffer(60)  # values up to 59, far beyond a +-1 range
        response = wrapper.infer(self.request(**{EFFORT_HISTORY: history}))
        self.assertGreater(response["torque.clipped_values"], 0)
        response = wrapper.infer(self.request(**{EFFORT_HISTORY: np.zeros_like(history)}))
        self.assertEqual(response["torque.clipped_values"], 0)

    def test_rtc_context_carries_history_and_does_not_mutate_the_request(self):
        wrapper = self.make_wrapper()
        previous = np.ones((22, 7), dtype=np.float32)
        request = self.request(**{
            "context.rtc.prev_chunk_left_over": previous,
            "context.rtc.consumed_steps": 8,
            "context.rtc.estimated_inference_delay_steps": 4,
        })
        history_before = request[EFFORT_HISTORY].copy()
        with patch.object(wrapper, "_infer_conditioned", return_value=(np.ones((30, 7)), 1.0)) as call:
            response = wrapper.infer(request)
        self.assertTrue(response["rtc.applied"])
        self.assertEqual(response["rtc.execute_from"], 4)
        self.assertEqual(response["torque.slots_from_oldest_row"], 0)
        self.assertEqual(call.call_args.args[0]["observation/joint_effort"].shape, (10, 7))
        self.assertEqual(call.call_args.args[1].shape, (30, 7))
        np.testing.assert_array_equal(previous, np.ones((22, 7)))
        np.testing.assert_array_equal(request[EFFORT_HISTORY], history_before)

    def test_warmup_supplies_history_to_both_paths(self):
        wrapper = self.make_wrapper()
        with patch.object(wrapper, "_infer_conditioned", return_value=(np.zeros((30, 7)), 1.0)) as call:
            wrapper.warmup()
        self.assertEqual(wrapper.policy.calls[-1]["observation/joint_effort"].shape, (10, 7))
        self.assertEqual(call.call_args.args[0]["observation/joint_effort"].shape, (10, 7))

    def test_checkpoint_preflight(self):
        layers = list(mod.REQUIRED_PARAMS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "params").mkdir()
            asset = root / f"assets/{mod.ASSET_ID}"
            asset.mkdir(parents=True)

            def write(present, committed=True, state_dim=77):
                leaves = {
                    f"{layer}.{p}": {"key_metadata": [{"key": k} for k in ("params", layer, p, "value")]}
                    for layer in present for p in ("kernel", "bias")
                }
                (root / "params/_METADATA").write_text(json.dumps({"tree_metadata": leaves}))
                (root / "_CHECKPOINT_METADATA").write_text(
                    json.dumps({"commit_timestamp_nsecs": 1 if committed else None}))
                q = lambda n: {"q01": [0] * n, "q99": [1] * n}  # noqa: E731
                (asset / "norm_stats.json").write_text(
                    json.dumps({"norm_stats": {"state": q(state_dim), "actions": q(14)}}))

            wrapper = self.make_wrapper()
            wrapper.checkpoint_path = root
            write(layers)
            wrapper._check_checkpoint()
            for missing in layers:
                with self.subTest(missing=missing):
                    write([layer for layer in layers if layer != missing])
                    with self.assertRaisesRegex(ValueError, f"missing {missing}"):
                        wrapper._check_checkpoint()
            write(layers, committed=False)
            with self.assertRaisesRegex(ValueError, "has not committed"):
                wrapper._check_checkpoint()
            write(layers, state_dim=8)
            with self.assertRaisesRegex(ValueError, "not the torque layout"):
                wrapper._check_checkpoint()


if __name__ == "__main__":
    unittest.main()
