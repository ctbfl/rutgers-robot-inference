"""Transport and input validation without OpenPI, GPUs, or a robot."""

import subprocess
import sys
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ruri.common import zmq as codec
from ruri.server.serve import handle_request
from ruri.server.wrappers.pi05.pi05_effort import Pi05EffortWrapper, Pi05TrainRTCEffortWrapper


class FakePolicy:
    def __init__(self):
        self._model = SimpleNamespace(action_horizon=30, delay_probs=[1] * 8)
        self.calls = []

    def infer(self, observation):
        self.calls.append(observation)
        return {"actions": np.zeros((30, 7), dtype=np.float32)}


class EffortWrapperTests(unittest.TestCase):
    def make_wrapper(self, cls=Pi05EffortWrapper, **kwargs):
        with patch.object(cls, "_load_policy", return_value=FakePolicy()):
            return cls("unused", default_prompt="row1 task", warmup=False, **kwargs)

    def observation(self):
        return {
            "observation.state": np.arange(7, dtype=np.float32),
            "observation.joint_effort": np.array([1, 2, 3, 4, 5, 6, -0.4], dtype=np.float32),
            "observation.images.top": np.zeros((16, 20, 3), dtype=np.uint8),
            "observation.images.wrist": np.zeros((16, 20, 3), dtype=np.uint8),
        }

    def test_lazy_import(self):
        subprocess.run([sys.executable, "-c", "import sys; import ruri.server.wrappers.pi05.pi05_effort; assert 'jax' not in sys.modules; assert 'pi05_effort' not in sys.modules"], check=True)

    def test_wire_mapping_metadata_and_seven_actions(self):
        for cls in (Pi05EffortWrapper, Pi05TrainRTCEffortWrapper):
            with self.subTest(cls=cls):
                wrapper = self.make_wrapper(cls)
                request = self.observation()
                response = codec.unpack(codec.pack(handle_request(wrapper, request)))
                self.assertEqual(response["action_chunk"].shape, (30, 7))
                sent = wrapper.policy.calls[-1]
                np.testing.assert_array_equal(sent["observation/joint_effort"], request["observation.joint_effort"])
                self.assertEqual(sent["observation/state"].shape, (7,))
                self.assertEqual(sent["prompt"], "row1 task")
                description = wrapper.describe()
                self.assertIn("observation.joint_effort", description["inputs"])
                self.assertEqual(description["outputs"]["output_chunk_size"], 30)
                self.assertEqual(description["policy"]["supports_rtc"], cls is Pi05TrainRTCEffortWrapper)

    def test_missing_force_is_not_filled_with_zero(self):
        wrapper = self.make_wrapper()
        request = self.observation()
        del request["observation.joint_effort"]
        with self.assertRaisesRegex(KeyError, "observation.joint_effort"):
            wrapper.infer(request)
        self.assertEqual(wrapper.policy.calls, [])

    def test_checkpoint_preflight_rejects_missing_effort_wrong_rtc_and_partial_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "params").mkdir()
            asset = root / "assets/tight_insertion_row1_v21"
            asset.mkdir(parents=True)
            (asset / "norm_stats.json").write_text("{}")
            marker = root / "_CHECKPOINT_METADATA"
            marker.write_text(json.dumps({"commit_timestamp_nsecs": 1}))
            metadata_path = root / "params/_METADATA"

            def metadata(layers):
                return {"tree_metadata": {
                    str((layer, name)): {"key_metadata": [{"key": k} for k in (
                        "params", layer, name, "value",
                    )]}
                    for layer in layers for name in ("kernel", "bias")
                }}

            wrapper = self.make_wrapper()
            wrapper.checkpoint_path = root
            metadata_path.write_text(json.dumps(metadata(["time_mlp_in"])))
            with self.assertRaisesRegex(ValueError, "Not an effort checkpoint"):
                wrapper._check_checkpoint()
            metadata_path.write_text(json.dumps(metadata([
                "effort_mlp_in", "effort_mlp_out", "tok_time_proj",
            ])))
            with self.assertRaisesRegex(ValueError, "Pi05TrainRTCEffortWrapper"):
                wrapper._check_checkpoint()
            metadata_path.write_text(json.dumps(metadata(["effort_mlp_in", "effort_mlp_out"])))
            wrapper._check_checkpoint()
            marker.write_text(json.dumps({"commit_timestamp_nsecs": None}))
            with self.assertRaisesRegex(ValueError, "has not committed"):
                wrapper._check_checkpoint()

    def test_invalid_effort_and_double_appended_state(self):
        wrapper = self.make_wrapper()
        for key, value in (
            ("observation.joint_effort", np.zeros(1)),
            ("observation.joint_effort", np.zeros((1, 7))),
            ("observation.joint_effort", [0] * 6 + [float("nan")]),
            ("observation.state", np.zeros(8)),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    wrapper.infer({**self.observation(), key: value})

    def test_rtc_context_still_aligns_seven_dimensional_actions(self):
        wrapper = self.make_wrapper(Pi05TrainRTCEffortWrapper)
        previous = np.ones((25, 7), dtype=np.float32)
        with patch.object(wrapper, "_infer_conditioned", return_value=(np.ones((30, 7)), 1.0)) as call:
            response = wrapper.infer({
                **self.observation(),
                "context.rtc.prev_chunk_left_over": previous,
                "context.rtc.consumed_steps": 5,
                "context.rtc.estimated_inference_delay_steps": 4,
            })
        self.assertTrue(response["rtc.applied"])
        self.assertEqual(response["rtc.execute_from"], 4)
        self.assertIn("observation/joint_effort", call.call_args.args[0])
        self.assertEqual(call.call_args.args[1].shape, (30, 7))
        np.testing.assert_array_equal(previous, np.ones((25, 7)))

    def test_warmup_supplies_effort_to_both_rtc_paths(self):
        wrapper = self.make_wrapper(Pi05TrainRTCEffortWrapper)
        with patch.object(wrapper, "_infer_conditioned", return_value=(np.zeros((30, 7)), 1.0)) as call:
            wrapper.warmup()
        self.assertIn("observation/joint_effort", wrapper.policy.calls[-1])
        self.assertIn("observation/joint_effort", call.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
