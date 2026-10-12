"""Checkpoint and wire-contract regression tests without OpenPI or a GPU."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ruri.common import zmq as codec
from ruri.server.serve import handle_request
from ruri.server.wrappers.pi05.pi05_sim_real import Pi05SimRealWrapper, SCHEMA, check_checkpoint
from ruri.server.wrappers.pi05.pi05 import Pi05Wrapper
from ruri.server.wrappers.pi05.pi05_train_rtc import Pi05TrainRTCWrapper
from ruri.server.wrappers.pi05.pi05_rtc import Pi05RTCWrapper


def make_checkpoint(root):
    (root / "params").mkdir()
    (root / "assets/sim_real_h30").mkdir(parents=True)
    stats = {key: {"mean": [0.] * 7, "std": [1.] * 7, "q01": [-2.] * 7, "q99": [2.] * 7}
             for key in ("state", "actions")}
    raw = json.dumps({"norm_stats": stats}).encode()
    (root / "assets/sim_real_h30/norm_stats.json").write_bytes(raw)
    spec = {
        "schema": SCHEMA,
        "model_class": "tight_insertion.sim_real.model.DomainRTCConfig",
        "observation_class": "tight_insertion.sim_real.model.DomainObservation",
        "model_config": {"action_dim": 32, "action_horizon": 30, "max_token_len": 200,
                         "dtype": "bfloat16", "paligemma_variant": "gemma_2b",
                         "action_expert_variant": "gemma_300m", "pi05": True,
                         "discrete_state_input": False, "delay_probs": [0.125] * 8},
        "indicator": {"encoding": {"sim": [1, 0], "real": [0, 1]}, "normalize_indicator": False,
                      "diffuse_indicator": False, "classifier_free_guidance": False, "domain_dropout": 0},
        "data_contract": {"physical_state_dim": 7, "physical_action_dim": 7, "model_action_dim": 32,
                          "fps": 30, "cameras": ["observation.images.top", "observation.images.hand"]},
        "inference_flow_steps": 10, "checkpoint_norm_sha256": hashlib.sha256(raw).hexdigest(),
    }
    (root / "model_spec.json").write_text(json.dumps(spec))
    leaves = [("domain_proj", "kernel"), ("tok_time_proj", "kernel"), ("tok_time_proj", "bias")]
    tree = {str((layer, name)): {
        "key_metadata": [{"key": x} for x in ("params", layer, name, "value")],
        "value_metadata": {"value_type": "np.ndarray"},
    } for layer, name in leaves}
    (root / "params/_METADATA").write_text(json.dumps({"tree_metadata": tree}))
    (root / "params/_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 1}))
    return spec


class FakePolicy:
    def __init__(self):
        self._model = SimpleNamespace(action_horizon=30, delay_probs=[.125] * 8)
        self.calls = []

    def infer(self, observation, **kwargs):
        self.calls.append((observation, kwargs))
        return {"actions": np.zeros((30, 7), np.float32), "policy_timing": {"infer_ms": 1.}}


class SimRealWrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = make_checkpoint(self.root)

    def make_wrapper(self, **kwargs):
        with patch.object(Pi05SimRealWrapper, "_load_policy", return_value=FakePolicy()):
            return Pi05SimRealWrapper(self.root, domain="real", default_prompt="insert A0",
                                      warmup=False, **kwargs)

    def observation(self):
        return {"observation.state": np.arange(7, dtype=np.float32),
                "observation.images.top": np.zeros((16, 20, 3), np.uint8),
                "observation.images.wrist": np.zeros((16, 20, 3), np.uint8)}

    def test_import_and_checkpoint_preflight_do_not_import_jax(self):
        code = ("import sys; from ruri.server.wrappers.pi05.pi05_sim_real import check_checkpoint; "
                "check_checkpoint(sys.argv[1]); assert 'jax' not in sys.modules")
        subprocess.run([sys.executable, "-c", code, str(self.root)], check=True)

    def test_unchanged_wire_contract(self):
        wrapper = self.make_wrapper()
        response = codec.unpack(codec.pack(handle_request(wrapper, self.observation())))
        self.assertEqual(response["action_chunk"].shape, (30, 7))
        self.assertFalse(response["rtc.applied"])
        metadata = codec.unpack(codec.pack(wrapper.describe()))
        self.assertEqual(metadata["outputs"]["output_chunk_size"], 30)
        self.assertTrue(metadata["policy"]["supports_rtc"])
        self.assertEqual(metadata["policy"]["domain_one_hot"], [0, 1])
        self.assertNotIn("domain_one_hot", metadata["inputs"])
        self.assertNotIn("action_valid_mask", metadata["inputs"])
        self.assertIn("observation.images.wrist", metadata["inputs"])

    def test_rtc_alignment_and_server_domain(self):
        wrapper = self.make_wrapper()
        previous = np.ones((25, 7), np.float32)
        response = wrapper.infer({**self.observation(), "domain_one_hot": [1, 0],
            "context.rtc.prev_chunk_left_over": previous, "context.rtc.consumed_steps": 5,
            "context.rtc.estimated_inference_delay_steps": 4})
        observation, kwargs = wrapper.policy.calls[-1]
        self.assertNotIn("domain_one_hot", observation)
        self.assertEqual(kwargs["prev_chunk"].shape, (30, 7))
        self.assertEqual(kwargs["delay"], 4)
        self.assertEqual(response["rtc.execute_from"], 4)
        self.assertEqual(response["rtc.method"], "training-time")
        np.testing.assert_array_equal(previous, np.ones((25, 7)))
        with self.assertRaisesRegex(ValueError, "alignment"):
            wrapper.infer({**self.observation(), "context.rtc.prev_chunk_left_over": previous,
                           "context.rtc.consumed_steps": 4})

    def test_warmup_exercises_both_paths(self):
        wrapper = self.make_wrapper()
        wrapper.warmup()
        self.assertEqual(len(wrapper.policy.calls), 2)
        self.assertEqual(wrapper.policy.calls[0][1], {})
        self.assertEqual(wrapper.policy.calls[1][1]["delay"], 4)

    def test_malformed_state_previous_and_output_rejected(self):
        wrapper = self.make_wrapper()
        for value in (np.zeros(8), np.zeros((1, 7)), [0] * 6 + [float("nan")]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                wrapper.infer({**self.observation(), "observation.state": value})
        with self.assertRaisesRegex(ValueError, "finite"):
            wrapper.infer({**self.observation(), "context.rtc.prev_chunk_left_over": np.full((30, 7), np.nan)})
        with patch.object(wrapper.policy, "infer", return_value={"actions": np.zeros((30, 32))}):
            with self.assertRaisesRegex(ValueError, "30, 7"):
                wrapper.infer(self.observation())

    def test_invalid_runtime_settings(self):
        for kwargs in ({"num_denoising_steps": 0}, {"default_inference_delay": 0},
                       {"default_inference_delay": 8}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_wrapper(**kwargs)
        with self.assertRaises(ValueError):
            Pi05SimRealWrapper(self.root, domain="robot", warmup=False)

    def test_plain_and_rtc_loaders_cannot_discard_domain_projection(self):
        with self.assertRaisesRegex(ValueError, "Pi05SimRealWrapper"):
            Pi05Wrapper(self.root, "baseline", warmup=False)
        with self.assertRaisesRegex(ValueError, "Pi05SimRealWrapper"):
            Pi05TrainRTCWrapper(self.root, warmup=False)
        with self.assertRaisesRegex(ValueError, "Pi05SimRealWrapper"):
            Pi05RTCWrapper(self.root, "baseline", warmup=False)

    def test_wrong_architecture_and_encoding_rejected(self):
        for section, key, value in (("model_config", "discrete_state_input", True),
                                    ("model_config", "action_horizon", 10),
                                    ("indicator", "encoding", {"real": [1, 0], "sim": [0, 1]})):
            spec = json.loads(json.dumps(self.spec))
            spec[section][key] = value
            (self.root / "model_spec.json").write_text(json.dumps(spec))
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_checkpoint(self.root)

    def test_norm_corruption_uncommitted_or_missing_parameters_rejected(self):
        path = self.root / "assets/sim_real_h30/norm_stats.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            check_checkpoint(self.root)
        path.write_bytes(original)
        commit = self.root / "params/_CHECKPOINT_METADATA"
        commit.write_text('{"commit_timestamp_nsecs": null}')
        with self.assertRaisesRegex(ValueError, "not committed"):
            check_checkpoint(self.root)
        commit.write_text('{"commit_timestamp_nsecs": 1}')
        (self.root / "params/_METADATA").write_text('{"tree_metadata": {}}')
        with self.assertRaisesRegex(ValueError, "domain_proj"):
            check_checkpoint(self.root)


if __name__ == "__main__":
    unittest.main()
