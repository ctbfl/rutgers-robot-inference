"""Optional CPU preprocessing tests in the FastWAM environment; no model load."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(os.environ.get("FASTWAM_ROOT", "/common/users/jh2400/FastWAM"))
AVAILABLE = (ROOT / "src/fastwam").is_dir() and all(
    importlib.util.find_spec(name) is not None for name in ("torch", "hydra", "torchvision")
)


@unittest.skipUnless(AVAILABLE, "requires the optional FastWAM environment")
class FastWAMRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_path = list(sys.path)
        sys.path.insert(0, str(ROOT / "src"))
        import torch
        from omegaconf import OmegaConf
        from ruri.server.wrappers.fastwam._runtime import FastWAMRuntime, DEFAULT_PROMPT

        cls.torch = torch
        cls.omega = OmegaConf
        cls.runtime_type = FastWAMRuntime
        cls.template = DEFAULT_PROMPT
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        sys.path[:] = cls.old_path
        cls.torch.set_num_threads(cls.old_threads)

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        shape = {
            "images": [{"key": k, "raw_shape": [3, 32, 40], "shape": [3, 16, 16]} for k in ("top", "hand")],
            "state": [{"key": "default", "raw_shape": 7, "shape": 7}],
            "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        }
        mask = {"default": [True] * 6 + [False]}
        image_transforms = [{"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"},
                            {"_target_": "torchvision.transforms.Resize", "size": [16, 16]}]
        self.cfg = self.omega.create({
            "model": {"_target_": "fastwam.runtime.create_fastwam", "proprio_dim": 7,
                      "action_dit_config": {"action_dim": 7}, "load_text_encoder": False,
                      "video_dit_config": {"video_attention_mask_mode": "first_frame_causal", "text_dim": 4}},
            "data": {"train": {"num_frames": 33, "action_video_freq_ratio": 4, "global_sample_stride": 1,
                "concat_multi_camera": "horizontal", "video_size": [16, 32],
                "text_embedding_cache_dir": str(self.folder), "context_len": 8, "use_text_embed_cache": True,
                "processor": {
                    "_target_": "fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor",
                    "shape_meta": shape, "num_obs_steps": 33, "num_output_cameras": 2,
                    "action_output_dim": 7, "proprio_output_dim": 7, "use_stepwise_action_norm": False,
                    "norm_default_mode": "q01/q99", "norm_exception_mode": None,
                    "delta_action_dim_mask": mask,
                    "action_state_transforms": [{"_target_": "fastwam.datasets.lerobot.transforms.relative_action.MaskedRelativeJointTransform",
                                                 "keys": ["default"], "dim_mask": mask}],
                    "action_state_merger": {"_target_": "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign"},
                    "train_transforms": image_transforms, "val_transforms": image_transforms,
                }}},
        })
        stats = {
            "state": {"default": {"global_q01": [-10.] * 7, "global_q99": [10.] * 7}},
            "action": {"default": {"global_q01": [-2.] * 6 + [0.], "global_q99": [2.] * 6 + [100.]}},
        }
        self.config_path = self.folder / "config.yaml"
        self.stats_path = self.folder / "dataset_stats.json"
        self.stats_path.write_text(json.dumps(stats))
        self.checkpoint = self.folder / "checkpoint.pt"
        self.torch.save({"dit": {}}, self.checkpoint)

    def runtime(self):
        self.omega.save(self.cfg, self.config_path)
        return self.runtime_type(checkpoint_path=self.checkpoint, root=ROOT, config_path=self.config_path,
                                 stats_path=self.stats_path, text_cache_dir=None, model_base_path=None, device="cpu")

    def test_preprocessing_and_inverse_no_crop_no_cumsum_no_gripper_rebase(self):
        r = self.runtime()
        state = np.array([1000, 2, 3, 4, 5, 6, 75], dtype=np.float32)
        images = {"top": np.zeros((32, 40, 3), dtype=np.uint8),
                  "hand": np.full((32, 40, 3), 255, dtype=np.uint8)}
        image, proprio = r.prepare(state, images)
        self.assertEqual(tuple(image.shape), (1, 3, 16, 32))
        self.assertTrue((image[..., :16] == -1).all())
        self.torch.testing.assert_close(image[..., 16:], self.torch.ones_like(image[..., 16:]), rtol=0, atol=1e-6)
        self.assertEqual(proprio[0].item(), 5)  # native input clamp
        normalized = self.torch.zeros(32, 7)
        normalized[:, :6] = .5  # each joint delta is +1, NOT cumulative
        result = r.restore_action(normalized, state)
        np.testing.assert_allclose(result[:, :6], np.tile(state[:6] + 1, (32, 1)))
        np.testing.assert_allclose(result[:, 6], 50)  # not 125
        self.assertEqual(result.shape, (32, 7))
        self.assertGreater(result[0, 0], 100)  # hardware clipping is not ours

    def test_context_zero_padding_all_true_mask_and_prefix(self):
        r = self.runtime()
        task = "test task"
        prompt = self.template.format(task=task)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        path = self.folder / f"{digest}.t5_len8.wan22ti2v5b.pt"
        self.torch.save({"context": self.torch.ones(8, 4), "mask": self.torch.tensor([True] * 3 + [False] * 5)}, path)
        context, mask = r.text_context(task)
        self.assertTrue(mask.all())
        self.assertTrue((context[:3] == 1).all())
        self.assertTrue((context[3:] == 0).all())
        self.assertIs(r.text_context(prompt)[0], context)
        with self.assertRaisesRegex(ValueError, "Uncached"):
            r.text_context("unknown")

    def test_rejects_wrong_camera_mask_horizon_and_model(self):
        mutations = [
            ("data.train.processor.shape_meta.images.0.key", "wrist"),
            ("data.train.processor.action_state_transforms.0.dim_mask.default.6", True),
            ("data.train.num_frames", 32),
            ("data.train.global_sample_stride", 2),
            ("model._target_", "fastwam.runtime.create_fastwam_idm"),
        ]
        original = self.omega.to_container(self.cfg)
        for key, value in mutations:
            with self.subTest(key=key):
                self.cfg = self.omega.create(original)
                self.omega.update(self.cfg, key, value)
                with self.assertRaises(ValueError):
                    self.runtime()

    def test_horizon_is_config_derived(self):
        self.cfg.data.train.num_frames = 17
        self.cfg.data.train.processor.num_obs_steps = 17
        self.assertEqual(self.runtime().horizon, 16)

    def test_missing_assets_and_legacy_checkpoint_fail_before_model_load(self):
        with self.assertRaisesRegex(ValueError, "complete FastWAM"):
            self.runtime().load_model()
        self.checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            self.runtime()

    def test_compile_flag_reaches_native_inference(self):
        r = self.runtime()
        model = SimpleNamespace(infer_action=Mock(return_value={"action": self.torch.zeros(32, 7)}))
        r.model = model
        images = {k: np.zeros((32, 40, 3), dtype=np.uint8) for k in ("top", "hand")}
        context = (self.torch.zeros(8, 4), self.torch.ones(8, dtype=self.torch.bool))
        for compiled in (False, True):
            r.compile_action_infer = compiled
            with patch.object(r, "text_context", return_value=context):
                action, _ = r.predict(np.zeros(7, dtype=np.float32), images, "task", steps=20, seed=42)
            self.assertIs(model.infer_action.call_args.kwargs["compile_action_infer"], compiled)
            self.assertEqual(action.shape, (32, 7))


if __name__ == "__main__":
    unittest.main()
