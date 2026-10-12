"""Backend-only code. Requires the FastWAM environment, not RURI base deps.

Only the reviewed two-camera, seven-dimensional masked-relative joint schema
is supported. Unknown training transforms fail closed instead of returning
well-shaped but incorrectly interpreted robot targets.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

logger = logging.getLogger(__name__)


def _file(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


class FastWAMRuntime:
    def __init__(self, *, checkpoint_path, root, config_path, stats_path,
                 text_cache_dir, model_base_path, device, compile_action_infer=False):
        self.root = root
        self.checkpoint_path = _file(checkpoint_path)
        # Standard training export: RUN/checkpoints/weights/step_NNNNNN.pt.
        run = self.checkpoint_path.parent.parent.parent
        self.config_path = _file(config_path or run / "config.yaml")
        self.stats_path = _file(stats_path or run / "dataset_stats.json")
        self.cfg = OmegaConf.load(self.config_path)
        self.data = self.cfg.data.train
        self._validate_config()
        self.horizon = int(self.data.num_frames) - 1
        self.device = torch.device(device)
        self.compile_action_infer = compile_action_infer
        if compile_action_infer and self.device.type != "cuda":
            raise ValueError("Compiled FastWAM inference requires a CUDA device")
        self.dtype = torch.bfloat16
        self.model_base = Path(model_base_path or root / "checkpoints").expanduser().resolve()
        cache = Path(text_cache_dir or self.data.text_embedding_cache_dir).expanduser()
        self.text_cache_dir = (cache if cache.is_absolute() else root / cache).resolve()
        self._text = {}  # bounded to one prompt; no cross-request image/action cache
        self.context_len = int(self.data.context_len)
        self.text_dim = int(self.cfg.model.video_dit_config.text_dim)
        self.processor = instantiate(self.data.processor).eval()
        stats = load_dataset_stats_from_json(str(self.stats_path))
        self.processor.set_normalizer_from_stats(stats)
        self.state_norm = self.processor.normalizer.normalizers["state"]["default"]
        self.action_norm = self.processor.normalizer.normalizers["action"]["default"]
        for norm in (self.state_norm, self.action_norm):
            if norm.scale.shape != (7,) or norm.offset.shape != (7,):
                raise ValueError("Expected seven-dimensional global normalization statistics")
            if not torch.isfinite(norm.scale).all() or not torch.isfinite(norm.offset).all() or (norm.scale <= 0).any():
                raise ValueError("Invalid normalization statistics")
        size = {"img_h": int(self.data.video_size[0]), "img_w": int(self.data.video_size[1])}
        self.image_finish = (
            ResizeSmallestSideAspectPreserving(size), CenterCrop(size),
            Normalize({"mean": 0.5, "std": 0.5}),
        )
        self.model = None
        self.checkpoint_step = None

    def _validate_config(self):
        d, m = self.data, self.cfg.model
        p = d.processor
        expected_mask = [True] * 6 + [False]
        if m._target_ != "fastwam.runtime.create_fastwam":
            raise ValueError("Only the standard FastWAM action-only backend is supported")
        if m.video_dit_config.video_attention_mask_mode != "first_frame_causal":
            raise ValueError("Action-only inference requires first_frame_causal video attention")
        if m.proprio_dim != 7 or m.action_dit_config.action_dim != 7:
            raise ValueError("Only seven-dimensional Piper checkpoints are supported")
        if d.concat_multi_camera != "horizontal" or d.get("camera_key") is not None:
            raise ValueError("Expected horizontal top/hand camera concatenation")
        if [x.key for x in p.shape_meta.images] != ["top", "hand"] or p.num_output_cameras != 2:
            raise ValueError("Expected exactly two cameras in top, hand order")
        for field in ("state", "action"):
            entries = p.shape_meta[field]
            if len(entries) != 1 or entries[0].key != "default" or entries[0].shape != 7 or entries[0].raw_shape != 7:
                raise ValueError(f"Unsupported {field} schema")
        transforms = p.action_state_transforms
        if (len(transforms) != 1
                or transforms[0]._target_ != "fastwam.datasets.lerobot.transforms.relative_action.MaskedRelativeJointTransform"
                or list(transforms[0]["keys"]) != ["default"]
                or list(transforms[0].dim_mask.default) != expected_mask
                or list(p.delta_action_dim_mask.default) != expected_mask):
            raise ValueError("Expected six relative joints and one absolute gripper")
        if p.action_state_merger._target_ != "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign":
            raise ValueError("Unsupported action/state merger")
        if p.use_stepwise_action_norm or p.action_output_dim != 7 or p.proprio_output_dim != 7:
            raise ValueError("Expected global seven-dimensional action/state normalization")
        # Deliberately restrict image transforms to the reviewed serving path.
        expected = ["fastwam.datasets.lerobot.transforms.image.ToTensor", "torchvision.transforms.Resize"]
        if [t._target_ for t in p.val_transforms] != expected:
            raise ValueError("Unsupported validation image transforms")
        horizon = int(d.num_frames) - 1
        ratio = int(d.action_video_freq_ratio)
        if (isinstance(d.num_frames, bool) or not isinstance(d.num_frames, int)
                or isinstance(d.action_video_freq_ratio, bool) or not isinstance(d.action_video_freq_ratio, int)
                or horizon <= 0 or ratio <= 0 or horizon % (4 * ratio) or d.global_sample_stride != 1):
            raise ValueError("Unsupported action/video horizon or sample stride")
        if p.num_obs_steps != d.num_frames:
            raise ValueError("Processor observation count disagrees with the training horizon")
        if len(d.video_size) != 2 or any(int(v) <= 0 or int(v) % 16 for v in d.video_size):
            raise ValueError("Video size must be positive multiples of 16")
        if not d.use_text_embed_cache or m.load_text_encoder:
            raise ValueError("This wrapper requires the cached-text training configuration")

    def text_context(self, task):
        # Accept either the task string used by RURI or the fully formatted
        # prompt used by offline FastWAM evaluation; never double-prefix it.
        prefix = DEFAULT_PROMPT.partition("{task}")[0]
        prompt = task if task.startswith(prefix) else DEFAULT_PROMPT.format(task=task)
        if prompt not in self._text:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            path = self.text_cache_dir / f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
            if not path.is_file():
                raise ValueError(f"Uncached FastWAM prompt {task!r}; precompute text embedding: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            context, mask = payload["context"], payload["mask"]
            if (not isinstance(context, torch.Tensor) or not context.is_floating_point()
                    or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool):
                raise ValueError(f"Invalid text embedding types in {path}")
            if context.shape != (self.context_len, self.text_dim) or mask.shape != (self.context_len,):
                raise ValueError(f"Invalid text embedding shapes in {path}")
            if not torch.isfinite(context).all():
                raise ValueError(f"Non-finite text embedding in {path}")
            context = context.clone()
            context[~mask] = 0
            # This all-true attention mask AFTER zeroing padding matches training.
            self._text = {prompt: (context, torch.ones_like(mask))}
        return self._text[prompt]

    def load_model(self):
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this FastWAM environment")
        # Complete MoT exports replace both experts. Legacy video-only checkpoints
        # must not leave an untrained action expert in a seemingly working server.
        logger.info("Loading FastWAM weights %s", self.checkpoint_path)
        payload = torch.load(self.checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        if "mot" not in payload or "proprio_encoder" not in payload:
            raise ValueError("Expected a complete FastWAM mot + proprio_encoder weights export")
        m = OmegaConf.create(OmegaConf.to_container(self.cfg.model, resolve=True))
        m.skip_dit_load_from_pretrain = True
        m.action_dit_pretrained_path = None
        m.compile_training_denoise = False
        # Frozen VAE is not in the fine-tuned export. Resolve it without chdir
        # and disallow automatic downloads. This environment override is startup-only.
        if m.redirect_common_files:
            vae = self.model_base / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
        else:
            vae = self.model_base / m.model_id / "Wan2.2_VAE.pth"
        _file(vae)
        overrides = {"DIFFSYNTH_MODEL_BASE_PATH": str(self.model_base), "DIFFSYNTH_SKIP_DOWNLOAD": "true"}
        previous = {key: os.environ.get(key) for key in overrides}
        try:
            os.environ.update(overrides)
            logger.info("Constructing FastWAM on CPU; frozen VAE: %s", vae)
            self.model = instantiate(m, device="cpu", model_dtype=self.dtype)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.model.mot.load_state_dict(payload["mot"], strict=True)
        self.model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        self.checkpoint_step = payload.get("step")
        del payload
        self.model.to(self.device).eval()
        # FastWAM.to() moves modules but does not update this routing attribute.
        self.model.device = self.device
        logger.info("FastWAM step %s loaded on %s; horizon=%s", self.checkpoint_step, self.device, self.horizon)

    def prepare(self, state, images):
        frames = []
        for meta in self.processor.shape_meta["images"]:
            frame = torch.from_numpy(images[meta["key"]]).permute(2, 0, 1).unsqueeze(0)
            for transform in self.processor.val_transforms:
                frame = transform(frame)
            if tuple(frame.shape[1:]) != tuple(meta["shape"]):
                raise ValueError("Image transforms do not match checkpoint shape_meta")
            frames.append(frame)
        image = torch.cat(frames, dim=-1)
        for transform in self.image_finish:
            image = transform(image)
        proprio = self.state_norm.forward(torch.from_numpy(state))
        return image, proprio

    def restore_action(self, action, state):
        if not isinstance(action, torch.Tensor) or action.shape != (self.horizon, 7):
            raise ValueError("Expected a full [horizon, 7] FastWAM action tensor")
        raw = self.action_norm.backward(action.detach().cpu().float())
        batch = {"action": {"default": raw}, "state": {"default": torch.from_numpy(state).unsqueeze(0)}}
        for transform in reversed(self.processor.action_state_transforms):
            batch = transform.backward(batch)
        # Do NOT call processor.postprocess(): its observation-prefix slicing
        # would discard all 32 predicted steps for num_obs_steps=33.
        return batch["action"]["default"].numpy()

    def predict(self, state, images, prompt, *, steps, seed):
        context, mask = self.text_context(prompt)
        with torch.inference_mode():
            image, proprio = self.prepare(state, images)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            prediction = self.model.infer_action(
                prompt=None, input_image=image, action_horizon=self.horizon,
                proprio=proprio, context=context, context_mask=mask,
                num_inference_steps=steps, seed=seed, rand_device="cpu",
                tiled=False, compile_action_infer=self.compile_action_infer,
            )["action"]
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            infer_ms = (time.perf_counter() - start) * 1000
            return self.restore_action(prediction, state), infer_ms

    def metadata(self):
        return {
            "checkpoint": str(self.checkpoint_path), "checkpoint_step": self.checkpoint_step,
            "config_path": str(self.config_path), "stats_path": str(self.stats_path),
            "stats_sha256": hashlib.sha256(self.stats_path.read_bytes()).hexdigest(),
            "text_cache_dir": str(self.text_cache_dir), "model_base_path": str(self.model_base),
            "device": str(self.device), "dtype": str(self.dtype),
            "compile_action_infer": self.compile_action_infer,
            "compile_mode": "reduce-overhead" if self.compile_action_infer else None,
            "video_size": list(self.data.video_size),
        }
