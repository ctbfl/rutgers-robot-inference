"""Serve row1 gripper-effort checkpoints with their exact training architecture.

Send seven positions plus ``observation.joint_effort`` (seven raw measurements).
Only effort[-1], the signed gripper force in N, conditions the model. The shared
training transform appends and normalizes it; returned actions remain (30, 7).
Imports of the model packages and JAX are deferred until construction.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .pi05 import (
    CONTEXT_CONSUMED_STEPS, CONTEXT_INFERENCE_DELAY, CONTEXT_PREV_CHUNK,
    OPENPI_STATE, Pi05Wrapper, _require,
)
from .pi05_train_rtc import Pi05TrainRTCWrapper


logger = logging.getLogger(__name__)
OPENPI_EFFORT = "observation/joint_effort"
EFFORT_PACKAGE = "/common/users/jh2400/pi05-effort"
RTC_PACKAGE = "/common/users/jh2400/pi05-rtc"
ASSET_ID = "tight_insertion_row1_v21"


class _EffortMixin:
    training_rtc = False
    INPUT_MAPPING = {
        **Pi05Wrapper.INPUT_MAPPING,
        "observation.joint_effort": OPENPI_EFFORT,
    }

    def __init__(
        self, checkpoint_path: str | Path, *,
        effort_package: str | Path = EFFORT_PACKAGE,
        train_rtc_package: str | Path = RTC_PACKAGE,
        default_inference_delay: int = 4,
        rtc_warmup: bool = True,
        **kwargs: Any,
    ):
        if kwargs.get("state_dim", 7) != 7:
            raise ValueError("Effort policies require seven positions and seven actions")
        if not 0 <= default_inference_delay <= 7:
            raise ValueError("default_inference_delay must be in the trained range 0-7")
        self.effort_package = Path(effort_package)
        self.train_rtc_package = Path(train_rtc_package)
        self.default_inference_delay = int(default_inference_delay)
        self.rtc_warmup = rtc_warmup
        name = "pi05_row1_rtc_effort_h30" if self.training_rtc else "pi05_row1_effort_h30"
        # RTC's original constructor pins its non-effort config. Initialize the
        # common wrapper directly, retaining its RTC inference implementation.
        Pi05Wrapper.__init__(self, checkpoint_path, name, **kwargs)

    def _check_checkpoint(self) -> None:
        for required in ("params/_METADATA", "_CHECKPOINT_METADATA",
                         f"assets/{ASSET_ID}/norm_stats.json"):
            if not (self.checkpoint_path / required).is_file():
                raise FileNotFoundError(f"Checkpoint is missing {required}: {self.checkpoint_path}")
        commit = json.loads((self.checkpoint_path / "_CHECKPOINT_METADATA").read_text())
        if not commit.get("commit_timestamp_nsecs"):
            raise ValueError("Checkpoint save has not committed; select a completed step")
        metadata = json.loads((self.checkpoint_path / "params/_METADATA").read_text())
        paths = {
            tuple(k["key"] for k in leaf["key_metadata"])
            for leaf in metadata["tree_metadata"].values()
        }
        for layer in ("effort_mlp_in", "effort_mlp_out"):
            for parameter in ("kernel", "bias"):
                if ("params", layer, parameter, "value") not in paths:
                    raise ValueError(f"Not an effort checkpoint: missing {layer}/{parameter}")
        has_rtc = any("tok_time_proj" in path for path in paths)
        if has_rtc != self.training_rtc:
            expected = "Pi05TrainRTCEffortWrapper" if has_rtc else "Pi05EffortWrapper"
            raise ValueError(f"Checkpoint architecture mismatch; use {expected}")

    def _load_policy(self):
        self._check_checkpoint()
        for directory in (self.effort_package, self.train_rtc_package):
            path = str(directory.expanduser().resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            import jax.numpy as jnp
            from openpi import transforms
            from openpi.models import model
            from openpi.policies.policy import Policy
            from openpi.training.config import AssetsConfig
            from pi05_effort.config import make_train_config
        except ImportError as exc:
            raise ImportError(
                "Use the OpenPI environment and provide effort_package and "
                "train_rtc_package directories containing pi05_effort/pi05_train_rtc"
            ) from exc

        variant = "rtc" if self.training_rtc else "plain"
        config = make_train_config(variant, "serve", self.checkpoint_path.parent)
        # Resolve ALL normalization reads against the checkpoint, including the
        # data config's validation. Serving does not need the training dataset.
        assets_dir = str(self.checkpoint_path / "assets")
        data = dataclasses.replace(
            config.data, assets=AssetsConfig(assets_dir=assets_dir, asset_id=ASSET_ID),
        ).create(config.assets_dirs, config.model)
        logger.info("Loading %s from %s", self.config_name, self.checkpoint_path)
        params = model.restore_params(self.checkpoint_path / "params", dtype=jnp.bfloat16)
        loaded = config.model.load(params, remove_extra_params=False)
        policy = Policy(
            loaded,
            transforms=[
                transforms.InjectDefaultPrompt(self.default_prompt),
                *data.data_transforms.inputs,
                transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                *data.model_transforms.inputs,
            ],
            output_transforms=[
                *data.model_transforms.outputs,
                transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                *data.data_transforms.outputs,
            ],
            sample_kwargs=(None if self.num_denoising_steps is None
                           else {"num_steps": self.num_denoising_steps}),
            metadata=config.policy_metadata,
        )
        if self.training_rtc:
            self._assert_rtc_capable(policy)
        return policy

    def _to_openpi_observation(self, inputs, prompt):
        state = np.asarray(_require(inputs, OPENPI_STATE, "robot state", "observation.state"))
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("observation.state must contain seven finite positions")
        effort = np.asarray(_require(
            inputs, OPENPI_EFFORT, "measured effort", "observation.joint_effort",
        ), dtype=np.float32)
        if effort.shape != (7,) or not np.isfinite(effort[-1]):
            raise ValueError("observation.joint_effort must have shape (7,) and finite effort[-1]")
        observation = super()._to_openpi_observation(inputs, prompt)
        observation[OPENPI_EFFORT] = effort.copy()
        return observation

    def warmup(self) -> None:
        if not self.default_prompt:
            logger.warning("Skipping effort warmup: provide default_prompt to compile at startup")
            return
        dummy = {
            "observation.state": np.zeros(7, dtype=np.float32),
            "observation.joint_effort": np.zeros(7, dtype=np.float32),
            "observation.images.top": np.zeros(self.warmup_image_shape, dtype=np.uint8),
            "observation.images.wrist": np.zeros(self.warmup_image_shape, dtype=np.uint8),
        }
        self.infer(dummy)
        if self.training_rtc and self.rtc_warmup:
            response = self.infer({
                **dummy,
                CONTEXT_PREV_CHUNK: np.zeros((self.output_chunk_size, 7), dtype=np.float32),
                CONTEXT_CONSUMED_STEPS: 0,
                CONTEXT_INFERENCE_DELAY: max(1, self.default_inference_delay),
            })
            if not response["rtc.applied"]:
                raise RuntimeError("Effort RTC warmup did not exercise prefix conditioning")
        logger.info("Effort policy warmup complete")

    def optional_more_metadata(self):
        return {
            **super().optional_more_metadata(),
            "name": "pi05_train_rtc_effort" if self.training_rtc else "pi05_effort",
            "gripper_effort": True,
            "effort_input_index": 6,
            "effort_units": "N (signed)",
            "state_dim": 7,
            "action_dim": 7,
            "supports_rtc": self.training_rtc,
        }


_EFFORT_METADATA = {"type": "state", "shape": [7], "used_index": 6}


class Pi05EffortWrapper(_EffortMixin, Pi05Wrapper):
    """Ordinary pi05 + gripper effort; supports blocking/rolling inference."""

    POLICY_METADATA = {"inputs": {
        **Pi05Wrapper.POLICY_METADATA["inputs"], OPENPI_EFFORT: _EFFORT_METADATA,
    }}


class Pi05TrainRTCEffortWrapper(_EffortMixin, Pi05TrainRTCWrapper):
    """Training-time RTC + gripper effort, including exact prefix conditioning."""

    training_rtc = True
    POLICY_METADATA = {"inputs": {
        **Pi05TrainRTCWrapper.POLICY_METADATA["inputs"], OPENPI_EFFORT: _EFFORT_METADATA,
    }}
