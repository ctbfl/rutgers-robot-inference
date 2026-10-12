"""Stateless, action-only FastWAM adapter for the single-Piper RURI contract.

State/actions use the same calibrated dataset coordinates as the client, NOT
radians or degrees. The backend handles training-statistics normalization and
chunk-relative joint targets; hardware calibration stays with the controller.
Images must be RGB uint8 HWC. RURI wrist maps to the training camera ``hand``.
"""

from __future__ import annotations

import sys
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ruri.server.wrappers.policy_wrapper import PolicyWrapper

logger = logging.getLogger(__name__)


class FastWAMWrapper(PolicyWrapper):
    INPUT_MAPPING = {
        "observation.state": "proprio",
        "observation.images.top": "top",
        "observation.images.wrist": "hand",
        "prompt": "prompt",
    }
    POLICY_METADATA = {
        "inputs": {
            "proprio": {"type": "state", "shape": [7]},
            "top": {"type": "image", "dtype": "uint8", "layout": "HWC", "color": "RGB"},
            "hand": {"type": "image", "dtype": "uint8", "layout": "HWC", "color": "RGB"},
            "prompt": {"type": "string", "required": False},
        }
    }

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        default_prompt: str,
        fastwam_root: str | Path = "/common/users/jh2400/FastWAM",
        config_path: str | Path | None = None,
        stats_path: str | Path | None = None,
        text_cache_dir: str | Path | None = None,
        model_base_path: str | Path | None = None,
        device: str = "cuda:0",
        diffusion_steps: int = 20,
        seed: int | None = 42,
        warmup: bool = True,
        compile_action_infer: bool = True,
        warmup_iters: int = 3,
    ):
        if not isinstance(default_prompt, str) or not default_prompt.strip():
            raise ValueError("default_prompt must be a non-empty training task string")
        if isinstance(diffusion_steps, bool) or not isinstance(diffusion_steps, int) or diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be a positive integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63):
            raise ValueError("seed must be None or an integer in [0, 2**63)")
        if not isinstance(compile_action_infer, bool):
            raise ValueError("compile_action_infer must be a boolean")
        if isinstance(warmup_iters, bool) or not isinstance(warmup_iters, int) or warmup_iters < 1:
            raise ValueError("warmup_iters must be a positive integer")
        root = Path(fastwam_root).expanduser().resolve()
        if not (root / "src/fastwam").is_dir():
            raise FileNotFoundError(f"Not a FastWAM source tree: {root}")
        source = str(root / "src")
        if source not in sys.path:
            sys.path.insert(0, source)

        # Neither torch nor FastWAM is required merely to import this module.
        from ._runtime import FastWAMRuntime

        self.default_prompt = default_prompt
        self.diffusion_steps = diffusion_steps
        self.seed = seed
        self.warmup_iters = warmup_iters
        self._lock = threading.Lock()
        self._runtime = FastWAMRuntime(
            checkpoint_path=checkpoint_path, root=root, config_path=config_path,
            stats_path=stats_path, text_cache_dir=text_cache_dir,
            model_base_path=model_base_path, device=device,
            compile_action_infer=compile_action_infer,
        )
        # Fail before allocating the model if the default instruction is uncached.
        self._runtime.text_context(default_prompt)
        self._runtime.load_model()
        if warmup:
            self.warmup()
        elif compile_action_infer:
            logger.warning("FastWAM compilation warmup skipped; the first request will pay compilation/capture cost")

    @property
    def output_chunk_size(self) -> int:
        return self._runtime.horizon

    def optional_more_metadata(self) -> dict[str, Any]:
        return {
            **self._runtime.metadata(),
            "default_prompt": self.default_prompt,
            "diffusion_steps": self.diffusion_steps,
            "seed": self.seed,
            "warmup_iters": self.warmup_iters,
            "action_dim": 7,
            "action_semantics": "absolute_targets",
            "coordinate_convention": "client_and_training_share_piper_calibration",
            "camera_order": ["top", "hand"],
            "text_mode": "cached_only",
            "supports_rtc": False,
            "generates_video": False,
        }

    def warmup(self) -> None:
        logger.info("Warming up FastWAM action inference")
        request = {
            "observation.state": np.zeros(7, dtype=np.float32),
            "observation.images.top": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        for iteration in range(self.warmup_iters):
            result = self.infer(request)
            logger.info("FastWAM warmup %d/%d: %.1f ms", iteration + 1, self.warmup_iters, result["timing.wrapper_ms"])
        logger.info("FastWAM warmup complete in %.1f ms", result["timing.wrapper_ms"])

    def _infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        if any(key.startswith("context.rtc.") for key in inputs):
            raise ValueError("FastWAM action-only wrapper does not support RTC conditioning")
        prompt = inputs.get("prompt", self.default_prompt)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        state = np.asarray(inputs["proprio"])
        if state.dtype.kind not in "fiu" or state.shape != (7,) or not np.all(np.isfinite(state)):
            raise ValueError("observation.state must contain exactly seven finite numeric values")
        with np.errstate(over="ignore"):
            state = state.astype(np.float32, copy=True)
        if not np.all(np.isfinite(state)):
            raise ValueError("observation.state cannot be represented as float32")
        images = {}
        for key in ("top", "hand"):
            value = inputs[key]
            if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
                raise ValueError(f"{key} image must be a uint8 NumPy array in RGB HWC order")
            if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) <= 0:
                raise ValueError(f"{key} image must have shape (H, W, 3) with nonzero H/W")
            images[key] = np.array(value, copy=True, order="C")

        # The public wrapper can also be used without the single-threaded REP server.
        with self._lock:
            action, infer_ms = self._runtime.predict(
                state, images, prompt, steps=self.diffusion_steps, seed=self.seed,
            )
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.output_chunk_size, 7) or not np.all(np.isfinite(action)):
            raise ValueError(f"FastWAM returned an invalid action chunk: {action.shape}")
        return {
            "action_chunk": np.ascontiguousarray(action),
            "prompt": prompt,
            "timing.infer_ms": infer_ms,
            "timing.wrapper_ms": (time.perf_counter() - start) * 1000,
        }
