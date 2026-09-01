"""
Pi0.5 policy wrapper backed by OpenPI (JAX).

This wrapper adapts an OpenPI Pi0.5 checkpoint to the RURI
``inputs: dict -> infer() -> response: dict`` contract.

Everything Pi0.5-specific lives here: checkpoint loading, image
normalization, prompt resolution, and action-chunk postprocessing. The
RURI server and the client-side scheduler stay unaware of JAX and OpenPI.

Runtime requirements
--------------------
This module imports ``openpi`` lazily, inside :meth:`Pi05Wrapper._load_policy`,
so that importing ``ruri`` on a robot-side machine (which only needs
numpy/pyzmq/msgpack) does not pull in JAX. The server process must run in
an environment where ``openpi`` is importable and the OpenPI repo's
``TrainConfig`` registry contains ``config_name``.

Input contract
--------------
Clients speak the RURI naming convention, flat with dot-separated
namespaces::

    {
        "observation.state":        (state_dim,) float,   # proprioception
        "observation.images.top":   (H, W, 3) uint8,      # third-person camera
        "observation.images.wrist": (H, W, 3) uint8,      # wrist camera
        "prompt":                   str,                  # optional per-request
    }

:attr:`Pi05Wrapper.INPUT_MAPPING` renames those to the keys OpenPI's
Pi0.5 transforms expect (``observation/state``, ``observation/image``,
``observation/wrist_image``), so :meth:`_infer` already receives OpenPI
naming. Keys absent from the mapping are forwarded unchanged.

Images follow the RURI transport format: ``np.ndarray``, ``dtype=np.uint8``,
shape ``(H, W, 3)``, channel order **RGB**. Resolution is free -- OpenPI's
model transforms resize to 224x224 internally.

Channel order is the one part of the contract nothing can check at runtime,
and it matters: this Pi0.5 was trained on LeRobot data, which decodes to
RGB, and OpenPI swaps no channels anywhere. A client that forwards BGR
straight from OpenCV will not raise, it will just quietly degrade the
policy.

Response
--------
Flat, matching the input convention::

    {
        "action_chunk": (horizon, action_dim) float32 np.ndarray,
        "prompt": str,
        "timing.infer_ms": float,
        "timing.wrapper_ms": float,
    }

``action_chunk`` is a numpy array. Serializing it for the wire is the
transport layer's job, not the wrapper's.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ruri.server.wrappers.policy_wrapper import PolicyWrapper

if TYPE_CHECKING:
    from openpi.policies.policy import Policy


logger = logging.getLogger(__name__)


# OpenPI's own observation keys, i.e. what _infer() sees after mapping.
OPENPI_STATE = "observation/state"
OPENPI_BASE_IMAGE = "observation/image"
OPENPI_WRIST_IMAGE = "observation/wrist_image"
OPENPI_PROMPT = "prompt"


# Real-time-chunking hints, forwarded unmapped. Defined here rather than in
# either RTC wrapper so that the test-time and training-time wrappers are
# guaranteed to speak the same wire keys without one depending on the other --
# a scheduler written against one works against the other unchanged, and
# retiring either wrapper leaves the other intact.
CONTEXT_PREV_CHUNK = "context.rtc.prev_chunk_left_over"
CONTEXT_CONSUMED_STEPS = "context.rtc.consumed_steps"
CONTEXT_INFERENCE_DELAY = "context.rtc.estimated_inference_delay_steps"


class Pi05Wrapper(PolicyWrapper):
    """
    Serve a fine-tuned OpenPI Pi0.5 checkpoint through the RURI interface.

    Example:
        >>> wrapper = Pi05Wrapper(
        ...     checkpoint_path="/common/users/jh2400/openpi_checkpoints/"
        ...                     "pi05_tight_insertion_E1/tight_insertion_E1_10k/9999",
        ...     config_name="pi05_tight_insertion_E1",
        ...     default_prompt="pick and place the object E into the first hole "
        ...                    "on the manipulation-net board.",
        ... )
        >>> response = wrapper.infer({
        ...     "observation.state": np.zeros(7, dtype=np.float32),
        ...     "observation.images.top": np.zeros((480, 640, 3), dtype=np.uint8),
        ...     "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        ... })
        >>> response["action_chunk"].shape
        (10, 7)

    Args:
        checkpoint_path:
            Directory of a single training step, e.g. ``.../9999``. It must
            contain ``params/`` and ``assets/`` -- normalization statistics
            are read from the checkpoint itself, not from the config's
            assets dir, so the policy always matches how it was trained.
        config_name:
            Name of the OpenPI ``TrainConfig`` used for this checkpoint
            (e.g. ``"pi05_tight_insertion_E1"``). It selects the data
            transforms, so it must be the config the checkpoint was
            trained with.
        default_prompt:
            Prompt used when a request carries none. Pi0.5 is
            language-conditioned and the tight-insertion configs train with
            ``prompt_from_task=True``, so inference must supply a prompt.
            Use the dataset's task string verbatim -- a paraphrase is
            off-distribution.
        num_denoising_steps:
            Flow-matching sample steps. ``None`` keeps the OpenPI default
            (10). Lowering it trades action quality for latency.
        state_dim:
            Proprioception width, used only to build the warmup observation.
        warmup:
            Run one dummy inference at construction to trigger JAX
            compilation, which otherwise costs tens of seconds on the first
            real request, mid-episode.
        warmup_image_shape:
            HWC shape of the dummy warmup images.
    """

    # The policy side keeps OpenPI's own spelling. OpenPI's Policy.infer()
    # consumes exactly this flat dict and nests it downstream itself, so
    # _infer() has no translation of its own to do.
    INPUT_MAPPING = {
        "observation.state": OPENPI_STATE,
        "observation.images.top": OPENPI_BASE_IMAGE,
        "observation.images.wrist": OPENPI_WRIST_IMAGE,
        "prompt": OPENPI_PROMPT,
    }

    # Keyed by OpenPI names; ruri_metadata() derives the client-facing view.
    POLICY_METADATA = {
        "inputs": {
            OPENPI_STATE: {"type": "state"},
            OPENPI_BASE_IMAGE: {"type": "image"},
            OPENPI_WRIST_IMAGE: {"type": "image"},
            OPENPI_PROMPT: {"type": "string"},
        },
    }

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_name: str,
        *,
        default_prompt: str | None = None,
        num_denoising_steps: int | None = None,
        state_dim: int = 7,
        warmup: bool = True,
        warmup_image_shape: tuple[int, int, int] = (480, 640, 3),
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.config_name = config_name
        self.default_prompt = default_prompt
        self.num_denoising_steps = num_denoising_steps
        self.state_dim = state_dim
        self.warmup_image_shape = warmup_image_shape

        self.policy = self._load_policy()

        if warmup:
            self.warmup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @property
    def output_chunk_size(self) -> int:
        """Chunk length the checkpoint was trained with."""
        return self.policy._model.action_horizon

    def _load_policy(self) -> "Policy":
        """Build the OpenPI policy, failing fast on a bad checkpoint path."""
        # Deferred so that importing ruri without OpenPI installed works.
        try:
            from openpi.policies import policy_config as openpi_policy_config
            from openpi.training import config as openpi_config
        except ImportError as exc:
            raise ImportError(
                "Pi05Wrapper requires OpenPI. Start the RURI server from an "
                "environment where `import openpi` works."
            ) from exc

        if not self.checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"OpenPI checkpoint directory does not exist: {self.checkpoint_path}"
            )
        for required in ("params", "assets"):
            if not (self.checkpoint_path / required).exists():
                raise FileNotFoundError(
                    f"Checkpoint {self.checkpoint_path} is missing '{required}/'. "
                    "Point checkpoint_path at a single training step directory "
                    "(e.g. .../9999), not at the run directory above it."
                )

        train_config = openpi_config.get_config(self.config_name)

        sample_kwargs: dict[str, Any] | None = None
        if self.num_denoising_steps is not None:
            sample_kwargs = {"num_steps": self.num_denoising_steps}

        logger.info(
            "Loading Pi0.5 policy config=%s checkpoint=%s",
            self.config_name,
            self.checkpoint_path,
        )
        policy = openpi_policy_config.create_trained_policy(
            train_config,
            self.checkpoint_path,
            default_prompt=self.default_prompt,
            sample_kwargs=sample_kwargs,
        )
        logger.info("Pi0.5 policy loaded")
        return policy

    def warmup(self) -> None:
        """Trigger JAX compilation with a throwaway observation."""
        if self.default_prompt is None:
            logger.warning(
                "Skipping warmup: no default_prompt is set, so no valid dummy "
                "observation can be built. The first real request will pay the "
                "JAX compilation cost."
            )
            return

        # Through the public infer(), so INPUT_MAPPING is exercised too.
        dummy = {
            "observation.state": np.zeros(self.state_dim, dtype=np.float32),
            "observation.images.top": np.zeros(self.warmup_image_shape, dtype=np.uint8),
            "observation.images.wrist": np.zeros(self.warmup_image_shape, dtype=np.uint8),
        }

        start = time.perf_counter()
        self.infer(dummy)
        logger.info("Pi0.5 warmup finished in %.1f s", time.perf_counter() - start)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Run one Pi0.5 inference request.

        `inputs` has already had INPUT_MAPPING applied, so observation
        keys are in OpenPI naming. See the module docstring for the full
        request and response shapes.
        """
        wrapper_start = time.perf_counter()

        prompt = self._resolve_prompt(inputs)
        openpi_observation = self._to_openpi_observation(inputs, prompt)

        result = self.policy.infer(openpi_observation)

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"Expected Pi0.5 actions of shape (horizon, action_dim), got {actions.shape}"
            )

        infer_ms = float(result.get("policy_timing", {}).get("infer_ms", float("nan")))
        wrapper_ms = (time.perf_counter() - wrapper_start) * 1000.0

        logger.debug(
            "Pi0.5 chunk shape=%s prompt=%r infer_ms=%.1f wrapper_ms=%.1f",
            actions.shape,
            prompt,
            infer_ms,
            wrapper_ms,
        )

        return {
            "action_chunk": actions,
            "prompt": prompt,
            "timing.infer_ms": infer_ms,
            "timing.wrapper_ms": wrapper_ms,
        }

    def _to_openpi_observation(
        self, inputs: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        """Pull the mapped keys out of the request and check the RURI contract."""
        state = _require(inputs, OPENPI_STATE, "robot state", "observation.state")
        base_image = _require(
            inputs, OPENPI_BASE_IMAGE, "third-person image", "observation.images.top"
        )
        wrist_image = _require(
            inputs, OPENPI_WRIST_IMAGE, "wrist image", "observation.images.wrist"
        )

        return {
            OPENPI_STATE: np.asarray(state, dtype=np.float32).reshape(-1),
            OPENPI_BASE_IMAGE: _validate_image(base_image, "observation.images.top"),
            OPENPI_WRIST_IMAGE: _validate_image(wrist_image, "observation.images.wrist"),
            OPENPI_PROMPT: prompt,
        }

    def _resolve_prompt(self, inputs: dict[str, Any]) -> str:
        """Per-request prompt if given, else the wrapper default."""
        prompt = inputs.get(OPENPI_PROMPT)
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

        if self.default_prompt:
            return self.default_prompt

        raise KeyError(
            "Pi0.5 is language-conditioned but no prompt was found. Send a "
            "'prompt' key in the request, or construct the wrapper with "
            "default_prompt=..."
        )

    def optional_more_metadata(self) -> dict[str, Any]:
        """
        Which checkpoint this server is actually running.

        Useful for a client to log or display, but nothing here is part of
        the input contract, so none of it is promised. describe() nests all
        of it under the global ``policy`` key; the standard ``inputs``
        section is assembled there too.
        """
        return {
            "name": "pi05",
            "config_name": self.config_name,
            "checkpoint_path": str(self.checkpoint_path),
            "default_prompt": self.default_prompt,
            "num_denoising_steps": self.num_denoising_steps,
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _require(
    inputs: dict[str, Any], key: str, what: str, ruri_key: str
) -> Any:
    """Fetch a mapped key, naming the RURI key the client should have sent."""
    value = inputs.get(key)
    if value is None:
        raise KeyError(
            f"Request is missing the {what}. Expected RURI key {ruri_key!r} "
            f"(mapped to {key!r}); request has {tuple(inputs)}."
        )
    return value


def _validate_image(value: Any, ruri_key: str) -> np.ndarray:
    """
    Check an image against the RURI transport format and return it C-contiguous.

    RURI specifies ``np.ndarray``, ``dtype=np.uint8``, shape ``(H, W, 3)``,
    channel order RGB. The first three are checked here; channel order is
    not checkable at runtime, see the module docstring on BGR.

    Resolution is unconstrained; OpenPI resizes to 224x224 downstream.
    """
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"{ruri_key!r} must be a np.ndarray per the RURI image format, got "
            f"{type(value).__name__}. Convert on the client (e.g. "
            "tensor.cpu().numpy()) before sending."
        )

    if value.ndim != 3 or value.shape[-1] != 3:
        hint = ""
        if value.ndim == 3 and value.shape[0] == 3:
            hint = (
                " This looks like CHW; RURI images are HWC, so transpose on the "
                "client with np.transpose(img, (1, 2, 0))."
            )
        raise ValueError(
            f"{ruri_key!r} must have shape (H, W, 3) per the RURI image format, "
            f"got {value.shape}.{hint}"
        )

    if value.dtype != np.uint8:
        hint = ""
        if np.issubdtype(value.dtype, np.floating):
            hint = (
                " Float images are not accepted; scale to 0-255 and cast on the "
                "client with (img * 255).astype(np.uint8)."
            )
        raise TypeError(
            f"{ruri_key!r} must be dtype uint8 per the RURI image format, got "
            f"{value.dtype}.{hint}"
        )

    # JAX needs a contiguous buffer; a client-side crop or slice may not be one.
    return np.ascontiguousarray(value)
