"""
DM0.5 policy wrapper backed by OpenDM (PyTorch).

This wrapper adapts a fine-tuned DM0.5 checkpoint to the RURI
``inputs: dict -> infer() -> response: dict`` contract.

Everything DM0.5-specific lives here: checkpoint loading, the OpenDM
transform pipeline, prompt resolution, and action-chunk postprocessing.
The RURI server and the client-side scheduler stay unaware of OpenDM.

How this loads a policy
-----------------------
OpenDM's serving path is normally its own Flask app. This wrapper does not
use it: HTTP would mean base64-encoding two 480x640 frames per request for
a process on the same machine. Instead it drives the same objects that app
drives, in process.

The load goes through the *training entry point*, not through hand-copied
settings. ``exp_module`` names the ``playground/`` module a checkpoint was
trained with, exactly as :class:`~ruri.server.wrappers.pi05.pi05.Pi05Wrapper`
names an OpenPI ``TrainConfig``, and
``DM05Exp._initialize_inference_runtime()`` derives everything from it:
chunk size, action mode, ``n_bins``, ``add_state``, the camera order, the
robot type, and the normalization statistics.

That indirection is the point. Chunk size, bin count and camera order have
to agree with training, and a DM0.5 served with any of them wrong still
returns a well-shaped action chunk -- it is just wrong. Reading them from
the training config makes the agreement structural instead of a comment.

Runtime requirements
--------------------
``opendm`` is imported lazily inside :meth:`DM05Wrapper._load_policy`, so
importing ``ruri`` on a robot-side machine that only has numpy/pyzmq/msgpack
still works.

The server process needs the OpenDM environment (torch 2.11 + transformers 5.x
+ peft; on this machine ``/common/users/jh2400/conda_envs/opendm``). Two
things about that environment are load-bearing:

    ``opendm_root`` goes on ``sys.path``   ``pip install -e`` exposes the
        ``opendm`` package but not ``playground/``, and ``exp_module`` lives
        in the latter.

    the model is loaded with cwd = ``opendm_root``   A LoRA checkpoint's
        ``adapter_config.json`` records ``base_model_name_or_path`` as it was
        passed at training time, and the training entry's default is the
        *relative* ``./checkpoints/DM05``. The chdir is scoped to the load and
        restored afterwards.

Loading a LoRA checkpoint merges the adapter into the base weights
(``merge_and_unload``), so inference costs the same as a full checkpoint.
Expect ~13 GB of VRAM and ~40 s of startup.

Input contract
--------------
Clients speak the RURI naming convention, flat with dot-separated
namespaces::

    {
        "observation.state":        (7,) float,        # 6 joints + gripper
        "observation.images.top":   (H, W, 3) uint8,   # third-person camera
        "observation.images.wrist": (H, W, 3) uint8,   # wrist camera
        "prompt":                   str,               # optional per-request
        "context.actions_per_chunk": int,              # optional truncation
    }

These are the same keys the Pi0.5 wrapper takes, so one client can be pointed
at either server for an A/B on the same robot.

:attr:`DM05Wrapper.INPUT_MAPPING` renames the cameras to DM0.5's own
vocabulary, which is *positional*: OpenDM has no camera names, only an
ordered image list zipped against ``image_prompts`` (here ``["Head",
"Left wrist"]``). ``observation.images.1`` / ``.2`` are the slot names
OpenDM's own HTTP API uses, and the mapping is the one place recording that
``top`` is slot 1 and ``wrist`` is slot 2. Getting that order wrong is
silent: the model receives a wrist frame where it expects an overhead one
and simply drives badly.

Images follow the RURI transport format: ``np.ndarray``, ``dtype=np.uint8``,
shape ``(H, W, 3)``, channel order **RGB**. Resolution is free -- OpenDM pads
to square and resizes to 448x448 internally.

Channel order is the one part of the contract nothing can check at runtime,
and it matters: this DM0.5 was fine-tuned on frames decoded from LeRobot mp4s,
which are RGB, and OpenDM swaps no channels anywhere. A client that forwards
BGR straight from OpenCV will not raise, it will just quietly degrade the
policy.

Response
--------
Flat, matching the input convention::

    {
        "action_chunk": (horizon, 7) float32 np.ndarray,
        "prompt": str,
        "timing.infer_ms": float,
        "timing.wrapper_ms": float,
    }

``action_chunk`` holds **absolute** joint targets in the same units and frame
as ``observation.state``. The tight-insertion checkpoints train with
``action_mode=relative``, so the model predicts per-joint deltas, but OpenDM's
output transform adds the current state back before returning (the gripper
dimension is absolute throughout). A client consumes this chunk exactly as it
consumes Pi0.5's.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ruri.server.wrappers.policy_wrapper import PolicyWrapper


logger = logging.getLogger(__name__)


# DM0.5's own observation keys, i.e. what _infer() sees after mapping. OpenDM
# identifies cameras by 1-based slot, ordered to match the registered
# `image_prompts`; there are no camera names to map onto.
DM05_STATE = "observation.state"
DM05_IMAGE_HEAD = "observation.images.1"
DM05_IMAGE_LEFT_WRIST = "observation.images.2"
DM05_PROMPT = "prompt"

# Optional truncation hint from the scheduler, forwarded unmapped.
CONTEXT_ACTIONS_PER_CHUNK = "context.actions_per_chunk"


class DM05Wrapper(PolicyWrapper):
    """
    Serve a fine-tuned OpenDM DM0.5 checkpoint through the RURI interface.

    Example:
        >>> wrapper = DM05Wrapper(
        ...     checkpoint_path="/common/users/jh2400/opendm/user_checkpoints/"
        ...                     "dm05_tight_insertion_lora/checkpoint-10000",
        ...     exp_module="playground.dm05_tight_insertion_lora",
        ...     default_prompt="pick up the metal object on the bottom right, "
        ...                    "and insert it into the bottom right hole",
        ... )
        >>> response = wrapper.infer({
        ...     "observation.state": np.zeros(7, dtype=np.float32),
        ...     "observation.images.top": np.zeros((480, 640, 3), dtype=np.uint8),
        ...     "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        ... })
        >>> response["action_chunk"].shape
        (50, 7)

    Args:
        checkpoint_path:
            A single step directory, e.g. ``.../checkpoint-10000``. For a LoRA
            run this holds ``adapter_config.json`` + ``adapter_model.safetensors``
            and the base weights are pulled in from the path the adapter
            records. It should also hold the ``norm_stats.json`` the trainer
            copies in at save time; see ``norm_stats_source`` in
            :meth:`optional_more_metadata` for which file was actually used.
        exp_module:
            Importable module path of the ``playground/`` training entry this
            checkpoint was trained with, e.g.
            ``"playground.dm05_tight_insertion_lora"``. It selects the dataset
            registration and the data transforms, not just the architecture,
            so it must be the entry the checkpoint was trained with.
        opendm_root:
            OpenDM repository root. Added to ``sys.path`` so ``exp_module`` is
            importable, and used as the cwd while the model loads so that a
            relative ``base_model_name_or_path`` resolves.
        default_prompt:
            Prompt used when a request carries none. DM0.5 is
            language-conditioned, so inference must supply one. Use the string
            the checkpoint was trained on verbatim -- a paraphrase is
            off-distribution.
        diffusion_steps:
            Flow-matching Euler steps. ``None`` keeps the training entry's
            value (10). Lowering it trades action quality for latency.
        state_dim:
            Proprioception width, used only to build the warmup observation.
        warmup:
            Run throwaway inferences at construction. Recommended: they pay
            for CUDA kernel autotuning and for capturing DM0.5's suffix CUDA
            graphs, which otherwise lands on the first real request,
            mid-episode.
        warmup_iters:
            How many. Two, because the first pass captures the suffix graph
            and the second exercises the replay path a real request takes.
        warmup_image_shape:
            HWC shape of the dummy warmup images.
    """

    # RURI camera names -> DM0.5's positional slots. `top` must be slot 1 and
    # `wrist` slot 2, matching image_prompts ["Head", "Left wrist"] in the
    # dataset registration.
    INPUT_MAPPING = {
        "observation.state": DM05_STATE,
        "observation.images.top": DM05_IMAGE_HEAD,
        "observation.images.wrist": DM05_IMAGE_LEFT_WRIST,
        "prompt": DM05_PROMPT,
    }

    # Keyed by DM0.5 names; ruri_metadata() derives the client-facing view.
    POLICY_METADATA = {
        "inputs": {
            DM05_STATE: {"type": "state"},
            DM05_IMAGE_HEAD: {"type": "image"},
            DM05_IMAGE_LEFT_WRIST: {"type": "image"},
            DM05_PROMPT: {"type": "string"},
        },
    }

    def __init__(
        self,
        checkpoint_path: str | Path,
        exp_module: str,
        *,
        opendm_root: str | Path = "/common/users/jh2400/opendm",
        default_prompt: str | None = None,
        diffusion_steps: int | None = None,
        state_dim: int = 7,
        warmup: bool = True,
        warmup_iters: int = 2,
        warmup_image_shape: tuple[int, int, int] = (480, 640, 3),
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.exp_module = exp_module
        self.opendm_root = Path(opendm_root).expanduser().resolve()
        self.default_prompt = default_prompt
        self.diffusion_steps = diffusion_steps
        self.state_dim = state_dim
        self.warmup_image_shape = warmup_image_shape

        self._exp = None
        self.policy = self._load_policy()

        if warmup:
            self.warmup(iters=warmup_iters)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_policy(self):
        """Build the OpenDM inference runtime, failing fast on a bad path."""
        self._check_checkpoint()

        if str(self.opendm_root) not in sys.path:
            sys.path.insert(0, str(self.opendm_root))

        # Deferred so that importing ruri without OpenDM installed works.
        try:
            exp_config = importlib.import_module(self.exp_module)
        except ImportError as exc:
            raise ImportError(
                f"DM05Wrapper could not import exp_module {self.exp_module!r} from "
                f"{self.opendm_root}. Start the RURI server from an environment "
                "where `import opendm` works, and check that opendm_root points at "
                "the OpenDM repository root."
            ) from exc

        # `task="inference"` only records intent; this wrapper drives the
        # runtime directly instead of calling exp.inference(), which would
        # start OpenDM's own Flask server.
        exp = exp_config.DM05Exp(task="inference")
        exp.model_config.model_name_or_path = str(self.checkpoint_path)
        if self.diffusion_steps is not None:
            exp.inference_config.diffusion_steps = int(self.diffusion_steps)

        logger.info(
            "Loading DM0.5 policy exp_module=%s checkpoint=%s",
            self.exp_module,
            self.checkpoint_path,
        )
        # A LoRA adapter_config.json records the base model path as it was
        # given at training time, which is relative to the repo root.
        with _chdir(self.opendm_root):
            exp._initialize_inference_runtime()

        self._exp = exp
        logger.info(
            "DM0.5 policy loaded: robot_type=%s cameras=%s chunk=%d action_dim=%d "
            "diffusion_steps=%d",
            exp.inference_config.default_robot_type,
            exp.inference_config.image_prompts,
            exp.model_config.chunk_size,
            exp.inference_config.output_action_dim,
            exp.inference_config.diffusion_steps,
        )
        return exp.inference_config

    def _check_checkpoint(self) -> None:
        """Reject a bad checkpoint here rather than 40 s into model loading."""
        if not self.checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"DM0.5 checkpoint directory does not exist: {self.checkpoint_path}"
            )

        is_lora = (self.checkpoint_path / "adapter_config.json").exists()
        is_full = (self.checkpoint_path / "config.json").exists()
        if not (is_lora or is_full):
            # Pointing at the run directory instead of a step directory is the
            # easy mistake, so name the steps that do exist.
            steps = sorted(
                p.name for p in self.checkpoint_path.glob("checkpoint-*") if p.is_dir()
            )
            hint = ""
            if steps:
                hint = (
                    " This looks like a training output directory; pass one of its "
                    f"step directories instead: {', '.join(steps)}."
                )
            raise FileNotFoundError(
                f"{self.checkpoint_path} has neither 'adapter_config.json' (LoRA) "
                f"nor 'config.json' (full checkpoint).{hint}"
            )

        if not (self.checkpoint_path / "norm_stats.json").exists():
            # Not fatal: OpenDM falls back to ./norm_stats/<dataset>_<hash>.json,
            # which is the same file for a checkpoint trained on this machine.
            # Worth saying out loud, because wrong normalization does not raise,
            # it just returns bad actions.
            logger.warning(
                "No norm_stats.json in %s; OpenDM will fall back to the "
                "norm_stats/ directory under %s. Verify it belongs to this "
                "checkpoint's dataset and chunk size.",
                self.checkpoint_path,
                self.opendm_root,
            )

    def warmup(self, iters: int = 2) -> None:
        """Trigger CUDA autotuning and suffix-graph capture with dummy inputs."""
        if self.default_prompt is None:
            logger.warning(
                "Skipping warmup: no default_prompt is set, so no valid dummy "
                "observation can be built. The first real request will pay the "
                "compilation cost."
            )
            return

        # Through the public infer(), so INPUT_MAPPING is exercised too.
        dummy = {
            "observation.state": np.zeros(self.state_dim, dtype=np.float32),
            "observation.images.top": np.zeros(self.warmup_image_shape, dtype=np.uint8),
            "observation.images.wrist": np.zeros(
                self.warmup_image_shape, dtype=np.uint8
            ),
        }

        start = time.perf_counter()
        for _ in range(max(1, iters)):
            self.infer(dict(dummy))
        logger.info(
            "DM0.5 warmup finished in %.1f s (%d iters)",
            time.perf_counter() - start,
            max(1, iters),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Run one DM0.5 inference request.

        `inputs` has already had INPUT_MAPPING applied, so observation keys
        are in DM0.5 slot naming. See the module docstring for the full
        request and response shapes.
        """
        wrapper_start = time.perf_counter()

        prompt = self._resolve_prompt(inputs)
        data = self._to_opendm_observation(inputs, prompt)

        actions = np.asarray(self.policy._predict(data), dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"Expected DM0.5 actions of shape (horizon, action_dim), got "
                f"{actions.shape}"
            )

        # The scheduler may want fewer actions than the model's horizon, e.g.
        # when it re-plans faster than the chunk is consumed.
        actions_per_chunk = inputs.get(CONTEXT_ACTIONS_PER_CHUNK)
        if actions_per_chunk is not None:
            actions = actions[:actions_per_chunk]

        model_latency = self.policy.last_model_latency_sec
        infer_ms = float("nan") if model_latency is None else model_latency * 1000.0
        wrapper_ms = (time.perf_counter() - wrapper_start) * 1000.0

        logger.debug(
            "DM0.5 chunk shape=%s prompt=%r infer_ms=%.1f wrapper_ms=%.1f",
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

    def _to_opendm_observation(
        self, inputs: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        """Pull the mapped keys out of the request and check the RURI contract."""
        state = _require(inputs, DM05_STATE, "robot state", "observation.state")
        head = _require(
            inputs, DM05_IMAGE_HEAD, "third-person image", "observation.images.top"
        )
        wrist = _require(
            inputs, DM05_IMAGE_LEFT_WRIST, "wrist image", "observation.images.wrist"
        )

        # _prepare_model_input parses the state as JSON, so it wants a list, not
        # an ndarray, and it wants PIL images rather than RURI's arrays. It fills
        # robot_type / speed / state_desc from the registered dataset when they
        # are left out, which is what keeps serving aligned with training.
        return self.policy._prepare_model_input(
            text=prompt,
            images=[
                _to_pil(head, "observation.images.top"),
                _to_pil(wrist, "observation.images.wrist"),
            ],
            states=np.asarray(state, dtype=np.float32).reshape(-1).tolist(),
            robot_type=None,
        )

    def _resolve_prompt(self, inputs: dict[str, Any]) -> str:
        """Per-request prompt if given, else the wrapper default."""
        prompt = inputs.get(DM05_PROMPT)
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

        if self.default_prompt:
            return self.default_prompt

        raise KeyError(
            "DM0.5 is language-conditioned but no prompt was found. Send a "
            "'prompt' key in the request, or construct the wrapper with "
            "default_prompt=..."
        )

    def optional_more_metadata(self) -> dict[str, Any]:
        """
        What this server is actually running.

        Useful for a client to log or display, but nothing here is part of
        the input contract, so none of it is promised. describe() nests all
        of it under the global ``policy`` key; the standard ``inputs``
        section is assembled there too.

        The settings that have to agree with training are reported rather
        than hard-coded, since they are read back off the loaded runtime.
        """
        exp = self._exp
        has_ckpt_stats = (self.checkpoint_path / "norm_stats.json").exists()
        return {
            "name": "dm05",
            "exp_module": self.exp_module,
            "checkpoint_path": str(self.checkpoint_path),
            "opendm_root": str(self.opendm_root),
            "default_prompt": self.default_prompt,
            "dataset_name": exp.data_config.dataset_name,
            "robot_type": self.policy.default_robot_type,
            "image_prompts": list(self.policy.image_prompts),
            "chunk_size": exp.model_config.chunk_size,
            "action_dim": self.policy.output_action_dim,
            "action_mode": exp.data_config.action_mode.value,
            "returns_absolute_actions": bool(self.policy.use_absolute_action),
            "diffusion_steps": self.policy.diffusion_steps,
            "norm_stats_source": "checkpoint" if has_ckpt_stats else "norm_stats_dir",
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


@contextlib.contextmanager
def _chdir(path: Path):
    """Temporarily change the working directory, restoring it on the way out."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _require(inputs: dict[str, Any], key: str, what: str, ruri_key: str) -> Any:
    """Fetch a mapped key, naming the RURI key the client should have sent."""
    value = inputs.get(key)
    if value is None:
        raise KeyError(
            f"Request is missing the {what}. Expected RURI key {ruri_key!r} "
            f"(mapped to {key!r}); request has {tuple(inputs)}."
        )
    return value


def _to_pil(value: Any, ruri_key: str):
    """
    Validate an image against the RURI transport format and convert it to PIL.

    RURI specifies ``np.ndarray``, ``dtype=np.uint8``, shape ``(H, W, 3)``,
    channel order RGB. The first three are checked here; channel order is
    not checkable at runtime, see the module docstring on BGR.

    Resolution is unconstrained; OpenDM pads to square and resizes to 448x448
    downstream. Pillow is imported here rather than at module scope so that
    importing ruri on a robot-side machine stays numpy-only.
    """
    from PIL import Image

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

    return Image.fromarray(np.ascontiguousarray(value), mode="RGB")
