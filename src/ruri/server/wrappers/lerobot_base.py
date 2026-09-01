"""
Shared base for LeRobot-backed policy wrappers.

Every LeRobot policy is loaded and run the same way -- read the config, pick the
policy class off its ``type`` string, restore the pre/post-processor pipelines
that were saved with the weights, then
``prepare -> preprocess -> predict_action_chunk -> postprocess``. That sequence
lives here. What differs between policies (which cameras, whether a language
prompt is required, how the RURI key names map onto the checkpoint's) is left to
subclasses, because those differences are real and papering over them produces a
server that runs and is quietly wrong.

Runtime requirements
--------------------
``lerobot`` is imported lazily inside :meth:`_load_policy`, so importing ``ruri``
on a robot-side machine that only has numpy/pyzmq/msgpack still works.

Note that this needs a *different environment* from the Pi0.5 wrapper. These
checkpoints carry the processor-pipeline format introduced after LeRobot 0.1.0,
and the OpenPI venv pins 0.1.0; loading them there fails. Serve LeRobot policies
from the LeRobot venv (0.5.2 on this machine) and Pi0.5 from the OpenPI one.

Input contract
--------------
Standard RURI naming, flat with dot-separated namespaces::

    {
        "observation.state":         (state_dim,) float,
        "observation.images.<name>": (H, W, 3) uint8,   # one per camera
        "prompt":                    str,               # optional
    }

``INPUT_MAPPING`` renames those to the checkpoint's own feature keys, so a
client speaks one vocabulary regardless of what a particular training run
happened to call its cameras. :meth:`_check_features_match_checkpoint` verifies
at load time that the subclass's mapping actually covers what the checkpoint
declares, and refuses to start otherwise -- a camera silently missing from the
observation is a policy that still returns actions, just bad ones.

Images follow the RURI transport format: ``np.ndarray``, ``dtype=np.uint8``,
shape ``(H, W, 3)``, channel order **RGB**. That is exactly what LeRobot's
``prepare_observation_for_inference`` consumes, so this wrapper hands frames
straight to it rather than reimplementing the ``/255`` and HWC->CHW permute.

Channel order is the one part of the contract nothing can check at runtime.
These policies were trained on LeRobot data, which decodes to RGB. A client
forwarding BGR straight from OpenCV will not raise; it will just degrade.

Response
--------
::

    {
        "action_chunk": (horizon, action_dim) float32 np.ndarray,
        "timing.infer_ms": float,
        "timing.wrapper_ms": float,
    }
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ruri.server.wrappers.policy_wrapper import PolicyWrapper


logger = logging.getLogger(__name__)



# LeRobot's own key for the language instruction.
LEROBOT_TASK = "task"


class LeRobotWrapper(PolicyWrapper):
    """
    Serve a LeRobot checkpoint through the RURI interface.

    Subclasses declare :attr:`POLICY_TYPE`, :attr:`INPUT_MAPPING` and
    :attr:`POLICY_METADATA`; everything else is inherited.

    Args:
        checkpoint_path:
            Either a ``pretrained_model/`` directory (the one holding
            ``config.json`` and ``model.safetensors``) or a checkpoint step
            directory containing one. Both are accepted because LeRobot's
            output layout nests them, and being strict about which level to
            point at buys nothing.
        device:
            Torch device for the policy. The saved postprocessor moves actions
            back to CPU on its own.
        task:
            Language instruction, used when a request carries no ``prompt``.
            Ignored by policies that are not language-conditioned, but still
            required as a key by the processor pipeline, so it is always set.
        robot_type:
            Passed through to the processor pipeline. Only meaningful for
            policies that condition on it.
        warmup:
            Run one dummy inference at construction. Torch does not pay a JAX-
            sized compile cost, but the first CUDA call still allocates
            workspaces and loads kernels, which is worth doing before the
            robot is waiting.
        warmup_image_shape:
            HWC shape of the dummy warmup frames.
    """

    # Expected value of the checkpoint's config `type` field. Checked on load.
    POLICY_TYPE: str | None = None

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        task: str = "",
        robot_type: str = "",
        warmup: bool = True,
        warmup_image_shape: tuple[int, int, int] = (480, 640, 3),
    ):
        self.checkpoint_path = _resolve_checkpoint_dir(checkpoint_path)
        self.device_name = device
        self.task = task
        self.robot_type = robot_type
        self.warmup_image_shape = warmup_image_shape

        self.policy, self.config, self.preprocessor, self.postprocessor = self._load_policy()
        self._check_features_match_checkpoint()

        if warmup:
            self.warmup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_policy(self):
        """Restore the policy and its saved processor pipelines."""
        # Deferred so that importing ruri without LeRobot installed works.
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        except ImportError as exc:
            raise ImportError(
                "LeRobot wrappers require lerobot and torch. Start the RURI "
                "server from an environment where `import lerobot` works -- "
                "note this is NOT the OpenPI venv, which pins lerobot 0.1.0 "
                "and cannot load processor-pipeline checkpoints."
            ) from exc

        config = PreTrainedConfig.from_pretrained(self.checkpoint_path)

        if self.POLICY_TYPE is not None and config.type != self.POLICY_TYPE:
            raise ValueError(
                f"{type(self).__name__} expects a {self.POLICY_TYPE!r} checkpoint, but "
                f"{self.checkpoint_path} declares type {config.type!r}. Use the "
                "wrapper matching the checkpoint, or the generic one."
            )

        # A single request carries a single observation. A policy trained to
        # consume a history of them cannot be served this way without the
        # client also sending that history, and quietly handing it one frame
        # produces a policy that still answers, just wrongly. Refuse instead.
        n_obs_steps = getattr(config, "n_obs_steps", 1)
        if n_obs_steps and n_obs_steps > 1:
            raise NotImplementedError(
                f"{self.checkpoint_path} was trained with n_obs_steps={n_obs_steps}, "
                "i.e. it conditions on a history of observations, but a RURI "
                "request carries exactly one. Serving it needs an observation "
                "history in the request contract (or server-side per-client "
                "state), neither of which this wrapper implements."
            )

        logger.info(
            "Loading LeRobot policy type=%s checkpoint=%s",
            config.type,
            self.checkpoint_path,
        )
        policy_cls = get_policy_class(config.type)
        policy = policy_cls.from_pretrained(self.checkpoint_path)
        policy.to(torch.device(self.device_name))
        policy.eval()

        # Load the normalization statistics that were saved alongside the
        # weights rather than recomputing them, so inference matches training
        # exactly. Same reasoning as the Pi0.5 wrapper reading norm stats out
        # of the checkpoint instead of the config assets dir.
        preprocessor, postprocessor = make_pre_post_processors(
            config, pretrained_path=str(self.checkpoint_path)
        )

        logger.info("LeRobot %s policy loaded", config.type)
        return policy, config, preprocessor, postprocessor

    def _check_features_match_checkpoint(self) -> None:
        """
        Fail loudly when INPUT_MAPPING and the checkpoint disagree.

        The mapping is a class attribute written by hand against one training
        run. Pointed at a checkpoint whose cameras are named differently, an
        unchecked wrapper would drop a camera into an unused key and serve a
        one-eyed policy without complaint.
        """
        expected = set(self.config.input_features)
        mapped = set(self.INPUT_MAPPING.values())

        missing = expected - mapped
        if missing:
            raise ValueError(
                f"{type(self).__name__}.INPUT_MAPPING does not produce "
                f"{sorted(missing)}, which {self.checkpoint_path} requires. "
                f"The mapping yields {sorted(mapped)}; the checkpoint declares "
                f"{sorted(expected)}. Fix the mapping to match this checkpoint."
            )

        unused = mapped - expected - {LEROBOT_TASK}
        if unused:
            logger.warning(
                "INPUT_MAPPING produces %s, which this checkpoint does not use; "
                "those inputs will be ignored.",
                sorted(unused),
            )

    @property
    def output_chunk_size(self) -> int:
        """Number of actions a single inference returns."""
        # chunk_size is None for policies that only expose n_action_steps.
        return self.config.chunk_size or self.config.n_action_steps

    @property
    def state_dim(self) -> int:
        return int(self.config.robot_state_feature.shape[0])

    @property
    def image_keys(self) -> list[str]:
        """Checkpoint-side camera keys, in the order the config declares them."""
        return list(self.config.image_features)

    def warmup(self) -> None:
        """Touch every CUDA path once, before the robot is waiting on it."""
        inverse = {policy: ruri for ruri, policy in self.INPUT_MAPPING.items()}
        dummy: dict[str, Any] = {
            inverse.get(key, key): np.zeros(self.warmup_image_shape, dtype=np.uint8)
            for key in self.image_keys
        }
        state_key = inverse.get("observation.state", "observation.state")
        dummy[state_key] = np.zeros(self.state_dim, dtype=np.float32)

        start = time.perf_counter()
        response = self.infer(dummy)
        logger.info(
            "LeRobot %s warmup finished in %.1f s (chunk %s)",
            self.config.type,
            time.perf_counter() - start,
            response["action_chunk"].shape,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Run one request.

        `inputs` has already had INPUT_MAPPING applied, so observation keys are
        in the checkpoint's own naming.
        """
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        wrapper_start = time.perf_counter()

        observation = self._to_lerobot_observation(inputs)
        task = self._resolve_task(inputs)
        device = torch.device(self.device_name)

        infer_start = time.perf_counter()
        with torch.inference_mode():
            # numpy HWC uint8 -> float32 CHW in [0, 1], batched, on device.
            # RURI's image format is precisely this function's input contract,
            # so use it rather than hand-rolling the conversion.
            batch = prepare_observation_for_inference(
                observation, device, task=task, robot_type=self.robot_type
            )
            batch = self.preprocessor(batch)
            # predict_action_chunk, not select_action: select_action hands back
            # one action at a time from an internal queue, which would make the
            # server stateful and hide the chunk from the scheduler. RURI's
            # scheduler owns chunk consumption.
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.postprocessor(chunk)

        if device.type == "cuda":
            # The postprocessor moves the result to CPU, which already syncs;
            # this is here so the timing stays honest if that ever changes.
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        actions = np.asarray(chunk[0].detach().cpu().numpy(), dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"Expected actions of shape (horizon, action_dim), got {actions.shape}"
            )

        wrapper_ms = (time.perf_counter() - wrapper_start) * 1000.0

        logger.debug(
            "%s chunk shape=%s infer_ms=%.1f wrapper_ms=%.1f",
            self.config.type,
            actions.shape,
            infer_ms,
            wrapper_ms,
        )

        return {
            "action_chunk": actions,
            "timing.infer_ms": infer_ms,
            "timing.wrapper_ms": wrapper_ms,
        }

    def _to_lerobot_observation(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Pull the mapped keys out of the request and check the RURI contract."""
        inverse = {policy: ruri for ruri, policy in self.INPUT_MAPPING.items()}

        state = inputs.get("observation.state")
        if state is None:
            raise KeyError(
                "Request is missing the robot state. Expected RURI key "
                f"{inverse.get('observation.state', 'observation.state')!r}; "
                f"request has {tuple(inputs)}."
            )

        observation: dict[str, np.ndarray] = {
            "observation.state": np.asarray(state, dtype=np.float32).reshape(-1)
        }

        for key in self.image_keys:
            image = inputs.get(key)
            if image is None:
                raise KeyError(
                    f"Request is missing the {key!r} camera. Expected RURI key "
                    f"{inverse.get(key, key)!r}; request has {tuple(inputs)}."
                )
            observation[key] = _validate_image(image, inverse.get(key, key))

        return observation

    def _resolve_task(self, inputs: dict[str, Any]) -> str:
        """Per-request prompt if given, else the wrapper default."""
        prompt = inputs.get(LEROBOT_TASK) or inputs.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
        return self.task

    def optional_more_metadata(self) -> dict[str, Any]:
        """Which checkpoint this server is actually running."""
        return {
            "name": self.config.type,
            "backend": "lerobot",
            "checkpoint_path": str(self.checkpoint_path),
            "n_action_steps": self.config.n_action_steps,
            "state_dim": self.state_dim,
            "image_keys": self.image_keys,
            "device": self.device_name,
            "task": self.task,
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_checkpoint_dir(path: str | Path) -> Path:
    """
    Accept either a ``pretrained_model/`` directory or its parent step directory.

    LeRobot writes ``<run>/checkpoints/<step>/pretrained_model/``, and pointing
    one level too high is the easy mistake, so handle it rather than erroring.
    """
    resolved = Path(path).expanduser().resolve()

    if not resolved.is_dir():
        raise FileNotFoundError(f"LeRobot checkpoint directory does not exist: {resolved}")

    if (resolved / "config.json").exists():
        return resolved

    nested = resolved / "pretrained_model"
    if (nested / "config.json").exists():
        return nested

    available = sorted(p.name for p in resolved.iterdir() if p.is_dir())
    raise FileNotFoundError(
        f"{resolved} contains no config.json and no pretrained_model/config.json. "
        "Point checkpoint_path at a step directory or the pretrained_model "
        f"directory inside it. Subdirectories here: {available}"
    )


def _validate_image(value: Any, ruri_key: str) -> np.ndarray:
    """
    Check an image against the RURI transport format.

    RURI specifies ``np.ndarray``, ``dtype=np.uint8``, shape ``(H, W, 3)``,
    channel order RGB. The first three are checked here; channel order is not
    checkable at runtime, see the module docstring on BGR.

    LeRobot's ``prepare_observation_for_inference`` assumes HWC uint8 and will
    happily permute something else into nonsense, which is why this runs first.
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

    # torch.from_numpy needs a writable, contiguous buffer; a client-side crop
    # or slice may be neither.
    return np.ascontiguousarray(value)
