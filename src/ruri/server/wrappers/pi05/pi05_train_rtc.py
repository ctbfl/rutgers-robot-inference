"""
Pi0.5 policy wrapper for **training-time** Real-Time Chunking.

This is a different method from :mod:`ruri.server.wrappers.pi05.pi05_rtc`, not a
variant of it, and the two are not interchangeable. Serving one method's
checkpoint through the other's wrapper does not raise -- it quietly produces a
worse policy -- so read this table before picking a server.

===================  ==========================================  ==================================================
                     ``pi05_rtc`` (test-time RTC)                 ``pi05_train_rtc`` (this module)
===================  ==========================================  ==================================================
paper                arXiv 2506.07339                            arXiv 2512.05964
checkpoint           any pi05 checkpoint, no retraining          **must** be trained with the delay-conditioned
                                                                 loss (``pi05_train_rtc_h30``); has an extra
                                                                 ``tok_time_proj`` parameter
how the prefix       PiGDM guidance term added to the velocity   the prefix is fed in as **clean input** at
gets in              at every Euler step; an approximation       per-token flow time 0 -- exact conditioning
knobs                ``prefix_attention_horizon``,               ``inference_delay`` only. There is no soft
                     ``prefix_attention_schedule``,              overlap and no guidance weight to tune; those
                     ``max_guidance_weight``                     parameters do not exist in this method
cost                 one extra VJP per flow step                 same as plain sampling
distribution shift   yes -- guidance steers the sampler off      no -- inference matches training exactly
                     the training distribution
===================  ==========================================  ==================================================

The practical consequence: only the ``inference_delay`` first actions are held
fixed, because those are the only ones physically already committed. The rest of
the chunk is completely free to react. Test-time RTC additionally drags the
overlap toward the previous chunk on a decaying schedule, which is what makes it
sluggish on contact-rich tasks when the execute horizon is short.

What goes wrong if you mix them up
----------------------------------
* **RTC-trained checkpoint through** ``pi05_rtc``: ``BaseModelConfig.load`` runs
  ``intersect_trees`` with ``remove_extra_params=True``, so ``tok_time_proj`` is
  dropped **silently** -- no error, no warning. A plain ``Pi0`` has no per-token
  time path anyway, so the entire training run is discarded and you get the old
  guided sampler on weights that were fine-tuned away from its assumptions.
  :class:`Pi05RTCWrapper` now refuses this combination.
* **Baseline checkpoint through this wrapper**: ``check_pytree_equality`` raises,
  because the checkpoint is missing a parameter the model needs. Loud, fine.
  :meth:`_load_policy` additionally asserts ``tok_time_proj`` is non-zero, which
  catches anything that slips past that.

Request contract
----------------
Deliberately identical to :class:`Pi05RTCWrapper`'s, so an existing scheduler
needs no changes::

    {
        "context.rtc.prev_chunk_left_over": (H - s, action_dim) float,
        "context.rtc.consumed_steps": int,                    # s
        "context.rtc.estimated_inference_delay_steps": int,   # d
    }

``prev_chunk_left_over`` is the previous ``action_chunk`` with its consumed rows
dropped, in **raw robot action space**, so its row 0 covers the same timestep as
row 0 of the chunk being requested. Only its first ``d`` rows are actually used
-- they are the actions the arm is committed to executing while this request is
in flight.

Response
--------
``Pi05Wrapper``'s fields plus::

    "rtc.applied":         bool
    "rtc.reason":          str | None       # why it fell back, if it did
    "rtc.method":          "training-time"  # never "test-time"
    "rtc.inference_delay": int              # after clamping
    "rtc.execute_from":    int              # == inference_delay

**Start executing at** ``rtc.execute_from``, not at row 0. Rows
``[0, execute_from)`` are the pinned prefix the arm is already running;
replaying them double-steps it. (Under test-time RTC those rows are only
approximately the committed ones, so the same rule applies there -- this wrapper
just states it.)

Delta action space
------------------
This config trains with ``DeltaActions``: every action in a chunk is a delta
against the state at the moment that chunk was generated. The previous chunk is
therefore expressed relative to ``s_old``, the new one relative to ``s_new``, and
handing the prefix over unchanged is wrong by ``s_old - s_new`` -- silently.
Rather than reimplement the conversion, this wrapper injects the previous chunk
into the observation dict as ``"actions"`` and lets OpenPI's own input pipeline
(``LiberoInputs`` forwards it, ``DeltaActions`` rebases it on the *current*
state, ``Normalize`` and ``PadStatesAndActions`` finish) do exactly what it did
during training. Verified end to end by ``pi05-rtc/serve_pi05_train_rtc.py
--self-test``.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ruri.server.wrappers.pi05.pi05 import (
    CONTEXT_CONSUMED_STEPS,
    CONTEXT_INFERENCE_DELAY,
    CONTEXT_PREV_CHUNK,
    OPENPI_BASE_IMAGE,
    OPENPI_PROMPT,
    OPENPI_STATE,
    OPENPI_WRIST_IMAGE,
    Pi05Wrapper,
)


logger = logging.getLogger(__name__)


# The TrainConfig for training-time RTC lives outside the OpenPI registry, in a
# side package that subclasses Pi0Config rather than modifying OpenPI.
PI05_TRAIN_RTC_PACKAGE = "/common/users/jh2400/pi05-rtc"
PI05_TRAIN_RTC_CONFIG = "pi05_train_rtc_h30"


class Pi05TrainRTCWrapper(Pi05Wrapper):
    """
    Serve a training-time RTC checkpoint with exact prefix conditioning.

    Example:
        >>> wrapper = Pi05TrainRTCWrapper(
        ...     checkpoint_path="/common/users/jh2400/openpi_checkpoints/"
        ...                     "pi05_train_rtc_h30/rtc_h30_10k/9999",
        ...     default_prompt="pick and place the object E into the first hole "
        ...                    "on the manipulation-net board.",
        ... )
        >>> response = wrapper.infer({
        ...     "observation.state": state,
        ...     "observation.images.top": top,
        ...     "observation.images.wrist": wrist,
        ...     "context.rtc.prev_chunk_left_over": previous[8:],
        ...     "context.rtc.consumed_steps": 8,
        ...     "context.rtc.estimated_inference_delay_steps": 4,
        ... })
        >>> response["rtc.execute_from"]
        4

    Args:
        default_inference_delay:
            Used when a request carries no
            ``context.rtc.estimated_inference_delay_steps``. Measured on this
            setup: ~120 ms end to end at 30 fps, i.e. 3-5 control steps, so 4.
            A client that measures its own round trip should send it instead.
        train_rtc_package:
            Directory containing the ``pi05_train_rtc`` package. Only used if
            the package is not already importable.
        rtc_warmup:
            Compile the prefix-conditioned path at startup. Passing
            ``prefix_actions`` changes the traced signature, so warming only
            the unconditioned path leaves a compile to land on the first real
            conditioned request, mid-episode.

        Remaining arguments are :class:`Pi05Wrapper`'s.
    """

    POLICY_METADATA = {
        "inputs": {
            OPENPI_STATE: {"type": "state"},
            OPENPI_BASE_IMAGE: {"type": "image"},
            OPENPI_WRIST_IMAGE: {"type": "image"},
            OPENPI_PROMPT: {"type": "string"},
            CONTEXT_PREV_CHUNK: {"type": "action_chunk", "optional": True},
            CONTEXT_CONSUMED_STEPS: {"type": "int", "optional": True},
            CONTEXT_INFERENCE_DELAY: {"type": "int", "optional": True},
        },
    }

    def __init__(
        self,
        checkpoint_path,
        config_name: str = PI05_TRAIN_RTC_CONFIG,
        *,
        default_inference_delay: int = 4,
        train_rtc_package: str | Path = PI05_TRAIN_RTC_PACKAGE,
        rtc_warmup: bool = True,
        **kwargs: Any,
    ):
        if config_name != PI05_TRAIN_RTC_CONFIG:
            raise ValueError(
                f"Pi05TrainRTCWrapper only serves {PI05_TRAIN_RTC_CONFIG!r}, got "
                f"{config_name!r}. A baseline config builds a plain Pi0, which has no "
                "per-token time path and would drop the checkpoint's tok_time_proj. "
                "For a baseline checkpoint use Pi05Wrapper or Pi05RTCWrapper."
            )
        # Set before super().__init__, which runs warmup(), which needs them.
        self.default_inference_delay = int(default_inference_delay)
        self.train_rtc_package = Path(train_rtc_package)
        self.rtc_warmup = rtc_warmup

        super().__init__(checkpoint_path, config_name, **kwargs)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @property
    def action_horizon(self) -> int:
        """Chunk length H the checkpoint was trained with."""
        return self.policy._model.action_horizon

    @property
    def max_inference_delay(self) -> int:
        """Largest delay the model was actually trained to condition on."""
        return len(self.policy._model.delay_probs) - 1

    def _load_policy(self):
        """
        Build the policy from ``Pi05TrainRTCConfig``, then prove it is one.

        ``Pi05Wrapper._load_policy`` resolves the config through OpenPI's
        registry, which cannot produce this model class -- hence the override.
        """
        try:
            from openpi.policies import policy_config as openpi_policy_config
        except ImportError as exc:
            raise ImportError(
                "Pi05TrainRTCWrapper requires OpenPI. Start the RURI server from "
                "an environment where `import openpi` works."
            ) from exc

        make_train_config = self._import_train_rtc()

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

        train_config = make_train_config("serve")

        sample_kwargs: dict[str, Any] | None = None
        if self.num_denoising_steps is not None:
            sample_kwargs = {"num_steps": self.num_denoising_steps}

        logger.info(
            "Loading Pi0.5 training-time RTC policy config=%s checkpoint=%s",
            train_config.name,
            self.checkpoint_path,
        )
        policy = openpi_policy_config.create_trained_policy(
            train_config,
            self.checkpoint_path,
            default_prompt=self.default_prompt,
            sample_kwargs=sample_kwargs,
        )
        self._assert_rtc_capable(policy)
        logger.info(
            "Pi0.5 training-time RTC policy loaded (H=%d, trained delays 0-%d)",
            policy._model.action_horizon,
            len(policy._model.delay_probs) - 1,
        )
        return policy

    def _import_train_rtc(self):
        """Import ``pi05_train_rtc``, adding its directory to sys.path if needed."""
        try:
            from pi05_train_rtc.config import make_train_config
        except ImportError:
            package = str(self.train_rtc_package.expanduser().resolve())
            if package not in sys.path:
                logger.info("adding %s to sys.path for pi05_train_rtc", package)
                sys.path.insert(0, package)
            try:
                from pi05_train_rtc.config import make_train_config
            except ImportError as exc:
                raise ImportError(
                    "Could not import `pi05_train_rtc`, which defines the model "
                    f"class for training-time RTC. Looked in {package!r}. Pass "
                    "train_rtc_package=... or install the package into the "
                    "server environment."
                ) from exc
        return make_train_config

    @staticmethod
    def _assert_rtc_capable(policy) -> None:
        """
        Fail loudly if the loaded model cannot actually do prefix conditioning.

        Two distinct mistakes land here. A model without ``embed_suffix_rtc``
        means the config built a plain ``Pi0``. A ``tok_time_proj`` that is
        still all zeros means it was freshly initialized rather than restored,
        i.e. the checkpoint was not trained with the delay-conditioned loss --
        the parameter is zero-initialized precisely so that an untrained model
        is numerically identical to stock pi05, which makes zero the signature
        of "never trained".
        """
        import flax.nnx as nnx
        import numpy as _np

        model = policy._model
        if not hasattr(model, "embed_suffix_rtc"):
            raise RuntimeError(
                f"Loaded model is a {type(model).__name__}, not a Pi05TrainRTC. "
                "It has no per-token time path, so prefix conditioning is impossible."
            )
        state = nnx.state(model, nnx.Param).to_pure_dict()
        kernel = state.get("tok_time_proj", {}).get("kernel")
        if kernel is None:
            raise RuntimeError("Loaded model has no tok_time_proj parameter.")
        norm = float(_np.linalg.norm(_np.asarray(kernel, dtype=_np.float32)))
        if norm == 0.0:
            raise RuntimeError(
                "tok_time_proj is all zeros, so this checkpoint was never trained "
                "with the delay-conditioned loss and prefix conditioning would be "
                "inert. Point checkpoint_path at a pi05_train_rtc_h30 run."
            )
        logger.info("tok_time_proj restored (||kernel||_F = %.4f)", norm)

    def warmup(self) -> None:
        """Compile both the unconditioned and the prefix-conditioned path."""
        super().warmup()

        if not self.rtc_warmup:
            return
        if self.default_prompt is None:
            logger.warning("Skipping RTC warmup: no default_prompt is set.")
            return

        horizon = self.action_horizon
        dummy = {
            "observation.state": np.zeros(self.state_dim, dtype=np.float32),
            "observation.images.top": np.zeros(self.warmup_image_shape, dtype=np.uint8),
            "observation.images.wrist": np.zeros(self.warmup_image_shape, dtype=np.uint8),
            # s = 0: full-length leftover. The delay is a runtime array, so any
            # delay traces to the same program once this shape is compiled.
            CONTEXT_PREV_CHUNK: np.zeros((horizon, self.state_dim), dtype=np.float32),
            CONTEXT_CONSUMED_STEPS: 0,
            CONTEXT_INFERENCE_DELAY: self.default_inference_delay,
        }

        start = time.perf_counter()
        response = self.infer(dummy)
        if not response.get("rtc.applied"):
            raise RuntimeError(
                "RTC warmup did not exercise the conditioned path "
                f"({response.get('rtc.reason')!r}); the first real conditioned "
                "request would pay the JAX compile instead."
            )
        logger.info(
            "Pi0.5 training-time RTC warmup finished in %.1f s",
            time.perf_counter() - start,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        wrapper_start = time.perf_counter()

        prompt = self._resolve_prompt(inputs)
        openpi_observation = self._to_openpi_observation(inputs, prompt)

        prev_chunk, delay, reason = self._resolve_rtc_context(inputs)

        if prev_chunk is None:
            result = self.policy.infer(openpi_observation)
            actions = np.asarray(result["actions"], dtype=np.float32)
            infer_ms = float(result.get("policy_timing", {}).get("infer_ms", float("nan")))
            rtc_info: dict[str, Any] = {
                "rtc.applied": False,
                "rtc.reason": reason,
                "rtc.method": "training-time",
                "rtc.inference_delay": 0,
                "rtc.execute_from": 0,
            }
        else:
            actions, infer_ms = self._infer_conditioned(openpi_observation, prev_chunk, delay)
            rtc_info = {
                "rtc.applied": True,
                "rtc.reason": None,
                "rtc.method": "training-time",
                "rtc.inference_delay": int(delay),
                "rtc.execute_from": int(delay),
            }

        if actions.ndim != 2:
            raise ValueError(
                f"Expected Pi0.5 actions of shape (horizon, action_dim), got {actions.shape}"
            )

        wrapper_ms = (time.perf_counter() - wrapper_start) * 1000.0

        logger.debug(
            "Pi0.5 train-RTC chunk shape=%s applied=%s delay=%s infer_ms=%.1f",
            actions.shape,
            rtc_info["rtc.applied"],
            rtc_info["rtc.inference_delay"],
            infer_ms,
        )

        return {
            "action_chunk": actions,
            "prompt": prompt,
            "timing.infer_ms": infer_ms,
            "timing.wrapper_ms": wrapper_ms,
            **rtc_info,
        }

    def _resolve_rtc_context(
        self, inputs: dict[str, Any]
    ) -> tuple[np.ndarray | None, int, str | None]:
        """
        Validate the RTC fields and clamp the delay into what is usable.

        Returns ``(padded_prev_chunk, delay, reason)``; a ``None`` chunk means
        fall back to unconditioned sampling and ``reason`` says why.

        Unlike the test-time wrapper there is no ``prefix_attention_horizon`` to
        resolve: this method pins exactly the ``delay`` actions that are
        physically committed and leaves the rest of the chunk free.
        """
        leftover = inputs.get(CONTEXT_PREV_CHUNK)
        if leftover is None:
            return None, 0, "no previous chunk supplied"

        if not isinstance(leftover, np.ndarray):
            raise TypeError(
                f"{CONTEXT_PREV_CHUNK!r} must be a np.ndarray, got "
                f"{type(leftover).__name__}."
            )

        horizon_length = self.action_horizon
        if leftover.ndim != 2 or leftover.shape[-1] != self.state_dim:
            raise ValueError(
                f"{CONTEXT_PREV_CHUNK!r} must have shape (H - s, {self.state_dim}) "
                f"in raw robot action space, got {leftover.shape}."
            )

        overlap = int(leftover.shape[0])
        if overlap == 0:
            return None, 0, "previous chunk fully consumed"
        if overlap > horizon_length:
            raise ValueError(
                f"{CONTEXT_PREV_CHUNK!r} has {overlap} rows, which exceeds the "
                f"model's action horizon of {horizon_length}."
            )

        # Cross-checked rather than trusted: a client that drifts by one step
        # here pins the wrong timesteps, which reads as a slightly worse policy.
        consumed = inputs.get(CONTEXT_CONSUMED_STEPS)
        if consumed is not None:
            consumed = int(consumed)
            if consumed < 0:
                raise ValueError(f"{CONTEXT_CONSUMED_STEPS!r} must be >= 0, got {consumed}.")
            if consumed + overlap != horizon_length:
                raise ValueError(
                    f"RTC alignment mismatch: {CONTEXT_CONSUMED_STEPS!r}={consumed} "
                    f"plus {overlap} leftover rows should equal the action horizon "
                    f"{horizon_length}. The leftover must be the previous chunk with "
                    "exactly its consumed rows dropped, so that its row 0 is the "
                    "timestep this request's row 0 covers."
                )

        delay = inputs.get(CONTEXT_INFERENCE_DELAY)
        delay = self.default_inference_delay if delay is None else int(delay)
        if delay < 0:
            raise ValueError(f"{CONTEXT_INFERENCE_DELAY!r} must be >= 0, got {delay}.")
        if delay == 0:
            return None, 0, "inference delay is 0, nothing is committed"

        # Beyond the trained support the conditioning is extrapolation: the model
        # simply never saw that many clean tokens. Clamp and say so.
        if delay > self.max_inference_delay:
            logger.warning(
                "inference delay %d exceeds the trained support (0-%d); clamping. "
                "If the client is chronically this late, retrain with a wider "
                "delay distribution rather than living with the clamp.",
                delay,
                self.max_inference_delay,
            )
            delay = self.max_inference_delay
        if delay > overlap:
            logger.warning(
                "inference delay %d exceeds the %d leftover rows; clamping. The "
                "client is running later than its execute horizon supports.",
                delay,
                overlap,
            )
            delay = overlap

        # Pad to H by repeating the last row. Only rows [0, delay) are ever read
        # -- the rest of the chunk is generated freely -- so the padding cannot
        # influence the result; repeating just keeps values in a sane range
        # through the delta and normalization transforms.
        if overlap < horizon_length:
            padding = np.repeat(leftover[-1:], horizon_length - overlap, axis=0)
            leftover = np.concatenate([leftover, padding], axis=0)

        return np.asarray(leftover, dtype=np.float32), delay, None

    def _infer_conditioned(
        self, openpi_observation: dict[str, Any], prev_chunk: np.ndarray, delay: int
    ) -> tuple[np.ndarray, float]:
        """
        One prefix-conditioned inference.

        Reimplements ``openpi.policies.policy.Policy.infer`` rather than calling
        it, because the previous chunk has to ride through the input transforms
        alongside the observation before reaching ``sample_actions``. That is the
        only coupling to OpenPI internals here, and the reason this wrapper pins
        its own TrainConfig.
        """
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        policy = self.policy

        inputs = dict(openpi_observation)
        # DeltaActions subtracts in place, so hand the pipeline a copy -- else
        # the caller's array is silently converted to deltas.
        inputs["actions"] = np.asarray(prev_chunk, dtype=np.float32).copy()

        transformed = policy._input_transform(inputs)
        prefix_model = transformed.pop("actions")

        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed)
        prefix_model = jnp.asarray(prefix_model)[np.newaxis, ...]

        policy._rng, sample_rng = jax.random.split(policy._rng)
        observation = _model.Observation.from_dict(batched)

        sample_kwargs = dict(policy._sample_kwargs)
        sample_kwargs["prefix_actions"] = prefix_model
        sample_kwargs["inference_delay"] = jnp.asarray(delay)

        start = time.monotonic()
        actions = policy._sample_actions(sample_rng, observation, **sample_kwargs)
        actions = jax.block_until_ready(actions)
        infer_ms = (time.monotonic() - start) * 1000.0

        outputs = {
            "state": np.asarray(batched["state"][0, ...]),
            "actions": np.asarray(actions[0, ...]),
        }
        outputs = policy._output_transform(outputs)

        return np.asarray(outputs["actions"], dtype=np.float32), infer_ms

    def optional_more_metadata(self) -> dict[str, Any]:
        metadata = super().optional_more_metadata()
        metadata.update(
            {
                "name": "pi05_train_rtc",
                "rtc_method": "training-time",
                "paper": "arXiv 2512.05964",
                "action_horizon": self.action_horizon,
                "trained_delay_support": f"0-{self.max_inference_delay}",
                "default_inference_delay": self.default_inference_delay,
            }
        )
        return metadata
