"""
Pi0.5 policy wrapper with Real-Time Chunking (RTC).

Same contract as :class:`~ruri.server.wrappers.pi05.pi05.Pi05Wrapper`, plus
three optional request fields that let the scheduler ask for a chunk that
continues smoothly from the one the robot is still executing.

Why
---
A chunked policy that re-plans from scratch has no reason to agree with the
actions already in flight, so every new chunk lands as a small discontinuity.
RTC conditions generation on the previous chunk during flow matching, so the
seam disappears. It is inference-time only -- the same checkpoint, no retraining.
:mod:`ruri.server.wrappers.pi05.rtc` holds the algorithm; this file holds the
plumbing.

Not to be confused with training-time RTC
-----------------------------------------
:mod:`ruri.server.wrappers.pi05.pi05_train_rtc` implements a *different* method
(arXiv 2512.05964) that requires its own checkpoint. This module is the
inference-time method (arXiv 2506.07339): PiGDM guidance, tunable soft overlap,
one extra VJP per flow step, and an approximation of the conditional that pushes
the sampler off the training distribution. The other module has none of those,
and its comparison table is worth reading before choosing.

Feeding a training-time RTC checkpoint to this wrapper would be silent rather
than loud -- ``BaseModelConfig.load`` drops the extra ``tok_time_proj``
parameter without a warning, discarding the entire training run and leaving
guidance running on weights fine-tuned away from its assumptions. Hence the
check in :meth:`Pi05RTCWrapper._load_policy`.

Request contract
----------------
Everything ``Pi05Wrapper`` accepts, plus::

    {
        "context.rtc.prev_chunk_left_over": (H - s, action_dim) float,
        "context.rtc.consumed_steps": int,                    # s
        "context.rtc.estimated_inference_delay_steps": int,   # d
    }

``prev_chunk_left_over`` is the unexecuted tail of the previous chunk, in raw
robot action space -- exactly the rows of a previous ``action_chunk`` response
that have not been sent to the arm yet. Because ``s`` rows were already
consumed, its index 0 already lines up with index 0 of the chunk being
requested, so the server does no shifting.

``consumed_steps`` is therefore redundant with ``len(prev_chunk_left_over)``,
and that is the point: the server cross-checks them and rejects a mismatch. A
scheduler that drifts by one step here produces a policy that is subtly, silently
worse, which is not a failure mode worth trusting a client to avoid.

``estimated_inference_delay_steps`` is how many further steps the client expects
to execute before this response can be acted on -- round-trip latency in
timesteps. Those actions are already committed, so RTC reproduces them
near-exactly and lets agreement decay over the rest of the overlap.

Timing budget
-------------
RTC needs ``d <= prefix_attention_horizon <= H - s``. With this checkpoint's
``action_horizon=10`` at 30 fps a whole chunk is only 333 ms, so the usable
range is narrow and shrinks as ``s`` grows. The server clamps rather than
failing, and reports what it actually used under ``rtc.*`` in the response --
watch those numbers, because a client that is chronically late will see the
horizon clamped to near ``d``, at which point RTC is holding the trajectory
still instead of steering it.

Failure policy
--------------
Absent or exhausted history is normal (first step of an episode; the client ran
its buffer dry), so those fall back to plain sampling and say so in
``rtc.applied`` / ``rtc.reason``. A malformed or inconsistent request is a
client bug and raises, because degrading quietly would hide it behind a policy
that merely looks a bit worse.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ruri.server.wrappers.pi05.pi05 import (
    CONTEXT_ACTIONS_PER_CHUNK,
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


class Pi05RTCWrapper(Pi05Wrapper):
    """
    Serve a fine-tuned OpenPI Pi0.5 checkpoint with RTC prefix guidance.

    Example:
        >>> wrapper = Pi05RTCWrapper(
        ...     checkpoint_path=".../tight_insertion_E1_10k/9999",
        ...     config_name="pi05_tight_insertion_E1",
        ...     default_prompt="pick and place the object E into the first hole "
        ...                    "on the manipulation-net board.",
        ... )
        >>> response = wrapper.infer({
        ...     "observation.state": state,
        ...     "observation.images.top": top,
        ...     "observation.images.wrist": wrist,
        ...     "context.rtc.prev_chunk_left_over": previous[3:],
        ...     "context.rtc.consumed_steps": 3,
        ...     "context.rtc.estimated_inference_delay_steps": 4,
        ... })
        >>> response["rtc.applied"]
        True

    Args:
        prefix_attention_horizon:
            Where agreement with the previous chunk reaches zero, in
            timesteps. ``None`` means the full chunk. Always clamped down to
            the actual overlap ``H - s``, so this is an upper bound, not a
            promise. Smaller values buy reactivity at the cost of smoothness.
        prefix_attention_schedule:
            How weight decays between the committed prefix and the horizon --
            ``"exp"`` (default), ``"linear"``, ``"ones"``, or ``"zeros"``. See
            :func:`ruri.server.wrappers.pi05.rtc.get_prefix_weights`.
        max_guidance_weight:
            Ceiling on the PiGDM guidance strength. The reference value, 10.0,
            is tuned for 10-step flow matching, which is the OpenPI default.
        rtc_warmup:
            Compile the RTC path at startup too. The plain and guided samplers
            are separate JAX programs, so warming only the plain one leaves a
            multi-second compile to land on the first guided request --
            mid-episode, which is exactly what warmup exists to prevent.

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
        config_name: str,
        *,
        prefix_attention_horizon: int | None = None,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 10.0,
        rtc_warmup: bool = True,
        **kwargs: Any,
    ):
        # Set before super().__init__, which runs warmup(), which needs them.
        self.prefix_attention_horizon = prefix_attention_horizon
        self.prefix_attention_schedule = prefix_attention_schedule
        self.max_guidance_weight = max_guidance_weight
        self.rtc_warmup = rtc_warmup
        self._rtc_sampler = None

        super().__init__(checkpoint_path, config_name, **kwargs)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @property
    def action_horizon(self) -> int:
        """Chunk length H the checkpoint was trained with."""
        return self.policy._model.action_horizon

    def _load_policy(self):
        """
        Load as usual, after refusing a checkpoint meant for the other method.

        ``BaseModelConfig.load`` intersects the checkpoint against the model's
        own parameter tree and drops the remainder, so a training-time RTC
        checkpoint loads here without complaint and silently loses the
        conditioning it was trained for. Reading the Orbax metadata costs ~20 ms
        and turns that into an error.
        """
        params_dir = self.checkpoint_path / "params"
        if params_dir.exists():
            try:
                import orbax.checkpoint as ocp

                metadata = ocp.PyTreeCheckpointer().metadata(params_dir)
                top_level = set(metadata.get("params", metadata))
            except Exception as exc:  # noqa: BLE001 - never block loading on a probe
                logger.debug("Could not read checkpoint metadata (%s); skipping check", exc)
            else:
                if "tok_time_proj" in top_level:
                    raise ValueError(
                        f"{self.checkpoint_path} is a training-time RTC checkpoint "
                        "(it has a tok_time_proj parameter), which this wrapper would "
                        "silently discard. Serve it with Pi05TrainRTCWrapper "
                        "(ruri.server.wrappers.pi05.pi05_train_rtc) instead; that is a "
                        "different method, not a different setting of this one."
                    )
        return super()._load_policy()

    def _build_rtc_sampler(self):
        """JIT the guided sampler. Deferred so importing ruri stays JAX-free."""
        from ruri.server.wrappers.pi05 import rtc as _rtc

        if self.prefix_attention_schedule not in _rtc.PREFIX_ATTENTION_SCHEDULES:
            raise ValueError(
                f"prefix_attention_schedule must be one of "
                f"{_rtc.PREFIX_ATTENTION_SCHEDULES}, got "
                f"{self.prefix_attention_schedule!r}."
            )

        # Mirror whatever the plain path samples with, so the two paths differ
        # only by the guidance term. `num_denoising_steps=None` means the
        # OpenPI default of 10.
        num_steps = self.num_denoising_steps or 10

        return _rtc.make_rtc_sampler(
            self.policy._model,
            num_steps=num_steps,
            prefix_attention_schedule=self.prefix_attention_schedule,
            max_guidance_weight=self.max_guidance_weight,
        )

    def warmup(self) -> None:
        """Compile both the plain and the guided sampler."""
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
            # A full-length leftover, i.e. s = 0: the widest prefix window, and
            # the shape everything narrower traces to as well, since the delay
            # and horizon are runtime arrays rather than compile-time constants.
            CONTEXT_PREV_CHUNK: np.zeros((horizon, self.state_dim), dtype=np.float32),
            CONTEXT_CONSUMED_STEPS: 0,
            CONTEXT_INFERENCE_DELAY: 0,
        }

        start = time.perf_counter()
        response = self.infer(dummy)
        if not response.get("rtc.applied"):
            raise RuntimeError(
                "RTC warmup did not exercise the guided path "
                f"({response.get('rtc.reason')!r}); the first real guided "
                "request would pay the JAX compile instead."
            )
        logger.info("Pi0.5 RTC warmup finished in %.1f s", time.perf_counter() - start)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Run one request, guided if the client supplied usable history.

        Mirrors ``Pi05Wrapper._infer``'s response, adding an ``rtc.*`` section
        that reports what was actually applied -- which is not always what was
        asked for, since the horizon is clamped to the available overlap.
        """
        wrapper_start = time.perf_counter()

        prompt = self._resolve_prompt(inputs)
        openpi_observation = self._to_openpi_observation(inputs, prompt)

        prev_chunk, delay, horizon, reason = self._resolve_rtc_context(inputs)

        if prev_chunk is None:
            result = self.policy.infer(openpi_observation)
            actions = np.asarray(result["actions"], dtype=np.float32)
            infer_ms = float(
                result.get("policy_timing", {}).get("infer_ms", float("nan"))
            )
            rtc_info: dict[str, Any] = {"rtc.applied": False, "rtc.reason": reason}
        else:
            actions, infer_ms = self._infer_guided(
                openpi_observation, prev_chunk, delay, horizon
            )
            rtc_info = {
                "rtc.applied": True,
                "rtc.reason": None,
                "rtc.inference_delay": int(delay),
                "rtc.prefix_attention_horizon": int(horizon),
                "rtc.schedule": self.prefix_attention_schedule,
            }

        if actions.ndim != 2:
            raise ValueError(
                f"Expected Pi0.5 actions of shape (horizon, action_dim), got {actions.shape}"
            )

        actions_per_chunk = inputs.get(CONTEXT_ACTIONS_PER_CHUNK)
        if actions_per_chunk is not None:
            actions = actions[:actions_per_chunk]

        wrapper_ms = (time.perf_counter() - wrapper_start) * 1000.0

        logger.debug(
            "Pi0.5 RTC chunk shape=%s applied=%s delay=%s horizon=%s infer_ms=%.1f",
            actions.shape,
            rtc_info["rtc.applied"],
            rtc_info.get("rtc.inference_delay"),
            rtc_info.get("rtc.prefix_attention_horizon"),
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
    ) -> tuple[np.ndarray | None, int, int, str | None]:
        """
        Validate the RTC fields and clamp them into a usable window.

        Returns ``(padded_prev_chunk, delay, horizon, reason)``. A ``None``
        chunk means fall back to plain sampling, and ``reason`` says why.

        Raises:
            TypeError / ValueError:
                The request carries RTC history but it is malformed or
                self-inconsistent. See the module docstring on why this is not
                a quiet fallback.
        """
        leftover = inputs.get(CONTEXT_PREV_CHUNK)
        if leftover is None:
            return None, 0, 0, "no previous chunk supplied"

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
            return None, 0, 0, "previous chunk fully consumed"
        if overlap > horizon_length:
            raise ValueError(
                f"{CONTEXT_PREV_CHUNK!r} has {overlap} rows, which exceeds the "
                f"model's action horizon of {horizon_length}."
            )

        # consumed_steps is checked against the leftover length rather than
        # trusted: the two disagreeing means the client's alignment is off, and
        # a misaligned prefix guides the chunk toward the wrong timesteps.
        consumed = inputs.get(CONTEXT_CONSUMED_STEPS)
        if consumed is not None:
            consumed = int(consumed)
            if consumed < 0:
                raise ValueError(
                    f"{CONTEXT_CONSUMED_STEPS!r} must be >= 0, got {consumed}."
                )
            if consumed + overlap != horizon_length:
                raise ValueError(
                    f"RTC alignment mismatch: {CONTEXT_CONSUMED_STEPS!r}={consumed} "
                    f"plus {overlap} leftover rows should equal the action "
                    f"horizon {horizon_length}. The leftover must be the "
                    "previous chunk with exactly its consumed rows dropped, so "
                    "that its row 0 is the timestep this request's row 0 covers."
                )

        delay = inputs.get(CONTEXT_INFERENCE_DELAY)
        delay = 0 if delay is None else int(delay)
        if delay < 0:
            raise ValueError(
                f"{CONTEXT_INFERENCE_DELAY!r} must be >= 0, got {delay}."
            )

        # d <= prefix_attention_horizon <= H - s. Both clamps are real: the
        # horizon cannot exceed the rows we actually have, and a client that is
        # later than the whole overlap gets everything pinned rather than an
        # error, because there is nothing better to do with that request.
        horizon = self.prefix_attention_horizon or horizon_length
        horizon = min(horizon, overlap)
        if delay > horizon:
            logger.warning(
                "RTC inference delay %d exceeds the usable overlap %d; the "
                "whole prefix will be pinned and the chunk cannot react. The "
                "client is running later than this chunk size supports.",
                delay,
                horizon,
            )
            delay = horizon

        # Pad to H by repeating the last known action. The weight schedule is
        # zero beyond `horizon <= overlap`, so the padding never influences the
        # result; repeating rather than zero-filling just keeps the array in a
        # sane range through the delta and normalization transforms.
        if overlap < horizon_length:
            padding = np.repeat(leftover[-1:], horizon_length - overlap, axis=0)
            leftover = np.concatenate([leftover, padding], axis=0)

        return np.asarray(leftover, dtype=np.float32), delay, horizon, None

    def _infer_guided(
        self,
        openpi_observation: dict[str, Any],
        prev_chunk: np.ndarray,
        delay: int,
        horizon: int,
    ) -> tuple[np.ndarray, float]:
        """
        One RTC-guided inference.

        Reimplements ``openpi.policies.policy.Policy.infer`` rather than calling
        it, because the previous chunk has to ride through the input transforms
        alongside the observation and then be handed to a different sampler.
        That means touching ``Policy``'s private attributes; it is the only
        coupling to OpenPI internals in this wrapper, and the reason the module
        pins a config name to a checkpoint.
        """
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        if self._rtc_sampler is None:
            self._rtc_sampler = self._build_rtc_sampler()

        policy = self.policy

        # The previous chunk arrives in raw robot space, the same space the
        # `action_chunk` response is in. Guidance happens in the model's space:
        # delta against the *current* state, quantile-normalized, padded to
        # action_dim. OpenPI's input pipeline already does all three to an
        # "actions" key (LiberoInputs forwards it, DeltaActions rebases it on
        # data["state"], Normalize and PadStatesAndActions finish the job), so
        # feed it through instead of reimplementing the chain and desyncing
        # from how the checkpoint was trained.
        inputs = dict(openpi_observation)
        inputs["actions"] = prev_chunk.copy()  # DeltaActions subtracts in place.

        transformed = policy._input_transform(inputs)
        prev_chunk_model = transformed.pop("actions")

        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed)
        prev_chunk_model = jnp.asarray(prev_chunk_model)[np.newaxis, ...]

        policy._rng, sample_rng = jax.random.split(policy._rng)
        observation = _model.Observation.from_dict(batched)

        start = time.monotonic()
        actions = self._rtc_sampler(
            sample_rng, observation, prev_chunk_model, delay, horizon
        )
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
                "name": "pi05_rtc",
                "action_horizon": self.action_horizon,
                "prefix_attention_horizon": self.prefix_attention_horizon,
                "prefix_attention_schedule": self.prefix_attention_schedule,
                "max_guidance_weight": self.max_guidance_weight,
            }
        )
        return metadata
