"""
Real-Time Chunking (RTC) sampling for OpenPI Pi0/Pi0.5 models.

RTC lets a chunked flow-matching policy keep moving while the next chunk is
still being computed. The robot executes the tail of the previous chunk during
inference, so by the time the new chunk lands, its first few actions are
already spoken for. RTC turns that into an inpainting problem: generate the new
chunk conditioned on the previous one, weighting the overlap so the committed
prefix is reproduced almost exactly and the agreement decays away over the rest.

The mechanism is a guidance term added to the learned velocity field at every
Euler step of the flow -- pseudoinverse guidance (PiGDM). Nothing is retrained;
this is entirely an inference-time intervention.

Reference: Black, Galliker, Levine, "Real-Time Execution of Action Chunking Flow
Policies" (arXiv 2506.07339). The two functions below are ports of the authors'
reference implementation, ``Physical-Intelligence/real-time-chunking-kinetix``
(``src/model.py``: ``get_prefix_weights`` and ``realtime_action``).

Time convention
---------------
The reference implementation integrates t: 0 -> 1 with t=0 noise and t=1 data.
OpenPI runs the opposite convention (``pi0.py``: t=1 noise, t=0 data,
``dt = -1/num_steps``), so the port is not a copy. Two things change:

1. The clean-sample estimate. Reference: ``x_1 = x_t + v_t * (1 - t)``.
   OpenPI:  ``x_0_hat = x_t - t * v_t``.

2. The sign of the correction. OpenPI steps ``x + dt * v`` with ``dt < 0``, so
   moving x along the correction requires *subtracting* it from the velocity.
   Getting this backwards does not crash -- it pushes the new chunk away from
   the previous one, which reads as a policy that has become erratic rather
   than as a bug. See ``_sample_actions_rtc``.

The guidance *magnitude* needs no conversion: written out, the reference's
``c * inv_r2`` is symmetric under t <-> 1-t. See :func:`guidance_weight`.

What this module does not do
----------------------------
It knows nothing about normalization, delta actions, or the RURI wire format.
``prev_chunk`` arrives already in the model's own action space -- normalized,
delta-relative-to-current-state, padded to ``action_dim``. Producing it is
:mod:`ruri.server.wrappers.pi05.pi05_rtc`'s job, and it does so by feeding the
previous chunk through OpenPI's own input transforms rather than reimplementing
them.

Importing this module pulls in JAX and OpenPI. Import it lazily, the way
``pi05.py`` defers its own OpenPI import, so that ``import ruri`` still works on
a robot-side machine.
"""

from __future__ import annotations

from typing import Any, Literal

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask


# Matches PrefixAttentionSchedule in the reference implementation.
PrefixAttentionSchedule = Literal["ones", "zeros", "linear", "exp"]

PREFIX_ATTENTION_SCHEDULES: tuple[str, ...] = ("ones", "zeros", "linear", "exp")

# The reference default, and what LeRobot ships for 10-step flow matching
# (Pi0, Pi0.5, SmolVLA). Raising it enforces continuity harder at the cost of
# reactivity.
DEFAULT_MAX_GUIDANCE_WEIGHT = 10.0

DEFAULT_PREFIX_ATTENTION_SCHEDULE: PrefixAttentionSchedule = "exp"


def get_prefix_weights(
    start: Any,
    end: Any,
    total: int,
    schedule: PrefixAttentionSchedule,
) -> jnp.ndarray:
    """
    Per-timestep agreement weights against the previous chunk.

    Ported from the reference implementation. ``start`` and ``end`` may be
    traced, so the caller can vary the inference delay every request without
    triggering a recompile; ``schedule`` and ``total`` are Python values and
    are baked in.

    Args:
        start:
            Inference delay ``d``, in timesteps. Indices below this are
            already committed to execution and get full weight.
        end:
            Prefix attention horizon. Weight reaches zero here and stays
            there, which is what leaves the tail of the chunk free to react
            to the new observation.
        total:
            Chunk length ``H``. Length of the returned vector.
        schedule:
            How weight decays between ``start`` and ``end``.

            ``ones``    Full weight everywhere before ``end``.
            ``zeros``   Full weight before ``start``, nothing after: a hard
                        prefix with no soft overlap.
            ``linear``  Linear decay from ``start`` to ``end``.
            ``exp``     Sharper-than-linear decay. The reference default.

    Returns:
        ``(total,)`` float array in [0, 1], non-increasing.
    """
    if schedule not in PREFIX_ATTENTION_SCHEDULES:
        raise ValueError(
            f"Unknown prefix attention schedule {schedule!r}; "
            f"expected one of {PREFIX_ATTENTION_SCHEDULES}."
        )

    index = jnp.arange(total)
    # A delay past the horizon would make the decay denominator negative; the
    # reference clamps rather than erroring, and so do we.
    start = jnp.minimum(start, end)

    if schedule == "ones":
        weights = jnp.ones(total, dtype=jnp.float32)
    elif schedule == "zeros":
        weights = (index < start).astype(jnp.float32)
    else:
        # 1 for index < start, then falling to 0 at index == end.
        weights = jnp.clip(
            (start - 1 - index).astype(jnp.float32) / (end - start + 1).astype(jnp.float32) + 1.0,
            0.0,
            1.0,
        )
        if schedule == "exp":
            weights = weights * jnp.expm1(weights) / (jnp.e - 1.0)

    return jnp.where(index >= end, 0.0, weights)


def guidance_weight(time: jnp.ndarray, max_guidance_weight: float) -> jnp.ndarray:
    """
    PiGDM guidance strength at flow time ``time``.

    The reference computes ``min(c * inv_r2, max_guidance_weight)`` with
    ``c = (1-t)/t`` and ``inv_r2 = (t^2 + (1-t)^2) / (1-t)^2``. Multiplied out
    that is::

        c * inv_r2 = ((1-t)^2 + t^2) / ((1-t) * t)

    which is symmetric under ``t <-> 1-t``, so it transfers to OpenPI's flipped
    time convention unchanged. It is U-shaped: 2.0 at t=0.5, blowing up at both
    ends, where the clamp takes over.

    The reference reaches the endpoint singularity via ``nan_to_num(posinf=...)``.
    A guarded denominator gets to the same place without ever materializing an
    inf, which matters here because the result multiplies a VJP output.
    """
    denominator = jnp.maximum((1.0 - time) * time, 1e-6)
    weight = ((1.0 - time) ** 2 + time**2) / denominator
    return jnp.minimum(weight, max_guidance_weight)


def _sample_actions_rtc(
    model: Any,
    rng: jax.Array,
    observation: _model.Observation,
    prev_chunk: jnp.ndarray,
    *,
    num_steps: int,
    inference_delay: jnp.ndarray,
    prefix_attention_horizon: jnp.ndarray,
    prefix_attention_schedule: PrefixAttentionSchedule,
    max_guidance_weight: float,
) -> jnp.ndarray:
    """
    Pi0/Pi0.5 flow-matching sampling with RTC prefix guidance.

    Mirrors ``Pi0.sample_actions`` step for step -- same prefix KV cache, same
    Euler ``while_loop``, same exit condition -- and adds the guidance term.
    Keeping the two visually diffable is deliberate: this function reaches into
    ``embed_prefix`` / ``embed_suffix`` / ``action_out_proj``, so it has to be
    re-checked whenever OpenPI's sampler changes.

    Args:
        model: A loaded ``openpi.models.pi0.Pi0``.
        rng: Sampling key, used only to draw the initial noise.
        observation: Already batched, not yet preprocessed.
        prev_chunk:
            ``(b, action_horizon, action_dim)``, in the model's own action
            space, aligned so that index j means the same wall-clock timestep
            as index j of the chunk being generated.
        num_steps: Flow-matching integration steps. Static.
        inference_delay: ``d``. May be traced.
        prefix_attention_horizon: May be traced.
        prefix_attention_schedule: Static.
        max_guidance_weight: Static.

    Returns:
        ``(b, action_horizon, action_dim)`` in the model's action space.
    """
    observation = _model.preprocess_observation(None, observation, train=False)

    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    # Prefix (images + language) is observation-only, so it is computed once and
    # cached. This is also why the VJP below stays affordable: it differentiates
    # the suffix pass only -- ten action tokens through the action expert --
    # while the cached prefix, which is nearly all of the compute, is a constant.
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    # (1, H, 1): broadcast over batch and action dim.
    weights = get_prefix_weights(
        inference_delay,
        prefix_attention_horizon,
        model.action_horizon,
        prefix_attention_schedule,
    )[None, :, None]

    def velocity(x_t: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_cross_mask = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate([prefix_cross_mask, suffix_attn_mask], axis=-1)
        suffix_positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        )

        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return model.action_out_proj(suffix_out[:, -model.action_horizon :])

    def step(carry):
        x_t, time = carry

        def denoiser(x: jnp.ndarray):
            v = velocity(x, time)
            # OpenPI's forward process is x_t = t*noise + (1-t)*actions with
            # v = noise - actions, so the clean sample falls out as x_t - t*v.
            return x - time * v, v

        # One forward plus one reverse pass. `vjp_fn` applies J^T, where
        # J = d(x_0_hat)/d(x_t) -- the pseudoinverse guidance of PiGDM, which is
        # what makes this better than simply overwriting the prefix each step.
        x_0_hat, vjp_fn, v_t = jax.vjp(denoiser, x_t, has_aux=True)

        error = (prev_chunk - x_0_hat) * weights
        (correction,) = vjp_fn(error)

        # Subtract, do not add: `dt` is negative, so this is the direction that
        # moves x_0_hat *toward* the previous chunk. See the module docstring.
        v_guided = v_t - guidance_weight(time, max_guidance_weight) * correction

        return x_t + dt * v_guided, time + dt

    def cond(carry):
        _, time = carry
        # Robust to floating-point error, as upstream.
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


def make_rtc_sampler(
    model: Any,
    *,
    num_steps: int,
    prefix_attention_schedule: PrefixAttentionSchedule = DEFAULT_PREFIX_ATTENTION_SCHEDULE,
    max_guidance_weight: float = DEFAULT_MAX_GUIDANCE_WEIGHT,
):
    """
    JIT-compile :func:`_sample_actions_rtc` against a frozen copy of ``model``.

    Follows ``openpi.shared.nnx_utils.module_jit``: split the module once, pass
    the state as a plain argument, and merge inside the traced function. That
    helper only accepts bound methods of an ``nnx.Module``, which this is not,
    so the pattern is reproduced here rather than reused.

    ``inference_delay`` and ``prefix_attention_horizon`` are passed as arrays,
    not Python ints, so that a delay that changes from request to request costs
    nothing. Everything genuinely structural is closed over and static.

    Returns:
        ``(rng, observation, prev_chunk, inference_delay,
        prefix_attention_horizon) -> actions``
    """
    graphdef, state = nnx.split(model)

    def fun(state, rng, observation, prev_chunk, inference_delay, prefix_attention_horizon):
        merged = nnx.merge(graphdef, state)
        return _sample_actions_rtc(
            merged,
            rng,
            observation,
            prev_chunk,
            num_steps=num_steps,
            inference_delay=inference_delay,
            prefix_attention_horizon=prefix_attention_horizon,
            prefix_attention_schedule=prefix_attention_schedule,
            max_guidance_weight=max_guidance_weight,
        )

    jitted = jax.jit(fun)

    def sample(rng, observation, prev_chunk, inference_delay, prefix_attention_horizon):
        return jitted(
            state,
            rng,
            observation,
            prev_chunk,
            jnp.asarray(inference_delay, dtype=jnp.int32),
            jnp.asarray(prefix_attention_horizon, dtype=jnp.int32),
        )

    return sample
