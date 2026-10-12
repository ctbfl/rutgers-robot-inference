"""Inference-only snapshot of tight_insertion.sim_real.model (2026-10-11).

Uses OpenPI directly so deployment needs neither the simulator nor pi05-rtc.
Parameter names and sampling match DomainRTCConfig / Pi05DomainRTC and
pi05_train_rtc.model.Pi05TrainRTC. See docs/pi05_sim_real.md for provenance.
"""

import dataclasses

import einops
import flax.nnx as nnx
from flax import struct
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import Pi0, make_attn_mask, posemb_sincos
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at


@struct.dataclass
class DomainObservation(_model.Observation):
    domain_one_hot: object = None
    action_valid_mask: object = None

    @classmethod
    def from_dict(cls, data):
        if "domain_one_hot" not in data or "action_valid_mask" not in data:
            raise ValueError("Explicit domain and action validity are required")
        obs = super().from_dict(data)
        return obs.replace(domain_one_hot=data["domain_one_hot"],
                           action_valid_mask=data["action_valid_mask"])


def preprocess(rng, observation, *, train=False):
    if observation.domain_one_hot is None or observation.action_valid_mask is None:
        raise ValueError("Domain/validity was lost before model preprocessing")
    processed = _model.preprocess_observation(rng, observation, train=train)
    return observation.replace(images=processed.images, image_masks=processed.image_masks)


@dataclasses.dataclass(frozen=True)
class DomainRTCConfig(Pi0Config):
    pi05: bool = True
    action_horizon: int = 30
    discrete_state_input: bool = False
    delay_probs: tuple[float, ...] = (0.15, 0.05, 0.10, 0.20, 0.20, 0.15, 0.10, 0.05)

    def create(self, rng):
        return Pi05DomainRTC(self, rngs=nnx.Rngs(rng))

    def load(self, params, *, remove_extra_params=False):
        # Orbax can omit this empty structural node. Never synthesize learned
        # weights or intersect trees: either could hide a wrong architecture.
        params = dict(params)
        projection = dict(params["domain_proj"])
        if projection.get("bias") is not None:
            raise ValueError("The domain projection must remain bias-free")
        projection["bias"] = None
        params["domain_proj"] = projection
        return super().load(params, remove_extra_params=False)

    def inputs_spec(self, *, batch_size=1):
        obs, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            obs = DomainObservation(
                **{f.name: getattr(obs, f.name) for f in dataclasses.fields(obs)},
                domain_one_hot=jax.ShapeDtypeStruct((batch_size, 2), jnp.float32),
                action_valid_mask=jax.ShapeDtypeStruct((batch_size, self.action_horizon), jnp.bool_),
            )
        return obs, actions


class Pi05DomainRTC(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        if not self.pi05:
            raise ValueError("Domain RTC requires pi05=True")
        self.delay_probs = tuple(config.delay_probs)
        self.max_delay = len(self.delay_probs)
        if self.max_delay > self.action_horizon:
            raise ValueError("Delay support exceeds action horizon")
        width = self.action_in_proj.out_features
        self.tok_time_proj = nnx.Linear(
            width, width, kernel_init=nnx.initializers.zeros_init(), rngs=rngs,
        )
        self.domain_proj = nnx.Linear(
            2, width, use_bias=False, kernel_init=nnx.initializers.zeros_init(), rngs=rngs,
        )

    def _tok_time_emb(self, tok_time):
        b, h = tok_time.shape
        emb = posemb_sincos(
            tok_time.reshape(-1), self.action_in_proj.out_features,
            min_period=4e-3, max_period=4.0,
        )
        return emb.reshape(b, h, -1)

    def embed_suffix_rtc(self, obs, noisy_actions, timestep, tok_time):
        tokens, mask, ar, cond = super().embed_suffix(obs, noisy_actions, timestep)
        tokens = tokens.at[:, -self.action_horizon:].add(self.tok_time_proj(self._tok_time_emb(tok_time)))
        tokens = tokens.at[:, -self.action_horizon:].add(self.domain_proj(obs.domain_one_hot)[:, None, :])
        mask = mask.at[:, -self.action_horizon:].set(obs.action_valid_mask)
        return tokens, mask, ar, cond

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError("Use tight_insertion.sim_real.model for training")

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None,
                       prefix_actions=None, inference_delay=0):
        observation = preprocess(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        if prefix_actions is None:
            prefix_actions = jnp.zeros_like(noise)
            inference_delay = 0
        is_prefix = jnp.arange(self.action_horizon) < jnp.asarray(inference_delay)
        is_prefix = jnp.broadcast_to(is_prefix, (batch_size, self.action_horizon))
        x_init = jnp.where(is_prefix[..., None], prefix_actions, noise)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            time_b = jnp.broadcast_to(time, (batch_size,))
            tok_time = jnp.where(is_prefix, 0.0, time_b[:, None])
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_rtc(
                observation, x_t, time_b, tok_time,
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions,
                kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            x_next = x_t + dt * v_t
            x_next = jnp.where(is_prefix[..., None], prefix_actions, x_next)
            return x_next, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (x_init, 1.0))
        return x_0
