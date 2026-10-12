"""OpenPI preprocessing and sampling with an explicit DomainObservation."""

import time

import jax
import jax.numpy as jnp
import numpy as np
from openpi import transforms
from openpi.models import model as base_model
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs
from openpi.shared import nnx_utils, normalize

from ._sim_real_model import DomainObservation, DomainRTCConfig


def make_transforms(config, stats):
    # Same order as training's Dataset, including six relative joints and an
    # absolute gripper. State is still needed to rebase/decode action chunks.
    delta_mask = transforms.make_bool_mask(6, -1)
    inputs = transforms.compose([
        LiberoInputs(config.model_type),
        transforms.DeltaActions(delta_mask),
        transforms.Normalize(stats, use_quantiles=True),
        transforms.ResizeImages(224, 224),
        transforms.TokenizePrompt(
            PaligemmaTokenizer(config.max_token_len),
            discrete_state_input=config.discrete_state_input,
        ),
        transforms.PadStatesAndActions(config.action_dim),
    ])
    outputs = transforms.compose([
        transforms.Unnormalize(stats, use_quantiles=True),
        transforms.AbsoluteActions(delta_mask),
        LiberoOutputs(),
    ])
    return inputs, outputs


class DomainPolicy:
    def __init__(self, checkpoint, spec, domain, num_steps):
        config_args = dict(spec["model_config"])
        config_args["delay_probs"] = tuple(config_args["delay_probs"])
        config = DomainRTCConfig(**config_args)
        # Match OpenPI's serving precision and keep weights on the device.
        # Host numpy leaves would be copied again on EVERY module_jit call.
        # Exported EMA files are not modified by the bfloat16 restore.
        params = base_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
        self._model = config.load(params)
        self._model.eval()
        self._assert_conditioning_parameters()
        stats = normalize.load(checkpoint / "assets/sim_real_h30")
        self._input_transform, self._output_transform = make_transforms(config, stats)
        self._sample_actions = nnx_utils.module_jit(self._model.sample_actions)
        self._rng = jax.random.key(0)
        self._sample_kwargs = {"num_steps": num_steps}
        self._domain = tuple(spec["indicator"]["encoding"][domain])

    def _assert_conditioning_parameters(self):
        domain = np.asarray(self._model.domain_proj.kernel.value, dtype=np.float32)
        time_kernel = np.asarray(self._model.tok_time_proj.kernel.value, dtype=np.float32)
        if not np.isfinite(domain).all() or not np.isfinite(time_kernel).all():
            raise ValueError("Non-finite domain/RTC conditioning parameters")
        if np.any(np.linalg.norm(domain, axis=1) == 0) or np.linalg.norm(time_kernel) == 0:
            raise ValueError("Untrained domain_proj or tok_time_proj; check the checkpoint")

    def prepare(self, observation, prev_chunk=None):
        inputs = dict(observation)
        if prev_chunk is not None:
            # DeltaActions mutates actions: never alter the client's buffer.
            inputs["actions"] = np.asarray(prev_chunk, dtype=np.float32).copy()
        inputs = self._input_transform(inputs)
        prefix = inputs.pop("actions", None)
        # LiberoInputs drops unknown fields, so inject AFTER all transforms.
        inputs["domain_one_hot"] = np.asarray(self._domain, dtype=np.float32)
        inputs["action_valid_mask"] = np.ones(self._model.action_horizon, dtype=bool)
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        obs = DomainObservation.from_dict(batched)
        prefix = None if prefix is None else jnp.asarray(prefix)[None, ...]
        return obs, prefix

    def infer(self, observation, *, prev_chunk=None, delay=0, noise=None):
        obs, prefix = self.prepare(observation, prev_chunk)
        self._rng, sample_rng = jax.random.split(self._rng)
        kwargs = dict(self._sample_kwargs)
        if prefix is not None:
            kwargs.update(prefix_actions=prefix, inference_delay=jnp.asarray(delay))
        if noise is not None:
            kwargs["noise"] = jnp.asarray(noise)
        start = time.monotonic()
        actions = jax.block_until_ready(self._sample_actions(sample_rng, obs, **kwargs))
        infer_ms = (time.monotonic() - start) * 1000.0
        outputs = self._output_transform({
            "state": np.asarray(obs.state[0]), "actions": np.asarray(actions[0]),
        })
        outputs["policy_timing"] = {"infer_ms": infer_ms}
        return outputs
