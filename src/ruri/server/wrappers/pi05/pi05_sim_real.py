"""Portable Pi05 sim/real training-time RTC wrapper; no client extensions.

Select domain='real' for a robot or domain='sim' for simulation on the server.
The client keeps the standard state / top / wrist / prompt / RTC wire keys.
Only construction imports OpenPI/JAX. Checkpoint preflight is CPU-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .pi05 import OPENPI_STATE, Pi05Wrapper
from .pi05_train_rtc import Pi05TrainRTCWrapper


ASSET_ID = "sim_real_h30"
SCHEMA = "pi05_sim_real_rtc_inference_v1"


def check_checkpoint(checkpoint_path):
    """Validate the v1 deployment contract without allocating model weights.

    This checks metadata, conditioning leaves and normalization integrity.
    The actual restore later checks every parameter key and shape strictly.
    """
    root = Path(checkpoint_path).expanduser().resolve()
    spec = json.loads((root / "model_spec.json").read_text())
    if spec.get("schema") != SCHEMA:
        raise ValueError(f"Expected checkpoint schema {SCHEMA!r}")
    if spec.get("model_class") != "tight_insertion.sim_real.model.DomainRTCConfig":
        raise ValueError("Expected DomainRTCConfig checkpoint")
    if spec.get("observation_class") != "tight_insertion.sim_real.model.DomainObservation":
        raise ValueError("Expected DomainObservation checkpoint")
    expected = {
        "action_dim": 32, "action_horizon": 30, "max_token_len": 200,
        "pi05": True, "discrete_state_input": False, "dtype": "bfloat16",
        "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m",
    }
    config = spec.get("model_config", {})
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported model_config.{key}: {config.get(key)!r}; expected {value!r}")
    allowed = set(expected) | {"delay_probs", "pytorch_compile_mode"}
    if set(config) - allowed:
        raise ValueError(f"Unsupported model_config keys: {set(config) - allowed}")
    probs = np.asarray(config.get("delay_probs", []), dtype=float)
    if (probs.ndim != 1 or not 2 <= probs.size <= 30 or not np.isfinite(probs).all()
            or np.any(probs < 0) or not np.isclose(probs.sum(), 1) or probs[-1] <= 0):
        raise ValueError("Invalid trained delay_probs")
    indicator = spec.get("indicator", {})
    if indicator.get("encoding") != {"sim": [1, 0], "real": [0, 1]}:
        raise ValueError("Unsupported domain encoding")
    for key in ("normalize_indicator", "diffuse_indicator", "classifier_free_guidance"):
        if indicator.get(key) is not False:
            raise ValueError(f"Unsupported indicator.{key}")
    if indicator.get("domain_dropout") != 0:
        raise ValueError("Unsupported domain_dropout")
    data = spec.get("data_contract", {})
    for key, expected_value in (("physical_state_dim", 7), ("physical_action_dim", 7),
                                ("model_action_dim", 32), ("fps", 30),
                                ("cameras", ["observation.images.top", "observation.images.hand"])):
        if data.get(key) != expected_value:
            raise ValueError(f"Unsupported data_contract.{key}")
    steps = spec.get("inference_flow_steps")
    if type(steps) is not int or steps <= 0:
        raise ValueError("inference_flow_steps must be a positive integer")
    norm_path = root / "assets" / ASSET_ID / "norm_stats.json"
    raw = norm_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec.get("checkpoint_norm_sha256"):
        raise ValueError("Checkpoint norm_stats SHA256 mismatch (or missing checksum)")
    stats = json.loads(raw).get("norm_stats", {})
    if set(stats) != {"state", "actions"}:
        raise ValueError("Expected shared state/actions normalization statistics")
    for name, fields in stats.items():
        for key in ("mean", "std", "q01", "q99"):
            value = np.asarray(fields.get(key), dtype=float)
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid norm_stats.{name}.{key}")
        if np.any(np.asarray(fields["q99"]) < fields["q01"]):
            raise ValueError(f"Reversed quantiles for {name}")
    commit = json.loads((root / "params/_CHECKPOINT_METADATA").read_text())
    if not commit.get("commit_timestamp_nsecs"):
        raise ValueError("Checkpoint params save has not committed")
    tree = json.loads((root / "params/_METADATA").read_text())["tree_metadata"]
    leaves = {
        tuple(entry["key"] for entry in item["key_metadata"])
        for item in tree.values()
        if item["value_metadata"].get("value_type") != "None"
    }
    for layer, parameter in (("domain_proj", "kernel"), ("tok_time_proj", "kernel"), ("tok_time_proj", "bias")):
        if ("params", layer, parameter, "value") not in leaves:
            raise ValueError(f"Missing checkpoint parameter {layer}/{parameter}")
    if ("params", "domain_proj", "bias", "value") in leaves:
        raise ValueError("domain_proj must remain bias-free")
    return spec


class Pi05SimRealWrapper(Pi05TrainRTCWrapper):
    def __init__(self, checkpoint_path, *, domain: str, default_prompt=None,
                 num_denoising_steps=None, default_inference_delay=4, warmup=True,
                 rtc_warmup=True, warmup_image_shape=(480, 640, 3)):
        if domain not in ("sim", "real"):
            raise ValueError("domain must explicitly be 'sim' or 'real'")
        self.domain = domain
        self.model_spec = check_checkpoint(checkpoint_path)
        if num_denoising_steps is None:
            num_denoising_steps = self.model_spec["inference_flow_steps"]
        if type(num_denoising_steps) is not int or num_denoising_steps <= 0:
            raise ValueError("num_denoising_steps must be a positive integer")
        max_delay = len(self.model_spec["model_config"]["delay_probs"]) - 1
        if type(default_inference_delay) is not int or not 1 <= default_inference_delay <= max_delay:
            raise ValueError(f"default_inference_delay must be between 1 and {max_delay}")
        self.default_inference_delay = default_inference_delay
        self.rtc_warmup = rtc_warmup
        # Reuse the RTC request/response contract without external train configs.
        Pi05Wrapper.__init__(
            self, checkpoint_path, SCHEMA, default_prompt=default_prompt,
            num_denoising_steps=num_denoising_steps, state_dim=7, warmup=warmup,
            warmup_image_shape=warmup_image_shape,
        )

    def _load_policy(self):
        from ._sim_real_policy import DomainPolicy
        return DomainPolicy(self.checkpoint_path, self.model_spec, self.domain, self.num_denoising_steps)

    def _to_openpi_observation(self, inputs, prompt):
        state = np.asarray(inputs.get(OPENPI_STATE), dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("observation.state must be finite with shape (7,)")
        return super()._to_openpi_observation(inputs, prompt)

    def _resolve_rtc_context(self, inputs):
        previous, delay, reason = super()._resolve_rtc_context(inputs)
        if previous is not None and not np.isfinite(previous).all():
            raise ValueError("RTC previous actions must be finite")
        return previous, delay, reason

    def _infer_conditioned(self, openpi_observation, prev_chunk, delay):
        result = self.policy.infer(openpi_observation, prev_chunk=prev_chunk, delay=delay)
        return np.asarray(result["actions"], dtype=np.float32), result["policy_timing"]["infer_ms"]

    def _infer(self, inputs):
        response = super()._infer(inputs)
        actions = response["action_chunk"]
        if actions.shape != (30, 7) or not np.isfinite(actions).all():
            raise ValueError("Expected finite sim/real actions of shape (30, 7)")
        return response

    def optional_more_metadata(self):
        metadata = super().optional_more_metadata()
        metadata.update({
            "name": "pi05_sim_real_rtc", "supports_rtc": True,
            "domain": self.domain, "domain_one_hot": self.model_spec["indicator"]["encoding"][self.domain],
            "model_schema": SCHEMA, "completed_updates": self.model_spec.get("completed_updates"),
            "parameter_kind": self.model_spec.get("parameter_kind"), "fps": 30,
            "checkpoint_norm_sha256": self.model_spec["checkpoint_norm_sha256"],
        })
        return metadata
