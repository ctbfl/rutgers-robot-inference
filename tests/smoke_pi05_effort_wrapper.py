"""Real checkpoint/data inference validation. Runs no robot and opens no server.

Run once per variant in the OpenPI environment on an available GPU.
"""

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np

from ruri.common import zmq as codec
from ruri.server.serve import handle_request
from ruri.server.wrappers.pi05.pi05_effort import Pi05EffortWrapper, Pi05TrainRTCEffortWrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=("plain", "rtc"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=5000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cls = Pi05TrainRTCEffortWrapper if args.variant == "rtc" else Pi05EffortWrapper
    wrapper = cls(args.checkpoint, warmup=False)

    import jax
    from openpi import transforms
    from openpi.training import data_loader
    from pi05_effort.config import make_train_config

    config = make_train_config(args.variant, "smoke_wrapper", args.output.parent)
    dc = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(dc, config.model.action_horizon, config.model)
    raw = dataset[args.index]

    def rgb(key):
        image = np.asarray(raw[key])
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    request = {
        "observation.state": np.asarray(raw["observation.state"]),
        "observation.joint_effort": np.asarray(raw["observation.joint_effort"]),
        "observation.images.top": rgb("observation.images.top"),
        "observation.images.wrist": rgb("observation.images.hand"),
        "prompt": raw["prompt"],
    }
    # Compare the independent training pipeline against the wrapper pipeline.
    reference = transforms.compose([
        *dc.repack_transforms.inputs, *dc.data_transforms.inputs,
        transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
        *dc.model_transforms.inputs,
    ])(copy.deepcopy(raw))
    mapped = {wrapper.INPUT_MAPPING.get(k, k): v for k, v in request.items()}
    online = wrapper.policy._input_transform(wrapper._to_openpi_observation(mapped, request["prompt"]))
    for key, value in online.items():
        for a, b in zip(jax.tree.leaves(value), jax.tree.leaves(reference[key]), strict=True):
            np.testing.assert_array_equal(a, b)

    rng = jax.random.key(42)

    def infer(observation):
        wrapper.policy._rng = rng
        result = codec.unpack(codec.pack(handle_request(wrapper, codec.unpack(codec.pack(observation)))))
        assert "error" not in result, result
        assert result["action_chunk"].shape == (30, 7)
        assert np.isfinite(result["action_chunk"]).all()
        return result

    first = infer(request)
    repeat = infer(request)
    np.testing.assert_array_equal(first["action_chunk"], repeat["action_chunk"])
    unrelated = copy.deepcopy(request)
    unrelated["observation.joint_effort"][:6] = 12345
    np.testing.assert_array_equal(first["action_chunk"], infer(unrelated)["action_chunk"])
    low, high = copy.deepcopy(request), copy.deepcopy(request)
    low["observation.joint_effort"][-1] = dc.norm_stats["state"].q01[7]
    high["observation.joint_effort"][-1] = dc.norm_stats["state"].q99[7]
    effort_diff = float(np.abs(infer(low)["action_chunk"] - infer(high)["action_chunk"]).max())
    assert effort_diff > 1e-6, "Effort input is inert"

    prefix_error = None
    conditioned_effort_diff = None
    if args.variant == "rtc":
        previous = first["action_chunk"][5:].copy()
        context = {
            "context.rtc.prev_chunk_left_over": previous,
            "context.rtc.consumed_steps": 5,
            "context.rtc.estimated_inference_delay_steps": 4,
        }
        conditioned = infer({**request, **context})
        assert conditioned["rtc.applied"] and conditioned["rtc.execute_from"] == 4
        prefix_error = float(np.abs(conditioned["action_chunk"][:4] - previous[:4]).max())
        np.testing.assert_allclose(conditioned["action_chunk"][:4], previous[:4], rtol=0, atol=2e-5)
        np.testing.assert_array_equal(previous, first["action_chunk"][5:])
        a, b = infer({**low, **context}), infer({**high, **context})
        conditioned_effort_diff = float(np.abs(a["action_chunk"][4:] - b["action_chunk"][4:]).max())
        assert conditioned_effort_diff > 1e-6

    # Warmup must compile the actual effort-aware entry points too.
    wrapper.default_prompt = request["prompt"]
    wrapper.warmup_image_shape = request["observation.images.top"].shape
    wrapper.warmup()
    result = {
        "status": "passed", "variant": args.variant, "checkpoint": args.checkpoint,
        "frame_index": args.index, "training_transform_exact": True,
        "fixed_rng_repeat_exact": True, "other_six_efforts_ignored": True,
        "warmup_passed": True, "shape": [30, 7],
        "effort_action_max_abs_diff": effort_diff,
        "rtc_prefix_max_abs_error": prefix_error,
        "rtc_effort_action_max_abs_diff": conditioned_effort_diff,
        "warm_infer_ms": repeat["timing.infer_ms"],
        "metadata": wrapper.describe(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
