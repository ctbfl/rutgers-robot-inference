"""Serve a Pi05 sim/real RTC checkpoint using the existing RURI wire contract."""

import argparse
import json
import logging
import os

from ruri.server.wrappers.pi05.pi05_sim_real import Pi05SimRealWrapper, check_checkpoint


def self_test(wrapper):
    """Exercise learned domain conditioning and absolute RTC prefix roundtrip.

    Synthetic images validate plumbing, not task accuracy. Opens no robot I/O.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from openpi.shared import normalize

    stats = normalize.load(wrapper.checkpoint_path / "assets/sim_real_h30")
    state = np.asarray(stats["state"].mean, dtype=np.float32)
    request = {
        "observation.state": state,
        "observation.images.top": np.zeros(wrapper.warmup_image_shape, dtype=np.uint8),
        "observation.images.wrist": np.zeros(wrapper.warmup_image_shape, dtype=np.uint8),
    }
    response = wrapper.infer(request)
    mapped = {wrapper.INPUT_MAPPING.get(k, k): v for k, v in request.items()}
    raw = wrapper._to_openpi_observation(mapped, wrapper._resolve_prompt(mapped))
    policy = wrapper.policy
    parameter_leaves = jax.tree.leaves(nnx.state(policy._model, nnx.Param))
    if not all(isinstance(x, jax.Array) for x in parameter_leaves):
        raise RuntimeError("Model weights must be device-resident JAX arrays")
    obs, _ = policy.prepare(raw)
    noise = jax.random.normal(jax.random.key(8), (1, 30, 32))
    predictions = []
    for domain in ([1., 0.], [0., 1.]):
        value = policy._sample_actions(
            jax.random.key(1), obs.replace(domain_one_hot=jnp.asarray([domain])),
            noise=noise, **policy._sample_kwargs,
        )
        predictions.append(np.asarray(value))
    difference = float(np.max(np.abs(predictions[0][:, :, :7] - predictions[1][:, :, :7])))
    if not all(np.isfinite(x).all() for x in predictions) or difference <= 1e-7:
        raise RuntimeError(f"Domain flip did not change finite model actions: {difference}")
    errors = {}
    previous = np.asarray(response["action_chunk"])
    # State changes between chunks: the previous ABSOLUTE prefix must survive
    # conversion to deltas relative to the new observation, then back again.
    new_state = state + np.array([1, -1, 2, -2, 1, -1, 0], np.float32)
    for delay in (1, wrapper.max_inference_delay):
        result = wrapper.infer({
            **request, "observation.state": new_state,
            "context.rtc.prev_chunk_left_over": previous[5:],
            "context.rtc.consumed_steps": 5,
            "context.rtc.estimated_inference_delay_steps": delay,
        })
        error = float(np.max(np.abs(result["action_chunk"][:delay] - previous[5:5 + delay])))
        if not result["rtc.applied"] or result["rtc.execute_from"] != delay or error >= 1e-3:
            raise RuntimeError(f"RTC prefix roundtrip failed at delay {delay}: {error}")
        errors[str(delay)] = error
    return {
        "passed": True, "devices": [str(d) for d in jax.devices()],
        "device_resident_parameters": True,
        "domain": wrapper.domain, "checkpoint": str(wrapper.checkpoint_path),
        "domain_flip_max_action_difference": difference,
        "absolute_prefix_max_errors": errors, "action_shape": list(previous.shape),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Export containing params/, assets/, model_spec.json")
    parser.add_argument("--domain", required=True, choices=("real", "sim"))
    parser.add_argument("--prompt", help="Exact training task string; clients can override per request")
    parser.add_argument("--num-denoising-steps", type=int, default=None, help="Default: checkpoint inference_flow_steps")
    parser.add_argument("--default-inference-delay", type=int, default=4)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--bind", default="tcp://*:5557")
    parser.add_argument("--name", default="pi05_sim_real_rtc")
    parser.add_argument("--advertise-host")
    parser.add_argument("--check-only", action="store_true", help="Check export metadata and norm hash without JAX/GPU")
    parser.add_argument("--self-test", action="store_true", help="Run model/domain/RTC smoke checks and exit; no server or robot")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.check_only:
        spec = check_checkpoint(args.checkpoint)
        print(json.dumps({"metadata_check_passed": True, "completed_updates": spec.get("completed_updates"),
                          "domain": args.domain, "full_weights_restored": False}, indent=2))
        return
    if (args.self_test or not args.no_warmup) and not args.prompt:
        parser.error("--prompt is required for warmup/self-test; use an exact training task string")
    # Avoid reserving most of a shared GPU before the model has loaded.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    if jax.default_backend() != "gpu":
        raise RuntimeError("JAX cannot use a GPU. Verify the deployment environment before loading the model.")
    wrapper = Pi05SimRealWrapper(
        args.checkpoint, domain=args.domain, default_prompt=args.prompt,
        num_denoising_steps=args.num_denoising_steps,
        default_inference_delay=args.default_inference_delay, warmup=not args.no_warmup,
    )
    if args.self_test:
        print(json.dumps(self_test(wrapper), indent=2))
        return
    from ruri.server.serve import serve
    serve(wrapper, args.bind, name=args.name, advertise_host=args.advertise_host)


if __name__ == "__main__":
    main()
