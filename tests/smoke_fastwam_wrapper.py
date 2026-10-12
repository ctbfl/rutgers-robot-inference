"""Opt-in real checkpoint/held-out-data parity check; NEVER opens robot hardware.

Run in the FastWAM environment with RURI_ROOT/src on PYTHONPATH. Dataset
side-effects (its stats export) are confined to a temporary work directory.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from ruri.server.wrappers.fastwam import FastWAMWrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fastwam-root", default="/common/users/jh2400/FastWAM")
    parser.add_argument("--index", type=int, default=0, help="Index into the held-out validation dataset")
    parser.add_argument("--prompt", default="insert the row of five objects, from left to right, into the bottom row of holes on the manipulation-net board.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile-action-infer", action="store_true",
                        help="Exercise compiled constructor warmup and compare against eager native inference")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.fastwam_root) / "src"))
    import torch
    from hydra.utils import instantiate
    from fastwam.utils import misc

    torch.set_num_threads(4)
    wrapper = FastWAMWrapper(args.checkpoint, fastwam_root=args.fastwam_root,
                             default_prompt=args.prompt, device=args.device, warmup=args.compile_action_infer,
                             compile_action_infer=args.compile_action_infer)
    runtime = wrapper._runtime
    with tempfile.TemporaryDirectory(prefix="ruri-fastwam-smoke-") as temporary:
        misc.register_work_dir(temporary)
        ds = instantiate(runtime.cfg.data.val, pretrained_norm_stats=str(runtime.stats_path),
                         text_embedding_cache_dir=str(runtime.text_cache_dir))
        base = ds.lerobot_dataset
        processor = base.processor
        try:
            base.processor = None
            raw = base[args.index]
        finally:
            base.processor = processor
        sample = ds[args.index]
        state = raw["state"]["default"][0].numpy().astype(np.float32)
        images = {key: raw["images"][key][0].permute(1, 2, 0).numpy().copy() for key in ("top", "hand")}
        image, proprio = runtime.prepare(state, images)
        torch.testing.assert_close(image[0], sample["video"][:, 0], rtol=0, atol=0)
        torch.testing.assert_close(proprio, sample["proprio"][0], rtol=0, atol=0)
        context, mask = runtime.text_context(args.prompt)
        torch.testing.assert_close(context, sample["context"], rtol=0, atol=0)
        torch.testing.assert_close(mask, sample["context_mask"], rtol=0, atol=0)

        # Exact elementary inverse test, including a gripper distinct from q0.
        target = raw["action"]["default"].float()
        batch = {"action": {"default": target.clone()}, "state": {"default": torch.from_numpy(state).unsqueeze(0)}}
        relative = processor.action_state_transform(copy.deepcopy(batch))["action"]["default"]
        # Use the affine transform directly: forward() clips at +/-5, which
        # cannot be inverted for outliers and is not an inverse-transform bug.
        normalized = relative * runtime.action_norm.scale + runtime.action_norm.offset
        restored = runtime.restore_action(normalized, state)
        np.testing.assert_allclose(restored, target.numpy(), rtol=1e-5, atol=1e-5)
        assert restored.shape == (wrapper.output_chunk_size, 7)

        request = {"observation.state": state, "observation.images.top": images["top"],
                   "observation.images.wrist": images["hand"]}
        response = wrapper.infer(request)
        # Independent input construction through the native validation dataset,
        # and independent output restoration (no runtime.prepare/restore here).
        with torch.inference_mode():
            prediction = runtime.model.infer_action(
                prompt=None, input_image=sample["video"][:, 0].unsqueeze(0),
                action_horizon=wrapper.output_chunk_size, proprio=sample["proprio"][0],
                context=sample["context"], context_mask=sample["context_mask"],
                num_inference_steps=20, seed=42, rand_device="cpu", tiled=False,
                compile_action_infer=False,
            )["action"]
        reference = processor.normalizer.normalizers["action"]["default"].backward(prediction).float()
        reference[:, :6] += torch.from_numpy(state[:6])
        normalized_error = (response["action_chunk"] - reference.numpy()) * runtime.action_norm.scale.numpy()
        if args.compile_action_infer:
            # BF16 kernel fusion changes rounding. This is an integration
            # regression tolerance in model-normalized units, not a safety bound.
            assert float(np.abs(normalized_error).max()) <= 0.01, "Compiled/eager action mismatch"
        else:
            np.testing.assert_allclose(response["action_chunk"], reference.numpy(), rtol=1e-5, atol=1e-5)
        repeat = wrapper.infer(request)
        np.testing.assert_array_equal(response["action_chunk"], repeat["action_chunk"])
        assert np.isfinite(repeat["action_chunk"]).all()
        print(json.dumps({
            "result": "PASS", "validation_index": args.index,
            "shape": list(repeat["action_chunk"].shape),
            "preprocessing_exact": True, "inverse_transform_passed": True,
            "native_action_max_abs_diff": float(np.max(np.abs(reference.numpy() - response["action_chunk"]))),
            "native_normalized_action_max_abs_diff": float(np.abs(normalized_error).max()),
            "compile_action_infer": args.compile_action_infer,
            "fixed_seed_repeat_exact": True,
            "first_infer_ms": response["timing.infer_ms"],
            "warm_infer_ms": repeat["timing.infer_ms"],
            "warm_wrapper_ms": repeat["timing.wrapper_ms"],
            "metadata": wrapper.describe(),
        }, indent=2))


if __name__ == "__main__":
    main()
