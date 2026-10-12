#!/usr/bin/env python
"""Serve a complete FastWAM weights export; no robot hardware is opened."""

import argparse
import logging

from ruri.server.serve import serve
from ruri.server.wrappers.fastwam import FastWAMWrapper


DEFAULT_CHECKPOINT = (
    "/common/users/jh2400/training_results/fastwam/260906_verified_zero2_row1/"
    "checkpoints/weights/step_020000.pt"
)
DEFAULT_PROMPT = (
    "insert the row of five objects, from left to right, into the bottom row "
    "of holes on the manipulation-net board."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fastwam-root", default="/common/users/jh2400/FastWAM")
    parser.add_argument("--config", help="Saved run config.yaml; inferred from checkpoint if omitted")
    parser.add_argument("--stats", help="Saved run dataset_stats.json; inferred if omitted")
    parser.add_argument("--text-cache-dir")
    parser.add_argument("--model-base-path", help="Local frozen-backbone assets; defaults to FASTWAM_ROOT/checkpoints")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", action="store_true", help="Use fresh action noise per request")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--compile-action-infer", action=argparse.BooleanOptionalAction, default=True,
                        help="Compile video prefill/action denoising with CUDA graph acceleration (default: enabled)")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--bind", default="tcp://*:5560")
    parser.add_argument("--name", default="fastwam")
    parser.add_argument("--advertise-host")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    import torch

    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    wrapper = FastWAMWrapper(
        args.checkpoint, default_prompt=args.prompt, fastwam_root=args.fastwam_root,
        config_path=args.config, stats_path=args.stats, text_cache_dir=args.text_cache_dir,
        model_base_path=args.model_base_path, device=args.device,
        diffusion_steps=args.diffusion_steps, seed=None if args.random_seed else args.seed,
        warmup=not args.no_warmup,
        compile_action_infer=args.compile_action_infer, warmup_iters=args.warmup_iters,
    )
    serve(wrapper, args.bind, name=args.name, advertise_host=args.advertise_host)


if __name__ == "__main__":
    main()
