#!/usr/bin/env python
"""
Example: serve the fine-tuned DM0.5 tight-insertion policy over ZeroMQ.

This wires the three pieces together:

    DM05Wrapper (OpenDM + PyTorch)  ->  ruri.server.serve.serve()  ->  ZMQ REP

Defaults point at the 10k-step LoRA checkpoint of the
``dm05_tight_insertion_lora`` run on this machine, so the example runs with no
arguments once that step exists.

Launch
------
DM0.5 needs the OpenDM environment (torch 2.11 + transformers 5.x + peft).
Prefer the shell script, which sets the GPU and the caches::

    /common/users/jh2400/rutgers-robot-inference/examples/shell_scripts/serve_dm05.sh

Directly, if you would rather::

    /common/users/jh2400/conda_envs/opendm/bin/python \\
      /common/users/jh2400/rutgers-robot-inference/examples/server_dm05.py

Startup takes ~40 s: loading the 11.7 GB base model, merging the LoRA adapter,
then two warmup inferences that pay for CUDA autotuning and DM0.5's suffix
graph capture. The bind address is printed only once the server can actually
answer. Steady-state VRAM is ~13 GB.

Request / response
------------------
Messages are MessagePack-encoded by ``ruri.common.zmq``, which carries numpy
arrays natively. A client sends::

    {
        "type": "infer",
        "observation.state":        np.ndarray (7,) float,
        "observation.images.top":   np.ndarray (H, W, 3) uint8, RGB,
        "observation.images.wrist": np.ndarray (H, W, 3) uint8, RGB,
        "prompt":                   str,   # optional, defaults to --prompt
    }

and gets back::

    {
        "action_chunk": np.ndarray (50, 7) float32,   # absolute joint targets
        "prompt": str,
        "timing.infer_ms": float,
        "timing.wrapper_ms": float,
    }

``{"type": "metadata"}`` returns the input contract instead. A failed request
comes back as ``{"error": "..."}``.

These are the same keys ``server_pi05.py`` serves, and both return absolute
joint targets in the dataset's units, so the same client can be pointed at
either for an A/B on the same robot. The horizons differ: Pi0.5 returns 10
actions, DM0.5 returns 50. Both return their whole chunk; the scheduler slices.
"""

import argparse
import logging

from ruri.server.serve import serve
from ruri.server.wrappers.dm05.dm05 import DM05Wrapper


# Steps 1000..9000 sit next to this one. There is no validation split on this
# dataset, so choosing between them means comparing on hardware.
DEFAULT_CHECKPOINT = (
    "/common/users/jh2400/opendm/user_checkpoints/"
    "dm05_tight_insertion_lora/checkpoint-10000"
)

# The playground training entry this checkpoint was trained with. It selects
# the dataset registration and the data transforms, not just the architecture.
DEFAULT_EXP_MODULE = "playground.dm05_tight_insertion_lora"

DEFAULT_OPENDM_ROOT = "/common/users/jh2400/opendm"

# Verbatim from the JSONL this run was trained on. Note this is NOT the
# LeRobot dataset's own task string -- the training set was converted with an
# overridden prompt, and a paraphrase is off-distribution.
DEFAULT_PROMPT = (
    "pick up the metal object on the bottom right, "
    "and insert it into the bottom right hole"
)

DEFAULT_BIND = "tcp://*:5555"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bind",
        default=DEFAULT_BIND,
        help=f"ZeroMQ bind address (default: {DEFAULT_BIND}).",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint step directory, e.g. .../checkpoint-10000.",
    )
    parser.add_argument(
        "--exp-module",
        default=DEFAULT_EXP_MODULE,
        help="Importable playground training entry this checkpoint was trained with.",
    )
    parser.add_argument(
        "--opendm-root",
        default=DEFAULT_OPENDM_ROOT,
        help="OpenDM repository root (holds playground/ and checkpoints/).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used when a request does not carry its own.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=None,
        help=(
            "Flow-matching Euler steps. Omit to keep the training entry's value "
            "(10); lower it to trade action quality for latency."
        ),
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help=(
            "Skip the startup warmup inferences. Not recommended: the CUDA "
            "autotune and suffix graph capture then land on the first real "
            "request instead."
        ),
    )
    parser.add_argument(
        "--name",
        default='dm05',
        help=(
            "Name this server registers under, so the RURI menu can list it and "
            "a client can pick it. It identifies this instance, not the wrapper "
            "class: two Pi0.5 servers on two checkpoints would both call "
            "themselves 'pi05'. Pass --name '' to run unannounced."
        ),
    )
    parser.add_argument(
        "--advertise-host",
        default=None,
        help=(
            "Hostname to publish to the menu in place of the bind wildcard. "
            "Defaults to this machine's own hostname; set it when that is not "
            "what the robot can resolve."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # The wrapper reports loading and warmup progress through `logging`.
    # OpenDM logs through loguru and reconfigures it on import; both land on
    # stdout, so the two streams interleave rather than one hiding the other.
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    policy = DM05Wrapper(
        checkpoint_path=args.checkpoint,
        exp_module=args.exp_module,
        opendm_root=args.opendm_root,
        default_prompt=args.prompt,
        diffusion_steps=args.diffusion_steps,
        warmup=not args.no_warmup,
    )

    for key, value in policy.optional_more_metadata().items():
        logging.info("  %s: %s", key, value)

    serve(
        policy=policy,
        bind_address=args.bind,
        name=args.name or None,
        advertise_host=args.advertise_host,
    )


if __name__ == "__main__":
    main()
