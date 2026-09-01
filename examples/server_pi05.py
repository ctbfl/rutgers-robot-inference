#!/usr/bin/env python
"""
Example: serve the fine-tuned Pi0.5 tight-insertion policy over ZeroMQ.

This wires the three pieces together:

    Pi05Wrapper (OpenPI + JAX)  ->  ruri.server.serve.serve()  ->  ZMQ REP

Defaults point at the 10k-step tight_insertion_E1 checkpoint on this
machine, so the example runs with no arguments.

Launch
------
Pi0.5 needs the OpenPI environment (JAX + the TrainConfig registry that
defines ``pi05_tight_insertion_E1``). RURI is already installed into it
as an editable package, so no PYTHONPATH is needed::

    cd /common/home/jh2400/projects/openpi
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \\
      ./.venv/bin/python \\
      /common/users/jh2400/rutgers-robot-inference/examples/server_pi05.py

JAX preallocates ``XLA_PYTHON_CLIENT_MEM_FRACTION`` of the card at startup,
so nvidia-smi reports the pool rather than real usage. 0.2 is plenty for one
Pi0.5 policy and leaves the card usable by others.

Startup takes ~25 s: restoring the params, then a warmup inference that
triggers JAX compilation. The bind address is printed only once the server
can actually answer.

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
        "action_chunk": np.ndarray (10, 7) float32,
        "prompt": str,
        "timing.infer_ms": float,
        "timing.wrapper_ms": float,
    }

``{"type": "metadata"}`` returns the input contract instead. A failed
request comes back as ``{"error": "..."}``.
"""

import argparse
import logging

from ruri.server.serve import serve
from ruri.server.wrappers.pi05.pi05 import Pi05Wrapper


# Steps 2000/4000/6000/8000 sit next to this one. There is no validation split
# on this dataset, so choosing between them means comparing on hardware.
DEFAULT_CHECKPOINT = (
    "/common/users/jh2400/openpi_checkpoints/"
    "pi05_tight_insertion_E1/tight_insertion_E1_10k/9999"
)

# Must be the TrainConfig this checkpoint was trained with: it selects the
# data transforms, not just the architecture.
DEFAULT_CONFIG_NAME = "pi05_tight_insertion_E1"

# Verbatim from the training set's meta/tasks.jsonl -- a paraphrase is
# off-distribution.
DEFAULT_PROMPT = (
    "pick and place the object E into the first hole "
    "on the manipulation-net board."
)

# One port per policy so several can run at once; the menu binds 5550.
# The port is how a client actually addresses a policy; --name is only a
# convenient label for the menu listing.
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
        help="Checkpoint step directory, containing params/ and assets/.",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="OpenPI TrainConfig name this checkpoint was trained with.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used when a request does not carry its own.",
    )
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        default=None,
        help=(
            "Flow-matching sample steps. Omit to keep the OpenPI default (10); "
            "lower it to trade action quality for latency."
        ),
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help=(
            "Skip the startup warmup inference. Not recommended: the JAX "
            "compile then lands on the first real request instead."
        ),
    )
    parser.add_argument(
        "--name",
        default='pi05',
        help=(
            "Label this server registers under, shown in the RURI menu listing. "
            "It is a convenience only: a policy is identified by its port, which "
            "is unique and which the client actually dials, and by the "
            "inputs/outputs contract it publishes. Two servers may share a "
            "label. Pass --name '' to run unannounced."
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
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    policy = Pi05Wrapper(
        checkpoint_path=args.checkpoint,
        config_name=args.config_name,
        default_prompt=args.prompt,
        num_denoising_steps=args.num_denoising_steps,
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
