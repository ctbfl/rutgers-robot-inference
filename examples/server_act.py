#!/usr/bin/env python
"""
Example: serve the fine-tuned ACT tight-insertion policy over ZeroMQ.

    ACTWrapper (LeRobot + PyTorch)  ->  ruri.server.serve.serve()  ->  ZMQ REP

Defaults point at the 100k-step tight_insertion_E1_act checkpoint on this
machine, so the example runs with no arguments.

Launch
------
ACT needs the **LeRobot** environment, not the OpenPI one. The two cannot be
shared: these checkpoints use the processor-pipeline format introduced after
LeRobot 0.1.0, and the OpenPI venv pins 0.1.0. RURI is installed editable into
the LeRobot venv, so no PYTHONPATH is needed::

    CUDA_VISIBLE_DEVICES=0 \\
      /common/home/jh2400/projects/lerobot/.venv/bin/python \\
      /common/users/jh2400/rutgers-robot-inference/examples/server_act.py

Startup is a few seconds: PyTorch loads the weights and runs one warmup
inference. There is no JAX-style compile to absorb, so this is far quicker than
the Pi0.5 servers.

Request / response
------------------
Messages are MessagePack-encoded by ``ruri.common.zmq``, which carries numpy
arrays natively. A client sends::

    {
        "type": "infer",
        "observation.state":        np.ndarray (7,) float,
        "observation.images.top":   np.ndarray (H, W, 3) uint8, RGB,
        "observation.images.wrist": np.ndarray (H, W, 3) uint8, RGB,
    }

and gets back::

    {
        "action_chunk": np.ndarray (100, 7) float32,
        "timing.infer_ms": float,
        "timing.wrapper_ms": float,
    }

Note ``observation.images.wrist``: the checkpoint calls that camera ``hand``,
and the wrapper's INPUT_MAPPING absorbs the difference so clients use the same
key here as for Pi0.5.

Two things differ from the Pi0.5 servers and matter to the scheduler:

  * The chunk is **100 steps**, i.e. 3.3 s at 30 fps, against Pi0.5's 10 steps
    / 333 ms. The whole chunk always comes back; re-plan well before it runs
    out and drop the tail.
  * Actions are **absolute joint targets**, not deltas against the current
    state, so they can go to the arm as-is.

``{"type": "metadata"}`` returns the input contract instead. A failed request
comes back as ``{"error": "..."}``.
"""

import argparse
import logging

from ruri.server.serve import serve
from ruri.server.wrappers.act.act import ACTWrapper


# Steps 020000/040000/060000/080000 sit next to this one, plus `last`. There is
# no validation split on this dataset, so choosing between them means comparing
# on hardware -- same situation as the Pi0.5 checkpoints.
DEFAULT_CHECKPOINT = (
    "/common/users/jh2400/lerobot_outputs/tight_insertion_E1_act"
    "/checkpoints/100000/pretrained_model"
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
        help=(
            "Checkpoint directory. Either a pretrained_model/ directory or the "
            "step directory containing it; both are accepted."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for the policy (default: cuda).",
    )
    parser.add_argument(
        "--task",
        default="",
        help=(
            "Language instruction used when a request carries no 'prompt'. ACT "
            "is not language-conditioned, so this is inert here; it exists "
            "because the LeRobot processor pipeline requires the key."
        ),
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help=(
            "Skip the startup warmup inference. The first real request then "
            "pays CUDA workspace allocation and kernel loading instead."
        ),
    )
    parser.add_argument(
        "--name",
        default='act',
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
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    policy = ACTWrapper(
        checkpoint_path=args.checkpoint,
        device=args.device,
        task=args.task,
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
