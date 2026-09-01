#!/usr/bin/env python
"""
Example: serve the fine-tuned Pi0.5 tight-insertion policy with Real-Time Chunking.

Same wiring as ``server_pi05.py``, with the RTC wrapper in place of the plain
one::

    Pi05RTCWrapper (OpenPI + JAX)  ->  ruri.server.serve.serve()  ->  ZMQ REP

RTC conditions each new chunk on the tail of the one the robot is still
executing, so successive chunks join without a seam. It is inference-time only:
the same 10k-step tight_insertion_E1 checkpoint, no retraining.

Launch
------
Identical to ``server_pi05.py`` -- the OpenPI environment, since RTC is a
different sampler over the same model::

    cd /common/home/jh2400/projects/openpi
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \\
      ./.venv/bin/python \\
      /common/users/jh2400/rutgers-robot-inference/examples/server_pi05_rtc.py

Startup is longer than the plain server, ~40 s rather than ~25 s: the guided
sampler is a separate JAX program and gets its own warmup compile. That is
deliberate. Skipping it would move the compile onto the first guided request,
mid-episode.

Request / response
------------------
Everything ``server_pi05.py`` accepts, plus three optional fields::

    {
        "type": "infer",
        ...,
        "context.rtc.prev_chunk_left_over": np.ndarray (H - s, 7) float32,
        "context.rtc.consumed_steps": int,                   # s
        "context.rtc.estimated_inference_delay_steps": int,  # d
    }

``prev_chunk_left_over`` is the previous ``action_chunk`` with its consumed rows
dropped, so its row 0 is the timestep this request's row 0 will cover. Omit all
three on the first step of an episode and the server samples normally.

The response adds an ``rtc.*`` section reporting what was actually applied::

    {
        "action_chunk": np.ndarray (10, 7) float32,
        ...,
        "rtc.applied": bool,
        "rtc.reason": str | None,               # why it fell back, if it did
        "rtc.inference_delay": int,             # after clamping
        "rtc.prefix_attention_horizon": int,    # after clamping
        "rtc.schedule": str,
    }

Watch the clamped values. This checkpoint's chunk is 10 steps, which at 30 fps
is 333 ms end to end, so the window RTC has to work in is narrow: the horizon
can never exceed ``H - s``. If it is being clamped down near the delay on every
request, the client is running late enough that RTC is mostly pinning the
trajectory rather than steering it.
"""

import argparse
import logging

from ruri.server.serve import serve
from ruri.server.wrappers.pi05.pi05_rtc import Pi05RTCWrapper


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
            "lower it to trade action quality for latency. RTC's guidance "
            "weights are tuned for 10, so change this and re-tune "
            "--max-guidance-weight."
        ),
    )
    parser.add_argument(
        "--prefix-attention-horizon",
        type=int,
        default=None,
        help=(
            "Timestep at which agreement with the previous chunk reaches zero. "
            "Omit for the full chunk. Always clamped down to the overlap the "
            "client actually sent, so this is an upper bound. Lower it to buy "
            "reactivity at the cost of smoothness."
        ),
    )
    parser.add_argument(
        "--prefix-attention-schedule",
        default="exp",
        choices=("exp", "linear", "ones", "zeros"),
        help=(
            "How agreement decays across the overlap (default: exp, the "
            "reference implementation's default). 'zeros' is a hard prefix "
            "with no soft overlap, i.e. the ablation."
        ),
    )
    parser.add_argument(
        "--max-guidance-weight",
        type=float,
        default=10.0,
        help=(
            "Ceiling on PiGDM guidance strength (default: 10.0, tuned for "
            "10-step flow matching). Higher enforces continuity harder."
        ),
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help=(
            "Skip both startup warmup inferences. Not recommended: the JAX "
            "compiles then land on the first real requests instead."
        ),
    )
    parser.add_argument(
        "--name",
        default='pi05_rtc',
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

    policy = Pi05RTCWrapper(
        checkpoint_path=args.checkpoint,
        config_name=args.config_name,
        default_prompt=args.prompt,
        num_denoising_steps=args.num_denoising_steps,
        prefix_attention_horizon=args.prefix_attention_horizon,
        prefix_attention_schedule=args.prefix_attention_schedule,
        max_guidance_weight=args.max_guidance_weight,
        warmup=not args.no_warmup,
        rtc_warmup=not args.no_warmup,
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
