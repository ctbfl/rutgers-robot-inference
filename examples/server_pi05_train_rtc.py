#!/usr/bin/env python
"""
Example: serve the Pi0.5 tight-insertion policy with **training-time** RTC.

    Pi05TrainRTCWrapper (OpenPI + JAX)  ->  ruri.server.serve.serve()  ->  ZMQ REP

This is not ``server_pi05_rtc.py`` with different flags -- it is a different
method and a different checkpoint, and the two are not interchangeable:

  server_pi05_rtc.py        arXiv 2506.07339, any pi05 checkpoint. Conditions on
                            the previous chunk with PiGDM guidance at sampling
                            time. Tunables: prefix attention horizon, schedule,
                            max guidance weight. Costs one extra VJP per flow
                            step, and the guidance is an approximation that
                            moves the sampler off the training distribution.

  server_pi05_train_rtc.py  arXiv 2512.05964, requires a pi05_train_rtc_h30
  (this file)               checkpoint. The committed prefix is fed in as clean
                            input at per-token flow time 0, which is exact
                            conditioning, not guidance. No soft overlap and no
                            guidance weight exist. Costs the same as plain
                            sampling.

Both wrappers now refuse the other's checkpoint, because the failure was
otherwise silent: loading an RTC-trained checkpoint under a baseline config
drops its ``tok_time_proj`` parameter without a warning.

Launch
------
The OpenPI environment, as with the other pi05 servers::

    cd /common/home/jh2400/projects/openpi
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \\
      ./.venv/bin/python \\
      /common/users/jh2400/rutgers-robot-inference/examples/server_pi05_train_rtc.py

Request / response
------------------
The request contract is identical to ``server_pi05_rtc.py``'s, so a scheduler
written for that server works here unchanged::

    "context.rtc.prev_chunk_left_over": np.ndarray (H - s, 7) float32,
    "context.rtc.consumed_steps": int,                   # s
    "context.rtc.estimated_inference_delay_steps": int,  # d

The response differs, and the difference is the point::

    "rtc.method":          "training-time"
    "rtc.inference_delay": int   # after clamping to the trained support
    "rtc.execute_from":    int   # == inference_delay -- START HERE, not at 0
    # no rtc.prefix_attention_horizon, no rtc.schedule: they do not exist here

Rows ``[0, rtc.execute_from)`` are the pinned prefix the arm is already
executing. Replaying them double-steps the robot.

Timing
------
H is 30 at 30 fps, i.e. a 1.0 s chunk, so there is real room here: with a
measured delay of 3-5 steps and an execute horizon of 8, roughly 22 of the 30
actions are regenerated freely every cycle. The 10-step chunk of the older
tight_insertion_E1 config left almost nothing free once the delay was pinned,
which is part of why RTC looked like a regression on it.
"""

import argparse
import logging

from ruri.server.serve import serve
from ruri.server.wrappers.pi05.pi05_train_rtc import Pi05TrainRTCWrapper


# Steps 2000/4000/6000/8000 sit next to this one. There is no validation split
# on this dataset, so choosing between them means comparing on hardware.
DEFAULT_CHECKPOINT = (
    "/common/users/jh2400/openpi_checkpoints/"
    "pi05_train_rtc_h30/rtc_h30_10k/9999"
)

# Verbatim from the training set's meta/tasks.jsonl -- a paraphrase is
# off-distribution.
DEFAULT_PROMPT = (
    "pick and place the object E into the first hole "
    "on the manipulation-net board."
)

# One port per policy so several can run at once; the menu binds 5550.
# The port is how a client actually addresses a policy; --name is only a
# convenient label for the menu listing.
DEFAULT_BIND = "tcp://*:5557"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bind", default=DEFAULT_BIND, help=f"ZeroMQ bind address (default: {DEFAULT_BIND}).")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint step directory, containing params/ and assets/. Must be a training-time RTC run.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used when a request does not carry its own.")
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        default=None,
        help=(
            "Flow-matching sample steps. Omit to keep the OpenPI default (10). "
            "Unlike the test-time wrapper there is no guidance weight tied to "
            "this number, so lowering it only trades action quality for latency."
        ),
    )
    parser.add_argument(
        "--default-inference-delay",
        type=int,
        default=4,
        help=(
            "Delay in control steps used when a request carries none (default: 4). "
            "Measured on this setup: ~120 ms end to end at 30 fps, i.e. 3-5 steps. "
            "A client that measures its own round trip should send it per request."
        ),
    )
    parser.add_argument(
        "--train-rtc-package",
        default="/common/users/jh2400/pi05-rtc",
        help="Directory containing the pi05_train_rtc package (the model class lives outside OpenPI).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip both startup warmup inferences. Not recommended: the JAX compiles land on the first real requests instead.",
    )
    parser.add_argument(
        "--name",
        default='pi05_train_rtc',
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
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    policy = Pi05TrainRTCWrapper(
        checkpoint_path=args.checkpoint,
        default_prompt=args.prompt,
        num_denoising_steps=args.num_denoising_steps,
        default_inference_delay=args.default_inference_delay,
        train_rtc_package=args.train_rtc_package,
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
