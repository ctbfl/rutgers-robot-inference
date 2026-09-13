#!/usr/bin/env python
"""Drive Single Piper with RTC and change the prompt without restarting.

On start the terminal asks for a prompt; the arm is enabled only after it is
entered. While running:

  q        pause: drop the queued actions, hold the last target and ask for a
           new prompt. Enter resumes with it; an empty line asks again.
  Esc      quit, also while typing a prompt.
  Ctrl-C   quit.

Quitting takes the same controller.stop() path as client_single_piper_rtc.py.
The server does not check the prompt: it should match the dataset's
meta/tasks.jsonl text exactly, since a paraphrase is off-distribution.
"""

from __future__ import annotations

import argparse
import logging

from ruri.client.controllers.single_piper import SinglePiperController
from ruri.client.schedulers import InteractiveRTCScheduler
from ruri.client.utils import inference_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policy-endpoint",
        required=True,
        help=(
            "explicit RTC policy ZeroMQ endpoint; inspect available policies "
            "with examples/list_policy_servers.py"
        ),
    )
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=5,
        help="request a fresh chunk after this many sent actions",
    )
    parser.add_argument("--rtc-latency-window", type=int, default=10)
    parser.add_argument(
        "--rtc-initial-delay-steps",
        type=int,
        default=4,
        help="conservative delay estimate before live RTC measurements exist",
    )
    parser.add_argument("--scheduler-log-dir", default="logs")

    parser.add_argument("--arm-side", default="left")
    parser.add_argument("--arm-role", default="main")
    parser.add_argument("--can-interface", default="can1")
    parser.add_argument(
        "--can-hardware-id",
        default="usb:1d50:606f:0042002F4759530820353131",
    )
    parser.add_argument("--head-camera-serial", default="827112071860")
    parser.add_argument("--wrist-camera-serial", default="002422064073")
    parser.add_argument(
        "--startup-home-skip-threshold-rad",
        type=float,
        default=0.05,
        help="skip startup homing when max joint error from home is below this",
    )
    parser.add_argument(
        "--record-effort",
        action="store_true",
        help="write every MIT telemetry packet of this run to logs/effort-*.jsonl",
    )
    parser.add_argument(
        "--send-effort",
        action="store_true",
        help=(
            "include observation.joint_effort in the observation sent to the "
            "policy; required by policies that declare it as an input"
        ),
    )
    parser.add_argument(
        "--diagnostic-log",
        default=None,
        help="MIT 100 Hz pre-abort q/qd/q_target JSONL path",
    )
    parser.add_argument(
        "--configure-can",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        with inference_client.connect(args) as policy:
            InteractiveRTCScheduler().run(
                controller=SinglePiperController,
                policy=policy,
                args=args,
            )
    except KeyboardInterrupt:
        logging.info("Stopped by user")


if __name__ == "__main__":
    main()
