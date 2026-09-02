#!/usr/bin/env python
"""Continuously drive Single Piper with latency-aligned RTC chunk replacement.

The policy's metadata declares its action horizon. After every configured
number of successfully sent actions, the scheduler requests another chunk
while continuing to execute the old one. On return, the elapsed prefix is
discarded and the unsent queue is replaced on the same absolute control
timeline. Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging

from ruri.client.controllers.single_piper import SinglePiperController
from ruri.client.policies import RemotePolicy
from ruri.client.schedulers import RTCScheduler


DEFAULT_PROMPT = (
    "pick and place the object E into the first hole "
    "on the manipulation-net board."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-endpoint",
        required=True,
        help=(
            "explicit RTC policy ZeroMQ endpoint; inspect available policies "
            "with examples/list_policy_servers.py"
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=5,
        help="request a fresh chunk after this many sent actions",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="stop after this many inferences; omitted means continuous",
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
        RTCScheduler().run(
            controller=SinglePiperController,
            policy=RemotePolicy,
            args=args,
        )
    except KeyboardInterrupt:
        logging.info("Stopped by user")


if __name__ == "__main__":
    main()
