#!/usr/bin/env python
"""Continuously drive the local Single Piper with the remote Pi0.5 policy.

The scheduler owns observation capture, asynchronous inference, rolling chunk
aggregation, and the 30 Hz action stream.  Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging

from ruri.client.controllers.single_piper import SinglePiperController
from ruri.client.policies import RemotePolicy
from ruri.client.schedulers import RollingScheduler


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
            "explicit policy ZeroMQ endpoint; inspect available policies with "
            "examples/list_policy_servers.py"
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.5)
    parser.add_argument(
        "--aggregate-fn-name",
        choices=("weighted_average", "latest_only", "average", "conservative"),
        default="weighted_average",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="stop after this many inferences; omitted means run continuously",
    )
    parser.add_argument(
        "--scheduler-log-dir",
        default="logs",
        help="local JSONL trace directory (default: logs)",
    )
    parser.add_argument(
        "--diagnostic-log",
        default=None,
        help="MIT 100 Hz pre-abort q/qd/q_target JSONL path",
    )

    # Current local hardware.  Controller still verifies the stable adapter ID
    # and Piper feedback signature before enabling motion.
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
    scheduler = RollingScheduler()
    try:
        scheduler.run(
            controller=SinglePiperController,
            policy=RemotePolicy,
            args=args,
        )
    except KeyboardInterrupt:
        logging.info("Stopped by user")


if __name__ == "__main__":
    main()
