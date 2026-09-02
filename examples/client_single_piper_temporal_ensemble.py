#!/usr/bin/env python
"""Run remote ACT with LeRobot-style per-step temporal ensembling."""

from __future__ import annotations

import argparse
import logging

from ruri.client.controllers.single_piper import SinglePiperController
from ruri.client.schedulers import TemporalEnsembleScheduler
from ruri.client.utils import inference_client


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
            "explicit ACT policy ZeroMQ endpoint; inspect available policies "
            "with examples/list_policy_servers.py"
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="LeRobot ACT exponential coefficient; positive favors older predictions",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="stop after this many ACT inferences; omitted means continuous",
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
            TemporalEnsembleScheduler().run(
                controller=SinglePiperController,
                policy=policy,
                args=args,
            )
    except KeyboardInterrupt:
        logging.info("Stopped by user")


if __name__ == "__main__":
    main()
