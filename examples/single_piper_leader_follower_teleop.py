#!/usr/bin/env python3
"""Run the RURI-owned, no-force-feedback Piper MIT teleop controller."""

from __future__ import annotations

import argparse
import logging
import time

from ruri.client.controllers.single_piper_leader_follower_teleop import (
    SinglePiperLeaderFollowerTeleopController,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-can-interface")
    parser.add_argument("--follower-can-interface")
    parser.add_argument("--telemetry-address", default="udp://127.0.0.1:6670")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--no-gripper", dest="use_gripper", action="store_false")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement: energize and move both arms",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required because this command moves both arms")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    controller = SinglePiperLeaderFollowerTeleopController(args)
    try:
        controller.start()
        print(
            "MIT teleop engaged. The follower tracks the leader; Ctrl-C stops "
            "the worker and leaves its final MIT hold."
        )
        while controller.arm_started:
            time.sleep(0.2)
        raise RuntimeError("MIT teleop worker exited unexpectedly; inspect its log above")
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
