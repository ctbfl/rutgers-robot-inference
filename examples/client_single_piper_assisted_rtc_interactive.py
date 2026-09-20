#!/usr/bin/env python
"""Paired Piper RTC with automatic human correction, without dataset recording.

Enter a task prompt to enable both arms. The right/secondary Piper is the
compliant leader; left/main is the task follower. Both receive a shared virtual
reference directly. Drag the right arm to correct it; background RTC inference
continues, and fresh predictions continue from the corrected state on release.

  q             pause both arms; enter another task prompt to resume.
  home          at the paused prompt: paired home at up to 1 rad/s, then hold.
  switch PORT   at the paused prompt: switch checkpoint on the same host.
  Esc / Ctrl-C  hold and exit through the assisted worker's normal stop path.

Startup enables and homes both arms with MIT, then starts policy execution.
Exit holds the current pose. No dataset, video or effort log is
written. Do not run this alongside another arm controller.
"""

from __future__ import annotations

import argparse
import logging

from ruri.client.controllers.single_piper_assisted import SinglePiperAssistedController
from ruri.client.schedulers.assisted_interactive import InteractiveAssistedRTC
from ruri.client.utils import inference_client


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy-endpoint",
        required=True,
        type=inference_client.normalize_endpoint,
        help="policy server, e.g. tcp://172.16.68.130:5557 or 172.16.68.130:5557",
    )
    parser.add_argument(
        "--can-interface",
        "--follower-can-interface",
        dest="follower_can_interface",
        default=None,
        help="left/main CAN interface; default discovers its registered USB ID",
    )
    parser.add_argument(
        "--leader-can-interface",
        default=None,
        help="right/secondary CAN interface; default discovers its registered USB ID",
    )
    parser.add_argument("--head-camera-serial", default="827112071860")
    parser.add_argument("--wrist-camera-serial", default="002422064073")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--execution-horizon", type=int, default=5)
    parser.add_argument("--rtc-latency-window", type=int, default=10)
    parser.add_argument("--rtc-initial-delay-steps", type=int, default=4)
    parser.add_argument("--policy-timeout-s", type=float, default=5.0)
    parser.add_argument("--policy-ready-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--startup-home", action=argparse.BooleanOptionalAction, default=True,
        help="home both arms with MIT before policy (default: enabled)",
    )
    parser.add_argument("--start-home-speed", type=float, default=0.2,
                        help="startup home peak reference speed in rad/s")
    parser.add_argument(
        "--configure-can", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        with inference_client.connect(args) as policy:
            InteractiveAssistedRTC().run(
                controller=SinglePiperAssistedController, policy=policy, args=args
            )
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopped by user")


if __name__ == "__main__":
    main()
