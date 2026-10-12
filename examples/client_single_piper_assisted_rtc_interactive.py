#!/usr/bin/env python
"""Paired Piper RTC with automatic human correction, without dataset recording.

Enter a task prompt to enable both arms. The right/secondary Piper is the
leader; left/main is the task follower. Both track the same virtual reference.
Close the open right gripper by 1cm to request control. It matches the left
gripper, then the right arm enters gravity control; the left blends to its readings
over 0.5s. Background inference continues and RTC restarts from current state
on release. Remote observation/action formats are unchanged.
Grippers use the validated 0-100 mm firmware range locally; the policy still
sends/receives 0-100 (for example, action 10 commands 10 mm).

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
from contextlib import nullcontext

from ruri.client.controllers.single_piper_assisted import SinglePiperAssistedController
from ruri.client.schedulers.assisted_interactive import InteractiveAssistedRTC
from ruri.client.utils import inference_client


def parse_args(argv=None, *, interactive=True):
    parser = argparse.ArgumentParser(
        description=(__doc__ if interactive else
                     "Remote RTC with dual Piper gravity handoff. Enable/home both arms, run until Ctrl-C; no recording."),
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    parser.add_argument("--simulate", action="store_true", help="no CAN or cameras; connect to policy as usual")
    if not interactive:
        parser.add_argument("--prompt", default=(
            "pick and place the object E into the first hole on the manipulation-net board."))
        parser.add_argument("--seconds", type=float, default=0., help="0 runs until Ctrl-C; duration starts after home")
    parser.add_argument(
        "--startup-home", action=argparse.BooleanOptionalAction, default=True,
        help="home both arms with MIT before policy (default: enabled)",
    )
    parser.add_argument("--start-home-speed", type=float, default=0.15,
                        help="startup home peak reference speed in rad/s")
    parser.add_argument(
        "--configure-can", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--log-level", default="INFO")
    # The same local controller settings accepted in the dual circle trial.
    parser.set_defaults(leader_only=False, gravity_handoff=True, human_trigger="gripper", rtc_during_policy=True,
                        gripper_max_m=.1,
                        reference_speed=4., max_reference_speed=4., policy_reference_speed=.5,
                        camera_enabled=True, record_effort=False, reset_latched_stop=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        with inference_client.connect(args) as policy:
            from ruri.client.controllers.single_piper.mit.leader_gravity_check import hardware_owner
            with (nullcontext() if args.simulate else hardware_owner()):
                InteractiveAssistedRTC().run(
                    controller=SinglePiperAssistedController, policy=policy, args=args
                )
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopped by user")


if __name__ == "__main__":
    main()
