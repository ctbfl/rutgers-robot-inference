#!/usr/bin/env python
"""Serve a row1 gripper-effort checkpoint (plain or training-time RTC)."""

import argparse
import logging

from ruri.server.serve import serve
from ruri.server.wrappers.pi05.pi05_effort import (
    EFFORT_PACKAGE, RTC_PACKAGE, Pi05EffortWrapper, Pi05TrainRTCEffortWrapper,
)

DEFAULT_PROMPT = (
    "insert the row of five objects, from left to right, into the bottom row "
    "of holes on the manipulation-net board."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="completed step with params/ and assets/")
    parser.add_argument("--variant", choices=("plain", "rtc"), default="plain")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--bind", default="tcp://*:5561")
    parser.add_argument("--name", default=None)
    parser.add_argument("--advertise-host", default=None)
    parser.add_argument("--effort-package", default=EFFORT_PACKAGE)
    parser.add_argument("--train-rtc-package", default=RTC_PACKAGE)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--default-inference-delay", type=int, default=4)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s: %(message)s")
    cls = Pi05TrainRTCEffortWrapper if args.variant == "rtc" else Pi05EffortWrapper
    policy = cls(
        args.checkpoint, default_prompt=args.prompt,
        effort_package=args.effort_package, train_rtc_package=args.train_rtc_package,
        num_denoising_steps=args.num_denoising_steps,
        default_inference_delay=args.default_inference_delay,
        warmup=not args.no_warmup,
    )
    serve(policy=policy, bind_address=args.bind,
          name=args.name if args.name is not None else policy.optional_more_metadata()["name"],
          advertise_host=args.advertise_host)


if __name__ == "__main__":
    main()
