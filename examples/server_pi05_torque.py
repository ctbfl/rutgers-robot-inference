#!/usr/bin/env python
"""Serve the whole_v1 torque-history training-time RTC Pi0.5 checkpoint.

The input contract -- including the timestamped effort history -- is published
in the server's metadata; run examples/list_policy_servers.py to read it.
"""

import argparse
import logging

from ruri.server.serve import serve
from ruri.server.wrappers.pi05.pi05_torque import (
    TORQUE_PACKAGE, TRAIN_RTC_PACKAGE, Pi05TrainRTCTorqueWrapper,
)

DEFAULT_CHECKPOINT = (
    "/common/users/jh2400/training_results/pi05_train_rtc/"
    "260915_whole_v1_torque_rtc_h30_4gpu_bs128/checkpoints/"
    "pi05_whole_v1_torque_rtc_h30/whole_v1_torque_rtc_h30_4gpu_bs128/10000"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="completed step directory with params/ and assets/")
    parser.add_argument("--bind", default="tcp://*:5563")
    parser.add_argument("--name", default="pi05_train_rtc_torque")
    parser.add_argument("--advertise-host", default=None)
    parser.add_argument("--torque-package", default=TORQUE_PACKAGE)
    parser.add_argument("--train-rtc-package", default=TRAIN_RTC_PACKAGE)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--default-inference-delay", type=int, default=4)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s: %(message)s")
    policy = Pi05TrainRTCTorqueWrapper(
        args.checkpoint,
        torque_package=args.torque_package,
        train_rtc_package=args.train_rtc_package,
        num_denoising_steps=args.num_denoising_steps,
        default_inference_delay=args.default_inference_delay,
        warmup=not args.no_warmup,
    )
    serve(policy=policy, bind_address=args.bind, name=args.name, advertise_host=args.advertise_host)


if __name__ == "__main__":
    main()
