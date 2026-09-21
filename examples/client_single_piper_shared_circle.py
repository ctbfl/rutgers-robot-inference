#!/usr/bin/env python
"""Right leader only: fixed MIT + gravity, circle policy and human admittance.

Default: offline IK audit. --simulate: worker/UDP/scheduler without hardware.
--execute: enable and home the right arm, then run until Ctrl-C (or --seconds).
Left arm and cameras are never opened. Ctrl-C requests HOLD; runtime hardware
faults retain MIT hold/home recovery where feedback permits it.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from piper_circle_policy import (
    CirclePolicy, DEFAULT_RADIUS, DEFAULT_PERIOD, DEFAULT_TCP_SPEED, DEFAULT_REFERENCE_SPEED,
    DEFAULT_POLICY_SPEED,
)
from ruri.client.controllers.single_piper_assisted import SinglePiperAssistedController
from ruri.client.controllers.single_piper_assisted.config import SinglePiperAssistedConfig
from ruri.client.schedulers.assisted import AssistedScheduler


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--simulate", action="store_true")
    parser.add_argument("--seconds", type=float, default=0., help="0 = run until Ctrl-C")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS, help="metres")
    parser.add_argument("--period", type=float, default=DEFAULT_PERIOD, help="seconds per lap; default is 0.5 rad/s circle angular speed")
    parser.add_argument("--tcp-speed", type=float, default=DEFAULT_TCP_SPEED, help="policy TCP lookahead speed, m/s")
    parser.add_argument("--reference-speed", type=float, default=DEFAULT_REFERENCE_SPEED,
                        help="final blended joint reference velocity limit, rad/s")
    parser.add_argument("--policy-speed", type=float, default=DEFAULT_POLICY_SPEED,
                        help="policy joint reference velocity limit before blending, rad/s")
    parser.add_argument("--admittance-damping", type=float, default=.1,
                        help="human reference damping B for all six joints; smaller follows faster")
    parser.add_argument("--center-offset", type=float, nargs=3, default=(.05, 0., .15), metavar=("X", "Y", "Z"))
    parser.add_argument("--tcp-offset", type=float, nargs=3, default=(0., 0., .1358), metavar=("X", "Y", "Z"))
    parser.add_argument("--leader-can-interface", default=None)
    parser.add_argument("--configure-can", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--simulate-intervention", action="store_true", help="simulation only: constrain J1 +0.2 rad for 2s")
    args = parser.parse_args(argv)
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error("seconds must be finite and nonnegative")
    if not np.isfinite(args.reference_speed) or not .01 <= args.reference_speed <= 4.:
        parser.error("reference-speed must be finite in [0.01, 4] rad/s")
    if not np.isfinite(args.policy_speed) or not .01 <= args.policy_speed <= args.reference_speed:
        parser.error("policy-speed must be finite in [0.01, reference-speed] rad/s")
    if not np.isfinite(args.admittance_damping) or args.admittance_damping <= 0:
        parser.error("admittance-damping must be positive and finite")
    if args.simulate_intervention and not args.simulate:
        parser.error("--simulate-intervention requires --simulate")
    return args


def main(argv=None):
    args = parse_args(argv)
    policy = CirclePolicy(radius=args.radius, period=args.period, tcp_speed=args.tcp_speed,
                          center_offset=args.center_offset, tcp_offset=args.tcp_offset)
    audit = policy.validate_path(policy_speed=args.policy_speed)
    print(json.dumps(audit, indent=2))
    if not (args.execute or args.simulate):
        return 0
    from ruri.client.controllers.single_piper.mit.leader_gravity_check import hardware_owner, torque_contract
    config = SinglePiperAssistedConfig(
        leader_only=True, camera_enabled=False, simulate=args.simulate,
        startup_home=True, start_home_speed=.15, reference_speed=args.reference_speed,
        policy_reference_speed=args.policy_speed, max_reference_speed=args.reference_speed,
        admittance_damping=(args.admittance_damping,)*6,
        leader_can_interface=args.leader_can_interface, configure_can=args.configure_can,
    )
    log_path = args.log or Path("logs") / f"right_shared_circle_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    scheduler = None
    robot = SinglePiperAssistedController(config)
    # Claim the workstation writer lock before starting the CAN-owning child.
    with (hardware_owner() if args.execute else nullcontext()), log_path.open("x") as log:
        header = {"kind": "setup", "simulation": args.simulate, "config": asdict(config),
                  "circle_audit": audit, "torque_contract": torque_contract() if args.execute else None}
        log.write(json.dumps(header, default=json_value)+"\n")
        try:
            robot.start()
            policy.started = time.monotonic()
            scheduler = AssistedScheduler(robot, policy, policy_version="circle-v1", prompt="follow moving circle point",
                                          control_hz=30., execution_horizon=1, rtc_initial_delay_steps=0)
            scheduler.resume()
            print(f"RUNNING: right leader only; drag to intervene, release to resume. Ctrl-C holds. Log: {log_path}", flush=True)
            began = next_tick = next_print = time.monotonic()
            injected = False
            while not args.seconds or time.monotonic()-began < args.seconds:
                sample = robot.get_sample()
                if args.simulate_intervention and not injected and time.monotonic()-began >= 3.:
                    from ruri.client.controllers.single_piper.normalization import normalize_pose
                    measured = sample["telemetry"]["leader"]
                    q = np.array(measured["q"]); q[0] += .2
                    robot.simulate_leader(normalize_pose(q, measured["gripper_width"]), duration_s=2.)
                    injected = True
                step = scheduler.step(sample)
                row = {"kind": "sample", "telemetry": sample["telemetry"], "trace": sample["trace"],
                       "scheduler": step, "circle": policy.last}
                log.write(json.dumps(row, default=json_value, allow_nan=False)+"\n")
                c = sample["telemetry"]["control"]
                if c["mode"] == "HOLD" and not c["armed"] and time.monotonic()-began > 1.:
                    raise RuntimeError(f"worker entered HOLD: {c['reason']}")
                if time.monotonic() >= next_print:
                    print(f"t={time.monotonic()-began:5.1f}s  authority={c['authority_phase']}  "
                          f"alpha={c['human_weight']:.2f}  confirm={c['intervention_confirm_s']:.2f}s  "
                          f"rise={c['human_rise_s']:.2f}s  "
                          f"quiet={c['human_quiet_s']:.2f}s  revision={c['reference_revision']}", flush=True)
                    next_print = time.monotonic()+1.
                    log.flush()
                next_tick += 1/30
                time.sleep(max(0., next_tick-time.monotonic()))
                next_tick = max(next_tick, time.monotonic())
        except KeyboardInterrupt:
            print("Stopping policy and requesting HOLD.", flush=True)
        finally:
            try:
                if scheduler is not None:
                    scheduler.close()
            finally:
                robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
