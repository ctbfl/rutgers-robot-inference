#!/usr/bin/env python3
"""Re-zero one joint of one arm, interactively.

calibrate_joint writes the joint's current position into the arm as its new
zero. It is a firmware change: it survives a power cycle, it shifts every
reading that joint will ever report, and it invalidates any dataset or policy
recorded in the old frame. So this tool never issues it, or the disable that
precedes it, without a typed confirmation, and it never moves the arm.

Why one joint at a time: the two arms here disagree about joint2 by 6.85 deg
while agreeing on every other joint to within 1.5, so joint2 is the only frame
worth touching and the rest should be left exactly as they are.

    tools/piper/set_joint_zero.py --arm left_arm --joint 2 --target -6.85

--target is the reading, in the joint's CURRENT frame, at which zero should be
set. Work it out from a mechanical stop rather than by eye: drive the joint
onto its stop, note what it reads, and add the offset you want that stop to
end up at.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ruri.client.controllers.single_piper import calibration_ranges as calib  # noqa: E402
from ruri.client.controllers.single_piper.discovery import discover_piper_can  # noqa: E402
from ruri.client.controllers.single_piper.hardware_registry import (  # noqa: E402
    ARM_NAMES,
    find_arm_by_name,
)
from ruri.client.controllers.single_piper.mit.leader_follower import (  # noqa: E402
    build_arm,
    read_sample,
)

SETTLE_WINDOW_S = 2.0
STABLE_DRIFT_DEG = 0.05


def confirm(prompt: str, word: str) -> bool:
    print(f"\n{prompt}")
    try:
        return input(f"  type {word!r} to proceed, anything else to abort: ").strip() == word
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def reading(arm, joint: int) -> float:
    return math.degrees(read_sample(arm, "arm").q[joint - 1])


def track(arm, joint: int, target: float) -> float:
    """Show the live reading until the operator stops. Returns the last value."""
    history: deque[tuple[float, float]] = deque()
    print("\nPosition the joint, then Ctrl-C to stop watching.")
    print("Drift is over the last 2 s: the reading must hold still on its own,")
    print("because it has to stay put between your hand leaving it and the write.\n")
    value = reading(arm, joint)
    try:
        while True:
            value = reading(arm, joint)
            now = time.monotonic()
            history.append((now, value))
            while history and now - history[0][0] > SETTLE_WINDOW_S:
                history.popleft()
            drift = max(v for _, v in history) - min(v for _, v in history)
            steady = "steady" if drift <= STABLE_DRIFT_DEG else "MOVING"
            print(f"\r  reading {value:8.2f} deg   target {target:8.2f}   "
                  f"error {value - target:+7.2f}   drift {drift:5.2f} [{steady}]  ",
                  end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=list(ARM_NAMES))
    parser.add_argument("--joint", required=True, type=int, choices=range(1, 7))
    parser.add_argument("--target", required=True, type=float,
                        help="reading in the current frame at which to set zero, deg")
    parser.add_argument("--tolerance", type=float, default=0.30,
                        help="refuse to write further than this from --target, deg")
    args = parser.parse_args()

    registration = find_arm_by_name(args.arm)
    device = discover_piper_can(hardware_id=registration.can_hardware_id)
    print(f"{args.arm} -> {device.interface}  (adapter {registration.can_hardware_id})")

    stored = calib.MEASURED_BY_ARM[args.arm][f"joint{args.joint}"]
    print(f"stored joint{args.joint} stops: "
          f"[{stored[0] / 1000:.2f}, {stored[1] / 1000:.2f}] deg "
          f"-> after this write they become "
          f"[{stored[0] / 1000 - args.target:.2f}, {stored[1] / 1000 - args.target:.2f}]")
    print("That file will be stale until joint{} is measured again "
          "(tools/piper/probe_arm_range.py).".format(args.joint))

    arm = build_arm(device.interface)
    enabled = arm.get_joints_enable_status_list()
    print(f"\njoint enable status: {enabled}")
    print(f"joint{args.joint} currently reads {reading(arm, args.joint):.2f} deg")

    if enabled and enabled[args.joint - 1]:
        if not confirm(
            f"joint{args.joint} is energized. Disabling it drops whatever it "
            f"holds up -- on joint2 that is the whole arm. Support the arm first.",
            "disable",
        ):
            print("Nothing was sent.")
            return 1
        arm.disable(args.joint)
        time.sleep(0.3)
        print(f"joint{args.joint} disabled; status now "
              f"{arm.get_joints_enable_status_list()}")
    else:
        print(f"joint{args.joint} is already de-energized; nothing to disable.")

    value = track(arm, args.joint, args.target)

    error = value - args.target
    print(f"\nfinal reading {value:.2f} deg, {error:+.2f} from target")
    if abs(error) > args.tolerance:
        print(f"That is further than --tolerance {args.tolerance:.2f}. Not writing.")
        print("Re-run once the joint is closer, or pass a larger --tolerance.")
        return 1

    if not confirm(
        f"Write joint{args.joint} zero at {value:.2f} deg. This is a firmware "
        f"change and it invalidates data and policies recorded in the old frame.",
        "calibrate",
    ):
        print("Nothing was written. The joint is still de-energized.")
        return 1

    ok = arm.calibrate_joint(args.joint)
    time.sleep(0.3)
    after = reading(arm, args.joint)
    print(f"\ncalibrate_joint returned {ok}; joint{args.joint} now reads {after:.2f} deg")
    if not ok:
        print("The arm did not confirm the write. Check the reading before trusting it.")
    print("\nNext: re-enable when ready, drive the joint onto its stop and check it "
          "reads what you expect, then re-measure and update the calibration file.")
    print(f"  tools/piper/probe_arm_range.py --arm {args.arm}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
