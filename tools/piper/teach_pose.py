#!/usr/bin/env python3
"""Record the pose you are holding, straight into the auto-restore pose log.

    *** RECEIVE ONLY. This never sends a CAN frame and never moves anything. ***

Reads the RIGHT/LEADER arm by default, because under leader_follower teleop the
follower's MIT target IS the leader's measured pose -- so the leader is the
command, and the command is what replays. The follower is only where it got to;
on this rig those differ by up to 1.7 deg on joint5. The MIT frames themselves
cannot be sniffed (candump sees no 0x15A-0x15F on these adapters), so reading
the leader is the only way to see the target.

Poses land in tools/piper/auto_restore_poses.jsonl, one JSON object per line,
the same file auto_restore_stage1.py's constants were taught from.

    tools/piper/teach_pose.py                      # interactive, one after another
    tools/piper/teach_pose.py "top-down grasp"     # capture one and exit
    tools/piper/teach_pose.py --follower "..."     # read the left arm instead
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ruri.client.controllers.single_piper.discovery import (  # noqa: E402
    discover_piper_can,
)
from ruri.client.controllers.single_piper.hardware_registry import (  # noqa: E402
    find_arm_registration,
)

POSE_LOG = Path(__file__).resolve().with_name("auto_restore_poses.jsonl")
JOINT_FRAMES = {0x2A5: (0, 1), 0x2A6: (2, 3), 0x2A7: (4, 5)}
GRIPPER_FRAME = 0x2A8
#: The nominal envelope, in degrees, so the margin column means something.
NOMINAL = ((-154.0, 154.0), (0.0, 195.0), (-175.0, 0.0),
           (-106.0, 106.0), (-75.0, 75.0), (-172.0, 172.0))


def sample(interface: str, seconds: float):
    """Collect every feedback frame on one bus for ``seconds``."""
    current = [None] * 6
    poses: list[list[float]] = []
    widths: list[float] = []
    forces: list[int] = []
    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((interface,))
    sock.settimeout(0.5)
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            try:
                frame = sock.recv(16)
            except socket.timeout:
                continue
            can_id, dlc = struct.unpack("=IB3x", frame[:8])
            can_id &= socket.CAN_EFF_MASK
            if dlc < 8:
                continue
            if can_id in JOINT_FRAMES:
                first, second = struct.unpack(">ii", frame[8:16])
                for slot, raw in zip(JOINT_FRAMES[can_id], (first, second)):
                    current[slot] = raw / 1000.0
                if all(v is not None for v in current):
                    poses.append(list(current))
            elif can_id == GRIPPER_FRAME:
                width, force = struct.unpack(">ih", frame[8:14])
                widths.append(width / 1000.0)
                forces.append(force)
    finally:
        sock.close()
    return np.array(poses), np.array(widths), np.array(forces)


def capture(interface, arm_name, role, seconds):
    poses, widths, forces = sample(interface, seconds)
    if len(poses) < 20:
        print(f"  only {len(poses)} samples -- is the arm powered and on {interface}?")
        return None
    median = np.median(poses, axis=0)
    spread = np.ptp(poses, axis=0)
    margin = [min(v - lo, hi - v) for v, (lo, hi) in zip(median, NOMINAL)]
    print("  q_deg  : " + " ".join(f"{v:+9.3f}" for v in median))
    print("  spread : " + " ".join(f"{v:9.3f}" for v in spread)
          + ("   <- HELD STILL" if spread.max() < 0.05 else "   <- STILL MOVING"))
    print("  margin : " + " ".join(f"{v:9.1f}" for v in margin)
          + ("   <- PAST A STOP" if min(margin) < 0 else ""))
    if len(widths):
        print(f"  gripper: {np.median(widths):.2f} mm   force {int(np.median(forces))}")
    return {
        "arm": f"{arm_name} ({role})",
        "can_interface": interface,
        "source": (
            "leader feedback frames = the follower MIT p_des target"
            if role == "leader" else "follower feedback frames (measured, not commanded)"
        ),
        "samples": int(len(poses)),
        "q_deg": [round(float(v), 3) for v in median],
        "q_rad": [round(math.radians(float(v)), 5) for v in median],
        "ptp_deg": [round(float(v), 3) for v in spread],
        "gripper_mm": None if not len(widths) else round(float(np.median(widths)), 2),
        "gripper_force": None if not len(forces) else int(np.median(forces)),
        "margin_to_nominal_deg": [round(float(v), 1) for v in margin],
    }


def write(record, label):
    record = {"recorded": dt.datetime.now().isoformat(timespec="seconds"),
              "label": label, **record}
    with POSE_LOG.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    total = sum(1 for _ in POSE_LOG.open())
    print(f"  saved as {label!r}  ({POSE_LOG.name}, {total} entries)\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("label", nargs="?", help="capture one pose under this name and exit")
    parser.add_argument("--follower", action="store_true",
                        help="read the LEFT/follower arm instead of the leader")
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="how long to average each capture (default 2)")
    args = parser.parse_args(argv)
    if not 0.5 <= args.seconds <= 30.0:
        raise SystemExit("--seconds must be in [0.5, 30]")

    role = "follower" if args.follower else "leader"
    side, slot = ("left", "main") if args.follower else ("right", "secondary")
    registration = find_arm_registration(side, slot)
    device = discover_piper_can(
        hardware_id=registration.can_hardware_id, configure=False
    )
    print(f"reading {role} = {registration.name} on {device.interface} "
          f"({args.seconds:.0f} s per capture, receive only)")

    if args.label:
        record = capture(device.interface, registration.name, role, args.seconds)
        if record is None:
            return 1
        write(record, args.label)
        return 0

    print("type a name and press Enter to capture it; empty line or Ctrl-C to stop\n")
    while True:
        try:
            label = input("name> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not label:
            break
        record = capture(device.interface, registration.name, role, args.seconds)
        if record is not None:
            write(record, label)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
