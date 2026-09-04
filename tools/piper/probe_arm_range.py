#!/usr/bin/env python3
"""Measure what a Piper arm physically reaches, joints and gripper together.

Arms are named, not addressed by cable: pass ``left_arm`` / ``right_arm`` and
the adapter serial in the hardware registry decides which SocketCAN interface
that is today.

Everything is decoded straight from the arm's own feedback frames -- joints
from 0x2A5/0x2A6/0x2A7, the gripper from 0x2A8 -- so the numbers are before any
SDK software limit, URDF value or dataset calibration table. That matters:
those four sources disagree with each other and with this hardware, most
severely on joint6, where the AgileX manual says +/-100 deg and both arms here
reach +/-172.

Listening is passive and sends nothing. ``--disable-torque`` additionally
de-energizes the arm so it can be dragged by hand, which is how the mechanical
stops are found; that is the only command this tool ever issues.

    tools/piper/probe_arm_range.py --disable-torque
    tools/piper/probe_arm_range.py --arm right_arm
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ruri.client.controllers.single_piper import calibration_ranges as calib  # noqa: E402
from ruri.client.controllers.single_piper.discovery import discover_piper_can  # noqa: E402
from ruri.client.controllers.single_piper.hardware_registry import (  # noqa: E402
    ARM_NAMES,
    find_arm_by_name,
)

JOINT_FRAMES = {0x2A5: (0, 1), 0x2A6: (2, 3), 0x2A7: (4, 5)}
GRIPPER_FRAME = 0x2A8
#: joints in 0.001 deg, gripper in micrometres -- the Piper raw units the
#: calibration files also use, so readings compare without conversion.
CHANNELS = (*(f"joint{i}" for i in range(1, 7)), "gripper")


class ArmRecorder:
    """Track one arm's extreme readings on every channel."""

    def __init__(self, name: str, interface: str, hardware_id: str):
        self.name = name
        self.interface = interface
        self.hardware_id = hardware_id
        self.low = {c: float("inf") for c in CHANNELS}
        self.high = {c: float("-inf") for c in CHANNELS}
        self.joint_frames = 0
        self.gripper_frames = 0

    def observe(self, channel: str, value: float) -> None:
        if value < self.low[channel]:
            self.low[channel] = value
        if value > self.high[channel]:
            self.high[channel] = value

    @property
    def seen_any(self) -> bool:
        return self.joint_frames > 0 or self.gripper_frames > 0

    def span(self, channel: str) -> float:
        return self.high[channel] - self.low[channel]


def read_bus(rec: ArmRecorder, stop: threading.Event) -> None:
    """Receive-only. A dropped adapter is retried rather than raised: an
    interface bounce mid-session would otherwise lose the whole record."""
    sock = None
    while not stop.is_set():
        try:
            if sock is None:
                sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                sock.bind((rec.interface,))
                sock.settimeout(0.5)
            frame = sock.recv(16)
        except socket.timeout:
            continue
        except OSError:
            if sock is not None:
                sock.close()
                sock = None
            time.sleep(0.5)
            continue

        can_id, dlc = struct.unpack("=IB3x", frame[:8])
        can_id &= socket.CAN_EFF_MASK
        if dlc < 8:
            continue
        if can_id in JOINT_FRAMES:
            first, second = struct.unpack(">ii", frame[8:16])
            for slot, raw in zip(JOINT_FRAMES[can_id], (first, second)):
                rec.observe(f"joint{slot + 1}", float(raw))
            rec.joint_frames += 1
        elif can_id == GRIPPER_FRAME:
            width_um, _force_mn = struct.unpack(">ih", frame[8:14])
            rec.observe("gripper", float(width_um))
            rec.gripper_frames += 1

    if sock is not None:
        sock.close()


def resolve(names: list[str]) -> list[ArmRecorder]:
    out = []
    for name in names:
        registration = find_arm_by_name(name)
        device = discover_piper_can(hardware_id=registration.can_hardware_id)
        out.append(ArmRecorder(name, device.interface, registration.can_hardware_id))
    return out


def disable(rec: ArmRecorder) -> None:
    from ruri.client.controllers.single_piper.mit.leader_follower import build_arm

    arm = build_arm(rec.interface)
    print(f"  {rec.name}: joint enable status {arm.get_joints_enable_status_list()}")
    arm.disable()
    time.sleep(0.3)
    print(f"  {rec.name}: de-energized, now {arm.get_joints_enable_status_list()}")


def unit(channel: str) -> str:
    return "um" if channel == "gripper" else "mdeg"


def human(channel: str, raw: float) -> str:
    return f"{raw / 1000.0:8.2f}" + (" mm" if channel == "gripper" else " deg")


def summarize(recorders: list[ArmRecorder]) -> None:
    for rec in recorders:
        if not rec.seen_any:
            print(f"\n{rec.name}: no feedback on {rec.interface}")
            continue
        stored = calib.MEASURED_BY_ARM.get(rec.name)
        print(f"\n=== {rec.name} ({rec.interface}, {rec.joint_frames} joint / "
              f"{rec.gripper_frames} gripper frames) ===\n")
        print(f"{'':>8} {'min':>12} {'max':>12} {'travel':>12}   vs stored calibration")
        for channel in CHANNELS:
            if rec.low[channel] == float("inf"):
                print(f"{channel:>8} {'-- not moved --':>38}")
                continue
            line = (f"{channel:>8} {human(channel, rec.low[channel])} "
                    f"{human(channel, rec.high[channel])} "
                    f"{human(channel, rec.span(channel))}")
            if stored is not None:
                s_lo, s_hi = stored[channel]
                line += (f"   {(rec.low[channel] - s_lo) / 1000.0:+7.2f} /"
                         f" {(rec.high[channel] - s_hi) / 1000.0:+7.2f}")
            print(line)
        if stored is not None:
            print("  (last column: how far this run reached beyond the stored file,"
                  " negative = stopped short)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", choices=[*ARM_NAMES, "all"],
                        help="repeatable; defaults to all arms")
    parser.add_argument("--disable-torque", action="store_true",
                        help="de-energize first so the arm can be dragged by hand")
    parser.add_argument("--json", type=Path,
                        help="also write the raw extremes here")
    args = parser.parse_args()

    chosen = args.arm or ["all"]
    names = list(ARM_NAMES) if "all" in chosen else list(dict.fromkeys(chosen))

    try:
        recorders = resolve(names)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    for rec in recorders:
        print(f"{rec.name} -> {rec.interface}  (adapter {rec.hardware_id})")

    stop = threading.Event()
    for rec in recorders:
        threading.Thread(target=read_bus, args=(rec, stop), daemon=True).start()
    time.sleep(1.5)

    if not any(rec.seen_any for rec in recorders):
        print("No feedback frames on any bus -- are the arms powered?", file=sys.stderr)
        stop.set()
        return 1

    if args.disable_torque:
        print("\nDe-energizing. A supported arm will sag under gravity once torque "
              "is removed.")
        for rec in recorders:
            disable(rec)

    print("\nDrive every joint onto both mechanical stops and run the gripper fully "
          "open and shut, then Ctrl-C.\n")
    try:
        while True:
            time.sleep(0.5)
            print("\r" + "   ".join(
                f"{rec.name}: j6 {rec.low['joint6'] / 1000:7.1f}/"
                f"{rec.high['joint6'] / 1000:<7.1f} "
                f"grip {rec.low['gripper'] / 1000:5.1f}/{rec.high['gripper'] / 1000:<5.1f}"
                for rec in recorders if rec.seen_any), end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()

    print()
    summarize(recorders)

    if args.json:
        payload = {
            rec.name: {
                "interface": rec.interface,
                "can_hardware_id": rec.hardware_id,
                "joint_frames": rec.joint_frames,
                "gripper_frames": rec.gripper_frames,
                "units": {"joints": "0.001 deg", "gripper": "micrometre"},
                "ranges": {c: [rec.low[c], rec.high[c]] for c in CHANNELS
                           if rec.low[c] != float("inf")},
            }
            for rec in recorders if rec.seen_any
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    if args.disable_torque:
        print("\nThe arms are still de-energized; nothing was re-enabled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
