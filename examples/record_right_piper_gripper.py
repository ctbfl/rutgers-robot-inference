#!/usr/bin/env python
"""Receive-only right-gripper CAN recording. Sends no commands or queries.

Run until Ctrl-C; raw frames and un-clamped readings are saved as JSONL.
Existing actuator commands are not changed by this recorder.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import can

from ruri.client.controllers.single_piper.discovery import read_can_hardware_id


RIGHT_USB = "usb:1d50:606f:002B00464759530920353131"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    channels = [p.name for p in Path("/sys/class/net").glob("can*")
                if read_can_hardware_id(p.name) == RIGHT_USB]
    if len(channels) != 1:
        raise RuntimeError(f"Expected one right-arm USB adapter, got {channels}")
    path = args.log or Path("logs") / f"right_gripper_manual_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    low, high, count = float("inf"), float("-inf"), 0
    started, next_print = time.monotonic(), 0.
    # SocketCAN reception only: no SDK initialization, motor writes, parameter
    # requests, CAN reconfiguration, homing, zeroing or gripper enable changes.
    with can.Bus(interface="socketcan", channel=channels[0],
                 can_filters=[{"can_id": i, "can_mask": 0x7FF, "extended": False}
                              for i in (0x2A8, 0x159, 0x47E)]) as bus, path.open("x", buffering=1) as log:
        log.write(json.dumps({"kind": "setup", "channel": channels[0], "hardware_id": RIGHT_USB,
                              "read_only": True, "started_unix_s": time.time()})+"\n")
        print(f"RECORDING right gripper on {channels[0]} (receive only): {path}", flush=True)
        try:
            while True:
                msg = bus.recv(timeout=.5)
                if msg is None:
                    continue
                elapsed = time.monotonic()-started
                data = bytes(msg.data)
                row = {"kind": "frame", "elapsed_s": elapsed, "unix_s": msg.timestamp,
                       "can_id": hex(msg.arbitration_id), "data_hex": data.hex()}
                if msg.arbitration_id == 0x2A8 and len(data) == 8:
                    raw_value = int.from_bytes(data[:4], "big", signed=True)
                    force = int.from_bytes(data[4:6], "big", signed=True)*.001
                    mode = "width" if data[7] == 0 else "angle" if data[7] == 1 else "unknown"
                    row.update(raw_value=raw_value, mode=mode, force_N=force,
                               status=data[6], enabled=bool(data[6] & 0x40))
                    if mode == "width":
                        width = raw_value*.001  # Raw micrometres -> mm; do not clamp.
                        row["width_mm"] = width
                        low, high = min(low, width), max(high, width)
                        count += 1
                        if elapsed >= next_print:
                            print(f"t={elapsed:7.2f}s width={width:7.2f}mm force={force:7.3f}N "
                                  f"range=[{low:.2f}, {high:.2f}]mm status=0x{data[6]:02x}", flush=True)
                            next_print = elapsed+.1
                log.write(json.dumps(row)+"\n")
        except KeyboardInterrupt:
            print(f"STOPPED: {count} width samples; range [{low:.2f}, {high:.2f}]mm; {path}", flush=True)


if __name__ == "__main__":
    main()
