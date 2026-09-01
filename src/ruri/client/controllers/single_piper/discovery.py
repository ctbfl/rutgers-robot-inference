"""Automatic discovery for the fixed one-Piper/two-RealSense setup."""

from __future__ import annotations

import socket
import struct
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PIPER_STATUS_ID = 0x2A1
PIPER_JOINT_ID_SETS = (
    frozenset((0x2A5, 0x2A6, 0x2A7)),
    frozenset((0x2AA, 0x2AB, 0x2AC)),
)


@dataclass(frozen=True, slots=True)
class RealSenseDevices:
    head_serial: str
    wrist_serial: str


@dataclass(frozen=True, slots=True)
class PiperCanDevice:
    interface: str
    frames_seen: int
    arbitration_ids: frozenset[int]
    hardware_id: str | None = None


def discover_realsense_devices(
    finder: Callable[[], list[dict[str, Any]]] | None = None,
) -> RealSenseDevices:
    """Require exactly one D435 and one D415, selected by model then serial."""
    if finder is None:
        try:
            from lerobot.cameras.realsense import RealSenseCamera
        except ImportError as exc:
            raise ImportError(
                "Single Piper camera discovery requires LeRobot's intelrealsense extra"
            ) from exc
        finder = RealSenseCamera.find_cameras

    devices = finder()
    head = [item for item in devices if "D435" in str(item.get("name", "")).upper()]
    wrist = [item for item in devices if "D415" in str(item.get("name", "")).upper()]
    if len(head) != 1 or len(wrist) != 1:
        summary = [(item.get("name"), item.get("id")) for item in devices]
        raise RuntimeError(
            "Single Piper requires exactly one D435 and one D415; "
            f"found D435={len(head)} D415={len(wrist)} devices={summary}"
        )
    return RealSenseDevices(head_serial=str(head[0]["id"]), wrist_serial=str(wrist[0]["id"]))


def list_can_interfaces(sys_class_net: Path = Path("/sys/class/net")) -> list[str]:
    """List Linux SocketCAN interfaces without relying on their names."""
    result = []
    for entry in sys_class_net.iterdir():
        try:
            if (entry / "type").read_text(encoding="ascii").strip() == "280":
                result.append(entry.name)
        except (FileNotFoundError, OSError):
            continue
    return sorted(result)


def read_can_hardware_id(
    interface: str,
    sys_class_net: Path = Path("/sys/class/net"),
) -> str:
    """Return an ID based on adapter hardware, independent of netdev name/port."""
    device_link = sys_class_net / interface / "device"
    try:
        device = device_link.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"Cannot resolve hardware behind CAN interface {interface!r}") from exc

    for candidate in (device, *device.parents):
        try:
            serial = (candidate / "serial").read_text(encoding="ascii").strip()
        except (FileNotFoundError, OSError):
            continue
        if not serial:
            continue
        try:
            vendor = (candidate / "idVendor").read_text(encoding="ascii").strip().lower()
            product = (candidate / "idProduct").read_text(encoding="ascii").strip().lower()
        except (FileNotFoundError, OSError):
            continue
        if vendor and product:
            return f"usb:{vendor}:{product}:{serial}"
    raise RuntimeError(
        f"CAN interface {interface!r} has no USB hardware serial; "
        "refusing to identify an arm from its transient interface name"
    )


def configure_can_interface(interface: str, bitrate: int) -> None:
    """Bring one SocketCAN interface up at the Piper bitrate; send no frames."""
    import os

    prefix = [] if hasattr(os, "geteuid") and os.geteuid() == 0 else ["sudo", "-n"]
    commands = (
        [*prefix, "ip", "link", "set", interface, "down"],
        [*prefix, "ip", "link", "set", interface, "type", "can", "bitrate", str(bitrate)],
        [*prefix, "ip", "link", "set", interface, "up"],
    )
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True)


def probe_piper_feedback(interface: str, timeout_s: float) -> PiperCanDevice:
    """Passively listen for a complete Piper status/joint feedback signature."""
    can_socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    ids: set[int] = set()
    frames = 0
    try:
        can_socket.bind((interface,))
        can_socket.settimeout(min(0.1, timeout_s))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                payload = can_socket.recv(16)
            except TimeoutError:
                continue
            if len(payload) != 16:
                continue
            can_id, _, _ = struct.unpack("=IB3x8s", payload)
            ids.add(can_id & socket.CAN_EFF_MASK)
            frames += 1
    finally:
        can_socket.close()
    return PiperCanDevice(interface, frames, frozenset(ids))


def _has_piper_signature(device: PiperCanDevice) -> bool:
    return PIPER_STATUS_ID in device.arbitration_ids and any(
        joint_ids.issubset(device.arbitration_ids) for joint_ids in PIPER_JOINT_ID_SETS
    )


def discover_piper_can(
    *,
    candidates: Sequence[str] | None = None,
    bitrate: int = 1_000_000,
    timeout_s: float = 1.0,
    configure: bool = True,
    hardware_id: str | None = None,
    configurator: Callable[[str, int], None] = configure_can_interface,
    prober: Callable[[str, float], PiperCanDevice] = probe_piper_feedback,
    identity_reader: Callable[[str], str] = read_can_hardware_id,
) -> PiperCanDevice:
    """Resolve stable adapter identity, then require valid Piper feedback."""
    interfaces = list(candidates) if candidates is not None else list_can_interfaces()
    if not interfaces:
        raise RuntimeError("No SocketCAN interfaces were found")

    identities = {}
    identity_errors = {}
    for interface in interfaces:
        try:
            identities[interface] = identity_reader(interface)
        except (OSError, RuntimeError) as exc:
            identity_errors[interface] = str(exc)

    if hardware_id is not None:
        interfaces = [
            interface for interface in interfaces if identities.get(interface) == hardware_id
        ]
        if len(interfaces) != 1:
            raise RuntimeError(
                f"Expected one CAN adapter with hardware ID {hardware_id!r}; "
                f"found interfaces={identities}, identity_errors={identity_errors}"
            )

    probes = []
    setup_errors = {}
    for interface in interfaces:
        try:
            if configure:
                configurator(interface, bitrate)
            probe = prober(interface, timeout_s)
            probes.append(replace(probe, hardware_id=identities.get(interface)))
        except (OSError, subprocess.SubprocessError) as exc:
            setup_errors[interface] = str(exc)

    active = [device for device in probes if _has_piper_signature(device)]
    if len(active) != 1:
        summary = {
            device.interface: {
                "frames": device.frames_seen,
                "ids": [hex(value) for value in sorted(device.arbitration_ids)],
            }
            for device in probes
        }
        raise RuntimeError(
            "Expected exactly one active Piper CAN bus; "
            f"found {len(active)} active, probes={summary}, setup_errors={setup_errors}"
        )
    return active[0]
