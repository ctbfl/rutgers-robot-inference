"""Local-only command and telemetry protocol for the managed MIT worker."""

from __future__ import annotations

import json
import select
import socket
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse


TELEMETRY_PROTOCOL = "piper_mit_telemetry"
COMMAND_PROTOCOL = "piper_mit_policy_command"
PROTOCOL_VERSION = 1


def parse_local_udp(address: str) -> tuple[str, int]:
    parsed = urlparse(address)
    if parsed.scheme != "udp" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("address must be local UDP, e.g. udp://127.0.0.1:6670")
    if parsed.port is None or not 1 <= parsed.port <= 65535:
        raise ValueError("UDP port must be between 1 and 65535")
    if parsed.path not in ("", "/") or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("UDP address cannot contain a path, params, query, or fragment")
    return "127.0.0.1", parsed.port


class MITTelemetryReceiver:
    def __init__(self, address: str):
        self.address = address
        self.endpoint = parse_local_udp(address)
        self._socket: socket.socket | None = None
        self._latest: dict[str, Any] | None = None
        self._latched: dict[str, Any] | None = None
        self._received_at: float | None = None

    @property
    def is_open(self) -> bool:
        return self._socket is not None

    def open(self) -> None:
        if self._socket is not None:
            return
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(self.endpoint)
        receiver.setblocking(False)
        self._socket = receiver

    def _drain(self, wait_s: float = 0.0) -> None:
        if self._socket is None:
            raise RuntimeError("MIT telemetry receiver is not open")
        if wait_s > 0 and not select.select([self._socket], [], [], wait_s)[0]:
            return
        while True:
            try:
                payload, _ = self._socket.recvfrom(65_535)
            except BlockingIOError:
                return
            packet = json.loads(payload)
            if not isinstance(packet, dict):
                continue
            if packet.get("protocol") != TELEMETRY_PROTOCOL or packet.get("version") != PROTOCOL_VERSION:
                continue
            self._latest = packet
            self._received_at = time.monotonic()

    def latest_engaged(self, timeout_s: float) -> Mapping[str, Any]:
        self._drain(timeout_s)
        if self._latest is None or self._received_at is None:
            raise TimeoutError(f"No MIT telemetry received from {self.address}")
        age = time.monotonic() - self._received_at
        if age > timeout_s:
            raise TimeoutError(f"MIT telemetry is stale by {age:.3f}s")
        phase = str(self._latest.get("phase", "")).lower()
        if phase != "engaged":
            reason = self._latest.get("reason")
            raise RuntimeError(f"MIT worker phase={phase!r} reason={reason!r}")
        return self._latest

    def latch_engaged(self, timeout_s: float) -> Mapping[str, Any]:
        """Latch one control packet for a complete 30 Hz collection frame."""
        packet = self.latest_engaged(timeout_s)
        self._latched = dict(packet)
        return self._latched

    def latched(self) -> Mapping[str, Any]:
        if self._latched is None:
            raise RuntimeError(
                "No MIT packet is latched; collect the observation before its action"
            )
        return self._latched

    def wait_for_engaged(self, timeout_s: float, freshness_s: float) -> Mapping[str, Any]:
        deadline = time.monotonic() + timeout_s
        last_phase = None
        while time.monotonic() < deadline:
            self._drain(min(0.1, deadline - time.monotonic()))
            if self._latest is None or self._received_at is None:
                continue
            last_phase = self._latest.get("phase")
            if time.monotonic() - self._received_at <= freshness_s and last_phase == "engaged":
                return self._latest
            if last_phase in ("aborted", "stopped"):
                raise RuntimeError(
                    f"MIT worker phase={last_phase!r} reason={self._latest.get('reason')!r}"
                )
        raise TimeoutError(f"Timed out waiting for MIT worker phase=engaged; last phase={last_phase!r}")

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self._latest = None
        self._latched = None
        self._received_at = None


_SHARED_TELEMETRY: dict[str, MITTelemetryReceiver] = {}


def get_shared_mit_telemetry(address: str) -> MITTelemetryReceiver:
    """Return the one local receiver shared by thin LeRobot adapters."""
    parse_local_udp(address)
    if address not in _SHARED_TELEMETRY:
        _SHARED_TELEMETRY[address] = MITTelemetryReceiver(address)
    return _SHARED_TELEMETRY[address]


class MITCommandSender:
    def __init__(self, address: str):
        self.destination = parse_local_udp(address)
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, q_rad: Sequence[float], gripper_width_m: float) -> None:
        q = [float(value) for value in q_rad]
        if len(q) != 6:
            raise ValueError("MIT command must contain six joints")
        self.sequence += 1
        packet = {
            "protocol": COMMAND_PROTOCOL,
            "version": PROTOCOL_VERSION,
            "session": self.session,
            "sequence": self.sequence,
            "monotonic_ns": time.monotonic_ns(),
            "q_rad": q,
            "gripper_width_m": float(gripper_width_m),
        }
        self._socket.sendto(
            json.dumps(packet, allow_nan=False, separators=(",", ":")).encode(),
            self.destination,
        )

    def close(self) -> None:
        self._socket.close()
