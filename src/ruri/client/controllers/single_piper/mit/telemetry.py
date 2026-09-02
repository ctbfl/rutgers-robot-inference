"""Non-blocking, read-only telemetry for :mod:`mit_teleop`.

The controller is the sole CAN owner.  This module only sends compact UDP
datagrams to a recorder on the same machine.  A full socket buffer or a missing
recorder is counted and ignored so dataset collection can never stall the
100 Hz control loop.
"""

from __future__ import annotations

import json
import math
import socket
import time
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

PROTOCOL_NAME = "piper_mit_telemetry"
PROTOCOL_VERSION = 1
DEFAULT_TELEMETRY_ADDRESS = "udp://127.0.0.1:6670"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}


@dataclass(frozen=True)
class UDPEndpoint:
    host: str
    port: int


def parse_udp_endpoint(address: str) -> UDPEndpoint:
    """Parse the deliberately local-only telemetry endpoint."""
    parsed = urlparse(address)
    if parsed.scheme != "udp" or not parsed.hostname or parsed.port is None:
        raise ValueError("telemetry address must look like udp://127.0.0.1:6670")
    if parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("telemetry is restricted to localhost")
    if parsed.path not in ("", "/") or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("telemetry address cannot contain a path, query, or fragment")
    if not 1 <= parsed.port <= 65535:
        raise ValueError("telemetry UDP port must be between 1 and 65535")
    host = "127.0.0.1" if parsed.hostname == "localhost" else parsed.hostname
    return UDPEndpoint(host=host, port=parsed.port)


def _finite_vector(values: Sequence[float] | None, size: int) -> list[float] | None:
    if values is None:
        return None
    result = [float(value) for value in values]
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError(f"telemetry vector must contain {size} finite values")
    return result


def _finite_optional(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("telemetry scalar must be finite")
    return result


class MITTelemetryPublisher:
    """Best-effort UDP publisher whose :meth:`publish` never raises."""

    def __init__(self, address: str):
        endpoint = parse_udp_endpoint(address)
        self.address = address
        self._destination = (endpoint.host, endpoint.port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self.sequence = 0
        self.sent = 0
        self.dropped = 0
        self.last_error: str | None = None

    def publish(
        self,
        *,
        phase: str,
        leader_q: Sequence[float] | None = None,
        leader_qd: Sequence[float] | None = None,
        follower_q: Sequence[float] | None = None,
        follower_qd: Sequence[float] | None = None,
        follower_target_q: Sequence[float] | None = None,
        follower_target_qd: Sequence[float] | None = None,
        leader_gripper_width: float | None = None,
        follower_gripper_width: float | None = None,
        follower_gripper_target: float | None = None,
        follower_gripper_force: float | None = None,
        overruns: int = 0,
        reason: str | None = None,
    ) -> bool:
        """Send one latest-state packet, dropping it on any telemetry error."""
        self.sequence += 1
        try:
            packet = {
                "protocol": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
                "sequence": self.sequence,
                "monotonic_ns": time.monotonic_ns(),
                "wall_time_ns": time.time_ns(),
                "phase": str(phase).lower(),
                "leader": {
                    "q": _finite_vector(leader_q, 6),
                    "qd": _finite_vector(leader_qd, 6),
                    "gripper_width": _finite_optional(leader_gripper_width),
                },
                "follower": {
                    "q": _finite_vector(follower_q, 6),
                    "qd": _finite_vector(follower_qd, 6),
                    "gripper_width": _finite_optional(follower_gripper_width),
                    "gripper_force": _finite_optional(follower_gripper_force),
                },
                "action": {
                    "q": _finite_vector(follower_target_q, 6),
                    "qd": _finite_vector(follower_target_qd, 6),
                    "gripper_width": _finite_optional(follower_gripper_target),
                },
                "overruns": int(overruns),
                "reason": None if reason is None else str(reason),
            }
            payload = json.dumps(
                packet,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._socket.sendto(payload, self._destination)
            self.sent += 1
            return True
        except (BlockingIOError, OSError, TypeError, ValueError) as exc:
            self.dropped += 1
            self.last_error = str(exc)
            return False

    def publish_status(self, phase: str, reason: str | None = None) -> bool:
        return self.publish(phase=phase, reason=reason)

    def close(self) -> None:
        self._socket.close()
