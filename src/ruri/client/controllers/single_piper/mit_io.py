"""Local-only command and telemetry protocol for the managed MIT worker."""

from __future__ import annotations

import json
import queue
import select
import socket
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


TELEMETRY_PROTOCOL = "piper_mit_telemetry"
COMMAND_PROTOCOL = "piper_mit_policy_command"
CONTROL_PROTOCOL = "piper_mit_teleop_control"
CONTROL_MAX_AGE_NS = 500_000_000
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


class _EffortTrace:
    """Append every telemetry packet to disk without delaying the caller.

    The receiver runs on the control client's own thread, so the file write has
    to leave it: one queue hand-off per packet, one writer thread, same shape as
    the schedulers' own trace. Line-buffered and flushed per record, because
    these runs end with Ctrl-C and a buffered tail would be the part worth
    keeping.
    """

    def __init__(self, path: Path, address: str):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._records: queue.SimpleQueue[dict[str, Any] | object] = queue.SimpleQueue()
        self._stop_token = object()
        self.error: Exception | None = None
        self._thread = threading.Thread(
            target=self._write_loop, name="ruri-effort-log", daemon=True
        )
        self._thread.start()
        self.record(
            {
                "kind": "trace_header",
                "schema_version": 1,
                "telemetry_address": address,
                "wall_time_ns": time.time_ns(),
            }
        )

    def record(self, packet: Mapping[str, Any]) -> None:
        self.check()
        self._records.put(dict(packet))

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"MIT telemetry log failed: {self.path}") from self.error

    def close(self) -> None:
        self._records.put(self._stop_token)
        self._thread.join()
        self._stream.close()
        self.check()

    def _write_loop(self) -> None:
        try:
            self._write_records()
        except Exception as exc:
            self.error = exc

    def _write_records(self) -> None:
        while True:
            record = self._records.get()
            if record is self._stop_token:
                return
            self._stream.write(
                json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
            )
            self._stream.flush()


class MITTelemetryReceiver:
    def __init__(self, address: str):
        self.address = address
        self.endpoint = parse_local_udp(address)
        self._socket: socket.socket | None = None
        self._latest: dict[str, Any] | None = None
        self._latched: dict[str, Any] | None = None
        self._received_at: float | None = None
        self._trace: _EffortTrace | None = None
        self._state_lock = threading.Lock()
        self._reader_stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._reader_error: Exception | None = None

    def record_to(self, path: Path, *, background: bool = False) -> None:
        """Keep every packet drained from here, not just the latest one.

        The effort a rollout produced is already in every telemetry packet, at
        the worker's full rate; this receiver simply throws all but the newest
        away. Recording is therefore a tee on the drain, and costs the control
        path one queue put per packet.

        With background=True, call after open(): a dedicated reader keeps
        receiving while the observation loop encodes video or waits in reset.
        Only latch_engaged() advances the per-frame observation/action latch.
        """
        if self._trace is not None:
            raise RuntimeError("MIT telemetry is already being recorded")
        if background and not self.is_open:
            raise RuntimeError("open telemetry before starting background recording")
        self._trace = _EffortTrace(Path(path), self.address)
        # The observer may already have a cached startup packet. Keep it too,
        # so the first dataset frame always has a corresponding raw record.
        if self._latest is not None:
            self._trace.record(self._latest)
        if background:
            self._reader_stop.clear()
            self._reader_error = None
            self._reader_thread = threading.Thread(
                target=self._read_loop, name="ruri-telemetry-reader", daemon=True
            )
            self._reader_thread.start()

    def _read_loop(self) -> None:
        try:
            while not self._reader_stop.is_set():
                self._receive(0.02)
        except Exception as exc:
            self._reader_error = exc

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

    def check_recording(self) -> None:
        """Surface background receive/write errors on the recording thread."""
        if self._reader_error is not None:
            raise RuntimeError("MIT background telemetry failed") from self._reader_error
        if self._trace is not None:
            self._trace.check()

    def _drain(self, wait_s: float = 0.0) -> None:
        self.check_recording()
        if self._reader_thread is None:
            self._receive(wait_s)

    def _receive(self, wait_s: float = 0.0) -> None:
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
            if self._trace is not None:
                self._trace.record(packet)
            with self._state_lock:
                self._latest = packet
                self._received_at = time.monotonic()

    def latest_engaged(self, timeout_s: float) -> Mapping[str, Any]:
        self._drain(timeout_s)
        with self._state_lock:
            packet, received_at = self._latest, self._received_at
        if packet is None or received_at is None:
            raise TimeoutError(f"No MIT telemetry received from {self.address}")
        age = time.monotonic() - received_at
        if age > timeout_s:
            raise TimeoutError(f"MIT telemetry is stale by {age:.3f}s")
        phase = str(packet.get("phase", "")).lower()
        if phase != "engaged":
            reason = packet.get("reason")
            raise RuntimeError(f"MIT worker phase={phase!r} reason={reason!r}")
        return packet

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
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join()
            self._reader_thread = None
        reader_error, self._reader_error = self._reader_error, None
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self._latest = None
        self._latched = None
        self._received_at = None
        if self._trace is not None:
            trace = self._trace
            self._trace = None
            trace.close()
        if reader_error is not None:
            raise RuntimeError("MIT background telemetry failed") from reader_error


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


class MITControlSender:
    """Ask a running teleop worker to do something. Currently only "home".

    Separate from MITCommandSender: that one streams policy targets to the
    policy worker, this one sends a rare, discrete request to the teleop worker.
    Sharing a socket would mean a dropped policy frame and an ignored request
    look the same on the wire.
    """

    def __init__(self, address: str):
        self.destination = parse_local_udp(address)
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def request_home(self) -> None:
        self.sequence += 1
        packet = {
            "protocol": CONTROL_PROTOCOL,
            "version": PROTOCOL_VERSION,
            "session": self.session,
            "sequence": self.sequence,
            "monotonic_ns": time.monotonic_ns(),
            "type": "home",
        }
        self._socket.sendto(
            json.dumps(packet, allow_nan=False, separators=(",", ":")).encode(),
            self.destination,
        )

    def close(self) -> None:
        self._socket.close()


class ControlRequestReceiver:
    """Non-blocking receiver for teleop control requests.

    ``take_home_request`` drains whatever arrived and reports whether a home was
    asked for, so a burst of duplicates during one 10 ms tick is one request.
    """

    def __init__(self, address: str):
        self.endpoint = parse_local_udp(address)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.bind(self.endpoint)
        except OSError:
            self._socket.close()
            raise
        self._socket.setblocking(False)
        self._seen: dict[str, int] = {}

    def take_home_request(self) -> bool:
        requested = False
        # Bound work per tick so a burst cannot starve the motor loop.
        for _ in range(64):
            try:
                payload, _ = self._socket.recvfrom(65_535)
            except BlockingIOError:
                return requested
            except OSError:
                return requested
            try:
                packet = json.loads(payload)
            except (ValueError, TypeError):
                continue
            if (
                not isinstance(packet, dict)
                or packet.get("protocol") != CONTROL_PROTOCOL
                or packet.get("version") != PROTOCOL_VERSION
                or packet.get("type") != "home"
            ):
                continue
            session = packet.get("session")
            sequence = packet.get("sequence")
            stamp = packet.get("monotonic_ns")
            if (
                not isinstance(session, str) or not session or len(session) > 128
                or type(sequence) is not int or type(stamp) is not int
                or not 0 <= time.monotonic_ns() - stamp <= CONTROL_MAX_AGE_NS
            ):
                continue
            if sequence <= self._seen.get(session, 0):
                continue
            self._seen[session] = sequence
            requested = True
        return requested

    def close(self) -> None:
        self._socket.close()
