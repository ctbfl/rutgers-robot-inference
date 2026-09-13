"""Local assisted commands and a single continuously drained telemetry socket."""

from __future__ import annotations

from collections import deque
import copy
import json
import select
import socket
import threading
import time
import uuid

from ruri.client.controllers.single_piper.mit.telemetry import (
    PROTOCOL_NAME, PROTOCOL_VERSION, parse_udp_endpoint,
)
from .control import PROTOCOL


class AssistedIO:
    """No CAN, camera, dataset or disk I/O. Trace overflow is explicit."""

    def __init__(self, config, worker_session: str):
        self.config = config
        self.worker_session = worker_session
        self.client_session = uuid.uuid4().hex
        self._receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        endpoint = parse_udp_endpoint(config.telemetry_address)
        self._receiver.bind((endpoint.host, endpoint.port))
        self._receiver.setblocking(False)
        self._sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sender.setblocking(False)
        endpoint = parse_udp_endpoint(config.command_address)
        self._destination = (endpoint.host, endpoint.port)
        self._send_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._condition = threading.Condition()
        self._latest = None
        self._trace = deque()
        self.trace_dropped = self.sequence_gaps = 0
        self._sequence = 0
        self._stop = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name="assisted-telemetry")
        self._thread.start()

    def send(self, kind: str, **fields) -> int:
        with self._send_lock:
            self._sequence += 1
            packet = {"protocol": PROTOCOL, "worker_session": self.worker_session,
                      "client_session": self.client_session, "sequence": self._sequence,
                      "monotonic_ns": time.monotonic_ns(), "kind": kind, **fields}
            self._sender.sendto(json.dumps(packet, allow_nan=False).encode(), self._destination)
            return self._sequence

    def _loop(self):
        last_heartbeat = 0.
        try:
            while not self._stop.is_set():
                if time.monotonic() - last_heartbeat >= 0.05:
                    try:
                        self.send("heartbeat")
                    except BlockingIOError:
                        pass
                    last_heartbeat = time.monotonic()
                if not select.select([self._receiver], [], [], 0.01)[0]:
                    continue
                # Bound each drain so a flooded socket cannot starve heartbeat.
                for _ in range(256):
                    try:
                        payload, source = self._receiver.recvfrom(65535)
                    except BlockingIOError:
                        break
                    if source[0] != "127.0.0.1":
                        continue
                    try:
                        packet = json.loads(payload)
                        control = packet.get("control", {})
                        if (packet.get("protocol") != PROTOCOL_NAME
                                or packet.get("version") != PROTOCOL_VERSION
                                or control.get("worker_session") != self.worker_session):
                            continue
                        stamp, seq = packet["monotonic_ns"], packet["sequence"]
                        if type(stamp) is not int or type(seq) is not int:
                            continue
                        age = (time.monotonic_ns() - stamp) / 1e9
                        if not -0.01 <= age <= self.config.telemetry_timeout_s:
                            continue
                    except (ValueError, TypeError, AttributeError, KeyError):
                        continue
                    with self._condition:
                        if self._latest and seq <= self._latest["sequence"]:
                            continue
                        if self._latest:
                            self.sequence_gaps += max(0, seq - self._latest["sequence"] - 1)
                        self._latest = packet
                        if len(self._trace) >= self.config.trace_queue_size:
                            self._trace.popleft()
                            self.trace_dropped += 1
                        self._trace.append(packet)
                        self._condition.notify_all()
        except BaseException as exc:
            self.error = exc
            with self._condition:
                self._condition.notify_all()

    def snapshot(self, wait_s=0.):
        deadline = time.monotonic() + wait_s
        with self._condition:
            while self._latest is None and self.error is None and time.monotonic() < deadline:
                self._condition.wait(max(0., deadline - time.monotonic()))
            if self.error:
                raise RuntimeError("assisted telemetry receiver failed") from self.error
            if self._latest is None:
                raise TimeoutError("no assisted telemetry")
            if (time.monotonic_ns() - self._latest["monotonic_ns"]) / 1e9 > self.config.telemetry_timeout_s:
                raise TimeoutError("assisted telemetry is stale")
            return copy.deepcopy(self._latest)

    def drain_trace(self):
        with self._condition:
            result = list(self._trace)
            self._trace.clear()
            return {"packets": result, "dropped": self.trace_dropped, "sequence_gaps": self.sequence_gaps}

    def request_mode(self, mode: str, timeout_s=1.):
        with self._request_lock:
            request_id = uuid.uuid4().hex
            epoch = self.snapshot()["control"]["epoch"]
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                self.send("mode", mode=mode, epoch=epoch, request_id=request_id)
                with self._condition:
                    self._condition.wait(min(0.025, max(0., deadline - time.monotonic())))
                control = self.snapshot()["control"]
                ack = control.get("last_request")
                if ack and ack["request_id"] == request_id:
                    if not ack["accepted"]:
                        # A worker timeout can revoke the epoch between snapshot
                        # and send. Retry that race, never other rejected modes.
                        if ack["reason"] == "stale control epoch":
                            epoch, request_id = control["epoch"], uuid.uuid4().hex
                            continue
                        raise RuntimeError(f"{mode} rejected: {ack['reason']}")
                    return control
            raise TimeoutError(f"worker did not acknowledge {mode}")

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.)
        self._receiver.close()
        self._sender.close()

