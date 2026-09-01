"""Blocking client proxy for a remote RURI policy server."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import zmq

from ruri.client._args import get_arg
from ruri.common.zmq import recv as recv_message
from ruri.common.zmq import send as send_message


class RemotePolicy:
    """Expose a remote ZeroMQ policy server through ``start/infer/stop``."""

    def __init__(self, args: Any):
        self.args = args
        endpoint = get_arg(args, "policy_endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("policy_endpoint must be a non-empty ZeroMQ address")
        self.endpoint = endpoint.strip()
        self.request_timeout_s = float(get_arg(args, "policy_timeout_s", 10.0))
        self.ready_timeout_s = float(get_arg(args, "policy_ready_timeout_s", 30.0))
        self.ready_retry_s = float(get_arg(args, "policy_ready_retry_s", 0.5))

        if (
            not math.isfinite(self.request_timeout_s)
            or not math.isfinite(self.ready_timeout_s)
            or self.request_timeout_s <= 0
            or self.ready_timeout_s <= 0
        ):
            raise ValueError("Policy request and ready timeouts must be positive")
        if not math.isfinite(self.ready_retry_s) or self.ready_retry_s < 0:
            raise ValueError("policy_ready_retry_s must be finite and non-negative")

        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._metadata: dict[str, Any] | None = None

    @property
    def is_started(self) -> bool:
        return self._socket is not None and self._metadata is not None

    @property
    def metadata(self) -> dict[str, Any] | None:
        return None if self._metadata is None else dict(self._metadata)

    def _open_socket(self, timeout_s: float) -> None:
        self._close_socket()
        if self._context is None:
            self._context = zmq.Context()

        timeout_ms = max(1, round(timeout_s * 1_000))
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.connect(self.endpoint)
        self._socket = socket

    def _close_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = None

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._socket is None:
            raise RuntimeError("RemotePolicy.start() must complete before infer()")

        try:
            send_message(self._socket, request)
            response = recv_message(self._socket)
        except zmq.Again as exc:
            raise TimeoutError(
                f"Timed out waiting for policy server at {self.endpoint}"
            ) from exc

        if not isinstance(response, dict):
            raise TypeError(
                f"Policy response must be a dict, got {type(response).__name__}"
            )
        if "error" in response:
            raise RuntimeError(f"Remote policy failed: {response['error']}")
        return response

    def start(self) -> None:
        """Block until the server answers a metadata readiness request."""
        if self.is_started:
            raise RuntimeError("RemotePolicy is already started")

        deadline = time.monotonic() + self.ready_timeout_s
        last_error: BaseException | None = None

        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                self.stop()
                raise TimeoutError(
                    f"Policy server {self.endpoint} was not ready within "
                    f"{self.ready_timeout_s:.3f}s"
                ) from last_error

            try:
                self._open_socket(min(self.request_timeout_s, remaining_s))
                metadata = self._request({"type": "metadata"})
                self._metadata = metadata

                # Readiness requests can use a short timeout, but normal
                # inference gets the full configured request timeout.
                self._open_socket(self.request_timeout_s)
                return
            except (
                OSError,
                RuntimeError,
                TimeoutError,
                TypeError,
                ValueError,
                zmq.ZMQError,
            ) as exc:
                last_error = exc
                self._close_socket()
                sleep_s = min(
                    self.ready_retry_s,
                    max(0.0, deadline - time.monotonic()),
                )
                if sleep_s:
                    time.sleep(sleep_s)

    def infer(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Send Scheduler-assembled policy inputs to the remote server."""
        request = dict(inputs)
        request["type"] = "infer"
        return self._request(request)

    def stop(self) -> None:
        """Idempotently release the client socket and context."""
        self._metadata = None
        self._close_socket()
        if self._context is not None:
            self._context.term()
        self._context = None

    def __enter__(self) -> "RemotePolicy":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
