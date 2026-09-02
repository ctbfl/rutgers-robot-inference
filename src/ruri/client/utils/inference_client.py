"""Connect client schedulers to one RURI policy server."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import zmq

from ruri.client._args import get_arg
from ruri.common.zmq import recv as recv_message
from ruri.common.zmq import send as send_message


@dataclass(slots=True)
class _InferenceCall:
    request: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    error: BaseException | None = None


class _PolicyConnection:
    """A connected policy handle whose ZeroMQ socket stays on one I/O thread."""

    def __init__(self, endpoint_or_args: str | Any):
        self.args = None if isinstance(endpoint_or_args, str) else endpoint_or_args
        endpoint = (
            endpoint_or_args
            if isinstance(endpoint_or_args, str)
            else get_arg(endpoint_or_args, "policy_endpoint")
        )
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("policy_endpoint must be a non-empty ZeroMQ address")
        self.endpoint = endpoint.strip()
        self.request_timeout_s = float(
            10.0
            if self.args is None
            else get_arg(self.args, "policy_timeout_s", 10.0)
        )
        self.ready_timeout_s = float(
            30.0
            if self.args is None
            else get_arg(self.args, "policy_ready_timeout_s", 30.0)
        )
        self.ready_retry_s = float(
            0.5
            if self.args is None
            else get_arg(self.args, "policy_ready_retry_s", 0.5)
        )
        if (
            not math.isfinite(self.request_timeout_s)
            or not math.isfinite(self.ready_timeout_s)
            or self.request_timeout_s <= 0
            or self.ready_timeout_s <= 0
        ):
            raise ValueError("Policy request and ready timeouts must be positive")
        if not math.isfinite(self.ready_retry_s) or self.ready_retry_s < 0:
            raise ValueError("policy_ready_retry_s must be finite and non-negative")

        self._metadata: dict[str, Any] | None = None
        self._output_chunk_size: int | None = None
        self._ready: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)
        self._calls: queue.Queue[_InferenceCall | object] = queue.Queue()
        self._stop_token = object()
        self._closed = False
        self._thread = threading.Thread(
            target=self._io_loop,
            name="ruri-policy-connection",
            daemon=True,
        )

    @property
    def is_connected(self) -> bool:
        return (
            not self._closed
            and self._thread.is_alive()
            and self._metadata is not None
            and self._output_chunk_size is not None
        )

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            raise RuntimeError("Policy connection has no server metadata")
        return dict(self._metadata)

    @property
    def output_chunk_size(self) -> int:
        if self._output_chunk_size is None:
            raise RuntimeError("Policy connection has no server metadata")
        return self._output_chunk_size

    @staticmethod
    def _metadata_output_chunk_size(metadata: Mapping[str, Any]) -> int:
        outputs = metadata.get("outputs")
        if not isinstance(outputs, Mapping):
            raise ValueError("Policy metadata must contain an 'outputs' mapping")
        value = outputs.get("output_chunk_size")
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(
                "Policy metadata outputs.output_chunk_size must be a positive integer"
            )
        return int(value)

    @staticmethod
    def _open_socket(
        context: zmq.Context,
        endpoint: str,
        timeout_s: float,
    ) -> zmq.Socket:
        timeout_ms = max(1, round(timeout_s * 1_000))
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.connect(endpoint)
        return socket

    @staticmethod
    def _request(socket: zmq.Socket, request: dict[str, Any]) -> dict[str, Any]:
        try:
            send_message(socket, request)
            response = recv_message(socket)
        except zmq.Again as exc:
            raise TimeoutError("Timed out waiting for policy server") from exc
        if not isinstance(response, dict):
            raise TypeError(
                f"Policy response must be a dict, got {type(response).__name__}"
            )
        if "error" in response:
            raise RuntimeError(f"Remote policy failed: {response['error']}")
        return response

    def _connect_ready_socket(self, context: zmq.Context) -> zmq.Socket:
        deadline = time.monotonic() + self.ready_timeout_s
        last_error: BaseException | None = None
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError(
                    f"Policy server {self.endpoint} was not ready within "
                    f"{self.ready_timeout_s:.3f}s"
                ) from last_error
            socket = self._open_socket(
                context,
                self.endpoint,
                min(self.request_timeout_s, remaining_s),
            )
            try:
                metadata = self._request(socket, {"type": "metadata"})
                output_chunk_size = self._metadata_output_chunk_size(metadata)
            except (
                OSError,
                RuntimeError,
                TimeoutError,
                TypeError,
                zmq.ZMQError,
            ) as exc:
                last_error = exc
                socket.close(linger=0)
                sleep_s = min(
                    self.ready_retry_s,
                    max(0.0, deadline - time.monotonic()),
                )
                if sleep_s:
                    time.sleep(sleep_s)
                continue

            self._metadata = metadata
            self._output_chunk_size = output_chunk_size
            socket.close(linger=0)
            return self._open_socket(context, self.endpoint, self.request_timeout_s)

    def _io_loop(self) -> None:
        context = zmq.Context()
        socket: zmq.Socket | None = None
        ready_reported = False
        try:
            try:
                socket = self._connect_ready_socket(context)
            except BaseException as exc:
                self._ready.put(exc)
                ready_reported = True
                return
            self._ready.put(None)
            ready_reported = True

            while True:
                call = self._calls.get()
                if call is self._stop_token:
                    return
                assert isinstance(call, _InferenceCall)
                try:
                    call.response = self._request(socket, call.request)
                except BaseException as exc:
                    call.error = exc
                    # A timed-out REQ socket cannot send another request until
                    # its missing reply arrives. Recreate it on the same I/O
                    # thread so a transient timeout does not poison all later
                    # inferences.
                    if isinstance(exc, (OSError, TimeoutError, zmq.ZMQError)):
                        socket.close(linger=0)
                        socket = self._open_socket(
                            context,
                            self.endpoint,
                            self.request_timeout_s,
                        )
                finally:
                    call.done.set()
        finally:
            if not ready_reported:
                self._ready.put(RuntimeError("Policy connection stopped during startup"))
            if socket is not None:
                socket.close(linger=0)
            context.term()

    def _connect(self) -> None:
        self._thread.start()
        error = self._ready.get()
        if error is not None:
            self._closed = True
            self._thread.join()
            raise error

    def infer(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("Policy connection is closed")
        request = dict(inputs)
        request["type"] = "infer"
        call = _InferenceCall(request=request)
        self._calls.put(call)
        call.done.wait()
        if call.error is not None:
            raise call.error
        assert call.response is not None
        return call.response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            self._calls.put(self._stop_token)
            self._thread.join(timeout=self.request_timeout_s + 1.0)
        self._metadata = None
        self._output_chunk_size = None

    def __enter__(self) -> "_PolicyConnection":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def connect(endpoint_or_args: str | Any) -> _PolicyConnection:
    """Connect to one policy endpoint and return its ready policy handle."""
    policy = _PolicyConnection(endpoint_or_args)
    policy._connect()
    return policy


__all__ = ["connect"]
