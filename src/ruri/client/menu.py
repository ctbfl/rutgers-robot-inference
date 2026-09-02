"""Read-only client for discovering live RURI policy servers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import zmq

from ruri.common.zmq import recv as recv_message
from ruri.common.zmq import send as send_message


DEFAULT_MENU_ENDPOINT = "tcp://172.16.68.130:5550"


def list_policies(
    endpoint: str = DEFAULT_MENU_ENDPOINT,
    *,
    timeout_s: float = 3.0,
) -> list[dict[str, Any]]:
    """Return validated entries from a RURI menu without selecting one."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("menu endpoint must be a non-empty ZeroMQ address")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("menu timeout must be finite and positive")

    timeout_ms = max(1, round(timeout_s * 1_000))
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.connect(endpoint.strip())
    try:
        try:
            send_message(socket, {"type": "list"})
            response = recv_message(socket)
        except zmq.Again as exc:
            raise TimeoutError(
                f"Timed out waiting for policy menu at {endpoint.strip()}"
            ) from exc
    finally:
        socket.close(linger=0)
        context.term()

    if not isinstance(response, Mapping):
        raise TypeError(
            f"Policy menu response must be a mapping, got {type(response).__name__}"
        )
    if "error" in response:
        raise RuntimeError(f"Policy menu failed: {response['error']}")
    policies = response.get("policies")
    if not isinstance(policies, list):
        raise ValueError("Policy menu response must contain a 'policies' list")

    validated: list[dict[str, Any]] = []
    for index, entry in enumerate(policies):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Policy menu entry {index} must be a mapping")
        policy_endpoint = entry.get("endpoint")
        describe = entry.get("describe")
        if not isinstance(policy_endpoint, str) or not policy_endpoint.strip():
            raise ValueError(
                f"Policy menu entry {index} has no valid endpoint"
            )
        if not isinstance(describe, Mapping):
            raise ValueError(
                f"Policy menu entry {index} has no valid describe metadata"
            )
        validated.append(
            {
                "name": entry.get("name"),
                "endpoint": policy_endpoint.strip(),
                "describe": dict(describe),
            }
        )
    return validated
