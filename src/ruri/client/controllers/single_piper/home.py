"""Home references shared by single-arm and assisted interactive clients."""

from __future__ import annotations

import math
import time

import numpy as np

from .normalization import normalize_pose, normalize_telemetry, validate_action


def home_actions(read_packet, *, control_hz, speed_rad_s=1.0, extra_settled=None):
    """Yield zero-joint references; the caller owns pacing and command delivery."""
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    if not np.isfinite(speed_rad_s) or not 0 < speed_rad_s <= 1.0:
        raise ValueError("home speed must be in (0, 1] rad/s")
    packet = read_packet()
    normalize_telemetry(packet)  # Validate state without clipping measured overshoot.
    start_q = np.asarray(packet["follower"]["q"], dtype=float)
    width = packet.get("action", {}).get(
        "gripper_width", packet["follower"]["gripper_width"]
    )
    home = validate_action(normalize_pose(np.zeros(6), width))
    steps = max(1, math.ceil(float(np.max(np.abs(start_q))) * control_hz / speed_rad_s))
    for step in range(steps + 1):
        yield normalize_pose(start_q * (1.0 - step / steps), width)

    deadline = time.monotonic() + 3.0
    settled_since = None
    while True:
        packet = read_packet()
        normalize_telemetry(packet)
        q = np.asarray(packet["follower"]["q"], dtype=float)
        qd = np.asarray(packet["follower"]["qd"], dtype=float)
        if qd.shape != (6,) or not np.all(np.isfinite(qd)):
            raise ValueError("MIT telemetry must contain six finite joint velocities")
        now = time.monotonic()
        if (
            np.max(np.abs(q)) <= 0.05
            and np.max(np.abs(qd)) <= 0.05
            and (extra_settled is None or extra_settled(packet))
        ):
            if settled_since is None:
                settled_since = now
            if now - settled_since >= 0.2:
                return
        else:
            settled_since = None
        if now >= deadline:
            raise TimeoutError(
                "Home did not settle within 3 seconds of the final target"
            )
        yield home.copy()
