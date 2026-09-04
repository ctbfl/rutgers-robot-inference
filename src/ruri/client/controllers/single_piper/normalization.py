"""Single-Piper hardware convention shared by collection and inference."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import calibration_ranges


JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_NAME = "gripper"
ACTION_NAMES = (*JOINT_NAMES, GRIPPER_NAME)
ACTION_KEYS = tuple(f"{name}.pos" for name in ACTION_NAMES)
EFFORT_KEYS = tuple(f"{name}.effort" for name in ACTION_NAMES)
ACTION_LOWER = np.asarray([-100.0] * 6 + [0.0], dtype=np.float32)
ACTION_UPPER = np.asarray([100.0] * 7, dtype=np.float32)

# Piper raw joint values are 0.001 degree and gripper values are micrometres.
#
# The envelope lives in calibration/piper_range.json rather than here, next to
# the two files recording what each arm on this rig actually reaches. It is the
# AgileX manual's figures with joint6 and the gripper corrected: the manual is
# accurate to 3.4 deg on joint1-joint5, but states joint6 as +/-100 deg when
# both arms measure +/-172, and the gripper as 70 mm against a measured 82.4 mm.
#
# Replaces the table inherited from WeGo, which additionally narrowed joint5 to
# +/-65 and gave joint6 the asymmetric [-100, 130] that matched neither the
# manual nor either arm, and which truncated joint6 on 29.2% of the frames in
# tight_insertion_row_1.
CALIBRATION_RANGES = {
    name: tuple(bounds) for name, bounds in calibration_ranges.NOMINAL.ranges.items()
}


def _normalize_raw(name: str, raw_value: float) -> float:
    minimum, maximum = CALIBRATION_RANGES[name]
    bounded = min(maximum, max(minimum, float(raw_value)))
    fraction = (bounded - minimum) / (maximum - minimum)
    return fraction * 100.0 if name == GRIPPER_NAME else fraction * 200.0 - 100.0


def _denormalize_raw(name: str, normalized_value: float) -> float:
    value = float(normalized_value)
    if not math.isfinite(value):
        raise ValueError(f"Piper action {name}.pos must be finite")
    minimum, maximum = CALIBRATION_RANGES[name]
    fraction = value / 100.0 if name == GRIPPER_NAME else (value + 100.0) / 200.0
    return minimum + fraction * (maximum - minimum)


def normalize_pose(q_rad: Sequence[float], gripper_width_m: float) -> np.ndarray:
    q = np.asarray(q_rad, dtype=np.float64)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("Piper pose must contain six finite joint values")
    width = float(gripper_width_m)
    if not math.isfinite(width):
        raise ValueError("Piper gripper width must be finite")

    result = [
        _normalize_raw(name, math.degrees(value) * 1_000.0)
        for name, value in zip(JOINT_NAMES, q, strict=True)
    ]
    result.append(_normalize_raw(GRIPPER_NAME, width * 1_000_000.0))
    return np.asarray(result, dtype=np.float32)


def validate_action(action: Any) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (7,):
        raise ValueError(f"Single Piper action must have shape (7,), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Single Piper action contains NaN or infinity")

    if np.any(values < ACTION_LOWER) or np.any(values > ACTION_UPPER):
        raise ValueError(
            "Single Piper action is outside its normalized target range"
        )
    return values.copy()


def denormalize_action(action: Any) -> tuple[np.ndarray, float]:
    values = validate_action(action)
    raw = {
        name: _denormalize_raw(name, value)
        for name, value in zip(ACTION_NAMES, values, strict=True)
    }
    q_rad = np.asarray(
        [math.radians(raw[name] / 1_000.0) for name in JOINT_NAMES],
        dtype=np.float64,
    )
    return q_rad, raw[GRIPPER_NAME] / 1_000_000.0


def normalize_telemetry(packet: Mapping[str, Any]) -> np.ndarray:
    follower = packet.get("follower")
    if not isinstance(follower, Mapping):
        raise ValueError("MIT telemetry is missing follower state")
    return normalize_pose(follower.get("q", ()), follower.get("gripper_width"))


def normalize_teleop_target(packet: Mapping[str, Any]) -> np.ndarray:
    """Read the follower target from the same MIT packet as its state."""
    action = packet.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("MIT telemetry is missing follower target")
    return normalize_pose(action.get("q", ()), action.get("gripper_width"))


def _recording_vector(
    values: Any,
    size: int,
    label: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"MIT telemetry {label} must contain {size} finite values")
    return result.copy()


def recording_observation(packet: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Decode one follower packet into RURI's LeRobot recording convention.

    ``observation.state`` preserves the normalized seven-value convention used
    by the existing Piper datasets. ``observation.joint_effort`` contains six
    signed joint torques in N*m followed by signed gripper force in N.
    """
    follower = packet.get("follower")
    if not isinstance(follower, Mapping):
        raise ValueError("MIT telemetry is missing follower state")

    joint_effort = _recording_vector(
        follower.get("joint_effort"),
        6,
        "joint effort",
    )
    gripper_effort = _recording_vector(
        [follower.get("gripper_force")],
        1,
        "gripper effort",
    )
    return {
        "observation.state": normalize_telemetry(packet),
        "observation.joint_effort": np.concatenate((joint_effort, gripper_effort)),
    }
