"""Load the reviewed RealSense controls for the single-Piper setup."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CAMERA_PARAMS_PATH = Path(__file__).with_name("cam_params.json")

_ROLES = ("head", "wrist")
_AUTO_CONTROLS = (
    "enable_auto_exposure",
    "enable_auto_white_balance",
)
_MANUAL_CONTROLS = (
    "exposure",
    "gain",
    "brightness",
    "contrast",
    "gamma",
    "hue",
    "saturation",
    "sharpness",
    "backlight_compensation",
)
CAMERA_CONTROL_ORDER = _MANUAL_CONTROLS + _AUTO_CONTROLS


@dataclass(frozen=True, slots=True)
class CameraRoleParams:
    model: str
    controls: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class CameraParams:
    warmup_frames: int
    roles: Mapping[str, CameraRoleParams]


def _expect_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label} has incorrect keys; missing={missing}, extra={extra}"
        )


def load_camera_params(path: str | Path = CAMERA_PARAMS_PATH) -> CameraParams:
    """Load and validate the camera controls before opening either device."""
    params_path = Path(path)
    try:
        raw = json.loads(params_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load camera parameters from {params_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"{params_path} must contain a JSON object")
    _expect_keys(raw, {"warmup_frames", *_ROLES}, str(params_path))

    warmup_frames = raw["warmup_frames"]
    if (
        isinstance(warmup_frames, bool)
        or not isinstance(warmup_frames, int)
        or warmup_frames < 0
    ):
        raise ValueError("camera warmup_frames must be a non-negative integer")

    roles: dict[str, CameraRoleParams] = {}
    expected_controls = set(CAMERA_CONTROL_ORDER)
    for role in _ROLES:
        role_raw = raw[role]
        if not isinstance(role_raw, dict):
            raise ValueError(f"camera role {role!r} must be a JSON object")
        _expect_keys(role_raw, {"model", "controls"}, f"camera role {role!r}")
        model = role_raw["model"]
        controls_raw = role_raw["controls"]
        if not isinstance(model, str) or not model:
            raise ValueError(f"camera role {role!r} model must be a non-empty string")
        if not isinstance(controls_raw, dict):
            raise ValueError(f"camera role {role!r} controls must be a JSON object")
        _expect_keys(
            controls_raw,
            expected_controls,
            f"camera role {role!r} controls",
        )

        controls: dict[str, float] = {}
        for name in CAMERA_CONTROL_ORDER:
            value = controls_raw[name]
            if name in _AUTO_CONTROLS:
                if not isinstance(value, bool):
                    raise ValueError(f"camera {role}.{name} must be boolean")
                controls[name] = float(value)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"camera {role}.{name} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"camera {role}.{name} must be finite")
            controls[name] = number

        if controls["exposure"] <= 0:
            raise ValueError(f"camera {role}.exposure must be positive")
        if controls["gain"] < 0:
            raise ValueError(f"camera {role}.gain must be non-negative")
        roles[role] = CameraRoleParams(model=model, controls=controls)

    return CameraParams(warmup_frames=warmup_frames, roles=roles)
