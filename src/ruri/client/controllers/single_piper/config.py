"""Configuration for the single-Piper robot setup."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ruri.client._args import get_arg


@dataclass(frozen=True, slots=True)
class SinglePiperConfig:
    """Machine-level settings; devices are discovered unless overridden."""

    arm_side: str | None = None
    arm_role: str | None = None
    can_hardware_id: str | None = None
    head_camera_serial: str | None = None
    wrist_camera_serial: str | None = None
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_timeout_ms: int = 500
    can_interface: str | None = None
    can_bitrate: int = 1_000_000
    can_probe_timeout_s: float = 1.0
    configure_can: bool = True

    telemetry_address: str = "udp://127.0.0.1:6670"
    command_address: str = "udp://127.0.0.1:6671"
    arm_connect_timeout_s: float = 120.0
    telemetry_timeout_s: float = 0.25
    command_timeout_s: float = 0.30
    worker_shutdown_timeout_s: float = 45.0
    startup_home_skip_threshold_rad: float = 0.05
    diagnostic_log: Path | None = None

    python_executable: Path | None = None

    @classmethod
    def from_args(cls, args: Any) -> "SinglePiperConfig":
        """Select this component's settings from the shared global args."""
        if args is None:
            return cls()
        if isinstance(args, cls):
            return args

        values = {
            item.name: get_arg(args, item.name, item.default)
            for item in fields(cls)
        }
        for name in ("diagnostic_log", "python_executable"):
            if values[name] is not None:
                values[name] = Path(values[name])
        return cls(**values)

    def __post_init__(self) -> None:
        if (self.arm_side is None) != (self.arm_role is None):
            raise ValueError("arm_side and arm_role must either both be set or both be omitted")
        if self.arm_side == "" or self.arm_role == "":
            raise ValueError("arm_side and arm_role cannot be empty")
        if self.camera_width <= 0 or self.camera_height <= 0 or self.camera_fps <= 0:
            raise ValueError("camera width, height, and fps must be positive")
        if self.camera_timeout_ms <= 0:
            raise ValueError("camera_timeout_ms must be positive")
        if self.can_bitrate <= 0 or self.can_probe_timeout_s <= 0:
            raise ValueError("CAN bitrate and probe timeout must be positive")
        if self.arm_connect_timeout_s <= 0 or self.telemetry_timeout_s <= 0:
            raise ValueError("arm and telemetry timeouts must be positive")
        if self.command_timeout_s <= 0 or self.worker_shutdown_timeout_s <= 0:
            raise ValueError("command and shutdown timeouts must be positive")
        if not 0.0 <= self.startup_home_skip_threshold_rad <= 0.05:
            raise ValueError(
                "startup_home_skip_threshold_rad must be between 0 and 0.05"
            )
