"""Legacy Piper observer configuration for LeRobot 0.5.2."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots import RobotConfig

from ruri.client.controllers.single_piper.mit_io import parse_local_udp


@RobotConfig.register_subclass("piper_mit_observer")
@dataclass(kw_only=True)
class PiperMITObserverConfig(RobotConfig):
    telemetry_address: str = "udp://127.0.0.1:6670"
    telemetry_timeout_s: float = 0.25
    arm_connect_timeout_s: float = 120.0
    camera_timeout_ms: int = 500
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    head_camera_serial: str | None = None
    wrist_camera_serial: str | None = None
    action_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        parse_local_udp(self.telemetry_address)
        if not 0.05 <= self.telemetry_timeout_s <= 2.0:
            raise ValueError("telemetry_timeout_s must be between 0.05 and 2.0")
        if not 1.0 <= self.arm_connect_timeout_s <= 600.0:
            raise ValueError("arm_connect_timeout_s must be between 1 and 600")
        if not 0.0 < self.action_tolerance <= 0.1:
            raise ValueError("action_tolerance must be in (0, 0.1]")
