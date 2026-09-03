"""Configuration for the dual-arm MIT teleop controller."""

from __future__ import annotations

from dataclasses import dataclass

from ruri.client.controllers.single_piper.config import SinglePiperConfig


@dataclass(frozen=True, slots=True)
class SinglePiperLeaderFollowerTeleopConfig(SinglePiperConfig):
    leader_side: str = "right"
    leader_role: str = "secondary"
    follower_side: str = "left"
    follower_role: str = "main"
    leader_can_hardware_id: str | None = None
    follower_can_hardware_id: str | None = None
    leader_can_interface: str | None = None
    follower_can_interface: str | None = None

    mit_rate_hz: float = 100.0
    leader_kd: float = 0.2
    follower_kp: float = 10.0
    follower_kd: float = 0.8
    engage_seconds: float = 2.0
    start_home_speed: float = 0.2
    max_start_gap: float = 0.35
    max_track_error: float = 0.35
    max_joint_speed: float = 4.0
    max_reference_speed: float = 3.0
    feedback_timeout_s: float = 0.2
    show_periodic_status: bool = True
    seconds: float = 0.0
    abort_hold_seconds: float = 10.0
    abort_home_speed: float = 0.2
    use_gripper: bool = True
    grip_force: float = 1.0
    grip_base: float = 0.1
    grip_gain: float = 1.0
    grip_max_force: float = 5.0

    def __post_init__(self) -> None:
        SinglePiperConfig.__post_init__(self)
        if not 10.0 <= self.mit_rate_hz <= 200.0:
            raise ValueError("mit_rate_hz must be between 10 and 200")
        if not 0.0 <= self.leader_kd <= 5.0:
            raise ValueError("leader_kd must be between 0 and 5")
        if not 0.0 < self.follower_kp <= 50.0:
            raise ValueError("follower_kp must be in (0, 50]")
        if not 0.0 < self.follower_kd <= 2.0:
            raise ValueError("follower_kd must be in (0, 2]")
        if self.engage_seconds < 0.5:
            raise ValueError("engage_seconds must be at least 0.5")
        if not 0.05 <= self.start_home_speed <= 0.5:
            raise ValueError("start_home_speed must be between 0.05 and 0.5")
        if not 0.0 < self.max_start_gap <= 0.5:
            raise ValueError("max_start_gap must be in (0, 0.5]")
        if not 0.0 < self.max_track_error <= 0.7:
            raise ValueError("max_track_error must be in (0, 0.7]")
        if not 0.0 < self.max_joint_speed <= 5.0:
            raise ValueError("max_joint_speed must be in (0, 5]")
        if not 0.0 < self.max_reference_speed <= 5.0:
            raise ValueError("max_reference_speed must be in (0, 5]")
        if not 0.05 <= self.feedback_timeout_s <= 1.0:
            raise ValueError("feedback_timeout_s must be between 0.05 and 1.0")
        if self.seconds < 0.0:
            raise ValueError("seconds cannot be negative")
        if not 0.5 <= self.abort_hold_seconds <= 60.0:
            raise ValueError("abort_hold_seconds must be between 0.5 and 60")
        if not 0.05 <= self.abort_home_speed <= 0.5:
            raise ValueError("abort_home_speed must be between 0.05 and 0.5")
        if not 0.0 < self.grip_force <= 5.0:
            raise ValueError("grip_force must be in (0, 5]")
        if not 0.0 < self.grip_base <= 1.0:
            raise ValueError("grip_base must be in (0, 1]")
        if not 0.0 <= self.grip_gain <= 5.0:
            raise ValueError("grip_gain must be in [0, 5]")
        if not self.grip_base <= self.grip_max_force <= 10.0:
            raise ValueError("grip_max_force must be >= grip_base and <= 10")
