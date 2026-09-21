"""Hardware settings for one policy follower and its actuated leader."""

import math
from dataclasses import dataclass, fields

from ruri.client.controllers.single_piper_leader_follower_teleop.config import (
    SinglePiperLeaderFollowerTeleopConfig,
)


@dataclass(frozen=True, slots=True)
class SinglePiperAssistedConfig(SinglePiperLeaderFollowerTeleopConfig):
    leader_only: bool = False
    gravity_handoff: bool = False
    follower_handoff_s: float = 0.5
    reset_latched_stop: bool = False
    camera_enabled: bool = True
    startup_home: bool = False
    leader_policy_kp: float = 3.0
    leader_policy_kd: float = 0.5
    alignment_speed: float = 0.2
    alignment_tolerance: float = 0.04
    gripper_reference_speed: float = 0.04
    heartbeat_timeout_s: float = 0.5
    reference_speed: float = 0.15
    policy_reference_speed: float = 0.15
    policy_reference_gain: float = 2.0
    interaction_deadzone: tuple[float, ...] = (0.08,) * 6
    interaction_bias: tuple[float, ...] = (0.0,) * 6
    admittance_damping: tuple[float, ...] = (2.0,) * 6
    human_trigger_angle_deg: float = 5.0
    human_confirm_s: float = 0.1
    interaction_filter_s: float = 0.05
    human_takeover_s: float = 0.5
    human_release_delay_s: float = 0.5
    human_release_s: float = 1.0
    human_quiet_range_deg: tuple[float, ...] = (0.1,) * 6
    interaction_release: float = 0.02
    policy_max_age_s: float = 1.0
    leader_sign: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    leader_offset_rad: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    trace_queue_size: int = 4096
    simulate: bool = False

    @property
    def uses_gravity_handoff(self):
        return self.leader_only or self.gravity_handoff

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(item.default, bool) and type(value) is not bool:
                raise ValueError(f"{item.name} must be a boolean")
            if type(item.default) in (float, int):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"{item.name} must be a finite number")
                if type(item.default) is int and type(value) is not int:
                    raise ValueError(f"{item.name} must be an integer")
        SinglePiperLeaderFollowerTeleopConfig.__post_init__(self)
        for name, low, high in (
            ("leader_policy_kp", 0.1, 10.0),
            ("leader_policy_kd", 0.01, 2.0),
            ("alignment_speed", 0.01, 0.5),
            ("alignment_tolerance", 0.005, 0.1),
            ("reference_speed", 0.01, 4.0),
            ("policy_reference_speed", 0.01, 4.0),
            ("policy_reference_gain", 0.01, 20.0),
            ("human_trigger_angle_deg", 0.1, 15.0),
            ("human_confirm_s", 0.05, 5.0),
            ("interaction_filter_s", 0.001, 0.5),
            ("human_takeover_s", 0.05, 5.0),
            ("follower_handoff_s", 0.05, 5.0),
            ("human_release_delay_s", 0.05, 5.0),
            ("human_release_s", 0.1, 5.0),
            ("interaction_release", 0.0, 10.0),
            ("gripper_reference_speed", 0.001, 0.1),
            ("heartbeat_timeout_s", 0.1, 2.0),
            ("policy_max_age_s", 0.1, 5.0),
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite in [{low}, {high}]")
        if self.human_release_delay_s >= self.human_release_s:
            raise ValueError("require release delay < release duration")
        if len(self.human_quiet_range_deg) != 6 or any(
            isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or not 0.001 <= v <= 1.
            for v in self.human_quiet_range_deg
        ):
            raise ValueError("human_quiet_range_deg requires six values in [0.001, 1] degrees")
        if self.reference_speed > self.max_reference_speed:
            raise ValueError("reference_speed exceeds max_reference_speed")
        if self.policy_reference_speed > self.reference_speed:
            raise ValueError("policy_reference_speed exceeds reference_speed")
        for name in ("interaction_deadzone", "interaction_bias", "admittance_damping"):
            values = getattr(self, name)
            if len(values) != 6 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in values
            ):
                raise ValueError(f"{name} must contain six finite numbers")
            if name == "interaction_deadzone" and any(v < 0 for v in values):
                raise ValueError("interaction deadzone cannot be negative")
            if name == "admittance_damping" and any(v <= 0 for v in values):
                raise ValueError("admittance damping must be positive")
        trigger = self.leader_policy_kp * math.radians(self.human_trigger_angle_deg)
        if any(d + self.interaction_release >= trigger for d in self.interaction_deadzone):
            raise ValueError("interaction release threshold must be below the angular trigger")
        if len(self.leader_sign) != 6 or any(
            v not in (-1.0, 1.0) for v in self.leader_sign
        ):
            raise ValueError("leader_sign must contain six +/-1 values")
        if len(self.leader_offset_rad) != 6 or not all(
            math.isfinite(v) for v in self.leader_offset_rad
        ):
            raise ValueError("leader_offset_rad must contain six finite values")
        if not self.use_gripper or not self.enforce_joint_limits:
            raise ValueError(
                "assisted seven-dimensional control requires grippers and limits"
            )
        if self.leader_only and (tuple(self.leader_sign) != (1.,)*6 or tuple(self.leader_offset_rad) != (0.,)*6):
            raise ValueError("leader-only control uses the physical leader joint coordinates")
        if type(self.simulate) is not bool:
            raise ValueError("simulate must be a boolean")
        if type(self.trace_queue_size) is not int or self.trace_queue_size < 100:
            raise ValueError("trace_queue_size must be at least 100")
