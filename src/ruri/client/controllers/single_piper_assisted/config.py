"""Hardware settings for one policy follower and its actuated leader."""

import math
from dataclasses import dataclass, fields

from ruri.client.controllers.single_piper_leader_follower_teleop.config import (
    SinglePiperLeaderFollowerTeleopConfig,
)


@dataclass(frozen=True, slots=True)
class SinglePiperAssistedConfig(SinglePiperLeaderFollowerTeleopConfig):
    leader_policy_kp: float = 3.0
    leader_policy_kd: float = 0.5
    gain_transition_s: float = 0.3
    alignment_speed: float = 0.2
    alignment_tolerance: float = 0.04
    human_reference_speed: float = 0.5
    gripper_reference_speed: float = 0.04
    heartbeat_timeout_s: float = 0.5
    intervention_error_rad: float = 0.035
    intervention_release_rad: float = 0.012
    intervention_gripper_error_m: float = 0.003
    intervention_gripper_release_m: float = 0.001
    intervention_confirm_s: float = 0.04
    intervention_release_s: float = 0.4
    replan_distance_rad: float = 0.05
    policy_max_age_s: float = 1.0
    leader_sign: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    leader_offset_rad: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    trace_queue_size: int = 4096
    simulate: bool = False

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
            ("gain_transition_s", 0.05, 3.0),
            ("alignment_speed", 0.01, 0.5),
            ("alignment_tolerance", 0.005, 0.1),
            ("human_reference_speed", 0.01, 2.0),
            ("gripper_reference_speed", 0.001, 0.1),
            ("heartbeat_timeout_s", 0.1, 2.0),
            ("intervention_error_rad", 0.005, 0.2),
            ("intervention_release_rad", 0.001, 0.1),
            ("intervention_gripper_error_m", 0.001, 0.02),
            ("intervention_gripper_release_m", 0.0001, 0.01),
            ("intervention_confirm_s", 0.01, 0.5),
            ("intervention_release_s", 0.1, 2.0),
            ("replan_distance_rad", 0.01, 0.3),
            ("policy_max_age_s", 0.1, 5.0),
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite in [{low}, {high}]")
        if (
            self.intervention_release_rad >= self.intervention_error_rad
            or self.intervention_gripper_release_m >= self.intervention_gripper_error_m
        ):
            raise ValueError(
                "intervention release thresholds must be below entry thresholds"
            )
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
        if type(self.simulate) is not bool:
            raise ValueError("simulate must be a boolean")
        if type(self.trace_queue_size) is not int or self.trace_queue_size < 100:
            raise ValueError("trace_queue_size must be at least 100")
