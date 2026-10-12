"""Worker admission/lifecycle for dual shared reference or right gravity handoff.

No episode, dataset, model or reward logic. The worker owns authority, actuator
references and freshness; the scheduler only supplies policy proposals.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np

from ruri.client.controllers.single_piper.calibration_ranges import (
    NOMINAL, measured_for_arm,
)
from ruri.client.controllers.single_piper.normalization import GRIPPER_NOMINAL_MAX_M, normalize_pose

from .config import SinglePiperAssistedConfig
from .reference import SharedReference
from .gravity_handoff import GravityHandoff
from .follower_handoff import FollowerHandoff
from .gripper_handoff import GripperHandoff

PROTOCOL = "ruri_assisted_control_v4"


def pose(values) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError(
            "pose must contain six finite radians and gripper width in metres"
        )
    return value.copy()


def physical_bounds(ranges):
    joints = [ranges[f"joint{i}"] for i in range(1, 7)]
    bounds = np.concatenate(
        (np.radians(np.asarray(joints) / 1000.0), np.asarray([ranges["gripper"]]) / 1e6)
    )
    return bounds[:, 0], bounds[:, 1]


def synchronized_bounds(config, leader_name="right_arm", follower_name="left_arm"):
    low, high = physical_bounds(NOMINAL)
    flo, fhi = (low, high) if config.leader_only else physical_bounds(measured_for_arm(follower_name))
    llo, lhi = physical_bounds(measured_for_arm(leader_name))
    llo, lhi = np.maximum(llo, low), np.minimum(lhi, high)
    sign, offset = np.r_[config.leader_sign, 1.], np.r_[config.leader_offset_rad, 0.]
    a, b = (llo-offset)*sign, (lhi-offset)*sign
    low = np.maximum.reduce((low, flo, np.minimum(a, b)))
    high = np.minimum.reduce((high, fhi, np.maximum(a, b)))
    if config.gripper_max_m is not None:
        # Optional gripper range override; joint envelopes stay unchanged.
        low[6], high[6] = 0., config.gripper_max_m
    if np.any(low >= high):
        raise ValueError("leader/follower have no common reference range")
    return low, high


def clamp_pose(values, gripper_max_m=None):
    low, high = physical_bounds(NOMINAL)
    if gripper_max_m is not None:
        low[6], high[6] = 0., gripper_max_m
    return np.clip(pose(values), low, high)


class AssistedControl:
    """The MIT worker is the only caller/mutator of this state machine."""

    def __init__(
        self,
        config: SinglePiperAssistedConfig,
        leader,
        follower,
        *,
        worker_session: str | None = None,
        clock=time.monotonic,
        leader_name="right_arm",
        follower_name="left_arm",
    ):
        self.config, self.clock = config, clock
        self.session = worker_session or uuid.uuid4().hex
        self.sign = np.r_[config.leader_sign, 1.0]
        self.offset = np.r_[config.leader_offset_rad, 0.0]
        self.lower, self.upper = synchronized_bounds(config, leader_name, follower_name)
        self.leader = pose(leader)
        self.follower = pose(follower)
        self.target = self.clamp(self.follower)
        self.leader_target = self.clamp(self.leader)
        self.anchor_leader = self.leader.copy()
        self.anchor_target = self.target.copy()
        self.mode = "HOLD"
        self.epoch = 0
        self.armed = self.aligning = False
        self.reason = "startup hold"
        self.last_request: dict[str, Any] | None = None
        self.last_sequence = -1
        self.command_id: int | None = None
        self.client_session: str | None = None
        self.last_heartbeat = self.last_command = self.last_tick = clock()
        self.leader_gain = 1.0
        self.rejected = 0
        self.reference_revision = 0
        self.policy_reference = self.target.copy()
        self.policy_observation_ns = None
        self.policy_valid = False
        self.intervention_id = 0
        self.correcting = False
        self.policy_error = np.zeros(7)
        self.coupling_error = np.zeros(7)
        self.source = "hold"
        self.follower_control_source = "hold"
        self.follower_gripper_source = "hold"
        reference_type = GravityHandoff if config.uses_gravity_handoff else SharedReference
        self.reference = reference_type(config, self.target[:6], self.lower[:6], self.upper[:6])
        self.target[:6] = self.reference.r
        self.reference_velocity = np.zeros(6)
        self.follower_velocity = np.zeros(6)
        self.follower_handoff = (FollowerHandoff(config, self.target[:6], self.lower[:6], self.upper[:6])
                                 if config.uses_gravity_handoff and not config.leader_only else None)
        leader_open = min(physical_bounds(NOMINAL)[1][6],
                          physical_bounds(measured_for_arm(leader_name))[1][6])
        if config.gripper_max_m is not None:
            leader_open = config.gripper_max_m
        self.gripper_handoff = (GripperHandoff(config, leader_open, self.lower[6], self.upper[6])
                                if config.human_trigger == "gripper" else None)

    def mapped(self, follower):
        return self.clamp(self.sign * follower + self.offset)

    def clamp(self, values):
        return clamp_pose(values, self.config.gripper_max_m)

    def normalize(self, q, width):
        scale = GRIPPER_NOMINAL_MAX_M/(self.config.gripper_max_m or GRIPPER_NOMINAL_MAX_M)
        return normalize_pose(q, width*scale)

    @property
    def aligned(self):
        gap = np.abs(self.leader - self.mapped(self.target))
        reference_gap = np.max(np.abs(self.leader_target - self.mapped(self.target)))
        return bool(
            np.max(gap[:6]) <= self.config.alignment_tolerance
            and gap[6] <= 0.003
            and reference_gap < 1e-6
        )

    def _revoke(self, mode: str, reason: str):
        previous_mode = self.mode
        self.epoch += 1
        self.mode, self.reason = mode, reason
        self.armed = self.aligning = False
        self.command_id = None
        # Lifecycle HOLD captures leader position; active sharing uses one r.
        if previous_mode == "HUMAN" or mode == "HOLD":
            self.leader_target = self.clamp(self.leader)
            if self.config.leader_only:
                self.target = np.clip(self.leader_target, self.lower, self.upper)
                self.leader_target = self.target.copy()
        self.correcting = self.policy_valid = False
        self.source = "human" if mode == "HUMAN" else "hold"
        self.follower_control_source = self.follower_gripper_source = "hold"
        self.policy_reference = self.target.copy()
        self.policy_observation_ns = None
        self.reference.hold(self.target[:6])
        if self.gripper_handoff is not None:
            self.gripper_handoff.reset()
        self.reference_velocity[:] = 0
        self.follower_velocity[:] = 0
        if self.follower_handoff is not None:
            self.follower_handoff.sync(self.target[:6], self.follower_velocity)
        if mode == "HUMAN":
            self.intervention_id += 1
            self.anchor_leader = self.leader.copy()
            self.anchor_target = self.target.copy()

    def fault(self, reason: str):
        if self.mode != "FAULT":
            self._revoke("FAULT", reason)

    def receive(self, packet: dict, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        try:
            if (
                packet.get("protocol") != PROTOCOL
                or packet.get("worker_session") != self.session
            ):
                raise ValueError("wrong protocol or worker session")
            sequence = packet.get("sequence")
            if type(sequence) is not int or sequence <= self.last_sequence:
                raise ValueError("replayed or out of order sequence")
            stamp = packet.get("monotonic_ns")
            if type(stamp) is not int or not -0.01 <= now - stamp / 1e9 <= 0.3:
                raise ValueError("stale command timestamp")
            client = packet.get("client_session")
            if not isinstance(client, str) or not client or len(client) > 128:
                raise ValueError("invalid client session")
            if self.client_session is not None and client != self.client_session:
                raise ValueError("worker already belongs to another client session")
            self.client_session = client
            self.last_sequence = sequence
            kind = packet.get("kind")
            if kind == "heartbeat":
                self.last_heartbeat = now
                return True
            if kind == "mode":
                request_id = packet.get("request_id")
                if (
                    not isinstance(request_id, str)
                    or not request_id
                    or len(request_id) > 128
                ):
                    raise ValueError("invalid request ID")
                if self.last_request and request_id == self.last_request["request_id"]:
                    return self.last_request["accepted"]
                accepted, reason = self._mode(packet.get("mode"), packet.get("epoch"))
                self.last_request = {
                    "request_id": request_id,
                    "accepted": accepted,
                    "reason": reason,
                    "monotonic_ns": int(now * 1e9),
                }
                if not accepted:
                    self.rejected += 1
                return accepted
            if kind != "target":
                raise ValueError("unknown command kind")
            if packet.get("epoch") != self.epoch or not (
                self.mode == "POLICY" or self.armed
            ):
                raise ValueError("policy lacks current control authority")
            if packet.get("reference_revision", 0) != self.reference_revision:
                raise ValueError("policy reference predates a physical correction")
            model_action = packet.get("model_action", True)
            if type(model_action) is not bool:
                raise ValueError("model_action must be a boolean")
            observation_ns = packet.get("observation_ns", stamp)
            if model_action and (
                type(observation_ns) is not int
                or not -0.01
                <= now - observation_ns / 1e9
                <= self.config.policy_max_age_s
            ):
                raise ValueError("policy observation is stale")
            target = np.clip(self.clamp(packet.get("target")), self.lower, self.upper)
            if model_action:
                self.policy_reference = target
                self.policy_observation_ns = observation_ns
                self.policy_valid = True
            self.mode, self.armed, self.reason = "POLICY", False, "policy command"
            self.command_id = sequence
            self.last_command = now
            return True
        except (ValueError, TypeError, OverflowError):
            self.rejected += 1
            return False

    def _mode(self, mode, epoch):
        if self.mode == "FAULT":
            return False, "FAULT requires worker restart and hardware inspection"
        if type(epoch) is not int or epoch != self.epoch:
            return False, "stale control epoch"
        if mode in ("HOLD", "HUMAN"):
            self._revoke(mode, "operator request")
            return True, mode
        if mode == "ALIGN" and self.mode == "HOLD":
            self._revoke("HOLD", "align leader while follower holds")
            self.aligning = True
            return True, "ALIGN"
        if mode == "POLICY" and self.mode == "HOLD":
            # Both arms receive the shared command directly. Measured pose or
            # gripper agreement must never gate policy startup/resume.
            self._revoke("HOLD", "waiting for fresh policy target")
            self.armed = True
            self.last_command = self.clock()
            return True, "armed"
        return False, "POLICY and ALIGN require HOLD"

    def _rebase_policy(self):
        """Invalidate predictions without moving r or changing actuator gains."""
        self.reference_revision += 1
        self.policy_reference = self.target.copy()
        self.policy_valid = False
        self.policy_observation_ns = None

    def _coupled_tick(self, now, dt):
        c = self.config
        leader = (self.leader-self.offset)*self.sign
        self.coupling_error = leader-self.follower
        self.policy_error = leader-self.policy_reference
        if self.policy_valid and now-self.policy_observation_ns/1e9 > c.policy_max_age_s:
            self._rebase_policy()
        candidate = self.policy_reference[:6] if self.policy_valid else None
        previous_phase = self.reference.phase
        was_intervening = self.reference.intervening
        handoff = self.follower_handoff
        recovery = {} if handoff is None else {
            "recovery_ready": handoff.ready(leader[:6]),
            "recovery_reference": self.target[:6].copy(),
        }
        gripper = self.gripper_handoff
        aligning_gripper = False
        if gripper is not None:
            if self.mode == "HUMAN":
                gripper.phase = "teleop"
            else:
                triggered, aligned = gripper.observe(leader[6], self.follower[6], dt)
                recovery.update(gripper_triggered=triggered, gripper_aligned=aligned)
            aligning_gripper = gripper.phase == "aligning"
        reference, self.reference_velocity = self.reference.step(
            leader[:6], candidate, dt, manual=self.mode == "HUMAN", **recovery
        )
        if handoff is not None and self.reference.phase == "human":
            self.target[:6], self.follower_velocity = handoff.follow(leader[:6], dt)
        else:
            self.target[:6] = reference
            self.follower_velocity = self.reference_velocity.copy()
            if handoff is not None:
                handoff.sync(reference, self.reference_velocity)
        if self.mode == "POLICY":
            self.correcting = self.reference.intervening
            if self.correcting and not was_intervening:
                self.intervention_id += 1
            if self.reference.phase != previous_phase:
                # Recovery already starts from current q and a fresh revision.
                # Finishing its gain ramp must not interrupt that fresh motion.
                finishing_handoff = (c.uses_gravity_handoff and previous_phase == "recover"
                                     and self.reference.phase == "policy")
                if not finishing_handoff:
                    self._rebase_policy()
        # Grippers have a separate position/force channel, outside the torque norm.
        if gripper is not None and aligning_gripper:
            self.target[6] = gripper.follower_hold
            self.follower_gripper_source = "hold"
        elif self.mode == "HUMAN" or self.correcting:
            self.target[6] = np.clip(leader[6], self.lower[6], self.upper[6])
            self.follower_gripper_source = "human"
        elif self.policy_valid:
            self.target[6] += float(np.clip(
                self.policy_reference[6]-self.target[6],
                -c.gripper_reference_speed*dt, c.gripper_reference_speed*dt
            ))
            self.follower_gripper_source = "policy"
        else:
            self.follower_gripper_source = "hold"
        self.leader_target = self.mapped(self.target)
        if handoff is not None and self.reference.phase == "human":
            self.leader_target[:6] = self.mapped(np.r_[reference, self.target[6]])[:6]
        if gripper is not None:
            if self.reference.phase == "human":
                gripper.phase = "teleop"
            elif previous_phase == "recover" and self.reference.phase == "policy":
                gripper.reset()
            if gripper.phase in ("opening", "armed", "aligning"):
                self.leader_target[6] = gripper.leader_target
        self.source = (
            "human" if self.mode == "HUMAN" or self.reference.human_latched else "shared" if self.correcting
            else "policy" if self.policy_valid else "hold"
        )
        # Attribute the target sent to the task arm, independently of leader
        # stiffness. Interpolation towards live leader readings is human-owned;
        # recovery follows policy targets even while leader alpha is still > 0.
        if c.uses_gravity_handoff:
            phase = self.reference.phase
            if phase == "human":
                if handoff is None or handoff.progress > 0.:
                    self.follower_control_source = "human"
                # At progress zero, follow() preserves the previous command.
            elif phase == "recover" and previous_phase != "recover":
                self.follower_control_source = "hold"  # Rebase tick; no new policy motion yet.
            elif phase not in ("gripper_align", "blend"):
                self.follower_control_source = "policy" if candidate is not None else "hold"
            # Braking/alignment retain the origin of the preceding reference.
        else:
            self.follower_control_source = self.source

    def tick(self, leader, follower, now: float | None = None):
        now = self.clock() if now is None else now
        dt = max(0.0, min(now - self.last_tick, 2.0 / self.config.mit_rate_hz))
        self.last_tick = now
        self.leader, self.follower = pose(leader), pose(follower)
        if self.mode != "FAULT":
            if now - self.last_heartbeat > self.config.heartbeat_timeout_s and (
                self.mode != "HOLD" or self.armed or self.aligning
            ):
                self._revoke("HOLD", "client heartbeat timeout")
            elif (
                self.mode == "POLICY" or self.armed
            ) and now - self.last_command > self.config.command_timeout_s:
                self._revoke("HOLD", "policy command timeout")
        if self.mode in ("POLICY", "HUMAN"):
            self._coupled_tick(now, dt)
        elif self.aligning:
            step = (
                np.r_[
                    np.full(6, self.config.alignment_speed),
                    self.config.gripper_reference_speed,
                ]
                * dt
            )
            self.leader_target += np.clip(
                self.mapped(self.target) - self.leader_target, -step, step
            )
        if self.mode not in ("POLICY", "HUMAN"):
            self.reference_velocity[:] = 0
            self.follower_velocity[:] = 0
        self.leader_gain = 1.-self.reference.alpha if self.config.uses_gravity_handoff else 1.
        return self.target.copy(), self.leader_target.copy()

    def status(self):
        return {
            "worker_session": self.session,
            "epoch": self.epoch,
            "mode": self.mode,
            "source": self.source,
            "follower_control_source": (self.follower_control_source
                if self.mode in ("POLICY", "HUMAN") else "hold"),
            "follower_gripper_source": (self.follower_gripper_source
                if self.mode in ("POLICY", "HUMAN") else "hold"),
            "follower_transition": (self.mode in ("POLICY", "HUMAN") and (
                self.reference.phase in ("gripper_align", "blend", "recover") or
                (self.reference.phase == "human" and self.follower_handoff is not None
                 and self.follower_handoff.progress < 1.))),
            "control_schema": "ruri_gravity_handoff_v1" if self.config.uses_gravity_handoff else "ruri_shared_reference_v1",
            "leader_only": self.config.leader_only,
            "controlled_arm": "right_arm" if self.config.leader_only else "left_arm",
            "follower_present": not self.config.leader_only,
            "follower_gravity_compensation": not self.config.leader_only,
            "virtual_reference": self.normalize(
                self.target[:6], self.target[6]
            ).tolist(),
            "reference_revision": self.reference_revision,
            "policy_reference": self.normalize(
                self.policy_reference[:6], self.policy_reference[6]
            ).tolist(),
            "policy_valid": self.policy_valid,
            "policy_observation_ns": self.policy_observation_ns,
            "intervention_active": self.correcting or self.mode == "HUMAN",
            "execution_owner": "shared_reference" if self.mode in ("POLICY", "HUMAN") else "hold",
            "human_weight": self.reference.alpha,
            "human_trigger": self.config.human_trigger,
            "gripper_handoff_phase": (self.gripper_handoff.phase
                if self.gripper_handoff is not None and self.mode in ("POLICY", "HUMAN") else "inactive"),
            "gripper_closure_m": self.gripper_handoff.closure if self.gripper_handoff else None,
            "leader_gripper_target_m": float(self.leader_target[6]),
            "gripper_trigger_width_m": (self.gripper_handoff.open_width-self.config.gripper_trigger_m
                                         if self.gripper_handoff else None),
            "human_weight_target": self.reference.alpha_target,
            "authority_phase": self.reference.phase,
            "human_exclusive": self.reference.human_latched,
            "intervention_confirm_s": self.reference.confirm_elapsed,
            "human_rise_s": self.reference.rise_elapsed,
            "intervention_trigger_active": self.reference.trigger_active,
            "filtered_interaction_error_rad": self.reference.filtered_error.tolist(),
            "human_quiet_s": self.reference.quiet_elapsed,
            "human_motion_range_deg": (np.degrees(self.reference.motion_range).tolist()
                                       if self.config.uses_gravity_handoff else None),
            "leader_impedance_scale": 1.-self.reference.alpha if self.config.uses_gravity_handoff else 1.,
            "reference_active": not (self.config.uses_gravity_handoff and self.reference.alpha == 1.),
            "applied_policy_velocity": (self.reference.velocity
                if self.config.uses_gravity_handoff and self.reference.phase in ("policy", "recover")
                else np.zeros(6) if self.config.uses_gravity_handoff
                else (1-self.reference.alpha)*self.reference.policy_velocity).tolist(),
            "interaction_estimate": self.reference.interaction.tolist(),
            "interaction_effective": self.reference.effective.tolist(),
            "reference_q": self.reference.r.tolist(),
            "reference_qd": self.reference_velocity.tolist(),
            "follower_reference_q": self.target[:6].tolist(),
            "follower_reference_qd": self.follower_velocity.tolist(),
            "follower_handoff_progress": self.follower_handoff.progress if self.follower_handoff else None,
            "follower_handoff_limited": self.follower_handoff.limited if self.follower_handoff else False,
            "policy_reference_velocity": self.reference.policy_velocity.tolist(),
            "human_reference_velocity": self.reference.human_velocity.tolist(),
            "intervention_id": self.intervention_id,
            "policy_error": self.policy_error.tolist(),
            "coupling_error": self.coupling_error.tolist(),
            "armed": self.armed,
            "aligning": self.aligning,
            "aligned": self.aligned,
            "reason": self.reason,
            "command_id": self.command_id,
            "last_request": self.last_request,
            "rejected": self.rejected,
            "action_lower": self.normalize(self.lower[:6], self.lower[6]).tolist(),
            "action_upper": self.normalize(self.upper[:6], self.upper[6]).tolist(),
        }
