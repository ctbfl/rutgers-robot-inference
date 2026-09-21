"""Worker admission/lifecycle with continuous shared-reference control.

Policy proposals and human spring displacement update one reference. Both arms
retain fixed impedance gains. No episode, dataset, model or reward logic.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np

from ruri.client.controllers.single_piper.calibration_ranges import (
    NOMINAL, measured_for_arm,
)
from ruri.client.controllers.single_piper.normalization import normalize_pose

from .config import SinglePiperAssistedConfig
from .reference import SharedReference

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
    if np.any(low >= high):
        raise ValueError("leader/follower have no common reference range")
    return low, high


def clamp_pose(values):
    low, high = physical_bounds(NOMINAL)
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
        self.target = clamp_pose(self.follower)
        self.leader_target = clamp_pose(self.leader)
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
        self.reference = SharedReference(config, self.target[:6], self.lower[:6], self.upper[:6])
        self.target[:6] = self.reference.r
        self.reference_velocity = np.zeros(6)

    def mapped(self, follower):
        return clamp_pose(self.sign * follower + self.offset)

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
            self.leader_target = clamp_pose(self.leader)
            if self.config.leader_only:
                self.target = np.clip(self.leader_target, self.lower, self.upper)
                self.leader_target = self.target.copy()
        self.correcting = self.policy_valid = False
        self.source = "human" if mode == "HUMAN" else "hold"
        self.policy_reference = self.target.copy()
        self.policy_observation_ns = None
        self.reference.hold(self.target[:6])
        self.reference_velocity[:] = 0
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
            target = np.clip(clamp_pose(packet.get("target")), self.lower, self.upper)
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
        self.target[:6], self.reference_velocity = self.reference.step(
            leader[:6], candidate, dt, manual=self.mode == "HUMAN"
        )
        if self.mode == "POLICY":
            self.correcting = self.reference.intervening
            if self.correcting and not was_intervening:
                self.intervention_id += 1
            if self.reference.phase != previous_phase:
                self._rebase_policy()
        # Grippers have a separate position/force channel, outside the torque norm.
        if self.mode == "HUMAN" or self.correcting:
            self.target[6] = np.clip(leader[6], self.lower[6], self.upper[6])
        elif self.policy_valid:
            self.target[6] += float(np.clip(
                self.policy_reference[6]-self.target[6],
                -c.gripper_reference_speed*dt, c.gripper_reference_speed*dt
            ))
        self.leader_target = self.mapped(self.target)
        self.source = (
            "human" if self.mode == "HUMAN" or self.reference.human_latched else "shared" if self.correcting
            else "policy" if self.policy_valid else "hold"
        )

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
        self.leader_gain = 1.0  # Diagnostic compatibility; never authority-dependent.
        return self.target.copy(), self.leader_target.copy()

    def status(self):
        return {
            "worker_session": self.session,
            "epoch": self.epoch,
            "mode": self.mode,
            "source": self.source,
            "control_schema": "ruri_shared_reference_v1",
            "leader_only": self.config.leader_only,
            "controlled_arm": "right_arm" if self.config.leader_only else "left_arm",
            "follower_present": not self.config.leader_only,
            "virtual_reference": normalize_pose(
                self.target[:6], self.target[6]
            ).tolist(),
            "reference_revision": self.reference_revision,
            "policy_reference": normalize_pose(
                self.policy_reference[:6], self.policy_reference[6]
            ).tolist(),
            "policy_valid": self.policy_valid,
            "policy_observation_ns": self.policy_observation_ns,
            "intervention_active": self.correcting or self.mode == "HUMAN",
            "execution_owner": "shared_reference" if self.mode in ("POLICY", "HUMAN") else "hold",
            "human_weight": self.reference.alpha,
            "human_weight_target": self.reference.alpha_target,
            "authority_phase": self.reference.phase,
            "human_exclusive": self.reference.human_latched,
            "intervention_confirm_s": self.reference.confirm_elapsed,
            "human_rise_s": self.reference.rise_elapsed,
            "intervention_trigger_active": self.reference.trigger_active,
            "filtered_interaction_error_rad": self.reference.filtered_error.tolist(),
            "human_quiet_s": self.reference.quiet_elapsed,
            "applied_policy_velocity": ((1-self.reference.alpha)*self.reference.policy_velocity).tolist(),
            "interaction_estimate": self.reference.interaction.tolist(),
            "interaction_effective": self.reference.effective.tolist(),
            "reference_q": self.reference.r.tolist(),
            "reference_qd": self.reference_velocity.tolist(),
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
            "action_lower": normalize_pose(self.lower[:6], self.lower[6]).tolist(),
            "action_upper": normalize_pose(self.upper[:6], self.upper[6]).tolist(),
        }
