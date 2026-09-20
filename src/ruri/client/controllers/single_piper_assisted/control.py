"""Deterministic control authority and references, independent of sockets/SDK.

Policy sets a shared virtual leader reference, sent directly to both arms.
During correction the physical leader supplies the shared reference; policy
inference continues as a preview. Release starts a fresh policy timeline.
Explicit manual HUMAN mode remains available with an anchored relative reference.
There are deliberately no episode, dataset, model or reward concepts here.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np

from ruri.client.controllers.single_piper.calibration_ranges import (
    NOMINAL,
)
from ruri.client.controllers.single_piper.normalization import normalize_pose

from .config import SinglePiperAssistedConfig

PROTOCOL = "ruri_assisted_control_v3"


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
    # Same policy/teleop action box on both arms. Each physical arm applies its
    # own measured hardware clamp afterwards at the SDK boundary.
    return physical_bounds(NOMINAL)


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
        self.leader_gain = 0.0
        self.rejected = 0
        self.reference_revision = 0
        self._revision_reference = self.target.copy()
        self.policy_reference = self.target.copy()
        self.policy_observation_ns = None
        self.policy_valid = False
        self.intervention_id = 0
        self.correcting = False
        self._deviation_since = self._agreement_since = None
        self.policy_error = np.zeros(7)
        self.coupling_error = np.zeros(7)
        self._previous_leader = (self.leader - self.offset) * self.sign
        self.source = "hold"

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
        # Preserve the final follower reference. Capture leader's actual pose
        # when leaving gravity-only operation; ramp stiffness separately.
        if previous_mode == "HUMAN" or mode == "HOLD":
            self.leader_target = clamp_pose(self.leader)
        self.correcting = self.policy_valid = False
        self._deviation_since = self._agreement_since = None
        self.source = "human" if mode == "HUMAN" else "hold"
        self.policy_reference = self.target.copy()
        self._previous_leader = (self.leader - self.offset) * self.sign
        self.policy_observation_ns = None
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
            target = clamp_pose(packet.get("target"))
            if model_action:
                self.policy_reference = target
                self.policy_observation_ns = observation_ns
                self.policy_valid = True
                if not self.correcting:
                    self.target = target.copy()
                    self.leader_target = self.mapped(target)
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

    def _rebase_policy(self, leader):
        """Revoke old predictions, not execution authority or leader/follower coupling."""
        self.reference_revision += 1
        self.policy_reference = np.clip(leader, self.lower, self.upper)
        self.target = self.policy_reference.copy()
        self._revision_reference = self.target.copy()
        self.leader_target = self.mapped(self.target)
        self.policy_valid = False
        self.policy_observation_ns = None

    def _coupled_tick(self, now, dt):
        c = self.config
        leader = (self.leader - self.offset) * self.sign
        delta = leader - self._previous_leader
        self._previous_leader = leader.copy()
        self.coupling_error = leader - self.follower
        self.policy_error = leader - self.policy_reference
        if (
            self.policy_valid
            and now - self.policy_observation_ns / 1e9 > c.policy_max_age_s
        ):
            self._rebase_policy(self.target)
        entry = np.r_[
            np.full(6, c.intervention_error_rad), c.intervention_gripper_error_m
        ]
        release = np.r_[
            np.full(6, c.intervention_release_rad), c.intervention_gripper_release_m
        ]
        shared_error = leader - self.target
        # A passive leader travelling toward the shared target does not move
        # that target. This prevents normal mechanical lag becoming a cascade.
        toward = np.sign(shared_error) * delta < -np.r_[np.full(6, 0.005), 0.001] * dt
        separated = np.abs(self.coupling_error) > release
        displaced = np.abs(shared_error) > entry
        override = bool(np.any(displaced & separated & ~toward))
        follower_healthy = (
            np.max(np.abs(self.follower[:6] - self.target[:6])) < c.max_track_error
            and abs(self.follower[6] - self.target[6]) < 0.015
        )
        if not self.correcting:
            if self.policy_valid and override and follower_healthy:
                if self._deviation_since is None:
                    self._deviation_since = now
                if now - self._deviation_since >= c.intervention_confirm_s:
                    self.correcting = True
                    self.intervention_id += 1
                    self.leader_gain = 0.0
                    self._agreement_since = None
                    self._rebase_policy(self.target)
            else:
                self._deviation_since = None
        if self.correcting:
            speed = np.r_[
                np.full(6, c.human_reference_speed), c.gripper_reference_speed
            ]
            corrected = self.target + np.clip(
                np.clip(leader, self.lower, self.upper) - self.target,
                -speed * dt,
                speed * dt,
            )
            self.target = corrected
            # Encoder jitter must not invalidate every background inference.
            # While human-led there is no executed RTC prefix to preserve;
            # rebase once motion leaves the release tolerance around its origin.
            if np.any(np.abs(corrected - self._revision_reference) > release):
                self._rebase_policy(corrected)
            # Position feedback cannot identify a hand. Quiet leader motion is
            # the explicit release proxy; never require the next policy action
            # to equal the human pose (a moving policy would never release).
            quiet = bool(
                np.all(
                    np.abs(delta)
                    <= np.r_[
                        np.full(6, c.intervention_release_speed_rad_s),
                        c.intervention_gripper_release_speed_m_s,
                    ]
                    * dt
                )
            )
            if quiet:
                if self._agreement_since is None:
                    self._agreement_since = now
                if (
                    now - self._agreement_since >= c.intervention_release_s
                    and self.policy_valid
                    and self.policy_observation_ns / 1e9 >= self._agreement_since
                    and np.all(
                        np.abs(np.clip(leader, self.lower, self.upper) - self.target)
                        < release
                    )
                    and follower_healthy
                ):
                    self.correcting = False
                    self._deviation_since = self._agreement_since = None
                    # Predictions during manual motion were previews, never an
                    # executed RTC prefix. Bootstrap again at the release pose.
                    self._rebase_policy(self.target)
            else:
                self._agreement_since = None
        self.source = (
            (
                "transition"
                if self._agreement_since is not None and self.policy_valid
                else "human"
            )
            if self.correcting
            else (
                "transition"
                if self._deviation_since is not None
                else "policy"
                if self.policy_valid
                else "hold"
            )
        )
        if not self.policy_valid:
            self.policy_reference = self.target.copy()
        # The virtual leader is the one shared reference. The physical leader
        # is a parallel actuator/input device, never in the policy -> follower path.
        self.leader_target = self.mapped(self.target)

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
        if self.mode == "POLICY":
            self._coupled_tick(now, dt)
        elif self.mode == "HUMAN":
            desired = self.anchor_target + self.sign * (
                self.leader - self.anchor_leader
            )
            desired = np.clip(desired, self.lower, self.upper)
            step = (
                np.r_[
                    np.full(6, self.config.human_reference_speed),
                    self.config.gripper_reference_speed,
                ]
                * dt
            )
            self.target += np.clip(desired - self.target, -step, step)
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
        # A takeover removes position stiffness immediately. When leaving HUMAN
        # stiffness rises from zero around the captured pose, not an old target.
        if self.mode == "HUMAN" or self.correcting:
            self.leader_gain = 0.0
        else:
            self.leader_gain = min(
                1.0, self.leader_gain + dt / self.config.gain_transition_s
            )
        return self.target.copy(), self.leader_target.copy()

    def status(self):
        return {
            "worker_session": self.session,
            "epoch": self.epoch,
            "mode": self.mode,
            "source": self.source,
            "control_schema": "ruri_virtual_leader_rtc_v1",
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
            "execution_owner": "leader"
            if self.correcting or self.mode == "HUMAN"
            else "policy"
            if self.mode == "POLICY" and self.policy_valid
            else "hold",
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
