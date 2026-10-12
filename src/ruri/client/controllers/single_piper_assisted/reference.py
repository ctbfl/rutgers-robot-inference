"""Six-joint shared reference; pure numerical state, no SDK or transport.

Policy proposals and estimated human interaction influence reference velocity.
Neither changes the MIT gains. Torque-like quantities use the pinned SDK's
legacy units; spring displacement is an interaction proxy, not a force sensor.
"""

from __future__ import annotations

import math
import numpy as np


def vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain six finite values")
    return result.copy()


class SharedReference:
    def __init__(self, config, initial, lower, upper):
        self.config = config
        self.lower, self.upper = vector(lower, "lower"), vector(upper, "upper")
        if np.any(self.lower >= self.upper):
            raise ValueError("shared reference bounds must have positive width")
        self.r = np.clip(vector(initial, "initial"), self.lower, self.upper)
        self.velocity = np.zeros(6)
        self.alpha = self.alpha_target = 0.0
        self.interaction = np.zeros(6)
        self.effective = np.zeros(6)
        self.human_velocity = np.zeros(6)
        self.policy_velocity = np.zeros(6)
        self.human_latched = False
        self.confirm_elapsed = self.rise_elapsed = self.quiet_elapsed = 0.0
        self.filtered_error = np.zeros(6)
        self.trigger_active = False
        self.phase = "policy"
        self.intervening = False
        self._rise_start = 0.0
        self._quiet_weight = 0.0

    def hold(self, reference=None):
        if reference is not None:
            self.r = np.clip(vector(reference, "hold"), self.lower, self.upper)
        self.velocity[:] = 0
        self.alpha = self.alpha_target = 0.0
        self.interaction[:] = self.effective[:] = 0
        self.human_velocity[:] = self.policy_velocity[:] = 0
        self.human_latched = False
        self.confirm_elapsed = self.rise_elapsed = self.quiet_elapsed = 0.0
        self.filtered_error[:] = 0
        self.trigger_active = False
        self.phase = "policy"
        self.intervening = False
        self._rise_start = 0.0
        self._quiet_weight = 0.0

    def _authority(self, dt, manual):
        """Confirm filtered displacement before a separate, evidence-gated rise.

        Motion alone never confirms or sustains authority. The admittance itself
        retains its smaller deadzone and unfiltered signal for responsive tracking.
        """
        c = self.config
        previous_phase = self.phase
        error = self.interaction / c.leader_policy_kp
        self.filtered_error += -math.expm1(-dt/c.interaction_filter_s)*(error-self.filtered_error)
        self.trigger_active = bool(np.max(np.abs(self.filtered_error)) > math.radians(c.human_trigger_angle_deg))
        filtered_effective = np.maximum(
            np.abs(c.leader_policy_kp*self.filtered_error)-np.asarray(c.interaction_deadzone), 0.
        )
        activity = np.linalg.norm(filtered_effective) > c.interaction_release

        if manual:
            self.intervening = True
            self.confirm_elapsed = c.human_confirm_s
            self.rise_elapsed = c.human_takeover_s
            self.quiet_elapsed = 0.0
            self.alpha = 1.0
            self.phase = "human"
        else:
            started = False
            if not self.intervening or self.phase == "recover":
                self.confirm_elapsed = min(c.human_confirm_s, self.confirm_elapsed+dt) if self.trigger_active else 0.
                if self.confirm_elapsed >= c.human_confirm_s-1e-12:
                    self.intervening = True
                    self.phase = "blend"
                    self._rise_start = self.alpha
                    self.rise_elapsed = self.quiet_elapsed = 0.0
                    started = True

            if self.intervening and not started:
                if activity and self.phase != "recover":
                    self.quiet_elapsed = 0.0
                    self.rise_elapsed = min(c.human_takeover_s, self.rise_elapsed+dt)
                    progress = self.rise_elapsed/c.human_takeover_s
                    self.alpha = self._rise_start+(1.-self._rise_start)*progress
                    if self.rise_elapsed >= c.human_takeover_s-1e-12:
                        self.alpha = 1.0
                        self.phase = "human"
                else:
                    if self.quiet_elapsed == 0:
                        self._quiet_weight = self.alpha
                    self.quiet_elapsed += dt
                    if self.quiet_elapsed >= c.human_release_s-1e-12:
                        self.intervening = False
                        self.alpha = 0.0
                        self.phase = "policy"
                        self.rise_elapsed = self.quiet_elapsed = 0.0
                    elif self.quiet_elapsed > c.human_release_delay_s+1e-12:
                        if self.phase != "recover":
                            self.confirm_elapsed = 0.0
                        self.phase = "recover"
                        self.alpha = self._quiet_weight * float(np.clip(
                            (c.human_release_s-self.quiet_elapsed)
                            / (c.human_release_s-c.human_release_delay_s), 0., 1.
                        ))
            elif not self.intervening:
                self.alpha = 0.0
                self.phase = "policy"
        self.human_latched = self.phase == "human"
        self.alpha_target = 0.0 if self.phase in ("policy", "recover") else self.alpha if self.quiet_elapsed else 1.0
        # Old HUMAN proposals cannot execute when policy first gets weight back.
        return previous_phase in ("blend", "human") and self.phase == "recover"

    def step(self, leader, policy, dt, *, manual=False):
        c = self.config
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("reference dt must be finite and nonnegative")
        q = vector(leader, "leader")
        target = None if policy is None else vector(policy, "policy")
        # Estimate against the reference already applied during the last cycle.
        self.interaction = c.leader_policy_kp * (q - self.r) - np.asarray(c.interaction_bias)
        self.effective = np.sign(self.interaction) * np.maximum(
            np.abs(self.interaction) - np.asarray(c.interaction_deadzone), 0.0
        )
        released = self._authority(dt, manual)
        self.human_velocity = self.effective / np.asarray(c.admittance_damping)
        # Do not execute a proposal retained during HUMAN on the release tick.
        # The worker invalidates its revision before accepting a fresh query.
        self.policy_velocity = np.zeros(6) if target is None or released else np.clip(
            c.policy_reference_gain * (target-self.r),
            -c.policy_reference_speed, c.policy_reference_speed
        )
        requested = (1-self.alpha)*self.policy_velocity + self.alpha*self.human_velocity
        velocity = np.clip(requested, -c.reference_speed, c.reference_speed)
        if dt > 0:
            next_r = np.clip(self.r + velocity*dt, self.lower, self.upper)
            self.velocity = (next_r-self.r)/dt
            self.r = next_r
        return self.r.copy(), self.velocity.copy()
