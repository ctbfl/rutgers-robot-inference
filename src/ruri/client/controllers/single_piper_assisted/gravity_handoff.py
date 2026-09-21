"""Right-leader policy tracking with a smooth handoff to gravity-only control.

Alpha blends actuator impedance, not human/policy reference velocities. During
HUMAN, r only records the measured pose; it exerts no spring force. No SDK I/O.
"""

from collections import deque
import math

import numpy as np

from .reference import SharedReference, vector


def smoothstep(value):
    u = float(np.clip(value, 0., 1.))
    return u*u*(3.-2.*u)


class GravityHandoff(SharedReference):
    """Shares the reference/status interface with the dual-arm controller."""

    def __init__(self, config, initial, lower, upper):
        super().__init__(config, initial, lower, upper)
        self._reset_handoff()

    def _reset_handoff(self):
        self._time = self._human_since = self._recovery_elapsed = 0.
        self._motion = deque()
        self.motion_range = np.zeros(6)
        self._entry_velocity = np.zeros(6)
        self.rebased = False

    def hold(self, reference=None):
        super().hold(reference)
        self._reset_handoff()

    def _begin_takeover(self):
        self.phase = "blend"
        self.intervening = True
        self._rise_start = self.alpha
        self.rise_elapsed = self.quiet_elapsed = 0.
        self._entry_velocity = self.velocity.copy()
        self._motion.clear()

    def _observe_stillness(self, q):
        c = self.config
        self._motion.append((self._time, q.copy()))
        cutoff = self._time-c.human_release_delay_s
        while len(self._motion) > 1 and self._motion[1][0] <= cutoff:
            self._motion.popleft()
        self.motion_range = np.ptp(np.asarray([item[1] for item in self._motion]), axis=0)
        # Also report the duration of the longest quiet suffix, for the console.
        low, high = q.copy(), q.copy()
        earliest = self._time
        threshold = np.radians(c.human_quiet_range_deg)
        for stamp, pose in reversed(self._motion):
            low, high = np.minimum(low, pose), np.maximum(high, pose)
            if np.any(high-low > threshold):
                break
            earliest = stamp
        self.quiet_elapsed = min(c.human_release_delay_s, self._time-earliest)
        return (self._time-self._human_since >= c.human_release_delay_s-1e-12
                and self.quiet_elapsed >= c.human_release_delay_s-1e-12)

    def step(self, leader, policy, dt, *, manual=False,
             recovery_ready=True, recovery_reference=None):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("reference dt must be finite and nonnegative")
        c = self.config
        q = vector(leader, "leader")
        target = None if policy is None else vector(policy, "policy")
        self._time += dt
        self.rebased = False
        self.interaction = c.leader_policy_kp*(q-self.r)-np.asarray(c.interaction_bias)
        self.effective[:] = self.human_velocity[:] = 0.  # No admittance branch.
        self.trigger_active = False
        self.policy_velocity = np.zeros(6) if target is None else np.clip(
            c.policy_reference_gain*(target-self.r), -c.policy_reference_speed, c.policy_reference_speed
        )

        entered = False
        if self.phase in ("policy", "recover"):
            error = self.interaction/c.leader_policy_kp
            self.filtered_error += -math.expm1(-dt/c.interaction_filter_s)*(error-self.filtered_error)
            self.trigger_active = bool(np.max(abs(self.filtered_error)) > math.radians(c.human_trigger_angle_deg))
            self.confirm_elapsed = min(c.human_confirm_s, self.confirm_elapsed+dt) if self.trigger_active else 0.
            if manual or self.confirm_elapsed >= c.human_confirm_s-1e-12:
                self._begin_takeover()
                entered = True

        if self.phase == "blend":
            # Confirm once, then finish unloading. The disappearing spring
            # signal during unloading is not evidence that the person released.
            if not entered:
                self.rise_elapsed = min(c.human_takeover_s, self.rise_elapsed+dt)
            progress = smoothstep(self.rise_elapsed/c.human_takeover_s)
            self.alpha = self._rise_start+(1.-self._rise_start)*progress
            velocity = (1.-progress)*self._entry_velocity
            if self.rise_elapsed >= c.human_takeover_s-1e-12:
                self.phase, self.alpha = "human", 1.
                self._human_since = self._time
                self._motion.clear()
                self._motion.append((self._time, q.copy()))
                self.filtered_error[:] = 0.
                self.quiet_elapsed = 0.
        elif self.phase == "human":
            velocity = np.zeros(6)
            if self._observe_stillness(q) and not manual and recovery_ready:
                self.phase = "recover"
                self._recovery_elapsed = self.confirm_elapsed = 0.
                self.filtered_error[:] = 0.
                self.rebased = True
                # Rebase while stiffness is still exactly zero. The enclosing
                # worker invalidates the old revision before any fresh motion.
                # A dual follower resumes from its last commanded reference,
                # after catching the leader. This avoids a second command jump.
                start = q if recovery_reference is None else vector(recovery_reference, "recovery reference")
                self.r = np.clip(start, self.lower, self.upper)
                self.policy_velocity[:] = 0.
        elif self.phase == "recover":
            self._recovery_elapsed += dt
            duration = c.human_release_s-c.human_release_delay_s
            self.alpha = 1.-smoothstep(self._recovery_elapsed/duration)
            velocity = self.policy_velocity.copy()
            if self._recovery_elapsed >= duration-1e-12:
                self.phase, self.alpha, self.intervening = "policy", 0., False
                self.quiet_elapsed = self.rise_elapsed = 0.
        else:
            self.alpha = 0.
            velocity = self.policy_velocity.copy()

        if self.phase == "human":
            self.r = np.clip(q, self.lower, self.upper)
            self.velocity[:] = 0.  # A passive shadow pose, not a servo command.
        elif dt > 0:
            velocity = np.clip(velocity, -c.reference_speed, c.reference_speed)
            new_r = np.clip(self.r+velocity*dt, self.lower, self.upper)
            self.velocity = (new_r-self.r)/dt
            self.r = new_r
        self.human_latched = self.alpha == 1.
        self.alpha_target = 1. if self.phase in ("blend", "human") else 0.
        return self.r.copy(), self.velocity.copy()
