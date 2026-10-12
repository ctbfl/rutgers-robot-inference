"""Transfer the follower from the shared goal to a live leader in joint space.

Only actuator references live here; no inference, SDK access or authority logic.
The transfer also preserves a moving policy reference at direct takeover.
"""

import numpy as np

from .reference import vector


class FollowerHandoff:
    def __init__(self, config, initial, lower, upper):
        self.config = config
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.sync(initial, np.zeros(6))

    def sync(self, reference, velocity):
        self.r = vector(reference, "follower reference")
        self.velocity = vector(velocity, "follower velocity")
        self.start = self.r.copy()
        self.start_velocity = self.velocity.copy()
        self.elapsed = 0.
        self.active = False
        self.limited = False

    @property
    def progress(self):
        return min(1., self.elapsed/self.config.follower_handoff_s) if self.active else 0.

    def ready(self, leader):
        # Wait rather than snap or abort if a large gap hit the velocity limit.
        goal = np.clip(leader, self.lower, self.upper)
        return (self.active and self.progress >= 1.-1e-12
                and np.all(abs(goal-self.r) <= np.radians(self.config.human_quiet_range_deg)))

    def follow(self, leader, dt):
        goal = np.clip(vector(leader, "leader"), self.lower, self.upper)
        if not self.active:
            self.active = True
            self.start = self.r.copy()
            self.start_velocity = self.velocity.copy()
            # First gravity-only tick sends exactly the preceding shared goal.
            self.elapsed = 0.
            return self.r.copy(), self.velocity.copy()
        else:
            self.elapsed += dt
        u = self.progress
        weight = u*u*u*(10.+u*(-15.+6.*u))
        # Quintic Hermite term: derivative is the old velocity at entry and
        # zero at the end. The original zero-velocity transfer is unchanged.
        tangent = self.config.follower_handoff_s*(u-6*u**3+8*u**4-3*u**5)
        desired = np.clip((1.-weight)*self.start+weight*goal+tangent*self.start_velocity,
                          self.lower, self.upper)
        delta = desired-self.r
        step = np.clip(delta, -self.config.reference_speed*dt, self.config.reference_speed*dt)
        self.limited = bool(np.any(abs(delta-step) > 1e-12))
        self.r += step
        self.velocity = step/dt if dt > 0. else np.zeros(6)
        return self.r.copy(), self.velocity.copy()
