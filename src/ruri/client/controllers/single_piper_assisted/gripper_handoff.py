"""Explicit takeover gesture, isolated from the task gripper's commands.

All widths are measured in metres. No hardware or inference dependencies.
"""

import numpy as np


class GripperHandoff:
    tolerance = .002
    settle_s = .05

    def __init__(self, config, open_width, lower, upper):
        self.config = config
        self.open_width = float(open_width)
        self.lower, self.upper = float(lower), float(upper)
        self.reset()

    def reset(self):
        self.phase = "opening"
        self.elapsed = self.closure = 0.
        self.leader_target = self.open_width
        self.follower_hold = None

    def observe(self, leader, follower, dt):
        """Return (new gesture, alignment complete); never mirror the gesture."""
        if self.phase == "opening":
            self.elapsed = self.elapsed+dt if leader >= self.open_width-self.tolerance else 0.
            if self.elapsed >= self.settle_s-1e-12:
                self.phase, self.elapsed = "armed", 0.
        elif self.phase == "armed":
            # Weak outward spring; absolute threshold, independent of velocity
            # and of how far past nominal full-open the measured stop lies.
            self.leader_target = self.open_width
            self.closure = max(0., self.open_width-leader)
            if self.closure >= self.config.gripper_trigger_m-1e-12:
                self.phase, self.elapsed = "aligning", 0.
                self.follower_hold = float(np.clip(follower, self.lower, self.upper))
                self.leader_target = float(leader)
                return True, False
        elif self.phase == "aligning":
            goal = float(np.clip(follower, self.lower, self.upper))
            step = self.config.gripper_alignment_speed*dt
            self.leader_target += float(np.clip(goal-self.leader_target, -step, step))
            aligned = (abs(leader-goal) <= self.tolerance
                       and abs(self.leader_target-goal) <= self.tolerance)
            self.elapsed = self.elapsed+dt if aligned else 0.
            return False, self.elapsed >= self.settle_s-1e-12
        return False, False

    def opening_force(self, measured):
        # Hardware trial: 0.05/0.1 N did not move from 42.7 mm; 0.2 N
        # repeatedly opened 40 -> 99.4..99.6 mm in about 0.75 s. Use that
        # effort until arrival, then the weak spring while waiting for a hand.
        if self.phase == "opening":
            return self.config.gripper_open_force
        return min(.05, self.config.gripper_open_force,
                   max(.005, 5.*abs(self.open_width-measured)))
