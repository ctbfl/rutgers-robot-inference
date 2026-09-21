"""Small closed-loop circle policy and offline TCP kinematics for the demo.

Coordinates: base +X forward, +Z up. The default TCP is the URDF gripper
midpoint, 135.8 mm along link6 +Z, not a calibrated attached-tool tip.
SciPy is needed only by this example, never by the shared-reference controller.
"""
from __future__ import annotations

import math
import time
import numpy as np
from scipy.optimize import least_squares

from ruri.client.controllers.single_piper.mit.gravity import PiperGravity, rpy_to_R, axis_angle_to_R
from ruri.client.controllers.single_piper.normalization import normalize_pose
from ruri.client.controllers.single_piper.calibration_ranges import NOMINAL
from ruri.client.controllers.single_piper_assisted.control import physical_bounds
from ruri.client.controllers.single_piper_assisted.config import SinglePiperAssistedConfig
from ruri.client.controllers.single_piper_assisted.control import synchronized_bounds
from ruri.server.wrappers.policy_wrapper import PolicyWrapper


DEFAULT_RADIUS = .10
DEFAULT_PERIOD = 2 * math.pi / .5  # Circle phase advances at 0.5 rad/s.
DEFAULT_TCP_SPEED = .10  # Allows catching a circle point moving at 0.05 m/s.
DEFAULT_REFERENCE_SPEED = 4.
DEFAULT_POLICY_SPEED = .5
RUNTIME_IK_TOLERANCE_M = .0005  # 0.5 mm for the moving demo; audit remains 0.05 mm.


class IKConvergenceError(ValueError):
    """No sufficiently accurate solution found for this requested TCP step."""


class PiperTCP:
    def __init__(self, tcp_offset=(0., 0., .1358), *, config=None):
        self.chain = PiperGravity()
        self.offset = np.asarray(tcp_offset, dtype=float)
        if self.offset.shape != (3,) or not np.isfinite(self.offset).all():
            raise ValueError("TCP offset must contain three finite metres")
        self.config = config or SinglePiperAssistedConfig(leader_only=True)
        self.low, self.high = (v[:6] for v in synchronized_bounds(self.config))
        self.origins_R = [rpy_to_R(*rpy) for rpy in self.chain.joint_rpy]

    def frames(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("FK requires six finite radians")
        R, p, origins, axes = np.eye(3), np.zeros(3), [], []
        for i in range(6):
            p = p + R @ self.chain.joint_xyz[i]
            R = R @ self.origins_R[i]
            origins.append(p.copy())
            axes.append(R @ self.chain.joint_axis[i])
            R = R @ axis_angle_to_R(self.chain.joint_axis[i], q[i])
        return p+R@self.offset, R, np.asarray(origins), np.asarray(axes)

    def fk(self, q):
        p, R, _, _ = self.frames(q)
        return p, R

    def ik(self, xyz, seed, tool_axis=None, *, position_tolerance=5e-5):
        xyz = np.asarray(xyz, dtype=float)
        seed = np.asarray(seed, dtype=float)
        if xyz.shape != (3,) or seed.shape != (6,) or not np.isfinite(np.r_[xyz, seed]).all():
            raise ValueError("IK requires finite xyz and six-joint seed")
        if not math.isfinite(position_tolerance) or position_tolerance <= 0:
            raise ValueError("IK position tolerance must be positive and finite")
        seed = np.clip(seed, self.low+1e-9, self.high-1e-9)
        axis = self.fk(seed)[1][:, 2] if tool_axis is None else np.asarray(tool_axis, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
            raise ValueError("invalid preferred tool axis")
        axis = axis / np.linalg.norm(axis)
        # Free pitch/yaw, with a repeatable roll preference. A position-only
        # solve can drift along the wrist null space on every lap.
        side = np.cross([0., 0., 1.], axis)
        if np.linalg.norm(side) < 1e-6:
            side = np.cross([1., 0., 0.], axis)
        side /= np.linalg.norm(side)
        desired_R = np.column_stack((np.cross(side, axis), side, axis))

        def residual(q):
            p, R = self.fk(q)
            return np.r_[1000*(p-xyz), .3*(R-desired_R).ravel(), .0001*(q-seed)]

        def jacobian(q):
            p, R, origins, axes = self.frames(q)
            rotation_jac = np.column_stack([np.cross(a, R.T).T.ravel() for a in axes])
            return np.vstack((1000*np.cross(axes, p-origins).T,
                              .3*rotation_jac, .0001*np.eye(6)))

        result = least_squares(residual, seed, jac=jacobian, bounds=(self.low, self.high),
                               max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        if not np.isfinite(result.x).all():
            raise IKConvergenceError("IK produced a non-finite solution")
        error = float(np.linalg.norm(self.fk(result.x)[0]-xyz))
        if error > position_tolerance:
            raise IKConvergenceError(f"IK did not reach TCP target; position error {error*1000:.3f} mm")
        return result.x.copy()


class CirclePolicy(PolicyWrapper):
    POLICY_METADATA = {"inputs": {"observation.state": {"type": "state"}}}

    def __init__(self, *, radius=DEFAULT_RADIUS, period=DEFAULT_PERIOD, tcp_speed=DEFAULT_TCP_SPEED, lookahead=.5,
                 center_offset=(.05, 0., .15), tcp_offset=(0., 0., .1358), clock=time.monotonic,
                 config=None):
        for name, value in (("radius", radius), ("period", period), ("tcp_speed", tcp_speed), ("lookahead", lookahead)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.kin = PiperTCP(tcp_offset, config=config)
        if tuple(self.kin.config.leader_sign) != (1.,)*6 or tuple(self.kin.config.leader_offset_rad) != (0.,)*6:
            raise ValueError("circle demo expects identical left/right joint coordinates")
        offset = np.asarray(center_offset, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("center offset must contain three finite metres")
        self.home_tcp = self.kin.fk(np.zeros(6))[0]
        self.center = self.home_tcp+offset
        self.radius, self.period, self.tcp_speed, self.lookahead = radius, period, tcp_speed, lookahead
        self.clock, self.started = clock, clock()
        self.last = None

    @property
    def output_chunk_size(self):
        return 1

    def goal(self, elapsed):
        phase = 2*np.pi*elapsed/self.period
        # Vertical YZ circle. Pitch is a soft preference, never a fixed attitude.
        xyz = self.center + self.radius*np.array([0., np.cos(phase), np.sin(phase)])
        return xyz, self.preferred_axis(xyz, elapsed)

    def preferred_axis(self, xyz, elapsed):
        phase = 2*np.pi*elapsed/self.period
        # Orient at the *next TCP point*, not toward a distant final point.
        # This avoids a wrist roll flip near the straight-wrist home singularity.
        pitch, yaw = .15+.25*np.sin(phase), math.atan2(xyz[1], xyz[0])
        return np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])

    def _infer(self, inputs):
        state = np.asarray(inputs["observation.state"], dtype=float)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("circle policy requires seven finite measured coordinates")
        # Measured states may overshoot nominal bounds; action validation must
        # not truncate or reject real encoder observations near home.
        low, high = physical_bounds(NOMINAL)
        measured = low+(np.r_[(state[:6]+100.)/200., state[6]/100.])*(high-low)
        q, width = measured[:6], float(np.clip(measured[6], 0., .08))
        elapsed = self.clock()-self.started
        goal, _ = self.goal(elapsed)
        current = self.kin.fk(q)[0]
        delta = goal-current
        # Recompute the straight approach from the latest measured TCP every call.
        step = delta*min(1., self.tcp_speed*self.lookahead/max(np.linalg.norm(delta), 1e-12))
        failure = None
        for scale in (1., .5, .25, .125):
            next_xyz = current+scale*step
            try:
                next_q = self.kin.ik(next_xyz, q, self.preferred_axis(next_xyz, elapsed),
                                     position_tolerance=RUNTIME_IK_TOLERANCE_M)
                status = "ok" if scale == 1. else "reduced_step"
                error_mm = 1000*float(np.linalg.norm(self.kin.fk(next_q)[0]-next_xyz))
                break
            except IKConvergenceError as exc:
                failure = str(exc)
        else:
            # A dragged pose can have no feasible straight-line Cartesian step.
            # Propose the latest bounded measured pose, never an old trajectory.
            # This is a normal policy result: it cannot request worker HOLD or
            # terminate gravity dragging. The next query tries the moving goal again.
            next_q = np.clip(q, self.kin.low, self.kin.high)
            next_xyz = self.kin.fk(next_q)[0]
            status, scale, error_mm = "waiting_ik", 0., None
        self.last = {"phase_time_s": elapsed, "goal_tcp": goal.tolist(),
                     "observed_tcp": current.tolist(), "proposed_tcp": next_xyz.tolist(),
                     "ik_status": status, "ik_step_scale": scale,
                     "ik_error_mm": error_mm, "ik_failure": failure}
        return {"action_chunk": normalize_pose(next_q, width)[None, :], "circle": self.last}

    def validate_path(self, samples=120, *, policy_speed=DEFAULT_POLICY_SPEED):
        """Check approach, entire loop, closure and gravity ceilings offline."""
        from ruri.client.controllers.single_piper.mit import leader_follower as mit
        if not math.isfinite(policy_speed) or policy_speed <= 0:
            raise ValueError("policy speed must be positive and finite")
        if type(samples) is not int or samples < 12:
            raise ValueError("path validation needs at least 12 circle samples")
        q = np.zeros(6)
        path, errors = [q.copy()], []
        first, axis = self.goal(0.)
        for f in np.linspace(0, 1, 60)[1:]:
            xyz = self.home_tcp + f*(first-self.home_tcp)
            q = self.kin.ik(xyz, q, self.preferred_axis(xyz, 0.))
            errors.append(np.linalg.norm(self.kin.fk(q)[0]-xyz))
            path.append(q.copy())
        loop = []
        for elapsed in np.linspace(0, self.period, samples+1):
            xyz, axis = self.goal(elapsed)
            q = self.kin.ik(xyz, q, axis)
            errors.append(np.linalg.norm(self.kin.fk(q)[0]-xyz))
            path.append(q.copy())
            loop.append(q.copy())
        path = np.asarray(path)
        loop = np.asarray(loop)
        loop_speed = np.max(np.abs(np.diff(loop, axis=0)), axis=0)/(self.period/samples)
        max_step = float(np.abs(np.diff(path, axis=0)).max())
        if max_step > .08:
            raise ValueError(f"IK path has a discontinuity: {max_step:.3f} rad between audit samples")
        if np.max(loop_speed) > policy_speed:
            raise ValueError(f"circle joint speed exceeds {policy_speed:g} rad/s; increase --period")
        peak_torque = {}
        gravity_arms = ("leader",) if self.kin.config.leader_only else ("leader", "follower")
        for arm in gravity_arms:
            gravity = mit.GravityCompensator(mit.ASSET_ROOT/f"calibration_{arm}.json")
            torques = np.array([mit.bounded_torque(gravity.torque(q), arm) for q in path])
            peak_torque[arm] = np.max(np.abs(torques), axis=0).tolist()
        return {"home_tcp_m": self.home_tcp.tolist(), "center_tcp_m": self.center.tolist(),
                "tcp_offset_m": self.kin.offset.tolist(), "radius_m": self.radius,
                "period_s": self.period, "angular_speed_rad_s": 2*math.pi/self.period,
                "policy_speed_limit_rad_s": policy_speed, "plane": "base YZ", "samples": len(path),
                "max_ik_error_mm": 1000*max(errors),
                "min_joint_margin_deg": float(np.rad2deg(np.minimum(path-self.kin.low, self.kin.high-path)).min()),
                "max_neighbor_joint_step_rad": max_step,
                "loop_peak_joint_speed_rad_s": loop_speed.tolist(),
                "loop_joint_closure_rad": float(np.abs(loop[-1]-loop[0]).max()),
                "peak_gravity_legacy": peak_torque,
                "follower_gravity_compensation": not self.kin.config.leader_only,
                "collision_checked": False}
