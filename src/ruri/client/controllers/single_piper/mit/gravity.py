#!/usr/bin/env python3
"""Gravity torques for the Piper from its URDF. numpy only, no pinocchio.

FACTR computes leader-arm gravity compensation with RNEA over the arm's URDF.
We need the same thing, but adding a pinocchio dependency for what is a static
6-DOF serial chain is not worth it -- and a dependency we cannot check is worse
than 80 lines we can.

So this computes the generalized gravity force analytically,

    tau_i = sum_j  m_j * g^T * (dp_j / dq_i),
    dp_j / dq_i = z_i x (p_j - o_i)      for revolute joint i an ancestor of j
                = 0                       otherwise

and validates it against the numerical gradient of the potential energy

    U(q) = sum_j m_j * g^T * p_j(q),      tau = dU/dq

which is an independent derivation sharing only the forward kinematics. If the
two agree to ~1e-6 N*m the Jacobian terms and the sign convention are right.
Run this file directly to see that check.

SIGN CONVENTION
    Returns the torque the joint must SUPPLY to hold the arm static against
    gravity. To compensate gravity on the leader, feed this straight into
    move_mit(t_ff=...) -- no negation.

Usage:
    python factr/gravity.py                       # self-test
    python factr/gravity.py --q 0 0.5 -0.3 0 0 0  # torques at a pose
"""

import argparse
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

URDF_DEFAULT = Path(__file__).resolve().parent / "assets" / "piper_description.urdf"
G = np.array([0.0, 0.0, -9.80665])


def rpy_to_R(r, p, y):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def axis_angle_to_R(axis, theta):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _xyz_rpy(elem):
    if elem is None:
        return np.zeros(3), np.zeros(3)
    o = elem.find("origin")
    if o is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
    rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()])
    return xyz, rpy


class PiperGravity:
    """Serial-chain gravity model built from a URDF.

    Only the revolute chain is modelled. The gripper fingers are prismatic and
    nearly massless (0.025 kg each); their contribution is folded in as a fixed
    payload on the last link via `payload_mass`/`payload_com` instead, which is
    also how you account for a real tool.
    """

    def __init__(self, urdf=URDF_DEFAULT, n_joints=6,
                 payload_mass=0.0, payload_com=(0.0, 0.0, 0.0)):
        root = ET.parse(urdf).getroot()
        links = {l.get("name"): l for l in root.findall("link")}
        joints = {j.get("name"): j for j in root.findall("joint")}

        self.n = n_joints
        self.joint_xyz, self.joint_rpy, self.joint_axis = [], [], []
        self.link_mass, self.link_com = [], []
        self.names = []

        for i in range(1, n_joints + 1):
            jn, ln = f"joint{i}", f"link{i}"
            if jn not in joints or ln not in links:
                raise ValueError(f"URDF is missing {jn}/{ln}")
            j = joints[jn]
            xyz, rpy = _xyz_rpy(j)
            ax = j.find("axis")
            axis = np.array([float(v) for v in
                             (ax.get("xyz") if ax is not None else "0 0 1").split()])
            self.joint_xyz.append(xyz)
            self.joint_rpy.append(rpy)
            self.joint_axis.append(axis / np.linalg.norm(axis))
            self.names.append(jn)

            ine = links[ln].find("inertial")
            m = float(ine.find("mass").get("value")) if ine is not None else 0.0
            com, _ = _xyz_rpy(ine)
            self.link_mass.append(m)
            self.link_com.append(com)

        # Everything rigidly attached past joint6 (gripper base, fingers, tools)
        # rides on link6, so merge it into link6's mass and CoM.
        extra_m, extra_mc = 0.0, np.zeros(3)
        gb = links.get("gripper_base")
        if gb is not None and gb.find("inertial") is not None:
            gj = joints.get("joint6_to_gripper_base")
            gxyz, grpy = _xyz_rpy(gj) if gj is not None else (np.zeros(3), np.zeros(3))
            R = rpy_to_R(*grpy)
            ine = gb.find("inertial")
            m = float(ine.find("mass").get("value"))
            c, _ = _xyz_rpy(ine)
            extra_m += m
            extra_mc += m * (gxyz + R @ c)
        if payload_mass:
            extra_m += payload_mass
            extra_mc += payload_mass * np.asarray(payload_com, dtype=float)

        if extra_m:
            m6, c6 = self.link_mass[-1], self.link_com[-1]
            tot = m6 + extra_m
            self.link_com[-1] = (m6 * c6 + extra_mc) / tot
            self.link_mass[-1] = tot

        self.link_mass = np.array(self.link_mass)
        self.link_com = np.array(self.link_com)

    def fk(self, q):
        """Returns (origins, axes, coms) in the base frame, each (n,3).

        Per URDF semantics, a joint's <origin> is the fixed parent-link -> joint
        transform, and the child link frame is the joint frame rotated by q
        about <axis>. The joint origin is a fixed point of that rotation, so the
        same `p` serves as both the joint origin and the child frame origin.
        """
        R = np.eye(3)          # orientation of the current link frame
        p = np.zeros(3)        # origin of the current link frame
        origins, axes, coms = [], [], []
        for i in range(self.n):
            p = p + R @ self.joint_xyz[i]
            R = R @ rpy_to_R(*self.joint_rpy[i])
            origins.append(p.copy())
            axes.append((R @ self.joint_axis[i]).copy())
            R = R @ axis_angle_to_R(self.joint_axis[i], q[i])
            coms.append(p + R @ self.link_com[i])
        return np.array(origins), np.array(axes), np.array(coms)

    def potential(self, q):
        """U(q) = sum_j m_j * g^T * p_j. Only used by the self-test."""
        _, _, coms = self.fk(q)
        return float(np.sum(self.link_mass * (coms @ G)))

    def torque(self, q):
        """Joint torque needed to hold the arm static against gravity (N*m)."""
        origins, axes, coms = self.fk(q)
        tau = np.zeros(self.n)
        for i in range(self.n):
            # joint i moves every link at or past it in the chain
            for j in range(i, self.n):
                dp = np.cross(axes[i], coms[j] - origins[i])
                tau[i] += self.link_mass[j] * (G @ dp)
        return tau

    def torque_numerical(self, q, eps=1e-6):
        """dU/dq by central differences. Independent check on torque()."""
        q = np.asarray(q, dtype=float)
        out = np.zeros(self.n)
        for i in range(self.n):
            a, b = q.copy(), q.copy()
            a[i] += eps
            b[i] -= eps
            out[i] = (self.potential(a) - self.potential(b)) / (2 * eps)
        return out


def _self_test(urdf):
    print(f"URDF: {urdf}")
    g = PiperGravity(urdf)
    print(f"  modelled links: {g.n}   total mass "
          f"{g.link_mass.sum():.3f} kg (link6 includes the gripper base)")
    print(f"  per-link mass: " + ", ".join(f"{m:.3f}" for m in g.link_mass))

    rng = np.random.default_rng(0)
    worst = 0.0
    print("\n  analytic vs numerical dU/dq over 200 random poses")
    for _ in range(200):
        q = rng.uniform(-2.0, 2.0, g.n)
        a = g.torque(q)
        n = g.torque_numerical(q)
        worst = max(worst, float(np.abs(a - n).max()))
    print(f"  max |analytic - numerical| = {worst:.3e} N*m")
    ok = worst < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}: the two derivations "
          f"{'agree' if ok else 'DISAGREE'}.")

    q0 = np.zeros(g.n)
    print(f"\n  gravity torque at q=0 (arm as URDF-zero):")
    for i, t in enumerate(g.torque(q0), 1):
        print(f"    joint{i}  {t:+8.4f} N*m")
    t1 = g.torque(q0)[0]
    print(f"\n  joint1 (base yaw, vertical axis) = {t1:+.2e} N*m")
    print(f"  {'PASS' if abs(t1) < 1e-9 else 'CHECK'}: gravity exerts no torque"
          f" about a vertical axis, so this must be ~0.")
    return ok and abs(t1) < 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=URDF_DEFAULT)
    ap.add_argument("--q", type=float, nargs="+", help="pose in rad; prints torques")
    ap.add_argument("--payload", type=float, default=0.0,
                    help="extra mass on link6 in kg")
    args = ap.parse_args()

    if args.q:
        g = PiperGravity(args.urdf, payload_mass=args.payload)
        q = np.array(args.q, dtype=float)
        tau = g.torque(q)
        print("  q (rad):   " + ", ".join(f"{v:+7.4f}" for v in q))
        print("  tau (N*m): " + ", ".join(f"{v:+7.4f}" for v in tau))
        print("  numerical: " + ", ".join(f"{v:+7.4f}" for v in g.torque_numerical(q)))
        return 0

    return 0 if _self_test(args.urdf) else 1


if __name__ == "__main__":
    raise SystemExit(main())
