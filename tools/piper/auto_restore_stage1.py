#!/usr/bin/env python3
"""Auto-restore, first block: feed a peg by hand and drop it into the straightener.

    *** WITH --execute THE FOLLOWER (left/main) ARM MOVES ON ITS OWN. ***

This is stage 1 of the restore routine reconstructed from the human
demonstration in ``auto_restore_0917_new``. It covers only up to the drop into
the straightening fixture; the wrist roll, the regrasp and the return to the
board are deliberately not here yet.

Sequence
--------
1. Move to RESTORE_START (see the constant for where that pose comes from) and
   settle. The demonstration lifted high before swinging the base; from this
   pose the base can be rotated directly, so no lift is planned.
2. Open the jaws to --grip-feed-open-mm, hold there and wait for Enter. Enter
   starts a ``--feed-seconds`` countdown during which you put the peg into the
   open jaws; the gripper closes when the countdown reaches zero.
   With --demo there is no pause: the arm instead descends to DEMO_GRASP, takes
   the peg off the board itself, and returns to RESTORE_START's shape at the
   board's base angle -- the same pivot the hand-fed path leaves from.
3. Rotate the base to the restore workspace with every other joint held still,
   then move the remaining joints onto RELEASE.
4. Open the gripper, dropping the peg into the straightener, then roll the
   wrist over to PREGRASP, open the jaws to --grip-pregrasp-mm and fly the
   descent PREGRASP -> GRASP. The jaws hold that width all the way in, then
   close onto the peg over --grip-close-seconds once GRASP is reached.
5. Carry the peg back out to PREGRASP, roll joint6 --unroll-deg back, lay the
   peg down via PRE_HORI_RELEASE to HORI_RELEASE on the restore table and let
   go, then stand off at TOP_DOWN_PRE_ADJUST and pick it up again from
   straight above at TOP_DOWN_GRASP. The point
   of putting it down and picking it up is to move the jaws onto a different
   pair of faces.
6. Move the other joints back onto their RESTORE_START values with the base
   still out at the table, then rotate the base home.
7. Stand off at TOP_DOWN_PRE_RELEASE, set the peg into its slot at
   TOP_DOWN_RELEASE, open the jaws, and return to RESTORE_START.

Control is the package's MIT position-impedance stream with follower gravity
feed-forward on quintic joint-space segments: the same path
``leader_gravity_id_collect.py`` uses to drive one arm autonomously. Two speeds
are kept, and a segment earns ``--fast-speed`` only once it has been watched
on the hardware at ``--transit-speed``. The whole routine has now been run, so
everything is fast except the last 12.5 deg onto the peg and back off it --
PREGRASP -> GRASP -> PREGRASP -- which stays at ``--transit-speed`` because
that is the leg that has to arrive accurately. Note that step 1 is fast too
although its path is not fixed: it starts from wherever the arm happens to be,
so it is the one segment no run can have validated in advance. There is no
``move_j``/``move_js`` and no firmware planner, and nothing here ever homes to
zero -- the arm starts and ends at RESTORE_START, not at all-zeros.

Read-only unless --execute: the dry run connects, reads feedback, builds the
whole plan and prints its durations, speeds and limit margins.

    python tools/piper/auto_restore_stage1.py                 # dry run
    python tools/piper/auto_restore_stage1.py --execute
    python tools/piper/auto_restore_stage1.py --execute --park-only
    python tools/piper/auto_restore_stage1.py --execute --demo
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ruri.client.controllers.single_piper import discovery
from ruri.client.controllers.single_piper.calibration_ranges import measured_for_arm
from ruri.client.controllers.single_piper.hardware_registry import (
    find_arm_by_hardware_id,
    find_arm_registration,
)
from ruri.client.controllers.single_piper.mit import leader_follower as mit
from ruri.client.controllers.single_piper.mit import policy_controller as pc

R2D = 180.0 / math.pi
REPO = Path(__file__).resolve().parents[2]
FOLLOWER_CALIBRATION = mit.ASSET_ROOT / "calibration_follower.json"
#: Hand-taught poses accumulate here, one JSON object per line. The same file
#: the poses at the top of this module were read out of.
POSE_LOG = Path(__file__).resolve().with_name("auto_restore_poses.jsonl")

#: Where the arm starts and ends. Read off the follower's own feedback frames
#: (0x2A5/0x2A6/0x2A7) on 2026-09-17 with the arm parked where the operator
#: wants the routine to begin -- it is not a pose from the demonstration, and
#: it is not home. Its j5 sits 3.8 deg from the +75 deg stop, so the routine
#: never moves j5 while at this pose.
RESTORE_START_DEG = (-2.227, 44.735, -46.823, -2.506, 71.192, -4.702)

#: Where the peg is dropped into the straightener. Taught by teleop on
#: 2026-09-17 and read off the LEADER's feedback frames, not the follower's:
#: in leader_follower the follower's MIT p_des is the leader's measured pose,
#: so the leader is the command and the follower is where it got to. Those
#: differ by -1.23 deg on joint5 here, and it is the command that replays.
#: (The MIT frames themselves are not observable -- candump sees no 0x15A-0x15F
#: on this adapter -- so the leader is the only way to read the target.)
#:
#: Joint5 sits 5.2 deg from its +75 stop here, the tightest margin of the three
#: taught poses. Joint4 was still drifting 1.29 deg while this was captured, so
#: it is a median of a 6 s window rather than a settled reading.
#:
#: Supersedes two earlier poses, both kept in auto_restore_poses.jsonl: the
#: demonstration's waypoint B (150.43, 93.36, -82.79, -1.90, 54.75, -8.63), and
#: a hand-taught (145.708, 94.369, -84.274, 7.542, 58.399, -21.042) that was
#: tried and reverted for dropping the peg worse despite better joint1 margin.
RELEASE_DEG = (144.233, 101.223, -96.282, 6.483, 69.796, -9.155)

#: The descent onto the peg in the straightener: hover, then grasp. Both were
#: taught by teleop on 2026-09-17 and read off the LEADER's feedback frames --
#: in leader_follower the follower's MIT p_des IS the leader's measured pose, so
#: the leader is the command and the follower is only where it got to. Here that
#: gap is -0.22 deg of joint5 at PREGRASP and +0.59 deg at GRASP, and it is the
#: command that replays. Logged in auto_restore_poses.jsonl.
#:
#: They share joint1 to 0.013 deg and joint6 to 0.105 deg, so the descent is a
#: pure change of the arm's shape at one base angle and one wrist roll: joint2
#: +7.8, joint3 +6.5, joint5 -12.5 deg.
#:
#: An intermediate CLOSE_GRASP_WAYPOINT between these two was removed on the
#: operator's instruction; the descent is now a single leg. Its last value was
#: (136.528, 75.975, -50.193, 15.082, 46.040, 146.135).
#:
#: Two things learned from the earlier poses, worth keeping:
#:   - Do not tune from the post-grasp readout. Before contact the steady-state
#:     offset repeats to 0.007 deg run to run; after contact it varied by 0.62
#:     deg between two runs, because the peg and the fixture set the pose, not
#:     the controller. Setting GRASP to a settled readout once stacked the
#:     offset and landed 1.675 deg off instead of the 1.0 deg wanted.
#:   - Corrections under about a degree do not take. kp=10 turns a 1 deg error
#:     into 0.175 N*m, which is below joint5's breakaway friction: in one run
#:     the joint sat 0.92 deg off target and did not move until the closing jaws
#:     shook it loose. Moving the GRASP joint5 command 0.675 deg moved the arm
#:     0.055 deg, a ratio of 0.08. Make corrections where the joint is still
#:     moving, and make them bigger than a degree.
PREGRASP_DEG = (141.822, 74.362, -62.773, 6.169, 58.712, 167.354)
GRASP_DEG = (141.809, 82.179, -56.312, 6.648, 46.188, 167.459)

#: How far joint6 is rolled back at PREGRASP once the peg is in the jaws. The
#: outbound roll RELEASE -> PREGRASP is +176.5 deg rather than a clean 180, so
#: this does not return joint6 to where it started; it is a fixed -180 from
#: wherever PREGRASP left it, which is what the operator taught.
UNROLL_DEG = 180.0

#: Lying the peg down on the restore table, then picking it up again from
#: straight above so the jaws end up on a different pair of faces. Taught by
#: teleop from the leader with tools/piper/teach_pose.py and logged in
#: auto_restore_poses.jsonl; all three were held still to 0.000 deg.
#:
#: PRE_HORI_RELEASE_DEG was re-taught after the first version: the original
#: (120.960, 106.224, -34.216, -2.845, -68.768, 0.774) sat with joint5 6.2 deg
#: from its stop, where this one has 29.6 deg. It is kept in the pose log.
#:
#: Two of these ask for joint5 PAST its nominal stop -- HORI_RELEASE at -75.140
#: and TOP_DOWN_GRASP at +75.372, against a +/-75 envelope. clamp_joint_target
#: trims every tick of that to the stop without saying so, and the arm settles
#: a degree or so short again. It is not a teaching slip: the demonstration
#: parked joint5 hard against +/-75 for 27% of its length and hit exactly
#: +75.00 at its own top-down grasp. It does mean neither pose has any joint5
#: margin left, and neither can be corrected in that direction.
PRE_HORI_RELEASE_DEG = (118.967, 69.055, -26.554, -3.832, -45.411, 5.057)
HORI_RELEASE_DEG = (120.960, 114.325, -34.669, -2.391, -75.140, 0.930)
#: Stood off above the peg before dropping onto it top-down. Taught after a
#: dry run showed the straight joint-space line from HORI_RELEASE to
#: TOP_DOWN_GRASP sagging 55 mm below both its endpoints, out at radius 539 mm
#: -- a curve through the table with the peg already lying on it. Splitting the
#: leg here is what keeps the jaws above the work.
TOP_DOWN_PRE_ADJUST_DEG = (116.493, 76.196, -92.826, 0.000, 75.366, -11.434)
TOP_DOWN_GRASP_DEG = (119.956, 120.244, -108.246, 5.296, 75.372, -13.852)

#: Standing off above the slot, then setting the peg into it. Both on the near
#: side of the base -- the only taught poses there, joint1 -38 and -47 against
#: +119 to +145 for everything else.
TOP_DOWN_PRE_RELEASE_DEG = (-38.514, 70.188, -41.922, -3.201, 67.734, -34.995)
TOP_DOWN_RELEASE_DEG = (-47.076, 73.560, -23.405, 6.226, 34.553, -52.946)

#: --demo only: pick the peg off the board instead of being handed it. Taught
#: by teleop from the leader; both share joint1 exactly (-31.077), so the
#: descent onto the peg is a pure change of the arm's shape, joint2 +8.1 and
#: joint5 -9.8 deg. Margins are comfortable at both, worst 7.4 deg on joint5.
DEMO_START_PRE_GRASP_DEG = (-31.077, 114.291, -91.226, -5.385, 67.632, -83.309)
DEMO_GRASP_DEG = (-31.077, 122.363, -86.269, -3.716, 57.856, -80.644)

_running = True


def _stop(_sig, _frame):
    global _running
    _running = False
    pc._running = False  # also interrupts the borrowed enable/hold routine


# --------------------------------------------------------------------- plan


@dataclass
class Segment:
    name: str
    kind: str  # transit | settle | grip
    q: np.ndarray  # (n, 6) joint reference, rad
    qd: np.ndarray  # (n, 6) reference velocity, rad/s
    #: Commanded gripper width: one value for the whole segment, or one per
    #: tick to walk the jaws in slowly. None leaves the last command standing.
    grip_m: float | np.ndarray | None = None
    info: dict = field(default_factory=dict)


def quintic_segment(name, start, end, speed, rate, min_seconds=1.0):
    start, end = np.asarray(start, float), np.asarray(end, float)
    duration = max(min_seconds, 1.875 * float(np.max(np.abs(end - start))) / speed)
    n = max(1, int(math.ceil(duration * rate)))
    q = np.empty((n, 6))
    qd = np.empty((n, 6))
    for i in range(n):
        alpha, alpha_dot = mit.quintic_blend((i + 1) / rate, duration)
        q[i] = start + alpha * (end - start)
        qd[i] = alpha_dot * (end - start)
    return Segment(name, "transit", q, qd)


def grip_ramp_segment(name, pose, start_m, end_m, seconds, rate):
    """Hold the pose while walking the jaws from ``start_m`` to ``end_m``.

    The gripper takes a position command, so closing onto something is a ramp
    or it is a lunge; there is no force-only mode to ask for instead.
    """
    n = max(1, int(round(seconds * rate)))
    alpha = np.array([mit.quintic_blend((i + 1) / rate, seconds)[0] for i in range(n)])
    return Segment(
        name,
        "grip",
        np.repeat(np.asarray(pose, float)[None], n, 0),
        np.zeros((n, 6)),
        start_m + alpha * (end_m - start_m),
    )


def hold_segment(name, kind, pose, seconds, rate, grip_m=None):
    n = max(1, int(round(seconds * rate)))
    return Segment(
        name,
        kind,
        np.repeat(np.asarray(pose, float)[None], n, 0),
        np.zeros((n, 6)),
        grip_m,
    )


def build_plan(args, start_q):
    """Everything after the arm is enabled and holding at its current pose."""
    rate = args.rate
    speed = args.transit_speed
    fast = args.fast_speed
    settle = args.settle
    restore = np.radians(RESTORE_START_DEG)
    release = np.radians(RELEASE_DEG)
    pregrasp = np.radians(PREGRASP_DEG)
    grasp = np.radians(GRASP_DEG)
    pre_hori = np.radians(PRE_HORI_RELEASE_DEG)
    hori = np.radians(HORI_RELEASE_DEG)
    topdown_pre = np.radians(TOP_DOWN_PRE_ADJUST_DEG)
    topdown = np.radians(TOP_DOWN_GRASP_DEG)
    pre_setting = np.radians(TOP_DOWN_PRE_RELEASE_DEG)
    setting = np.radians(TOP_DOWN_RELEASE_DEG)
    unrolled = pregrasp.copy()
    unrolled[5] -= math.radians(UNROLL_DEG)

    #: RESTORE_START's shape, parked at whatever base angle that leg starts or
    #: ends on. Each leg reaches its pivot by changing the arm's shape alone,
    #: then crosses the room by moving the base alone. The return pivot follows
    #: PREGRASP's base angle, so its base swing is a strict subset of the sweep
    #: the outbound one already made.
    def pivot_at(pose):
        out = restore.copy()
        out[0] = pose[0]
        return out

    close_m = args.grip_close_mm / 1000.0
    open_m = args.grip_open_mm / 1000.0
    pregrasp_m = args.grip_pregrasp_mm / 1000.0
    feed_open_m = args.grip_feed_open_mm / 1000.0
    demo_pre = np.radians(DEMO_START_PRE_GRASP_DEG)
    demo_grasp = np.radians(DEMO_GRASP_DEG)
    grasp_m = args.grip_grasp_mm / 1000.0

    plan = [
        quintic_segment("1_to_restore_start", start_q, restore, fast, rate),
        hold_segment("1_settle", "settle", restore, settle, rate),
        hold_segment("1b_open_for_feed", "grip", restore, args.grip_seconds,
                     rate, feed_open_m),
    ]
    if args.demo:
        # --demo: take the peg off the board instead of being handed it. Ends
        # on RESTORE_START's shape at the board's base angle, which is the same
        # pivot the hand-fed path leaves from, so stage 3 is unchanged.
        plan += [
            quintic_segment("d1_to_demo_pre_grasp", restore, demo_pre, speed, rate),
            quintic_segment("d2_to_demo_grasp", demo_pre, demo_grasp, speed, rate),
            hold_segment("d3_settle", "settle", demo_grasp, settle, rate),
            grip_ramp_segment("d4_close_on_peg", demo_grasp, feed_open_m, grasp_m,
                              args.grip_close_seconds, rate),
            hold_segment("d5_grip_settle", "settle", demo_grasp, settle, rate),
            quintic_segment("d6_arm_to_pivot", demo_grasp, pivot_at(demo_grasp),
                            speed, rate),
        ]
        here = pivot_at(demo_grasp)
    else:
        # 2 (wait for Enter, then the feed countdown) is not a planned segment:
        # it is open-ended, so it runs as a hold loop in execute_plan.
        plan += [
            hold_segment("2_close_gripper", "grip", restore, args.grip_seconds,
                         rate, close_m),
        ]
        here = restore
    plan += [
        quintic_segment("3a_base_to_workspace", here, pivot_at(release), fast, rate),
        quintic_segment("3b_arm_to_release", pivot_at(release), release, fast, rate),
        hold_segment("3_settle", "settle", release, settle, rate),
        hold_segment("4_open_gripper", "grip", release, args.grip_seconds, rate, open_m),
        quintic_segment("4b_roll_to_pregrasp", release, pregrasp, fast, rate),
        hold_segment("4b_jaws_to_pregrasp", "grip", pregrasp, args.grip_seconds,
                     rate, pregrasp_m),
        quintic_segment("4c_pregrasp_to_grasp", pregrasp, grasp, speed, rate),
        hold_segment("4e_grasp_settle", "settle", grasp, settle, rate),
        grip_ramp_segment("4e2_close_on_peg", grasp, pregrasp_m, grasp_m,
                          args.grip_close_seconds, rate),
        hold_segment("4e3_grip_settle", "settle", grasp, settle, rate),
        quintic_segment("4f_grasp_to_pregrasp", grasp, pregrasp, speed, rate),
        # 5: lay the peg down on the restore table and pick it up from above.
        quintic_segment("5a_unroll_at_pregrasp", pregrasp, unrolled, fast, rate),
        quintic_segment("5b_to_pre_hori", unrolled, pre_hori, fast, rate),
        quintic_segment("5c_to_hori_release", pre_hori, hori, fast, rate),
        hold_segment("5d_settle", "settle", hori, settle, rate),
        hold_segment("5e_release_horizontal", "grip", hori,
                     args.grip_seconds, rate, open_m),
        quintic_segment("5f_to_topdown_pre", hori, topdown_pre, fast, rate),
        quintic_segment("5g_to_topdown", topdown_pre, topdown, fast, rate),
        hold_segment("5h_settle", "settle", topdown, settle, rate),
        grip_ramp_segment("5i_close_topdown", topdown, open_m, grasp_m,
                          args.grip_close_seconds, rate),
        hold_segment("5j_grip_settle", "settle", topdown, settle, rate),
        # 6: default shape, then the base alone carries it back across.
        quintic_segment("6a_arm_to_pivot", topdown, pivot_at(topdown), fast, rate),
        quintic_segment("6b_base_to_start", pivot_at(topdown), restore, fast, rate),
        hold_segment("6c_settle", "settle", restore, settle, rate),
        # 7: set the peg into its slot, let go, and come back.
        quintic_segment("7a_to_pre_release", restore, pre_setting, fast, rate),
        quintic_segment("7b_to_topdown_release", pre_setting, setting, fast, rate),
        hold_segment("7c_settle", "settle", setting, settle, rate),
        hold_segment("7d_release_peg", "grip", setting, args.grip_seconds, rate, open_m),
        quintic_segment("7e_back_to_home", setting, restore, fast, rate),
        hold_segment("7f_settle", "settle", restore, settle, rate),
    ]
    if args.park_only:
        # Just the entry move. Used to walk the arm back onto RESTORE_START
        # after a run left it out at the fixture, without auto-home -- homing
        # goes to all-zeros, which is not where this routine lives.
        return plan[:2]
    return plan


def describe_plan(plan, rate, arm_name, wait_index):
    lower = np.maximum(mit.JOINT_LIMIT_LOWER, _measured_bounds(arm_name)[0])
    upper = np.minimum(mit.JOINT_LIMIT_UPPER, _measured_bounds(arm_name)[1])
    allq = np.concatenate([s.q for s in plan])
    allqd = np.concatenate([s.qd for s in plan])
    total = sum(len(s.q) for s in plan) / rate

    print(f"\nplan: {len(plan)} segments, {total:.1f} s of motion "
          f"(plus an open-ended wait for Enter)")
    for index, segment in enumerate(plan):
        if index == wait_index:
            print(f"  {'--':>6s}  2_wait_for_enter          "
                  f"hold until Enter, then the feed countdown")
        seconds = len(segment.q) / rate
        moved = float(np.max(np.abs(segment.q[-1] - segment.q[0]))) * R2D
        if segment.grip_m is None:
            grip = ""
        elif np.ndim(segment.grip_m) == 0:
            grip = f"  grip -> {float(segment.grip_m) * 1000:.1f} mm"
        else:
            grip = (f"  grip {segment.grip_m[0] * 1000:.1f} -> "
                    f"{segment.grip_m[-1] * 1000:.1f} mm over {seconds:.1f}s")
        peak = float(np.max(np.abs(segment.qd)))
        print(f"  {seconds:5.1f}s  {segment.name:24s} "
              f"max joint move {moved:6.1f} deg  peak {peak:4.2f} rad/s{grip}")

    print("  reference min (deg): " + " ".join(f"{v * R2D:+7.1f}" for v in allq.min(0)))
    print("  reference max (deg): " + " ".join(f"{v * R2D:+7.1f}" for v in allq.max(0)))
    print("  |reference speed| max (rad/s): "
          + " ".join(f"{v:5.2f}" for v in np.abs(allqd).max(0)))

    margin = np.minimum(allq.min(0) - lower, upper - allq.max(0)) * R2D
    print("  margin to the tighter of nominal/measured stops (deg): "
          + " ".join(f"{v:7.1f}" for v in margin))
    tight = [f"J{i + 1} {margin[i]:.1f} deg" for i in range(6) if margin[i] < 5.0]
    if tight:
        print("  NARROW MARGIN: " + ", ".join(tight))
    clipped = int(np.sum(np.any((allq < lower) | (allq > upper), axis=1)))
    if clipped:
        print(f"  ticks outside those stops (would be clamped): {clipped}")
    jumps = [float(np.max(np.abs(b.q[0] - a.q[-1]))) for a, b in zip(plan[:-1], plan[1:])]
    print(f"  largest step between segments: {max(jumps, default=0.0):.4f} rad")


def _measured_bounds(arm_name):
    ranges = measured_for_arm(arm_name)
    lo = np.radians([ranges[f"joint{i}"][0] / 1000.0 for i in range(1, 7)])
    hi = np.radians([ranges[f"joint{i}"][1] / 1000.0 for i in range(1, 7)])
    return lo, hi


# ------------------------------------------------------------------ logging


class RunLogger:
    """Every commanded and measured value of every tick, written at the end.

    Tuning a pose by pushing the arm around only tells you anything if the
    number you pushed it to survives the session, so this records the whole run
    rather than the moments someone remembered to look at.
    """

    FIELDS = ("t_s", "q_ref", "q", "qd", "joint_effort",
              "grip_ref_m", "grip_m", "grip_force")

    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows = {key: [] for key in self.FIELDS}
        self.segment = []
        self.start = time.perf_counter()

    def add(self, segment, **values):
        self.segment.append(segment)
        for key in self.FIELDS:
            self.rows[key].append(values[key])

    def close(self, metadata):
        if not self.segment:
            return None
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in self.rows.items()}
        arrays["segment"] = np.asarray(self.segment)
        path = self.dir / "ticks.npz"
        np.savez_compressed(path, **arrays)
        (self.dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return path


def append_pose(q_deg, grip_mm, grip_force, label, interface, arm_name, extra=None):
    """Append one hand-taught pose to POSE_LOG and return the record."""
    nominal = list(zip(mit.JOINT_LIMIT_LOWER * R2D, mit.JOINT_LIMIT_UPPER * R2D))
    record = {
        "recorded": dt.datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "arm": arm_name,
        "can_interface": interface,
        "source": "auto_restore_stage1.py --adjust-at-grasp, measured MIT feedback",
        "q_deg": [round(float(v), 3) for v in q_deg],
        "q_rad": [round(math.radians(float(v)), 5) for v in q_deg],
        "gripper_mm": round(float(grip_mm), 2),
        "gripper_force": round(float(grip_force), 3),
        "margin_to_nominal_deg": [
            round(min(v - lo, hi - v), 1) for v, (lo, hi) in zip(q_deg, nominal)
        ],
    }
    record.update(extra or {})
    with POSE_LOG.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


# ---------------------------------------------------------------- execution


class EnterWatcher:
    """Read one Enter from stdin without stalling the 100 Hz command stream."""

    def __init__(self):
        self.pressed = threading.Event()
        self._thread = threading.Thread(target=self._wait, daemon=True)

    def start(self):
        self._thread.start()

    def _wait(self):
        try:
            input()
        except (EOFError, OSError):
            pass
        self.pressed.set()


class Streamer:
    """One MIT tick: read feedback, check it, command joints and gripper."""

    def __init__(self, arm, gravity, gripper, args, logger=None):
        self.arm = arm
        self.gravity = gravity
        self.gripper = gripper
        self.args = args
        self.logger = logger
        self.watchdog = mit.FeedbackWatchdog(args.feedback_timeout)
        self.dt = 1.0 / args.rate
        self.next_tick = time.perf_counter()
        self.next_status = self.next_tick
        self.overruns = 0
        self.grip_goal = None

    def tick(self, q_des, qd_des, label):
        now = time.perf_counter()
        sample = mit.read_sample(self.arm, label)
        self.watchdog.observe(sample.stamps, now)
        q_des, qd_des = mit.clamp_joint_target(np.asarray(q_des, float),
                                               np.asarray(qd_des, float))
        mit.send_position_impedance(
            self.arm, sample, q_des, qd_des, self.gravity,
            self.args.kp, self.args.kd, self.args.max_reference_speed, label,
        )
        if self.grip_goal is not None:
            self.gripper.move_gripper_m(self.grip_goal, self.args.grip_force)
        if self.logger is not None:
            grip = mit.read_gripper(self.gripper)
            self.logger.add(
                label,
                t_s=now - self.logger.start,
                q_ref=q_des, q=sample.q, qd=sample.qd,
                joint_effort=sample.joint_effort,
                grip_ref_m=-1.0 if self.grip_goal is None else self.grip_goal,
                grip_m=math.nan if grip is None else grip.width,
                grip_force=math.nan if grip is None else grip.force,
            )

        track = float(np.max(np.abs(sample.q - q_des)))
        speed = float(np.max(np.abs(sample.qd)))
        if track > self.args.max_track_error:
            joint = int(np.argmax(np.abs(sample.q - q_des))) + 1
            raise mit.TeleopAbort(f"J{joint} tracking error {track:.3f} rad")
        if speed > self.args.max_joint_speed:
            joint = int(np.argmax(np.abs(sample.qd))) + 1
            raise mit.TeleopAbort(f"J{joint} speed {speed:.3f} rad/s")
        if now >= self.next_status:
            mit.require_can_mit_enabled(self.arm, label)
            self.next_status = now + 0.1

        self.next_tick += self.dt
        slack = self.next_tick - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        else:
            self.overruns += 1
            self.next_tick = time.perf_counter()
        return sample, track


def wait_for_feed(streamer, pose, args):
    """Hold at ``pose`` until Enter, then count the feed window down and close."""
    watcher = EnterWatcher()
    watcher.start()
    print("\n  >>> PUT THE PEG IN WHEN THE COUNTDOWN STARTS. "
          "Press Enter to begin. <<<")
    while _running and not watcher.pressed.is_set():
        streamer.tick(pose, np.zeros(6), "wait for enter")
    if not _running:
        return False

    print(f"  feeding window: gripper closes to "
          f"{args.grip_close_mm:.1f} mm in {args.feed_seconds:.0f} s")
    start = time.perf_counter()
    spoken = None
    while _running:
        left = args.feed_seconds - (time.perf_counter() - start)
        if left <= 0:
            break
        whole = int(math.ceil(left))
        if whole != spoken:
            print(f"    closing in {whole} ...")
            spoken = whole
        streamer.tick(pose, np.zeros(6), "feed window")
    return _running


def adjust_at_grasp(streamer, pose, args, context):
    """Hold at GRASP while the operator nudges the arm, then record where it is.

    The MIT loop keeps commanding ``pose``, so the arm springs back the moment
    it is let go: what gets recorded is the MEASURED pose while it is being
    held, not a pose the arm will stay at on its own. Push it where you want
    it, keep it there, and press Enter with the other hand.
    """
    watcher = EnterWatcher()
    watcher.start()
    print("\n  >>> ADJUST THE GRASP BY HAND, HOLD IT THERE, PRESS ENTER TO "
          "RECORD. <<<")
    print("      (the arm is commanded to GRASP and will spring back on release)")
    next_print = 0.0
    sample = None
    while _running and not watcher.pressed.is_set():
        sample, _ = streamer.tick(pose, np.zeros(6), "adjust at grasp")
        if time.perf_counter() >= next_print:
            offset = (sample.q - pose) * R2D
            print("      measured " + " ".join(f"{v * R2D:+7.2f}" for v in sample.q)
                  + "   offset " + " ".join(f"{v:+6.2f}" for v in offset))
            next_print = time.perf_counter() + 0.5
    if not _running or sample is None:
        return False

    grip = mit.read_gripper(streamer.gripper)
    record = append_pose(
        sample.q * R2D,
        0.0 if grip is None else grip.width * 1000.0,
        0.0 if grip is None else grip.force,
        args.adjust_label,
        context["interface"],
        context["arm_name"],
        extra={
            "commanded_q_deg": [round(float(v) * R2D, 3) for v in pose],
            "offset_from_commanded_deg": [
                round(float(v) * R2D, 3) for v in (sample.q - pose)
            ],
            "run_log": context["log_dir"],
        },
    )
    print(f"\n  recorded to {POSE_LOG.name}:")
    print("    q_deg  " + " ".join(f"{v:+8.3f}" for v in record["q_deg"]))
    print("    offset " + " ".join(f"{v:+8.3f}" for v in record["offset_from_commanded_deg"]))
    print(f"    gripper {record['gripper_mm']:.2f} mm, force {record['gripper_force']:.2f}")
    return True


def execute_plan(streamer, plan, wait_index, args, context):
    """Run every segment; return None on completion or an abort reason."""
    for number, segment in enumerate(plan):
        if number == wait_index and not wait_for_feed(streamer, segment.q[0], args):
            return None
        if (args.adjust_at_grasp and segment.name == "4e2_close_on_peg"
                and not adjust_at_grasp(streamer, segment.q[0], args, context)):
            return None
        ramp = segment.grip_m is not None and np.ndim(segment.grip_m) > 0
        if segment.grip_m is not None:
            target = segment.grip_m[-1] if ramp else float(segment.grip_m)
            if not ramp:
                streamer.grip_goal = float(segment.grip_m)
            print(f"  [{number + 1}/{len(plan)}] {segment.name}: "
                  f"gripper -> {target * 1000:.1f} mm at {args.grip_force:.1f} N"
                  + (f" over {len(segment.q) / args.rate:.1f}s" if ramp else ""))
        next_print = 0.0
        try:
            for i in range(len(segment.q)):
                if not _running:
                    return None
                if ramp:
                    streamer.grip_goal = float(segment.grip_m[i])
                _, track = streamer.tick(segment.q[i], segment.qd[i], segment.name)
                if time.perf_counter() >= next_print:
                    print(f"  [{number + 1}/{len(plan)}] {segment.name:24s} "
                          f"{i + 1:4d}/{len(segment.q):4d} "
                          f"track={track * R2D:5.2f} deg "
                          f"overruns={streamer.overruns}")
                    next_print = time.perf_counter() + 2.0
        except (mit.TeleopAbort, ValueError, OSError) as exc:
            return str(exc)
    print(f"plan complete, overruns={streamer.overruns}")
    return None


def resolve_follower(args):
    registration = find_arm_registration(args.follower_side, args.follower_role)
    device = discovery.discover_piper_can(
        candidates=None if args.follower_can is None else [args.follower_can],
        configure=args.configure_can,
        hardware_id=registration.can_hardware_id,
    )
    resolved = find_arm_by_hardware_id(device.hardware_id)
    if resolved != registration:
        raise RuntimeError(
            f"{device.interface} is {resolved.name}, not {registration.name}"
        )
    return device.interface, registration


def hold_here(arm, gravity, gripper, args, seconds, label):
    """Best-effort: keep the arm where it is. Never a path to zero."""
    try:
        hold = mit.read_sample(arm, label).q.copy()
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            sample = mit.read_sample(arm, label)
            mit.send_position_impedance(
                arm, sample, hold, np.zeros(6), gravity,
                args.kp, args.kd, args.max_reference_speed, label,
            )
            grip = mit.read_gripper(gripper)
            if grip is not None:
                gripper.move_gripper_m(grip.width, args.grip_force)
            time.sleep(1.0 / args.rate)
    except Exception as exc:  # noqa: BLE001 - best-effort hold
        print(f"{label} warning: {exc}")


def run(args):
    gravity = mit.GravityCompensator(args.follower_calibration)
    interface, registration = resolve_follower(args)
    print(f"follower: {registration.name} on {interface}")

    arm = mit.build_arm(interface)
    started = False
    aborted = None
    logger = None
    try:
        mit.reject_teaching_mode(arm, "follower")
        sample = mit.read_sample(arm, "follower")
        ctrl, mode, _ = mit.arm_status_text(arm)
        print(f"  ctrl/mode: {ctrl} / {mode}")
        print("  q(deg) now:    " + " ".join(f"{v * R2D:+7.1f}" for v in sample.q))
        print("  RESTORE_START: " + " ".join(f"{v:+7.1f}" for v in RESTORE_START_DEG))
        gap = float(np.max(np.abs(sample.q - np.radians(RESTORE_START_DEG)))) * R2D
        print(f"  largest joint gap to RESTORE_START: {gap:.2f} deg")
        mit.bounded_torque(gravity.torque(sample.q), "follower")

        gripper = arm.init_effector("agx_gripper")
        time.sleep(0.5)
        grip = mit.read_gripper(gripper)
        if grip is None:
            raise mit.TeleopAbort("missing follower gripper feedback")
        print(f"  gripper: {grip.width * 1000:.1f} mm")

        plan = build_plan(args, sample.q)
        wait_index = next(
            (i for i, s in enumerate(plan) if s.name == "2_close_gripper"), -1
        )
        describe_plan(plan, args.rate, registration.name, wait_index)

        if not args.execute:
            print("\nDRY RUN: nothing enabled, nothing sent.")
            return 0

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
        print("\n*** THE FOLLOWER ARM WILL MOVE. WORKSPACE CLEAR, E-STOP IN HAND. ***")
        for i in range(5, 0, -1):
            if not _running:
                return 0
            print(f"    starting in {i} ...")
            time.sleep(1.0)

        arm.set_follower_mode()
        time.sleep(0.3)
        pc.enable_follower_with_hold(
            arm, gravity, args.kp, args.kd, args.max_reference_speed
        )
        started = True
        ctrl, mode, _ = mit.arm_status_text(arm)
        print(f"  follower enabled: ctrl={ctrl} mode={mode}")

        log_dir = args.log_dir or (
            REPO / "logs" / "auto_restore"
            / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        logger = RunLogger(log_dir)
        print(f"  logging every tick to {log_dir}")
        context = {
            "interface": interface,
            "arm_name": registration.name,
            "log_dir": str(log_dir),
        }
        streamer = Streamer(arm, gravity, gripper, args, logger)
        streamer.grip_goal = grip.width  # hold the width it already has
        aborted = execute_plan(streamer, plan, wait_index, args, context)
    except (mit.TeleopAbort, ValueError, OSError) as exc:
        aborted = str(exc)
    finally:
        if aborted:
            print(f"\nABORTED: {aborted}")
        if logger is not None:
            written = logger.close({
                "created": dt.datetime.now().isoformat(timespec="seconds"),
                "arm": registration.name,
                "can_interface": interface,
                "aborted": aborted,
                "poses_deg": {
                    "RESTORE_START": list(RESTORE_START_DEG),
                    "RELEASE": list(RELEASE_DEG),
                    "PREGRASP": list(PREGRASP_DEG),
                    "GRASP": list(GRASP_DEG),
                    "PRE_HORI_RELEASE": list(PRE_HORI_RELEASE_DEG),
                    "HORI_RELEASE": list(HORI_RELEASE_DEG),
                    "TOP_DOWN_PRE_ADJUST": list(TOP_DOWN_PRE_ADJUST_DEG),
                    "TOP_DOWN_GRASP": list(TOP_DOWN_GRASP_DEG),
                    "TOP_DOWN_PRE_RELEASE": list(TOP_DOWN_PRE_RELEASE_DEG),
                    "TOP_DOWN_RELEASE": list(TOP_DOWN_RELEASE_DEG),
                    "UNROLL_DEG": UNROLL_DEG,
                },
                "args": {k: str(v) for k, v in vars(args).items()},
                "units": {
                    "t_s": "seconds since the first tick",
                    "q_ref/q/qd": "rad, rad/s",
                    "joint_effort": "N*m",
                    "grip_ref_m/grip_m": "m (grip_ref_m -1 = no command yet)",
                    "grip_force": "N",
                },
            })
            if written is not None:
                print(f"run log: {written}")
        if started:
            hold_here(arm, gravity, gripper, args, args.final_hold_seconds, "final hold")
            print("FOLLOWER REMAINS ENERGIZED AT ITS FINAL MIT COMMAND; "
                  "support it before disabling torque.")
        try:
            arm.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"disconnect warning: {exc}")
    return 1 if aborted else 0


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--execute", action="store_true",
                   help="ENERGIZE the follower and run the plan")
    p.add_argument("--adjust-at-grasp", action="store_true",
                   help="pause at GRASP so the arm can be nudged by hand, and "
                        "append the measured pose to auto_restore_poses.jsonl")
    p.add_argument("--adjust-label", default="grasp adjusted by hand at GRASP",
                   help="label written with the pose recorded by --adjust-at-grasp")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="where the per-tick run log goes "
                        "(default: logs/auto_restore/<timestamp>)")
    p.add_argument("--demo", action="store_true",
                   help="pick the peg off the board at DEMO_GRASP instead of "
                        "waiting to be handed it; skips the Enter/feed pause")
    p.add_argument("--park-only", action="store_true",
                   help="run only the entry move onto RESTORE_START, then stop; "
                        "the way back from the fixture, since homing goes to zero")
    p.add_argument("--follower-can", default=None,
                   help="default: discover by registered hardware ID")
    p.add_argument("--follower-side", default="left")
    p.add_argument("--follower-role", default="main")
    p.add_argument("--configure-can", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--follower-calibration", type=Path, default=FOLLOWER_CALIBRATION)

    p.add_argument("--transit-speed", type=float, default=0.25,
                   help="peak rad/s of the unverified entry move onto RESTORE_START")
    p.add_argument("--fast-speed", type=float, default=1.0,
                   help="peak rad/s of the eye-verified segments of the routine; "
                        "equals --max-reference-speed, so raising it further "
                        "needs that raised too or the velocity feed-forward clips")
    p.add_argument("--settle", type=float, default=1.0,
                   help="seconds held still at the end of each move")
    p.add_argument("--feed-seconds", type=float, default=3.0,
                   help="countdown after Enter before the gripper closes")
    p.add_argument("--grip-seconds", type=float, default=1.5,
                   help="seconds held while the gripper reaches its commanded width")
    p.add_argument("--grip-close-mm", type=float, default=5.0,
                   help="commanded closed width; well under any peg so the jaws "
                        "clamp at --grip-force on whatever is fed in")
    p.add_argument("--grip-open-mm", type=float, default=40.0,
                   help="released width; the demonstration opened to 36.6 mm")
    p.add_argument("--grip-feed-open-mm", type=float, default=50.0,
                   help="width the jaws open to before the feed wait, so there "
                        "is room to put the peg in by hand")
    p.add_argument("--grip-pregrasp-mm", type=float, default=45.9,
                   help="width held throughout the descent onto the peg and back; "
                        "the jaws' measured opening at PREGRASP")
    p.add_argument("--grip-grasp-mm", type=float, default=5.0,
                   help="commanded width when closing on the peg in the fixture; "
                        "like --grip-close-mm this sits under the peg so the jaws "
                        "stop on force, not on a preset width")
    p.add_argument("--grip-close-seconds", type=float, default=3.0,
                   help="seconds to walk the jaws shut onto the peg")
    p.add_argument("--grip-force", type=float, default=1.0)
    p.add_argument("--final-hold-seconds", type=float, default=1.0)

    p.add_argument("--rate", type=float, default=100.0)
    p.add_argument("--kp", type=float, default=mit.DEFAULT_FOLLOWER_KP)
    p.add_argument("--kd", type=float, default=mit.DEFAULT_FOLLOWER_KD)
    p.add_argument("--max-reference-speed", type=float, default=1.0)
    p.add_argument("--max-track-error", type=float, default=0.35)
    p.add_argument("--max-joint-speed", type=float, default=2.0)
    p.add_argument("--feedback-timeout", type=float, default=0.2)
    return p


def validate(args):
    checks = (
        ("transit_speed", 0.05, 1.0), ("fast_speed", 0.05, 1.5),
        ("settle", 0.0, 10.0),
        ("feed_seconds", 1.0, 30.0), ("grip_seconds", 0.5, 10.0),
        ("grip_close_mm", 0.0, 80.0), ("grip_open_mm", 0.0, 80.0),
        ("grip_pregrasp_mm", 0.0, 80.0), ("grip_grasp_mm", 0.0, 80.0),
        ("grip_feed_open_mm", 0.0, 80.0),
        ("grip_close_seconds", 0.5, 20.0),
        ("grip_force", 0.1, 5.0), ("final_hold_seconds", 0.0, 30.0),
        ("rate", 50.0, 200.0), ("kp", 1.0, 20.0), ("kd", 0.1, 2.0),
        ("max_reference_speed", 0.1, 3.0), ("max_track_error", 0.05, 0.5),
        ("max_joint_speed", 0.5, 4.0), ("feedback_timeout", 0.05, 1.0),
    )
    for name, low, high in checks:
        value = getattr(args, name)
        if not (math.isfinite(value) and low <= value <= high):
            raise SystemExit(f"--{name.replace('_', '-')} must be in [{low}, {high}]")
    if args.grip_close_mm >= args.grip_open_mm:
        raise SystemExit("--grip-close-mm must be below --grip-open-mm")
    if max(args.transit_speed, args.fast_speed) > args.max_reference_speed:
        raise SystemExit(
            "--max-reference-speed must cover --transit-speed and --fast-speed"
        )


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
