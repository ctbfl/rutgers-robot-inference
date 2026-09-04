"""Piper joint envelopes for the leader/follower teleop controller.

The three files in ``calibration/`` are copies of the ones under
``single_piper``: the same nominal envelope this project normalizes against,
and the same two per-arm measurements. They are duplicated rather than imported
so that this controller's calibration is legible on its own, and a unit test
asserts the copies never drift from the originals.

What this module adds over the single_piper one is the teleop role binding.
Role and arm are two different things: ``leader``/``follower`` is the job an
arm is doing this session, ``left_arm``/``right_arm`` is which arm in the room
it is. On this rig the leader is the right arm and the follower is the left,
and ``LEADER_ARM``/``FOLLOWER_ARM`` name that binding so a reader never has to
infer it from a CAN interface.
"""

from __future__ import annotations

from pathlib import Path

from ruri.client.controllers.single_piper.calibration_ranges import (
    RANGE_NAMES,
    PiperRanges,
    load_ranges,
)


CALIBRATION_DIR = Path(__file__).with_name("calibration")
NOMINAL_PATH = CALIBRATION_DIR / "piper_range.json"
LEADER_PATH = CALIBRATION_DIR / "right_arm_real.json"
FOLLOWER_PATH = CALIBRATION_DIR / "left_arm_real.json"

#: The envelope normalization and the follower-target clamp both use.
NOMINAL = load_ranges(NOMINAL_PATH)

#: Which physical arm plays each teleop role on this rig.
LEADER_ARM = "right_arm"
FOLLOWER_ARM = "left_arm"

#: What each arm physically reaches. Never used for normalization -- only for
#: deciding whether a commanded target is safe on that particular arm.
LEADER = load_ranges(LEADER_PATH)
FOLLOWER = load_ranges(FOLLOWER_PATH)

__all__ = [
    "CALIBRATION_DIR", "NOMINAL_PATH", "LEADER_PATH", "FOLLOWER_PATH",
    "RANGE_NAMES", "PiperRanges", "load_ranges", "NOMINAL",
    "LEADER_ARM", "FOLLOWER_ARM", "LEADER", "FOLLOWER",
]
