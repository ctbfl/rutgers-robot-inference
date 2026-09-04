"""The calibration files must agree with each other and with the code."""

from __future__ import annotations

import json
import unittest

import numpy as np

from ruri.client.controllers.single_piper import calibration_ranges as base
from ruri.client.controllers.single_piper.normalization import CALIBRATION_RANGES
from ruri.client.controllers.single_piper_leader_follower_teleop import (
    calibration_ranges as teleop,
)


class CopiesDoNotDriftTests(unittest.TestCase):
    def test_teleop_calibration_files_are_byte_identical_copies(self):
        # The teleop controller keeps its own copies so its calibration reads on
        # its own. Hand-copying a joint table between packages is exactly how the
        # same wrong joint6 row ended up in three separate repositories.
        for name in ("piper_range.json", "left_arm_real.json", "right_arm_real.json"):
            self.assertEqual(
                (base.CALIBRATION_DIR / name).read_bytes(),
                (teleop.CALIBRATION_DIR / name).read_bytes(),
                f"{name} has drifted between the two controllers",
            )

    def test_teleop_roles_map_to_the_documented_arms(self):
        # Role (leader/follower) and arm (left/right) are separate facts; this
        # pins the binding between them so a log line naming one is unambiguous
        # about the other.
        self.assertEqual(teleop.LEADER_ARM, "right_arm")
        self.assertEqual(teleop.FOLLOWER_ARM, "left_arm")
        self.assertEqual(
            teleop.LEADER.ranges, base.measured_for_arm(teleop.LEADER_ARM).ranges
        )
        self.assertEqual(
            teleop.FOLLOWER.ranges, base.measured_for_arm(teleop.FOLLOWER_ARM).ranges
        )

    def test_every_arm_name_resolves_end_to_end(self):
        from ruri.client.controllers.single_piper.hardware_registry import (
            ARM_NAMES, REGISTERED_ARMS, find_arm_by_name,
        )

        self.assertEqual(sorted(a.name for a in REGISTERED_ARMS), sorted(ARM_NAMES))
        for name in ARM_NAMES:
            registration = find_arm_by_name(name)
            self.assertEqual(
                base.arm_name_for_hardware_id(registration.can_hardware_id), name
            )
            self.assertIn(name, base.MEASURED_BY_ARM)

    def test_measured_files_declare_the_arm_they_describe(self):
        for name, path in (("left_arm", base.LEFT_ARM_PATH),
                           ("right_arm", base.RIGHT_ARM_PATH)):
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["name"], name, path.name)


class NominalIsTheSingleSourceTests(unittest.TestCase):
    def test_normalizer_uses_the_nominal_envelope(self):
        self.assertEqual(
            {k: tuple(v) for k, v in CALIBRATION_RANGES.items()},
            {k: tuple(v) for k, v in base.NOMINAL.ranges.items()},
        )


class MeasurementsAreConsistentTests(unittest.TestCase):
    def test_nominal_stays_close_to_what_both_arms_reach(self):
        # The nominal envelope is not either arm's travel, but it must not be so
        # far inside one that a reachable pose is truncated on every episode, nor
        # so far outside that a commanded target drives an arm into a hard stop.
        for name in base.RANGE_NAMES:
            n_lo, n_hi = base.NOMINAL[name]
            for arm in (base.LEFT_ARM, base.RIGHT_ARM):
                a_lo, a_hi = arm[name]
                span = a_hi - a_lo
                self.assertLess(abs(n_lo - a_lo), 0.10 * span, f"{name} lower")
                self.assertLess(abs(n_hi - a_hi), 0.10 * span, f"{name} upper")

    def test_both_arms_agree_on_travel(self):
        # Two independently dragged arms; the stops are real, so the widths match.
        # joint2's 6.2 deg zero offset shifts its endpoints but not its width.
        for name in base.RANGE_NAMES:
            left = base.LEFT_ARM[name][1] - base.LEFT_ARM[name][0]
            right = base.RIGHT_ARM[name][1] - base.RIGHT_ARM[name][0]
            self.assertLess(abs(left - right), 0.01 * max(left, right), name)


class FilesAreWellFormedTests(unittest.TestCase):
    def test_every_file_declares_its_units_and_provenance(self):
        for path in sorted(base.CALIBRATION_DIR.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("units", raw, path.name)
            self.assertEqual(raw["units"]["joints"], "0.001 deg", path.name)
            self.assertEqual(raw["units"]["gripper"], "micrometre", path.name)
            self.assertIn(raw["kind"], {"nominal", "measured"}, path.name)
            if raw["kind"] == "measured":
                self.assertIn("can_hardware_id", raw, path.name)
                self.assertIn("measured_on", raw, path.name)

    def test_a_malformed_range_is_rejected(self):
        with self.assertRaises(ValueError):
            base._pair([5.0, 5.0], "test")
        with self.assertRaises(ValueError):
            base._pair([float("nan"), 1.0], "test")
        with self.assertRaises(ValueError):
            base._pair([1.0], "test")


if __name__ == "__main__":
    unittest.main()

