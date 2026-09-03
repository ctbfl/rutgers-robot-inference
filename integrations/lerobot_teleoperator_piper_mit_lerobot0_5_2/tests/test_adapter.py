from __future__ import annotations

import unittest

import numpy as np

from lerobot_teleoperator_piper_mit_lerobot0_5_2 import PiperMIT, PiperMITConfig
from ruri.client.controllers.single_piper.normalization import (
    ACTION_KEYS,
    normalize_pose,
)


class FakeTelemetry:
    is_open = True

    def __init__(self, packet):
        self.packet = packet

    def latest_engaged(self, _timeout_s):
        return self.packet

    def latched(self):
        return self.packet


class PiperMITTests(unittest.TestCase):
    def test_action_comes_from_the_already_latched_ruri_packet(self):
        q = np.asarray([0.1, 0.8, -0.7, 0.2, -0.1, 0.3])
        packet = {
            "phase": "engaged",
            "sequence": 42,
            "action": {"q": (q + 0.02).tolist(), "gripper_width": 0.04},
        }
        teleop = PiperMIT(PiperMITConfig(id="test"))
        teleop.telemetry = FakeTelemetry(packet)
        teleop.connect()

        action = teleop.get_action()

        expected = normalize_pose(q + 0.02, 0.04)
        self.assertEqual(
            action,
            {key: float(value) for key, value in zip(ACTION_KEYS, expected, strict=True)},
        )
        self.assertEqual(packet["sequence"], 42)
        teleop.disconnect()


if __name__ == "__main__":
    unittest.main()
