from __future__ import annotations

import unittest

from ruri.client.controllers.single_piper.discovery import PiperCanDevice
from ruri.client.controllers.single_piper.hardware_registry import find_arm_registration
from ruri.client.controllers.single_piper_leader_follower_teleop import (
    SinglePiperLeaderFollowerTeleopConfig,
    SinglePiperLeaderFollowerTeleopController,
)


class FakeTelemetry:
    def __init__(self, _address):
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def wait_for_engaged(self, *_):
        return {"phase": "engaged"}

    def close(self):
        self.closed = True


class FakeWorker:
    def __init__(self, config, leader_can, follower_can):
        self.config = config
        self.leader_can = leader_can
        self.follower_can = follower_can
        self.running = False
        self.logs = []

    def start(self):
        self.running = True

    def stop(self, **_):
        self.running = False


class SinglePiperTeleopTests(unittest.TestCase):
    def test_discovers_registered_pair_and_starts_one_dual_arm_worker(self):
        discovered = []
        created = {}
        telemetry = FakeTelemetry("unused")

        def can_discovery(**kwargs):
            hardware_id = kwargs["hardware_id"]
            discovered.append(hardware_id)
            right = find_arm_registration("right", "secondary").can_hardware_id
            interface = "renamed-right" if hardware_id == right else "renamed-left"
            return PiperCanDevice(
                interface=interface,
                frames_seen=4,
                arbitration_ids=frozenset((0x2A1, 0x2A5, 0x2A6, 0x2A7)),
                hardware_id=hardware_id,
            )

        def worker_factory(config, leader_can, follower_can):
            worker = FakeWorker(config, leader_can, follower_can)
            created["worker"] = worker
            return worker

        controller = SinglePiperLeaderFollowerTeleopController(
            SinglePiperLeaderFollowerTeleopConfig(configure_can=False),
            can_discovery=can_discovery,
            telemetry_factory=lambda _: telemetry,
            worker_factory=worker_factory,
        )
        controller.start()

        self.assertEqual(
            discovered,
            [
                find_arm_registration("right", "secondary").can_hardware_id,
                find_arm_registration("left", "main").can_hardware_id,
            ],
        )
        self.assertEqual(created["worker"].leader_can, "renamed-right")
        self.assertEqual(created["worker"].follower_can, "renamed-left")
        self.assertTrue(controller.arm_started)
        self.assertTrue(telemetry.opened)
        self.assertTrue(telemetry.closed)
        self.assertEqual(controller.status()["follower_can_interface"], "renamed-left")

        controller.stop()
        self.assertFalse(controller.is_connected)
        self.assertFalse(created["worker"].running)

    def test_rejects_scheduler_action_path(self):
        controller = SinglePiperLeaderFollowerTeleopController(
            SinglePiperLeaderFollowerTeleopConfig(configure_can=False)
        )
        with self.assertRaisesRegex(RuntimeError, "Leader motion"):
            controller.send_action([0.0] * 7)


if __name__ == "__main__":
    unittest.main()
