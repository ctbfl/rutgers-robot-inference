from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from ruri.client.controllers.single_piper import SinglePiperConfig, SinglePiperController
from ruri.client.controllers.single_piper.discovery import (
    PiperCanDevice,
    discover_piper_can,
    discover_realsense_devices,
)
from ruri.client.controllers.single_piper.hardware_registry import find_arm_registration
from ruri.client.controllers.single_piper.normalization import (
    denormalize_action,
    normalize_pose,
    validate_action,
)


class FakeCamera:
    def __init__(self, value: int):
        self.value = value
        self.is_connected = False
        self.latest_timestamp = float(value)

    def connect(self):
        self.is_connected = True

    def async_read(self, timeout_ms=0):
        return np.full((4, 6, 3), self.value, dtype=np.uint8)

    def disconnect(self):
        self.is_connected = False


class FakeTelemetry:
    def __init__(self, _address):
        self.packet = {
            "phase": "engaged",
            "follower": {"q": [0.0] * 6, "gripper_width": 0.034},
        }

    def open(self):
        pass

    def wait_for_engaged(self, *_):
        return self.packet

    def latest_engaged(self, *_):
        return self.packet

    def close(self):
        pass


class FakeCommands:
    def __init__(self, _address):
        self.sent = []

    def send(self, q, gripper):
        self.sent.append((np.asarray(q), gripper))

    def close(self):
        pass


class FakeWorker:
    def __init__(self, *_):
        self.running = False
        self.logs = []

    def start(self):
        self.running = True

    def stop(self, **_):
        self.running = False


class DiscoveryTests(unittest.TestCase):
    def test_discovers_one_camera_of_each_model(self):
        devices = discover_realsense_devices(
            lambda: [
                {"name": "Intel RealSense D415", "id": "wrist"},
                {"name": "Intel RealSense D435", "id": "head"},
            ]
        )
        self.assertEqual(devices.head_serial, "head")
        self.assertEqual(devices.wrist_serial, "wrist")

    def test_rejects_ambiguous_camera_setup(self):
        with self.assertRaisesRegex(RuntimeError, "exactly one D435"):
            discover_realsense_devices(
                lambda: [
                    {"name": "Intel RealSense D435", "id": "one"},
                    {"name": "Intel RealSense D435", "id": "two"},
                    {"name": "Intel RealSense D415", "id": "wrist"},
                ]
            )

    def test_discovers_only_bus_with_complete_piper_signature(self):
        def probe(interface, _timeout):
            ids = {0x2A1, 0x2A5, 0x2A6, 0x2A7} if interface == "can1" else set()
            return PiperCanDevice(interface, len(ids), frozenset(ids))

        result = discover_piper_can(
            candidates=["can0", "can1"],
            configure=False,
            prober=probe,
            identity_reader=lambda interface: f"usb:test:{interface}",
        )
        self.assertEqual(result.interface, "can1")

    def test_rejects_two_powered_pipers_instead_of_guessing_roles(self):
        def probe(interface, _timeout):
            ids = frozenset((0x2A1, 0x2A5, 0x2A6, 0x2A7))
            return PiperCanDevice(interface, 4, ids)

        with self.assertRaisesRegex(RuntimeError, "found 2 active"):
            discover_piper_can(
                candidates=["renamed-a", "renamed-b"],
                configure=False,
                prober=probe,
                identity_reader=lambda interface: f"usb:test:{interface}",
            )

    def test_registered_hardware_id_survives_can_interface_rename(self):
        left = find_arm_registration("left", "main")
        probed = []

        def probe(interface, _timeout):
            probed.append(interface)
            return PiperCanDevice(
                interface,
                4,
                frozenset((0x2A1, 0x2A5, 0x2A6, 0x2A7)),
            )

        result = discover_piper_can(
            candidates=["can9", "can42"],
            hardware_id=left.can_hardware_id,
            configure=False,
            prober=probe,
            identity_reader=lambda interface: {
                "can9": "usb:test:some-other-adapter",
                "can42": left.can_hardware_id,
            }[interface],
        )
        self.assertEqual(result.interface, "can42")
        self.assertEqual(result.hardware_id, left.can_hardware_id)
        self.assertEqual(probed, ["can42"])


class NormalizationTests(unittest.TestCase):
    def test_pose_round_trip(self):
        q = np.asarray([-0.2, 1.0, -0.8, 0.1, 0.5, -0.3])
        normalized = normalize_pose(q, 0.034)
        restored_q, restored_gripper = denormalize_action(normalized)
        np.testing.assert_allclose(restored_q, q, atol=1e-6)
        self.assertAlmostEqual(restored_gripper, 0.034, places=6)

    def test_finite_policy_overshoot_is_clipped_without_mutating_input(self):
        action = np.asarray(
            [-101.5, 102.0, -100.0, 100.0, 0.0, 500.0, -3.0],
            dtype=np.float32,
        )
        original = action.copy()

        accepted = validate_action(action)

        np.testing.assert_array_equal(action, original)
        np.testing.assert_array_equal(
            accepted,
            [-100.0, 100.0, -100.0, 100.0, 0.0, 100.0, 0.0],
        )

    def test_non_finite_policy_action_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            validate_action([0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0])


class ControllerTests(unittest.TestCase):
    def test_startup_home_skip_threshold_is_safety_bounded(self):
        self.assertEqual(SinglePiperConfig().startup_home_skip_threshold_rad, 0.05)
        with self.assertRaisesRegex(ValueError, "must be between 0 and 0.05"):
            SinglePiperConfig(startup_home_skip_threshold_rad=0.051)

    def test_start_is_a_full_readiness_gate_with_shared_args(self):
        args = SimpleNamespace(configure_can=False, camera_fps=15)

        def can_discovery(**_):
            return PiperCanDevice(
                "can-test",
                4,
                frozenset((0x2A1, 0x2A5, 0x2A6, 0x2A7)),
                find_arm_registration("left", "main").can_hardware_id,
            )

        controller = SinglePiperController(
            args,
            camera_discovery=lambda: type("Devices", (), {
                "head_serial": "head", "wrist_serial": "wrist"
            })(),
            can_discovery=can_discovery,
            camera_factory=lambda *_: FakeCamera(10),
            telemetry_factory=FakeTelemetry,
            command_factory=FakeCommands,
            worker_factory=FakeWorker,
        )

        self.assertIs(controller.args, args)
        self.assertEqual(controller.config.camera_fps, 15)
        self.assertFalse(controller.is_connected)

        controller.start()

        self.assertTrue(controller.is_connected)
        self.assertTrue(controller.arm_started)
        controller.stop()
        self.assertFalse(controller.is_connected)

    def test_standard_observation_and_action_injection(self):
        commands = FakeCommands("unused")

        def can_discovery(**_):
            return PiperCanDevice(
                "can-test",
                4,
                frozenset((0x2A1, 0x2A5, 0x2A6, 0x2A7)),
                find_arm_registration("left", "main").can_hardware_id,
            )

        controller = SinglePiperController(
            SinglePiperConfig(configure_can=False),
            camera_discovery=lambda: type("Devices", (), {
                "head_serial": "head", "wrist_serial": "wrist"
            })(),
            can_discovery=can_discovery,
            camera_factory=lambda serial, _: FakeCamera(10 if serial == "head" else 20),
            telemetry_factory=FakeTelemetry,
            command_factory=lambda _: commands,
            worker_factory=FakeWorker,
        )
        controller.connect()
        self.assertFalse(controller.arm_started)
        self.assertEqual(controller.status()["arm_side"], "left")
        self.assertEqual(controller.status()["arm_role"], "main")
        controller.start_arm()
        observation = controller.get_observation()

        self.assertEqual(
            set(observation),
            {
                "observation.state",
                "observation.images.top",
                "observation.images.wrist",
            },
        )
        self.assertEqual(observation["observation.state"].shape, (7,))
        self.assertEqual(int(observation["observation.images.top"][0, 0, 0]), 10)
        self.assertEqual(int(observation["observation.images.wrist"][0, 0, 0]), 20)

        action = normalize_pose([0.0] * 6, 0.034)
        np.testing.assert_array_equal(controller.send_action(action), action)
        self.assertEqual(len(commands.sent), 1)
        np.testing.assert_allclose(commands.sent[0][0], np.zeros(6), atol=1e-6)
        self.assertAlmostEqual(commands.sent[0][1], 0.034, places=6)

        overshoot = np.asarray(
            [101.0, -102.0, 103.0, -104.0, 105.0, -106.0, 110.0],
            dtype=np.float32,
        )
        expected = np.asarray(
            [100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(controller.send_action(overshoot), expected)
        expected_q, expected_gripper = denormalize_action(expected)
        np.testing.assert_allclose(commands.sent[1][0], expected_q)
        self.assertAlmostEqual(commands.sent[1][1], expected_gripper)

        controller.disconnect()
        self.assertFalse(controller.is_connected)

    def test_rejects_an_active_but_unregistered_can_adapter(self):
        def can_discovery(**_):
            return PiperCanDevice(
                "can-test",
                4,
                frozenset((0x2A1, 0x2A5, 0x2A6, 0x2A7)),
                "usb:test:unregistered",
            )

        controller = SinglePiperController(
            SinglePiperConfig(configure_can=False),
            camera_discovery=lambda: type("Devices", (), {
                "head_serial": "head", "wrist_serial": "wrist"
            })(),
            can_discovery=can_discovery,
            camera_factory=lambda *_: FakeCamera(10),
        )
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            controller.connect()


if __name__ == "__main__":
    unittest.main()
