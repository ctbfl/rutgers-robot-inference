from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import math

import numpy as np

from ruri.client.controllers.single_piper import SinglePiperConfig, SinglePiperController
from ruri.client.controllers.single_piper.camera_params import load_camera_params
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
from ruri.client.controllers.single_piper.single_piper import (
    _configure_realsense_camera,
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
            "follower": {
                "q": [0.0] * 6,
                "qd": [0.1] * 6,
                "joint_effort": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
                "gripper_width": 0.034,
                "gripper_force": 1.5,
            },
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

    def test_recording_does_not_truncate_a_pose_outside_the_box(self):
        # Clamping here instead of in the control path recorded joint6 at
        # bit-exact -100 for 29.2% of tight_insertion_row_1 while the arm was
        # physically past the limit. The follower may legitimately overshoot
        # its clamped target by a small tracking error; the dataset carries it.
        from ruri.client.controllers.single_piper.normalization import (
            CALIBRATION_RANGES,
        )

        # A degree past joint6's lower bound, whatever that bound currently is.
        limit_deg = CALIBRATION_RANGES["joint6"][0] / 1000.0
        past = math.radians(limit_deg - 1.0)
        q = np.asarray([0.0, math.radians(90.0), 0.0, 0.0, 0.0, past])

        normalized = normalize_pose(q, 0.034)

        self.assertLess(normalized[5], -100.0)

    def test_finite_policy_overshoot_is_rejected_without_mutating_input(self):
        action = np.asarray(
            [-101.5, 102.0, -100.0, 100.0, 0.0, 500.0, -3.0],
            dtype=np.float32,
        )
        original = action.copy()

        with self.assertRaisesRegex(ValueError, "outside its normalized target range"):
            validate_action(action)

        np.testing.assert_array_equal(action, original)

    def test_non_finite_policy_action_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            validate_action([0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0])


class ControllerTests(unittest.TestCase):
    def test_teleop_observer_reads_state_and_target_without_touching_can(self):
        class TeleopTelemetry(FakeTelemetry):
            def __init__(self, address):
                super().__init__(address)
                self.packet["action"] = {
                    "q": [0.1] * 6,
                    "gripper_width": 0.02,
                }
                self.closed = False

            def latch_engaged(self, *_):
                return self.packet

            def close(self):
                self.closed = True

        telemetry = TeleopTelemetry("unused")

        def forbid_can(**_):
            raise AssertionError("observer must not discover or open CAN")

        controller = SinglePiperController(
            SinglePiperConfig(configure_can=False),
            camera_discovery=lambda: SimpleNamespace(
                head_serial="head", wrist_serial="wrist"
            ),
            can_discovery=forbid_can,
            camera_factory=lambda serial, _: FakeCamera(
                10 if serial == "head" else 20
            ),
            telemetry_factory=lambda _: telemetry,
        )

        controller.connect_teleop_observer()
        self.assertEqual(list(controller.cameras), ["top", "wrist"])
        self.assertIsNotNone(controller.cameras["top"])
        self.assertIsNotNone(controller.cameras["wrist"])
        observation, target = controller.get_teleop_sample()

        self.assertTrue(controller.teleop_observer_attached)
        self.assertIsNone(controller.status()["can_interface"])
        self.assertEqual(int(observation["observation.images.top"][0, 0, 0]), 10)
        np.testing.assert_array_equal(
            observation["observation.state"], normalize_pose([0.0] * 6, 0.034)
        )
        np.testing.assert_array_equal(target, normalize_pose([0.1] * 6, 0.02))

        recording, recording_target = controller.get_teleop_recording_sample()
        self.assertEqual(
            set(recording),
            {
                "observation.state",
                "observation.joint_effort",
                "observation.images.top",
                "observation.images.wrist",
            },
        )
        np.testing.assert_array_equal(
            recording["observation.state"],
            normalize_pose([0.0] * 6, 0.034),
        )
        np.testing.assert_array_equal(
            recording["observation.joint_effort"],
            np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5], dtype=np.float32),
        )
        np.testing.assert_array_equal(recording_target, target)
        controller.disconnect()
        self.assertEqual(controller.cameras, {"top": None, "wrist": None})
        self.assertTrue(telemetry.closed)

    def test_reviewed_camera_controls_are_loaded_from_json(self):
        params = load_camera_params()
        self.assertEqual(params.warmup_frames, 30)
        self.assertEqual(params.roles["head"].model, "Intel RealSense D435")
        self.assertEqual(params.roles["head"].controls["exposure"], 200.0)
        self.assertEqual(params.roles["wrist"].model, "Intel RealSense D415")
        self.assertEqual(params.roles["wrist"].controls["exposure"], 195.0)
        for role in ("head", "wrist"):
            controls = params.roles[role].controls
            self.assertEqual(controls["gain"], 64.0)
            self.assertEqual(controls["brightness"], 0.0)
            self.assertEqual(controls["enable_auto_exposure"], 0.0)
            self.assertEqual(controls["enable_auto_white_balance"], 1.0)

    def test_realsense_controls_are_locked_then_warmed_up(self):
        option_names = (
            "enable_auto_exposure",
            "enable_auto_white_balance",
            "brightness",
            "contrast",
            "gamma",
            "hue",
            "saturation",
            "sharpness",
            "exposure",
            "gain",
            "white_balance",
            "backlight_compensation",
        )
        fake_rs = SimpleNamespace(
            option=SimpleNamespace(**{name: name for name in option_names})
        )

        class FakeRange:
            min = -1000.0
            max = 10000.0
            step = 1.0
            default = 1.0

        class FakeSensor:
            def __init__(self):
                self.values = {name: 1.0 for name in option_names}
                self.values["white_balance"] = 4600.0

            def supports(self, _option):
                return True

            def get_option_range(self, option):
                value = FakeRange()
                value.default = self.values[option]
                return value

            def set_option(self, option, value):
                self.values[option] = value

            def get_option(self, option):
                return self.values[option]

        class ConfigurableCamera(FakeCamera):
            def __init__(self):
                super().__init__(0)
                self.sensor = FakeSensor()
                device = SimpleNamespace(first_color_sensor=lambda: self.sensor)
                self.rs_profile = SimpleNamespace(get_device=lambda: device)
                self.read_count = 0

            def async_read(self, timeout_ms=0):
                self.read_count += 1
                return super().async_read(timeout_ms)

        config = SinglePiperConfig()
        for role, expected_exposure in (("head", 200.0), ("wrist", 195.0)):
            camera = ConfigurableCamera()
            with patch.dict("sys.modules", {"pyrealsense2": fake_rs}):
                controls = _configure_realsense_camera(camera, role, config)

            self.assertEqual(camera.read_count, 30)
            self.assertEqual(controls["enable_auto_exposure"], 0.0)
            self.assertEqual(controls["exposure"], expected_exposure)
            self.assertEqual(controls["gain"], 64.0)
            self.assertEqual(controls["brightness"], 0.0)
            self.assertEqual(controls["contrast"], 50.0)
            self.assertEqual(controls["gamma"], 300.0)
            self.assertEqual(controls["hue"], 0.0)
            self.assertEqual(controls["saturation"], 64.0)
            self.assertEqual(controls["sharpness"], 50.0)
            self.assertEqual(controls["backlight_compensation"], 0.0)
            self.assertEqual(controls["enable_auto_white_balance"], 1.0)
            self.assertEqual(controls["white_balance"], 4600.0)

    def test_connect_applies_camera_configurator_after_each_camera_connects(self):
        calls = []

        def can_discovery(**_):
            return PiperCanDevice(
                "can-test",
                4,
                frozenset((0x2A1, 0x2A5, 0x2A6, 0x2A7)),
                find_arm_registration("left", "main").can_hardware_id,
            )

        def configure_camera(camera, role, _config):
            self.assertTrue(camera.is_connected)
            calls.append(role)
            return {"exposure": 200.0 if role == "head" else 195.0}

        controller = SinglePiperController(
            SinglePiperConfig(configure_can=False),
            camera_discovery=lambda: SimpleNamespace(
                head_serial="head", wrist_serial="wrist"
            ),
            can_discovery=can_discovery,
            camera_factory=lambda *_: FakeCamera(10),
            camera_configurator=configure_camera,
        )
        controller.connect()

        self.assertEqual(calls, ["head", "wrist"])
        self.assertEqual(controller.status()["head_camera_controls"]["exposure"], 200.0)
        self.assertEqual(controller.status()["wrist_camera_controls"]["exposure"], 195.0)
        controller.disconnect()

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
        self.assertIsNone(controller.send_action(action))
        self.assertEqual(len(commands.sent), 1)
        np.testing.assert_allclose(commands.sent[0][0], np.zeros(6), atol=1e-6)
        self.assertAlmostEqual(commands.sent[0][1], 0.034, places=6)

        overshoot = np.asarray(
            [101.0, -102.0, 103.0, -104.0, 105.0, -106.0, 110.0],
            dtype=np.float32,
        )
        with self.assertRaisesRegex(ValueError, "outside its normalized target range"):
            controller.send_action(overshoot)
        self.assertEqual(len(commands.sent), 1)

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
