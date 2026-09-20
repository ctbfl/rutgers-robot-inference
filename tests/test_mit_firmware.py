"""Exercise the vendored SDK at the CAN byte boundary without opening a bus."""

from contextlib import ExitStack
from types import SimpleNamespace
import threading
import time
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from ruri.client.controllers.single_piper.mit import leader_follower as mit
from ruri.client.controllers.single_piper.mit import policy_controller as pc

try:
    import can
    import pyAgxArm as sdk
except ImportError:
    sdk = None


@unittest.skipIf(sdk is None, "requires the vendored pyAgxArm and RURI piper extra")
class FirmwareBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Creating a driver and packing/parsing frames must remain fully offline.
        self.stack.enter_context(patch("socket.socket", side_effect=AssertionError("no hardware in tests")))
        config = sdk.create_agx_arm_config(
            robot=sdk.ArmModel.PIPER, firmeware_version=sdk.PiperFW.V189,
            interface="virtual", channel="offline-firmware-test",
        )
        self.arm = sdk.AgxArmFactory.create_arm(config)
        self.frames = []
        self.arm._ctx.comm = SimpleNamespace(send=self.frames.append)

    def build(self, info):
        factory = self.stack.enter_context(
            patch.object(sdk.AgxArmFactory, "create_arm", return_value=self.arm)
        )
        self.stack.enter_context(patch.object(self.arm, "connect"))
        self.disconnect = self.stack.enter_context(patch.object(self.arm, "disconnect"))
        self.query = self.stack.enter_context(
            patch.object(self.arm, "get_firmware", return_value=info)
        )
        self.stack.enter_context(patch.object(mit.time, "sleep"))
        self.stack.enter_context(patch(
            "ruri.client.controllers.single_piper.discovery.read_can_hardware_id",
            return_value="usb:1d50:606f:0042002F4759530820353131",
        ))
        return factory

    def test_build_selects_v189_and_reads_firmware_before_any_control(self):
        factory = self.build({"software_version": "S-V1.9-1\x00 "})
        wrapped = mit.build_arm("offline-test")
        self.assertEqual(factory.call_args.args[0]["firmeware_version"], "v189")
        self.assertEqual(wrapped.firmware_info["software_version"], "S-V1.9-1")
        self.assertEqual(wrapped.arm_name, "left_arm")
        self.assertEqual(self.frames, [])
        self.query.assert_called_once()
        self.disconnect.assert_not_called()

    def test_old_missing_and_malformed_firmware_disconnect_without_control(self):
        for info in (None, {}, {"software_version": "S-V1.8-2"},
                     {"software_version": "S-V1.8-8"}, {"software_version": "unknown"}):
            with self.subTest(info=info), ExitStack() as local:
                local.enter_context(patch.object(sdk.AgxArmFactory, "create_arm", return_value=self.arm))
                local.enter_context(patch.object(self.arm, "connect"))
                disconnect = local.enter_context(patch.object(self.arm, "disconnect"))
                local.enter_context(patch.object(self.arm, "get_firmware", return_value=info))
                local.enter_context(patch.object(mit.time, "sleep"))
                local.enter_context(patch(
                    "ruri.client.controllers.single_piper.discovery.read_can_hardware_id",
                    return_value="usb:1d50:606f:0042002F4759530820353131",
                ))
                with self.assertRaises(mit.TeleopAbort):
                    mit.build_arm("offline-test")
                disconnect.assert_called_once()
                self.assertEqual(self.frames, [])

    def test_firmware_query_can_retry_without_enabling(self):
        self.build(None)
        self.query.side_effect = [None, {"software_version": "S-V1.9-1"}]
        mit.build_arm("offline-test")
        self.assertEqual(self.query.call_count, 2)
        self.assertEqual(self.frames, [])

    def test_v189_accepts_documented_lower_boundary(self):
        self.assertEqual(
            mit.require_v189_firmware({"software_version": "S-V1.8-9"}, "test"),
            {"software_version": "S-V1.8-9"},
        )

    def test_real_mode_frame_and_signed_12bit_torque_for_every_joint(self):
        for joint in range(1, 7):
            for torque in (-16.0, -3.0, 0.0, 3.0, 16.0):
                with self.subTest(joint=joint, torque=torque):
                    self.arm.move_mit(joint, p_des=0.25, v_des=-0.1, kp=10, kd=0.8, t_ff=torque)
                    frame = self.frames[-1]
                    self.assertEqual(frame.arbitration_id, 0x159 + joint)
                    self.assertEqual(len(frame.data), 8)
                    # Decode independently according to the V188/V189 wire layout.
                    code = ((frame.data[6] & 0x0F) << 8) | frame.data[7]
                    decoded = code * 32.0 / 4095 - 16.0
                    self.assertAlmostEqual(decoded, torque, delta=32.0 / 4095)
                    p = int.from_bytes(frame.data[:2], "big") * 25.0 / 65535 - 12.5
                    self.assertAlmostEqual(p, 0.25, delta=25.0 / 65535)
        mode_frames = [f for f in self.frames if f.arbitration_id == 0x151]
        self.assertTrue(mode_frames)
        self.assertEqual(mode_frames[0].data[0], 0x01)  # CAN control
        self.assertEqual(mode_frames[0].data[1], 0x06)  # New MIT mode
        self.assertEqual(mode_frames[0].data[3], 0xAD)  # MIT enable selector

    def test_six_joint_support_uses_new_frames_after_auto_mode_is_disabled(self):
        self.arm.set_auto_set_motion_mode_enabled(False)
        sample = mit.ArmSample(np.zeros(6), np.zeros(6), np.zeros(6), (1.0,) * 7)
        gravity = SimpleNamespace(torque=lambda q: np.array([1, 2, 3, -1, -2, -3]))
        mit.send_position_impedance(self.arm, sample, sample.q, sample.qd, gravity, 10, 0.8, 3, "test")
        self.assertEqual([f.arbitration_id for f in self.frames], list(range(0x15A, 0x160)))

    def test_real_feedback_decodes_new_mit_and_rejects_old_mit_code(self):
        for mode, expected in ((0x06, "MOVE_MIT"), (0x04, "UNKNOWN")):
            self.arm._parser.parse_packet(can.Message(
                arbitration_id=0x2A1, is_extended_id=False, timestamp=10.0,
                data=[1, 0, mode, 0, 0, 0, 0, 0],
            ))
            self.assertIn(expected, mit.arm_status_text(self.arm)[1])

    def test_existing_torque_units_are_preserved_with_the_pinned_sdk(self):
        for joint, expected in ((1, 4.725), (4, 0.95844)):
            self.arm._parser.parse_packet(can.Message(
                arbitration_id=0x250 + joint, is_extended_id=False, timestamp=10.0,
                data=[0, 0, 0x03, 0xE8, 0, 0, 0, 0],  # 1 A
            ))
            self.assertAlmostEqual(self.arm.get_motor_states(joint).msg.torque, expected)

    def test_leader_gravity_real_frames_have_no_position_spring_or_duplicate_damping(self):
        self.arm.set_auto_set_motion_mode_enabled(False)
        gravity = mit.GravityCompensator(mit.ASSET_ROOT / "calibration_leader.json")
        q = np.array([.1, .3, -.4, .2, .3, -.1])
        qd = np.array([.1, -.2, .3, -.4, .2, -.1])
        sample = mit.ArmSample(q, qd, np.zeros(6), (1.0,) * 7)
        torque = mit.send_leader(self.arm, sample, gravity, mit.DEFAULT_LEADER_KD)
        np.testing.assert_allclose(torque, gravity.torque(q) - .2 * qd)
        self.assertEqual(len(self.frames), 6)
        for joint, frame in enumerate(self.frames):
            self.assertEqual(frame.arbitration_id, 0x15A + joint)
            kp_code = ((frame.data[3] & 15) << 8) | frame.data[4]
            kd_code = (frame.data[5] << 4) | (frame.data[6] >> 4)
            self.assertEqual(kp_code, 0)
            # kd=0 encoded in the SDK's signed [-5,5] range; software already
            # supplies damping through t_ff, so no additional kd=.2 is sent.
            self.assertEqual(kd_code, 2047)
            t_code = ((frame.data[6] & 15) << 8) | frame.data[7]
            self.assertAlmostEqual(t_code * 32 / 4095 - 16, torque[joint], delta=32 / 4095)

    def test_old_and_new_encoders_preserve_current_intent_with_firmware_b_scaling(self):
        config = sdk.create_agx_arm_config(robot=sdk.ArmModel.PIPER,
                                           firmeware_version=sdk.PiperFW.DEFAULT,
                                           interface="virtual", channel="offline-old")
        old = sdk.AgxArmFactory.create_arm(config)
        old_frames = []
        old._ctx.comm = SimpleNamespace(send=old_frames.append)
        old.set_auto_set_motion_mode_enabled(False)
        self.arm.set_auto_set_motion_mode_enabled(False)
        for joint in range(1, 7):
            k, b = config["joint_torque_k"][joint-1], config["joint_torque_b"][joint-1]
            for requested in (-3., -.5, 0., .5, 3.):
                old.move_mit(joint, kp=0., kd=0., t_ff=requested)
                self.arm.move_mit(joint, kp=0., kd=0., t_ff=requested)
                a, z = old_frames[-1].data, self.frames[-1].data
                # Pinned DEFAULT codec puts torque above the CRC nibble.
                old_code = ((a[6] & 15) << 4) | (a[7] >> 4)
                new_code = ((z[6] & 15) << 8) | z[7]
                old_current = (old_code * 16 / 255 - 8) / k
                new_current = (new_code * 32 / 4095 - 16) / (b*k)
                desired_current = requested / (b*k)
                self.assertAlmostEqual(old_current, desired_current, delta=16 / 255 / k)
                self.assertAlmostEqual(new_current, desired_current, delta=32 / 4095 / (b*k))


@unittest.skipIf(sdk is None, "requires the vendored pyAgxArm and RURI piper extra")
class VirtualCANStartupTests(unittest.TestCase):
    def test_firmware_query_and_mit_handshake_through_real_sdk_threads(self):
        channel = f"ruri-v189-{uuid.uuid4()}"
        stop = threading.Event()
        received = []
        failures = []
        # The payload shape and version match the 2026-09-20 hardware capture.
        firmware = bytearray(88)
        for start, value in ((0, b"H-V1.2-1"), (16, b"10"), (32, b"ARM_MC"),
                             (60, b"S-V1.9-1"), (68, b"260920"), (76, b"15")):
            firmware[start:start + len(value)] = value
        peer = can.Bus(interface="virtual", channel=channel)

        def send(cid, data):
            peer.send(can.Message(arbitration_id=cid, is_extended_id=False, data=data))

        def device():
            enabled = False
            mode = 0
            next_feedback = time.monotonic()
            try:
                while not stop.is_set():
                    msg = peer.recv(timeout=0.001)
                    if msg is not None:
                        received.append(msg)
                        if msg.arbitration_id == 0x4AF:
                            for i in range(0, 88, 8):
                                send(0x4AF, firmware[i:i + 8])
                        elif msg.arbitration_id == 0x471:
                            enabled = True
                        elif msg.arbitration_id == 0x151:
                            mode = msg.data[1]
                    if time.monotonic() >= next_feedback:
                        send(0x2A1, [1 if mode else 0, 0, mode, 0, 0, 0, 0, 0])
                        for cid in range(0x2A5, 0x2A8):
                            send(cid, [0] * 8)
                        for joint in range(6):
                            send(0x251 + joint, [0] * 8)
                            send(0x261 + joint, [0, 230, 0, 30, 30, 0x40 if enabled else 0, 0, 0])
                        next_feedback = time.monotonic() + 0.005
            except Exception as exc:
                failures.append(exc)

        thread = threading.Thread(target=device, daemon=True)
        thread.start()
        arm = None
        real_config = sdk.create_agx_arm_config
        try:
            with patch("socket.socket", side_effect=AssertionError("no physical sockets")), \
                    patch.object(sdk, "create_agx_arm_config", side_effect=lambda **kw: real_config(interface="virtual", **kw)), \
                    patch("ruri.client.controllers.single_piper.discovery.read_can_hardware_id",
                          return_value="usb:1d50:606f:0042002F4759530820353131"), \
                    patch.object(pc, "_running", True):
                arm = mit.build_arm(channel)
                self.assertEqual(arm.firmware_info["software_version"], "S-V1.9-1")
                self.assertEqual({msg.arbitration_id for msg in received}, {0x4AF})
                pc.enable_follower_with_hold(
                    arm, SimpleNamespace(torque=lambda q: np.zeros(6)), 10, 0.8, 3,
                )
                self.assertIn("MIT", mit.arm_status_text(arm)[1])
                self.assertFalse(arm.get_auto_set_motion_mode_enabled())
                modes = [msg for msg in received if msg.arbitration_id == 0x151]
                self.assertTrue(modes)
                self.assertTrue(all(msg.data[1] == 0x06 for msg in modes))
                self.assertTrue(set(range(0x15A, 0x160)).issubset(
                    {msg.arbitration_id for msg in received}
                ))
        finally:
            if arm is not None:
                arm.disconnect()
            stop.set()
            thread.join(timeout=2)
            peer.shutdown()
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
