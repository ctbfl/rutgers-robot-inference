import json
import socket
import unittest

import numpy as np

from ruri.client.controllers.single_piper.mit.telemetry import (
    MITTelemetryPublisher,
    parse_udp_endpoint,
)


class EndpointTests(unittest.TestCase):
    def test_accepts_only_local_udp(self):
        endpoint = parse_udp_endpoint("udp://localhost:6670")
        self.assertEqual(endpoint.host, "127.0.0.1")
        self.assertEqual(endpoint.port, 6670)

        for address in (
            "tcp://127.0.0.1:6670",
            "udp://192.168.1.20:6670",
            "udp://127.0.0.1:6670/path",
        ):
            with self.subTest(address=address), self.assertRaises(ValueError):
                parse_udp_endpoint(address)


class PublisherTests(unittest.TestCase):
    def test_udp_packet_contains_exact_state_and_action(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1.0)
        port = receiver.getsockname()[1]
        publisher = MITTelemetryPublisher(f"udp://127.0.0.1:{port}")
        try:
            q = np.linspace(-0.3, 0.3, 6)
            qd = np.linspace(-0.2, 0.2, 6)
            sent = publisher.publish(
                phase="engaged",
                leader_q=q + 0.01,
                leader_qd=qd,
                follower_q=q,
                follower_qd=qd - 0.01,
                follower_target_q=q + 0.02,
                follower_target_qd=qd + 0.02,
                leader_gripper_width=0.04,
                follower_gripper_width=0.039,
                follower_gripper_target=0.04,
                follower_gripper_force=-1.2,
                overruns=3,
            )
            payload, _ = receiver.recvfrom(65535)
            packet = json.loads(payload)
        finally:
            publisher.close()
            receiver.close()

        self.assertTrue(sent)
        self.assertEqual(packet["protocol"], "piper_mit_telemetry")
        self.assertEqual(packet["version"], 1)
        self.assertEqual(packet["phase"], "engaged")
        self.assertEqual(packet["sequence"], 1)
        np.testing.assert_allclose(packet["follower"]["q"], q)
        np.testing.assert_allclose(packet["action"]["q"], q + 0.02)
        self.assertAlmostEqual(packet["action"]["gripper_width"], 0.04)
        self.assertEqual(packet["overruns"], 3)

    def test_bad_sample_is_dropped_instead_of_raising(self):
        publisher = MITTelemetryPublisher("udp://127.0.0.1:6670")
        try:
            self.assertFalse(
                publisher.publish(phase="engaged", follower_q=[float("nan")] * 6)
            )
            self.assertEqual(publisher.dropped, 1)
            self.assertIsNotNone(publisher.last_error)
        finally:
            publisher.close()


if __name__ == "__main__":
    unittest.main()
