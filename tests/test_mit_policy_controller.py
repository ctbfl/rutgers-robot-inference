from __future__ import annotations

import json
import socket
import time
import unittest

import numpy as np
from ruri.client.controllers.single_piper.mit_io import (
    MITCommandSender as PolicyCommandSender,
)

from ruri.client.controllers.single_piper.mit.policy_controller import (
    PolicyCommandReceiver,
    startup_home_is_within_threshold,
    write_diagnostic_log,
)


def free_udp_address() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"udp://127.0.0.1:{port}"


class CommandProtocolTests(unittest.TestCase):
    def test_startup_home_skip_uses_max_absolute_joint_error(self):
        self.assertTrue(
            startup_home_is_within_threshold(
                [0.049, -0.05, 0.0, 0.01, -0.02, 0.03], 0.05
            )
        )
        self.assertFalse(
            startup_home_is_within_threshold(
                [0.0, 0.0, 0.051, 0.0, 0.0, 0.0], 0.05
            )
        )

    def test_sender_receiver_round_trip_and_sequence(self):
        address = free_udp_address()
        receiver = PolicyCommandReceiver(address, max_age_s=0.3)
        sender = PolicyCommandSender(address)
        try:
            sender.send([0.1, 0.2, -0.3, 0.4, -0.5, 0.6], 0.04)
            time.sleep(0.01)
            target = receiver.receive_latest()
            self.assertIsNotNone(target)
            self.assertEqual(target.sequence, 1)
            np.testing.assert_allclose(target.q, [0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
            self.assertAlmostEqual(target.gripper_width, 0.04)
        finally:
            sender.close()
            receiver.close()

    def test_diagnostic_log_contains_header_and_samples(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.jsonl"
            write_diagnostic_log(
                path,
                [{"kind": "controller_sample", "tracking_error": 0.1}],
                {"outcome": "aborted: test", "rate_hz": 100.0},
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(records[0]["kind"], "trace_header")
        self.assertEqual(records[0]["outcome"], "aborted: test")
        self.assertEqual(records[1]["tracking_error"], 0.1)

    def test_receiver_coalesces_pending_commands_to_latest(self):
        address = free_udp_address()
        receiver = PolicyCommandReceiver(address, max_age_s=0.3)
        sender = PolicyCommandSender(address)
        try:
            base = np.array([0.01, 0.02, -0.03, 0.04, -0.05, 0.06])
            sender.send(base, 0.01)
            sender.send(2.0 * base, 0.02)
            sender.send(3.0 * base, 0.03)
            time.sleep(0.01)
            target = receiver.receive_latest()
            self.assertIsNotNone(target)
            self.assertEqual(target.sequence, 3)
            np.testing.assert_allclose(target.q, 3.0 * base)
            self.assertAlmostEqual(target.gripper_width, 0.03)
        finally:
            sender.close()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
