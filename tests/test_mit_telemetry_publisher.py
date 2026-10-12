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
                follower_joint_effort=qd + 2.0,
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
        np.testing.assert_allclose(packet["follower"]["joint_effort"], qd + 2.0)
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


class EffortRecordingTests(unittest.TestCase):
    """--record-effort keeps every packet, not just the newest."""

    def test_background_recording_keeps_packets_without_polling_and_preserves_latch(self):
        import tempfile
        import time
        from pathlib import Path
        from ruri.client.controllers.single_piper.mit_io import MITTelemetryReceiver

        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1',0))
            port=probe.getsockname()[1]
        address=f'udp://127.0.0.1:{port}'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.jsonl'
            receiver=MITTelemetryReceiver(address)
            receiver.open()
            sender=MITTelemetryPublisher(address)
            try:
                sender.publish(phase='engaged')
                original=receiver.latch_engaged(1.)
                receiver.record_to(path,background=True)
                for _ in range(24):
                    sender.publish(phase='engaged')
                    time.sleep(.01)
                deadline=time.monotonic()+1
                while receiver._latest['sequence']!=25 and time.monotonic()<deadline:
                    time.sleep(.005)
                self.assertEqual(receiver._latest['sequence'],25)
                self.assertEqual(receiver.latched(),original)
                self.assertEqual(receiver.latch_engaged(1.)['sequence'],25)
            finally:
                sender.close();receiver.close()
            packets=[json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual([p['sequence'] for p in packets if 'sequence' in p],list(range(1,26)))

    def test_trace_disk_error_is_reported(self):
        import tempfile
        import time
        from pathlib import Path
        from unittest.mock import MagicMock,patch
        from ruri.client.controllers.single_piper.mit_io import _EffortTrace

        with tempfile.TemporaryDirectory() as tmp:
            stream=MagicMock()
            stream.write.side_effect=OSError('disk full')
            with patch.object(Path,'open',return_value=stream):
                trace=_EffortTrace(Path(tmp)/'trace.jsonl','udp://127.0.0.1:12345')
            deadline=time.monotonic()+1
            while trace.error is None and time.monotonic()<deadline:
                time.sleep(.005)
            with self.assertRaisesRegex(RuntimeError,'log failed'):
                trace.check()
            with self.assertRaisesRegex(RuntimeError,'log failed'):
                trace.close()

    def test_every_drained_packet_reaches_the_log(self):
        import json
        import tempfile
        import time
        from pathlib import Path

        from ruri.client.controllers.single_piper.mit_io import MITTelemetryReceiver

        address = "udp://127.0.0.1:6689"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "effort.jsonl"
            receiver = MITTelemetryReceiver(address)
            receiver.record_to(path)
            receiver.open()
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                for index in range(5):
                    packet = {
                        "protocol": "piper_mit_telemetry",
                        "version": 1,
                        "sequence": index,
                        "phase": "engaged",
                        "follower": {"joint_effort": [float(index)] * 6,
                                     "gripper_force": float(index)},
                    }
                    sender.sendto(json.dumps(packet).encode(), ("127.0.0.1", 6689))
                time.sleep(0.05)
                # One poll, five packets: the receiver keeps only the newest in
                # memory, so this is exactly the loss the log exists to prevent.
                receiver._drain(0.1)
                self.assertEqual(receiver._latest["sequence"], 4)
            finally:
                sender.close()
                receiver.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[0]["kind"], "trace_header")
            self.assertEqual([r["sequence"] for r in records[1:]], [0, 1, 2, 3, 4])
            self.assertEqual(records[1]["follower"]["joint_effort"], [0.0] * 6)
