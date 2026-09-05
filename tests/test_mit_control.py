import json
import socket
import time
import unittest
from unittest.mock import patch

from ruri.client.controllers.single_piper import mit_io


class ControlRequestTests(unittest.TestCase):
    def setUp(self):
        # Ask the OS for a free port without hardcoding a shared service port.
        with patch.object(mit_io, "parse_local_udp", return_value=("127.0.0.1", 0)):
            self.receiver = mit_io.ControlRequestReceiver("unused")
        self.addCleanup(self.receiver.close)
        self.endpoint = self.receiver._socket.getsockname()
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(self.sender.close)

    def send(self, **overrides):
        packet = dict(protocol=mit_io.CONTROL_PROTOCOL, version=1, type="home",
                      session="test", sequence=1, monotonic_ns=time.monotonic_ns())
        packet.update(overrides)
        self.sender.sendto(json.dumps(packet).encode(), self.endpoint)

    def test_sender_round_trip_and_repeated_requests(self):
        sender = mit_io.MITControlSender(f"udp://127.0.0.1:{self.endpoint[1]}")
        self.addCleanup(sender.close)
        sender.request_home()
        self.assertTrue(self.receiver.take_home_request())
        self.assertFalse(self.receiver.take_home_request())
        sender.request_home()
        self.assertTrue(self.receiver.take_home_request())

    def test_replay_and_out_of_order_are_ignored(self):
        self.send(sequence=2)
        self.assertTrue(self.receiver.take_home_request())
        self.send(sequence=2)
        self.send(sequence=1)
        self.assertFalse(self.receiver.take_home_request())
        self.send(session="new", sequence=1)
        self.assertTrue(self.receiver.take_home_request())

    def test_stale_future_and_malformed_packets_are_ignored(self):
        self.sender.sendto(b"not json", self.endpoint)
        self.sender.sendto(b"[]", self.endpoint)
        for overrides in (
            {"monotonic_ns": time.monotonic_ns() - 1_000_000_000},
            {"monotonic_ns": time.monotonic_ns() + 1_000_000_000},
            {"monotonic_ns": None}, {"sequence": True}, {"session": ""},
            {"protocol": mit_io.COMMAND_PROTOCOL}, {"type": "unknown"},
        ):
            self.send(**overrides)
        self.assertFalse(self.receiver.take_home_request())
        self.send()
        self.assertTrue(self.receiver.take_home_request())
