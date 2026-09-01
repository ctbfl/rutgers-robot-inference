from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from ruri.client.policies import RemotePolicy


class RemotePolicyTests(unittest.TestCase):
    def test_start_waits_for_metadata_readiness(self):
        args = SimpleNamespace(
            policy_endpoint="tcp://policy.example:5555",
            policy_ready_timeout_s=1.0,
        )
        policy = RemotePolicy(args)
        fake_socket = Mock()

        def open_socket(_timeout_s):
            policy._socket = fake_socket

        policy._open_socket = Mock(side_effect=open_socket)
        policy._request = Mock(return_value={"inputs": {"observation.state": {}}})

        policy.start()

        self.assertTrue(policy.is_started)
        self.assertEqual(policy.metadata, {"inputs": {"observation.state": {}}})
        self.assertEqual(policy._open_socket.call_count, 2)
        policy.stop()

    def test_infer_transports_scheduler_assembled_inputs(self):
        args = SimpleNamespace(
            policy_endpoint="tcp://policy.example:5555",
        )
        policy = RemotePolicy(args)
        fake_socket = object()
        policy._socket = fake_socket
        response = {"action_chunk": np.zeros((4, 7), dtype=np.float32)}

        with (
            patch("ruri.client.policies.remote_policy.send_message") as send,
            patch(
                "ruri.client.policies.remote_policy.recv_message",
                return_value=response,
            ),
        ):
            actual = policy.infer(
                {
                    "observation.state": np.zeros(7, dtype=np.float32),
                    "prompt": "perform task",
                    "context.actions_per_chunk": 4,
                }
            )

        self.assertIs(actual, response)
        request = send.call_args.args[1]
        self.assertIs(send.call_args.args[0], fake_socket)
        self.assertEqual(request["type"], "infer")
        self.assertEqual(request["prompt"], "perform task")
        self.assertEqual(request["context.actions_per_chunk"], 4)
        np.testing.assert_array_equal(
            request["observation.state"], np.zeros(7, dtype=np.float32)
        )

    def test_server_error_is_raised(self):
        args = SimpleNamespace(policy_endpoint="tcp://policy.example:5555")
        policy = RemotePolicy(args)
        policy._socket = object()

        with (
            patch("ruri.client.policies.remote_policy.send_message"),
            patch(
                "ruri.client.policies.remote_policy.recv_message",
                return_value={"error": "bad request"},
            ),
            self.assertRaisesRegex(RuntimeError, "bad request"),
        ):
            policy.infer({})


if __name__ == "__main__":
    unittest.main()
