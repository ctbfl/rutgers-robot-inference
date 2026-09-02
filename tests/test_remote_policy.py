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
        metadata = {
            "inputs": {"observation.state": {}},
            "outputs": {"output_chunk_size": 30},
        }
        policy._request = Mock(return_value=metadata)

        policy.start()

        self.assertTrue(policy.is_started)
        self.assertEqual(policy.metadata, metadata)
        self.assertEqual(policy.output_chunk_size, 30)
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
                }
            )

        self.assertIs(actual, response)
        request = send.call_args.args[1]
        self.assertIs(send.call_args.args[0], fake_socket)
        self.assertEqual(request["type"], "infer")
        self.assertEqual(request["prompt"], "perform task")
        self.assertNotIn("context.actions_per_chunk", request)
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

    def test_start_rejects_metadata_without_output_chunk_size(self):
        args = SimpleNamespace(
            policy_endpoint="tcp://policy.example:5555",
            policy_ready_timeout_s=1.0,
        )
        policy = RemotePolicy(args)
        fake_socket = Mock()

        def open_socket(_timeout_s):
            policy._socket = fake_socket

        policy._open_socket = Mock(side_effect=open_socket)
        policy._request = Mock(return_value={"inputs": {}})

        with self.assertRaisesRegex(ValueError, "outputs"):
            policy.start()

        self.assertFalse(policy.is_started)
        self.assertIsNone(policy.metadata)

    def test_output_chunk_size_must_be_a_positive_integer(self):
        invalid_values = (None, True, 0, -1, 3.5, "30")
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                RemotePolicy._metadata_output_chunk_size(
                    {"outputs": {"output_chunk_size": value}}
                )


if __name__ == "__main__":
    unittest.main()
