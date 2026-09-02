from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from ruri.client.utils import inference_client


class InferenceClientTests(unittest.TestCase):
    @staticmethod
    def _transport(responses, *, socket_count=2):
        context = Mock()
        sockets = [Mock() for _ in range(socket_count)]
        context.socket.side_effect = sockets
        return (
            patch(
                "ruri.client.utils.inference_client.zmq.Context",
                return_value=context,
            ),
            patch("ruri.client.utils.inference_client.send_message"),
            patch(
                "ruri.client.utils.inference_client.recv_message",
                side_effect=responses,
            ),
            context,
            sockets,
        )

    def test_connect_waits_for_metadata_and_returns_policy_handle(self):
        args = SimpleNamespace(
            policy_endpoint="tcp://policy.example:5555",
            policy_ready_timeout_s=1.0,
        )
        metadata = {
            "inputs": {"observation.state": {}},
            "outputs": {"output_chunk_size": 30},
        }
        context_patch, send_patch, recv_patch, context, sockets = self._transport(
            [metadata]
        )
        with context_patch, send_patch as send, recv_patch:
            policy = inference_client.connect(args)
            try:
                self.assertTrue(policy.is_connected)
                self.assertEqual(policy.metadata, metadata)
                self.assertEqual(policy.output_chunk_size, 30)
                self.assertIs(policy.args, args)
            finally:
                policy.close()

        self.assertEqual(context.socket.call_count, 2)
        self.assertEqual(send.call_args.args[1], {"type": "metadata"})
        for socket in sockets:
            socket.close.assert_called_once_with(linger=0)
        context.term.assert_called_once_with()

    def test_infer_transports_scheduler_assembled_inputs(self):
        metadata = {"outputs": {"output_chunk_size": 4}}
        response = {"action_chunk": np.zeros((4, 7), dtype=np.float32)}
        context_patch, send_patch, recv_patch, _, _ = self._transport(
            [metadata, response]
        )
        with context_patch, send_patch as send, recv_patch:
            with inference_client.connect("tcp://policy.example:5555") as policy:
                actual = policy.infer(
                    {
                        "observation.state": np.zeros(7, dtype=np.float32),
                        "prompt": "perform task",
                    }
                )

        self.assertIs(actual, response)
        request = send.call_args_list[1].args[1]
        self.assertEqual(request["type"], "infer")
        self.assertEqual(request["prompt"], "perform task")
        self.assertNotIn("context.actions_per_chunk", request)
        np.testing.assert_array_equal(
            request["observation.state"], np.zeros(7, dtype=np.float32)
        )

    def test_server_error_is_raised(self):
        context_patch, send_patch, recv_patch, _, _ = self._transport(
            [
                {"outputs": {"output_chunk_size": 1}},
                {"error": "bad request"},
            ]
        )
        with context_patch, send_patch, recv_patch:
            with inference_client.connect("tcp://policy.example:5555") as policy:
                with self.assertRaisesRegex(RuntimeError, "bad request"):
                    policy.infer({})

    def test_timeout_recreates_req_socket_for_the_next_call(self):
        import zmq

        context_patch, send_patch, recv_patch, context, sockets = self._transport(
            [
                {"outputs": {"output_chunk_size": 1}},
                zmq.Again(),
                {"action_chunk": np.zeros((1, 7), dtype=np.float32)},
            ],
            socket_count=3,
        )
        with context_patch, send_patch, recv_patch:
            with inference_client.connect("tcp://policy.example:5555") as policy:
                with self.assertRaises(TimeoutError):
                    policy.infer({})
                response = policy.infer({})

        self.assertEqual(context.socket.call_count, 3)
        sockets[1].close.assert_called_once_with(linger=0)
        self.assertEqual(response["action_chunk"].shape, (1, 7))

    def test_connect_rejects_metadata_without_output_chunk_size(self):
        context_patch, send_patch, recv_patch, _, _ = self._transport(
            [{"inputs": {}}]
        )
        with context_patch, send_patch, recv_patch:
            with self.assertRaisesRegex(ValueError, "outputs"):
                inference_client.connect("tcp://policy.example:5555")

    def test_connect_rejects_empty_endpoint(self):
        with self.assertRaisesRegex(ValueError, "policy_endpoint"):
            inference_client.connect(" ")


if __name__ == "__main__":
    unittest.main()
