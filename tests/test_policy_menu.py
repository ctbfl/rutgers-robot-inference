from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from ruri.client.menu import DEFAULT_MENU_ENDPOINT, list_policies


class PolicyMenuTests(unittest.TestCase):
    def test_lists_validated_policies_without_selecting_one(self):
        response = {
            "policies": [
                {
                    "name": "pi05_rtc",
                    "endpoint": "tcp://policy-host:5556",
                    "describe": {
                        "outputs": {"output_chunk_size": 30},
                        "config": {"name": "pi05_h30"},
                    },
                }
            ]
        }
        context = Mock()
        socket = context.socket.return_value

        with (
            patch("ruri.client.menu.zmq.Context", return_value=context),
            patch("ruri.client.menu.send_message") as send,
            patch("ruri.client.menu.recv_message", return_value=response),
        ):
            policies = list_policies()

        socket.connect.assert_called_once_with(DEFAULT_MENU_ENDPOINT)
        send.assert_called_once_with(socket, {"type": "list"})
        self.assertEqual(policies, response["policies"])
        socket.close.assert_called_once_with(linger=0)
        context.term.assert_called_once_with()

    def test_rejects_menu_entry_without_describe_metadata(self):
        context = Mock()
        response = {
            "policies": [
                {"name": "broken", "endpoint": "tcp://host:5555"}
            ]
        }

        with (
            patch("ruri.client.menu.zmq.Context", return_value=context),
            patch("ruri.client.menu.send_message"),
            patch("ruri.client.menu.recv_message", return_value=response),
            self.assertRaisesRegex(ValueError, "describe metadata"),
        ):
            list_policies("tcp://menu:5550")


if __name__ == "__main__":
    unittest.main()
