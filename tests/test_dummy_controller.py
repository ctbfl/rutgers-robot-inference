from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ruri.client.controllers import DummyController
from ruri.client.schedulers import BlockingScheduler


class DummyControllerTests(unittest.TestCase):
    def test_lifecycle_observation_and_action_feedback(self):
        args = SimpleNamespace(
            dummy_state_dim=3,
            dummy_image_height=4,
            dummy_image_width=6,
            dummy_image_value=12,
        )
        controller = DummyController(args)

        controller.start()
        observation = controller.get_observation()
        accepted = controller.send_action([1.0, 2.0, 3.0])
        next_observation = controller.get_observation()

        self.assertEqual(observation["observation.state"].shape, (3,))
        self.assertEqual(observation["observation.images.top"].shape, (4, 6, 3))
        self.assertEqual(int(observation["observation.images.wrist"][0, 0, 0]), 12)
        np.testing.assert_array_equal(accepted, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            next_observation["observation.state"], accepted
        )
        self.assertEqual(controller.action_history.shape, (1, 3))

        controller.stop()
        self.assertFalse(controller.is_connected)

    def test_scheduler_flows_two_chunks_through_dummy_controller(self):
        args = SimpleNamespace(
            control_hz=30.0,
            max_chunks=2,
            prompt="dummy task",
            actions_per_chunk=2,
            dummy_state_dim=7,
            dummy_image_height=4,
            dummy_image_width=6,
        )
        created = {}

        def controller_factory(received_args):
            controller = DummyController(received_args)
            created["controller"] = controller
            return controller

        class FakePolicy:
            def __init__(self, received_args):
                self.args = received_args
                self.calls = 0

            def start(self):
                pass

            def infer(self, inputs):
                self.calls += 1
                self.assert_inputs(inputs)
                if self.calls == 2:
                    np.testing.assert_array_equal(
                        inputs["observation.state"],
                        np.full(7, 2.0, dtype=np.float32),
                    )
                return {
                    "action_chunk": np.asarray(
                        [
                            np.full(7, self.calls, dtype=np.float32),
                            np.full(7, self.calls + 1, dtype=np.float32),
                        ]
                    )
                }

            def assert_inputs(self, inputs):
                if self.args is not args:
                    raise AssertionError("Policy did not receive shared args")
                if inputs["prompt"] != args.prompt:
                    raise AssertionError("Prompt did not flow through Scheduler")
                if inputs["context.actions_per_chunk"] != 2:
                    raise AssertionError("Chunk context did not flow through Scheduler")

            def stop(self):
                pass

        with patch("ruri.client.schedulers.blocking.time.sleep"):
            BlockingScheduler().run(
                controller_factory,
                FakePolicy,
                args=args,
            )

        controller = created["controller"]
        self.assertEqual(controller.observation_count, 2)
        self.assertEqual(controller.action_history.shape, (4, 7))
        np.testing.assert_array_equal(
            controller.action_history[:, 0],
            np.asarray([1.0, 2.0, 2.0, 3.0]),
        )
        np.testing.assert_array_equal(
            controller.state,
            np.full(7, 3.0, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
