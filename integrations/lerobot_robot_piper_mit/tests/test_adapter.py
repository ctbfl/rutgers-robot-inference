from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from lerobot_robot_piper_mit import PiperMITObserver, PiperMITObserverConfig


class FakeRURIController:
    def __init__(self, *_args, **_kwargs):
        self.is_connected = False
        self.teleop_observer_attached = False
        self.cameras = {"top": object(), "wrist": object()}

    def connect_teleop_observer(self):
        self.is_connected = True
        self.teleop_observer_attached = True

    def get_teleop_recording_sample(self):
        return (
            {
                "observation.state": np.arange(7, dtype=np.float32),
                "observation.joint_effort": np.arange(7, dtype=np.float32) + 10,
                "observation.images.top": np.full((2, 3, 3), 10, np.uint8),
                "observation.images.wrist": np.full((2, 3, 3), 20, np.uint8),
            },
            np.arange(7, dtype=np.float32) + 1,
        )

    def disconnect(self):
        self.is_connected = False
        self.teleop_observer_attached = False


class PiperMITObserverTests(unittest.TestCase):
    def test_adapter_uses_ruri_sample_and_validates_same_frame_target(self):
        with patch(
            "lerobot_robot_piper_mit.piper_mit_observer.SinglePiperController",
            FakeRURIController,
        ):
            observer = PiperMITObserver(
                PiperMITObserverConfig(
                    id="test",
                    camera_width=3,
                    camera_height=2,
                )
            )
        self.assertEqual(list(observer.cameras), ["top", "hand"])
        self.assertEqual(list(observer.cameras.values()), [None, None])
        observer.connect()
        self.assertIs(observer.cameras["top"], observer.controller.cameras["top"])
        self.assertIs(observer.cameras["hand"], observer.controller.cameras["wrist"])
        observation = observer.get_observation()
        action = {
            key: float(index + 1)
            for index, key in enumerate(observer.action_features)
        }

        returned = observer.send_action(action)

        self.assertEqual(set(observation), set(observer.observation_features))
        np.testing.assert_array_equal(
            [observation[f"joint{index}.effort"] for index in range(1, 7)]
            + [observation["gripper.effort"]],
            np.arange(7, dtype=np.float32) + 10,
        )
        np.testing.assert_array_equal(observation["top"], 10)
        np.testing.assert_array_equal(observation["hand"], 20)
        self.assertEqual(returned, action)
        observer.disconnect()
        self.assertEqual(list(observer.cameras.values()), [None, None])


if __name__ == "__main__":
    unittest.main()
