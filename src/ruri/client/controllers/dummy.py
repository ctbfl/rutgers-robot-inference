"""In-memory Controller for testing the complete client inference flow."""

from __future__ import annotations

from typing import Any

import numpy as np

from ruri.client._args import get_arg
from ruri.client.controllers.robot_setup_controller import RobotSetupController


class DummyController(RobotSetupController):
    """Produce synthetic observations and record every accepted action."""

    def __init__(self, args: Any):
        self.args = args
        self.state_dim = int(get_arg(args, "dummy_state_dim", 7))
        self.image_height = int(get_arg(args, "dummy_image_height", 480))
        self.image_width = int(get_arg(args, "dummy_image_width", 640))
        image_value = int(get_arg(args, "dummy_image_value", 0))

        if self.state_dim <= 0:
            raise ValueError("dummy_state_dim must be positive")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("Dummy image height and width must be positive")
        if not 0 <= image_value <= 255:
            raise ValueError("dummy_image_value must be in [0, 255]")

        initial_state = get_arg(args, "dummy_initial_state", None)
        if initial_state is None:
            self._state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            self._state = self._validate_action(initial_state)

        image_shape = (self.image_height, self.image_width, 3)
        self._top_image = np.full(image_shape, image_value, dtype=np.uint8)
        self._wrist_image = np.full(image_shape, image_value, dtype=np.uint8)
        self._actions: list[np.ndarray] = []
        self._connected = False
        self._started = False
        self._observation_count = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def observation_count(self) -> int:
        return self._observation_count

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def action_history(self) -> np.ndarray:
        if not self._actions:
            return np.empty((0, self.state_dim), dtype=np.float32)
        return np.stack(self._actions).copy()

    def connect(self) -> None:
        if self._connected:
            raise RuntimeError("DummyController is already connected")
        self._connected = True

    def start(self) -> None:
        if self._started:
            raise RuntimeError("DummyController is already started")
        if not self._connected:
            self.connect()
        self._started = True

    def get_observation(self) -> dict[str, np.ndarray]:
        self._require_started()
        self._observation_count += 1
        return {
            "observation.state": self._state.copy(),
            "observation.images.top": self._top_image.copy(),
            "observation.images.wrist": self._wrist_image.copy(),
        }

    def send_action(self, action: np.ndarray) -> np.ndarray:
        self._require_started()
        accepted = self._validate_action(action)
        self._state = accepted
        self._actions.append(accepted.copy())
        return accepted.copy()

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.is_connected,
            "started": self.is_started,
            "observation_count": self.observation_count,
            "action_count": len(self._actions),
            "state": self.state,
        }

    def disconnect(self) -> None:
        self._started = False
        self._connected = False

    def stop(self) -> None:
        self.disconnect()

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("DummyController.start() must complete first")

    def _validate_action(self, action: Any) -> np.ndarray:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (self.state_dim,):
            raise ValueError(
                f"Dummy action must have shape ({self.state_dim},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Dummy action contains NaN or infinity")
        return values.copy()
