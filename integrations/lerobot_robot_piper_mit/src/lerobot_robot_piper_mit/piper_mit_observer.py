"""RURI Piper observer adapter for LeRobot 0.6.1."""

from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ruri.client.controllers.single_piper import SinglePiperController
from ruri.client.controllers.single_piper.mit_io import get_shared_mit_telemetry
from ruri.client.controllers.single_piper.normalization import ACTION_KEYS, EFFORT_KEYS

from .config_piper_mit_observer import PiperMITObserverConfig

logger = logging.getLogger(__name__)


def _named_vector(keys: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in zip(keys, np.asarray(values), strict=True)
    }


class PiperMITObserver(Robot):
    """LeRobot-shaped view of RURI camera and MIT telemetry observations."""

    config_class = PiperMITObserverConfig
    name = "piper_follower"

    def __init__(self, config: PiperMITObserverConfig):
        super().__init__(config)
        self.config = config
        self.controller = SinglePiperController(
            config,
            telemetry_factory=get_shared_mit_telemetry,
        )
        # LeRobot creates the dataset before connecting the robot and uses this
        # mapping to size its image-writer pool. Keep both slots present before
        # hardware discovery, then replace the placeholders after connect().
        self.cameras: dict[str, Any | None] = {"top": None, "hand": None}
        self._latched_action: dict[str, float] | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {
            **{key: float for key in ACTION_KEYS},
            **{key: float for key in EFFORT_KEYS},
            "top": (self.config.camera_height, self.config.camera_width, 3),
            "hand": (self.config.camera_height, self.config.camera_width, 3),
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in ACTION_KEYS}

    @property
    def is_connected(self) -> bool:
        return (
            self.controller.is_connected
            and self.controller.teleop_observer_attached
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.controller.connect_teleop_observer()
        controller_cameras = self.controller.cameras
        self.cameras["top"] = controller_cameras["top"]
        self.cameras["hand"] = controller_cameras["wrist"]
        logger.info("%s connected through RURI without opening CAN", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> dict[str, Any]:
        observation, target = self.controller.get_teleop_recording_sample()
        state = _named_vector(ACTION_KEYS, observation["observation.state"])
        effort = _named_vector(
            EFFORT_KEYS,
            observation["observation.joint_effort"],
        )
        self._latched_action = _named_vector(ACTION_KEYS, target)
        return {
            **state,
            **effort,
            "top": observation["observation.images.top"],
            "hand": observation["observation.images.wrist"],
        }

    @check_if_not_connected
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Validate/no-op, then return the target required by LeRobot's API."""
        if self._latched_action is None:
            raise RuntimeError("get_observation() must latch this frame first")
        if set(action) != set(self._latched_action):
            raise ValueError(
                f"expected action keys {list(self._latched_action)}, got {list(action)}"
            )
        for key, expected in self._latched_action.items():
            value = float(action[key])
            if not math.isfinite(value):
                raise ValueError(f"action {key} is not finite")
            if abs(value - expected) > self.config.action_tolerance:
                raise ValueError(
                    f"action {key}={value:.6f} does not match this frame's "
                    f"RURI MIT target {expected:.6f}"
                )
        return dict(self._latched_action)

    @check_if_not_connected
    def request_home(self) -> bool:
        """Ask the teleop worker to send the arms home between episodes.

        This adapter never opens CAN, so it cannot move anything itself; the
        request travels to the worker that does, which is free to refuse it.
        """
        return self.controller.request_home()

    @check_if_not_connected
    def disconnect(self) -> None:
        self.controller.disconnect()
        self.cameras.update(top=None, hand=None)
        self._latched_action = None
        logger.info("%s disconnected from RURI telemetry and cameras", self)
