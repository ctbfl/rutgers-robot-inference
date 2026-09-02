from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

from lerobot.teleoperators import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ruri.client.controllers.single_piper.mit_io import get_shared_mit_telemetry
from ruri.client.controllers.single_piper.normalization import (
    ACTION_KEYS,
    normalize_teleop_target,
)

from .config_piper_mit import PiperMITConfig

logger = logging.getLogger(__name__)


class PiperMIT(Teleoperator):
    """Return the follower target latched by RURI for the current frame."""

    config_class = PiperMITConfig
    name = "piper_mit"

    def __init__(self, config: PiperMITConfig):
        super().__init__(config)
        self.config = config
        self.telemetry = get_shared_mit_telemetry(config.telemetry_address)
        self._connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in ACTION_KEYS}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if not self.telemetry.is_open:
            raise ConnectionError("The RURI Piper observer must connect first")
        self.telemetry.latest_engaged(self.config.telemetry_timeout_s)
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> dict[str, Any]:
        values = normalize_teleop_target(self.telemetry.latched())
        return {
            key: float(value)
            for key, value in zip(ACTION_KEYS, values, strict=True)
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        self._connected = False
        logger.info("%s disconnected", self)
