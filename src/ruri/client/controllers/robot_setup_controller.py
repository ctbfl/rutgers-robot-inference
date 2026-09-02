"""Hardware boundary used by client-side inference schedulers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RobotSetupController(ABC):
    """Minimal interface between robot hardware and an inference scheduler.

    A controller owns hardware lifecycle and the robot-specific coordinate
    convention.  It neither talks to a policy server nor decides when an
    action chunk should be requested, replaced, or aggregated.
    """

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the controller's observation devices are connected."""

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Describe this Controller's target space for Scheduler-side clipping."""
        return None

    @abstractmethod
    def start(self) -> None:
        """Block until all hardware is ready for observation and action."""

    @abstractmethod
    def connect(self) -> None:
        """Connect observation devices without implicitly commanding motion."""

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Return one observation using standard RURI field names."""

    @abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        """Send one already-scheduled target without changing its semantics."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release all resources owned by the controller."""

    @abstractmethod
    def stop(self) -> None:
        """Idempotently stop control and release all owned resources."""

    def __enter__(self) -> "RobotSetupController":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
