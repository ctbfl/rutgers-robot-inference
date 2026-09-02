"""RURI-owned MIT leader/follower teleoperation controller."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ruri.client.controllers.single_piper import SinglePiperController
from ruri.client.controllers.single_piper.discovery import PiperCanDevice, discover_piper_can
from ruri.client.controllers.single_piper.hardware_registry import (
    find_arm_by_hardware_id,
    find_arm_registration,
)
from ruri.client.controllers.single_piper.mit_io import MITTelemetryReceiver
from ruri.client.controllers.single_piper.mit_process import ManagedMITTeleopProcess

from .config import SinglePiperLeaderFollowerTeleopConfig


class SinglePiperLeaderFollowerTeleopController(SinglePiperController):
    """Own two CAN buses while exposing one follower-arm teleop controller."""

    def __init__(
        self,
        args: Any | None = None,
        *,
        can_discovery: Callable[..., PiperCanDevice] = discover_piper_can,
        telemetry_factory: Callable[[str], Any] = MITTelemetryReceiver,
        worker_factory: Callable[..., Any] = ManagedMITTeleopProcess,
    ):
        super().__init__(
            args,
            can_discovery=can_discovery,
            telemetry_factory=telemetry_factory,
        )
        self.config = SinglePiperLeaderFollowerTeleopConfig.from_args(args)
        self._teleop_worker_factory = worker_factory
        self._leader_can: PiperCanDevice | None = None
        self._follower_can: PiperCanDevice | None = None

    @property
    def is_connected(self) -> bool:
        return self._leader_can is not None and self._follower_can is not None

    def _discover_registered_arm(
        self,
        *,
        interface: str | None,
        hardware_id: str | None,
        side: str,
        role: str,
    ) -> PiperCanDevice:
        expected = hardware_id or find_arm_registration(side, role).can_hardware_id
        device = self._can_discovery(
            candidates=None if interface is None else [interface],
            bitrate=self.config.can_bitrate,
            timeout_s=self.config.can_probe_timeout_s,
            configure=self.config.configure_can,
            hardware_id=expected,
        )
        if device.hardware_id is None:
            raise RuntimeError("CAN discovery did not return a stable hardware ID")
        registered = find_arm_by_hardware_id(device.hardware_id)
        if registered.side != side or registered.role != role:
            raise RuntimeError(
                f"CAN adapter resolved to {registered.side}/{registered.role}, "
                f"expected {side}/{role}"
            )
        return device

    def connect(self) -> None:
        if self.is_connected:
            raise RuntimeError("leader/follower teleop controller is already connected")
        self._leader_can = self._discover_registered_arm(
            interface=self.config.leader_can_interface,
            hardware_id=self.config.leader_can_hardware_id,
            side=self.config.leader_side,
            role=self.config.leader_role,
        )
        try:
            self._follower_can = self._discover_registered_arm(
                interface=self.config.follower_can_interface,
                hardware_id=self.config.follower_can_hardware_id,
                side=self.config.follower_side,
                role=self.config.follower_role,
            )
        except Exception:
            self._leader_can = None
            raise
        if self._leader_can.interface == self._follower_can.interface:
            self._leader_can = self._follower_can = None
            raise RuntimeError("leader and follower resolved to the same CAN interface")

    def start_arm(self) -> None:
        if not self.is_connected:
            raise RuntimeError("connect() must complete before start_arm()")
        if self.arm_started:
            raise RuntimeError("leader/follower teleop is already started")
        telemetry = self._telemetry_factory(self.config.telemetry_address)
        telemetry.open()
        try:
            self._worker = self._teleop_worker_factory(
                self.config,
                self._leader_can.interface,
                self._follower_can.interface,
            )
            self._worker.start()
            telemetry.wait_for_engaged(
                self.config.arm_connect_timeout_s,
                self.config.telemetry_timeout_s,
            )
        except Exception as exc:
            logs = list(getattr(self._worker, "logs", ()))
            if self._worker is not None:
                self._worker.stop(allow_watchdog_recovery=False)
            self._worker = None
            raise RuntimeError(
                f"MIT teleop worker failed to become engaged; logs={logs[-20:]}"
            ) from exc
        finally:
            telemetry.close()
        self._arm_started = True

    def get_observation(self) -> dict[str, Any]:
        raise RuntimeError(
            "The CAN-owning teleop controller does not collect frames; attach the "
            "RURI LeRobot observer on its telemetry address"
        )

    def send_action(self, action: np.ndarray) -> None:
        del action
        raise RuntimeError("Leader motion, not send_action(), drives teleoperation")

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.is_connected,
            "arm_started": self.arm_started,
            "leader_can_interface": (
                None if self._leader_can is None else self._leader_can.interface
            ),
            "follower_can_interface": (
                None if self._follower_can is None else self._follower_can.interface
            ),
            "telemetry_address": self.config.telemetry_address,
        }

    def disconnect(self) -> None:
        super().disconnect()
        self._leader_can = None
        self._follower_can = None

    def stop(self) -> None:
        self.disconnect()
