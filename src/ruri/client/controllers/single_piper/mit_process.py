"""Lifecycle manager for the proven Piper MIT control loop."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from ruri.client.controllers.single_piper.config import SinglePiperConfig


logger = logging.getLogger(__name__)


class ManagedMITProcess:
    """Start and supervise the MIT loop as an implementation detail of Controller."""

    def __init__(self, config: SinglePiperConfig, can_interface: str):
        self.config = config
        self.can_interface = can_interface
        self.process: subprocess.Popen[str] | None = None
        self.logs: deque[str] = deque(maxlen=500)
        self._log_thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        if self.running:
            raise RuntimeError("MIT worker is already running")
        python = self.config.python_executable or Path(sys.executable)
        command = [
            str(python),
            "-u",
            "-m",
            "ruri.client.controllers.single_piper.mit.policy_controller",
            "--follower-can",
            self.can_interface,
            "--command-address",
            self.config.command_address,
            "--telemetry-address",
            self.config.telemetry_address,
            "--command-timeout",
            str(self.config.command_timeout_s),
            "--startup-home-skip-threshold-rad",
            str(self.config.startup_home_skip_threshold_rad),
            "--execute",
        ]
        if self.config.diagnostic_log is not None:
            diagnostic_log = self.config.diagnostic_log.expanduser().resolve()
            diagnostic_log.parent.mkdir(parents=True, exist_ok=True)
            command.extend(("--diagnostic-log", str(diagnostic_log)))
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._log_thread = threading.Thread(target=self._copy_logs, daemon=True)
        self._log_thread.start()

    def _copy_logs(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            text = line.rstrip()
            self.logs.append(text)
            logger.info("[Piper MIT] %s", text)

    def wait(self, timeout_s: float) -> int | None:
        if self.process is None:
            return 0
        try:
            return self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None

    def stop(self, *, allow_watchdog_recovery: bool) -> None:
        if self.process is None:
            return
        if allow_watchdog_recovery and self.running:
            # No new command is sent. The worker's command watchdog performs
            # its configured hold/home recovery and exits on its own.
            if self.wait(self.config.worker_shutdown_timeout_s) is not None:
                self.process = None
                return
        if self.running:
            os.killpg(self.process.pid, signal.SIGINT)
            if self.wait(5.0) is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.wait(5.0)
        self.process = None


class ManagedMITTeleopProcess(ManagedMITProcess):
    """Launch the packaged leader/follower MIT worker as the sole CAN owner."""

    def __init__(self, config, leader_can_interface: str, follower_can_interface: str):
        super().__init__(config, follower_can_interface)
        self.leader_can_interface = leader_can_interface
        self.follower_can_interface = follower_can_interface

    def start(self) -> None:
        if self.running:
            raise RuntimeError("MIT teleop worker is already running")
        python = self.config.python_executable or Path(sys.executable)
        command = [
            str(python),
            "-u",
            "-m",
            "ruri.client.controllers.single_piper.mit.leader_follower",
            "--leader-can",
            self.leader_can_interface,
            "--follower-can",
            self.follower_can_interface,
            "--telemetry-address",
            self.config.telemetry_address,
            "--rate",
            str(self.config.mit_rate_hz),
            "--leader-kd",
            str(self.config.leader_kd),
            "--follower-kp",
            str(self.config.follower_kp),
            "--follower-kd",
            str(self.config.follower_kd),
            "--engage-seconds",
            str(self.config.engage_seconds),
            "--start-home-speed",
            str(self.config.start_home_speed),
            "--max-start-gap",
            str(self.config.max_start_gap),
            "--max-track-error",
            str(self.config.max_track_error),
            "--max-joint-speed",
            str(self.config.max_joint_speed),
            "--max-reference-speed",
            str(self.config.max_reference_speed),
            "--feedback-timeout",
            str(self.config.feedback_timeout_s),
            "--seconds",
            str(self.config.seconds),
            "--abort-hold-seconds",
            str(self.config.abort_hold_seconds),
            "--abort-home-speed",
            str(self.config.abort_home_speed),
            "--grip-force",
            str(self.config.grip_force),
            "--grip-base",
            str(self.config.grip_base),
            "--grip-gain",
            str(self.config.grip_gain),
            "--grip-max-force",
            str(self.config.grip_max_force),
            "--execute",
        ]
        if not self.config.show_periodic_status:
            command.append("--quiet-status")
        if not self.config.use_gripper:
            command.append("--no-gripper")
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._log_thread = threading.Thread(target=self._copy_logs, daemon=True)
        self._log_thread.start()
