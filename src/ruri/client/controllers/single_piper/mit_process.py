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


def resolve_teleop_root(configured: Path | None) -> Path:
    candidates = []
    if configured is not None:
        candidates.append(configured)
    if root := os.environ.get("RURI_PIPER_TELEOP_ROOT"):
        candidates.append(Path(root))
    candidates.append(Path(__file__).resolve().parents[6] / "piper_teleop_agx")
    for candidate in candidates:
        if (candidate / "mit_policy_controller.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find piper_teleop_agx/mit_policy_controller.py; set "
        "SinglePiperConfig.teleop_root or RURI_PIPER_TELEOP_ROOT"
    )


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
        root = resolve_teleop_root(self.config.teleop_root)
        python = self.config.python_executable or Path(sys.executable)
        command = [
            str(python),
            "-u",
            str(root / "mit_policy_controller.py"),
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
            cwd=root,
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
