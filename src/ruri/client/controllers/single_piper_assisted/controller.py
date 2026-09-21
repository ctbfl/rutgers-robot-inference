"""Assisted hardware boundary with one camera owner and one telemetry receiver."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ruri.client.controllers.robot_setup_controller import RobotSetupController
from ruri.client.controllers.single_piper import SinglePiperController
from ruri.client.controllers.single_piper.home import home_actions
from ruri.client.controllers.single_piper.mit_process import ManagedMITProcess
from ruri.client.controllers.single_piper.normalization import (
    ACTION_LOWER,
    ACTION_UPPER,
    denormalize_action,
    normalize_pose,
    normalize_teleop_target,
    recording_observation,
)
from ruri.client.controllers.single_piper_leader_follower_teleop import (
    SinglePiperLeaderFollowerTeleopController,
)

from .config import SinglePiperAssistedConfig
from .control import synchronized_bounds
from .io import AssistedIO


class ManagedAssistedProcess(ManagedMITProcess):
    def __init__(self, config, session, leader_can=None, follower_can=None):
        super().__init__(config, follower_can)
        self.session, self.leader_can = session, leader_can

    def start(self):
        if self.running:
            raise RuntimeError("assisted worker is already running")
        config_json = json.dumps(
            asdict(self.config), default=lambda v: str(v) if isinstance(v, Path) else v
        )
        command = [
            str(self.config.python_executable or sys.executable),
            "-u",
            "-m",
            "ruri.client.controllers.single_piper.mit.assisted",
            "--config-json",
            config_json,
            "--worker-session",
            self.session,
        ]
        if not self.config.simulate:
            command += [
                "--leader-can",
                self.leader_can,
                "--execute",
            ]
        if not self.config.simulate and not self.config.leader_only:
            command += ["--follower-can", self.can_interface]
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

    def stop(self, *, allow_watchdog_recovery=False):
        process = self.process
        allow_watchdog_recovery = allow_watchdog_recovery or any(
            line.startswith("ASSISTED RECOVERY:") for line in self.logs
        )
        try:
            # A runtime FAULT owns its MIT recovery. Wait for it to finish
            # before sending a shutdown signal that would cancel the home.
            super().stop(allow_watchdog_recovery=allow_watchdog_recovery)
        finally:
            if self._log_thread is not None:
                self._log_thread.join(timeout=1.0)
            if process is not None and process.stdout is not None:
                process.stdout.close()


class SinglePiperAssistedController(RobotSetupController):
    def __init__(self, args=None):
        self.config = SinglePiperAssistedConfig.from_args(args)
        self._camera_owner = None
        self._discovery = None
        self._worker = None
        self._io = None
        self._connected = False
        self._session = uuid.uuid4().hex
        self._observation_sequence = 0
        self._policy_epoch = None
        self.last_command_id = None
        self._camera_clock_offset = time.monotonic_ns() - time.perf_counter_ns()

    @property
    def is_connected(self):
        return self._connected

    @property
    def action_bounds(self):
        if self._io is not None:
            state = self._io.snapshot()["control"]
            return np.asarray(state["action_lower"], dtype=np.float32), np.asarray(
                state["action_upper"], dtype=np.float32
            )
        low, high = synchronized_bounds(self.config)
        return normalize_pose(low[:6], low[6]), normalize_pose(high[:6], high[6])

    def connect(self):
        if self.is_connected:
            raise RuntimeError("assisted controller is already connected")
        try:
            if not self.config.simulate:
                self._discovery = SinglePiperLeaderFollowerTeleopController(self.config)
                if self.config.leader_only:
                    c = self.config
                    self._discovery._leader_can = self._discovery._discover_registered_arm(
                        interface=c.leader_can_interface, hardware_id=c.leader_can_hardware_id,
                        side=c.leader_side, role=c.leader_role,
                    )
                else:
                    self._discovery.connect()
                if self.config.camera_enabled:
                    self._camera_owner = SinglePiperController(self.config)
                    self._camera_owner.connect_cameras()
            self._connected = True
        except BaseException:
            self.disconnect()
            raise

    def start(self):
        if self._worker is not None:
            raise RuntimeError("assisted worker already started")
        try:
            # Claim telemetry before configuring CAN or opening cameras. An
            # existing client on this control channel must fail startup early.
            self._io = AssistedIO(self.config, self._session)
            if not self.is_connected:
                self.connect()
            can = {} if self._discovery is None else self._discovery.status()
            self._worker = ManagedAssistedProcess(
                self.config,
                self._session,
                can.get("leader_can_interface"),
                can.get("follower_can_interface"),
            )
            self._worker.start()
            deadline = time.monotonic() + self.config.arm_connect_timeout_s
            while True:
                if not self._worker.running:
                    raise RuntimeError(
                        f"assisted worker exited: {list(self._worker.logs)[-20:]}"
                    )
                try:
                    packet = self._io.snapshot(wait_s=0.1)
                    if packet["phase"] == "engaged":
                        break
                    raise RuntimeError(f"worker startup failed: {packet.get('reason')}")
                except TimeoutError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"assisted worker not ready: {list(self._worker.logs)[-20:]}"
                        )
        except BaseException:
            self.disconnect()
            raise

    def status(self):
        if self._io is None:
            return {"connected": self.is_connected, "mode": "STOPPED"}
        packet = self._io.snapshot()
        return {
            **packet["control"],
            "connected": self.is_connected,
            "phase": packet["phase"],
            "target": normalize_teleop_target(packet),
            "monotonic_ns": packet["monotonic_ns"],
            "sequence": packet["sequence"],
        }

    def request_mode(self, mode):
        if self._io is None:
            raise RuntimeError("worker is not started")
        result = self._io.request_mode(mode)
        if mode == "POLICY":
            self._policy_epoch = result["epoch"]
        return result

    def send_action(
        self, action, *, reference_revision=None, observation_ns=None, model_action=True
    ):
        if self._io is None or self._policy_epoch is None:
            raise RuntimeError("POLICY must be explicitly armed before sending targets")
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("action must contain seven finite values")
        q, grip = denormalize_action(np.clip(values, ACTION_LOWER, ACTION_UPPER))
        if reference_revision is None:
            reference_revision = self.status()["reference_revision"]
        self.last_command_id = self._io.send(
            "target",
            epoch=self._policy_epoch,
            target=np.r_[q, grip].tolist(),
            reference_revision=reference_revision,
            observation_ns=time.monotonic_ns()
            if observation_ns is None
            else observation_ns,
            model_action=model_action,
        )

    def simulate_leader(self, normalized_pose, *, duration_s=0.0):
        if not self.config.simulate or self._io is None:
            raise RuntimeError("leader injection is available only in simulation")
        q, grip = denormalize_action(normalized_pose)
        self._io.send(
            "simulation_leader", target=np.r_[q, grip].tolist(), duration_s=duration_s
        )

    def iter_home_actions(self, *, control_hz, speed_rad_s=1.0):
        """Shared home references; completion checks both physical arms."""
        if self._io is None or self._policy_epoch is None:
            raise RuntimeError("Arm shared control before homing")

        def leader_settled(packet):
            q = np.asarray(packet["leader"]["q"])
            qd = np.asarray(packet["leader"]["qd"])
            return bool(
                q.shape == qd.shape == (6,)
                and np.all(np.isfinite(q))
                and np.all(np.isfinite(qd))
                and np.max(np.abs(q - self.config.leader_offset_rad)) <= 0.05
                and np.max(np.abs(qd)) <= 0.05
            )

        yield from home_actions(
            lambda: self._controlled_packet(self._io.snapshot()),
            control_hz=control_hz,
            speed_rad_s=speed_rad_s,
            extra_settled=leader_settled,
        )

    def _images(self):
        images, times = {}, {}
        if not self.config.camera_enabled:
            return images, times
        for name in ("top", "wrist"):
            if self.config.simulate:
                # Changing RGB buffers exercise real image storage/encoding.
                stamp = time.monotonic_ns()
                image = np.zeros(
                    (self.config.camera_height, self.config.camera_width, 3),
                    dtype=np.uint8,
                )
                image[..., 0] = self._observation_sequence % 256
                image[..., 1 if name == "top" else 2] = np.arange(
                    image.shape[1], dtype=np.uint8
                )
            else:
                camera = self._camera_owner.cameras[name]
                # LeRobot 0.6.1 publishes these together under frame_lock. Its
                # stamp is host post-receive perf_counter, not sensor exposure.
                with camera.frame_lock:
                    frame, timestamp = (
                        camera.latest_color_frame,
                        camera.latest_timestamp,
                    )
                    if frame is None or timestamp is None:
                        raise TimeoutError(f"{name} has no image")
                    image = SinglePiperController._image(frame, name)
                    stamp = int(timestamp * 1e9) + self._camera_clock_offset
                if (time.monotonic_ns() - stamp) / 1e6 > self.config.camera_timeout_ms:
                    raise TimeoutError(f"{name} image is stale")
            image.setflags(write=False)
            images[f"observation.images.{name}"] = image
            times[name] = {
                "host_received_ns": stamp,
                "frame_id": f"{name}:{stamp}",
                "clock": "host_monotonic",
                "device_capture_ns": None,
            }
        return images, times

    def _controlled_packet(self, packet):
        if not self.config.leader_only:
            return packet
        # Adapt the legacy observation/home helpers locally. Wire telemetry
        # leaves the absent follower null; the real observation is the leader.
        return {**packet, "follower": {
            **packet["leader"],
            "joint_effort": packet["control"]["leader_joint_effort"],
            "gripper_force": packet["control"]["leader_gripper_force"],
        }}

    def get_sample(self):
        if self._io is None:
            raise RuntimeError("worker is not started")
        self._observation_sequence += 1
        images, camera_times = self._images()
        packet = self._io.snapshot()
        if packet["phase"] != "engaged":
            raise RuntimeError(
                f"worker phase={packet['phase']}: {packet.get('reason')}"
            )
        observation = {**recording_observation(self._controlled_packet(packet)), **images}
        for value in observation.values():
            value.setflags(write=False)
        return {
            "observation_id": f"{self._session}:{self._observation_sequence}",
            "monotonic_ns": time.monotonic_ns(),
            "observation": observation,
            "action": normalize_teleop_target(packet),
            "telemetry": packet,
            "cameras": camera_times,
            "trace": self._io.drain_trace(),
        }

    def get_observation(self):
        return self.get_sample()["observation"]

    def disconnect(self):
        recovering = False
        try:
            if self._io is not None:
                try:
                    recovering = self._io.snapshot()["phase"] in (
                        "recovering",
                        "aborted",
                    )
                    if not recovering:
                        self._io.request_mode("HOLD", timeout_s=0.2)
                except (RuntimeError, TimeoutError, OSError):
                    pass
                # The worker may have faulted while the HOLD acknowledgement
                # was pending. Re-read before deciding whether to signal it.
                try:
                    recovering |= self._io.snapshot()["phase"] in (
                        "recovering",
                        "aborted",
                    )
                except (RuntimeError, TimeoutError, OSError):
                    pass
            if self._worker is not None:
                self._worker.stop(allow_watchdog_recovery=recovering)
        finally:
            if self._io is not None:
                self._io.close()
            if self._camera_owner is not None:
                self._camera_owner.disconnect()
            if self._discovery is not None:
                self._discovery.disconnect()
            self._worker = self._io = self._camera_owner = self._discovery = None
            self._connected = False
            self._policy_epoch = None

    def stop(self):
        self.disconnect()
