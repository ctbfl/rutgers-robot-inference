"""Controller for one Piper arm, one head D435, and one wrist D415."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ruri.client.controllers.robot_setup_controller import RobotSetupController
from ruri.client.controllers.single_piper.config import SinglePiperConfig
from ruri.client.controllers.single_piper.discovery import (
    PiperCanDevice,
    RealSenseDevices,
    discover_piper_can,
    discover_realsense_devices,
)
from ruri.client.controllers.single_piper.hardware_registry import (
    ArmHardwareRegistration,
    find_arm_by_hardware_id,
    find_arm_registration,
)
from ruri.client.controllers.single_piper.mit_io import (
    MITCommandSender,
    MITTelemetryReceiver,
)
from ruri.client.controllers.single_piper.mit_process import ManagedMITProcess
from ruri.client.controllers.single_piper.normalization import (
    denormalize_action,
    normalize_telemetry,
    validate_action,
)


logger = logging.getLogger(__name__)


_REALSENSE_FACTORY_DEFAULT_OPTIONS = (
    "brightness",
    "contrast",
    "gamma",
    "hue",
    "saturation",
    "sharpness",
    "exposure",
    "gain",
    "white_balance",
    "backlight_compensation",
)


def _set_realsense_option(sensor: Any, rs: Any, name: str, value: float) -> float:
    option = getattr(rs.option, name, None)
    if option is None or not sensor.supports(option):
        raise RuntimeError(f"RealSense color sensor does not support {name!r}")
    option_range = sensor.get_option_range(option)
    if not option_range.min <= value <= option_range.max:
        raise ValueError(
            f"RealSense {name}={value:g} is outside "
            f"[{option_range.min:g}, {option_range.max:g}]"
        )
    sensor.set_option(option, float(value))
    actual = float(sensor.get_option(option))
    tolerance = max(abs(float(option_range.step)) / 2.0, 1e-6)
    if abs(actual - value) > tolerance:
        raise RuntimeError(
            f"RealSense rejected {name}={value:g}; read back {actual:g}"
        )
    return actual


def _configure_realsense_camera(
    camera: Any,
    role: str,
    config: SinglePiperConfig,
) -> Mapping[str, float]:
    """Reproduce the UVC state used for the training images.

    Dataset collection and inference both keep LeRobot's RGB async-read path.
    The original head-camera brightness setup lived outside the recorder and
    persisted in the camera across processes, so set it explicitly here rather
    than depending on whichever program last opened either device.
    """
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError(
            "Single Piper camera controls require pyrealsense2"
        ) from exc

    profile = getattr(camera, "rs_profile", None)
    if profile is None:
        raise RuntimeError("LeRobot RealSense camera has no active rs_profile")
    sensor = profile.get_device().first_color_sensor()

    # Manual writes are ignored while the corresponding auto control is on.
    for name in ("enable_auto_exposure", "enable_auto_white_balance"):
        _set_realsense_option(sensor, rs, name, 0.0)
    for name in _REALSENSE_FACTORY_DEFAULT_OPTIONS:
        option = getattr(rs.option, name, None)
        if option is not None and sensor.supports(option):
            default = float(sensor.get_option_range(option).default)
            _set_realsense_option(sensor, rs, name, default)

    if role == "head":
        exposure = config.head_camera_exposure
        gain = config.head_camera_gain
        brightness = config.head_camera_brightness
    elif role == "wrist":
        exposure = config.wrist_camera_exposure
        gain = config.wrist_camera_gain
        brightness = config.wrist_camera_brightness
    else:
        raise ValueError(f"unknown Piper camera role {role!r}")

    readback = {
        "enable_auto_exposure": _set_realsense_option(
            sensor, rs, "enable_auto_exposure", 0.0
        ),
        "exposure": _set_realsense_option(sensor, rs, "exposure", exposure),
        "gain": _set_realsense_option(sensor, rs, "gain", gain),
        "brightness": _set_realsense_option(sensor, rs, "brightness", brightness),
        "enable_auto_white_balance": _set_realsense_option(
            sensor, rs, "enable_auto_white_balance", 1.0
        ),
    }

    # Match the capture setup's 30-frame discard after changing UVC controls.
    for _ in range(config.camera_controls_warmup_frames):
        camera.async_read(timeout_ms=config.camera_timeout_ms)
    white_balance = getattr(rs.option, "white_balance", None)
    if white_balance is not None and sensor.supports(white_balance):
        readback["white_balance"] = float(sensor.get_option(white_balance))
    return readback


def _skip_camera_configuration(
    _camera: Any,
    _role: str,
    _config: SinglePiperConfig,
) -> Mapping[str, float]:
    return {}


class SinglePiperController(RobotSetupController):
    """Hardware controller exposing standard RURI observations.

    ``connect`` discovers hardware and starts both camera streams, but never
    enables or moves the arm. ``start_arm`` is the explicit motion gate: it
    launches the persistent MIT child process, which becomes the sole CAN
    socket owner and performs its existing startup/home safety sequence.
    """

    def __init__(
        self,
        args: Any | None = None,
        *,
        camera_discovery: Callable[[], RealSenseDevices] = discover_realsense_devices,
        can_discovery: Callable[..., PiperCanDevice] = discover_piper_can,
        camera_factory: Callable[[str, SinglePiperConfig], Any] | None = None,
        camera_configurator: (
            Callable[[Any, str, SinglePiperConfig], Mapping[str, float]] | None
        ) = None,
        telemetry_factory: Callable[[str], Any] = MITTelemetryReceiver,
        command_factory: Callable[[str], Any] = MITCommandSender,
        worker_factory: Callable[[SinglePiperConfig, str], Any] = ManagedMITProcess,
    ):
        # Scheduler, Controller, and Policy all receive the same complete args
        # object.  The Controller selects its own fields internally.  Passing a
        # SinglePiperConfig directly remains supported for focused use/tests.
        self.args = args
        self.config = SinglePiperConfig.from_args(args)
        self._camera_discovery = camera_discovery
        self._can_discovery = can_discovery
        using_default_camera_factory = camera_factory is None
        self._camera_factory = camera_factory or self._default_camera_factory
        self._camera_configurator = camera_configurator or (
            _configure_realsense_camera
            if using_default_camera_factory
            else _skip_camera_configuration
        )
        self._telemetry_factory = telemetry_factory
        self._command_factory = command_factory
        self._worker_factory = worker_factory

        self._head: Any | None = None
        self._wrist: Any | None = None
        self._telemetry: Any | None = None
        self._commands: Any | None = None
        self._worker: Any | None = None
        self._devices: RealSenseDevices | None = None
        self._can: PiperCanDevice | None = None
        self._arm: ArmHardwareRegistration | None = None
        self._arm_started = False
        self._sent_action = False
        self._camera_controls: dict[str, dict[str, float]] = {}

    @staticmethod
    def _default_camera_factory(serial: str, config: SinglePiperConfig) -> Any:
        try:
            from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
        except ImportError as exc:
            raise ImportError(
                "Single Piper cameras require LeRobot's intelrealsense extra"
            ) from exc
        return RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=serial,
                fps=config.camera_fps,
                width=config.camera_width,
                height=config.camera_height,
                use_depth=False,
            )
        )

    @property
    def is_connected(self) -> bool:
        return bool(
            self._head is not None
            and self._wrist is not None
            and self._head.is_connected
            and self._wrist.is_connected
        )

    @property
    def arm_started(self) -> bool:
        return self._arm_started and self._worker is not None and self._worker.running

    def start(self) -> None:
        """Connect all devices, enable the arm, and block until MIT is engaged."""
        if not self.is_connected:
            self.connect()
        self.start_arm()

    def connect(self) -> None:
        if self.is_connected:
            raise RuntimeError("SinglePiperController is already connected")

        discovered = self._camera_discovery()
        self._devices = RealSenseDevices(
            head_serial=self.config.head_camera_serial or discovered.head_serial,
            wrist_serial=self.config.wrist_camera_serial or discovered.wrist_serial,
        )
        can_candidates = None if self.config.can_interface is None else [self.config.can_interface]
        can_hardware_id = self.config.can_hardware_id
        if can_hardware_id is None and self.config.arm_side is not None:
            can_hardware_id = find_arm_registration(
                self.config.arm_side,
                self.config.arm_role,
            ).can_hardware_id
        self._can = self._can_discovery(
            candidates=can_candidates,
            bitrate=self.config.can_bitrate,
            timeout_s=self.config.can_probe_timeout_s,
            configure=self.config.configure_can,
            hardware_id=can_hardware_id,
        )
        if self._can.hardware_id is None:
            raise RuntimeError("CAN discovery did not return a stable hardware ID")
        self._arm = find_arm_by_hardware_id(self._can.hardware_id)

        self._head = self._camera_factory(self._devices.head_serial, self.config)
        self._wrist = self._camera_factory(self._devices.wrist_serial, self.config)
        connected = []
        try:
            self._head.connect()
            connected.append(self._head)
            self._camera_controls["head"] = dict(
                self._camera_configurator(self._head, "head", self.config)
            )
            self._wrist.connect()
            connected.append(self._wrist)
            self._camera_controls["wrist"] = dict(
                self._camera_configurator(self._wrist, "wrist", self.config)
            )
        except Exception:
            for camera in reversed(connected):
                camera.disconnect()
            self._head = self._wrist = None
            self._camera_controls.clear()
            raise
        logger.info(
            "Single Piper observations ready: CAN=%s head-D435=%s controls=%s "
            "wrist-D415=%s controls=%s",
            self._can.interface,
            self._devices.head_serial,
            self._camera_controls.get("head", {}),
            self._devices.wrist_serial,
            self._camera_controls.get("wrist", {}),
        )

    def start_arm(self) -> None:
        """Explicitly start the managed MIT worker; this enables and homes the arm."""
        if not self.is_connected or self._can is None:
            raise RuntimeError("connect() must complete before start_arm()")
        if self.arm_started:
            raise RuntimeError("Single Piper arm is already started")

        self._telemetry = self._telemetry_factory(self.config.telemetry_address)
        self._telemetry.open()
        self._worker = self._worker_factory(self.config, self._can.interface)
        try:
            self._worker.start()
            self._telemetry.wait_for_engaged(
                self.config.arm_connect_timeout_s,
                self.config.telemetry_timeout_s,
            )
            self._commands = self._command_factory(self.config.command_address)
        except Exception as exc:
            logs = list(getattr(self._worker, "logs", ()))
            self._worker.stop(allow_watchdog_recovery=False)
            self._telemetry.close()
            self._worker = self._telemetry = None
            raise RuntimeError(f"MIT worker failed to become engaged; logs={logs[-20:]}") from exc
        self._arm_started = True
        self._sent_action = False

    @staticmethod
    def _image(frame: Any, label: str) -> np.ndarray:
        image = np.asarray(frame)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{label} must be HWC RGB uint8, got {image.shape} {image.dtype}")
        return np.ascontiguousarray(image).copy()

    def get_camera_observation(self) -> dict[str, np.ndarray]:
        """Read one fresh frame from each already-streaming camera."""
        if not self.is_connected:
            raise RuntimeError("SinglePiperController is not connected")
        top = self._image(
            self._head.async_read(timeout_ms=self.config.camera_timeout_ms),
            "head D435",
        )
        wrist = self._image(
            self._wrist.async_read(timeout_ms=self.config.camera_timeout_ms),
            "wrist D415",
        )
        return {
            "observation.images.top": top,
            "observation.images.wrist": wrist,
        }

    def get_observation(self) -> dict[str, Any]:
        if not self.arm_started or self._telemetry is None:
            raise RuntimeError("start_arm() must complete before reading a full observation")
        cameras = self.get_camera_observation()
        packet = self._telemetry.latest_engaged(self.config.telemetry_timeout_s)
        return {
            "observation.state": normalize_telemetry(packet),
            **cameras,
        }

    def send_action(self, action: np.ndarray) -> np.ndarray:
        """Inject one normalized target; timing/chunk policy remains scheduler-owned."""
        if not self.arm_started or self._commands is None:
            raise RuntimeError("start_arm() must complete before sending an action")
        accepted = validate_action(action)
        q_rad, gripper_width_m = denormalize_action(accepted)
        self._commands.send(q_rad, gripper_width_m)
        self._sent_action = True
        return accepted.copy()

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.is_connected,
            "arm_started": self.arm_started,
            "arm_side": None if self._arm is None else self._arm.side,
            "arm_role": None if self._arm is None else self._arm.role,
            "can_interface": None if self._can is None else self._can.interface,
            "can_hardware_id": None if self._can is None else self._can.hardware_id,
            "can_detection": "usb_serial+piper_feedback_signature",
            "can_feedback_ids": (
                []
                if self._can is None
                else [hex(value) for value in sorted(self._can.arbitration_ids)]
            ),
            "head_camera_serial": None if self._devices is None else self._devices.head_serial,
            "wrist_camera_serial": None if self._devices is None else self._devices.wrist_serial,
            "head_camera_controls": dict(self._camera_controls.get("head", {})),
            "wrist_camera_controls": dict(self._camera_controls.get("wrist", {})),
            "last_head_frame_time": getattr(self._head, "latest_timestamp", None),
            "last_wrist_frame_time": getattr(self._wrist, "latest_timestamp", None),
        }

    def disconnect(self) -> None:
        if self._commands is not None:
            self._commands.close()
            self._commands = None
        if self._worker is not None:
            self._worker.stop(allow_watchdog_recovery=self._sent_action)
            self._worker = None
        if self._telemetry is not None:
            self._telemetry.close()
            self._telemetry = None
        for camera in (self._wrist, self._head):
            if camera is not None and camera.is_connected:
                camera.disconnect()
        self._head = self._wrist = None
        self._camera_controls.clear()
        self._arm_started = False
        self._sent_action = False

    def stop(self) -> None:
        """Stop control and release resources; safe to call after partial start."""
        self.disconnect()
