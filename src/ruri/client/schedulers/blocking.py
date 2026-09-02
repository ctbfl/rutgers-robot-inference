"""Minimal chunk-at-a-time blocking inference scheduler."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ruri.client._args import get_arg
from ruri.client.schedulers._actions import clip_target


logger = logging.getLogger(__name__)


class _BlockingTrace:
    """Write blocking scheduler events without delaying target delivery."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._records: queue.SimpleQueue[dict[str, Any] | object] = (
            queue.SimpleQueue()
        )
        self._stop_token = object()
        self._thread = threading.Thread(
            target=self._write_loop,
            name="ruri-blocking-log",
            daemon=True,
        )
        self._thread.start()

    def record(self, event: str, **fields: Any) -> None:
        self._records.put(
            {
                "event": event,
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "thread": threading.current_thread().name,
                **fields,
            }
        )

    def close(self) -> None:
        self._records.put(self._stop_token)
        self._thread.join()
        self._stream.close()

    def _write_loop(self) -> None:
        while True:
            record = self._records.get()
            if record is self._stop_token:
                return
            self._stream.write(
                json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
            )
            self._stream.flush()


class BlockingScheduler:
    """Own the full observation -> inference -> chunk execution loop."""

    def __init__(self) -> None:
        self.last_run_stats: dict[str, int | float | str] | None = None
        self.last_log_path: Path | None = None

    def run(
        self,
        controller: Callable[[Any], Any],
        policy: Any,
        *,
        args: Any,
    ) -> None:
        """Run until ``max_chunks`` is reached or the caller interrupts it.

        ``policy`` is an already-connected handle returned by
        ``ruri.client.utils.inference_client.connect``.  The scheduler owns
        Controller lifecycle but not the caller-owned policy connection.
        """
        robot = controller(args)

        control_hz = float(get_arg(args, "control_hz", 30.0))
        max_chunks = get_arg(args, "max_chunks", None)
        action_profile = str(get_arg(args, "action_profile", "none"))
        execute_actions_per_chunk = get_arg(
            args, "execute_actions_per_chunk", None
        )
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if max_chunks is not None and max_chunks < 0:
            raise ValueError("max_chunks cannot be negative")
        if action_profile not in ("none", "minimum_jerk"):
            raise ValueError("action_profile must be 'none' or 'minimum_jerk'")
        if (
            execute_actions_per_chunk is not None
            and execute_actions_per_chunk <= 0
        ):
            raise ValueError("execute_actions_per_chunk must be positive")

        trace = self._start_trace(args)
        if trace is not None:
            self.last_log_path = trace.path
            logger.info("BlockingScheduler log: %s", trace.path)
            trace.record(
                "run_start",
                control_hz=control_hz,
                max_chunks=max_chunks,
                execute_actions_per_chunk=execute_actions_per_chunk,
                action_profile=action_profile,
            )

        try:
            output_chunk_size = int(policy.output_chunk_size)
            if (
                execute_actions_per_chunk is not None
                and execute_actions_per_chunk > output_chunk_size
            ):
                raise ValueError(
                    "execute_actions_per_chunk cannot exceed server metadata "
                    f"outputs.output_chunk_size: {execute_actions_per_chunk} > "
                    f"{output_chunk_size}"
                )
            if trace is not None:
                trace.record(
                    "policy_ready", output_chunk_size=output_chunk_size
                )
            robot.start()
            if trace is not None:
                trace.record("controller_ready")
            self.last_run_stats = self._run_loop(
                robot,
                policy,
                args,
                control_hz,
                max_chunks,
                output_chunk_size,
                action_profile,
                execute_actions_per_chunk,
                trace,
            )
        except BaseException as error:
            if trace is not None:
                trace.record(
                    "scheduler_error",
                    error_type=type(error).__name__,
                    error=str(error),
                    traceback=traceback.format_exc(),
                )
            logger.exception(
                "BlockingScheduler failed; log=%s",
                None if trace is None else trace.path,
            )
            raise
        finally:
            try:
                robot.stop()
            finally:
                if trace is not None:
                    trace.record("run_stop", stats=self.last_run_stats)
                    trace.close()

    @staticmethod
    def _run_loop(
        controller: Any,
        policy: Any,
        args: Any,
        control_hz: float,
        max_chunks: int | None,
        output_chunk_size: int,
        action_profile: str = "none",
        execute_actions_per_chunk: int | None = None,
        trace: _BlockingTrace | None = None,
    ) -> dict[str, int | float | str]:
        period_s = 1.0 / control_hz
        chunk_index = 0
        action_timestep = 0
        stats: dict[str, int | float | str] = {
            "chunks_received": 0,
            "actions_sent": 0,
            "clipped_actions": 0,
            "output_chunk_size": output_chunk_size,
            "action_profile": action_profile,
        }

        while max_chunks is None or chunk_index < max_chunks:
            if trace is not None:
                trace.record("observation_started", chunk_index=chunk_index)
            observation_started_ns = time.monotonic_ns()
            policy_inputs = dict(controller.get_observation())
            if trace is not None:
                trace.record(
                    "observation_returned",
                    chunk_index=chunk_index,
                    duration_ms=(time.monotonic_ns() - observation_started_ns)
                    / 1_000_000.0,
                )
            prompt = get_arg(args, "prompt", None)
            if prompt is not None:
                policy_inputs["prompt"] = prompt

            inference_started_ns = time.monotonic_ns()
            if trace is not None:
                trace.record("inference_started", chunk_index=chunk_index)
            response = policy.infer(policy_inputs)
            action_chunk = BlockingScheduler._action_chunk(response)
            if len(action_chunk) != output_chunk_size:
                raise ValueError(
                    "Policy returned an action chunk whose horizon does not match "
                    "metadata outputs.output_chunk_size: "
                    f"{len(action_chunk)} != {output_chunk_size}"
                )
            returned_horizon = len(action_chunk)
            stats["chunks_received"] = int(stats["chunks_received"]) + 1
            if trace is not None:
                trace.record(
                    "inference_returned",
                    chunk_index=chunk_index,
                    duration_ms=(time.monotonic_ns() - inference_started_ns)
                    / 1_000_000.0,
                    chunk_shape=list(action_chunk.shape),
                    action_min=float(np.min(action_chunk)),
                    action_max=float(np.max(action_chunk)),
                )
            if execute_actions_per_chunk is not None:
                if execute_actions_per_chunk > len(action_chunk):
                    raise ValueError(
                        "execute_actions_per_chunk cannot exceed the returned "
                        f"action chunk length: {execute_actions_per_chunk} > "
                        f"{len(action_chunk)}"
                    )
                action_chunk = action_chunk[:execute_actions_per_chunk]
            if action_profile == "minimum_jerk":
                state = np.asarray(
                    policy_inputs.get("observation.state"), dtype=np.float32
                )
                action_chunk = BlockingScheduler._minimum_jerk_chunk(
                    start_action=state,
                    action_chunk=action_chunk,
                    control_hz=control_hz,
                    max_velocity=get_arg(args, "profile_max_velocity", None),
                    max_acceleration=get_arg(
                        args, "profile_max_acceleration", None
                    ),
                )

            if trace is not None:
                trace.record(
                    "chunk_ready",
                    chunk_index=chunk_index,
                    returned_horizon=returned_horizon,
                    executed_horizon=len(action_chunk),
                    action_profile=action_profile,
                )

            next_tick = time.monotonic()
            for chunk_offset, action in enumerate(action_chunk):
                action_values = np.asarray(action, dtype=np.float32).copy()
                target, clipped = clip_target(controller, action_values)
                send_started_ns = time.monotonic_ns()
                try:
                    controller.send_action(target)
                except Exception as error:
                    if trace is not None:
                        trace.record(
                            "action",
                            timestep=action_timestep,
                            chunk_index=chunk_index,
                            chunk_offset=chunk_offset,
                            outcome="rejected",
                            action=target.tolist(),
                            pre_clip_action=(
                                action_values.tolist() if clipped else None
                            ),
                            error_type=type(error).__name__,
                            error=str(error),
                            traceback=traceback.format_exc(),
                        )
                    raise
                stats["actions_sent"] = int(stats["actions_sent"]) + 1
                if clipped:
                    stats["clipped_actions"] = int(stats["clipped_actions"]) + 1
                if trace is not None:
                    trace.record(
                        "action",
                        timestep=action_timestep,
                        chunk_index=chunk_index,
                        chunk_offset=chunk_offset,
                        outcome="sent",
                        action=target.tolist(),
                        pre_clip_action=(
                            action_values.tolist() if clipped else None
                        ),
                        clipped=clipped,
                        send_duration_ms=(time.monotonic_ns() - send_started_ns)
                        / 1_000_000.0,
                    )
                action_timestep += 1
                next_tick += period_s
                remaining_s = next_tick - time.monotonic()
                if remaining_s > 0:
                    time.sleep(remaining_s)
                else:
                    # Do not burst actions in an attempt to catch up with
                    # deadlines already missed by a blocking implementation.
                    next_tick = time.monotonic()

            chunk_index += 1

        return stats

    @staticmethod
    def _start_trace(args: Any) -> _BlockingTrace | None:
        if not bool(get_arg(args, "scheduler_log_enabled", True)):
            return None
        configured_path = get_arg(args, "scheduler_log_path", None)
        if configured_path is None:
            log_dir = Path(get_arg(args, "scheduler_log_dir", "logs"))
            timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
            path = log_dir / f"blocking-{timestamp}-{os.getpid()}.jsonl"
        else:
            path = Path(configured_path)
        return _BlockingTrace(path.expanduser().resolve())

    @staticmethod
    def _action_chunk(response: Any) -> np.ndarray:
        if not isinstance(response, Mapping):
            raise TypeError(
                f"Policy response must be a mapping, got {type(response).__name__}"
            )
        if "action_chunk" not in response:
            raise KeyError("Policy response is missing 'action_chunk'")

        action_chunk = np.asarray(response["action_chunk"], dtype=np.float32)
        if action_chunk.ndim != 2 or action_chunk.shape[0] == 0:
            raise ValueError(
                "action_chunk must have non-empty shape (horizon, action_dim), "
                f"got {action_chunk.shape}"
            )
        if not np.all(np.isfinite(action_chunk)):
            raise ValueError("action_chunk contains NaN or infinity")
        return action_chunk

    @staticmethod
    def _minimum_jerk_chunk(
        *,
        start_action: Any,
        action_chunk: np.ndarray,
        control_hz: float,
        max_velocity: Any = None,
        max_acceleration: Any = None,
    ) -> np.ndarray:
        """Time-warp a chunk along the same polyline and in the same duration.

        The quintic time law ``10u^3 - 15u^4 + 6u^5`` has zero velocity and
        acceleration at both endpoints.  The output retains the original
        number of control ticks and final action, but intermediate waypoints
        are reached at new times so the middle of the motion can run faster.
        """
        chunk = BlockingScheduler._action_chunk({"action_chunk": action_chunk})
        start = np.asarray(start_action, dtype=np.float32)
        if start.shape != (chunk.shape[1],):
            raise ValueError(
                "minimum_jerk requires observation.state to match action_dim: "
                f"{start.shape} != ({chunk.shape[1]},)"
            )
        if not np.all(np.isfinite(start)):
            raise ValueError("minimum_jerk start action contains NaN or infinity")

        horizon = chunk.shape[0]
        path = np.vstack((start, chunk))
        u = np.arange(1, horizon + 1, dtype=np.float64) / horizon
        progress = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        path_position = np.minimum(progress * horizon, horizon)
        segment = np.minimum(path_position.astype(np.int64), horizon - 1)
        fraction = (path_position - segment)[:, None]
        profiled = path[segment] + fraction * (path[segment + 1] - path[segment])
        profiled[-1] = chunk[-1]
        profiled = np.asarray(profiled, dtype=np.float32)

        # Include the held endpoint in the finite-difference check, because the
        # following blocking inference interval commands zero target velocity.
        samples = np.vstack((start, profiled, profiled[-1]))
        velocities = np.diff(samples, axis=0) * control_hz
        accelerations = np.diff(velocities, axis=0) * control_hz
        BlockingScheduler._check_profile_limit(
            values=velocities,
            configured=max_velocity,
            action_dim=chunk.shape[1],
            label="profile_max_velocity",
        )
        BlockingScheduler._check_profile_limit(
            values=accelerations,
            configured=max_acceleration,
            action_dim=chunk.shape[1],
            label="profile_max_acceleration",
        )
        logger.debug(
            "Minimum-jerk chunk horizon=%d peak_velocity=%s peak_acceleration=%s",
            horizon,
            np.max(np.abs(velocities), axis=0).tolist(),
            np.max(np.abs(accelerations), axis=0).tolist(),
        )
        return profiled

    @staticmethod
    def _check_profile_limit(
        *,
        values: np.ndarray,
        configured: Any,
        action_dim: int,
        label: str,
    ) -> None:
        if configured is None:
            return
        limits = np.asarray(configured, dtype=np.float64)
        if limits.ndim == 0:
            limits = np.full(action_dim, float(limits))
        if limits.shape != (action_dim,):
            raise ValueError(f"{label} must be scalar or shape ({action_dim},)")
        if not np.all(np.isfinite(limits)) or np.any(limits <= 0):
            raise ValueError(f"{label} must contain finite positive values")
        peaks = np.max(np.abs(values), axis=0)
        exceeded = np.flatnonzero(peaks > limits + 1e-6)
        if exceeded.size:
            details = ", ".join(
                f"dim{index}={peaks[index]:.3f}>{limits[index]:.3f}"
                for index in exceeded
            )
            raise ValueError(
                f"Fixed-duration minimum-jerk trajectory exceeds {label}: "
                f"{details}. The requested path is infeasible in the original "
                "duration; increase the limit or allow more time."
            )
