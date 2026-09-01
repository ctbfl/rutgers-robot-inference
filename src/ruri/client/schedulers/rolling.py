"""Asynchronous rolling action-chunk scheduler."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from ruri.client._args import get_arg


logger = logging.getLogger(__name__)

_AGGREGATE_WEIGHTS = {
    "weighted_average": (0.3, 0.7),
    "latest_only": (0.0, 1.0),
    "average": (0.5, 0.5),
    "conservative": (0.7, 0.3),
}


class _SchedulerTrace:
    """Non-blocking JSONL writer for control-loop and inference events."""

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
            name="ruri-scheduler-log",
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


class RollingScheduler:
    """Continuously execute actions while fetching and rolling policy chunks."""

    def __init__(self) -> None:
        self.last_run_stats: dict[str, int | float | str] | None = None
        self.last_log_path: Path | None = None

    def run(
        self,
        controller: Callable[[Any], Any],
        policy: Callable[[Any], Any],
        *,
        args: Any,
    ) -> None:
        """Run a fixed-rate executor and an asynchronous inference worker.

        The exact same global ``args`` object is passed to both factories.  An
        initial chunk is fetched before execution starts.  Afterwards a new
        observation is requested when the fraction of actions remaining in the
        current chunk reaches ``chunk_size_threshold``.
        """
        robot = controller(args)
        remote_policy = policy(args)

        control_hz = float(get_arg(args, "control_hz", 30.0))
        max_chunks = get_arg(args, "max_chunks", None)
        threshold = float(get_arg(args, "chunk_size_threshold", 0.5))
        aggregate_name = str(get_arg(args, "aggregate_fn_name", "weighted_average"))
        configured_horizon = get_arg(args, "actions_per_chunk", None)

        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if max_chunks is not None and max_chunks < 0:
            raise ValueError("max_chunks cannot be negative")
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("chunk_size_threshold must be between 0 and 1")
        if aggregate_name not in _AGGREGATE_WEIGHTS:
            available = ", ".join(_AGGREGATE_WEIGHTS)
            raise ValueError(
                f"Unknown aggregate_fn_name {aggregate_name!r}; available: {available}"
            )
        if configured_horizon is not None and configured_horizon <= 0:
            raise ValueError("actions_per_chunk must be positive")

        trace = self._start_trace(args)
        if trace is not None:
            self.last_log_path = trace.path
            logger.info("RollingScheduler log: %s", trace.path)
            trace.record(
                "run_start",
                control_hz=control_hz,
                max_chunks=max_chunks,
                actions_per_chunk=configured_horizon,
                chunk_size_threshold=threshold,
                aggregate_fn_name=aggregate_name,
            )

        worker: threading.Thread | None = None
        stop_worker = threading.Event()
        ready_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        request_queue: queue.Queue[object] = queue.Queue(maxsize=1)
        result_queue: queue.Queue[tuple[str, int | None, Any]] = queue.Queue(maxsize=1)
        stop_token = object()
        latest_step = [-1]
        latest_step_lock = threading.Lock()

        def inference_worker() -> None:
            try:
                # The policy's complete lifecycle stays on one thread.  This
                # matters for transports such as ZeroMQ sockets, which must not
                # be reused from a different thread after initial inference.
                try:
                    remote_policy.start()
                except Exception as error:
                    if trace is not None:
                        trace.record(
                            "policy_start_error",
                            error_type=type(error).__name__,
                            error=str(error),
                            traceback=traceback.format_exc(),
                        )
                    ready_queue.put(("error", error))
                    return
                if trace is not None:
                    trace.record("policy_ready")
                ready_queue.put(("ok", None))
                while not stop_worker.is_set():
                    request = request_queue.get()
                    if request is stop_token:
                        return
                    request_id = int(request)
                    inference_started_ns = time.monotonic_ns()
                    if trace is not None:
                        trace.record("inference_started", request_id=request_id)
                    try:
                        observation = robot.get_observation()
                        with latest_step_lock:
                            observation_step = latest_step[0]
                        # ACT action_chunk[0] is the first command to execute
                        # *after* this observation.  latest_step is the command
                        # already sent, so anchoring offset zero there would
                        # discard action_chunk[0] and shift every subsequent
                        # action one control tick early.
                        first_action_step = observation_step + 1
                        inputs = self._policy_inputs(observation, args)
                        chunk = self._action_chunk(remote_policy.infer(inputs))
                        if (
                            configured_horizon is not None
                            and chunk.shape[0] != int(configured_horizon)
                        ):
                            raise ValueError(
                                "Policy returned an action chunk whose horizon does "
                                "not match actions_per_chunk: "
                                f"{chunk.shape[0]} != {configured_horizon}"
                            )
                        if trace is not None:
                            trace.record(
                                "inference_result",
                                request_id=request_id,
                                observation_timestep=observation_step,
                                first_action_timestep=first_action_step,
                                duration_ms=(
                                    time.monotonic_ns() - inference_started_ns
                                )
                                / 1_000_000.0,
                                chunk_shape=list(chunk.shape),
                                action_min=float(np.min(chunk)),
                                action_max=float(np.max(chunk)),
                            )
                        result_queue.put(("ok", first_action_step, chunk))
                    except Exception as error:
                        if trace is not None:
                            trace.record(
                                "inference_error",
                                request_id=request_id,
                                duration_ms=(
                                    time.monotonic_ns() - inference_started_ns
                                )
                                / 1_000_000.0,
                                error_type=type(error).__name__,
                                error=str(error),
                                traceback=traceback.format_exc(),
                            )
                        result_queue.put(("error", None, error))
            finally:
                remote_policy.stop()
                if trace is not None:
                    trace.record("policy_stopped")

        try:
            worker = threading.Thread(
                target=inference_worker,
                name="ruri-rolling-inference",
                daemon=True,
            )
            worker.start()

            # Check the remote dependency before enabling real hardware.
            ready_status, ready_payload = ready_queue.get()
            if ready_status == "error":
                raise ready_payload
            robot.start()
            if trace is not None:
                trace.record("controller_ready")

            if max_chunks == 0:
                self.last_run_stats = self._empty_stats(aggregate_name)
                return

            if trace is not None:
                trace.record("inference_requested", request_id=1, reason="initial")
            request_queue.put_nowait(1)
            initial_status, _, initial_payload = result_queue.get()
            if initial_status == "error":
                raise initial_payload
            initial_chunk = initial_payload
            horizon = int(configured_horizon or initial_chunk.shape[0])

            old_weight, new_weight = _AGGREGATE_WEIGHTS[aggregate_name]
            self.last_run_stats = self._execute(
                robot=robot,
                initial_chunk=initial_chunk,
                horizon=horizon,
                control_hz=control_hz,
                max_chunks=max_chunks,
                threshold=threshold,
                old_weight=old_weight,
                new_weight=new_weight,
                aggregate_name=aggregate_name,
                request_queue=request_queue,
                result_queue=result_queue,
                latest_step=latest_step,
                latest_step_lock=latest_step_lock,
                trace=trace,
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
                "RollingScheduler failed; log=%s",
                None if trace is None else trace.path,
            )
            raise
        finally:
            try:
                # Stop hardware command production immediately on any failure.
                robot.stop()
            finally:
                stop_worker.set()
                if worker is not None and worker.is_alive():
                    try:
                        request_queue.put_nowait(stop_token)
                    except queue.Full:
                        pass
                if worker is not None:
                    worker.join(timeout=1.0)
                if trace is not None:
                    trace.record("run_stop", stats=self.last_run_stats)
                    trace.close()

    @staticmethod
    def _execute(
        *,
        robot: Any,
        initial_chunk: np.ndarray,
        horizon: int,
        control_hz: float,
        max_chunks: int | None,
        threshold: float,
        old_weight: float,
        new_weight: float,
        aggregate_name: str,
        request_queue: queue.Queue[object],
        result_queue: queue.Queue[tuple[str, int | None, Any]],
        latest_step: list[int],
        latest_step_lock: threading.Lock,
        trace: _SchedulerTrace | None,
    ) -> dict[str, int | float | str]:
        future_actions = {
            timestep: action.copy() for timestep, action in enumerate(initial_chunk)
        }
        chunks_requested = 1
        chunks_received = 1
        actions_sent = 0
        hold_ticks = 0
        stale_actions = 0
        overlap_actions = 0
        inference_in_flight = False
        last_action: np.ndarray | None = None
        period_s = 1.0 / control_hz
        next_tick = time.monotonic()

        while True:
            if inference_in_flight:
                try:
                    status, first_action_step, payload = result_queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    inference_in_flight = False
                    if status == "error":
                        raise payload
                    chunks_received += 1
                    future_actions, stale, overlap = RollingScheduler._merge_chunks(
                        current=future_actions,
                        incoming=payload,
                        first_timestep=int(first_action_step),
                        latest_timestep=latest_step[0],
                        old_weight=old_weight,
                        new_weight=new_weight,
                    )
                    stale_actions += stale
                    overlap_actions += overlap
                    if trace is not None:
                        trace.record(
                            "chunk_merged",
                            request_id=chunks_received,
                            first_action_timestep=int(first_action_step),
                            latest_timestep=latest_step[0],
                            stale_actions=stale,
                            overlap_actions=overlap,
                            queue_size=len(future_actions),
                            queue_first_timestep=(
                                None if not future_actions else min(future_actions)
                            ),
                            queue_last_timestep=(
                                None if not future_actions else max(future_actions)
                            ),
                        )

            no_more_chunks = max_chunks is not None and chunks_requested >= max_chunks
            if not future_actions and not inference_in_flight and no_more_chunks:
                break

            expected_step = latest_step[0] + 1
            while future_actions and min(future_actions) < expected_step:
                del future_actions[min(future_actions)]
                stale_actions += 1

            if expected_step in future_actions:
                action = future_actions.pop(expected_step)
                action_source = "policy"
            elif last_action is not None:
                # Keep the MIT target stream alive if inference exceeds the
                # remaining lookahead.  Its timestep still advances, so stale
                # predictions are discarded when the response arrives.
                action = last_action
                action_source = "hold"
                hold_ticks += 1
            else:
                raise RuntimeError("RollingScheduler has no action available to execute")

            action_values = np.asarray(action, dtype=np.float32).copy()
            send_started_ns = time.monotonic_ns()
            try:
                accepted = robot.send_action(action_values)
            except Exception as error:
                if trace is not None:
                    trace.record(
                        "action",
                        timestep=expected_step,
                        source=action_source,
                        outcome="rejected",
                        action=action_values.tolist(),
                        queue_size=len(future_actions),
                        inference_in_flight=inference_in_flight,
                        chunks_requested=chunks_requested,
                        chunks_received=chunks_received,
                        send_duration_ms=(time.monotonic_ns() - send_started_ns)
                        / 1_000_000.0,
                        error_type=type(error).__name__,
                        error=str(error),
                        traceback=traceback.format_exc(),
                    )
                raise
            accepted_values = (
                action_values
                if accepted is None
                else np.asarray(accepted, dtype=np.float32).copy()
            )
            if trace is not None:
                trace.record(
                    "action",
                    timestep=expected_step,
                    source=action_source,
                    outcome="sent",
                    action=action_values.tolist(),
                    accepted_action=accepted_values.tolist(),
                    clipped=not np.array_equal(action_values, accepted_values),
                    queue_size=len(future_actions),
                    inference_in_flight=inference_in_flight,
                    chunks_requested=chunks_requested,
                    chunks_received=chunks_received,
                    send_duration_ms=(time.monotonic_ns() - send_started_ns)
                    / 1_000_000.0,
                )
            last_action = action_values
            with latest_step_lock:
                latest_step[0] = expected_step
            actions_sent += 1

            can_request = max_chunks is None or chunks_requested < max_chunks
            remaining_ratio = len(future_actions) / horizon
            if (
                can_request
                and not inference_in_flight
                and remaining_ratio <= threshold
            ):
                chunks_requested += 1
                if trace is not None:
                    trace.record(
                        "inference_requested",
                        request_id=chunks_requested,
                        reason="queue_threshold",
                        latest_timestep=latest_step[0],
                        queue_size=len(future_actions),
                        remaining_ratio=remaining_ratio,
                    )
                request_queue.put_nowait(chunks_requested)
                inference_in_flight = True

            next_tick += period_s
            remaining_s = next_tick - time.monotonic()
            if remaining_s > 0:
                time.sleep(remaining_s)
            else:
                # Do not burst commands to catch up with missed deadlines.
                next_tick = time.monotonic()

        return {
            "actions_sent": actions_sent,
            "chunks_requested": chunks_requested,
            "chunks_received": chunks_received,
            "hold_ticks": hold_ticks,
            "stale_actions": stale_actions,
            "overlap_actions": overlap_actions,
            "aggregate_fn_name": aggregate_name,
            "chunk_size_threshold": threshold,
        }

    @staticmethod
    def _merge_chunks(
        *,
        current: Mapping[int, np.ndarray],
        incoming: np.ndarray,
        first_timestep: int,
        latest_timestep: int,
        old_weight: float,
        new_weight: float,
    ) -> tuple[dict[int, np.ndarray], int, int]:
        """Return LeRobot-style replacement queue indexed by control timestep."""
        merged: dict[int, np.ndarray] = {}
        stale = 0
        overlap = 0
        for offset, new_action in enumerate(incoming):
            timestep = first_timestep + offset
            if timestep <= latest_timestep:
                stale += 1
                continue
            if timestep in current:
                merged[timestep] = (
                    old_weight * current[timestep] + new_weight * new_action
                ).astype(np.float32, copy=False)
                overlap += 1
            else:
                merged[timestep] = new_action.copy()
        return merged, stale, overlap

    @staticmethod
    def _policy_inputs(observation: Any, args: Any) -> dict[str, Any]:
        inputs = dict(observation)
        prompt = get_arg(args, "prompt", None)
        if prompt is not None:
            inputs["prompt"] = prompt
        actions_per_chunk = get_arg(args, "actions_per_chunk", None)
        if actions_per_chunk is not None:
            inputs["context.actions_per_chunk"] = actions_per_chunk
        return inputs

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
    def _empty_stats(aggregate_name: str) -> dict[str, int | float | str]:
        return {
            "actions_sent": 0,
            "chunks_requested": 0,
            "chunks_received": 0,
            "hold_ticks": 0,
            "stale_actions": 0,
            "overlap_actions": 0,
            "aggregate_fn_name": aggregate_name,
            "chunk_size_threshold": 0.0,
        }

    @staticmethod
    def _start_trace(args: Any) -> _SchedulerTrace | None:
        if not bool(get_arg(args, "scheduler_log_enabled", True)):
            return None
        configured_path = get_arg(args, "scheduler_log_path", None)
        if configured_path is None:
            log_dir = Path(get_arg(args, "scheduler_log_dir", "logs"))
            timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
            path = log_dir / f"rolling-{timestamp}-{os.getpid()}.jsonl"
        else:
            path = Path(configured_path)
        return _SchedulerTrace(path.expanduser().resolve())
