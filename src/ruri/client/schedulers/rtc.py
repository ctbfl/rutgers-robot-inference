"""Latency-aligned real-time chunk scheduler.

The control loop owns one monotonically increasing action timestep.  Every
inference request is anchored to that timeline immediately before the remote
policy call.  When the response arrives, actions whose timesteps have already
been sent are discarded and the remaining action queue is replaced atomically.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ruri.client._args import get_arg


logger = logging.getLogger(__name__)


class _RTCTrace:
    """Write scheduler events without blocking the control loop on disk I/O."""

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
            name="ruri-rtc-log",
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


@dataclass(frozen=True)
class _InferenceToken:
    request_id: int
    reason: str
    trigger_step: int
    triggered_ns: int


@dataclass(frozen=True)
class _InferenceResult:
    token: _InferenceToken
    status: str
    request_last_sent_step: int | None
    anchor_step: int | None
    sent_ns: int | None
    received_ns: int
    returned_after_step: int | None
    estimated_delay_steps: int | None
    consumed_steps: int | None
    rtc_applied: bool | None
    chunk: np.ndarray | None = None
    error: BaseException | None = None


@dataclass(frozen=True)
class _QueuedAction:
    value: np.ndarray
    chunk_id: int
    chunk_offset: int


@dataclass(frozen=True)
class _SelectedAction:
    timestep: int
    value: np.ndarray
    source: str
    chunk_id: int
    chunk_offset: int


class _ActionTimeline:
    """Thread-safe action timeline; only the control thread mutates it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_sent_step = -1
        self._future: dict[int, _QueuedAction] = {}
        self._last_accepted: _QueuedAction | None = None

    def latest_sent_step(self) -> int:
        with self._lock:
            return self._latest_sent_step

    def request_snapshot(self) -> tuple[int, np.ndarray | None, int]:
        """Bind a request to the current chunk cursor and its complete tail."""
        with self._lock:
            latest = self._latest_sent_step
            step = latest + 1
            first = self._future.get(step)
            if first is None:
                consumed = (
                    0
                    if self._last_accepted is None
                    else self._last_accepted.chunk_offset + 1
                )
                return latest, None, consumed

            values: list[np.ndarray] = []
            while step in self._future:
                queued = self._future[step]
                if queued.chunk_id != first.chunk_id:
                    break
                values.append(queued.value.copy())
                step += 1
            return latest, np.stack(values), first.chunk_offset

    def current_chunk_consumed_steps(self) -> int:
        with self._lock:
            queued = self._future.get(self._latest_sent_step + 1)
            if queued is not None:
                return queued.chunk_offset
            if self._last_accepted is not None:
                return self._last_accepted.chunk_offset + 1
            return 0

    def install(
        self,
        *,
        chunk: np.ndarray,
        chunk_id: int,
        anchor_step: int,
    ) -> dict[str, int | None]:
        """Replace every unsent action with the still-live suffix of ``chunk``."""
        with self._lock:
            next_step = self._latest_sent_step + 1
            if anchor_step > next_step:
                raise RuntimeError(
                    "RTC chunk is anchored in the future: "
                    f"anchor_step={anchor_step}, next_step={next_step}"
                )
            expired = next_step - anchor_step
            if expired >= len(chunk):
                return {
                    "installed": 0,
                    "expired": len(chunk),
                    "replaced": 0,
                    "next_step": next_step,
                    "last_step": None,
                }

            replaced = len(self._future)
            future: dict[int, _QueuedAction] = {}
            for offset in range(expired, len(chunk)):
                timestep = anchor_step + offset
                future[timestep] = _QueuedAction(
                    value=chunk[offset].copy(),
                    chunk_id=chunk_id,
                    chunk_offset=offset,
                )
            self._future = future
            return {
                "installed": len(future),
                "expired": expired,
                "replaced": replaced,
                "next_step": next_step,
                "last_step": max(future),
            }

    def pending_count(self) -> int:
        with self._lock:
            return len(self._future)

    def bounds(self) -> tuple[int | None, int | None]:
        with self._lock:
            if not self._future:
                return None, None
            return min(self._future), max(self._future)

    def select_next(self) -> _SelectedAction | None:
        with self._lock:
            timestep = self._latest_sent_step + 1
            queued = self._future.get(timestep)
            if queued is not None:
                return _SelectedAction(
                    timestep=timestep,
                    value=queued.value.copy(),
                    source="policy",
                    chunk_id=queued.chunk_id,
                    chunk_offset=queued.chunk_offset,
                )
            if self._last_accepted is None:
                return None
            return _SelectedAction(
                timestep=timestep,
                value=self._last_accepted.value.copy(),
                source="hold",
                chunk_id=self._last_accepted.chunk_id,
                chunk_offset=self._last_accepted.chunk_offset,
            )

    def commit(self, selected: _SelectedAction, accepted: np.ndarray) -> None:
        """Advance the timeline only after the controller accepted the command."""
        with self._lock:
            expected = self._latest_sent_step + 1
            if selected.timestep != expected:
                raise RuntimeError(
                    "RTC action commit is out of order: "
                    f"selected={selected.timestep}, expected={expected}"
                )
            if selected.source == "policy":
                queued = self._future.pop(selected.timestep, None)
                if queued is None:
                    raise RuntimeError(
                        f"RTC action {selected.timestep} disappeared before commit"
                    )
            self._latest_sent_step = selected.timestep
            self._last_accepted = _QueuedAction(
                value=accepted.copy(),
                chunk_id=selected.chunk_id,
                chunk_offset=selected.chunk_offset,
            )


class RTCScheduler:
    """Execute policy chunks continuously with exact latency compensation."""

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
        """Run a standalone RTC request/execute/replace state machine."""
        control_hz = float(get_arg(args, "control_hz", 30.0))
        execution_horizon = int(get_arg(args, "execution_horizon", 5))
        max_chunks = get_arg(args, "max_chunks", None)
        latency_window = int(get_arg(args, "rtc_latency_window", 10))
        configured_initial_delay_steps = get_arg(
            args, "rtc_initial_delay_steps", None
        )
        if configured_initial_delay_steps is not None:
            configured_initial_delay_steps = int(configured_initial_delay_steps)

        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if max_chunks is not None and max_chunks < 0:
            raise ValueError("max_chunks cannot be negative")
        if latency_window <= 0:
            raise ValueError("rtc_latency_window must be positive")
        if (
            configured_initial_delay_steps is not None
            and configured_initial_delay_steps < 0
        ):
            raise ValueError("rtc_initial_delay_steps cannot be negative")

        robot = controller(args)
        remote_policy = policy(args)
        timeline = _ActionTimeline()
        trace = self._start_trace(args)
        if trace is not None:
            self.last_log_path = trace.path
            logger.info("RTCScheduler log: %s", trace.path)
            trace.record(
                "run_start",
                control_hz=control_hz,
                execution_horizon=execution_horizon,
                rtc_initial_delay_steps=configured_initial_delay_steps,
                max_chunks=max_chunks,
            )

        ready_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        request_queue: queue.Queue[_InferenceToken | object] = queue.Queue(maxsize=1)
        result_queue: queue.Queue[_InferenceResult] = queue.Queue(maxsize=1)
        stop_worker = threading.Event()
        stop_token = object()

        def inference_worker() -> None:
            try:
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
                output_chunk_size = int(remote_policy.output_chunk_size)
                initial_delay_steps = (
                    min(4, output_chunk_size)
                    if configured_initial_delay_steps is None
                    else configured_initial_delay_steps
                )
                delay_history: deque[int] = deque(
                    [initial_delay_steps], maxlen=latency_window
                )
                if trace is not None:
                    trace.record(
                        "policy_ready",
                        output_chunk_size=output_chunk_size,
                        rtc_initial_delay_steps=initial_delay_steps,
                    )
                ready_queue.put(
                    ("ok", (output_chunk_size, initial_delay_steps))
                )

                while not stop_worker.is_set():
                    token = request_queue.get()
                    if token is stop_token:
                        return
                    assert isinstance(token, _InferenceToken)
                    received_ns = time.monotonic_ns()
                    try:
                        observation = robot.get_observation()
                        request_last_step, previous_tail, consumed_steps = (
                            timeline.request_snapshot()
                        )
                        anchor_step = request_last_step + 1
                        raw_estimate = max(delay_history)
                        prefix_length = 0 if previous_tail is None else len(previous_tail)
                        estimated_delay = min(raw_estimate, prefix_length)
                        inputs = self._policy_inputs(
                            observation=observation,
                            args=args,
                            estimated_delay_steps=estimated_delay,
                            previous_tail=previous_tail,
                            consumed_steps=consumed_steps,
                        )
                        sent_ns = time.monotonic_ns()
                        if trace is not None:
                            trace.record(
                                "inference_started",
                                request_id=token.request_id,
                                trigger_step=token.trigger_step,
                                request_last_sent_step=request_last_step,
                                anchor_step=anchor_step,
                                estimated_delay_steps=estimated_delay,
                                raw_estimated_delay_steps=raw_estimate,
                                consumed_steps=consumed_steps,
                                previous_tail_shape=(
                                    None
                                    if previous_tail is None
                                    else list(previous_tail.shape)
                                ),
                            )
                        response = remote_policy.infer(inputs)
                        chunk = self._action_chunk(response)
                        if len(chunk) != output_chunk_size:
                            raise ValueError(
                                "Policy returned an action chunk whose horizon does "
                                "not match metadata outputs.output_chunk_size: "
                                f"{len(chunk)} != {output_chunk_size}"
                            )
                        rtc_requested = previous_tail is not None
                        rtc_applied = response.get("rtc.applied")
                        if rtc_requested and rtc_applied is not True:
                            raise RuntimeError(
                                "Remote policy did not apply RTC guidance: "
                                f"reason={response.get('rtc.reason')!r}"
                            )
                        received_ns = time.monotonic_ns()
                        duration_s = (received_ns - sent_ns) / 1_000_000_000.0
                        returned_after_step = timeline.latest_sent_step()
                        actual_delay = max(
                            0, returned_after_step - request_last_step
                        )
                        measured_delay = max(
                            actual_delay, math.ceil(duration_s * control_hz)
                        )
                        delay_history.append(measured_delay)
                        if trace is not None:
                            trace.record(
                                "inference_returned",
                                request_id=token.request_id,
                                anchor_step=anchor_step,
                                returned_after_step=returned_after_step,
                                actual_delay_steps=actual_delay,
                                measured_delay_steps=measured_delay,
                                rtc_requested=rtc_requested,
                                rtc_applied=rtc_applied,
                                rtc_reason=response.get("rtc.reason"),
                                rtc_inference_delay=response.get(
                                    "rtc.inference_delay"
                                ),
                                rtc_prefix_attention_horizon=response.get(
                                    "rtc.prefix_attention_horizon"
                                ),
                                duration_ms=duration_s * 1_000.0,
                                chunk_shape=list(chunk.shape),
                                action_min=float(np.min(chunk)),
                                action_max=float(np.max(chunk)),
                            )
                        result_queue.put(
                            _InferenceResult(
                                token=token,
                                status="ok",
                                request_last_sent_step=request_last_step,
                                anchor_step=anchor_step,
                                sent_ns=sent_ns,
                                received_ns=received_ns,
                                returned_after_step=returned_after_step,
                                estimated_delay_steps=estimated_delay,
                                consumed_steps=consumed_steps,
                                rtc_applied=(rtc_applied is True),
                                chunk=chunk,
                            )
                        )
                    except Exception as error:
                        received_ns = time.monotonic_ns()
                        if trace is not None:
                            trace.record(
                                "inference_error",
                                request_id=token.request_id,
                                error_type=type(error).__name__,
                                error=str(error),
                                traceback=traceback.format_exc(),
                            )
                        result_queue.put(
                            _InferenceResult(
                                token=token,
                                status="error",
                                request_last_sent_step=None,
                                anchor_step=None,
                                sent_ns=None,
                                received_ns=received_ns,
                                returned_after_step=timeline.latest_sent_step(),
                                estimated_delay_steps=None,
                                consumed_steps=None,
                                rtc_applied=None,
                                error=error,
                            )
                        )
            finally:
                remote_policy.stop()
                if trace is not None:
                    trace.record("policy_stopped")

        worker: threading.Thread | None = None
        try:
            worker = threading.Thread(
                target=inference_worker,
                name="ruri-rtc-inference",
                daemon=True,
            )
            worker.start()
            ready_status, ready_payload = ready_queue.get()
            if ready_status == "error":
                raise ready_payload
            output_chunk_size, initial_delay_steps = ready_payload
            output_chunk_size = int(output_chunk_size)
            initial_delay_steps = int(initial_delay_steps)
            if execution_horizon > output_chunk_size:
                raise ValueError(
                    "execution_horizon cannot exceed server metadata "
                    "outputs.output_chunk_size: "
                    f"{execution_horizon} > {output_chunk_size}"
                )
            if initial_delay_steps > output_chunk_size:
                raise ValueError(
                    "rtc_initial_delay_steps cannot exceed server metadata "
                    "outputs.output_chunk_size: "
                    f"{initial_delay_steps} > {output_chunk_size}"
                )

            robot.start()
            if trace is not None:
                trace.record("controller_ready")

            if max_chunks == 0:
                self.last_run_stats = self._empty_stats()
                return

            self.last_run_stats = self._execute(
                robot=robot,
                timeline=timeline,
                request_queue=request_queue,
                result_queue=result_queue,
                control_hz=control_hz,
                execution_horizon=execution_horizon,
                max_chunks=max_chunks,
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
                "RTCScheduler failed; log=%s",
                None if trace is None else trace.path,
            )
            raise
        finally:
            try:
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
        timeline: _ActionTimeline,
        request_queue: queue.Queue[_InferenceToken | object],
        result_queue: queue.Queue[_InferenceResult],
        control_hz: float,
        execution_horizon: int,
        max_chunks: int | None,
        trace: _RTCTrace | None,
    ) -> dict[str, int | float | str]:
        period_s = 1.0 / control_hz
        request_id = 0
        inference_in_flight = False
        stats: dict[str, int | float | str] = {
            "actions_sent": 0,
            "policy_actions_sent": 0,
            "hold_ticks": 0,
            "chunks_requested": 0,
            "chunks_received": 0,
            "chunks_installed": 0,
            "stale_chunks": 0,
            "expired_actions": 0,
            "replaced_actions": 0,
            "max_actual_delay_steps": 0,
            "execution_horizon": execution_horizon,
        }

        def can_request() -> bool:
            return max_chunks is None or request_id < max_chunks

        def trigger(reason: str) -> None:
            nonlocal request_id, inference_in_flight
            request_id += 1
            token = _InferenceToken(
                request_id=request_id,
                reason=reason,
                trigger_step=timeline.latest_sent_step(),
                triggered_ns=time.monotonic_ns(),
            )
            if trace is not None:
                trace.record(
                    "inference_triggered",
                    request_id=request_id,
                    reason=reason,
                    trigger_step=token.trigger_step,
                    queue_size=timeline.pending_count(),
                    consumed_steps=timeline.current_chunk_consumed_steps(),
                )
            request_queue.put_nowait(token)
            inference_in_flight = True
            stats["chunks_requested"] = int(stats["chunks_requested"]) + 1

        def receive(*, block: bool) -> bool:
            nonlocal inference_in_flight
            try:
                result = result_queue.get() if block else result_queue.get_nowait()
            except queue.Empty:
                return False
            inference_in_flight = False
            if result.status == "error":
                assert result.error is not None
                raise result.error
            assert result.chunk is not None
            assert result.anchor_step is not None
            assert result.request_last_sent_step is not None
            stats["chunks_received"] = int(stats["chunks_received"]) + 1
            install = timeline.install(
                chunk=result.chunk,
                chunk_id=result.token.request_id,
                anchor_step=result.anchor_step,
            )
            expired = int(install["expired"] or 0)
            installed = int(install["installed"] or 0)
            stats["expired_actions"] = int(stats["expired_actions"]) + expired
            stats["replaced_actions"] = int(stats["replaced_actions"]) + int(
                install["replaced"] or 0
            )
            actual_delay = max(
                0, timeline.latest_sent_step() - result.request_last_sent_step
            )
            stats["max_actual_delay_steps"] = max(
                int(stats["max_actual_delay_steps"]), actual_delay
            )
            if installed:
                stats["chunks_installed"] = int(stats["chunks_installed"]) + 1
            else:
                stats["stale_chunks"] = int(stats["stale_chunks"]) + 1
            first_step, last_step = timeline.bounds()
            if trace is not None:
                trace.record(
                    "chunk_installed" if installed else "chunk_stale",
                    request_id=result.token.request_id,
                    anchor_step=result.anchor_step,
                    returned_after_step=result.returned_after_step,
                    installed_after_step=timeline.latest_sent_step(),
                    actual_delay_steps=actual_delay,
                    consumed_steps=result.consumed_steps,
                    rtc_applied=result.rtc_applied,
                    expired_actions=expired,
                    installed_actions=installed,
                    replaced_actions=int(install["replaced"] or 0),
                    queue_first_step=first_step,
                    queue_last_step=last_step,
                )
            return True

        # Bootstrap: no action exists until the first complete chunk arrives.
        trigger("initial")
        receive(block=True)
        # The initial inference is intentionally blocking. Start the control
        # clock only after it returns so the first two actions cannot burst.
        next_tick = time.monotonic()

        while True:
            if inference_in_flight:
                receive(block=False)

            pending = timeline.pending_count()
            if (
                not inference_in_flight
                and can_request()
                and (
                    timeline.current_chunk_consumed_steps() >= execution_horizon
                    or pending == 0
                )
            ):
                trigger("execution_horizon" if pending else "queue_empty")

            no_more_chunks = not can_request()
            if pending == 0 and not inference_in_flight and no_more_chunks:
                break

            selected = timeline.select_next()
            if selected is None:
                raise RuntimeError("RTCScheduler has no action available to execute")

            action_values = selected.value.copy()
            send_started_ns = time.monotonic_ns()
            try:
                accepted = robot.send_action(action_values)
            except Exception as error:
                if trace is not None:
                    trace.record(
                        "action",
                        timestep=selected.timestep,
                        source=selected.source,
                        chunk_id=selected.chunk_id,
                        chunk_offset=selected.chunk_offset,
                        outcome="rejected",
                        action=action_values.tolist(),
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
            timeline.commit(selected, accepted_values)
            stats["actions_sent"] = int(stats["actions_sent"]) + 1
            if selected.source == "policy":
                stats["policy_actions_sent"] = (
                    int(stats["policy_actions_sent"]) + 1
                )
            else:
                stats["hold_ticks"] = int(stats["hold_ticks"]) + 1
            if trace is not None:
                trace.record(
                    "action",
                    timestep=selected.timestep,
                    source=selected.source,
                    chunk_id=selected.chunk_id,
                    chunk_offset=selected.chunk_offset,
                    outcome="sent",
                    action=action_values.tolist(),
                    accepted_action=accepted_values.tolist(),
                    clipped=not np.array_equal(action_values, accepted_values),
                    queue_size=timeline.pending_count(),
                    inference_in_flight=inference_in_flight,
                    send_duration_ms=(time.monotonic_ns() - send_started_ns)
                    / 1_000_000.0,
                )

            pending = timeline.pending_count()
            if (
                not inference_in_flight
                and can_request()
                and (
                    timeline.current_chunk_consumed_steps() >= execution_horizon
                    or pending == 0
                )
            ):
                trigger("execution_horizon" if pending else "queue_empty")

            next_tick += period_s
            remaining_s = next_tick - time.monotonic()
            if remaining_s > 0:
                time.sleep(remaining_s)
            else:
                next_tick = time.monotonic()

        return stats

    @staticmethod
    def _policy_inputs(
        *,
        observation: Any,
        args: Any,
        estimated_delay_steps: int,
        previous_tail: np.ndarray | None,
        consumed_steps: int,
    ) -> dict[str, Any]:
        inputs = dict(observation)
        prompt = get_arg(args, "prompt", None)
        if prompt is not None:
            inputs["prompt"] = prompt
        if previous_tail is not None:
            inputs["context.rtc.prev_chunk_left_over"] = previous_tail
            inputs["context.rtc.consumed_steps"] = consumed_steps
            inputs[
                "context.rtc.estimated_inference_delay_steps"
            ] = estimated_delay_steps
        return inputs

    @staticmethod
    def _action_chunk(response: Any) -> np.ndarray:
        if not isinstance(response, Mapping):
            raise TypeError(
                f"Policy response must be a mapping, got {type(response).__name__}"
            )
        if "action_chunk" not in response:
            raise KeyError("Policy response is missing 'action_chunk'")
        chunk = np.asarray(response["action_chunk"], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            raise ValueError(
                "action_chunk must have non-empty shape (horizon, action_dim), "
                f"got {chunk.shape}"
            )
        if not np.all(np.isfinite(chunk)):
            raise ValueError("action_chunk contains NaN or infinity")
        return chunk

    @staticmethod
    def _empty_stats() -> dict[str, int | float | str]:
        return {
            "actions_sent": 0,
            "policy_actions_sent": 0,
            "hold_ticks": 0,
            "chunks_requested": 0,
            "chunks_received": 0,
            "chunks_installed": 0,
            "stale_chunks": 0,
            "expired_actions": 0,
            "replaced_actions": 0,
            "max_actual_delay_steps": 0,
            "execution_horizon": 0,
        }

    @staticmethod
    def _start_trace(args: Any) -> _RTCTrace | None:
        if not bool(get_arg(args, "scheduler_log_enabled", True)):
            return None
        configured_path = get_arg(args, "scheduler_log_path", None)
        if configured_path is None:
            log_dir = Path(get_arg(args, "scheduler_log_dir", "logs"))
            timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
            path = log_dir / f"rtc-{timestamp}-{os.getpid()}.jsonl"
        else:
            path = Path(configured_path)
        return _RTCTrace(path.expanduser().resolve())
