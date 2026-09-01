"""LeRobot-style temporal ensembling on an absolute control timeline."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ruri.client._args import get_arg


logger = logging.getLogger(__name__)


class _TemporalEnsembleTrace:
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
            name="ruri-temporal-ensemble-log",
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
    triggered_after_step: int


@dataclass(frozen=True)
class _InferenceResult:
    token: _InferenceToken
    status: str
    observation_step: int | None
    anchor_step: int | None
    chunk: np.ndarray | None = None
    duration_ms: float | None = None
    error: BaseException | None = None


@dataclass
class _EnsembleCell:
    weighted_sum: np.ndarray
    weight_sum: float
    count: int

    def value(self) -> np.ndarray:
        return np.asarray(self.weighted_sum / self.weight_sum, dtype=np.float32)


@dataclass(frozen=True)
class _SelectedAction:
    timestep: int
    value: np.ndarray
    source: str
    contributors: int


class _ActionEnsemble:
    """Online exponential ensemble keyed by absolute action timestep."""

    def __init__(self, coefficient: float):
        self.coefficient = float(coefficient)
        self._lock = threading.Lock()
        self._latest_sent_step = -1
        self._future: dict[int, _EnsembleCell] = {}
        self._last_accepted: np.ndarray | None = None

    def latest_sent_step(self) -> int:
        with self._lock:
            return self._latest_sent_step

    def next_action_step(self) -> int:
        with self._lock:
            return self._latest_sent_step + 1

    def add_chunk(self, chunk: np.ndarray, anchor_step: int) -> dict[str, Any]:
        with self._lock:
            next_step = self._latest_sent_step + 1
            stale = added = overlaps = 0
            first_detail: dict[str, Any] | None = None
            max_contributors = 0
            for offset, incoming in enumerate(chunk):
                timestep = anchor_step + offset
                if timestep < next_step:
                    stale += 1
                    continue
                action = np.asarray(incoming, dtype=np.float64)
                cell = self._future.get(timestep)
                old_action = None if cell is None else cell.value()
                if cell is None:
                    cell = _EnsembleCell(
                        weighted_sum=action.copy(),
                        weight_sum=1.0,
                        count=1,
                    )
                    self._future[timestep] = cell
                    added += 1
                else:
                    if action.shape != cell.weighted_sum.shape:
                        raise ValueError(
                            "Temporal ensemble action dimension changed: "
                            f"{action.shape} != {cell.weighted_sum.shape}"
                        )
                    weight = math.exp(-self.coefficient * cell.count)
                    if not math.isfinite(weight) or weight <= 0:
                        raise ValueError("Temporal ensemble weight is not finite and positive")
                    cell.weighted_sum += weight * action
                    cell.weight_sum += weight
                    cell.count += 1
                    overlaps += 1
                max_contributors = max(max_contributors, cell.count)
                if first_detail is None:
                    first_detail = {
                        "timestep": timestep,
                        "old_action": (
                            None if old_action is None else old_action.tolist()
                        ),
                        "new_action": action.astype(np.float32).tolist(),
                        "ensembled_action": cell.value().tolist(),
                        "contributors": cell.count,
                    }
            return {
                "stale": stale,
                "added": added,
                "overlaps": overlaps,
                "first_detail": first_detail,
                "max_contributors": max_contributors,
                "pending": len(self._future),
            }

    def pending_count(self) -> int:
        with self._lock:
            return len(self._future)

    def select_next(self) -> _SelectedAction | None:
        with self._lock:
            timestep = self._latest_sent_step + 1
            cell = self._future.get(timestep)
            if cell is not None:
                return _SelectedAction(
                    timestep=timestep,
                    value=cell.value(),
                    source="ensemble",
                    contributors=cell.count,
                )
            if self._last_accepted is None:
                return None
            return _SelectedAction(
                timestep=timestep,
                value=self._last_accepted.copy(),
                source="hold",
                contributors=0,
            )

    def commit(self, selected: _SelectedAction, accepted: np.ndarray) -> None:
        with self._lock:
            expected = self._latest_sent_step + 1
            if selected.timestep != expected:
                raise RuntimeError(
                    "Temporal ensemble action commit is out of order: "
                    f"{selected.timestep} != {expected}"
                )
            if selected.source == "ensemble":
                if self._future.pop(selected.timestep, None) is None:
                    raise RuntimeError(
                        f"Temporal ensemble action {selected.timestep} disappeared"
                    )
            self._latest_sent_step = selected.timestep
            self._last_accepted = accepted.copy()


class TemporalEnsembleScheduler:
    """Query a full ACT chunk every step and ensemble overlapping predictions."""

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
        control_hz = float(get_arg(args, "control_hz", 30.0))
        horizon = int(get_arg(args, "actions_per_chunk", 100))
        coefficient = float(get_arg(args, "temporal_ensemble_coeff", 0.01))
        max_chunks = get_arg(args, "max_chunks", None)
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if horizon <= 0:
            raise ValueError("actions_per_chunk must be positive")
        if not math.isfinite(coefficient):
            raise ValueError("temporal_ensemble_coeff must be finite")
        if max_chunks is not None and max_chunks < 0:
            raise ValueError("max_chunks cannot be negative")

        robot = controller(args)
        remote_policy = policy(args)
        ensemble = _ActionEnsemble(coefficient)
        trace = self._start_trace(args)
        if trace is not None:
            self.last_log_path = trace.path
            logger.info("TemporalEnsembleScheduler log: %s", trace.path)
            trace.record(
                "run_start",
                control_hz=control_hz,
                actions_per_chunk=horizon,
                temporal_ensemble_coeff=coefficient,
                max_chunks=max_chunks,
            )

        ready_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        request_queue: queue.Queue[_InferenceToken | object] = queue.Queue(maxsize=1)
        result_queue: queue.Queue[_InferenceResult] = queue.Queue()
        stop_worker = threading.Event()
        stop_token = object()

        def inference_worker() -> None:
            try:
                try:
                    remote_policy.start()
                except Exception as error:
                    ready_queue.put(("error", error))
                    return
                if trace is not None:
                    trace.record("policy_ready")
                ready_queue.put(("ok", None))
                while not stop_worker.is_set():
                    token = request_queue.get()
                    if token is stop_token:
                        return
                    assert isinstance(token, _InferenceToken)
                    started_ns = time.monotonic_ns()
                    try:
                        observation = robot.get_observation()
                        observation_step = ensemble.latest_sent_step()
                        anchor_step = observation_step + 1
                        if trace is not None:
                            trace.record(
                                "inference_started",
                                request_id=token.request_id,
                                observation_timestep=observation_step,
                                first_action_timestep=anchor_step,
                            )
                        response = remote_policy.infer(
                            self._policy_inputs(observation, args)
                        )
                        chunk = self._action_chunk(response)
                        if len(chunk) != horizon:
                            raise ValueError(
                                "Policy returned an action chunk whose horizon does "
                                f"not match actions_per_chunk: {len(chunk)} != {horizon}"
                            )
                        duration_ms = (
                            time.monotonic_ns() - started_ns
                        ) / 1_000_000.0
                        if trace is not None:
                            trace.record(
                                "inference_returned",
                                request_id=token.request_id,
                                observation_timestep=observation_step,
                                first_action_timestep=anchor_step,
                                duration_ms=duration_ms,
                                chunk_shape=list(chunk.shape),
                                action_min=float(np.min(chunk)),
                                action_max=float(np.max(chunk)),
                            )
                        result_queue.put(
                            _InferenceResult(
                                token=token,
                                status="ok",
                                observation_step=observation_step,
                                anchor_step=anchor_step,
                                chunk=chunk,
                                duration_ms=duration_ms,
                            )
                        )
                    except Exception as error:
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
                                observation_step=None,
                                anchor_step=None,
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
                name="ruri-temporal-ensemble-inference",
                daemon=True,
            )
            worker.start()
            ready_status, ready_payload = ready_queue.get()
            if ready_status == "error":
                raise ready_payload
            robot.start()
            if trace is not None:
                trace.record("controller_ready")
            if max_chunks == 0:
                self.last_run_stats = self._empty_stats(coefficient)
                return
            self.last_run_stats = self._execute(
                robot=robot,
                ensemble=ensemble,
                request_queue=request_queue,
                result_queue=result_queue,
                control_hz=control_hz,
                coefficient=coefficient,
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
                "TemporalEnsembleScheduler failed; log=%s",
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
        ensemble: _ActionEnsemble,
        request_queue: queue.Queue[_InferenceToken | object],
        result_queue: queue.Queue[_InferenceResult],
        control_hz: float,
        coefficient: float,
        max_chunks: int | None,
        trace: _TemporalEnsembleTrace | None,
    ) -> dict[str, int | float | str]:
        period_s = 1.0 / control_hz
        request_id = 0
        inference_in_flight = False
        stats: dict[str, int | float | str] = {
            "actions_sent": 0,
            "ensemble_actions_sent": 0,
            "hold_ticks": 0,
            "chunks_requested": 0,
            "chunks_received": 0,
            "stale_predictions": 0,
            "overlap_predictions": 0,
            "max_contributors": 0,
            "temporal_ensemble_coeff": coefficient,
        }

        def can_request() -> bool:
            return max_chunks is None or request_id < max_chunks

        def trigger() -> None:
            nonlocal request_id, inference_in_flight
            request_id += 1
            token = _InferenceToken(
                request_id=request_id,
                triggered_after_step=ensemble.latest_sent_step(),
            )
            request_queue.put_nowait(token)
            inference_in_flight = True
            stats["chunks_requested"] = int(stats["chunks_requested"]) + 1
            if trace is not None:
                trace.record(
                    "inference_requested",
                    request_id=request_id,
                    triggered_after_timestep=token.triggered_after_step,
                )

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
            update = ensemble.add_chunk(result.chunk, result.anchor_step)
            stats["chunks_received"] = int(stats["chunks_received"]) + 1
            stats["stale_predictions"] = int(stats["stale_predictions"]) + int(
                update["stale"]
            )
            stats["overlap_predictions"] = int(
                stats["overlap_predictions"]
            ) + int(update["overlaps"])
            stats["max_contributors"] = max(
                int(stats["max_contributors"]), int(update["max_contributors"])
            )
            if trace is not None:
                trace.record(
                    "chunk_ensembled",
                    request_id=result.token.request_id,
                    observation_timestep=result.observation_step,
                    first_action_timestep=result.anchor_step,
                    installed_after_timestep=ensemble.latest_sent_step(),
                    stale_predictions=int(update["stale"]),
                    new_predictions=int(update["added"]),
                    overlap_predictions=int(update["overlaps"]),
                    pending_actions=int(update["pending"]),
                    max_contributors=int(update["max_contributors"]),
                    first_overlap=update["first_detail"],
                )
            return True

        trigger()
        receive(block=True)
        next_tick = time.monotonic()

        while True:
            if inference_in_flight:
                receive(block=False)

            no_more_requests = not can_request()
            if (
                ensemble.pending_count() == 0
                and not inference_in_flight
                and no_more_requests
            ):
                break

            selected = ensemble.select_next()
            if selected is None:
                raise RuntimeError(
                    "TemporalEnsembleScheduler has no action available to execute"
                )
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
                        contributors=selected.contributors,
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
            ensemble.commit(selected, accepted_values)
            stats["actions_sent"] = int(stats["actions_sent"]) + 1
            if selected.source == "ensemble":
                stats["ensemble_actions_sent"] = (
                    int(stats["ensemble_actions_sent"]) + 1
                )
            else:
                stats["hold_ticks"] = int(stats["hold_ticks"]) + 1
            if trace is not None:
                trace.record(
                    "action",
                    timestep=selected.timestep,
                    source=selected.source,
                    contributors=selected.contributors,
                    outcome="sent",
                    action=action_values.tolist(),
                    accepted_action=accepted_values.tolist(),
                    clipped=not np.array_equal(action_values, accepted_values),
                    pending_actions=ensemble.pending_count(),
                    inference_in_flight=inference_in_flight,
                    send_duration_ms=(time.monotonic_ns() - send_started_ns)
                    / 1_000_000.0,
                )

            if not inference_in_flight and can_request():
                trigger()

            next_tick += period_s
            remaining_s = next_tick - time.monotonic()
            if remaining_s > 0:
                time.sleep(remaining_s)
            else:
                next_tick = time.monotonic()

        return stats

    @staticmethod
    def _policy_inputs(observation: Any, args: Any) -> dict[str, Any]:
        inputs = dict(observation)
        prompt = get_arg(args, "prompt", None)
        if prompt is not None:
            inputs["prompt"] = prompt
        inputs["context.actions_per_chunk"] = int(
            get_arg(args, "actions_per_chunk", 100)
        )
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
    def _empty_stats(coefficient: float) -> dict[str, int | float | str]:
        return {
            "actions_sent": 0,
            "ensemble_actions_sent": 0,
            "hold_ticks": 0,
            "chunks_requested": 0,
            "chunks_received": 0,
            "stale_predictions": 0,
            "overlap_predictions": 0,
            "max_contributors": 0,
            "temporal_ensemble_coeff": coefficient,
        }

    @staticmethod
    def _start_trace(args: Any) -> _TemporalEnsembleTrace | None:
        if not bool(get_arg(args, "scheduler_log_enabled", True)):
            return None
        configured_path = get_arg(args, "scheduler_log_path", None)
        if configured_path is None:
            log_dir = Path(get_arg(args, "scheduler_log_dir", "logs"))
            timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
            path = log_dir / f"temporal-ensemble-{timestamp}-{os.getpid()}.jsonl"
        else:
            path = Path(configured_path)
        return _TemporalEnsembleTrace(path.expanduser().resolve())
