"""RTC scheduler whose prompt can be changed without restarting the client.

It runs the same request/execute/replace timeline as ``RTCScheduler`` and adds
an operator loop on the terminal:

* The first prompt is typed before the Controller starts, so the arm is not
  enabled while someone is typing.
* ``q`` pauses. Every unsent action is dropped, a chunk still being computed is
  discarded when it returns, and the last sent target is re-sent every control
  tick. The re-sending is load-bearing: once commanded, the MIT worker aborts to
  hold-then-home and exits when no command arrives for its command timeout.
* Entering a new prompt resumes the way the run started. The first request
  carries no RTC prefix, and its chunk is executed from row 0 however many hold
  ticks were sent while it was being computed.
* ``Esc`` or Ctrl-C ends the run through the usual ``controller.stop()`` path.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ruri.client._args import get_arg
from ruri.client.schedulers._actions import clip_target
from ruri.client.schedulers.rtc import (
    RTCScheduler,
    _ActionTimeline,
    _InferenceResult,
    _RTCTrace,
)
from ruri.client.utils.prompt_terminal import PromptTerminal


logger = logging.getLogger(__name__)

FIRST_PROMPT_MESSAGE = (
    "Type the task prompt and press Enter; the arm starts after that. "
    "Esc or Ctrl-C quits."
)
PAUSED_PROMPT_MESSAGE = (
    "Paused, holding the last target. Type a new prompt and press Enter. "
    "Esc or Ctrl-C quits."
)

# Requests that begin a motion segment: no RTC prefix, executed from row 0.
_FRESH_REASONS = frozenset({"initial", "resume"})


@dataclass(frozen=True)
class _PromptToken:
    request_id: int
    reason: str
    trigger_step: int
    triggered_ns: int
    prompt: str
    epoch: int


class _PausableTimeline(_ActionTimeline):
    """Action timeline that can drop its unsent actions on a pause."""

    def clear_future(self) -> int:
        """Drop every unsent action; the last sent target remains the hold."""
        with self._lock:
            dropped = len(self._future)
            self._future = {}
            return dropped


class InteractiveRTCScheduler:
    """RTC execution with a pause and re-prompt loop driven from the terminal."""

    def __init__(self) -> None:
        self.last_run_stats: dict[str, int | float | str] | None = None
        self.last_log_path: Path | None = None

    def run(
        self,
        controller: Callable[[Any], Any],
        policy: Any,
        *,
        args: Any,
        terminal: Any | None = None,
    ) -> None:
        """Ask for a prompt, start the arm, then run until Esc or Ctrl-C.

        ``terminal`` defaults to a ``PromptTerminal`` on this process's TTY.
        """
        control_hz = float(get_arg(args, "control_hz", 30.0))
        execution_horizon = int(get_arg(args, "execution_horizon", 5))
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
        if latency_window <= 0:
            raise ValueError("rtc_latency_window must be positive")
        if (
            configured_initial_delay_steps is not None
            and configured_initial_delay_steps < 0
        ):
            raise ValueError("rtc_initial_delay_steps cannot be negative")

        robot = controller(args)
        timeline = _PausableTimeline()
        trace = RTCScheduler._start_trace(args)
        if trace is not None:
            self.last_log_path = trace.path
            logger.info("InteractiveRTCScheduler log: %s", trace.path)
            trace.record(
                "run_start",
                scheduler="interactive_rtc",
                control_hz=control_hz,
                execution_horizon=execution_horizon,
                rtc_initial_delay_steps=configured_initial_delay_steps,
            )

        ready_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        request_queue: queue.Queue[_PromptToken | object] = queue.Queue(maxsize=1)
        result_queue: queue.Queue[_InferenceResult] = queue.Queue(maxsize=1)
        stop_worker = threading.Event()
        stop_token = object()

        def inference_worker() -> None:
            try:
                try:
                    output_chunk_size = int(policy.output_chunk_size)
                except Exception as error:
                    ready_queue.put(("error", error))
                    return
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
                ready_queue.put(("ok", (output_chunk_size, initial_delay_steps)))

                while not stop_worker.is_set():
                    token = request_queue.get()
                    if token is stop_token:
                        return
                    assert isinstance(token, _PromptToken)
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
                        inputs = RTCScheduler._policy_inputs(
                            observation=observation,
                            args={"prompt": token.prompt},
                            estimated_delay_steps=estimated_delay,
                            previous_tail=previous_tail,
                            consumed_steps=consumed_steps,
                        )
                        sent_ns = time.monotonic_ns()
                        if trace is not None:
                            trace.record(
                                "inference_started",
                                request_id=token.request_id,
                                reason=token.reason,
                                epoch=token.epoch,
                                prompt=token.prompt,
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
                        response = policy.infer(inputs)
                        chunk = RTCScheduler._action_chunk(response)
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
                                epoch=token.epoch,
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
                if trace is not None:
                    trace.record("inference_worker_stopped")

        worker: threading.Thread | None = None
        owned_terminal: PromptTerminal | None = None
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

            if terminal is None:
                owned_terminal = terminal = PromptTerminal()
                owned_terminal.open()
            prompt = self._first_prompt(terminal)
            if prompt is None:
                logger.info("Quit before the arm was started")
                if trace is not None:
                    trace.record("exit_requested", phase="first_prompt")
                self.last_run_stats = self._new_stats(execution_horizon)
                return
            if trace is not None:
                trace.record("prompt_set", prompt=prompt, epoch=0)
            logger.info("Prompt: %r", prompt)

            robot.start()
            if trace is not None:
                trace.record("controller_ready")
            logger.info(
                "Running. Press q to pause and change the prompt; "
                "Esc or Ctrl-C quits."
            )

            self.last_run_stats = self._execute(
                robot=robot,
                timeline=timeline,
                terminal=terminal,
                request_queue=request_queue,
                result_queue=result_queue,
                control_hz=control_hz,
                execution_horizon=execution_horizon,
                prompt=prompt,
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
                "InteractiveRTCScheduler failed; log=%s",
                None if trace is None else trace.path,
            )
            raise
        finally:
            try:
                if owned_terminal is not None:
                    owned_terminal.close()
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
    def _first_prompt(terminal: Any) -> str | None:
        """Block until the operator enters a prompt; None means they quit."""
        terminal.request_prompt(FIRST_PROMPT_MESSAGE)
        while True:
            event = terminal.poll_event(0.1)
            if event is None:
                continue
            if event.kind == "exit":
                return None
            if event.kind == "prompt":
                return event.text

    @staticmethod
    def _execute(
        *,
        robot: Any,
        timeline: _PausableTimeline,
        terminal: Any,
        request_queue: queue.Queue[_PromptToken | object],
        result_queue: queue.Queue[_InferenceResult],
        control_hz: float,
        execution_horizon: int,
        prompt: str,
        trace: _RTCTrace | None,
    ) -> dict[str, int | float | str]:
        period_s = 1.0 / control_hz
        request_id = 0
        # Bumped on every pause; a chunk requested under an older epoch is
        # discarded when it returns.
        epoch = 0
        paused = False
        needs_fresh_chunk = True
        inference_in_flight = False
        stats = InteractiveRTCScheduler._new_stats(execution_horizon)

        def bump(name: str, amount: int = 1) -> None:
            stats[name] = int(stats[name]) + amount

        def trigger(reason: str) -> None:
            nonlocal request_id, inference_in_flight
            request_id += 1
            token = _PromptToken(
                request_id=request_id,
                reason=reason,
                trigger_step=timeline.latest_sent_step(),
                triggered_ns=time.monotonic_ns(),
                prompt=prompt,
                epoch=epoch,
            )
            if trace is not None:
                trace.record(
                    "inference_triggered",
                    request_id=request_id,
                    reason=reason,
                    epoch=epoch,
                    trigger_step=token.trigger_step,
                    queue_size=timeline.pending_count(),
                    consumed_steps=timeline.current_chunk_consumed_steps(),
                )
            request_queue.put_nowait(token)
            inference_in_flight = True
            bump("chunks_requested")

        def maybe_trigger() -> None:
            if paused or inference_in_flight:
                return
            if needs_fresh_chunk:
                trigger("resume" if int(stats["actions_sent"]) else "initial")
                return
            pending = timeline.pending_count()
            if (
                timeline.current_chunk_consumed_steps() >= execution_horizon
                or pending == 0
            ):
                trigger("execution_horizon" if pending else "queue_empty")

        def receive(timeout_s: float) -> None:
            nonlocal inference_in_flight, needs_fresh_chunk
            try:
                result = (
                    result_queue.get(timeout=timeout_s)
                    if timeout_s > 0
                    else result_queue.get_nowait()
                )
            except queue.Empty:
                return
            inference_in_flight = False
            if result.status == "error":
                assert result.error is not None
                raise result.error
            token = result.token
            if token.epoch != epoch:
                # Requested before a pause; the operator has replaced its prompt.
                bump("discarded_chunks")
                if trace is not None:
                    trace.record(
                        "chunk_discarded",
                        request_id=token.request_id,
                        request_epoch=token.epoch,
                        epoch=epoch,
                    )
                return
            assert result.chunk is not None
            assert result.anchor_step is not None
            assert result.request_last_sent_step is not None
            bump("chunks_received")
            fresh = token.reason in _FRESH_REASONS
            # A fresh chunk begins its segment as the first chunk of a run does:
            # row 0 is the next action sent. Anchoring it where it was requested
            # would expire the rows covered by the hold ticks sent meanwhile and
            # jump the target ahead of a standing arm.
            anchor_step = (
                timeline.latest_sent_step() + 1 if fresh else result.anchor_step
            )
            install = timeline.install(
                chunk=result.chunk,
                chunk_id=token.request_id,
                anchor_step=anchor_step,
            )
            if fresh:
                needs_fresh_chunk = False
            expired = int(install["expired"] or 0)
            installed = int(install["installed"] or 0)
            replaced = int(install["replaced"] or 0)
            bump("expired_actions", expired)
            bump("replaced_actions", replaced)
            actual_delay = max(
                0, timeline.latest_sent_step() - result.request_last_sent_step
            )
            stats["max_actual_delay_steps"] = max(
                int(stats["max_actual_delay_steps"]), actual_delay
            )
            bump("chunks_installed" if installed else "stale_chunks")
            first_step, last_step = timeline.bounds()
            if trace is not None:
                trace.record(
                    "chunk_installed" if installed else "chunk_stale",
                    request_id=token.request_id,
                    reason=token.reason,
                    epoch=epoch,
                    anchor_step=anchor_step,
                    request_anchor_step=result.anchor_step,
                    returned_after_step=result.returned_after_step,
                    installed_after_step=timeline.latest_sent_step(),
                    actual_delay_steps=actual_delay,
                    consumed_steps=result.consumed_steps,
                    rtc_applied=result.rtc_applied,
                    expired_actions=expired,
                    installed_actions=installed,
                    replaced_actions=replaced,
                    queue_first_step=first_step,
                    queue_last_step=last_step,
                )

        def handle_events() -> bool:
            """Apply queued keys; False once the operator asked to quit."""
            nonlocal paused, epoch, needs_fresh_chunk, prompt
            while (event := terminal.poll_event(0.0)) is not None:
                if event.kind == "exit":
                    logger.info("Quit requested")
                    if trace is not None:
                        trace.record(
                            "exit_requested",
                            phase="paused" if paused else "running",
                            epoch=epoch,
                        )
                    return False
                if event.kind == "pause" and not paused:
                    paused = True
                    epoch += 1
                    needs_fresh_chunk = True
                    dropped = timeline.clear_future()
                    bump("pauses")
                    bump("dropped_actions", dropped)
                    if trace is not None:
                        trace.record(
                            "paused",
                            epoch=epoch,
                            dropped_actions=dropped,
                            last_sent_step=timeline.latest_sent_step(),
                            inference_in_flight=inference_in_flight,
                        )
                    terminal.request_prompt(PAUSED_PROMPT_MESSAGE)
                elif event.kind == "prompt" and paused:
                    prompt = event.text
                    paused = False
                    bump("prompt_changes")
                    if trace is not None:
                        trace.record("prompt_set", prompt=prompt, epoch=epoch)
                    logger.info("Prompt: %r", prompt)
            return True

        next_tick = time.monotonic()
        while handle_events():
            if inference_in_flight:
                receive(0.0)
            maybe_trigger()

            selected = timeline.select_next()
            if selected is None:
                # Nothing has been commanded yet. The MIT worker holds home by
                # itself and does not arm its command watchdog before the first
                # command, so there is no target to re-send while waiting.
                if inference_in_flight:
                    receive(period_s)
                else:
                    time.sleep(period_s)
                # Start the control clock when the first chunk lands, as
                # RTCScheduler does, so its first two actions cannot burst.
                next_tick = time.monotonic()
                continue

            action_values = selected.value.copy()
            target, clipped = clip_target(robot, action_values)
            send_started_ns = time.monotonic_ns()
            try:
                robot.send_action(target)
            except Exception as error:
                if trace is not None:
                    trace.record(
                        "action",
                        timestep=selected.timestep,
                        source=selected.source,
                        chunk_id=selected.chunk_id,
                        chunk_offset=selected.chunk_offset,
                        paused=paused,
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
            timeline.commit(selected, target)
            bump("actions_sent")
            bump("policy_actions_sent" if selected.source == "policy" else "hold_ticks")
            if paused:
                bump("paused_ticks")
            if trace is not None:
                trace.record(
                    "action",
                    timestep=selected.timestep,
                    source=selected.source,
                    chunk_id=selected.chunk_id,
                    chunk_offset=selected.chunk_offset,
                    paused=paused,
                    epoch=epoch,
                    outcome="sent",
                    action=target.tolist(),
                    pre_clip_action=(action_values.tolist() if clipped else None),
                    clipped=clipped,
                    queue_size=timeline.pending_count(),
                    inference_in_flight=inference_in_flight,
                    send_duration_ms=(time.monotonic_ns() - send_started_ns)
                    / 1_000_000.0,
                )

            maybe_trigger()

            next_tick += period_s
            remaining_s = next_tick - time.monotonic()
            if remaining_s > 0:
                time.sleep(remaining_s)
            else:
                next_tick = time.monotonic()

        return stats

    @staticmethod
    def _new_stats(execution_horizon: int) -> dict[str, int | float | str]:
        return {
            "actions_sent": 0,
            "policy_actions_sent": 0,
            "hold_ticks": 0,
            "paused_ticks": 0,
            "chunks_requested": 0,
            "chunks_received": 0,
            "chunks_installed": 0,
            "stale_chunks": 0,
            "discarded_chunks": 0,
            "expired_actions": 0,
            "replaced_actions": 0,
            "dropped_actions": 0,
            "max_actual_delay_steps": 0,
            "pauses": 0,
            "prompt_changes": 0,
            "execution_horizon": execution_horizon,
        }
