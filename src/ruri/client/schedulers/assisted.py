"""RTC action timeline with local interruption and correction invalidation.

``step(sample)`` consumes the caller's one observation snapshot. The caller owns
the periodic loop, optional recording and operator UI; this object owns inference,
chunk execution and control-mode requests. Inference never holds the mode lock.
"""

from __future__ import annotations

import math
import logging
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np

from ._actions import clip_target
from .rtc import RTCScheduler, _ActionTimeline
from ruri.client._args import get_arg


logger = logging.getLogger(__name__)


def policy_version(policy):
    metadata = policy.metadata
    return str(metadata.get("policy_version")
               or metadata.get("policy", {}).get("checkpoint_path") or policy.endpoint)


@dataclass(frozen=True)
class InferenceToken:
    generation: int
    epoch: int
    policy_version: str
    observation_id: str
    sent_at: float
    reference_revision: int
    observation_ns: int
    request_id: int
    anchor_step: int
    request_last_step: int
    consumed_steps: int
    estimated_delay_steps: int
    rtc_requested: bool
    scheduler_instance_id: str = ""


class AssistedScheduler:
    def __init__(
        self,
        controller,
        policy,
        *,
        policy_version,
        prompt,
        control_hz=30.0,
        inference_timeout_s=5.0,
        execution_horizon=None,
        rtc_latency_window=10,
        rtc_initial_delay_steps=None,
        max_observation_age_s=1.0,
        rtc_during_policy=False,
        request_adapter=None,
        request_interval_steps=None,
    ):
        if not policy_version or not prompt:
            raise ValueError("policy_version and prompt are required")
        for name, value in (
            ("control_hz", control_hz),
            ("inference_timeout_s", inference_timeout_s),
            ("max_observation_age_s", max_observation_age_s),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.controller, self.policy = controller, policy
        # Optional data/protocol adapter. Default execution and RTC behavior
        # remain unchanged; adapters cannot replace the action timeline.
        self.request_adapter = request_adapter
        self.scheduler_instance_id = uuid.uuid4().hex
        self.policy_version, self.prompt = policy_version, prompt
        self.control_hz, self.timeout = control_hz, inference_timeout_s
        self.max_observation_age_s = max_observation_age_s
        if type(rtc_during_policy) is not bool:
            raise ValueError("rtc_during_policy must be a boolean")
        self.rtc_during_policy = rtc_during_policy
        self._rtc_segment = None
        self._reference_revision = None
        self.horizon = int(policy.output_chunk_size)
        if request_interval_steps is not None and (
                type(request_interval_steps) is not int or
                not 1 <= request_interval_steps < self.horizon):
            raise ValueError("request_interval_steps must be null or an integer below the chunk horizon")
        self.request_interval_steps = request_interval_steps
        self._sample_step = 0
        self._request_grid = None
        self._last_request_sample_step = None
        self.execute = (
            min(5, self.horizon)
            if execution_horizon is None
            else int(execution_horizon)
        )
        if not 1 <= self.execute <= self.horizon:
            raise ValueError("execution_horizon must fit the model's output chunk")
        if type(rtc_latency_window) is not int or rtc_latency_window < 1:
            raise ValueError("rtc_latency_window must be a positive integer")
        initial_delay = (
            min(4, self.horizon)
            if rtc_initial_delay_steps is None
            else rtc_initial_delay_steps
        )
        if type(initial_delay) is not int or not 0 <= initial_delay <= self.horizon:
            raise ValueError("rtc_initial_delay_steps must fit the action horizon")
        self._delay_history = deque([initial_delay], maxlen=rtc_latency_window)
        self._initial_delay = initial_delay
        self._request_id = 0
        self._generation = 0
        self._want_policy = False
        self._epoch = None
        self._timeline = _ActionTimeline()
        self._pending = None
        self._requests = queue.Queue(maxsize=1)
        self._results = queue.Queue(maxsize=2)
        self._closed = threading.Event()
        self._lock = threading.RLock()
        self._thread = threading.Thread(
            target=self._infer, name="assisted-inference", daemon=True
        )
        self._thread.start()
        self.last_step = {}
        self.discarded_results = 0
        self._chunk_token = None

    def _infer(self):
        while not self._closed.is_set():
            try:
                item = self._requests.get(timeout=0.05)
            except queue.Empty:
                continue
            token, policy, inputs, horizon, timeline = item
            try:
                result = policy.infer(inputs)
                if (
                    result.get("policy_version", token.policy_version)
                    != token.policy_version
                ):
                    raise ValueError(
                        "inference response came from a different policy version"
                    )
                chunk = np.asarray(result["action_chunk"], dtype=np.float32)
                if chunk.shape != (horizon, 7) or not np.all(np.isfinite(chunk)):
                    raise ValueError(
                        f"policy returned invalid chunk {chunk.shape}; expected {(horizon, 7)}"
                    )
                if token.rtc_requested and result.get("rtc.applied") is not True:
                    raise ValueError(
                        f"remote policy did not apply RTC: {result.get('rtc.reason')}"
                    )
                output = (
                    token,
                    chunk,
                    None,
                    time.monotonic_ns(),
                    timeline.latest_sent_step(),
                    {k: v for k, v in result.items() if k.startswith("rtc.")},
                )
            except Exception as exc:
                output = (
                    token,
                    None,
                    exc,
                    time.monotonic_ns(),
                    timeline.latest_sent_step(),
                    {},
                )
            while not self._closed.is_set():
                try:
                    self._results.put(output, timeout=0.05)
                    break
                except queue.Full:
                    continue

    def _invalidate(self):
        self._generation += 1
        self._timeline = _ActionTimeline()
        self._chunk_token = None
        self._epoch = None
        self._reference_revision = None
        # Keep the in-flight request known until it returns: at most one GPU
        # call is ever active and obsolete results cannot create request floods.
        self._rtc_segment = None

    def pause(self):
        with self._lock:
            self._want_policy = False
            self._invalidate()
            return self.controller.request_mode("HOLD")

    def takeover(self):
        with self._lock:
            self._want_policy = False
            self._invalidate()
            return self.controller.request_mode("HUMAN")

    def resume(self):
        with self._lock:
            state = self.controller.status()
            if state["mode"] != "HOLD":
                raise RuntimeError(
                    "release human control into HOLD before resuming policy"
                )
            self._invalidate()
            self._want_policy = True

    def replace_policy(self, policy, version, *, prompt=None, allow_human=False):
        with self._lock:
            allowed = ("HOLD", "HUMAN") if allow_human else ("HOLD",)
            if self._want_policy or self.controller.status()["mode"] not in allowed:
                raise RuntimeError("pause before replacing policy")
            horizon = int(policy.output_chunk_size)
            if horizon != self.horizon:
                raise ValueError(
                    "policy replacement changed the configured action horizon"
                )
            if not version:
                raise ValueError("immutable policy version is required")
            self._invalidate()
            self.policy, self.policy_version = policy, version
            self._delay_history.clear()
            self._delay_history.append(self._initial_delay)
            if prompt is not None:
                self.prompt = prompt

    def _rtc_allowed(self, state):
        schema = state.get("control_schema")
        if schema not in ("ruri_shared_reference_v1", "ruri_gravity_handoff_v1"):
            return True
        # RTC conditions the next chunk on targets that will still be submitted
        # during inference, as in the ordinary RTC client. That assumption is
        # valid only in autonomous tracking, not unloading, dragging or recovery.
        return (self.rtc_during_policy and schema == "ruri_gravity_handoff_v1"
                and state.get("authority_phase") == "policy"
                and not state.get("intervention_active", False)
                and state.get("human_weight", 0.) == 0.)

    def _request(self, sample, state):
        if self._pending is not None:
            return
        grid = (self._generation, state["epoch"], state["reference_revision"])
        if self.request_interval_steps is not None:
            # Anchor to collection ticks, not chunk consumption or completion
            # time. A late response skips a grid point; it never shifts phase.
            if self._request_grid is not None and self._request_grid[:3] == grid:
                if ((self._sample_step - self._request_grid[3]) % self.request_interval_steps
                        or self._last_request_sample_step == self._sample_step):
                    return
        elif (
            self._timeline.pending_count()
            and self._timeline.current_chunk_consumed_steps() < self.execute
        ):
            return
        last_step, tail, consumed = self._timeline.request_snapshot()
        if not self._rtc_allowed(state):
            tail, consumed = None, 0
        delay = min(max(self._delay_history), 0 if tail is None else len(tail))
        self._request_id += 1
        if self._request_id == 1:
            logger.info("Starting policy inference for task %r", self.prompt)
        token = InferenceToken(
            self._generation,
            state["epoch"],
            self.policy_version,
            sample["observation_id"],
            time.monotonic(),
            self._reference_revision,
            sample["monotonic_ns"],
            self._request_id,
            last_step + 1,
            last_step,
            consumed,
            delay,
            tail is not None,
            self.scheduler_instance_id,
        )
        inputs = RTCScheduler._policy_inputs(
            observation=sample["observation"],
            args={"prompt": self.prompt},
            estimated_delay_steps=delay,
            previous_tail=tail,
            consumed_steps=consumed,
        )
        policy = self.policy
        if self.request_adapter is not None:
            policy, inputs = self.request_adapter.prepare(token, policy, inputs)
        self._requests.put_nowait((token, policy, inputs, self.horizon, self._timeline))
        self._pending = token
        if self.request_interval_steps is not None:
            if self._request_grid is None or self._request_grid[:3] != grid:
                self._request_grid = (*grid, self._sample_step)
            self._last_request_sample_step = self._sample_step

    def step(self, sample):
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("scheduler is closed")
            self._sample_step += 1
            self.last_step = {
                "waiting": False,
                "policy_version": self.policy_version,
                "inference_results": [],
            }
            state = self.controller.status()
            if state.get("phase", "engaged") != "engaged" or state["mode"] == "FAULT":
                self._want_policy = False
                self._invalidate()
                raise RuntimeError(f"assisted worker fault: {state.get('reason')}")
            if self._epoch is not None and state["epoch"] != self._epoch:
                self._want_policy = False
                self._invalidate()
            revision = state["reference_revision"]
            rtc_segment = self._rtc_allowed(state)
            if self._want_policy and (revision != self._reference_revision or (
                    self._rtc_segment is not None and rtc_segment != self._rtc_segment)):
                # A physical correction starts a fresh RTC segment, never a
                # control-mode change. Nothing from the previous prefix survives.
                self._generation += 1
                self._timeline = _ActionTimeline()
                self._chunk_token = None
                self._reference_revision = revision
            self._rtc_segment = rtc_segment
            while True:
                try:
                    token, chunk, error, completed_ns, returned_step, rtc = (
                        self._results.get_nowait()
                    )
                except queue.Empty:
                    break
                if token == self._pending:
                    self._pending = None
                audit = {
                    "inference": token,
                    "action_chunk": chunk,
                    "completed_ns": completed_ns,
                    "returned_after_step": returned_step,
                    "rtc": rtc,
                    "disposition": "accepted",
                    "error": None if error is None else str(error),
                }
                self.last_step["inference_results"].append(audit)
                if (
                    token.generation != self._generation
                    or token.epoch != state["epoch"]
                    or token.policy_version != self.policy_version
                    or token.reference_revision != revision
                    or not self._want_policy
                ):
                    audit["disposition"] = "discarded_obsolete"
                    self.discarded_results += 1
                    continue
                if error is not None:
                    audit["disposition"] = "failed"
                    self.pause()
                    raise RuntimeError("policy inference failed; holding") from error
                if (
                    time.monotonic() - token.observation_ns / 1e9
                    > self.max_observation_age_s
                ):
                    audit["disposition"] = "discarded_stale"
                    self.discarded_results += 1
                    continue
                actual_delay = max(0, returned_step - token.request_last_step)
                duration = completed_ns / 1e9 - token.sent_at
                self._delay_history.append(
                    max(actual_delay, math.ceil(duration * self.control_hz))
                )
                if state.get("intervention_active") and state["mode"] == "POLICY":
                    # Continue querying during intervention. The worker ignores
                    # proposals in gravity mode and uses fresh ones in recovery.
                    target, _ = clip_target(self.controller, chunk[0].copy())
                    self.controller.send_action(
                        target, reference_revision=revision,
                        observation_ns=token.observation_ns, model_action=True,
                    )
                    audit["disposition"] = "blended_proposal"
                    continue
                install = self._timeline.install(
                    chunk=chunk,
                    chunk_id=token.request_id,
                    anchor_step=(token.anchor_step if self._rtc_allowed(state)
                                 else self._timeline.latest_sent_step()+1),
                )
                audit["install"] = install
                if install["installed"]:
                    self._chunk_token = token
                else:
                    audit["disposition"] = "discarded_expired"
                    self.discarded_results += 1
            if (
                self._pending is not None
                and self._want_policy
                and time.monotonic() - self._pending.sent_at > self.timeout
            ):
                self.pause()
                raise TimeoutError("policy inference deadline exceeded; holding")
            if not self._want_policy:
                return self.last_step
            observed_control = sample.get("telemetry", {}).get("control", {})
            if observed_control.get("epoch", state["epoch"]) != state["epoch"]:
                return self.last_step
            if state["mode"] == "HOLD" and not state["armed"]:
                state = {**state, **self.controller.request_mode("POLICY")}
                self._epoch = state["epoch"]
            if state["mode"] not in ("HOLD", "POLICY"):
                self._want_policy = False
                return self.last_step
            if observed_control.get("reference_revision", revision) != revision:
                return self.last_step
            if (
                self._chunk_token is not None
                and time.monotonic() - self._chunk_token.observation_ns / 1e9
                > self.max_observation_age_s
            ):
                self._generation += 1
                self._timeline = _ActionTimeline()
                self._chunk_token = None
            self._request(sample, state)
            selected = self._timeline.select_next()
            if selected is None:
                # Bootstrap/rebase waiting does not advance the action timeline.
                # Thus the first fresh unconditioned chunk starts at row zero.
                self.controller.send_action(
                    np.asarray(state["policy_reference"]),
                    reference_revision=revision,
                    model_action=False,
                )
                self.last_step.update(
                    waiting=True,
                    command_id=self.controller.last_command_id,
                    reference_revision=revision,
                    model_action=False,
                )
            else:
                target, clipped = clip_target(self.controller, selected.value.copy())
                model_action = selected.source == "policy"
                self.controller.send_action(
                    target,
                    reference_revision=revision,
                    observation_ns=self._chunk_token.observation_ns,
                    model_action=model_action,
                )
                self._timeline.commit(selected, target)
                self.last_step.update(
                    command_id=self.controller.last_command_id,
                    proposal=selected.value.copy(),
                    target=target.copy(),
                    clipped=clipped,
                    row=selected.chunk_offset,
                    timestep=selected.timestep,
                    inference=self._chunk_token,
                    reference_revision=revision,
                    model_action=model_action,
                    waiting=not model_action,
                )
                # Same trigger rule and complete raw prefix as RTCScheduler.
                self._request(sample, state)
            return self.last_step

    def close(self):
        with self._lock:
            if self._closed.is_set():
                return
            self._want_policy = False
            self._invalidate()
            try:
                state = self.controller.status()
                # Recovery owns the arm until disconnect waits for it to finish.
                # That worker no longer acknowledges normal mode requests.
                if (state.get("phase", "engaged") == "engaged"
                        and state.get("mode") not in ("FAULT", "STOPPED")):
                    self.controller.request_mode("HOLD")
            except (RuntimeError, TimeoutError, OSError) as exc:
                # Recovery can also begin between the status read and HOLD.
                # Cleanup must not replace the fault that caused this shutdown.
                logger.warning("Scheduler cleanup could not confirm HOLD: %s", exc)
            finally:
                self._closed.set()
        self._thread.join(timeout=0.2)


def run_assisted_rtc(*, controller, policy, args):
    """Continuous remote RTC execution without dataset or trace writers."""
    seconds = float(get_arg(args, "seconds", 0.))
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and nonnegative")
    hz = float(get_arg(args, "control_hz", 30.))
    robot, scheduler = controller(args), None
    stats = {"steps": 0, "inference_results": 0, "rtc_results": 0}
    try:
        scheduler = AssistedScheduler(
            robot, policy, policy_version=policy_version(policy),
            prompt=get_arg(args, "prompt"), control_hz=hz,
            execution_horizon=get_arg(args, "execution_horizon", 5),
            rtc_latency_window=get_arg(args, "rtc_latency_window", 10),
            rtc_initial_delay_steps=get_arg(args, "rtc_initial_delay_steps", 4),
            inference_timeout_s=get_arg(args, "policy_timeout_s", 5.),
            rtc_during_policy=True,
        )
        robot.start()
        scheduler.resume()
        logger.info("Dual Piper RTC running: right leader intervention, left follower observation; no recording")
        began = next_tick = time.monotonic()
        previous_phase = None
        while not seconds or time.monotonic()-began < seconds:
            sample = robot.get_sample()
            if "observation.joint_effort" not in policy.metadata.get("inputs", {}):
                sample["observation"].pop("observation.joint_effort", None)
            step = scheduler.step(sample)
            stats["steps"] += 1
            for result in step["inference_results"]:
                stats["inference_results"] += 1
                stats["rtc_results"] += int(result["rtc"].get("rtc.applied") is True)
            state = robot.status()
            if state["mode"] == "HOLD" and not state["armed"]:
                raise RuntimeError(f"worker entered HOLD: {state.get('reason')}")
            phase = state["authority_phase"]
            if phase != previous_phase:
                logger.info("Authority: %s; alpha=%.2f", phase, state["human_weight"])
                previous_phase = phase
            next_tick += 1./hz
            time.sleep(max(0., next_tick-time.monotonic()))
            next_tick = max(next_tick, time.monotonic())
        return stats
    finally:
        try:
            if scheduler is not None:
                scheduler.close()
        finally:
            robot.stop()
