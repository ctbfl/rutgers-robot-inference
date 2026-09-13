"""RTC action timeline with local interruption and correction invalidation.

``step(sample)`` consumes the caller's one observation snapshot. Workflow owns
the periodic loop, recording and operator UI; this object owns only inference,
chunk execution and control-mode requests. Inference never holds the mode lock.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ._actions import clip_target
from .rtc import RTCScheduler, _ActionTimeline


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
        self.policy_version, self.prompt = policy_version, prompt
        self.control_hz, self.timeout = control_hz, inference_timeout_s
        self.max_observation_age_s = max_observation_age_s
        self._reference_revision = None
        self.horizon = int(policy.output_chunk_size)
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

    def replace_policy(self, policy, version, *, prompt=None):
        with self._lock:
            if self._want_policy or self.controller.status()["mode"] != "HOLD":
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
            if prompt is not None:
                self.prompt = prompt

    def _request(self, sample, state):
        if self._pending is not None:
            return
        if (
            self._timeline.pending_count()
            and self._timeline.current_chunk_consumed_steps() < self.execute
        ):
            return
        last_step, tail, consumed = self._timeline.request_snapshot()
        delay = min(max(self._delay_history), 0 if tail is None else len(tail))
        self._request_id += 1
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
        )
        inputs = RTCScheduler._policy_inputs(
            observation=sample["observation"],
            args={"prompt": self.prompt},
            estimated_delay_steps=delay,
            previous_tail=tail,
            consumed_steps=consumed,
        )
        self._requests.put_nowait(
            (token, self.policy, inputs, self.horizon, self._timeline)
        )
        self._pending = token

    def step(self, sample):
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("scheduler is closed")
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
            if self._want_policy and revision != self._reference_revision:
                # A physical correction starts a fresh RTC segment, never a
                # control-mode change. Nothing from the previous prefix survives.
                self._generation += 1
                self._timeline = _ActionTimeline()
                self._chunk_token = None
                self._reference_revision = revision
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
                install = self._timeline.install(
                    chunk=chunk,
                    chunk_id=token.request_id,
                    anchor_step=token.anchor_step,
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
                if not state["aligned"]:
                    if not state["aligning"]:
                        self.controller.request_mode("ALIGN")
                    return self.last_step
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
            try:
                self.pause()
            finally:
                self._closed.set()
        self._thread.join(timeout=0.2)
