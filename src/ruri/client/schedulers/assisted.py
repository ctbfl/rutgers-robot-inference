"""Interruptible ordinary-chunk scheduler for an assisted controller.

``step(sample)`` consumes the caller's one observation snapshot. Workflow owns
the periodic loop, recording and operator UI; this object owns only inference,
chunk execution and control-mode requests. Inference never holds the mode lock.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ._actions import clip_target


@dataclass(frozen=True)
class InferenceToken:
    generation: int
    epoch: int
    policy_version: str
    observation_id: str
    sent_at: float
    reference_revision: int
    observation_ns: int


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
        max_target_speed=20.0,
        execution_horizon=None,
        inference_interval_s=0.2,
        max_observation_age_s=1.0,
    ):
        if not policy_version or not prompt:
            raise ValueError("policy_version and prompt are required")
        for name, value in (
            ("control_hz", control_hz),
            ("inference_timeout_s", inference_timeout_s),
            ("max_target_speed", max_target_speed),
            ("inference_interval_s", inference_interval_s),
            ("max_observation_age_s", max_observation_age_s),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.controller, self.policy = controller, policy
        self.policy_version, self.prompt = policy_version, prompt
        self.control_hz, self.timeout = control_hz, inference_timeout_s
        self.max_target_speed = max_target_speed
        self.inference_interval_s = inference_interval_s
        self.max_observation_age_s = max_observation_age_s
        self._last_request_at = 0.0
        self._reference_revision = None
        self.horizon = int(policy.output_chunk_size)
        self.execute = (
            self.horizon if execution_horizon is None else int(execution_horizon)
        )
        if not 1 <= self.execute <= self.horizon:
            raise ValueError("execution_horizon must fit the model's output chunk")
        self._generation = 0
        self._want_policy = False
        self._epoch = None
        self._chunk = deque()
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
        self._last_target = None
        self._chunk_token = None

    def _infer(self):
        while not self._closed.is_set():
            try:
                item = self._requests.get(timeout=0.05)
            except queue.Empty:
                continue
            token, policy, inputs, horizon = item
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
                output = (token, chunk, None, time.monotonic_ns())
            except Exception as exc:
                output = (token, None, exc, time.monotonic_ns())
            while not self._closed.is_set():
                try:
                    self._results.put(output, timeout=0.05)
                    break
                except queue.Full:
                    continue

    def _invalidate(self):
        self._generation += 1
        self._chunk.clear()
        self._chunk_token = None
        self._last_target = None
        self._epoch = None
        self._reference_revision = None
        self._last_request_at = 0.0
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
                # Physical corrections invalidate predictions but keep execution
                # and periodic inference running in the same authority epoch.
                self._generation += 1
                self._chunk.clear()
                self._chunk_token = None
                self._last_target = np.asarray(state["policy_reference"]).copy()
                self._reference_revision = revision
                self._last_request_at = 0.0
            while True:
                try:
                    token, chunk, error, completed_ns = self._results.get_nowait()
                except queue.Empty:
                    break
                if token == self._pending:
                    self._pending = None
                audit = {
                    "inference": token,
                    "action_chunk": chunk,
                    "completed_ns": completed_ns,
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
                self._chunk = deque(enumerate(chunk[: self.execute]))
                self._chunk_token = token
            if (
                self._pending is not None
                and self._want_policy
                and time.monotonic() - self._pending.sent_at > self.timeout
            ):
                self.pause()
                raise TimeoutError("policy inference deadline exceeded; holding")
            if not self._want_policy:
                return self.last_step
            sample_epoch = sample.get("telemetry", {}).get("control", {}).get("epoch")
            if sample_epoch is not None and sample_epoch != state["epoch"]:
                return self.last_step
            if state["mode"] == "HOLD" and not state["armed"]:
                if not state["aligned"]:
                    if not state["aligning"]:
                        self.controller.request_mode("ALIGN")
                    return self.last_step
                state = {**state, **self.controller.request_mode("POLICY")}
                self._epoch = state["epoch"]
                self._last_target = np.asarray(state["policy_reference"]).copy()
            if state["mode"] not in ("HOLD", "POLICY"):
                self._want_policy = False
                return self.last_step
            if (
                sample.get("telemetry", {})
                .get("control", {})
                .get("reference_revision", revision)
                != revision
            ):
                return self.last_step
            now = time.monotonic()
            if (
                self._chunk_token
                and now - self._chunk_token.observation_ns / 1e9
                > self.max_observation_age_s
            ):
                self._chunk.clear()
            if (
                self._pending is None
                and now - self._last_request_at >= self.inference_interval_s
            ):
                token = InferenceToken(
                    self._generation,
                    state["epoch"],
                    self.policy_version,
                    sample["observation_id"],
                    now,
                    revision,
                    sample["monotonic_ns"],
                )
                inputs = {**sample["observation"], "prompt": self.prompt}
                self._requests.put_nowait((token, self.policy, inputs, self.horizon))
                self._pending = token
                self._last_request_at = now
            if self._chunk:
                row, proposal = self._chunk.popleft()
                target, clipped = clip_target(self.controller, proposal)
                max_step = self.max_target_speed / self.control_hz
                target = self._last_target + np.clip(
                    target - self._last_target, -max_step, max_step
                )
                self.controller.send_action(
                    target,
                    reference_revision=revision,
                    observation_ns=self._chunk_token.observation_ns,
                    model_action=True,
                )
                self._last_target = target.copy()
                self.last_step.update(
                    {
                        "command_id": self.controller.last_command_id,
                        "proposal": proposal.copy(),
                        "target": target.copy(),
                        "clipped": clipped,
                        "row": row,
                        "inference": self._chunk_token,
                        "reference_revision": revision,
                        "model_action": True,
                    }
                )
            else:
                self.last_step["waiting"] = True
                # Keep the MIT reference alive while the bounded inference call
                # runs. The separately enforced inference deadline still holds.
                if self._last_target is not None:
                    self.controller.send_action(
                        np.asarray(state["policy_reference"]),
                        reference_revision=revision,
                        model_action=False,
                    )
                    self.last_step["command_id"] = self.controller.last_command_id
                    self.last_step.update(
                        reference_revision=revision, model_action=False
                    )
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
