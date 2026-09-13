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
    ):
        if not policy_version or not prompt:
            raise ValueError("policy_version and prompt are required")
        for name, value in (
            ("control_hz", control_hz),
            ("inference_timeout_s", inference_timeout_s),
            ("max_target_speed", max_target_speed),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.controller, self.policy = controller, policy
        self.policy_version, self.prompt = policy_version, prompt
        self.control_hz, self.timeout = control_hz, inference_timeout_s
        self.max_target_speed = max_target_speed
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
                output = (token, chunk, None)
            except Exception as exc:
                output = (token, None, exc)
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
            state = self.controller.status()
            if state.get("phase", "engaged") != "engaged" or state["mode"] == "FAULT":
                self._want_policy = False
                self._invalidate()
                raise RuntimeError(f"assisted worker fault: {state.get('reason')}")
            if self._epoch is not None and state["epoch"] != self._epoch:
                self._want_policy = False
                self._invalidate()
            while True:
                try:
                    token, chunk, error = self._results.get_nowait()
                except queue.Empty:
                    break
                if token == self._pending:
                    self._pending = None
                if (
                    token.generation != self._generation
                    or token.epoch != state["epoch"]
                    or token.policy_version != self.policy_version
                    or not self._want_policy
                ):
                    self.discarded_results += 1
                    continue
                if error is not None:
                    self.pause()
                    raise RuntimeError("policy inference failed; holding") from error
                self._chunk = deque(enumerate(chunk[: self.execute]))
                self._chunk_token = token
            if (
                self._pending is not None
                and self._want_policy
                and self._pending.generation == self._generation
                and time.monotonic() - self._pending.sent_at > self.timeout
            ):
                self.pause()
                raise TimeoutError("policy inference deadline exceeded; holding")
            self.last_step = {"waiting": False, "policy_version": self.policy_version}
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
                self._last_target = np.asarray(state["target"]).copy()
            if state["mode"] not in ("HOLD", "POLICY"):
                self._want_policy = False
                return self.last_step
            if self._chunk:
                row, proposal = self._chunk.popleft()
                target, clipped = clip_target(self.controller, proposal)
                max_step = self.max_target_speed / self.control_hz
                target = self._last_target + np.clip(
                    target - self._last_target, -max_step, max_step
                )
                self.controller.send_action(target)
                self._last_target = target.copy()
                self.last_step.update(
                    {
                        "command_id": self.controller.last_command_id,
                        "proposal": proposal.copy(),
                        "target": target.copy(),
                        "clipped": clipped,
                        "row": row,
                        "inference": self._chunk_token,
                    }
                )
            else:
                self.last_step["waiting"] = True
                if self._pending is None:
                    token = InferenceToken(
                        self._generation,
                        state["epoch"],
                        self.policy_version,
                        sample["observation_id"],
                        time.monotonic(),
                    )
                    inputs = {**sample["observation"], "prompt": self.prompt}
                    self._requests.put_nowait(
                        (token, self.policy, inputs, self.horizon)
                    )
                    self._pending = token
                # Keep the MIT reference alive while the bounded inference call
                # runs. The separately enforced inference deadline still holds.
                if self._last_target is not None:
                    self.controller.send_action(self._last_target)
                    self.last_step["command_id"] = self.controller.last_command_id
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
