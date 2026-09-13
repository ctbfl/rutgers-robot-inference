from __future__ import annotations

import os
import threading
import time
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from ruri.client.schedulers import InteractiveRTCScheduler
from ruri.client.utils.prompt_terminal import PromptTerminal, TerminalEvent


def wait_until(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not reached in time")
        time.sleep(0.001)


class FakeTerminal:
    def __init__(self, events=(), log: list[str] | None = None):
        self._lock = threading.Lock()
        self._events = deque(events)
        self.prompt_requests: list[str] = []
        self.log = log

    def push(self, kind: str, text: str | None = None) -> None:
        with self._lock:
            self._events.append(TerminalEvent(kind, text))

    def request_prompt(self, message: str) -> None:
        with self._lock:
            self.prompt_requests.append(message)
        if self.log is not None:
            self.log.append("request_prompt")

    def poll_event(self, timeout_s: float = 0.0):
        with self._lock:
            if self._events:
                return self._events.popleft()
        if timeout_s > 0:
            time.sleep(min(timeout_s, 0.001))
        return None


def recording_controller(instances: dict, log: list[str] | None = None):
    class FakeController:
        def __init__(self, received_args):
            self.actions: list[int] = []
            instances["controller"] = self

        def start(self):
            if log is not None:
                log.append("controller.start")

        def get_observation(self):
            return {"observation.state": np.zeros(1)}

        def send_action(self, action):
            self.actions.append(int(action[0]))

        def stop(self):
            if log is not None:
                log.append("controller.stop")

    return FakeController


def chunk(start: int, size: int) -> np.ndarray:
    return np.arange(start, start + size, dtype=np.float32)[:, None]


class InteractiveRTCSchedulerTests(unittest.TestCase):
    def test_pause_discards_in_flight_chunk_holds_and_resumes_from_row_zero(self):
        instances: dict[str, object] = {}
        args = SimpleNamespace(
            control_hz=200.0,
            execution_horizon=2,
            scheduler_log_enabled=False,
        )
        terminal = FakeTerminal([TerminalEvent("prompt", "task A")])

        class ScriptedPolicy:
            output_chunk_size = 4

            def __init__(self):
                self.inputs: list[dict] = []

            def infer(self, inputs):
                self.inputs.append(inputs)
                call = len(self.inputs)
                controller = instances["controller"]
                if call == 1:
                    return {"action_chunk": chunk(0, 4)}
                if call == 2:
                    # Pause while this RTC request is in flight and keep it in
                    # flight through a stretch of paused hold ticks.
                    terminal.push("pause")
                    wait_until(lambda: len(terminal.prompt_requests) == 2)
                    paused_at = len(controller.actions)
                    wait_until(lambda: len(controller.actions) >= paused_at + 10)
                    terminal.push("prompt", "task B")
                    return {"action_chunk": chunk(10, 4), "rtc.applied": True}
                if call == 3:
                    # Hold ticks keep going while the resume chunk is computed;
                    # none of them may consume its rows.
                    requested_at = len(controller.actions)
                    wait_until(lambda: len(controller.actions) >= requested_at + 5)
                    return {"action_chunk": chunk(20, 4)}
                terminal.push("exit")
                return {"action_chunk": chunk(30, 4), "rtc.applied": True}

        policy = ScriptedPolicy()
        scheduler = InteractiveRTCScheduler()
        scheduler.run(
            recording_controller(instances),
            policy,
            args=args,
            terminal=terminal,
        )

        actions = instances["controller"].actions
        self.assertFalse(set(actions) & {10, 11, 12, 13})
        resume_index = actions.index(20)
        self.assertEqual(actions[resume_index : resume_index + 2], [20, 21])
        held = actions[resume_index - 15 : resume_index]
        self.assertEqual(len(set(held)), 1, held)
        self.assertIn(held[0], {0, 1, 2, 3})

        self.assertEqual(policy.inputs[0]["prompt"], "task A")
        self.assertNotIn("context.rtc.prev_chunk_left_over", policy.inputs[0])
        self.assertEqual(policy.inputs[1]["prompt"], "task A")
        self.assertEqual(policy.inputs[2]["prompt"], "task B")
        self.assertNotIn("context.rtc.prev_chunk_left_over", policy.inputs[2])

        stats = scheduler.last_run_stats
        self.assertEqual(stats["pauses"], 1)
        self.assertEqual(stats["prompt_changes"], 1)
        self.assertEqual(stats["discarded_chunks"], 1)
        self.assertGreaterEqual(stats["paused_ticks"], 10)

    def test_pause_before_first_chunk_sends_nothing_and_restarts(self):
        instances: dict[str, object] = {}
        args = SimpleNamespace(
            control_hz=200.0,
            execution_horizon=2,
            scheduler_log_enabled=False,
        )
        terminal = FakeTerminal([TerminalEvent("prompt", "task A")])
        sent_while_paused: list[int] = []

        class ScriptedPolicy:
            output_chunk_size = 2

            def __init__(self):
                self.inputs: list[dict] = []

            def infer(self, inputs):
                self.inputs.append(inputs)
                call = len(self.inputs)
                controller = instances["controller"]
                if call == 1:
                    terminal.push("pause")
                    wait_until(lambda: len(terminal.prompt_requests) == 2)
                    time.sleep(0.05)
                    sent_while_paused.append(len(controller.actions))
                    terminal.push("prompt", "task B")
                    return {"action_chunk": chunk(10, 2)}
                if call == 2:
                    return {"action_chunk": chunk(20, 2)}
                terminal.push("exit")
                return {"action_chunk": chunk(30, 2), "rtc.applied": True}

        policy = ScriptedPolicy()
        InteractiveRTCScheduler().run(
            recording_controller(instances),
            policy,
            args=args,
            terminal=terminal,
        )

        actions = instances["controller"].actions
        self.assertEqual(sent_while_paused, [0])
        self.assertEqual(actions[:2], [20, 21])
        self.assertNotIn(10, actions)
        self.assertEqual(policy.inputs[1]["prompt"], "task B")

    def test_first_prompt_is_read_before_the_controller_starts(self):
        log: list[str] = []
        instances: dict[str, object] = {}
        args = SimpleNamespace(
            control_hz=200.0,
            execution_horizon=1,
            scheduler_log_enabled=False,
        )
        terminal = FakeTerminal([TerminalEvent("prompt", "task A")], log=log)

        class ExitingPolicy:
            output_chunk_size = 2

            def infer(self, inputs):
                log.append("infer")
                terminal.push("exit")
                return {"action_chunk": chunk(0, 2)}

        InteractiveRTCScheduler().run(
            recording_controller(instances, log),
            ExitingPolicy(),
            args=args,
            terminal=terminal,
        )

        self.assertEqual(log[:3], ["request_prompt", "controller.start", "infer"])
        self.assertEqual(log[-1], "controller.stop")

    def test_escape_at_first_prompt_quits_without_starting_the_arm(self):
        log: list[str] = []
        instances: dict[str, object] = {}
        args = SimpleNamespace(control_hz=200.0, scheduler_log_enabled=False)
        terminal = FakeTerminal([TerminalEvent("exit")], log=log)

        class UnusedPolicy:
            output_chunk_size = 5

            def infer(self, inputs):
                log.append("infer")
                return {"action_chunk": chunk(0, 5)}

        InteractiveRTCScheduler().run(
            recording_controller(instances, log),
            UnusedPolicy(),
            args=args,
            terminal=terminal,
        )

        self.assertEqual(log, ["request_prompt", "controller.stop"])


class PromptTerminalTests(unittest.TestCase):
    def setUp(self):
        self.master, self.slave = os.openpty()
        self.terminal = PromptTerminal(input_fd=self.slave, output_fd=self.slave)
        self.terminal.open()

    def tearDown(self):
        self.terminal.close()
        os.close(self.master)
        os.close(self.slave)

    def test_q_pauses_outside_prompt_entry_and_is_text_inside_it(self):
        os.write(self.master, b"q")
        self.assertEqual(self.terminal.poll_event(1.0), TerminalEvent("pause"))
        self.terminal.request_prompt("prompt?")
        os.write(self.master, b"q\r")
        self.assertEqual(self.terminal.poll_event(1.0), TerminalEvent("prompt", "q"))

    def test_empty_line_asks_again_and_backspace_edits(self):
        self.terminal.request_prompt("prompt?")
        os.write(self.master, b"\r")
        self.assertIsNone(self.terminal.poll_event(0.2))
        os.write(self.master, b"pick ab\x7fc\r")
        self.assertEqual(
            self.terminal.poll_event(1.0), TerminalEvent("prompt", "pick ac")
        )

    def test_lone_escape_quits_but_arrow_keys_do_not(self):
        self.terminal.request_prompt("prompt?")
        os.write(self.master, b"\x1b[A")
        self.assertIsNone(self.terminal.poll_event(0.2))
        os.write(self.master, b"\x1b")
        self.assertEqual(self.terminal.poll_event(1.0), TerminalEvent("exit"))

    def test_second_q_does_not_pause_again_after_the_prompt(self):
        os.write(self.master, b"qq")
        self.assertEqual(self.terminal.poll_event(1.0), TerminalEvent("pause"))
        time.sleep(0.1)
        self.terminal.request_prompt("prompt?")
        os.write(self.master, b"x\r")
        self.assertEqual(self.terminal.poll_event(1.0), TerminalEvent("prompt", "x"))
        self.assertIsNone(self.terminal.poll_event(0.1))


if __name__ == "__main__":
    unittest.main()
