from __future__ import annotations

import os
import copy
import queue
import threading
import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zmq

from ruri.client.schedulers import InteractiveRTCScheduler
from ruri.client.utils import inference_client
from ruri.client.utils.prompt_terminal import PromptTerminal, TerminalEvent
from ruri.common.zmq import recv as recv_message, send as send_message


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


class LocalPolicyServer:
    """Real loopback RURI transport, with scripted actions and no hardware."""

    def __init__(self, metadata, infer):
        self.metadata = metadata
        self.infer = infer
        self.inputs = []
        self.errors = []
        self.ready = queue.Queue()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with zmq.Context() as context, context.socket(zmq.REP) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            port = socket.bind_to_random_port("tcp://127.0.0.1")
            self.ready.put(f"tcp://127.0.0.1:{port}")
            while not self.stop.is_set():
                if not socket.poll(50):
                    continue
                request = recv_message(socket)
                try:
                    if request["type"] == "metadata":
                        response = self.metadata
                    else:
                        self.inputs.append(request)
                        response = self.infer(request, len(self.inputs))
                except Exception as error:
                    self.errors.append(error)
                    response = {"error": str(error)}
                send_message(socket, response)

    def __enter__(self):
        self.thread.start()
        self.endpoint = self.ready.get(timeout=2)
        self.port = self.endpoint.rsplit(":", 1)[1]
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise AssertionError("Loopback policy server did not stop")
        if self.errors:
            raise AssertionError(self.errors)


class PortSwitchTests(unittest.TestCase):
    @staticmethod
    def metadata():
        return {
            "inputs": {"observation.state": {"type": "state"}},
            "outputs": {"output_chunk_size": 30},
            "policy": {"name": "pi05_train_rtc", "config_name": "pi05_train_rtc_h30",
                       "rtc_method": "training-time", "checkpoint_path": "/checkpoint/A"},
        }

    def test_real_transport_switches_checkpoints_and_resets_rtc_delay(self):
        instances = {}
        terminal = FakeTerminal([TerminalEvent("prompt", "A0")])
        metadata = self.metadata()
        new_metadata = copy.deepcopy(metadata)
        new_metadata["policy"]["checkpoint_path"] = "/checkpoint/B"
        old_held = []

        def old_infer(inputs, call):
            if call == 1:
                return {"action_chunk": chunk(0, 30)}
            if call == 3:
                return {"action_chunk": chunk(400, 30)}
            if call == 4:
                terminal.push("exit")
                return {"action_chunk": chunk(500, 30), "rtc.applied": True}
            terminal.push("pause")
            wait_until(lambda: len(terminal.prompt_requests) == 2)
            terminal.push("prompt", f"switch {new_server.port}")
            actions = instances["controller"].actions
            held_at = len(actions)
            wait_until(lambda: len(actions) >= held_at + 12)
            old_held.extend(actions[held_at:])
            return {"action_chunk": chunk(100, 30), "rtc.applied": True}

        def new_infer(inputs, call):
            if call == 1:
                return {"action_chunk": chunk(200, 30)}
            terminal.push("pause")
            wait_until(lambda: len(terminal.prompt_requests) == 3)
            terminal.push("prompt", f"switch {old_server.port}")
            return {"action_chunk": chunk(300, 30), "rtc.applied": True}

        with LocalPolicyServer(new_metadata, new_infer) as new_server:
            with LocalPolicyServer(metadata, old_infer) as old_server:
                with inference_client.connect(old_server.endpoint) as policy:
                    scheduler = InteractiveRTCScheduler()
                    scheduler.run(recording_controller(instances), policy,
                        args=SimpleNamespace(control_hz=100, execution_horizon=5,
                            rtc_initial_delay_steps=4, scheduler_log_enabled=False),
                        terminal=terminal)
                self.assertEqual(len(old_server.inputs), 4)
                self.assertNotIn("context.rtc.prev_chunk_left_over", old_server.inputs[2])
                self.assertEqual(old_server.inputs[2]["prompt"], "A0")
            self.assertEqual(new_server.inputs[0]["prompt"], "A0")
            self.assertNotIn("context.rtc.prev_chunk_left_over", new_server.inputs[0])
            self.assertEqual(new_server.inputs[1]["context.rtc.estimated_inference_delay_steps"], 4)
            self.assertTrue(np.all(new_server.inputs[1]["context.rtc.prev_chunk_left_over"] >= 200))
        actions = instances["controller"].actions
        self.assertEqual(len(set(old_held)), 1)
        self.assertFalse(set(actions) & set(range(100, 130)))
        self.assertFalse(set(actions) & set(range(300, 330)))
        start = actions.index(200)
        self.assertEqual(actions[start:start + 5], list(range(200, 205)))
        self.assertEqual(scheduler.last_run_stats["switches_completed"], 2)

    def test_pause_or_exit_during_switch_keeps_the_new_policy_from_moving(self):
        for event in ("pause", "exit"):
            with self.subTest(event=event):
                instances = {}
                terminal = FakeTerminal([TerminalEvent("prompt", "A0")])
                stopped = threading.Event()
                metadata = self.metadata()
                closed = []

                class Controller(recording_controller(instances)):
                    def stop(self):
                        stopped.set()

                class Policy:
                    endpoint = "tcp://host:5557"
                    output_chunk_size = 30

                    def __init__(self):
                        self.metadata = metadata
                        self.calls = 0

                    def infer(self, inputs):
                        self.calls += 1
                        if self.calls == 1:
                            return {"action_chunk": chunk(0, 30)}
                        terminal.push("pause")
                        wait_until(lambda: len(terminal.prompt_requests) == 2)
                        terminal.push("prompt", "switch 5558")
                        return {"action_chunk": chunk(100, 30), "rtc.applied": True}

                def no_infer(inputs):
                    raise AssertionError("Switch must not resume after q or exit")

                candidate = SimpleNamespace(metadata=metadata, output_chunk_size=30,
                    close=lambda: closed.append(True), infer=no_infer)

                def connect(args):
                    terminal.push(event)
                    if event == "exit":
                        wait_until(stopped.is_set)
                    else:
                        actions = instances["controller"].actions
                        held_at = len(actions)
                        wait_until(lambda: len(actions) >= held_at + 5)
                    return candidate

                def operator():
                    wait_until(lambda: len(terminal.prompt_requests) == 3)
                    terminal.push("exit")

                operator_thread = None
                if event == "pause":
                    operator_thread = threading.Thread(target=operator, daemon=True)
                    operator_thread.start()
                scheduler = InteractiveRTCScheduler()
                with patch("ruri.client.schedulers.rtc_interactive.inference_client.connect", side_effect=connect):
                    scheduler.run(Controller, Policy(),
                        args=SimpleNamespace(control_hz=200, execution_horizon=5,
                            scheduler_log_enabled=False), terminal=terminal)
                if operator_thread is not None:
                    operator_thread.join(timeout=2)
                self.assertTrue(stopped.is_set())
                self.assertEqual(closed, [True])
                self.assertLess(max(instances["controller"].actions), 30)

    def test_failed_switch_holds_and_can_resume_the_original_connection(self):
        for failure in ("timeout", "inputs", "horizon", "rtc_method"):
            with self.subTest(failure=failure):
                instances = {}
                terminal = FakeTerminal([TerminalEvent("prompt", "A0")])
                metadata = self.metadata()
                candidate = SimpleNamespace(metadata=copy.deepcopy(metadata),
                    output_chunk_size=30, close=lambda: closed.append(True))
                closed = []
                if failure == "inputs":
                    candidate.metadata["inputs"] = {}
                elif failure == "horizon":
                    candidate.output_chunk_size = 20
                elif failure == "rtc_method":
                    candidate.metadata["policy"]["rtc_method"] = "test-time"

                class Policy:
                    endpoint = "tcp://172.16.68.130:5557"
                    output_chunk_size = 30

                    def __init__(self):
                        self.metadata = metadata
                        self.inputs = []

                    def infer(self, inputs):
                        self.inputs.append(inputs)
                        if len(self.inputs) == 1:
                            return {"action_chunk": chunk(0, 30)}
                        if len(self.inputs) == 2:
                            terminal.push("pause")
                            wait_until(lambda: len(terminal.prompt_requests) == 2)
                            terminal.push("prompt", "switch 5558")
                            # Superseded old request failure must not stop the
                            # control loop before the switch can be attempted.
                            raise TimeoutError("old request timed out")
                        terminal.push("exit")
                        return {"action_chunk": chunk(50, 30)}

                def connect(args):
                    self.assertEqual(args["policy_endpoint"], "tcp://172.16.68.130:5558")
                    actions = instances["controller"].actions
                    held_at = len(actions)
                    wait_until(lambda: len(actions) >= held_at + 5)
                    self.assertEqual(len(set(actions[held_at:])), 1)
                    if failure == "timeout":
                        raise TimeoutError("new port unreachable")
                    return candidate

                def operator():
                    wait_until(lambda: len(terminal.prompt_requests) == 3)
                    terminal.push("prompt", "A0")

                thread = threading.Thread(target=operator, daemon=True)
                thread.start()
                scheduler = InteractiveRTCScheduler()
                policy = Policy()
                with patch("ruri.client.schedulers.rtc_interactive.inference_client.connect", side_effect=connect):
                    scheduler.run(recording_controller(instances), policy,
                        args=SimpleNamespace(control_hz=200, execution_horizon=5,
                            scheduler_log_enabled=False), terminal=terminal)
                thread.join(timeout=2)
                self.assertIn("still on tcp://172.16.68.130:5557", terminal.prompt_requests[-1])
                self.assertEqual(scheduler.last_run_stats["switches_failed"], 1)
                self.assertEqual(len(policy.inputs), 3)
                self.assertNotIn("context.rtc.prev_chunk_left_over", policy.inputs[2])
                self.assertEqual(closed, [] if failure == "timeout" else [True])


class InteractiveRTCSchedulerTests(unittest.TestCase):
    def test_home_discards_old_rtc_holds_and_resumes_without_a_prefix(self):
        instances = {}
        terminal = FakeTerminal([TerminalEvent("prompt", "task A")])

        class HomeController(recording_controller(instances)):
            def iter_home_actions(self, **kwargs):
                self.home_args = kwargs
                for value in [-10, -20, -30]:
                    yield np.array([value], dtype=np.float32)

        class Policy:
            output_chunk_size = 4

            def __init__(self):
                self.inputs = []

            def infer(self, inputs):
                self.inputs.append(inputs)
                call = len(self.inputs)
                controller = instances["controller"]
                if call == 1:
                    return {"action_chunk": chunk(0, 4)}
                if call == 2:
                    terminal.push("pause")
                    wait_until(lambda: len(terminal.prompt_requests) == 2)
                    terminal.push("prompt", "home")
                    wait_until(lambda: len(terminal.prompt_requests) == 3)
                    held_at = len(controller.actions)
                    wait_until(lambda: len(controller.actions) >= held_at + 5)
                    self.held = controller.actions[held_at:]
                    terminal.push("prompt", "task B")
                    return {"action_chunk": chunk(10, 4), "rtc.applied": True}
                if call == 3:
                    return {"action_chunk": chunk(20, 4)}
                terminal.push("exit")
                return {"action_chunk": chunk(30, 4), "rtc.applied": True}

        policy = Policy()
        scheduler = InteractiveRTCScheduler()
        scheduler.run(HomeController, policy, args=SimpleNamespace(
            control_hz=200, execution_horizon=2, scheduler_log_enabled=False
        ), terminal=terminal)
        controller = instances["controller"]
        self.assertEqual(controller.home_args, {"control_hz": 200, "speed_rad_s": 1.0})
        start = controller.actions.index(-10)
        self.assertEqual(controller.actions[start:start + 3], [-10, -20, -30])
        self.assertEqual(set(policy.held), {-30})
        self.assertFalse(set(controller.actions) & {10, 11, 12, 13})
        self.assertEqual([x["prompt"] for x in policy.inputs[:3]], ["task A", "task A", "task B"])
        self.assertNotIn("context.rtc.prev_chunk_left_over", policy.inputs[2])
        resume = controller.actions.index(20)
        self.assertEqual(controller.actions[resume:resume + 2], [20, 21])
        self.assertEqual(scheduler.last_run_stats["homes_completed"], 1)
        self.assertEqual(scheduler.last_run_stats["discarded_chunks"], 1)

    def test_q_can_cancel_home_before_the_first_policy_chunk(self):
        instances = {}
        terminal = FakeTerminal([TerminalEvent("prompt", "task A")])

        class HomeController(recording_controller(instances)):
            def iter_home_actions(self, **kwargs):
                for value in range(-10, -100, -1):
                    yield np.array([value], dtype=np.float32)

            def send_action(self, action):
                super().send_action(action)
                if int(action[0]) == -12 and self.actions.count(-12) == 1:
                    terminal.push("pause")

        class Policy:
            output_chunk_size = 4

            def infer(self, inputs):
                terminal.push("pause")
                wait_until(lambda: len(terminal.prompt_requests) == 2)
                terminal.push("prompt", "home")
                wait_until(lambda: len(terminal.prompt_requests) == 3)
                controller = instances["controller"]
                wait_until(lambda: controller.actions.count(-12) >= 5)
                terminal.push("exit")
                return {"action_chunk": chunk(0, 4)}

        scheduler = InteractiveRTCScheduler()
        scheduler.run(HomeController, Policy(), args=SimpleNamespace(
            control_hz=200, execution_horizon=2, scheduler_log_enabled=False
        ), terminal=terminal)
        actions = instances["controller"].actions
        self.assertEqual(actions[:3], [-10, -11, -12])
        self.assertEqual(set(actions[2:]), {-12})
        self.assertEqual(scheduler.last_run_stats["homes_cancelled"], 1)
        self.assertEqual(scheduler.last_run_stats["homes_completed"], 0)
        self.assertEqual(scheduler.last_run_stats["chunks_requested"], 1)

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
