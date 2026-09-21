"""Unrecorded paired interactive runs against the real simulated MIT worker."""

import importlib.util
import os
import socket
import tempfile
import threading
import unittest
from contextlib import chdir
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ruri.client.controllers.single_piper_assisted import SinglePiperAssistedController
from ruri.client.schedulers.assisted_interactive import InteractiveAssistedRTC
from ruri.client.utils.prompt_terminal import TerminalEvent
from tests.test_rtc_interactive_scheduler import FakeTerminal, wait_until


def udp_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return f"udp://127.0.0.1:{sock.getsockname()[1]}"


class HoldingPolicy:
    endpoint = "tcp://localhost:5557"
    output_chunk_size = 30

    def __init__(self):
        self.inputs = []
        self.closed = False
        self.metadata = {
            "inputs": {
                "observation.state": {},
                "observation.images.top": {},
                "observation.images.wrist": {},
            },
            "outputs": {"output_chunk_size": 30},
            "policy": {"rtc_method": "training-time", "checkpoint_path": "test-A"},
        }

    def infer(self, inputs):
        self.inputs.append(inputs)
        return {
            "action_chunk": np.tile(inputs["observation.state"], (30, 1)),
            "rtc.applied": True,
            "rtc.method": "training-time",
        }

    def close(self):
        self.closed = True


class AssistedInteractiveTests(unittest.TestCase):
    def setUp(self):
        self.instances = {}
        self.errors = []
        self.terminal = FakeTerminal([TerminalEvent("prompt", "A0")])
        self.args = SimpleNamespace(
            control_hz=30,
            execution_horizon=5,
            rtc_initial_delay_steps=4,
            policy_timeout_s=3,
        )

    def controller(self, args):
        robot = SinglePiperAssistedController(
            {
                "simulate": True,
                "command_address": udp_address(),
                "telemetry_address": udp_address(),
                "camera_width": 32,
                "camera_height": 24,
                "startup_home": getattr(args, "startup_home", False),
                "arm_connect_timeout_s": 10,
            }
        )
        self.instances["robot"] = robot
        return robot

    def run_operator(self, operator):
        def guarded():
            try:
                operator()
            except Exception as error:
                self.errors.append(error)
            finally:
                self.terminal.push("exit")

        thread = threading.Thread(target=guarded, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)

    def run_demo(self, policy):
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            stats = InteractiveAssistedRTC().run(
                controller=self.controller,
                policy=policy,
                args=self.args,
                terminal=self.terminal,
            )
            self.assertEqual(os.listdir(directory), [], "demo wrote data files")
        if self.errors:
            raise self.errors[0]
        return stats

    def test_physical_correction_and_release_keep_policy_running_without_recording(
        self,
    ):
        policy = HoldingPolicy()
        epochs = []

        def operator():
            wait_until(lambda: len(policy.inputs) >= 2, timeout_s=5)
            robot = self.instances["robot"]
            epochs.append(robot.status()["epoch"])
            corrected = robot.get_observation()["observation.state"].copy()
            corrected[0] += 4
            calls = len(policy.inputs)
            # Keep pulling through the new confirmation delay and authority rise.
            robot.simulate_leader(corrected, duration_s=1.7)
            wait_until(lambda: robot.status()["intervention_active"])
            self.assertEqual(robot.status()["mode"], "POLICY")
            wait_until(lambda: len(policy.inputs) >= calls + 2)
            wait_until(lambda: robot.status()["source"] == "policy", timeout_s=4)
            epochs.append(robot.status()["epoch"])
            self.assertEqual(epochs[0], epochs[1])
            self.assertGreater(
                policy.inputs[-1]["observation.state"][0], corrected[0] - 3.5
            )
            self.terminal.push("pause")
            wait_until(lambda: len(self.terminal.prompt_requests) == 2)
            self.terminal.push("prompt", "B0")
            wait_until(lambda: any(x["prompt"] == "B0" for x in policy.inputs))

        self.run_operator(operator)
        stats = self.run_demo(policy)
        self.assertEqual(stats["corrections"], 1)
        first_b = next(x for x in policy.inputs if x["prompt"] == "B0")
        self.assertNotIn("context.rtc.prev_chunk_left_over", first_b)
        self.assertTrue(all("observation.joint_effort" not in x for x in policy.inputs))

    def test_home_checks_both_arms_then_switches_checkpoint(self):
        policy = HoldingPolicy()
        candidate = HoldingPolicy()
        candidate.endpoint = "tcp://localhost:5558"
        candidate.metadata["policy"]["checkpoint_path"] = "test-B"

        def operator():
            wait_until(lambda: len(policy.inputs) >= 2, timeout_s=5)
            self.terminal.push("pause")
            wait_until(lambda: len(self.terminal.prompt_requests) == 2)
            self.terminal.push("prompt", "home")
            wait_until(lambda: len(self.terminal.prompt_requests) == 3, timeout_s=5)
            self.assertIn("Both arms reached home", self.terminal.prompt_requests[-1])
            sample = self.instances["robot"].get_sample()
            for arm in ("leader", "follower"):
                self.assertLess(np.max(np.abs(sample["telemetry"][arm]["q"])), 0.05)
            self.assertEqual(sample["telemetry"]["control"]["mode"], "HOLD")
            self.terminal.push("prompt", "switch 5558")
            wait_until(lambda: len(candidate.inputs) >= 2)

        self.run_operator(operator)
        original_status = SinglePiperAssistedController.status

        def unaligned_status(robot):
            return {**original_status(robot), "aligned": False, "aligning": False}

        with patch(
            "ruri.client.schedulers.assisted_interactive.inference_client.connect",
            return_value=candidate,
        ) as connect, patch.object(
            SinglePiperAssistedController, "status", unaligned_status
        ):
            stats = self.run_demo(policy)
        self.assertEqual(
            connect.call_args.args[0]["policy_endpoint"], candidate.endpoint
        )
        self.assertEqual(stats["homes"], 1)
        self.assertEqual(stats["switches"], 1)
        self.assertTrue(candidate.closed)
        self.assertEqual(candidate.inputs[0]["prompt"], "A0")
        self.assertNotIn("context.rtc.prev_chunk_left_over", candidate.inputs[0])
        self.assertNotIn("context.rtc.estimated_inference_delay_steps", candidate.inputs[1])

    def test_exit_before_prompt_does_not_start_arms(self):
        self.terminal = FakeTerminal([TerminalEvent("exit")])
        with patch.object(SinglePiperAssistedController, "start") as start:
            stats = self.run_demo(HoldingPolicy())
        start.assert_not_called()
        self.assertEqual(stats["steps"], 0)

    def test_startup_homes_both_arms_before_first_policy_observation(self):
        self.args.startup_home = True
        policy = HoldingPolicy()

        def operator():
            wait_until(lambda: len(policy.inputs) >= 1, timeout_s=10)
            robot = self.instances["robot"]
            packet = robot.get_sample()["telemetry"]
            for name in ("leader", "follower"):
                np.testing.assert_allclose(packet[name]["q"], np.zeros(6), atol=1e-5)
            from ruri.client.controllers.single_piper.normalization import denormalize_action
            q, _ = denormalize_action(policy.inputs[0]["observation.state"])
            np.testing.assert_allclose(q, np.zeros(6), atol=1e-5)
            logs = list(robot._worker.logs)
            started = next(i for i, line in enumerate(logs) if "STARTUP HOME:" in line)
            completed = next(i for i, line in enumerate(logs) if "STARTUP HOME COMPLETE" in line)
            ready = next(i for i, line in enumerate(logs) if "ASSISTED READY: both arms homed" in line)
            self.assertLess(started, completed)
            self.assertLess(completed, ready)

        self.run_operator(operator)
        self.run_demo(policy)

    def test_can_alias_maps_to_follower_and_leader_uses_registered_discovery(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "examples/client_single_piper_assisted_rtc_interactive.py"
        )
        spec = importlib.util.spec_from_file_location("assisted_demo", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.parse_args(
            ["--policy-endpoint", "tcp://host:5557", "--can-interface", "can0"]
        )
        self.assertEqual(args.follower_can_interface, "can0")
        self.assertTrue(args.startup_home)
        self.assertIsNone(args.leader_can_interface)

    def test_busy_control_channel_fails_before_hardware_discovery(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            robot = SinglePiperAssistedController(
                {"telemetry_address": f"udp://127.0.0.1:{occupied.getsockname()[1]}"}
            )
            with patch.object(robot, "connect") as connect:
                with self.assertRaises(OSError):
                    robot.start()
            connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
