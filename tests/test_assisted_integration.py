"""Real UDP/subprocess tests; never open CAN or cameras."""

import socket
import threading
import time
import unittest

import numpy as np

from ruri.client.controllers.single_piper_assisted import SinglePiperAssistedController
from ruri.client.schedulers.assisted import AssistedScheduler


def udp_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return f"udp://127.0.0.1:{sock.getsockname()[1]}"


class AssistedIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.robot = SinglePiperAssistedController(
            {
                "simulate": True,
                "command_address": udp_address(),
                "telemetry_address": udp_address(),
                "camera_width": 32,
                "camera_height": 24,
                "arm_connect_timeout_s": 5.0,
            }
        )
        self.addCleanup(self.robot.stop)
        self.robot.start()
        self.traces = []
        self.inference_results = []

    def drive(self, scheduler, condition, timeout=4.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sample = self.robot.get_sample()
            self.traces.extend(sample["trace"]["packets"])
            result = scheduler.step(sample)
            self.inference_results.extend(result.get("inference_results", []))
            if condition():
                return
            time.sleep(1.0 / 30.0)
        self.fail(f"condition not reached; status={self.robot.status()}")

    def test_policy_human_resume_and_version_switch_share_real_worker(self):
        class Policy:
            output_chunk_size = 6

            def infer(self, inputs):
                target = inputs["observation.state"].copy()
                target[0] += 2.0
                return {
                    "action_chunk": np.tile(target, (6, 1)),
                    "rtc.applied": True,
                    "rtc.method": "training-time",
                }

        scheduler = AssistedScheduler(
            self.robot, Policy(), policy_version="v0", prompt="test"
        )
        self.addCleanup(scheduler.close)
        before = self.robot.get_sample()
        self.assertEqual(before["telemetry"]["control"]["mode"], "HOLD")
        self.assertEqual(
            before["observation"]["observation.images.top"].shape, (24, 32, 3)
        )
        scheduler.resume()
        self.drive(scheduler, lambda: self.robot.status()["target"][0] > 0.5)
        self.assertEqual(self.robot.status()["mode"], "POLICY")
        scheduler.takeover()
        self.assertEqual(self.robot.status()["mode"], "HUMAN")
        q = self.robot.get_sample()["observation"]["observation.state"].copy()
        q[0] += 5.0
        self.robot.simulate_leader(q)
        self.drive(scheduler, lambda: self.robot.status()["target"][0] > 3.0)
        scheduler.pause()
        scheduler.replace_policy(Policy(), "v1")
        scheduler.resume()
        self.drive(scheduler, lambda: self.robot.status()["mode"] == "POLICY")
        self.assertEqual(scheduler.policy_version, "v1")
        trace = self.robot.get_sample()["trace"]
        self.assertEqual(trace["dropped"], 0)
        sources = {p["control"]["source"] for p in self.traces}
        self.assertTrue({"policy", "human", "hold"}.issubset(sources))
        sequences = [p["sequence"] for p in self.traces]
        self.assertEqual(sequences, sorted(set(sequences)))

    def test_late_inference_cannot_reclaim_human_control(self):
        entered, release = threading.Event(), threading.Event()

        class SlowPolicy:
            output_chunk_size = 4

            def infer(self, inputs):
                entered.set()
                release.wait(3.0)
                return {
                    "action_chunk": np.tile(inputs["observation.state"], (4, 1)),
                    "rtc.applied": True,
                }

        scheduler = AssistedScheduler(
            self.robot,
            SlowPolicy(),
            policy_version="slow",
            prompt="test",
            inference_timeout_s=0.1,
        )
        self.addCleanup(scheduler.close)
        self.addCleanup(release.set)
        scheduler.resume()
        self.drive(scheduler, entered.is_set)
        start = time.monotonic()
        scheduler.takeover()
        self.assertLess(time.monotonic() - start, 0.2)
        # Exceed the inference deadline during human control. It must not issue
        # HOLD or interrupt the operator because of an obsolete GPU request.
        time.sleep(0.15)
        scheduler.step(self.robot.get_sample())
        self.assertEqual(self.robot.status()["mode"], "HUMAN")
        release.set()
        self.drive(scheduler, lambda: scheduler.discarded_results == 1)
        self.assertEqual(self.robot.status()["mode"], "HUMAN")

    def test_wrong_served_version_enters_hold_without_executing_chunk(self):
        class WrongVersion:
            output_chunk_size = 4

            def infer(self, inputs):
                return {
                    "action_chunk": np.tile(inputs["observation.state"] + 10.0, (4, 1)),
                    "policy_version": "wrong",
                }

        scheduler = AssistedScheduler(
            self.robot, WrongVersion(), policy_version="expected", prompt="test"
        )
        self.addCleanup(scheduler.close)
        initial = np.array(self.robot.status()["target"])
        scheduler.resume()
        with self.assertRaisesRegex(RuntimeError, "policy inference failed"):
            self.drive(scheduler, lambda: False)
        self.assertEqual(self.robot.status()["mode"], "HOLD")
        np.testing.assert_allclose(self.robot.status()["target"], initial)

    def test_automatic_correction_keeps_inference_running_and_rejects_old_result(self):
        entered, release = threading.Event(), threading.Event()
        states = []

        class Policy:
            output_chunk_size = 12

            def infer(self, inputs):
                states.append(inputs["observation.state"].copy())
                if len(states) == 2:
                    entered.set()
                    release.wait(2.0)
                return {
                    "action_chunk": np.tile(inputs["observation.state"], (12, 1)),
                    "rtc.applied": True,
                }

        scheduler = AssistedScheduler(
            self.robot, Policy(), policy_version="continuous", prompt="test"
        )
        self.addCleanup(scheduler.close)
        self.addCleanup(release.set)
        scheduler.resume()
        self.drive(scheduler, entered.is_set)
        epoch = self.robot.status()["epoch"]
        target = states[-1].copy()
        target[0] += 4.0
        self.robot.simulate_leader(target, duration_s=0.5)
        self.drive(scheduler, lambda: self.robot.status()["intervention_active"])
        self.assertEqual(self.robot.status()["mode"], "POLICY")
        self.assertEqual(self.robot.status()["epoch"], epoch)
        release.set()
        self.drive(
            scheduler, lambda: scheduler.discarded_results >= 1 and len(states) >= 4
        )
        # The next query uses the actual follower observation during catch-up,
        # not a fictitious exact copy of the human's leader pose.
        self.assertGreater(states[-1][0], target[0] - 1.5)
        self.drive(scheduler, lambda: self.robot.status()["source"] == "policy")
        self.assertEqual(self.robot.status()["epoch"], epoch)
        self.assertEqual(self.robot.status()["intervention_id"], 1)
        self.assertLess(np.max(np.abs(self.robot.status()["coupling_error"])), 0.02)
        # Periodic inference was issued while a previous chunk still had rows.
        self.assertGreater(len(states), 3)
        obsolete = [
            r
            for r in self.inference_results
            if r["disposition"] == "discarded_obsolete"
        ]
        self.assertTrue(obsolete)
        np.testing.assert_allclose(obsolete[0]["action_chunk"][0], states[1])
        self.assertEqual(obsolete[0]["inference"].epoch, epoch)


if __name__ == "__main__":
    unittest.main()
