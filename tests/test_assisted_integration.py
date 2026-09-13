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
        self.robot = SinglePiperAssistedController({"simulate": True,
            "command_address": udp_address(), "telemetry_address": udp_address(),
            "camera_width": 32, "camera_height": 24, "arm_connect_timeout_s": 5.})
        self.addCleanup(self.robot.stop)
        self.robot.start()
        self.traces = []

    def drive(self, scheduler, condition, timeout=4.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sample = self.robot.get_sample()
            self.traces.extend(sample["trace"]["packets"])
            scheduler.step(sample)
            if condition():
                return
            time.sleep(1./30.)
        self.fail(f"condition not reached; status={self.robot.status()}")

    def test_policy_human_resume_and_version_switch_share_real_worker(self):
        class Policy:
            output_chunk_size = 6
            def infer(self, inputs):
                target = inputs["observation.state"].copy()
                target[0] += 2.
                return {"action_chunk": np.tile(target, (6, 1))}
        scheduler = AssistedScheduler(self.robot, Policy(), policy_version="v0", prompt="test")
        self.addCleanup(scheduler.close)
        before = self.robot.get_sample()
        self.assertEqual(before["telemetry"]["control"]["mode"], "HOLD")
        self.assertEqual(before["observation"]["observation.images.top"].shape, (24,32,3))
        scheduler.resume()
        self.drive(scheduler, lambda: self.robot.status()["target"][0] > .5)
        self.assertEqual(self.robot.status()["mode"], "POLICY")
        scheduler.takeover()
        self.assertEqual(self.robot.status()["mode"], "HUMAN")
        q = self.robot.get_sample()["observation"]["observation.state"].copy()
        q[0] += 5.
        self.robot.simulate_leader(q)
        self.drive(scheduler, lambda: self.robot.status()["target"][0] > 3.)
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
                release.wait(3.)
                return {"action_chunk": np.tile(inputs["observation.state"], (4,1))}
        scheduler = AssistedScheduler(self.robot, SlowPolicy(), policy_version="slow", prompt="test",
                                      inference_timeout_s=.1)
        self.addCleanup(scheduler.close)
        self.addCleanup(release.set)
        scheduler.resume()
        self.drive(scheduler, entered.is_set)
        start = time.monotonic()
        scheduler.takeover()
        self.assertLess(time.monotonic() - start, .2)
        # Exceed the inference deadline during human control. It must not issue
        # HOLD or interrupt the operator because of an obsolete GPU request.
        time.sleep(.15)
        scheduler.step(self.robot.get_sample())
        self.assertEqual(self.robot.status()["mode"], "HUMAN")
        release.set()
        self.drive(scheduler, lambda: scheduler.discarded_results == 1)
        self.assertEqual(self.robot.status()["mode"], "HUMAN")


if __name__ == "__main__":
    unittest.main()
