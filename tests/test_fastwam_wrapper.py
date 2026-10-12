"""RURI-only contract tests: no FastWAM, torch, CUDA or hardware required."""

from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import zmq

from ruri.client.controllers.dummy import DummyController
from ruri.client.schedulers import BlockingScheduler
from ruri.client.utils import inference_client
from ruri.common import zmq as codec
from ruri.server.serve import handle_request
from ruri.server.wrappers.fastwam import FastWAMWrapper


class FakeRuntime:
    horizon = 32

    def __init__(self, **kwargs):
        self.calls = []
        self.compile_action_infer = kwargs.get("compile_action_infer", False)

    def load_model(self):
        pass

    def text_context(self, task):
        pass

    def metadata(self):
        return {"checkpoint_step": 20000, "compile_action_infer": self.compile_action_infer}

    def predict(self, state, images, prompt, **kwargs):
        if prompt != "task":
            raise ValueError("Uncached FastWAM prompt")
        self.calls.append((state, images, kwargs))
        return np.tile(state, (32, 1)), 1.0


class FastWAMWrapperTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root / "src/fastwam").mkdir(parents=True)
        runtime = SimpleNamespace(FastWAMRuntime=FakeRuntime)
        self.module_patch = patch.dict(sys.modules, {"ruri.server.wrappers.fastwam._runtime": runtime})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        original_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        self.wrapper = FastWAMWrapper("unused.pt", fastwam_root=self.root, default_prompt="task", warmup=False)

    def observation(self):
        return {
            "observation.state": np.arange(7, dtype=np.float32),
            "observation.images.top": np.full((16, 20, 3), 12, dtype=np.uint8),
            "observation.images.wrist": np.full((16, 20, 3), 34, dtype=np.uint8),
        }

    def test_lazy_import(self):
        subprocess.run([sys.executable, "-c", "import sys; from ruri.server.wrappers.fastwam import FastWAMWrapper; assert 'torch' not in sys.modules; assert 'fastwam' not in sys.modules"], check=True)

    def test_metadata_and_full_chunk(self):
        metadata = handle_request(self.wrapper, {"type": "metadata"})
        self.assertEqual(metadata["outputs"], {"output_chunk_size": 32})
        self.assertIn("observation.images.wrist", metadata["inputs"])
        self.assertNotIn("hand", metadata["inputs"])
        self.assertFalse(metadata["policy"]["supports_rtc"])
        self.assertTrue(metadata["policy"]["compile_action_infer"])
        request = self.observation()
        request["context.actions_per_chunk"] = 1  # selection is scheduler-owned
        response = codec.unpack(codec.pack(handle_request(self.wrapper, request)))
        self.assertEqual(response["action_chunk"].shape, (32, 7))
        self.assertEqual(response["action_chunk"].dtype, np.float32)
        self.assertGreaterEqual(response["timing.wrapper_ms"], 0)
        state, images, opts = self.wrapper._runtime.calls[-1]
        self.assertEqual(images["top"][0, 0, 0], 12)
        self.assertEqual(images["hand"][0, 0, 0], 34)
        self.assertEqual(opts, {"steps": 20, "seed": 42})
        self.assertFalse(np.shares_memory(state, request["observation.state"]))

    def test_invalid_inputs(self):
        changes = [
            ("observation.state", np.zeros((1, 7))),
            ("observation.state", np.full(7, np.nan)),
            ("observation.state", np.full(7, 1e300)),
            ("observation.state", ["1"] * 7),
            ("observation.images.top", np.zeros((16, 20, 3), dtype=np.float32)),
            ("observation.images.wrist", np.zeros((3, 16, 20), dtype=np.uint8)),
            ("prompt", None), ("prompt", "unknown"),
            ("context.rtc.consumed_steps", 1),
        ]
        for key, value in changes:
            with self.subTest(key=key, value_type=type(value)), self.assertRaises(ValueError):
                self.wrapper.infer({**self.observation(), key: value})

    def test_noncontiguous_images_and_independent_requests(self):
        request = self.observation()
        request["observation.images.top"] = request["observation.images.top"][:, ::-1]
        self.wrapper.infer(request)
        request["observation.state"] += 10
        self.wrapper.infer(request)
        calls = self.wrapper._runtime.calls
        self.assertTrue(calls[0][1]["top"].flags.c_contiguous)
        np.testing.assert_array_equal(calls[1][0] - calls[0][0], 10)

    def test_rejects_invalid_backend_output(self):
        for action in (np.zeros((0, 7)), np.full((32, 7), np.nan)):
            with patch.object(self.wrapper._runtime, "predict", return_value=(action, 1.0)):
                with self.assertRaisesRegex(ValueError, "invalid action chunk"):
                    self.wrapper.infer(self.observation())

    def test_warmup_uses_public_contract(self):
        self.wrapper.warmup()
        self.assertEqual(len(self.wrapper._runtime.calls), 3)

    def test_compile_opt_out_and_custom_warmup(self):
        wrapper = FastWAMWrapper("unused.pt", fastwam_root=self.root, default_prompt="task",
                                 compile_action_infer=False, warmup_iters=2)
        self.assertFalse(wrapper.describe()["policy"]["compile_action_infer"])
        self.assertEqual(len(wrapper._runtime.calls), 2)

    def test_compile_failure_is_not_silently_downgraded(self):
        with patch.object(self.wrapper._runtime, "predict", side_effect=RuntimeError("compiler failed")):
            with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                self.wrapper.warmup()

    def test_rejects_invalid_compilation_options(self):
        for opts in ({"compile_action_infer": "true"}, {"warmup_iters": 0}, {"warmup_iters": True}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                FastWAMWrapper("unused.pt", fastwam_root=self.root, default_prompt="task", **opts)

    def test_zmq_client_and_dummy_blocking_scheduler(self):
        # Loopback only. No serve()/registry/menu process or Piper imports.
        ready, stopped = threading.Event(), threading.Event()
        endpoints, errors = [], []

        def server():
            try:
                with zmq.Context() as context, context.socket(zmq.REP) as socket:
                    port = socket.bind_to_random_port("tcp://127.0.0.1")
                    endpoints.append(f"tcp://127.0.0.1:{port}")
                    ready.set()
                    while not stopped.is_set():
                        if not socket.poll(50):
                            continue
                        try:
                            response = handle_request(self.wrapper, codec.recv(socket))
                        except Exception as exc:
                            response = {"error": str(exc)}
                        codec.send(socket, response)
            except Exception as exc:
                errors.append(exc)
                ready.set()

        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        try:
            self.assertTrue(ready.wait(5))
            self.assertFalse(errors)
            args = SimpleNamespace(policy_endpoint=endpoints[0], prompt="task", control_hz=10000,
                                   max_chunks=2, execute_actions_per_chunk=2, scheduler_log_enabled=False)
            robot = DummyController(args)
            with inference_client.connect(args) as policy:
                self.assertEqual(policy.output_chunk_size, 32)
                with self.assertRaisesRegex(RuntimeError, "Uncached"):
                    policy.infer({**self.observation(), "prompt": "unknown"})
                BlockingScheduler().run(controller=lambda _: robot, policy=policy, args=args)
            self.assertEqual(robot.action_history.shape, (4, 7))
        finally:
            stopped.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)


if __name__ == "__main__":
    unittest.main()
