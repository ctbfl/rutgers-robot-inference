"""Small CPU tests of OpenPI transforms and domain transport, without weights."""

import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("openpi"), "requires the OpenPI server environment")
class SimRealRuntimeTests(unittest.TestCase):
    def make_policy(self):
        import jax
        from openpi.shared.normalize import NormStats
        from ruri.server.wrappers.pi05 import _sim_real_policy as runtime
        from ruri.server.wrappers.pi05._sim_real_model import DomainRTCConfig

        # Exercise the real TokenizePrompt transform without downloading assets.
        class Tokenizer:
            def tokenize(self, prompt, state=None):
                if state is not None:
                    raise AssertionError("This checkpoint was trained without discrete state tokens")
                return np.ones(200, np.int32), np.ones(200, bool)

        stats = {name: NormStats(mean=np.zeros(7), std=np.ones(7),
                                q01=-np.arange(1, 8.), q99=np.arange(1, 8.))
                 for name in ("state", "actions")}
        policy = runtime.DomainPolicy.__new__(runtime.DomainPolicy)
        with patch.object(runtime, "PaligemmaTokenizer", return_value=Tokenizer()):
            policy._input_transform, policy._output_transform = runtime.make_transforms(DomainRTCConfig(), stats)
        policy._model = SimpleNamespace(action_horizon=30)
        policy._domain = (0., 1.)
        policy._rng = jax.random.key(0)
        policy._sample_kwargs = {"num_steps": 10}
        return policy

    def observation(self):
        return {"observation/state": np.arange(7, dtype=np.float32),
                "observation/image": np.zeros((32, 40, 3), np.uint8),
                "observation/wrist_image": np.zeros((32, 40, 3), np.uint8), "prompt": "insert A0"}

    def test_domain_survives_preprocessing_and_prefix_is_rebased(self):
        from ruri.server.wrappers.pi05._sim_real_model import DomainObservation, preprocess
        policy = self.make_policy()
        request = self.observation()
        previous = np.tile(np.arange(7, dtype=np.float32) + 3., (30, 1))
        original = previous.copy()
        obs, prefix = policy.prepare(request, previous)
        self.assertIsInstance(obs, DomainObservation)
        self.assertEqual(obs.state.shape, (1, 32))
        np.testing.assert_array_equal(obs.domain_one_hot, [[0, 1]])
        np.testing.assert_array_equal(obs.action_valid_mask, np.ones((1, 30), bool))
        processed = preprocess(None, obs)
        np.testing.assert_array_equal(processed.domain_one_hot, [[0, 1]])
        np.testing.assert_array_equal(processed.action_valid_mask, obs.action_valid_mask)
        decoded = policy._output_transform({"state": np.asarray(obs.state[0]), "actions": np.asarray(prefix[0])})
        np.testing.assert_allclose(decoded["actions"], previous, atol=2e-6)
        # Moving the current joints changes normalized prefix joints, while
        # the absolute gripper stays unchanged. The supplied chunk is untouched.
        new_request = {**request, "observation/state": request["observation/state"] + 1}
        new_obs, new_prefix = policy.prepare(new_request, previous)
        self.assertTrue(np.all(np.asarray(new_prefix[0, :, :6]) < np.asarray(prefix[0, :, :6])))
        np.testing.assert_array_equal(new_prefix[:, :, 6], prefix[:, :, 6])
        decoded = policy._output_transform({"state": np.asarray(new_obs.state[0]), "actions": np.asarray(new_prefix[0])})
        np.testing.assert_allclose(decoded["actions"], previous, atol=2e-6)
        np.testing.assert_array_equal(previous, original)

    def test_both_sampling_paths_receive_domain_observation(self):
        import jax.numpy as jnp
        from ruri.server.wrappers.pi05._sim_real_model import DomainObservation
        policy = self.make_policy()
        calls = []

        def sample(rng, obs, **kwargs):
            calls.append((obs, kwargs))
            self.assertIsInstance(obs, DomainObservation)
            np.testing.assert_array_equal(obs.domain_one_hot, [[0, 1]])
            return kwargs.get("prefix_actions", jnp.zeros((1, 30, 32)))

        policy._sample_actions = sample
        policy.infer(self.observation())
        previous = np.ones((30, 7), np.float32)
        result = policy.infer(self.observation(), prev_chunk=previous, delay=4)
        self.assertNotIn("prefix_actions", calls[0][1])
        self.assertEqual(int(calls[1][1]["inference_delay"]), 4)
        np.testing.assert_allclose(result["actions"], previous, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
