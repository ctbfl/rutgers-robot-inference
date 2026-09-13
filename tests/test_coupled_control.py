"""Continuous policy execution and human corrections share one control epoch."""

import unittest
from dataclasses import replace

import numpy as np

from ruri.client.controllers.single_piper_assisted.config import (
    SinglePiperAssistedConfig,
)
from ruri.client.controllers.single_piper_assisted.control import (
    PROTOCOL,
    AssistedControl,
)


class CoupledControlTests(unittest.TestCase):
    def setUp(self):
        self.now, self.sequence = 10.0, 0
        self.q = np.array([0.0, 0.4, -0.4, 0.0, 0.0, 0.0, 0.04])
        self.control = AssistedControl(
            SinglePiperAssistedConfig(), self.q, self.q, clock=lambda: self.now
        )
        self.send("mode", mode="POLICY", request_id="start")
        self.send("target", target=self.q.tolist())

    def send(self, kind, **fields):
        self.sequence += 1
        packet = {
            "protocol": PROTOCOL,
            "worker_session": self.control.session,
            "client_session": "test",
            "sequence": self.sequence,
            "monotonic_ns": int(self.now * 1e9),
            "kind": kind,
            "epoch": self.control.epoch,
            "reference_revision": self.control.reference_revision,
            **fields,
        }
        return self.control.receive(packet)

    def tick(self, leader=None, follower=None, dt=0.01):
        self.now += dt
        self.send("heartbeat")
        self.control.tick(
            self.q if leader is None else leader,
            self.control.target.copy() if follower is None else follower,
        )

    def intervene(self, joint=0, delta=0.09):
        leader = self.q.copy()
        leader[joint] += delta
        for _ in range(8):
            self.tick(leader)
        self.assertTrue(self.control.correcting)
        return leader

    def test_follower_tracks_actual_leader_before_detection_and_policy_never_disarms(
        self,
    ):
        epoch = self.control.epoch
        leader = self.q.copy()
        leader[0] += 0.09
        self.tick(leader)
        self.assertGreater(self.control.target[0], 0.0)
        self.assertFalse(self.control.correcting)
        self.intervene()
        self.assertEqual(self.control.mode, "POLICY")
        self.assertEqual(self.control.epoch, epoch)
        self.assertEqual(self.control.intervention_id, 1)
        np.testing.assert_allclose(self.control.target, leader)
        np.testing.assert_allclose(self.control.policy_reference, leader)
        self.assertFalse(
            self.send("target", target=self.q.tolist(), reference_revision=0)
        )

    def test_fresh_policy_continues_from_corrected_state_without_hold_or_alignment(
        self,
    ):
        epoch = self.control.epoch
        leader = self.intervene()
        for _ in range(60):
            model_target = leader.copy()
            model_target[0] += 0.002
            self.assertTrue(self.send("target", target=model_target.tolist()))
            self.tick(model_target)
            leader = model_target
            self.assertEqual(self.control.mode, "POLICY")
            self.assertFalse(self.control.aligning)
        self.assertEqual(self.control.epoch, epoch)
        self.assertFalse(self.control.correcting)
        self.assertEqual(self.control.source, "policy")
        self.assertGreater(self.control.target[0], 0.15)

    def test_waiting_reference_cannot_be_mistaken_for_policy_agreement(self):
        leader = self.intervene()
        for _ in range(60):
            self.send(
                "target",
                target=self.control.policy_reference.tolist(),
                model_action=False,
            )
            self.tick(leader)
        self.assertTrue(self.control.correcting)
        self.assertEqual(self.control.source, "human")
        self.assertFalse(self.control.policy_valid)

    def test_follower_obstruction_and_short_transient_are_not_human_labels(self):
        follower = self.q.copy()
        follower[0] -= 0.1
        for _ in range(8):
            self.tick(self.q, follower)
        self.assertFalse(self.control.correcting)
        leader = self.q.copy()
        leader[0] += 0.06
        self.tick(leader)
        self.tick(self.q)
        self.assertEqual(self.control.source, "policy")
        self.assertEqual(self.control.intervention_id, 0)

    def test_gripper_only_correction_and_signed_mapping(self):
        self.intervene(6, 0.006)
        self.assertEqual(self.control.mode, "POLICY")
        self.assertGreater(self.control.target[6], self.q[6])
        config = replace(
            SinglePiperAssistedConfig(), leader_sign=(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        )
        self.control = AssistedControl(config, self.q, self.q, clock=lambda: self.now)
        self.send("mode", mode="POLICY", request_id="mapped")
        self.send("target", target=self.q.tolist())
        leader = self.q.copy()
        leader[0] -= 0.02
        self.tick(leader)
        self.assertGreater(self.control.target[0], 0.0)

    def test_observation_expiry_is_not_refreshed_by_keepalive_commands(self):
        self.control.config = replace(self.control.config, policy_max_age_s=0.1)
        self.tick(dt=0.11)
        self.assertEqual(self.control.mode, "POLICY")
        self.assertFalse(self.control.policy_valid)
        self.assertEqual(self.control.reference_revision, 1)
        self.assertFalse(
            self.send(
                "target",
                target=self.q.tolist(),
                observation_ns=int((self.now - 0.2) * 1e9),
            )
        )
        self.assertTrue(self.send("target", target=self.q.tolist(), model_action=False))
        self.assertFalse(self.control.policy_valid)
        self.assertTrue(self.send("target", target=self.q.tolist()))
        self.assertTrue(self.control.policy_valid)


if __name__ == "__main__":
    unittest.main()
