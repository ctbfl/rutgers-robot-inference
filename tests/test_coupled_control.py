"""Continuous reference control, stale action invalidation and mapped coordinates."""
import unittest
from dataclasses import replace
import numpy as np
from ruri.client.controllers.single_piper_assisted.config import SinglePiperAssistedConfig
from ruri.client.controllers.single_piper_assisted.control import PROTOCOL, AssistedControl


class CoupledControlTests(unittest.TestCase):
    def setUp(self):
        self.now, self.sequence = 10., 0
        self.q = np.array([0., .4, -.4, 0., 0., 0., .04])
        self.control = AssistedControl(SinglePiperAssistedConfig(human_confirm_s=.5), self.q, self.q, clock=lambda: self.now)
        self.send("mode", mode="POLICY", request_id="start")
        self.send("target", target=self.q.tolist())

    def send(self, kind, **fields):
        self.sequence += 1
        return self.control.receive({"protocol": PROTOCOL, "worker_session": self.control.session,
            "client_session": "test", "sequence": self.sequence, "monotonic_ns": int(self.now*1e9),
            "kind": kind, "epoch": self.control.epoch, "reference_revision": self.control.reference_revision, **fields})

    def tick(self, leader=None, follower=None, dt=.01):
        self.now += dt
        self.send("heartbeat")
        old = self.control.target.copy()
        self.control.tick(old if leader is None else leader, old if follower is None else follower)
        self.assertLessEqual(np.max(np.abs(self.control.target[:6]-old[:6])), .15*min(dt,.02)+1e-12)
        self.assertEqual(self.control.leader_gain, 1.)
        np.testing.assert_allclose(self.control.target, self.control.leader_target)

    def intervene(self):
        leader = self.q.copy(); leader[0] += .2
        for _ in range(80):
            self.tick(leader)
        self.assertTrue(self.control.correcting)
        return leader

    def test_intervention_moves_reference_without_disarming_or_pose_snap(self):
        epoch = self.control.epoch
        leader = self.intervene()
        self.assertEqual(self.control.epoch, epoch)
        self.assertEqual(self.control.mode, "POLICY")
        self.assertEqual(self.control.intervention_id, 1)
        self.assertGreater(self.control.target[0], 0.)
        self.assertLess(self.control.target[0], leader[0])
        self.assertFalse(self.send("target", target=self.q.tolist(), reference_revision=0))
        self.assertEqual(self.control.source, "shared")

    def test_fresh_policy_contributes_during_partial_authority(self):
        leader = self.intervene()
        proposal = self.q.copy(); proposal[0] = -.2
        self.send("target", target=proposal.tolist())
        self.tick(leader)
        ref = self.control.reference
        self.assertGreater(ref.alpha, 0.)
        self.assertLess(ref.alpha, 1.)
        self.assertLess(ref.policy_velocity[0], 0.)
        self.assertGreater(ref.human_velocity[0], 0.)
        np.testing.assert_allclose(ref.velocity, np.clip((1-ref.alpha)*ref.policy_velocity+ref.alpha*ref.human_velocity, -.15, .15))

    def test_release_decays_alpha_and_invalidates_intervention_chunk(self):
        self.intervene()
        revision, epoch = self.control.reference_revision, self.control.epoch
        # Released arm tracks the applied reference. Fresh policy keeps running.
        for _ in range(150):
            goal = self.control.target.copy(); goal[0] += .04
            self.send("target", target=goal.tolist())
            self.tick()
        self.assertFalse(self.control.correcting)
        self.assertLess(self.control.reference.alpha, .001)
        self.assertEqual(self.control.reference_revision, revision+2)
        self.assertEqual(self.control.epoch, epoch)
        self.assertEqual(self.control.source, "policy")
        self.assertFalse(self.send("target", target=self.q.tolist(), reference_revision=revision))

    def test_keepalive_is_neither_an_action_nor_evidence_of_release(self):
        self.intervene()
        # Sustain a displacement relative to moving r, as a person pulling does.
        for _ in range(50):
            leader = self.control.target.copy(); leader[0] += .2
            self.send("target", target=self.control.policy_reference.tolist(), model_action=False)
            self.tick(leader)
        self.assertTrue(self.control.correcting)
        self.assertFalse(self.control.policy_valid)

    def test_follower_gripper_and_short_noise_do_not_trigger_joint_authority(self):
        leader, follower = self.q.copy(), self.q.copy()
        leader[6] += .02; follower[0] -= .2
        for _ in range(30):
            self.tick(leader, follower)
        self.assertEqual(self.control.reference.alpha, 0.)
        leader[0] += .02  # Below the spring deadzone.
        self.tick(leader, follower)
        self.tick(self.q)
        self.assertFalse(self.control.correcting)
        self.assertEqual(self.control.intervention_id, 0)

    def test_distant_proposal_is_velocity_limited_and_shared(self):
        target = self.q.copy(); target[0] += .8; target[6] += .012
        before = self.control.target.copy()
        self.send("target", target=target.tolist())
        np.testing.assert_allclose(self.control.target, before)
        self.tick()
        self.assertAlmostEqual(self.control.target[0], .0015)
        self.assertAlmostEqual(self.control.target[6], .0404)
        self.assertFalse(self.control.correcting)

    def test_observation_expiry_is_not_refreshed_by_keepalive(self):
        self.control.config = replace(self.control.config, policy_max_age_s=.1)
        self.tick(dt=.11)
        self.assertFalse(self.control.policy_valid)
        self.assertEqual(self.control.reference_revision, 1)
        self.assertFalse(self.send("target", target=self.q.tolist(), observation_ns=int((self.now-.2)*1e9)))
        self.assertTrue(self.send("target", target=self.q.tolist(), model_action=False))
        self.assertFalse(self.control.policy_valid)
        self.assertTrue(self.send("target", target=self.q.tolist()))
        self.assertTrue(self.control.policy_valid)

    def test_full_takeover_and_release_invalidate_chunks_without_changing_epoch(self):
        epoch = self.control.epoch
        for _ in range(110):
            leader = self.control.target.copy(); leader[1] += .1
            self.send('target', target=self.q.tolist())
            self.tick(leader)
        self.assertTrue(self.control.reference.human_latched)
        self.assertEqual(self.control.source, 'human')
        self.assertEqual(self.control.reference.alpha, 1.)
        revision = self.control.reference_revision
        self.assertFalse(self.send('target', target=self.q.tolist(), reference_revision=revision-1))
        for _ in range(200):
            proposal = self.control.target.copy(); proposal[0] += .5
            self.send('target', target=proposal.tolist())
            was_latched = self.control.reference.human_latched
            self.tick()
            if was_latched and not self.control.reference.human_latched:
                break
        self.assertFalse(self.control.reference.human_latched)
        self.assertEqual(self.control.reference_revision, revision+1)
        self.assertFalse(self.control.policy_valid)
        np.testing.assert_allclose(self.control.reference.policy_velocity, 0.)
        self.assertFalse(self.send('target', target=self.q.tolist(), reference_revision=revision))
        self.assertEqual(self.control.epoch, epoch)

    def test_joint_limit_projection_and_signed_reference_velocity(self):
        config = replace(self.control.config, leader_sign=(-1., 1., 1., 1., 1., 1.))
        control = AssistedControl(config, self.q, self.q, clock=lambda: self.now)
        control.mode = "POLICY"
        control.policy_reference[0] = .5
        control.policy_observation_ns = int(self.now*1e9)
        control.policy_valid = True
        self.now += .01
        control.tick(self.q, self.q)
        self.assertAlmostEqual(control.target[0], .0015)
        self.assertAlmostEqual(control.leader_target[0], -.0015)
        self.assertAlmostEqual(control.reference_velocity[0], .15)


if __name__ == "__main__":
    unittest.main()
