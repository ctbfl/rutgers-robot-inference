from dataclasses import replace
import unittest

import numpy as np

from ruri.client.controllers.single_piper_assisted.config import SinglePiperAssistedConfig
from ruri.client.controllers.single_piper_assisted.control import AssistedControl, PROTOCOL


class AssistedControlTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.q = np.asarray([0., 0.4, -0.4, 0., 0., 0., 0.04])
        self.control = AssistedControl(SinglePiperAssistedConfig(), self.q, self.q,
                                       clock=lambda: self.now)
        self.seq = 0

    def packet(self, kind, **fields):
        self.seq += 1
        return {"protocol": PROTOCOL, "worker_session": self.control.session,
                "client_session": "test", "monotonic_ns": int(self.now * 1e9),
                "sequence": self.seq, "kind": kind, "epoch": self.control.epoch, **fields}

    def mode(self, mode, **fields):
        return self.control.receive(self.packet("mode", mode=mode, request_id=f"req{self.seq}", **fields))

    def tick(self, leader=None, dt=.01):
        self.now += dt
        self.control.receive(self.packet("heartbeat"))
        return self.control.tick(self.q if leader is None else leader, self.q)

    def test_takeover_has_continuous_reference_and_rejects_old_policy(self):
        self.assertTrue(self.mode("POLICY"))
        target = self.q.copy()
        target[0] += .01
        self.assertTrue(self.control.receive(self.packet("target", target=target.tolist())))
        old_epoch = self.control.epoch
        self.control.leader[0] -= .1
        leader = self.control.leader.copy()
        self.assertTrue(self.mode("HUMAN"))
        before = self.control.target.copy()
        self.tick(leader)
        self.assertLess(self.control.target[0], before[0])
        self.assertLessEqual(np.max(np.abs(self.control.target-before)), .00150001)
        self.assertEqual(self.control.leader_gain, 1.)
        self.assertFalse(self.control.receive(self.packet("target", target=self.q.tolist(), epoch=old_epoch)))
        self.assertFalse(self.control.receive(self.packet("target", target=self.q.tolist())))
        leader[0] += .2
        self.tick(leader)
        self.assertGreater(self.control.target[0], before[0])
        self.assertLessEqual(self.control.target[0] - before[0], .0050001)

    def test_hold_release_accepts_policy_without_alignment(self):
        self.mode("HUMAN")
        leader = self.q.copy()
        leader[0] += .2
        self.tick(leader)
        self.assertTrue(self.mode("HOLD"))
        held = self.control.target.copy()
        self.assertFalse(self.control.aligned)
        self.assertTrue(self.mode("POLICY"))
        self.assertEqual(self.control.mode, "HOLD")
        self.assertTrue(self.control.armed)
        self.assertTrue(self.control.receive(self.packet("target", target=held.tolist())))
        self.assertEqual(self.control.mode, "POLICY")
        np.testing.assert_allclose(self.control.target, held)
        self.tick(self.control.leader)
        np.testing.assert_allclose(self.control.leader_target, self.control.mapped(self.control.target))
        self.assertLessEqual(np.max(np.abs(self.control.target-held)), .00150001)

    def test_homed_arms_with_different_grippers_start_policy_directly(self):
        # Passive CAN sample from the real run that was stuck waiting for ALIGN.
        leader = [-0.00325, -0.00586, -0.00469, 0., -0.02896, -0.03805, 0.0196]
        follower = [0., -0.00403, 0., -0.0077, 0.01717, 0., 0.04396]
        self.control = AssistedControl(SinglePiperAssistedConfig(), leader, follower,
                                       clock=lambda: self.now)
        self.assertFalse(self.control.aligned)
        self.assertTrue(self.mode("POLICY"))
        self.assertTrue(self.control.receive(self.packet("target", target=self.q.tolist())))
        np.testing.assert_allclose(self.control.policy_reference, self.q)
        initial = self.control.target.copy()
        self.now += .01
        self.control.tick(leader, follower)
        self.assertLessEqual(np.max(np.abs(self.control.target[:6]-initial[:6])), .00150001)
        np.testing.assert_allclose(self.control.leader_target, self.control.target)
        np.testing.assert_allclose(self.control.leader, leader)
        np.testing.assert_allclose(self.control.follower, follower)

    def test_duplicate_ack_does_not_advance_epoch_and_fault_is_latched(self):
        request = self.packet("mode", mode="HUMAN", request_id="same")
        self.assertTrue(self.control.receive(request))
        epoch = self.control.epoch
        replay = self.packet("mode", mode="HUMAN", request_id="same", epoch=epoch-1)
        self.assertTrue(self.control.receive(replay))
        self.assertEqual(self.control.epoch, epoch)
        self.control.fault("feedback stale")
        self.assertFalse(self.mode("HOLD"))
        self.assertFalse(self.mode("HUMAN"))
        self.tick(dt=2.)
        self.assertEqual(self.control.mode, "FAULT")

    def test_heartbeat_and_policy_timeout_hold_without_home_or_resume(self):
        self.mode("POLICY")
        self.control.receive(self.packet("target", target=self.q.tolist()))
        self.tick(dt=1.1)
        self.assertEqual(self.control.mode, "HOLD")
        np.testing.assert_allclose(self.control.target, self.q)
        self.assertTrue(self.mode("HUMAN"))
        self.now += 1.
        self.control.tick(self.q, self.q)
        self.assertEqual(self.control.mode, "HOLD")
        self.control.receive(self.packet("heartbeat"))
        self.control.tick(self.q, self.q)
        self.assertEqual(self.control.mode, "HOLD")

    def test_mapping_clamping_timestamp_and_session_admission(self):
        self.mode("POLICY")
        target = self.q.copy()
        target[0] += 1.
        self.assertTrue(self.control.receive(self.packet("target", target=target.tolist())))
        target = self.q.copy()
        target[5] = 100.
        self.assertTrue(self.control.receive(self.packet("target", target=target.tolist())))
        self.assertAlmostEqual(self.control.policy_reference[5], self.control.upper[5])
        self.assertEqual(self.control.target[5], 0.)
        self.assertFalse(self.control.receive(self.packet("target", target=self.q.tolist(), worker_session="old")))
        self.assertFalse(self.control.receive(self.packet("target", target=self.q.tolist(), monotonic_ns=0)))
        self.assertFalse(self.control.receive(self.packet("target", target=self.q.tolist(), client_session="intruder")))
        config = replace(SinglePiperAssistedConfig(), leader_sign=(-1., 1., 1., 1., 1., 1.))
        mapped = AssistedControl(config, self.q, self.q, clock=lambda: self.now)
        self.assertEqual(mapped.mapped(target)[0], -target[0])
        self.assertTrue(np.all(mapped.lower < mapped.upper))

    def test_measured_overshoot_is_clamped_for_hold_and_can_align(self):
        measured = self.q.copy()
        measured[1:3] = [-0.0346, 0.033]
        self.control = AssistedControl(SinglePiperAssistedConfig(), measured, measured,
                                       clock=lambda: self.now)
        np.testing.assert_allclose(self.control.follower, measured)
        np.testing.assert_allclose(self.control.target[1:3], [0., 0.])
        self.assertTrue(self.mode("ALIGN"))
        self.tick(measured)
        self.assertTrue(self.control.aligned)
        self.assertTrue(self.mode("POLICY"))


if __name__ == "__main__":
    unittest.main()
