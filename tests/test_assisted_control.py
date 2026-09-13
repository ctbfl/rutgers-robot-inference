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
        np.testing.assert_allclose(self.control.target, before)
        self.assertEqual(self.control.leader_gain, 0.)
        self.assertFalse(self.control.receive(self.packet("target", target=self.q.tolist(), epoch=old_epoch)))
        self.assertFalse(self.control.receive(self.packet("target", target=self.q.tolist())))
        leader[0] += .1
        self.tick(leader)
        self.assertGreater(self.control.target[0], before[0])
        self.assertLessEqual(self.control.target[0] - before[0], .0050001)

    def test_hold_release_requires_explicit_alignment_and_fresh_command(self):
        self.mode("HUMAN")
        leader = self.q.copy()
        leader[0] += .2
        self.tick(leader)
        self.assertTrue(self.mode("HOLD"))
        held = self.control.target.copy()
        self.assertFalse(self.mode("POLICY"))
        self.assertTrue(self.mode("ALIGN"))
        for _ in range(120):
            leader = self.control.leader_target.copy()
            self.tick(leader)
        np.testing.assert_allclose(self.control.target, held)
        self.assertTrue(self.control.aligned)
        self.assertTrue(self.mode("POLICY"))
        self.assertEqual(self.control.mode, "HOLD")
        self.assertTrue(self.control.armed)
        self.assertTrue(self.control.receive(self.packet("target", target=held.tolist())))
        self.assertEqual(self.control.mode, "POLICY")

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

    def test_mapping_intersection_timestamp_session_and_jump_admission(self):
        self.mode("POLICY")
        target = self.q.copy()
        target[0] += 1.
        self.assertFalse(self.control.receive(self.packet("target", target=target.tolist())))
        target = self.q.copy()
        target[5] = 100.
        self.assertFalse(self.control.receive(self.packet("target", target=target.tolist())))
        self.assertFalse(self.control.receive(self.packet("target", target=self.q.tolist(), worker_session="old")))
        self.assertFalse(self.control.receive(self.packet("target", target=self.q.tolist(), monotonic_ns=0)))
        self.assertFalse(self.control.receive(self.packet("target", target=self.q.tolist(), client_session="intruder")))
        config = replace(SinglePiperAssistedConfig(), leader_sign=(-1., 1., 1., 1., 1., 1.))
        mapped = AssistedControl(config, self.q, self.q, clock=lambda: self.now)
        self.assertEqual(mapped.mapped(target)[0], -target[0])
        self.assertTrue(np.all(mapped.lower < mapped.upper))


if __name__ == "__main__":
    unittest.main()
