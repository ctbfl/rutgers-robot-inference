"""Teaching gestures and operator-latched takeover without physical hardware."""

import unittest

import numpy as np

from ruri.client.controllers.single_piper_assisted.config import SinglePiperAssistedConfig
from ruri.client.controllers.single_piper_assisted.control import AssistedControl, PROTOCOL


class TeachingTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.sequence = 0
        self.leader = np.array([0., .4, -.4, 0., 0., 0., .1])
        self.follower = self.leader.copy()
        config = SinglePiperAssistedConfig(gravity_handoff=True, human_trigger="gripper",
                                          gripper_max_m=.1)
        self.control = AssistedControl(config, self.leader, self.follower, clock=lambda: self.now)

    def send(self, kind, **fields):
        self.sequence += 1
        return self.control.receive(dict(protocol=PROTOCOL, worker_session=self.control.session,
            client_session="test", monotonic_ns=int(self.now*1e9), sequence=self.sequence,
            kind=kind, epoch=self.control.epoch, **fields))

    def mode(self, mode):
        return self.send("mode", mode=mode, request_id=f"mode-{self.sequence}")

    def tick(self, count=1):
        for _ in range(count):
            self.now += .01
            self.send("heartbeat")
            self.control.tick(self.leader, self.follower, self.now)

    def test_teaching_confirms_minimum_only_after_reopening_and_rearms(self):
        self.assertTrue(self.mode("TEACH"))
        self.tick(60)
        self.leader[6] = .08
        self.tick(10)
        self.leader[6] = .04
        self.tick(100)
        self.assertEqual(self.control.teaching_events, [])
        self.leader[6] = .042
        self.tick(10)
        self.assertEqual(self.control.teaching_events, [])
        self.leader[6] = .044
        self.tick(5)
        event, = self.control.teaching_events
        self.assertAlmostEqual(event["gripper_m"], .04)
        np.testing.assert_allclose(event["q_rad"], self.follower[:6])
        self.assertEqual(event["sequence"], 1)
        self.assertEqual(self.control.gripper_handoff.phase, "opening")
        self.leader[6] = .1
        self.tick(5)
        self.leader[6] = .06
        self.tick(5)
        self.leader[6] = .064
        self.tick(5)
        self.assertEqual([e["sequence"] for e in self.control.teaching_events], [1, 2])
        self.assertTrue(self.mode("HOLD"))
        self.assertFalse(self.control.teaching)

    def test_takeover_requires_intervention_and_keeps_human_until_hold(self):
        self.assertFalse(self.mode("TAKEOVER"))
        self.assertTrue(self.mode("POLICY"))
        self.assertTrue(self.send("target", target=self.follower.tolist()))
        self.tick(5)
        self.assertFalse(self.mode("TAKEOVER"))
        self.leader[6] = .09
        self.tick()
        self.assertTrue(self.control.correcting)
        previous_epoch = self.control.epoch
        self.assertTrue(self.mode("TAKEOVER"))
        self.assertGreater(self.control.epoch, previous_epoch)
        self.leader[6] = self.follower[6]
        # No more policy commands: fresh heartbeats alone retain manual control.
        self.tick(200)
        self.assertEqual(self.control.mode, "HUMAN")
        self.assertTrue(self.control.manual_takeover)
        self.assertEqual(self.control.reference.phase, "human")
        self.assertTrue(self.mode("HOLD"))
        self.assertFalse(self.control.manual_takeover)

    def test_teaching_still_obeys_heartbeat_timeout(self):
        self.assertTrue(self.mode("TEACH"))
        self.now += 2.
        self.control.tick(self.leader, self.follower, self.now)
        self.assertEqual(self.control.mode, "HOLD")
        self.assertFalse(self.control.teaching)


if __name__ == "__main__":
    unittest.main()
