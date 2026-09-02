from __future__ import annotations

import unittest

import numpy as np

from ruri.client.controllers.single_piper.mit import leader_follower, policy_controller


class MITWorkerTests(unittest.TestCase):
    def test_packaged_gravity_calibrations_and_urdfs_are_self_contained(self):
        for filename in ("calibration_leader.json", "calibration_follower.json"):
            gravity = leader_follower.GravityCompensator(
                leader_follower.ASSET_ROOT / filename
            )
            self.assertTrue(gravity.urdf.is_file())
            torque = gravity.torque(np.zeros(6))
            self.assertEqual(torque.shape, (6,))
            self.assertTrue(np.all(np.isfinite(torque)))

    def test_workers_default_to_packaged_calibrations(self):
        teleop_args = leader_follower.build_parser().parse_args([])
        policy_args = policy_controller.build_parser().parse_args([])
        self.assertEqual(
            teleop_args.leader_calibration.parent,
            leader_follower.ASSET_ROOT,
        )
        self.assertEqual(
            policy_args.follower_calibration.parent,
            policy_controller.ASSET_ROOT,
        )

    def test_importing_worker_does_not_require_eager_sdk_import(self):
        self.assertNotIn("pyAgxArm", leader_follower.__dict__)


if __name__ == "__main__":
    unittest.main()
