"""CLI and controller agree on configurable feedback speed protection."""

import unittest

from ruri.client.controllers.single_piper.mit import leader_follower
from ruri.client.controllers.single_piper_leader_follower_teleop.config import SinglePiperLeaderFollowerTeleopConfig


class SpeedGuardConfigTests(unittest.TestCase):
    def test_default_stays_at_four(self):
        self.assertEqual(SinglePiperLeaderFollowerTeleopConfig().max_joint_speed, 4.)
        self.assertEqual(leader_follower.build_parser().parse_args([]).max_joint_speed, 4.)

    def test_cli_and_config_accept_nine_and_reject_outside_range(self):
        for speed in (.1, 5., 9.):
            with self.subTest(speed=speed):
                config = SinglePiperLeaderFollowerTeleopConfig(max_joint_speed=speed)
                self.assertEqual(config.max_joint_speed, speed)
                args = leader_follower.build_parser().parse_args(["--max-joint-speed", str(speed)])
                leader_follower.validate_args(args)
        for speed in (0., -1., 9.01, float("nan"), float("inf")):
            with self.subTest(speed=speed):
                with self.assertRaises(ValueError):
                    SinglePiperLeaderFollowerTeleopConfig(max_joint_speed=speed)
                args = leader_follower.build_parser().parse_args(["--max-joint-speed", str(speed)])
                with self.assertRaises(ValueError):
                    leader_follower.validate_args(args)


if __name__ == "__main__":
    unittest.main()
