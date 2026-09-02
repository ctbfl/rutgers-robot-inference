from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ruri.client.controllers.single_piper.config import SinglePiperConfig
from ruri.client.controllers.single_piper.mit_process import (
    ManagedMITProcess,
    ManagedMITTeleopProcess,
)
from ruri.client.controllers.single_piper_leader_follower_teleop import (
    SinglePiperLeaderFollowerTeleopConfig,
)


class ManagedMITProcessTests(unittest.TestCase):
    def test_forwards_absolute_diagnostic_log_to_worker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostic_log = Path(temp_dir) / "logs" / "mit.jsonl"
            config = SinglePiperConfig(
                python_executable=Path("/usr/bin/python3"),
                diagnostic_log=diagnostic_log,
            )
            process = MagicMock()
            process.stdout = ()
            process.poll.return_value = None

            with patch(
                "ruri.client.controllers.single_piper.mit_process.subprocess.Popen",
                return_value=process,
            ) as popen:
                ManagedMITProcess(config, "can-test").start()

            command = popen.call_args.args[0]
            self.assertEqual(
                command[:4],
                [
                    "/usr/bin/python3",
                    "-u",
                    "-m",
                    "ruri.client.controllers.single_piper.mit.policy_controller",
                ],
            )
            index = command.index("--diagnostic-log")
            self.assertEqual(command[index + 1], str(diagnostic_log.resolve()))
            self.assertTrue(diagnostic_log.parent.is_dir())

    def test_teleop_worker_launches_packaged_module_with_both_can_interfaces(self):
        config = SinglePiperLeaderFollowerTeleopConfig(
            python_executable=Path("/usr/bin/python3"),
        )
        process = MagicMock()
        process.stdout = ()
        process.poll.return_value = None

        with patch(
            "ruri.client.controllers.single_piper.mit_process.subprocess.Popen",
            return_value=process,
        ) as popen:
            ManagedMITTeleopProcess(config, "can-right", "can-left").start()

        command = popen.call_args.args[0]
        self.assertEqual(
            command[:4],
            [
                "/usr/bin/python3",
                "-u",
                "-m",
                "ruri.client.controllers.single_piper.mit.leader_follower",
            ],
        )
        self.assertEqual(command[command.index("--leader-can") + 1], "can-right")
        self.assertEqual(command[command.index("--follower-can") + 1], "can-left")
        self.assertIn("--execute", command)


if __name__ == "__main__":
    unittest.main()
