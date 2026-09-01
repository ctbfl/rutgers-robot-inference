from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ruri.client.controllers.single_piper.config import SinglePiperConfig
from ruri.client.controllers.single_piper.mit_process import ManagedMITProcess


class ManagedMITProcessTests(unittest.TestCase):
    def test_forwards_absolute_diagnostic_log_to_worker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "teleop"
            root.mkdir()
            (root / "mit_policy_controller.py").write_text("", encoding="utf-8")
            diagnostic_log = Path(temp_dir) / "logs" / "mit.jsonl"
            config = SinglePiperConfig(
                teleop_root=root,
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
            index = command.index("--diagnostic-log")
            self.assertEqual(command[index + 1], str(diagnostic_log.resolve()))
            self.assertTrue(diagnostic_log.parent.is_dir())


if __name__ == "__main__":
    unittest.main()
