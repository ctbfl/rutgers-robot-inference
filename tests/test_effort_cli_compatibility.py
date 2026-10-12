"""Merged upstream and original effort flags select the same observation path."""

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruri.client.controllers.single_piper.config import SinglePiperConfig


class EffortCompatibilityTests(unittest.TestCase):
    def test_python_configuration_accepts_both_spellings(self):
        for setting in ("send_effort", "include_effort_observation"):
            with self.subTest(setting=setting):
                self.assertTrue(SinglePiperConfig(**{setting: True}).send_effort)
                args = SimpleNamespace(**{setting: True})
                self.assertTrue(SinglePiperConfig.from_args(args).send_effort)
        self.assertFalse(SinglePiperConfig().send_effort)

    def test_client_cli_aliases_select_send_effort(self):
        root = Path(__file__).resolve().parents[1]
        for script in ("client_single_piper_blocking.py", "client_single_piper_rtc.py"):
            module = runpy.run_path(str(root / "examples" / script))
            for flag in ("--send-effort", "--include-effort-observation"):
                with self.subTest(script=script, flag=flag):
                    with patch.object(sys, "argv", [script, "--policy-endpoint", "tcp://localhost:5557", flag]):
                        config = SinglePiperConfig.from_args(module["parse_args"]())
                    self.assertTrue(config.send_effort)


if __name__ == "__main__":
    unittest.main()
