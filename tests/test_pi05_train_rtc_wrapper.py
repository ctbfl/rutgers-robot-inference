"""Train-config selection without OpenPI, GPUs, or a checkpoint."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruri.server.wrappers.pi05.pi05_train_rtc import Pi05TrainRTCWrapper


def package_config(exp_name):
    return ("package", exp_name)


class TrainConfigFileTests(unittest.TestCase):
    def import_config(self, train_config_file=None):
        # Skip __init__, which would load a checkpoint.
        wrapper = Pi05TrainRTCWrapper.__new__(Pi05TrainRTCWrapper)
        if train_config_file is not None:
            wrapper.train_config_file = train_config_file
        with patch.object(Pi05TrainRTCWrapper, "_import_train_rtc", return_value=package_config):
            return wrapper._import_train_config()

    def test_package_config_without_file(self):
        self.assertIs(self.import_config(), package_config)

    def test_run_config_file_replaces_package_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_config.py"
            path.write_text("def make_train_config(exp_name='run'):\n    return ('run', exp_name)\n")
            self.assertEqual(self.import_config(path)("serve"), ("run", "serve"))

    def test_file_without_factory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_config.py"
            path.write_text("NAME = 'no factory here'\n")
            with self.assertRaises(AttributeError):
                self.import_config(path)

    def test_run_config_supports_dataclasses_with_postponed_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_config.py"
            path.write_text(
                "from __future__ import annotations\nfrom dataclasses import dataclass\n"
                "@dataclass\nclass Config:\n    exp_name: str\n"
                "def make_train_config(exp_name):\n    return Config(exp_name)\n"
            )
            self.assertEqual(self.import_config(path)("serve").exp_name, "serve")

    def test_missing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                self.import_config(Path(tmp) / "run_config.py")


if __name__ == "__main__":
    unittest.main()
