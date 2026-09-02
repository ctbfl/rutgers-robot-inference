"""LeRobot registration entrypoint for RURI MIT frame targets."""

from .config_piper_mit import PiperMITConfig
from .piper_mit import PiperMIT

__all__ = ["PiperMIT", "PiperMITConfig"]
