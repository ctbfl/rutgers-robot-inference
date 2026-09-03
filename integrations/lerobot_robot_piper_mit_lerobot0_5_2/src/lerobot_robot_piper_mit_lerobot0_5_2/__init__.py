"""Legacy LeRobot 0.5.2 registration entrypoint for the RURI Piper observer."""

from .config_piper_mit_observer import PiperMITObserverConfig
from .piper_mit_observer import PiperMITObserver

__all__ = ["PiperMITObserver", "PiperMITObserverConfig"]
