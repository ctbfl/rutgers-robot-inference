"""Robot-side components for Rutgers Robot Inference."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BlockingScheduler",
    "RollingScheduler",
    "RTCScheduler",
    "TemporalEnsembleScheduler",
]


def __getattr__(name: str) -> Any:
    """Keep Controller-only environments free of transport dependencies."""
    if name == "BlockingScheduler":
        from ruri.client.schedulers import BlockingScheduler

        return BlockingScheduler
    if name == "RollingScheduler":
        from ruri.client.schedulers import RollingScheduler

        return RollingScheduler
    if name == "RTCScheduler":
        from ruri.client.schedulers import RTCScheduler

        return RTCScheduler
    if name == "TemporalEnsembleScheduler":
        from ruri.client.schedulers import TemporalEnsembleScheduler

        return TemporalEnsembleScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
