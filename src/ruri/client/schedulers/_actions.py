"""Final action-boundary handling shared by client schedulers."""

from __future__ import annotations

from typing import Any

import numpy as np


def clip_target(controller: Any, action: Any) -> tuple[np.ndarray, bool]:
    """Return the Scheduler-owned final target and whether clipping occurred."""
    values = np.asarray(action, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Action target must be a non-empty vector, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Action target contains NaN or infinity")

    bounds = getattr(controller, "action_bounds", None)
    if bounds is None:
        return values.copy(), False
    lower = np.asarray(bounds[0], dtype=np.float32)
    upper = np.asarray(bounds[1], dtype=np.float32)
    if lower.shape != values.shape or upper.shape != values.shape:
        raise ValueError(
            "Controller action bounds do not match target shape: "
            f"target={values.shape}, lower={lower.shape}, upper={upper.shape}"
        )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Controller action bounds must be finite")
    if np.any(lower > upper):
        raise ValueError("Controller action lower bounds exceed upper bounds")

    target = np.clip(values, lower, upper)
    return target, not np.array_equal(values, target)


__all__ = ["clip_target"]
