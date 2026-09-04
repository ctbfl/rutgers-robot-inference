"""Load the Piper joint envelopes stored beside this module.

Three files live in ``calibration/``. ``piper_range.json`` is the nominal
envelope: the abstract Piper this project normalizes against and clamps teleop
targets to. ``left_arm_real.json`` and ``right_arm_real.json`` are what the two
arms on this rig actually reach, measured with their motors off.

The nominal envelope is deliberately not either arm's measurement. It has to
stay fixed across arms and across datasets, because normalizing two datasets
with different tables makes the same physical angle carry two different labels,
and no amount of dataset-statistics normalization downstream can undo that.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CALIBRATION_DIR = Path(__file__).with_name("calibration")
NOMINAL_PATH = CALIBRATION_DIR / "piper_range.json"
LEFT_ARM_PATH = CALIBRATION_DIR / "left_arm_real.json"
RIGHT_ARM_PATH = CALIBRATION_DIR / "right_arm_real.json"

RANGE_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper")


@dataclass(frozen=True, slots=True)
class PiperRanges:
    """One envelope: seven ``(minimum, maximum)`` pairs in Piper raw units."""

    kind: str
    ranges: Mapping[str, tuple[float, float]]

    def __getitem__(self, name: str) -> tuple[float, float]:
        return self.ranges[name]


def _pair(raw: Any, label: str) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{label} must be a [minimum, maximum] pair")
    low, high = (float(v) for v in raw)
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"{label} must be finite")
    if low >= high:
        raise ValueError(f"{label} minimum must be below its maximum")
    return low, high


def load_ranges(path: str | Path = NOMINAL_PATH) -> PiperRanges:
    """Read and validate one calibration file."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load Piper calibration from {source}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("ranges"), dict):
        raise ValueError(f"{source} must contain a JSON object with a 'ranges' object")

    entries = raw["ranges"]
    missing = [name for name in RANGE_NAMES if name not in entries]
    extra = sorted(set(entries) - set(RANGE_NAMES))
    if missing or extra:
        raise ValueError(f"{source} ranges missing={missing}, extra={extra}")

    return PiperRanges(
        kind=str(raw.get("kind", "unknown")),
        ranges={name: _pair(entries[name], f"{source.name} {name}") for name in RANGE_NAMES},
    )


NOMINAL = load_ranges(NOMINAL_PATH)
LEFT_ARM = load_ranges(LEFT_ARM_PATH)
RIGHT_ARM = load_ranges(RIGHT_ARM_PATH)

#: Travel is a property of a physical arm, so it is keyed on the arm's name.
MEASURED_BY_ARM = {"left_arm": LEFT_ARM, "right_arm": RIGHT_ARM}


def measured_for_arm(name: str) -> PiperRanges:
    """Return what ``left_arm`` or ``right_arm`` physically reaches."""
    try:
        return MEASURED_BY_ARM[name]
    except KeyError:
        raise RuntimeError(
            f"No measured calibration for {name!r}; "
            f"known arms are {sorted(MEASURED_BY_ARM)}, files live in {CALIBRATION_DIR}"
        ) from None


def arm_name_for_hardware_id(hardware_id: str) -> str:
    """Which physical arm is on this USB-CAN adapter.

    This is the only place the two layers meet. Above it everything speaks in
    ``left_arm`` / ``right_arm``; the adapter serial is just how we recognise
    which arm a cable currently leads to.
    """
    from .hardware_registry import find_arm_by_hardware_id

    return find_arm_by_hardware_id(hardware_id).name

__all__ = [
    "CALIBRATION_DIR", "NOMINAL_PATH", "LEFT_ARM_PATH", "RIGHT_ARM_PATH",
    "RANGE_NAMES", "PiperRanges", "load_ranges", "NOMINAL", "LEFT_ARM", "RIGHT_ARM",
    "MEASURED_BY_ARM", "measured_for_arm", "arm_name_for_hardware_id",
]
