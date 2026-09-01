"""Small helpers for the shared client argument object."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_MISSING = object()


def get_arg(args: Any, name: str, default: Any = _MISSING) -> Any:
    """Read one value from a shared Namespace-like or mapping argument object."""
    if isinstance(args, Mapping):
        if name in args:
            return args[name]
    elif hasattr(args, name):
        return getattr(args, name)

    if default is _MISSING:
        raise AttributeError(f"Global args is missing required field {name!r}")
    return default
