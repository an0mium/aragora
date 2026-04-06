"""Deprecated compatibility shim for :mod:`aragora.gauntlet`."""

from __future__ import annotations

import importlib
import warnings

warnings.warn(
    "aragora.modes.gauntlet is deprecated. Use aragora.gauntlet instead.",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = importlib.import_module("aragora.gauntlet")
__all__ = list(getattr(_canonical, "__all__", ()))


def __getattr__(name: str):
    return getattr(_canonical, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
