"""Deprecated compatibility shim for :mod:`aragora.scheduler.receipt_retention`."""

from __future__ import annotations

import importlib
import warnings

warnings.warn(
    "aragora.schedulers.receipt_retention is deprecated. "
    "Use aragora.scheduler.receipt_retention instead.",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = importlib.import_module("aragora.scheduler.receipt_retention")
__all__ = list(
    getattr(
        _canonical,
        "__all__",
        [name for name in dir(_canonical) if not name.startswith("_")],
    )
)


def __getattr__(name: str):
    return getattr(_canonical, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
