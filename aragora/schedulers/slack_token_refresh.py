"""Deprecated compatibility shim for :mod:`aragora.scheduler.slack_token_refresh`."""

from __future__ import annotations

import importlib
import warnings

warnings.warn(
    "aragora.schedulers.slack_token_refresh is deprecated. "
    "Use aragora.scheduler.slack_token_refresh instead.",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = importlib.import_module("aragora.scheduler.slack_token_refresh")
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
