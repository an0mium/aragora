"""Deprecated compatibility shim for :mod:`aragora.ops`."""

from __future__ import annotations

import importlib
import warnings

warnings.warn(
    "aragora.operations is deprecated. Use aragora.ops instead.",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = importlib.import_module("aragora.ops")
_key_rotation = importlib.import_module("aragora.ops.key_rotation")
_extra_exports = {
    "KeyRotationResult": _key_rotation,
}

__all__ = list(dict.fromkeys([*getattr(_canonical, "__all__", ()), *_extra_exports]))


def __getattr__(name: str):
    module = _extra_exports.get(name)
    if module is not None:
        return getattr(module, name)
    return getattr(_canonical, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
