"""Standalone Aragora debate wedge.

This package intentionally exposes only the minimal offline debate surface that is
truthful for the ``aragora-debate`` distribution:

- ``Environment`` and core message/result types
- ``DebateProtocol`` for debate configuration
- ``Arena`` for running a minimal async debate with mock or real agents
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

__version__ = "2.8.0"

_EXPORT_MAP = {
    "Agent": ("aragora.core", "Agent"),
    "Critique": ("aragora.core", "Critique"),
    "DebateProtocol": ("aragora.debate", "DebateProtocol"),
    "DebateResult": ("aragora.core", "DebateResult"),
    "Environment": ("aragora.core", "Environment"),
    "Message": ("aragora.core", "Message"),
    "Vote": ("aragora.core", "Vote"),
    "Arena": ("aragora.debate", "Arena"),
    # Golden 5 simplified API surface
    "debate": ("aragora.golden", "debate"),
    "remember": ("aragora.golden", "remember"),
    "recall": ("aragora.golden", "recall"),
    "review": ("aragora.golden", "review"),
    "workflow": ("aragora.golden", "workflow"),
    "receipt": ("aragora.golden", "receipt"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = sorted(_EXPORT_MAP)


_ROOT_GOLDEN_COLLISIONS = {"review", "workflow"}
_GOLDEN_SUBMODULE_HINTS = {
    "review": (
        "aragora.review.protocol",
        "aragora.review.provider_slots",
        "aragora.review.reviewer_output",
    ),
    "workflow": (
        "aragora.workflow.engine",
        "aragora.workflow.templates",
        "aragora.workflow.patterns",
        "aragora.workflow.types",
    ),
}


def _golden_collision_export(name: str) -> Any:
    module_name, attr_name = _EXPORT_MAP[name]
    value = getattr(importlib.import_module(module_name), attr_name)

    for hint in _GOLDEN_SUBMODULE_HINTS.get(name, ()):
        try:
            importlib.import_module(hint)
        except ImportError:
            continue

    prefix = f"aragora.{name}."
    for loaded_name, loaded_module in tuple(sys.modules.items()):
        if not loaded_name.startswith(prefix):
            continue
        child_name = loaded_name[len(prefix) :].split(".", 1)[0]
        if child_name:
            setattr(value, child_name, loaded_module)
    globals()[name] = value
    return value


class _AragoraModule(ModuleType):
    def __getattribute__(self, name: str) -> Any:
        if name in _ROOT_GOLDEN_COLLISIONS:
            return _golden_collision_export(name)
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _AragoraModule
