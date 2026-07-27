"""Standalone Aragora debate wedge.

This package intentionally exposes only the minimal offline debate surface that is
truthful for the ``aragora-debate`` distribution:

- ``Environment`` and core message/result types
- ``DebateProtocol`` for debate configuration
- ``Arena`` for running a minimal async debate with mock or real agents
"""

from __future__ import annotations

import importlib
from typing import Any

# Re-export the package version from the canonical single source of truth
# (``aragora/__version__.py``), which CI keeps aligned with ``pyproject.toml``
# via ``scripts/check_version_alignment.py``. Deriving it here instead of
# hard-coding a literal prevents ``aragora.__version__`` from silently drifting
# away from the declared/installed version. The version module is stdlib-only
# and side-effect free, so importing it at package init is safe.
from aragora.__version__ import __version__ as __version__

_EXPORT_MAP = {
    "Agent": ("aragora.core", "Agent"),
    "Critique": ("aragora.core", "Critique"),
    "DebateProtocol": ("aragora.debate", "DebateProtocol"),
    "DebateResult": ("aragora.core", "DebateResult"),
    "Environment": ("aragora.core", "Environment"),
    "Message": ("aragora.core", "Message"),
    "Vote": ("aragora.core", "Vote"),
    "Arena": ("aragora.debate", "Arena"),
    # Golden 5 simplified API surface.
    #
    # NOTE (#8780): ``debate``, ``review``, and ``workflow`` collide with same-named
    # subpackages. Once a subpackage is imported, the import system rebinds
    # the package attribute to the module object, bypassing this lazy map.
    # The matching subpackage ``__init__`` modules therefore make those modules
    # callable (delegating to ``aragora.golden``) so the Golden API callables work
    # in every import order. Keep that guard in sync when editing these entries.
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
