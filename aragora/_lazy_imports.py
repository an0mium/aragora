"""Lazy module import helpers for the CLI.

The CLI parser eagerly imports every command module so it can register the
full set of subparsers. Some command modules only need an optional
third-party dependency (e.g. ``httpx``) at *runtime* — when the command is
actually executed against a server — not at import time.

``lazy_module`` returns a tiny proxy that defers the real import until the
first attribute access. This keeps the parser (and therefore the headline
``aragora ask`` command) importable on a base install that does not have the
optional dependency installed, while leaving every ``module.attr`` call site
unchanged. If the dependency really is missing, the original
``ModuleNotFoundError`` is raised on first use, pointing the user at the
extra they need to install.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


class _LazyModule:
    """Proxy that imports the backing module on first attribute access."""

    __slots__ = ("_lazy_name", "_lazy_mod")

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_lazy_name", name)
        object.__setattr__(self, "_lazy_mod", None)

    def _load(self) -> ModuleType:
        mod = object.__getattribute__(self, "_lazy_mod")
        if mod is None:
            name = object.__getattribute__(self, "_lazy_name")
            mod = importlib.import_module(name)
            object.__setattr__(self, "_lazy_mod", mod)
        return mod

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._load(), attr)


def lazy_module(name: str) -> Any:
    """Return a lazy proxy for the importable module ``name``."""
    return _LazyModule(name)
