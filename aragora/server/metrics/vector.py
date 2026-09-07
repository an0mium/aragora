"""Deprecated import location for ``aragora.server.metrics.vector``.

This metrics submodule moved DOWN to
:mod:`aragora.observability.server_metrics.vector` during the P4a layering work so
that foundation/infrastructure modules can reach it without importing
``aragora.server``. Importing from ``aragora.server.metrics.vector`` still works
but is deprecated; import from
``aragora.observability.server_metrics.vector`` instead.
"""

from __future__ import annotations

import warnings

from aragora.observability.server_metrics import vector as _target

warnings.warn(
    "aragora.server.metrics.vector is deprecated; import from "
    "aragora.observability.server_metrics.vector instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(getattr(_target, "__all__", [n for n in dir(_target) if not n.startswith("_")]))


def __getattr__(name):
    return getattr(_target, name)


def __dir__():
    return sorted(set(__all__) | set(dir(_target)))
