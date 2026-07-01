"""Deprecated import location for ``aragora.server.metrics.debate``.

This metrics submodule moved DOWN to
:mod:`aragora.observability.server_metrics.debate` during the P4a layering work so
that foundation/infrastructure modules can reach it without importing
``aragora.server``. Importing from ``aragora.server.metrics.debate`` still works
but is deprecated; import from
``aragora.observability.server_metrics.debate`` instead.
"""

from __future__ import annotations

import warnings

from aragora.observability.server_metrics import debate as _target

warnings.warn(
    "aragora.server.metrics.debate is deprecated; import from "
    "aragora.observability.server_metrics.debate instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(getattr(_target, "__all__", [n for n in dir(_target) if not n.startswith("_")]))


def __getattr__(name):
    return getattr(_target, name)


def __dir__():
    return sorted(set(__all__) | set(dir(_target)))
