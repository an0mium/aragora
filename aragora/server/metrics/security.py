"""Deprecated import location for ``aragora.server.metrics.security``.

This metrics submodule moved DOWN to
:mod:`aragora.observability.server_metrics.security` during the P4a layering work so
that foundation/infrastructure modules can reach it without importing
``aragora.server``. Importing from ``aragora.server.metrics.security`` still works
but is deprecated; import from
``aragora.observability.server_metrics.security`` instead.
"""

from __future__ import annotations

import warnings

from aragora.observability.server_metrics import security as _target

warnings.warn(
    "aragora.server.metrics.security is deprecated; import from "
    "aragora.observability.server_metrics.security instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(getattr(_target, "__all__", [n for n in dir(_target) if not n.startswith("_")]))


def __getattr__(name):
    return getattr(_target, name)


def __dir__():
    return sorted(set(__all__) | set(dir(_target)))
