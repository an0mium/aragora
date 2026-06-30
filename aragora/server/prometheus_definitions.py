"""Deprecated import location for the Prometheus metric definitions.

The Prometheus metric definitions moved DOWN to
:mod:`aragora.observability.prometheus_definitions` during the P4a layering work
so that foundation/infrastructure modules can reach them without importing
``aragora.server``. Importing from ``aragora.server.prometheus_definitions``
still works but is deprecated; import from
``aragora.observability.prometheus_definitions`` instead.
"""

from __future__ import annotations

import warnings

from aragora.observability import prometheus_definitions as _target

warnings.warn(
    "aragora.server.prometheus_definitions is deprecated; import from "
    "aragora.observability.prometheus_definitions instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(getattr(_target, "__all__", [n for n in dir(_target) if not n.startswith("_")]))


def __getattr__(name):
    return getattr(_target, name)


def __dir__():
    return sorted(set(__all__) | set(dir(_target)))
