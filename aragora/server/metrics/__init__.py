"""Deprecated import location for the server Prometheus metrics package.

The ``aragora.server.metrics`` surface (API/debate/billing/agent/security/vector/
gateway/knowledge-mound counters plus the ``ACTIVE_DEBATES`` gauge and
``track_debate_outcome`` helper) moved DOWN to
:mod:`aragora.observability.server_metrics` during the P4a layering work so that
foundation/infrastructure modules can reach it without importing
``aragora.server``. Importing from ``aragora.server.metrics`` still works but is
deprecated; import from ``aragora.observability.server_metrics`` instead.
"""

from __future__ import annotations

import warnings

from aragora.observability import server_metrics as _target

warnings.warn(
    "aragora.server.metrics is deprecated; import from "
    "aragora.observability.server_metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(getattr(_target, "__all__", [n for n in dir(_target) if not n.startswith("_")]))


def __getattr__(name):
    return getattr(_target, name)


def __dir__():
    return sorted(set(__all__) | set(dir(_target)))
