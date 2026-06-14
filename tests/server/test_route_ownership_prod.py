"""Route-ownership assertions for resolved collisions (run-20260611 PROD lane).

PR #8098 froze 50 route collisions in ``test_route_collisions.py``. This lane
resolves a distinct subset of them (disjoint from the cluster-1 work in PR
#8163) and pins the *correct owner* of each formerly-colliding path so that a
future re-introduction of the duplicate claim fails loudly here, not silently
in production.

For each path we assert exactly one handler class claims it (no collision) AND
that the surviving handler is the canonical, behavior-preserving owner:

* ``/api/v1/debates/{id}/summary`` -> ``DebatesHandler``
  Canonical debate-resource owner; already the live first-wins winner, so
  existing clients see no change. ``ExplainabilityHandler`` carried a duplicate
  (shadowed-since-registration) ``/summary`` claim that is removed.

* ``/api/v1/notifications/history`` -> ``NotificationHistoryHandler``
  This is a genuine *fix*: the social ``NotificationsHandler`` claimed the path
  first but its ``handle()`` has no ``/history`` branch (returns ``None``), so
  the endpoint was broken. ``NotificationHistoryHandler._get_history`` is the
  real implementation (with ``notifications:read`` RBAC); removing the dead
  claim restores it.
"""

from __future__ import annotations

from functools import lru_cache

from aragora.server.handler_registry import HANDLER_REGISTRY
from aragora.server.handler_registry.core import _DeferredImport


@lru_cache(maxsize=1)
def _claims() -> dict[str, set[str]]:
    """Map every exact (normalized-version) route string to the handler names that claim it."""
    out: dict[str, set[str]] = {}
    for _attr, entry in HANDLER_REGISTRY:
        cls = entry.resolve() if isinstance(entry, _DeferredImport) else entry
        if cls is None:
            continue
        for raw in getattr(cls, "ROUTES", None) or []:
            if not isinstance(raw, str):
                continue
            out.setdefault(raw, set()).add(cls.__name__)
    return out


def _owners(path: str) -> set[str]:
    return _claims().get(path, set())


def test_debates_summary_owned_only_by_debates_handler() -> None:
    owners = _owners("/api/v1/debates/*/summary")
    assert owners == {"DebatesHandler"}, (
        "Expected DebatesHandler to be the sole owner of "
        f"/api/v1/debates/*/summary, got {sorted(owners)}. ExplainabilityHandler "
        "must not re-claim this path."
    )


def test_notifications_history_owned_only_by_history_handler() -> None:
    owners = _owners("/api/v1/notifications/history")
    assert owners == {"NotificationHistoryHandler"}, (
        "Expected NotificationHistoryHandler (the real implementation) to be the "
        "sole owner of /api/v1/notifications/history, got "
        f"{sorted(owners)}. The social NotificationsHandler must not claim it — "
        "its handle() has no /history branch and shadowed the working handler."
    )
