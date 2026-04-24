"""Covers the ``DevCoordinationStore`` optional-import contract in ``swarm.py``.

Tier B PR 4 restructured the sentinel ``DevCoordinationStore = None`` pattern
into a :data:`typing.TYPE_CHECKING`-guarded import so mypy narrows the type
correctly while runtime continues to tolerate the optional dependency being
unavailable (e.g. in stripped-down installs that omit ``aragora.nomic``).

The tests below guard that contract so future edits do not regress it.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from aragora.cli.commands import swarm


# ---------------------------------------------------------------------------
# Contract: the optional import resolves when ``aragora.nomic.dev_coordination``
# is available, exposes the real class, and survives a re-import when the
# dependency is pruned from ``sys.modules``.
# ---------------------------------------------------------------------------


def test_dev_coordination_store_is_resolved_in_normal_runtime() -> None:
    """Happy-path: the module exposes the real class when nomic is importable."""

    assert swarm.DevCoordinationStore is not None
    # Must point at the real implementation, not a sentinel shadow.
    from aragora.nomic.dev_coordination import DevCoordinationStore as real_cls

    assert swarm.DevCoordinationStore is real_cls


def test_dev_coordination_store_helper_returns_class_or_none() -> None:
    """``_dev_coordination_store_cls`` mirrors the module-level sentinel."""

    returned = swarm._dev_coordination_store_cls()
    assert returned is swarm.DevCoordinationStore


def test_dev_coordination_store_helper_handles_none_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the optional import is unavailable, the helper returns ``None``."""

    monkeypatch.setattr(swarm, "DevCoordinationStore", None)
    assert swarm._dev_coordination_store_cls() is None


def test_swarm_module_reimport_tolerates_missing_dev_coordination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading ``aragora.cli.commands.swarm`` with a stubbed ImportError for
    ``aragora.nomic.dev_coordination`` must fall back to ``None`` and must
    NOT raise.

    This protects the ``try/except ImportError`` fallback that runtime relies
    on in environments that strip the nomic subsystem.
    """

    # Drop any cached copy so the fresh import path is re-exercised.
    monkeypatch.delitem(sys.modules, "aragora.cli.commands.swarm", raising=False)

    # Force the optional dependency to appear unavailable by swapping its
    # module-level entry with one that raises on attribute access. We use
    # ``sys.modules`` so the ``from aragora.nomic.dev_coordination import
    # DevCoordinationStore`` line inside swarm.py resolves to our stub.
    class _Raiser:
        def __getattr__(self, name: str) -> object:
            raise ImportError("simulated: dev_coordination unavailable")

    monkeypatch.setitem(sys.modules, "aragora.nomic.dev_coordination", _Raiser())

    try:
        reimported = importlib.import_module("aragora.cli.commands.swarm")
    finally:
        # Restore the real module so other tests keep working.
        monkeypatch.delitem(sys.modules, "aragora.cli.commands.swarm", raising=False)

    assert reimported.DevCoordinationStore is None
    assert reimported._dev_coordination_store_cls() is None


def test_swarm_module_annotations_expose_class_to_type_checkers() -> None:
    """Smoke-check that ``DevCoordinationStore`` is a name on the swarm module.

    The TYPE_CHECKING guard in swarm.py must also bind the name unconditionally
    for runtime dispatch — losing that binding would silently regress the
    optional-import contract (the sentinel ``None`` path depends on the name
    existing).
    """

    assert hasattr(swarm, "DevCoordinationStore")
    assert hasattr(swarm, "_dev_coordination_store_cls")
