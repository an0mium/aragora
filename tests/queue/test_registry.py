"""Tests for the domain-free job-handler registry (P4a queue inversion Q1).

Covers the enabler surface introduced by the queue/job-handler registry
inversion (docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §4.1, §5.2, §10 Q1):

- ``register_worker`` / ``register_job_handler`` / ``get_registered_workers`` /
  ``get_registered_job_handlers`` / ``get_job_handler`` / ``reset_registry`` on
  ``aragora.queue.worker`` (re-exported from ``aragora.queue``).
- ``DebateWorker`` + ``DebateExecutor`` re-exports from ``aragora.queue`` are
  KEPT.
- The ``create_default_executor`` compatibility re-export is KEPT from
  ``aragora.queue`` while the underlying symbol stays defined in
  ``aragora.queue.worker`` (it only relocates in Q2).
- The registry functions carry ZERO domain imports.
"""

from __future__ import annotations

import inspect

import pytest

from aragora.queue import worker
from aragora.queue.worker import (
    DebateExecutor,
    DebateWorker,
    get_job_handler,
    get_registered_job_handlers,
    get_registered_workers,
    register_job_handler,
    register_worker,
    registered_job_handler_names,
    registered_worker_names,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate each test: empty registry before and after."""
    reset_registry()
    yield
    reset_registry()


async def _noop_handler(job: object) -> dict[str, object]:
    return {}


def test_register_worker_and_get():
    instance = object()
    register_worker("w1", instance)

    registered = get_registered_workers()
    assert registered["w1"] is instance
    assert "w1" in registered_worker_names()


def test_register_worker_is_keyed_idempotent():
    first, second = object(), object()
    register_worker("dup", first)
    register_worker("dup", second)

    registered = get_registered_workers()
    assert registered["dup"] is second
    assert registered_worker_names().count("dup") == 1


def test_get_registered_workers_returns_copy_not_live_reference():
    register_worker("w1", object())
    snapshot = get_registered_workers()
    snapshot["injected"] = object()

    assert "injected" not in get_registered_workers()


def test_register_job_handler_and_get():
    register_job_handler("demo", _noop_handler)

    assert get_job_handler("demo") is _noop_handler
    assert get_registered_job_handlers()["demo"] is _noop_handler
    assert "demo" in registered_job_handler_names()


def test_register_job_handler_is_keyed_idempotent():
    register_job_handler("dup", _noop_handler)
    register_job_handler("dup", _noop_handler)

    assert get_job_handler("dup") is _noop_handler
    assert registered_job_handler_names().count("dup") == 1


def test_register_job_handler_rejects_different_duplicate():
    async def first(job: object) -> dict[str, object]:
        return {}

    async def second(job: object) -> dict[str, object]:
        return {}

    register_job_handler("dup", first)

    with pytest.raises(ValueError, match="job handler already registered"):
        register_job_handler("dup", second)

    assert get_job_handler("dup") is first


def test_get_job_handler_returns_none_when_unregistered():
    assert get_job_handler("missing") is None


def test_get_registered_job_handlers_returns_copy_not_live_reference():
    register_job_handler("demo", _noop_handler)
    snapshot = get_registered_job_handlers()
    snapshot["injected"] = _noop_handler

    assert "injected" not in get_registered_job_handlers()


def test_reset_registry_clears_both_registries():
    register_worker("w1", object())
    register_job_handler("demo", _noop_handler)

    reset_registry()

    assert get_registered_workers() == {}
    assert get_registered_job_handlers() == {}
    assert registered_worker_names() == []
    assert registered_job_handler_names() == []


def test_worker_and_job_handler_registries_are_independent():
    """Registering under the same key in each registry must not collide."""
    register_worker("shared_name", "worker-instance")
    register_job_handler("shared_name", _noop_handler)

    assert get_registered_workers()["shared_name"] == "worker-instance"
    assert get_job_handler("shared_name") is _noop_handler


def test_debate_worker_and_executor_alias_reexported_from_queue_package():
    """DebateWorker + DebateExecutor stay re-exported (§5.2: both are public)."""
    import aragora.queue as queue_pkg

    assert queue_pkg.DebateWorker is DebateWorker
    assert queue_pkg.DebateExecutor is DebateExecutor
    assert "DebateWorker" in queue_pkg.__all__
    assert "DebateExecutor" in queue_pkg.__all__


def test_create_default_executor_reexport_kept_from_queue_package():
    """The Q1 registry enabler keeps the documented factory import path."""
    import aragora.queue as queue_pkg

    # The underlying symbol is untouched - it only relocates out of worker.py
    # in Q2, so it must still be directly importable from its defining module.
    assert queue_pkg.create_default_executor is worker.create_default_executor
    assert "create_default_executor" in queue_pkg.__all__
    assert hasattr(worker, "create_default_executor")


def test_registry_functions_reexported_from_queue_package():
    import aragora.queue as queue_pkg

    names = (
        "register_worker",
        "register_job_handler",
        "get_registered_workers",
        "get_registered_job_handlers",
        "get_job_handler",
        "registered_worker_names",
        "registered_job_handler_names",
        "reset_registry",
    )
    for name in names:
        assert hasattr(queue_pkg, name), f"{name} missing from aragora.queue"
        assert name in queue_pkg.__all__, f"{name} missing from aragora.queue.__all__"
        assert getattr(queue_pkg, name) is getattr(worker, name)


def test_registry_functions_have_zero_domain_imports():
    """The registry's own functions must not reference any domain package.

    ``worker.py`` also hosts ``create_default_executor`` (pending relocation to
    a debate-layer home in Q2), which legitimately lazy-imports domain packages
    inside its own body. Scoping this static guard to just the registry
    callables (instead of the whole file) avoids a false positive on that
    pre-existing, unrelated code; the full-layer grimp/import-linter re-check
    is the authoritative proof that ``aragora.queue`` gains no new edge.
    """
    domain_packages = (
        "agents",
        "debate",
        "gauntlet",
        "integrations",
        "memory",
        "nomic",
        "ranking",
        "server",
    )
    registry_callables = (
        register_worker,
        register_job_handler,
        get_registered_workers,
        get_registered_job_handlers,
        get_job_handler,
        registered_worker_names,
        registered_job_handler_names,
        reset_registry,
    )
    for fn in registry_callables:
        source = inspect.getsource(fn)
        offenders = [pkg for pkg in domain_packages if f"aragora.{pkg}" in source]
        assert not offenders, f"{fn.__name__} references domain packages: {offenders}"
