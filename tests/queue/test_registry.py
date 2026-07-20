"""Tests for the domain-free job-handler registry (P4a queue inversion Q1/Q2/Q4).

Covers the enabler surface introduced by the queue/job-handler registry
inversion (docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §4.1, §5.2, §10 Q1/Q2):

- ``register_worker`` / ``register_job_handler`` / ``get_registered_workers`` /
  ``get_registered_job_handlers`` / ``get_job_handler`` / ``reset_registry`` on
  ``aragora.queue.worker`` (re-exported from ``aragora.queue``).
- ``DebateWorker`` + ``DebateExecutor`` re-exports from ``aragora.queue`` are
  KEPT.
- The ``create_default_executor`` re-export is DROPPED from ``aragora.queue``
  (no shim, §10 Q2). The factory itself relocated out of ``aragora.queue.worker``
  to its debate-layer home ``aragora.debate.queue_executor``.
- The registry functions carry ZERO domain imports.

Also covers the Q4 registry-hardening rider: ``register_worker`` warns (but
still allows) overwriting a name with a different instance, and
``register_job_handler`` rejects a non-callable or non-async handler at
registration time.
"""

from __future__ import annotations

import inspect
import logging

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


def test_register_worker_overwrite_with_different_instance_logs_warning(caplog):
    """Overwriting with a different instance is still allowed (Q4 rider: warn,
    not fail-closed - a worker may legitimately restart under the same name)
    but must be visible instead of silent."""
    first, second = object(), object()
    register_worker("dup", first)

    with caplog.at_level(logging.WARNING, logger="aragora.queue.worker"):
        register_worker("dup", second)

    assert get_registered_workers()["dup"] is second
    assert "Overwriting registered worker" in caplog.text
    assert "'dup'" in caplog.text


def test_register_worker_reregister_same_instance_no_warning(caplog):
    instance = object()
    register_worker("w1", instance)

    with caplog.at_level(logging.WARNING, logger="aragora.queue.worker"):
        register_worker("w1", instance)

    assert get_registered_workers()["w1"] is instance
    assert "Overwriting registered worker" not in caplog.text


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


def test_register_job_handler_rejects_non_callable():
    with pytest.raises(TypeError, match="must be callable"):
        register_job_handler("not-callable", "not-a-function")  # type: ignore[arg-type]

    assert get_job_handler("not-callable") is None


def test_register_job_handler_rejects_sync_function():
    def sync_handler(job: object) -> dict[str, object]:
        return {}

    with pytest.raises(TypeError, match="must be an async callable"):
        register_job_handler("sync", sync_handler)  # type: ignore[arg-type]

    assert get_job_handler("sync") is None


def test_register_job_handler_accepts_bound_async_method():
    """Regression guard: the async-callable validation must not reject the
    bound-method registration pattern the conflict-detection tests rely on."""

    class Handler:
        async def handle(self, job: object) -> dict[str, object]:
            return {}

    instance = Handler()
    register_job_handler("bound-ok", instance.handle)

    registered = get_job_handler("bound-ok")
    assert registered is not None
    # A fresh `instance.handle` access is a distinct bound-method object each
    # time (no `is` identity across accesses), so compare __self__/__func__
    # like `_same_job_handler` does rather than `is instance.handle`.
    assert registered.__self__ is instance
    assert registered.__func__ is Handler.handle


def test_register_job_handler_type_validation_precedes_conflict_check():
    """A badly-typed re-registration must raise TypeError (validation), not
    ValueError (conflict), and must not clobber the existing handler."""
    register_job_handler("typed", _noop_handler)

    def sync_handler(job: object) -> dict[str, object]:
        return {}

    with pytest.raises(TypeError, match="must be an async callable"):
        register_job_handler("typed", sync_handler)  # type: ignore[arg-type]

    assert get_job_handler("typed") is _noop_handler


def test_register_job_handler_is_idempotent_for_same_bound_method():
    class Handler:
        async def handle(self, job: object) -> dict[str, object]:
            return {}

    instance = Handler()

    register_job_handler("bound", instance.handle)
    register_job_handler("bound", instance.handle)

    registered = get_job_handler("bound")
    assert registered is not None
    assert registered.__self__ is instance
    assert registered.__func__ is Handler.handle
    assert registered_job_handler_names().count("bound") == 1


def test_register_job_handler_rejects_same_method_on_different_instance():
    class Handler:
        async def handle(self, job: object) -> dict[str, object]:
            return {}

    first = Handler()
    second = Handler()

    register_job_handler("bound", first.handle)

    with pytest.raises(ValueError, match="job handler already registered"):
        register_job_handler("bound", second.handle)

    registered = get_job_handler("bound")
    assert registered is not None
    assert registered.__self__ is first
    assert registered.__func__ is Handler.handle


def test_register_job_handler_rejects_different_method_on_same_instance():
    class Handler:
        async def handle(self, job: object) -> dict[str, object]:
            return {}

        async def other(self, job: object) -> dict[str, object]:
            return {}

    instance = Handler()

    register_job_handler("bound", instance.handle)

    with pytest.raises(ValueError, match="job handler already registered"):
        register_job_handler("bound", instance.other)

    registered = get_job_handler("bound")
    assert registered is not None
    assert registered.__self__ is instance
    assert registered.__func__ is Handler.handle


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


def test_create_default_executor_reexport_dropped_from_queue_package():
    """The domain-coupled factory re-export is dropped (no shim, §5.2)."""
    import aragora.queue as queue_pkg

    assert not hasattr(queue_pkg, "create_default_executor")
    assert "create_default_executor" not in queue_pkg.__all__

    # Q2 relocated the factory itself out of worker.py to its debate-layer
    # home; the queue package (and its worker module) no longer defines it.
    assert not hasattr(worker, "create_default_executor")

    from aragora.debate.queue_executor import create_default_executor

    assert callable(create_default_executor)


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

    Scoped to just the registry callables (instead of the whole file) so a
    future addition to ``worker.py`` can't silently introduce a domain import
    without failing this guard; the full-layer grimp/import-linter re-check is
    the authoritative proof that ``aragora.queue`` gains no new edge.
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
