"""Shared fixtures for the audit test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_audit_registry():
    # ``AuditRegistry`` is a process-wide singleton exposed as the module-global
    # ``audit_registry``. Some tests mutate it in place (register/clear/discover)
    # without restoring it, so under pytest-randomly an auditor left registered by
    # one test leaks into a later test that assumes a clean registry. Snapshot the
    # registry before each test and restore it afterwards so registrations never
    # cross test boundaries.
    from aragora.audit.registry import audit_registry

    snapshot = {
        "_auditors": dict(audit_registry._auditors),
        "_auditor_classes": dict(audit_registry._auditor_classes),
        "_presets": dict(audit_registry._presets),
        "_legacy_auditors": dict(audit_registry._legacy_auditors),
    }
    try:
        yield
    finally:
        for attr, saved in snapshot.items():
            current = getattr(audit_registry, attr)
            current.clear()
            current.update(saved)
