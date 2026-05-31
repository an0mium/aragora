"""
Tests for durable local-mode audit session persistence.

Regression coverage for the defect where a session created by
``aragora audit --local create`` could not be retrieved by a subsequent
separate ``aragora audit start/status/...`` invocation because sessions
were only held in a process-global in-memory dict.

Each new Python process re-created an empty auditor, so the documented
create -> start -> status workflow could not complete across CLI
invocations in local mode.

These tests simulate separate CLI invocations by constructing a *fresh*
``DocumentAuditor`` instance pointed at the same on-disk data directory
(monkeypatching ``get_default_data_dir`` per the repo testing convention).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from aragora.audit.document_auditor import (
    AuditFinding,
    AuditSession,
    AuditStatus,
    AuditType,
    DocumentAuditor,
    FindingSeverity,
    FindingStatus,
)


@pytest.fixture
def patched_data_dir(tmp_path, monkeypatch):
    """Point the audit session store at an isolated temp data dir.

    Patches ``get_default_data_dir`` (the function, not a constant) on the
    persistence module so the auditor's SQLite store writes under tmp_path.
    """
    from aragora.persistence import db_config

    monkeypatch.setattr(db_config, "get_default_data_dir", lambda: tmp_path)
    # Also patch any re-export the auditor imports directly.
    import aragora.audit.session_store as session_store

    monkeypatch.setattr(session_store, "get_default_data_dir", lambda: tmp_path)
    return tmp_path


def _make_persistent_auditor() -> DocumentAuditor:
    """Construct an auditor configured to persist local sessions to disk."""
    return DocumentAuditor(persist_sessions=True)


class TestCrossInvocationPersistence:
    """Session created in one process must be readable in a later process."""

    def test_session_survives_fresh_auditor(self, patched_data_dir):
        # --- Process 1: create the session ---
        auditor1 = _make_persistent_auditor()
        session = asyncio.run(
            auditor1.create_session(
                document_ids=["doc1", "doc2"],
                audit_types=["security", "quality"],
                name="Test",
            )
        )
        session_id = session.id
        assert session_id

        # --- Process 2: a brand-new auditor (simulating a new CLI process) ---
        auditor2 = _make_persistent_auditor()
        loaded = auditor2.get_session(session_id)

        assert loaded is not None, "Session must be retrievable across invocations"
        assert loaded.id == session_id
        assert loaded.name == "Test"
        assert loaded.document_ids == ["doc1", "doc2"]
        assert AuditType.SECURITY in loaded.audit_types
        assert AuditType.QUALITY in loaded.audit_types
        assert loaded.status == AuditStatus.PENDING

    def test_status_after_create(self, patched_data_dir):
        """The documented create -> status workflow works across processes."""
        auditor1 = _make_persistent_auditor()
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Workflow"))

        # Fresh auditor — mimics ``aragora audit --local status <id>``.
        auditor2 = _make_persistent_auditor()
        loaded = auditor2.get_session(session.id)
        assert loaded is not None
        assert loaded.status == AuditStatus.PENDING
        assert loaded.name == "Workflow"

    def test_state_transition_persists(self, patched_data_dir):
        """A state transition in one process is visible to the next."""
        auditor1 = _make_persistent_auditor()
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Transition"))

        # Mutate + persist (simulating start/progress).
        session.status = AuditStatus.RUNNING
        session.started_at = datetime.now(timezone.utc)
        session.progress = 0.42
        session.current_phase = "verification"
        auditor1.save_session(session)

        auditor2 = _make_persistent_auditor()
        loaded = auditor2.get_session(session.id)
        assert loaded is not None
        assert loaded.status == AuditStatus.RUNNING
        assert loaded.progress == pytest.approx(0.42)
        assert loaded.current_phase == "verification"
        assert loaded.started_at is not None

    def test_findings_persist_faithfully(self, patched_data_dir):
        """Nested findings (with enums + datetimes) round-trip intact."""
        auditor1 = _make_persistent_auditor()
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Findings"))
        session.status = AuditStatus.COMPLETED
        session.findings = [
            AuditFinding(
                session_id=session.id,
                document_id="doc1",
                audit_type=AuditType.SECURITY,
                category="security",
                severity=FindingSeverity.HIGH,
                title="Hardcoded credential",
                description="API key in source",
                evidence_text="API_KEY = 'sk-123'",
                recommendation="Use a secrets manager",
                found_by="initial_scanner",
                confirmed_by=["verifier"],
                status=FindingStatus.OPEN,
                tags=["secret"],
            )
        ]
        auditor1.save_session(session)

        auditor2 = _make_persistent_auditor()
        loaded = auditor2.get_session(session.id)
        assert loaded is not None
        assert loaded.status == AuditStatus.COMPLETED
        assert len(loaded.findings) == 1
        f = loaded.findings[0]
        assert f.title == "Hardcoded credential"
        assert f.severity == FindingSeverity.HIGH
        assert f.audit_type == AuditType.SECURITY
        assert f.status == FindingStatus.OPEN
        assert f.confirmed_by == ["verifier"]
        assert f.tags == ["secret"]
        assert isinstance(f.created_at, datetime)

        # Findings retrievable via the public accessor too.
        public = auditor2.get_findings(session.id)
        assert len(public) == 1
        assert public[0].title == "Hardcoded credential"


class TestInMemoryModeUnchanged:
    """API/server (non-persistent) behavior must be unchanged."""

    def test_default_auditor_does_not_persist(self, patched_data_dir):
        # Default DocumentAuditor() (server/API mode) stays in-memory only.
        auditor1 = DocumentAuditor()
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Ephemeral"))

        auditor2 = DocumentAuditor()
        assert auditor2.get_session(session.id) is None
