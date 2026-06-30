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

    def test_default_store_path_survives_cwd_changes(self, tmp_path, monkeypatch):
        """Default local audit storage is stable across CLI working dirs."""
        import aragora.audit.session_store as session_store

        home = tmp_path / "home"
        cwd1 = tmp_path / "cwd1"
        cwd2 = tmp_path / "cwd2"
        home.mkdir()
        cwd1.mkdir()
        cwd2.mkdir()
        monkeypatch.delenv("ARAGORA_DATA_DIR", raising=False)
        monkeypatch.setattr(session_store.Path, "home", lambda: home)

        monkeypatch.chdir(cwd1)
        auditor1 = _make_persistent_auditor()
        session = asyncio.run(
            auditor1.create_session(document_ids=["doc1"], name="CWD Independent")
        )

        monkeypatch.chdir(cwd2)
        auditor2 = _make_persistent_auditor()
        loaded = auditor2.get_session(session.id)

        assert loaded is not None
        assert loaded.name == "CWD Independent"
        assert session_store.AuditSessionStore().db_path == home / ".aragora" / "audit_sessions.db"

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

    def test_status_filter_applies_before_limit(self, patched_data_dir):
        """Persisted status filtering must not under-return because of LIMIT."""
        auditor1 = _make_persistent_auditor()
        completed = asyncio.run(auditor1.create_session(document_ids=["old"], name="Completed"))
        completed.status = AuditStatus.COMPLETED
        completed.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        auditor1.save_session(completed)

        pending = asyncio.run(auditor1.create_session(document_ids=["new"], name="Pending"))
        pending.status = AuditStatus.PENDING
        pending.created_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
        auditor1.save_session(pending)

        auditor2 = _make_persistent_auditor()
        sessions = auditor2.list_sessions(status=AuditStatus.COMPLETED, limit=1)
        assert [s.id for s in sessions] == [completed.id]

    def test_progress_updates_persist_for_fresh_auditor(self, patched_data_dir):
        auditor1 = _make_persistent_auditor()
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Progress"))
        session.status = AuditStatus.RUNNING
        session.current_phase = "verification"
        auditor1._notify_progress(session, 0.7)

        auditor2 = _make_persistent_auditor()
        loaded = auditor2.get_session(session.id)
        assert loaded is not None
        assert loaded.status == AuditStatus.RUNNING
        assert loaded.current_phase == "verification"
        assert loaded.progress == pytest.approx(0.7)

    def test_standard_pipeline_persists_intermediate_findings(self, patched_data_dir, monkeypatch):
        auditor1 = _make_persistent_auditor()
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Findings"))

        async def fake_initial_scan(session_arg, chunks):
            return [AuditFinding(session_id=session_arg.id, document_id="doc1", title="Initial")]

        async def fake_type_audit(session_arg, chunks, audit_type):
            return []

        async def fake_verify(session_arg, findings):
            return findings

        monkeypatch.setattr(auditor1, "_initial_scan", fake_initial_scan)
        monkeypatch.setattr(auditor1, "_run_type_audit", fake_type_audit)
        monkeypatch.setattr(auditor1, "_verify_findings", fake_verify)

        asyncio.run(auditor1._execute_standard_pipeline(session, []))

        auditor2 = _make_persistent_auditor()
        loaded = auditor2.get_session(session.id)
        assert loaded is not None
        assert [finding.title for finding in loaded.findings] == ["Initial"]


class TestCrossProcessControlPersistence:
    """Cross-process control operations update durable local-mode state."""

    def test_cross_process_pause_and_cancel_mark_running_sessions(self, patched_data_dir):
        auditor1 = _make_persistent_auditor()
        pause_session = asyncio.run(
            auditor1.create_session(document_ids=["doc1"], name="Pause Running")
        )
        cancel_session = asyncio.run(
            auditor1.create_session(document_ids=["doc2"], name="Cancel Running")
        )
        for session in (pause_session, cancel_session):
            session.status = AuditStatus.RUNNING
            auditor1.save_session(session)

        auditor2 = _make_persistent_auditor()
        assert asyncio.run(auditor2.pause_audit(pause_session.id)) is True
        assert asyncio.run(auditor2.cancel_audit(cancel_session.id)) is True

        auditor3 = _make_persistent_auditor()
        assert auditor3.get_session(pause_session.id).status == AuditStatus.PAUSED
        assert auditor3.get_session(cancel_session.id).status == AuditStatus.CANCELLED

    def test_external_pause_survives_stale_runner_progress_write(self, patched_data_dir):
        import aragora.audit.document_auditor as document_auditor

        auditor1 = _make_persistent_auditor()
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Pause Race"))
        session.status = AuditStatus.RUNNING
        auditor1.save_session(session)

        auditor2 = _make_persistent_auditor()
        assert asyncio.run(auditor2.pause_audit(session.id)) is True

        # The original process still has a stale RUNNING object. Progress writes
        # must not clobber the externally requested PAUSED durable state.
        session.current_phase = "verification"
        with pytest.raises(document_auditor._AuditPaused):
            auditor1._notify_progress(session, 0.5)

        auditor3 = _make_persistent_auditor()
        assert auditor3.get_session(session.id).status == AuditStatus.PAUSED

    def test_resume_after_cross_process_pause_persists_completion(
        self,
        patched_data_dir,
        monkeypatch,
    ):
        auditor1 = _make_persistent_auditor()
        session = asyncio.run(
            auditor1.create_session(document_ids=["doc1"], name="Resume Persisted")
        )
        session.status = AuditStatus.RUNNING
        auditor1.save_session(session)

        auditor2 = _make_persistent_auditor()
        assert asyncio.run(auditor2.pause_audit(session.id)) is True

        auditor3 = _make_persistent_auditor()

        async def fake_execute(session_arg):
            session_arg.progress = 0.5
            session_arg.findings = [
                AuditFinding(session_id=session_arg.id, document_id="doc1", title="Resumed")
            ]

        monkeypatch.setattr(auditor3, "_execute_audit", fake_execute)
        resumed = asyncio.run(auditor3.resume_audit(session.id))
        assert resumed.status == AuditStatus.COMPLETED

        auditor4 = _make_persistent_auditor()
        loaded = auditor4.get_session(session.id)
        assert loaded.status == AuditStatus.COMPLETED
        assert loaded.progress == pytest.approx(0.5)
        assert [finding.title for finding in loaded.findings] == ["Resumed"]

    def test_external_cancel_survives_stale_runner_failure_write(self, patched_data_dir):
        auditor1 = _make_persistent_auditor()
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Cancel Race"))
        session.status = AuditStatus.RUNNING
        auditor1.save_session(session)

        auditor2 = _make_persistent_auditor()
        assert asyncio.run(auditor2.cancel_audit(session.id)) is True

        session.status = AuditStatus.FAILED
        session.errors.append("late failure")
        auditor1.save_session(session)

        auditor3 = _make_persistent_auditor()
        loaded = auditor3.get_session(session.id)
        assert loaded.status == AuditStatus.CANCELLED
        assert loaded.errors == []

    def test_terminal_status_survives_stale_runner_progress_write(self, patched_data_dir):
        auditor1 = _make_persistent_auditor()
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Terminal Race"))
        session.status = AuditStatus.RUNNING
        auditor1.save_session(session)

        auditor2 = _make_persistent_auditor()
        completed = auditor2.get_session(session.id)
        assert completed is not None
        completed.status = AuditStatus.COMPLETED
        completed.progress = 1.0
        completed.completed_at = datetime(2026, 1, 4, tzinfo=timezone.utc)
        auditor2.save_session(completed, force=True)

        # The stale runner must not make a terminal session appear active again.
        session.current_phase = "verification"
        auditor1._notify_progress(session, 0.5)

        loaded = _make_persistent_auditor().get_session(session.id)
        assert loaded is not None
        assert loaded.status == AuditStatus.COMPLETED
        assert loaded.progress == pytest.approx(1.0)

    def test_external_cancel_interrupts_running_process(
        self,
        patched_data_dir,
        monkeypatch,
    ):
        runner = _make_persistent_auditor()
        session = asyncio.run(runner.create_session(document_ids=["doc1"], name="Cancel Live"))

        async def fake_execute(session_arg):
            controller = _make_persistent_auditor()
            assert await controller.cancel_audit(session_arg.id) is True
            runner._notify_progress(session_arg, 0.5)
            pytest.fail("cancelled persisted state should abort progress writes")

        monkeypatch.setattr(runner, "_execute_audit", fake_execute)

        result = asyncio.run(runner.run_audit(session.id))

        assert result.status == AuditStatus.CANCELLED
        loaded = _make_persistent_auditor().get_session(session.id)
        assert loaded is not None
        assert loaded.status == AuditStatus.CANCELLED

    def test_external_pause_interrupts_running_process(
        self,
        patched_data_dir,
        monkeypatch,
    ):
        runner = _make_persistent_auditor()
        session = asyncio.run(runner.create_session(document_ids=["doc1"], name="Pause Live"))

        async def fake_execute(session_arg):
            controller = _make_persistent_auditor()
            assert await controller.pause_audit(session_arg.id) is True
            runner._notify_progress(session_arg, 0.5)
            pytest.fail("paused persisted state should abort progress writes")

        monkeypatch.setattr(runner, "_execute_audit", fake_execute)

        result = asyncio.run(runner.run_audit(session.id))

        assert result.status == AuditStatus.PAUSED
        assert result.completed_at is None
        loaded = _make_persistent_auditor().get_session(session.id)
        assert loaded is not None
        assert loaded.status == AuditStatus.PAUSED
        assert loaded.completed_at is None

    def test_run_audit_rejects_terminal_sessions(self, patched_data_dir):
        auditor = _make_persistent_auditor()
        session = asyncio.run(auditor.create_session(document_ids=["doc1"], name="Terminal Start"))
        session.status = AuditStatus.CANCELLED
        auditor.save_session(session, force=True)

        with pytest.raises(ValueError, match="Cannot start session"):
            asyncio.run(auditor.run_audit(session.id))

        fresh = _make_persistent_auditor().get_session(session.id)
        assert fresh is not None
        assert fresh.status == AuditStatus.CANCELLED

    def test_cancel_preserves_terminal_sessions(self, patched_data_dir):
        auditor1 = _make_persistent_auditor()
        session = asyncio.run(
            auditor1.create_session(document_ids=["doc1"], name="Completed Terminal")
        )
        completed_at = datetime(2026, 1, 3, tzinfo=timezone.utc)
        session.status = AuditStatus.COMPLETED
        session.completed_at = completed_at
        auditor1.save_session(session)

        auditor2 = _make_persistent_auditor()
        assert asyncio.run(auditor2.cancel_audit(session.id)) is False

        auditor3 = _make_persistent_auditor()
        loaded = auditor3.get_session(session.id)
        assert loaded.status == AuditStatus.COMPLETED
        assert loaded.completed_at == completed_at


class TestInMemoryModeUnchanged:
    """API/server (non-persistent) behavior must be unchanged."""

    def test_default_auditor_does_not_persist(self, patched_data_dir):
        # Default DocumentAuditor() (server/API mode) stays in-memory only.
        auditor1 = DocumentAuditor()
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Ephemeral"))

        auditor2 = DocumentAuditor()
        assert auditor2.get_session(session.id) is None

    def test_default_global_auditor_does_not_persist(self, patched_data_dir, monkeypatch):
        import aragora.audit.document_auditor as document_auditor_module

        monkeypatch.setattr(document_auditor_module, "_auditor", None)
        monkeypatch.setattr(document_auditor_module, "_persistent_auditor", None)

        auditor1 = document_auditor_module.get_document_auditor()
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Server"))

        # Simulate a later process: the default singleton must remain in-memory only.
        monkeypatch.setattr(document_auditor_module, "_auditor", None)
        auditor2 = document_auditor_module.get_document_auditor()
        assert auditor2.get_session(session.id) is None

    def test_cli_opt_in_global_auditor_persists(self, patched_data_dir, monkeypatch):
        import aragora.audit.document_auditor as document_auditor_module

        monkeypatch.setattr(document_auditor_module, "_auditor", None)
        monkeypatch.setattr(document_auditor_module, "_persistent_auditor", None)

        auditor1 = document_auditor_module.get_document_auditor(persist_sessions=True)
        session = asyncio.run(auditor1.create_session(document_ids=["doc1"], name="Local"))

        # Simulate a separate local CLI invocation.
        monkeypatch.setattr(document_auditor_module, "_persistent_auditor", None)
        auditor2 = document_auditor_module.get_document_auditor(persist_sessions=True)
        loaded = auditor2.get_session(session.id)
        assert loaded is not None
        assert loaded.name == "Local"
