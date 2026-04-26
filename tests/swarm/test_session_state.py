from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aragora.swarm.session_state import (
    SessionState,
    SessionStateStore,
    classify_session_blocker,
    load_resume_context_for_issue,
    summarize_session_blocker,
)
from aragora.swarm.supervisor_workers import _record_session_state


def _dt(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def test_session_state_roundtrip_serialization() -> None:
    created_at = _dt("2026-04-13T09:30:00+00:00")
    updated_at = _dt("2026-04-13T10:00:00+00:00")
    state = SessionState(
        session_id="bc01-repair-session",
        status="needs_human",
        issue_number=5247,
        target_agent="codex",
        runner_type="codex",
        worktree_path="/tmp/aragora-bc01",
        branch_name="codex/bc01-session-state",
        pr_url="https://github.com/synaptent/aragora/pull/9999",
        resume_hint="resume after review",
        retry_count=2,
        phase="repair",
        attempts=[
            {
                "at": "2026-04-13T09:45:00+00:00",
                "exit_code": 1,
                "changed_files": ["aragora/swarm/session_state.py"],
                "test_output": "AssertionError: boom",
                "worker_outcome": "needs_human",
            }
        ],
        repair_journal=[{"at": "2026-04-13T09:50:00+00:00", "note": "tighten retry path"}],
        metadata={"receipt_id": "rcpt-123", "phase": "skeleton"},
        created_at=created_at,
        updated_at=updated_at,
    )

    restored = SessionState.from_dict(state.to_dict())

    assert restored.to_dict() == state.to_dict()


def test_session_state_defaults_to_explore_phase() -> None:
    state = SessionState(session_id="default-phase")

    assert state.phase == "explore"
    assert state.attempts == []
    assert state.repair_journal == []


def test_session_state_store_save_and_load(tmp_path: Path) -> None:
    store = SessionStateStore(state_dir=tmp_path)
    state = SessionState(
        session_id="save-load",
        issue_number=6001,
        target_agent="codex",
        worktree_path="/tmp/worktree",
    )

    saved = store.save(state)
    loaded = store.load("save-load")

    assert saved == tmp_path / "save-load.json"
    assert loaded is not None
    assert loaded.session_id == "save-load"
    assert loaded.issue_number == 6001
    assert loaded.target_agent == "codex"


def test_session_state_store_lists_by_issue_number(tmp_path: Path) -> None:
    store = SessionStateStore(state_dir=tmp_path)
    older = SessionState(
        session_id="older",
        issue_number=7001,
        updated_at=_dt("2026-04-13T08:00:00+00:00"),
    )
    newer = SessionState(
        session_id="newer",
        issue_number=7001,
        updated_at=_dt("2026-04-13T09:00:00+00:00"),
    )
    other = SessionState(
        session_id="other",
        issue_number=7002,
        updated_at=_dt("2026-04-13T10:00:00+00:00"),
    )

    store.save(older)
    store.save(newer)
    store.save(other)

    items = store.list_sessions(issue_number=7001)

    assert [item.session_id for item in items] == ["newer", "older"]


def test_session_state_store_cleanup_old_removes_stale_files(tmp_path: Path) -> None:
    store = SessionStateStore(state_dir=tmp_path)
    stale = SessionState(
        session_id="stale",
        updated_at=_dt("2026-04-10T10:00:00+00:00"),
    )
    fresh = SessionState(
        session_id="fresh",
        updated_at=_dt("2026-04-13T10:00:00+00:00"),
    )

    stale_path = store.save(stale)
    fresh_path = store.save(fresh)
    stale.updated_at = _dt("2026-04-10T10:00:00+00:00")
    fresh.updated_at = _dt("2026-04-13T10:00:00+00:00")
    stale_path.write_text(json.dumps(stale.to_dict(), indent=2) + "\n", encoding="utf-8")
    fresh_path.write_text(json.dumps(fresh.to_dict(), indent=2) + "\n", encoding="utf-8")
    removed = store.cleanup_old(
        older_than=_dt("2026-04-12T00:00:00+00:00"),
        now=_dt("2026-04-13T12:00:00+00:00"),
    )

    assert removed == [stale_path]
    assert not stale_path.exists()
    assert fresh_path.exists()


def test_session_state_store_load_missing_file_returns_none(tmp_path: Path) -> None:
    store = SessionStateStore(state_dir=tmp_path)

    assert store.load("missing-session") is None


def test_session_state_store_default_path_uses_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    store = SessionStateStore()

    assert store.state_dir == tmp_path / ".aragora" / "sessions"


def test_record_attempt_updates_retry_count_and_last_attempt() -> None:
    state = SessionState(session_id="resume", phase="verify")

    recorded = state.record_attempt(
        1,
        ["aragora/swarm/session_state.py", "tests/swarm/test_session_state.py"],
        "AssertionError: expected resume context",
        "needs_human",
    )

    assert state.retry_count == 1
    assert recorded["changed_files"] == [
        "aragora/swarm/session_state.py",
        "tests/swarm/test_session_state.py",
    ]
    assert state.last_attempt() == recorded


def test_should_resume_when_previous_attempt_changed_files() -> None:
    state = SessionState(session_id="resume-files", phase="repair")
    state.record_attempt(1, ["aragora/swarm/session_state.py"], "", "needs_human")

    assert state.should_resume() is True


def test_should_resume_when_repair_journal_exists_without_attempts() -> None:
    state = SessionState(
        session_id="resume-journal",
        phase="repair",
        repair_journal=[{"at": "2026-04-13T11:00:00+00:00", "note": "retry with narrowed scope"}],
    )

    assert state.should_resume() is True


def test_should_not_resume_for_clean_exploration_state() -> None:
    state = SessionState(session_id="fresh-start", phase="explore")

    assert state.should_resume() is False
    assert state.resume_context() == ""


def test_resume_context_summarizes_prior_attempts() -> None:
    state = SessionState(
        session_id="resume-context",
        phase="repair",
        resume_hint="continue from the failing pytest lane",
    )
    state.record_attempt(
        1,
        ["aragora/swarm/session_state.py"],
        "AssertionError: blocker evidence missing",
        "needs_human",
    )

    text = state.resume_context()

    assert "Resume from phase: repair" in text
    assert "continue from the failing pytest lane" in text
    assert "Attempt 1" in text
    assert "AssertionError: blocker evidence missing" in text


def test_resume_context_includes_repair_journal_details() -> None:
    state = SessionState(
        session_id="resume-journal-details",
        phase="repair",
        repair_journal=[
            {
                "at": "2026-04-13T11:00:00+00:00",
                "worker_outcome": "merge_gate_failed",
                "failure_reason": "merge_gate_failed",
                "changed_paths": ["aragora/swarm/supervisor_workers.py"],
                "blocker_evidence": "Quality Gates failed on lint-run",
                "failing_verification": {
                    "command": "python -m pytest tests/swarm/test_supervisor.py -q",
                    "exit_code": 1,
                    "stderr_tail": "AssertionError: boom",
                },
            }
        ],
    )

    text = state.resume_context()

    assert "Repair journal entries available: 1" in text
    assert "merge_gate_failed" in text
    assert "python -m pytest tests/swarm/test_supervisor.py -q" in text
    assert "Quality Gates failed on lint-run" in text


def test_blocker_evidence_roundtrips_through_serialization() -> None:
    state = SessionState(
        session_id="blocker-roundtrip", blocker_evidence="pytest timed out on lane"
    )

    restored = SessionState.from_dict(state.to_dict())

    assert restored.blocker_evidence == "pytest timed out on lane"


def test_set_blocker_and_clear_blocker() -> None:
    state = SessionState(session_id="set-blocker")

    state.set_blocker("ModuleNotFoundError: missing helper")
    assert state.blocker_evidence == "ModuleNotFoundError: missing helper"

    state.clear_blocker()
    assert state.blocker_evidence is None


def test_record_session_state_creates_session_from_leased_work_order(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    work_order = {
        "work_order_id": "wo-lease",
        "issue_number": 5502,
        "target_agent": "codex",
        "branch": "codex/issue-5502-session-lifecycle",
        "worktree_path": "/tmp/codex-session-lifecycle",
        "metadata": {"supervisor_run_id": "run-lease"},
    }

    _record_session_state(work_order, status="leased", phase="dispatch")

    session_data = work_order["metadata"]["session_state"]
    assert session_data["status"] == "leased"
    assert session_data["phase"] == "dispatch"
    assert session_data["branch_name"] == "codex/issue-5502-session-lifecycle"
    assert session_data["worktree_path"] == "/tmp/codex-session-lifecycle"
    assert session_data["target_agent"] == "codex"
    assert session_data["runner_type"] == "codex"
    assert session_data["metadata"]["supervisor_run_id"] == "run-lease"


def test_record_session_state_syncs_branch_pr_and_blocker_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    work_order = {
        "work_order_id": "wo-dispatch",
        "issue_number": 5502,
        "target_agent": "claude",
        "branch": "claude/bc01-lifecycle",
        "worktree_path": "/tmp/claude-bc01-lifecycle",
        "pr_url": "https://github.com/synaptent/aragora/pull/12345",
        "receipt_id": "rcpt-123",
        "review_status": "pending_heterogeneous_review",
        "lease_id": "lease-123",
        "metadata": {
            "supervisor_run_id": "run-dispatch",
            "session_state": {
                "session_id": "resume-dispatch",
                "phase": "dispatch",
                "status": "created",
            },
        },
    }

    _record_session_state(
        work_order,
        status="dispatched",
        phase="edit",
        blocker_evidence="pytest timed out in tests/swarm/test_supervisor.py",
    )

    session_data = work_order["metadata"]["session_state"]
    assert session_data["session_id"] == "resume-dispatch"
    assert session_data["status"] == "dispatched"
    assert session_data["phase"] == "edit"
    assert session_data["branch_name"] == "claude/bc01-lifecycle"
    assert session_data["pr_url"] == "https://github.com/synaptent/aragora/pull/12345"
    assert session_data["blocker_evidence"] == "pytest timed out in tests/swarm/test_supervisor.py"
    assert session_data["metadata"]["receipt_id"] == "rcpt-123"
    assert session_data["metadata"]["review_status"] == "pending_heterogeneous_review"
    store = SessionStateStore()
    restored = store.load("resume-dispatch")
    assert restored is not None
    assert restored.pr_url == "https://github.com/synaptent/aragora/pull/12345"
    assert restored.branch_name == "claude/bc01-lifecycle"


def test_record_session_state_persists_terminal_attempt_and_publish_phase(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    work_order = {
        "work_order_id": "wo-publish",
        "issue_number": 5515,
        "target_agent": "codex",
        "branch": "codex/bc01-publish-phase",
        "worktree_path": "/tmp/codex-bc01-publish-phase",
        "status": "completed",
        "worker_outcome": "completed",
        "exit_code": 0,
        "changed_paths": ["aragora/swarm/session_state.py"],
        "stdout_tail": "worker output",
        "stderr_tail": "",
        "pr_url": "https://github.com/synaptent/aragora/pull/5515",
        "review_status": "pending_heterogeneous_review",
        "metadata": {
            "supervisor_run_id": "run-publish",
            "repair_journal": [
                {
                    "at": "2026-04-13T12:00:00+00:00",
                    "worker_outcome": "completed",
                    "failure_reason": "merge_gate_failed",
                    "changed_paths": ["aragora/swarm/session_state.py"],
                    "blocker_evidence": "Previous merge gate failure",
                    "failing_verification": {
                        "command": "python -m pytest tests/swarm/test_session_state.py -q",
                        "exit_code": 1,
                        "stderr_tail": "AssertionError: before retry",
                    },
                }
            ],
        },
    }

    _record_session_state(work_order, status="completed", phase="terminal")

    session_data = work_order["metadata"]["session_state"]
    assert session_data["phase"] == "publish"
    assert session_data["repair_journal"][0]["failure_reason"] == "merge_gate_failed"
    assert session_data["attempts"][-1]["worker_outcome"] == "completed"
    assert session_data["attempts"][-1]["changed_files"] == ["aragora/swarm/session_state.py"]
    assert session_data["attempts"][-1]["failing_verification"]["command"] == (
        "python -m pytest tests/swarm/test_session_state.py -q"
    )

    store = SessionStateStore()
    restored = store.load("swarm-wo-publish")
    assert restored is not None
    assert restored.phase == "publish"
    assert restored.repair_journal[0]["blocker_evidence"] == "Previous merge gate failure"
    assert restored.attempts[-1]["stdout_tail"] == "worker output"


def test_classify_session_blocker_import_error() -> None:
    state = SessionState(session_id="import-blocker")
    state.record_attempt(1, [], "ModuleNotFoundError: No module named 'aragora.foo'", "failed")

    result = classify_session_blocker(state)

    assert result["blocker_type"] == "import_error"
    assert "ModuleNotFoundError" in result["evidence"]


def test_classify_session_blocker_dependency_missing() -> None:
    state = SessionState(session_id="dependency-blocker")
    state.record_attempt(127, [], "ruff: command not found", "failed")

    result = classify_session_blocker(state)

    assert result["blocker_type"] == "dependency_missing"
    assert "command not found" in result["evidence"]


def test_classify_session_blocker_timeout_from_exit_code() -> None:
    state = SessionState(session_id="timeout-blocker")
    state.record_attempt(-1, [], "Timed out after 600s", "failed")

    result = classify_session_blocker(state)

    assert result["blocker_type"] == "timeout"
    assert "Timed out" in result["evidence"]


def test_classify_session_blocker_scope_too_broad() -> None:
    state = SessionState(session_id="scope-blocker")
    state.set_blocker("Issue was quarantined by task sanitizer: file scope spans 6 files.")

    result = classify_session_blocker(state)

    assert result["blocker_type"] == "scope_too_broad"
    assert "file scope spans 6 files" in result["evidence"]


def test_classify_session_blocker_defaults_to_test_failure() -> None:
    state = SessionState(session_id="test-blocker")
    state.record_attempt(1, [], "AssertionError: expected blocker evidence", "failed")

    result = classify_session_blocker(state)

    assert result["blocker_type"] == "test_failure"
    assert "AssertionError" in result["evidence"]


def test_session_state_record_attempt_builds_resume_payload() -> None:
    state = SessionState(session_id="issue-8101", issue_number=8101)

    state.record_attempt(
        status="needs_human",
        outcome="blocked",
        exit_code=1,
        changed_files=[
            "aragora/swarm/boss_loop.py",
            "aragora/swarm/boss_loop.py",
            "tests/swarm/test_boss_loop.py",
        ],
        target_agent="codex",
        runner_type="codex",
        branch_name="codex/session2-boss-loop-state",
        resume_hint="pytest -q tests/swarm/test_boss_loop.py failed",
        metadata={
            "failure_reason": "pytest -q tests/swarm/test_boss_loop.py failed",
            "failing_verification": {
                "command": "pytest -q tests/swarm/test_boss_loop.py",
                "exit_code": 1,
                "stderr_tail": "assert False",
            },
        },
    )

    context = state.resume_payload()

    assert state.retry_count == 1
    assert state.status == "needs_human"
    assert context["resume_hint"] == "pytest -q tests/swarm/test_boss_loop.py failed"
    assert context["target_agent"] == "codex"
    assert context["repair_journal"][0]["changed_paths"] == [
        "aragora/swarm/boss_loop.py",
        "tests/swarm/test_boss_loop.py",
    ]
    assert context["last_attempt"]["failing_verification"]["command"] == (
        "pytest -q tests/swarm/test_boss_loop.py"
    )


def test_session_state_store_latest_for_issue_returns_newest(tmp_path: Path) -> None:
    store = SessionStateStore(state_dir=tmp_path)
    older = SessionState(
        session_id="older-issue",
        issue_number=8102,
        updated_at=_dt("2026-04-13T08:00:00+00:00"),
    )
    newer = SessionState(
        session_id="newer-issue",
        issue_number=8102,
        updated_at=_dt("2026-04-13T09:00:00+00:00"),
    )

    store.save(older)
    store.save(newer)

    latest = store.latest_for_issue(8102)

    assert latest is not None
    assert latest.session_id == "newer-issue"


def test_session_state_store_latest_for_issue_filters_by_repo_slug(tmp_path: Path) -> None:
    store = SessionStateStore(state_dir=tmp_path)
    current_repo = SessionState(
        session_id="current-repo-issue",
        issue_number=8102,
        updated_at=_dt("2026-04-13T08:00:00+00:00"),
        metadata={"repo_slug": "synaptent/aragora"},
    )
    other_repo = SessionState(
        session_id="other-repo-issue",
        issue_number=8102,
        updated_at=_dt("2026-04-13T09:00:00+00:00"),
        metadata={"repo_slug": "other/repo"},
    )

    store.save(current_repo)
    store.save(other_repo)

    latest = store.latest_for_issue(8102, repo_slug="synaptent/aragora")

    assert latest is not None
    assert latest.session_id == "current-repo-issue"


def test_session_state_store_record_attempt_reuses_latest_issue_session(tmp_path: Path) -> None:
    store = SessionStateStore(state_dir=tmp_path)
    existing = SessionState(session_id="issue-8103", issue_number=8103, retry_count=1)
    store.save(existing)

    updated = store.record_attempt(
        issue_number=8103,
        status="needs_human",
        outcome="blocked",
        exit_code=1,
        changed_files=["aragora/swarm/boss_loop.py"],
        target_agent="codex",
        runner_type="codex",
        resume_hint="verification failed",
        metadata={"failure_reason": "verification failed"},
    )
    reloaded = store.load("issue-8103")

    assert updated.session_id == "issue-8103"
    assert reloaded is not None
    assert reloaded.retry_count == 2
    assert reloaded.attempts[-1]["exit_code"] == 1
    assert reloaded.attempts[-1]["changed_paths"] == ["aragora/swarm/boss_loop.py"]


def test_session_state_store_record_attempt_does_not_cross_repo_slug(tmp_path: Path) -> None:
    store = SessionStateStore(state_dir=tmp_path)
    other_repo_state = SessionState(
        session_id="issue-other-repo-8103",
        issue_number=8103,
        retry_count=1,
        metadata={"repo_slug": "other/repo"},
    )
    store.save(other_repo_state)

    updated = store.record_attempt(
        issue_number=8103,
        repo_slug="synaptent/aragora",
        status="needs_human",
        outcome="blocked",
        exit_code=1,
        changed_files=["aragora/swarm/boss_loop.py"],
        target_agent="codex",
        runner_type="codex",
        resume_hint="verification failed",
        metadata={"failure_reason": "verification failed"},
    )

    assert updated.session_id == "issue-synaptent-aragora-8103"
    assert updated.metadata["repo_slug"] == "synaptent/aragora"
    reloaded_other = store.load("issue-other-repo-8103")
    assert reloaded_other is not None
    assert reloaded_other.retry_count == 1


def test_load_resume_context_for_issue_filters_by_repo_slug(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    store = SessionStateStore()
    store.record_attempt(
        issue_number=1734,
        repo_slug="synaptent/aragora",
        status="needs_human",
        outcome="blocked",
        exit_code=1,
        changed_files=["aragora/swarm/session_state.py"],
        resume_hint="current repo hint",
        metadata={"failure_reason": "current repo failure"},
    )
    store.record_attempt(
        issue_number=1734,
        repo_slug="other/repo",
        status="needs_human",
        outcome="blocked",
        exit_code=1,
        changed_files=["other/repo.py"],
        resume_hint="other repo hint",
        metadata={"failure_reason": "other repo failure"},
    )

    resume_context = load_resume_context_for_issue(1734, repo_slug="synaptent/aragora")

    assert "current repo hint" in resume_context
    assert "other repo hint" not in resume_context


def test_summarize_session_blocker_prefers_failing_verification_summary() -> None:
    state = SessionState(
        session_id="issue-8104",
        issue_number=8104,
        retry_count=2,
        attempts=[
            {
                "at": "2026-04-13T10:00:00+00:00",
                "worker_outcome": "blocked",
                "exit_code": 1,
                "failure_reason": "pytest failed",
                "failing_verification": {
                    "command": "pytest -q tests/swarm/test_boss_loop.py",
                    "exit_code": 1,
                },
            }
        ],
    )

    summary = summarize_session_blocker(state)

    assert summary == (
        "Issue #8104 exhausted retries; last blocker was failing verification "
        "`pytest -q tests/swarm/test_boss_loop.py` (exit 1)."
    )


def test_summarize_session_blocker_reports_no_committed_changes() -> None:
    state = SessionState(
        session_id="issue-8105",
        issue_number=8105,
        retry_count=2,
        resume_hint="Worker exited cleanly with no deliverable",
        attempts=[
            {
                "at": "2026-04-13T10:30:00+00:00",
                "worker_outcome": "clean_exit_no_deliverable",
                "exit_code": 0,
                "failure_reason": "Worker exited cleanly with no deliverable",
                "changed_paths": [],
            }
        ],
    )

    summary = summarize_session_blocker(state)

    assert summary == (
        "Issue #8105 exhausted retries; last attempt made no committed file changes: "
        "Worker exited cleanly with no deliverable."
    )
