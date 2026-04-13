from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aragora.swarm.session_state import SessionState, SessionStateStore


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
        metadata={"receipt_id": "rcpt-123", "phase": "skeleton"},
        created_at=created_at,
        updated_at=updated_at,
    )

    restored = SessionState.from_dict(state.to_dict())

    assert restored.to_dict() == state.to_dict()


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
