"""Tests for read-only Codex session/automation ingest (codex_source).

Synthetic temp Codex homes only -- no dependency on a real ~/.codex. Verifies
parsing, recency filtering, thread-name join, ledger extraction, and the
defensive contract that malformed lines never raise.
"""

from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from pathlib import Path

from aragora.swarm.agent_bridge import codex_source as cs

NOW = datetime(2026, 6, 15, 21, 0, 0, tzinfo=UTC)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _rollout(home: Path, *, day: str, ts: str, session_id: str, records: list[dict]) -> Path:
    year, month, dd = day.split("-")
    path = home / "sessions" / year / month / dd / f"rollout-{ts}-{session_id}.jsonl"
    _write_jsonl(path, records)
    return path


def test_summarize_rollout_extracts_meta_and_last_message(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    path = _rollout(
        home,
        day="2026-06-15",
        ts="2026-06-15T15-40-37",
        session_id="019ecd03-aaaa",
        records=[
            {
                "timestamp": "2026-06-15T20:40:46.911Z",
                "type": "session_meta",
                "payload": {
                    "id": "019ecd03-aaaa",
                    "cwd": "/work/.codex/worktrees/d48e/aragora",
                    "originator": "Codex Desktop",
                    "model_provider": "openai",
                    "timestamp": "2026-06-15T20:40:37.152Z",
                },
            },
            {
                "timestamp": "2026-06-15T20:41:00Z",
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "first thing"},
            },
            {
                "timestamp": "2026-06-15T20:42:00Z",
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "  last thing  "},
            },
        ],
    )
    summary = cs.summarize_rollout(path)
    assert summary is not None
    assert summary.session_id == "019ecd03-aaaa"
    assert summary.cwd == "/work/.codex/worktrees/d48e/aragora"
    assert summary.originator == "Codex Desktop"
    assert summary.model_provider == "openai"
    assert summary.agent_message_count == 2
    assert summary.last_agent_message == "last thing"
    assert summary.updated_at == "2026-06-15T20:42:00Z"


def test_recent_sessions_filters_by_window_and_joins_thread_name(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    _rollout(
        home,
        day="2026-06-15",
        ts="2026-06-15T20-40-00",
        session_id="recent-1",
        records=[
            {
                "timestamp": "2026-06-15T20:50:00Z",
                "type": "session_meta",
                "payload": {"id": "recent-1", "cwd": "/repo", "originator": "Codex Desktop"},
            }
        ],
    )
    _rollout(
        home,
        day="2026-06-13",
        ts="2026-06-13T01-00-00",
        session_id="old-1",
        records=[
            {
                "timestamp": "2026-06-13T01:00:00Z",
                "type": "session_meta",
                "payload": {"id": "old-1", "cwd": "/repo"},
            }
        ],
    )
    _write_jsonl(
        home / "session_index.jsonl",
        [{"id": "recent-1", "thread_name": "Repo hygiene", "updated_at": "2026-06-15T20:50:00Z"}],
    )

    out = cs.recent_sessions(home, hours=6.0, now=NOW)
    ids = [s.session_id for s in out]
    assert ids == ["recent-1"]  # old-1 is outside the 6h window
    assert out[0].thread_name == "Repo hygiene"


def test_recent_sessions_no_window_returns_all(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    _rollout(
        home,
        day="2026-06-13",
        ts="2026-06-13T01-00-00",
        session_id="old-1",
        records=[
            {
                "timestamp": "2026-06-13T01:00:00Z",
                "type": "session_meta",
                "payload": {"id": "old-1", "cwd": "/repo"},
            }
        ],
    )
    out = cs.recent_sessions(home, hours=None, now=NOW)
    assert [s.session_id for s in out] == ["old-1"]


def test_malformed_lines_are_skipped_not_raised(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    path = home / "sessions" / "2026" / "06" / "15" / "rollout-2026-06-15T20-40-00-bad.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "not json at all\n"
        + json.dumps(
            {
                "timestamp": "2026-06-15T20:50:00Z",
                "type": "session_meta",
                "payload": {"id": "bad", "cwd": "/repo"},
            }
        )
        + "\n{ truncated half-written line",
        encoding="utf-8",
    )
    summary = cs.summarize_rollout(path)
    assert summary is not None
    assert summary.session_id == "bad"


def test_rollout_without_meta_falls_back_to_filename_id(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    path = _rollout(
        home,
        day="2026-06-15",
        ts="2026-06-15T20-40-00",
        session_id="from-filename",
        records=[{"timestamp": "2026-06-15T20:50:00Z", "type": "event_msg", "payload": {}}],
    )
    summary = cs.summarize_rollout(path)
    assert summary is not None
    # No session_meta -> id falls back to the filename stem (timestamp+uuid),
    # which still uniquely identifies the rollout.
    assert summary.session_id == "2026-06-15T20-40-00-from-filename"


def test_read_ledgers_extracts_structured_fields(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    _write_jsonl(
        home / "automations" / "overnight-conductor" / "ledger.jsonl",
        [
            {
                "action": {
                    "kind": "merge_ready_prompt",
                    "reason": "first mergeable non-draft PR",
                    "target": {
                        "pr": 8436,
                        "head": "bd238556853d8d7afa8c19cea1e599d0730d5984",
                        "branch": "dependabot/x",
                        "url": "https://github.com/synaptent/aragora/pull/8436",
                    },
                },
                "forbidden_actions": ["merge", "mark_ready"],
                "git": {"head": "abc", "origin_main": "abc", "status_ok": True},
                "summary": {"open_pr_count": 10, "runner_blockers": []},
                "generated_at": "2026-06-15T20:14:49+00:00",
                "repo_root": "/tmp/cycle/aragora",
            }
        ],
    )
    out = cs.read_ledgers(home, hours=24.0, now=NOW)
    assert len(out) == 1
    entry = out[0]
    assert entry.automation == "overnight-conductor"
    assert entry.pr == 8436
    assert entry.head.startswith("bd23855")
    assert entry.forbidden_actions == ["merge", "mark_ready"]
    assert entry.open_pr_count == 10
    assert entry.git_status_ok is True


def test_read_ledgers_window_filters_old_records(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    _write_jsonl(
        home / "automations" / "stale" / "ledger.jsonl",
        [{"action": {"kind": "x"}, "generated_at": "2026-06-10T00:00:00Z"}],
    )
    assert cs.read_ledgers(home, hours=12.0, now=NOW) == []


def test_parse_iso_handles_z_naive_and_garbage() -> None:
    assert cs._parse_iso("2026-06-15T20:40:46.911Z") is not None
    naive = cs._parse_iso("2026-06-15T20:40:46")
    assert naive is not None and naive.tzinfo is UTC
    assert cs._parse_iso("not a date") is None
    assert cs._parse_iso(None) is None


def test_missing_codex_home_returns_empty(tmp_path: Path) -> None:
    home = tmp_path / "does-not-exist"
    assert cs.recent_sessions(home, now=NOW) == []
    assert cs.read_ledgers(home, now=NOW) == []
    assert cs.read_session_index(home) == {}
