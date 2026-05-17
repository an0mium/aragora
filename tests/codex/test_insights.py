"""Tests for the codex activity-intelligence analysis layer."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aragora.codex import insights
from aragora.codex.desktop_inspector import SessionSummary, ThreadSummary
from aragora.codex.desktop_paths import resolve


def _make_thread(
    *,
    id: str,
    title: str = "t",
    model: str = "gpt-5.4",
    tokens: int = 100,
    branch: str | None = "main",
    archived: bool = False,
    rollout: Path,
    first_user_message: str = "please fix #6009",
) -> ThreadSummary:
    now = datetime.now(UTC)
    return ThreadSummary(
        id=id,
        title=title,
        cwd="/repo",
        model=model,
        rollout_path=rollout,
        created_at=now - timedelta(hours=1),
        updated_at=now,
        tokens_used=tokens,
        archived=archived,
        git_sha=None,
        git_branch=branch,
        source="vscode",
        first_user_message=first_user_message,
    )


def _make_summary(
    rollout: Path,
    *,
    tool_calls: dict[str, int] | None = None,
    events_scanned: int = 10,
    last_event_age_minutes: float | None = 1.0,
    last_event_type: str | None = "agent_message",
    model: str | None = "openai",
    first_user: str = "fix #5908 please",
    last_user: str = "thanks",
) -> SessionSummary:
    started = datetime.now(UTC) - timedelta(minutes=30)
    last = (
        datetime.now(UTC) - timedelta(minutes=last_event_age_minutes)
        if last_event_age_minutes is not None
        else None
    )
    counts = {"agent_message": 4, "tool_call": sum((tool_calls or {}).values())}
    if last_event_type:
        counts[last_event_type] = counts.get(last_event_type, 0) + 1
    return SessionSummary(
        rollout_path=rollout,
        events_scanned=events_scanned,
        truncated=False,
        event_type_counts=counts,
        tool_call_counts=tool_calls or {},
        first_user_message=first_user,
        last_user_message=last_user,
        model_provider=model,
        started_at=started,
        last_event_at=last,
    )


# -- summarize_patterns -------------------------------------------------------


def test_summarize_patterns_aggregates_threads(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    pattern, pairs = insights.summarize_patterns(since=timedelta(hours=4))
    assert pattern.thread_count >= 2  # fixture has 2 recent threads
    assert pattern.archived_excluded is True
    assert pattern.distinct_cwds >= 1
    # tool call distribution is non-empty (recent rollout has a Read call)
    assert pattern.tool_call_distribution
    assert all(isinstance(v, int) for v in pattern.tool_call_distribution.values())


def test_summarize_patterns_window_excludes_old(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    pattern, _ = insights.summarize_patterns(since=timedelta(minutes=1))
    # 10-minute-old fixtures are outside a 1-minute window.
    assert pattern.thread_count == 0


# -- detect_anomalies ---------------------------------------------------------


def test_detect_anomalies_runaway_tool_calls(tmp_path: Path) -> None:
    rollout = tmp_path / "r.jsonl"
    rollout.write_text("{}\n", encoding="utf-8")
    thread = _make_thread(id="a", rollout=rollout)
    summary = _make_summary(rollout, tool_calls={"Read": 300})
    anomalies = insights.detect_anomalies(
        [(thread, summary)],
        runaway_tool_calls=200,
    )
    kinds = {a.kind for a in anomalies}
    assert "runaway_tool_calls" in kinds
    assert any(a.severity == "high" for a in anomalies if a.kind == "runaway_tool_calls")


def test_detect_anomalies_token_cap_exceeded(tmp_path: Path) -> None:
    rollout = tmp_path / "r.jsonl"
    rollout.write_text("{}\n", encoding="utf-8")
    thread = _make_thread(id="b", rollout=rollout, tokens=999_999)
    summary = _make_summary(rollout, tool_calls={"Read": 1})
    anomalies = insights.detect_anomalies(
        [(thread, summary)],
        token_cap=100_000,
    )
    assert any(a.kind == "token_cap_exceeded" for a in anomalies)


def test_detect_anomalies_stuck_turn(tmp_path: Path) -> None:
    rollout = tmp_path / "r.jsonl"
    rollout.write_text("{}\n", encoding="utf-8")
    thread = _make_thread(id="c", rollout=rollout)
    summary = _make_summary(
        rollout,
        tool_calls={"Read": 1},
        last_event_age_minutes=20.0,
        last_event_type="turn_start",
    )
    anomalies = insights.detect_anomalies(
        [(thread, summary)],
        stuck_turn_minutes=5,
    )
    assert any(a.kind == "stuck_turn" for a in anomalies)


def test_detect_anomalies_returns_empty_when_clean(tmp_path: Path) -> None:
    rollout = tmp_path / "r.jsonl"
    rollout.write_text("{}\n", encoding="utf-8")
    thread = _make_thread(id="d", rollout=rollout, tokens=10)
    summary = _make_summary(rollout, tool_calls={"Read": 1})
    assert insights.detect_anomalies([(thread, summary)]) == []


def test_detect_anomalies_sorted_high_first(tmp_path: Path) -> None:
    rollout = tmp_path / "r.jsonl"
    rollout.write_text("{}\n", encoding="utf-8")
    high_thread = _make_thread(id="high", rollout=rollout)
    high_summary = _make_summary(rollout, tool_calls={"Read": 500})
    medium_thread = _make_thread(id="med", rollout=rollout, tokens=999_999)
    medium_summary = _make_summary(rollout, tool_calls={"Read": 5})
    anomalies = insights.detect_anomalies(
        [(medium_thread, medium_summary), (high_thread, high_summary)]
    )
    assert anomalies[0].severity == "high"


# -- crossref_work_board ------------------------------------------------------


def test_crossref_extracts_pr_and_issue_refs(tmp_path: Path) -> None:
    rollout = tmp_path / "r.jsonl"
    rollout.write_text("{}\n", encoding="utf-8")
    thread = _make_thread(id="e", rollout=rollout, title="Working on #7240")
    summary = _make_summary(rollout, first_user="please fix #5908 and review #6009")
    crossref = insights.crossref_work_board([(thread, summary)])
    assert len(crossref) == 1
    refs = set(crossref[0].pr_references)
    assert "#5908" in refs
    assert "#6009" in refs
    assert "#7240" in refs


def test_crossref_redacts_secret_rollout_path(tmp_path: Path) -> None:
    rollout = tmp_path / "sk-or-v1-abcdefghijklmnopqrstuvwxyz" / "r.jsonl"
    rollout.parent.mkdir()
    rollout.write_text("{}\n", encoding="utf-8")
    thread = _make_thread(id="secret-path", rollout=rollout, title="Working on #7240")
    summary = _make_summary(rollout)

    crossref = insights.crossref_work_board([(thread, summary)])

    assert "sk-or-v1-abcdefghijklmnopqrstuvwxyz" not in crossref[0].rollout_path


def test_crossref_handles_no_refs(tmp_path: Path) -> None:
    rollout = tmp_path / "r.jsonl"
    rollout.write_text("{}\n", encoding="utf-8")
    thread = _make_thread(
        id="f",
        rollout=rollout,
        title="just a chat",
        first_user_message="hello world",
    )
    summary = _make_summary(rollout, first_user="hello", last_user="bye")
    crossref = insights.crossref_work_board([(thread, summary)])
    assert crossref[0].pr_references == ()
    assert crossref[0].issue_references == ()


# -- build_digest -------------------------------------------------------------


def test_build_digest_has_required_fields(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    digest = insights.build_digest(since=timedelta(hours=4))
    assert digest.schema_version == insights.INSPECTOR_SCHEMA_VERSION
    assert digest.thread_count >= 2
    assert digest.sha256
    assert len(digest.sha256) == 64
    payload = digest.to_dict()
    assert "patterns" in payload
    assert "anomalies" in payload
    assert "crossref" in payload
    assert "inspector_summaries" in payload


def test_build_digest_redacts_secret_rollout_paths(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    digest = insights.build_digest(since=timedelta(hours=4), include_archived=True)
    payload = json.dumps(digest.to_dict(), sort_keys=True)

    assert "sk-or-v1-abcdefghijklmnopqrstuvwxyz" not in payload
    assert "ghp_FAKELEAK" not in payload
    assert "sk-proj-FAKE-LEAK" not in payload


def test_build_digest_is_deterministic_for_same_window(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    # Two digests built in quick succession differ in window_until timestamps
    # but the patterns section should match (same threads, same counts).
    first = insights.build_digest(since=timedelta(hours=4))
    second = insights.build_digest(since=timedelta(hours=4))
    assert first.patterns.thread_count == second.patterns.thread_count
    assert first.patterns.tool_call_distribution == second.patterns.tool_call_distribution


def test_digest_hmac_is_set_when_key_present(
    fake_codex_home, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    import base64

    monkeypatch.setenv(
        "ARAGORA_CONTEXT_SIGNING_KEY",
        base64.b64encode(b"test-signing-key-bytes").decode("ascii"),
    )
    digest = insights.build_digest(since=timedelta(hours=4))
    assert digest.hmac_sha256 is not None
    assert digest.signed_at is not None


def test_digest_hmac_is_none_when_key_absent(
    fake_codex_home, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ARAGORA_CONTEXT_SIGNING_KEY", raising=False)
    digest = insights.build_digest(since=timedelta(hours=4))
    assert digest.hmac_sha256 is None
    assert digest.signed_at is None


# -- persist_digest -----------------------------------------------------------


def test_persist_digest_writes_json(fake_codex_home, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    digest = insights.build_digest(since=timedelta(hours=4))
    target = insights.persist_digest(digest, root=tmp_path)
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["sha256"] == digest.sha256
    assert payload["schema_version"] == insights.INSPECTOR_SCHEMA_VERSION


# -- ingest_digest_into_km ----------------------------------------------------


def test_ingest_digest_into_km_handles_missing_aragora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest_path = tmp_path / "d.json"
    digest_path.write_text("{}", encoding="utf-8")
    # Force PATH so 'aragora' resolves to nothing.
    monkeypatch.setenv("PATH", "/usr/nowhere")
    ok, detail = insights.ingest_digest_into_km(digest_path, timeout_seconds=2.0)
    assert ok is False
    assert detail


def test_ingest_digest_into_km_returns_false_for_missing_file(tmp_path: Path) -> None:
    ok, detail = insights.ingest_digest_into_km(tmp_path / "missing.json")
    assert ok is False
    assert "not found" in detail
