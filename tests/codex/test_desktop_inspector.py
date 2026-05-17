"""Tests for the codex desktop inspector data layer."""

from __future__ import annotations

import inspect as _inspect
import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from aragora.codex import desktop_inspector as inspector
from aragora.codex.desktop_paths import resolve
from aragora.codex.duration import parse_duration
from aragora.codex.jsonl_stream import iter_jsonl
from aragora.codex.sqlite_ro import sqlite_ro


# -- duration -----------------------------------------------------------------


def test_parse_duration_supports_all_units() -> None:
    assert parse_duration("90s") == timedelta(seconds=90)
    assert parse_duration("30m") == timedelta(minutes=30)
    assert parse_duration("4h") == timedelta(hours=4)
    assert parse_duration("1d") == timedelta(days=1)


def test_parse_duration_strips_whitespace() -> None:
    assert parse_duration("  4h  ") == timedelta(hours=4)


def test_parse_duration_rejects_garbage() -> None:
    for value in ("4", "h", "4hours", "-1h", "", "4.5h"):
        with pytest.raises(ValueError):
            parse_duration(value)


# -- paths --------------------------------------------------------------------


def test_resolve_honors_env_var(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    paths = resolve()
    assert paths.home == fake_codex_home.home
    assert paths.sqlite_path == fake_codex_home.home / "state_5.sqlite"
    assert paths.sessions_root == fake_codex_home.home / "sessions"
    assert paths.global_state_path.name == ".codex-global-state.json"


def test_resolve_expands_explicit_tilde_home() -> None:
    paths = resolve("~/aragora-codex-test-home")
    assert paths.home == Path.home() / "aragora-codex-test-home"


# -- jsonl stream -------------------------------------------------------------


def test_iter_jsonl_is_streaming() -> None:
    assert _inspect.isgeneratorfunction(iter_jsonl)


def test_iter_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    sample = tmp_path / "sample.jsonl"
    sample.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    assert list(iter_jsonl(sample)) == [{"a": 1}, {"b": 2}]


def test_iter_jsonl_raises_with_line_context(tmp_path: Path) -> None:
    sample = tmp_path / "bad.jsonl"
    sample.write_text('{"a":1}\nnot json\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError) as excinfo:
        list(iter_jsonl(sample))
    assert "bad.jsonl:2" in excinfo.value.msg


def test_iter_jsonl_non_strict_stops_at_partial_line(tmp_path: Path) -> None:
    sample = tmp_path / "partial.jsonl"
    sample.write_text('{"a":1}\n{"partial":', encoding="utf-8")
    assert list(iter_jsonl(sample, strict=False)) == [{"a": 1}]


# -- sqlite read-only ---------------------------------------------------------


def test_sqlite_ro_rejects_writes(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    with sqlite_ro(fake_codex_home.home / "state_5.sqlite") as conn:
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            conn.execute(
                "INSERT INTO threads (id, rollout_path, created_at, updated_at, source,"
                " model_provider, cwd, title, sandbox_policy, approval_mode) VALUES"
                " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("x", "x", 0, 0, "s", "p", "c", "t", "sp", "ap"),
            )
        assert "readonly" in str(excinfo.value).lower()


def test_sqlite_ro_handles_uri_reserved_path_chars(tmp_path: Path) -> None:
    db_path = tmp_path / "codex ?# state.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sample (value) VALUES ('ok')")

    with sqlite_ro(db_path) as conn:
        row = conn.execute("SELECT value FROM sample").fetchone()

    assert row["value"] == "ok"


# -- inspector listing --------------------------------------------------------


def test_list_active_threads_default_window(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    threads = inspector.list_active_threads(since=timedelta(hours=4))
    ids = [t.id for t in threads]
    assert fake_codex_home.recent_thread_id in ids
    assert fake_codex_home.secret_titled_thread_id in ids
    assert fake_codex_home.archived_thread_id not in ids  # archived excluded by default
    assert fake_codex_home.old_thread_id not in ids


def test_list_active_threads_window_excludes_older(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    threads = inspector.list_active_threads(since=timedelta(minutes=5))
    ids = [t.id for t in threads]
    # 10-minute-old threads should be outside a 5-minute window.
    assert fake_codex_home.recent_thread_id not in ids


def test_list_active_threads_include_archived(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    threads = inspector.list_active_threads(since=timedelta(days=2), include_archived=True)
    ids = [t.id for t in threads]
    assert fake_codex_home.archived_thread_id in ids


def test_list_active_threads_redacts_secrets_in_title_and_message(
    fake_codex_home,
) -> None:  # type: ignore[no-untyped-def]
    threads = inspector.list_active_threads(since=timedelta(hours=4))
    secret_thread = next(t for t in threads if t.id == fake_codex_home.secret_titled_thread_id)
    assert "sk-proj-FAKE-LEAK-XYZ" not in secret_thread.title
    assert "[REDACTED]" in secret_thread.title
    assert "ghp_FAKELEAK" not in secret_thread.first_user_message


def test_list_active_threads_redacts_printable_metadata_fields(
    fake_codex_home,
) -> None:  # type: ignore[no-untyped-def]
    threads = inspector.list_active_threads(since=timedelta(hours=4))
    secret_thread = next(t for t in threads if t.id == fake_codex_home.secret_titled_thread_id)

    serialized = json.dumps(secret_thread.to_dict(), default=str)
    assert "ghp_FAKELEAK1234567890ABCD" not in serialized
    assert "sk-or-v1-abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "[REDACTED]" in serialized


def test_redact_display_covers_github_fine_grained_tokens() -> None:
    token = "github_pat_11ABCDEFG_fakeFineGrainedToken_abcdefghijklmnopqrstuvwxyz"

    redacted = inspector.redact_display(f"branch-{token}-suffix")

    assert redacted is not None
    assert token not in redacted
    assert "[REDACTED]" in redacted


# -- inspector summarize ------------------------------------------------------


def test_summarize_session_counts_events_and_tools(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    summary = inspector.summarize_session(fake_codex_home.recent_rollout)
    assert summary.event_type_counts.get("agent_message") == 2
    assert summary.event_type_counts.get("tool_call") == 1
    assert summary.event_type_counts.get("session_meta") == 1
    assert summary.tool_call_counts.get("Read") == 1
    assert summary.model_provider == "openai"
    assert summary.first_user_message
    assert summary.last_user_message


def test_summarize_session_redacts_user_messages(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    summary = inspector.summarize_session(fake_codex_home.recent_rollout)
    combined = summary.first_user_message + " " + summary.last_user_message
    assert "sk-proj-FAKE-LEAK-12345" not in combined
    assert "ghp_FAKELEAK12345678901234" not in combined
    assert "[REDACTED]" in combined


def test_summarize_session_truncates_to_max_events(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    # max_events of 2 should mark truncated and stop reporting later events.
    summary = inspector.summarize_session(fake_codex_home.recent_rollout, max_events=2)
    assert summary.truncated is True
    assert summary.events_scanned == 2


def test_summarize_session_tolerates_partial_trailing_line(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    with fake_codex_home.recent_rollout.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "agent_message", "payload":')
    summary = inspector.summarize_session(fake_codex_home.recent_rollout)
    assert summary.event_type_counts.get("agent_message") == 2


# -- inspector full event stream ---------------------------------------------


def test_iter_session_events_redacts_by_default(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    events = list(inspector.iter_session_events(fake_codex_home.recent_rollout))
    serialized = json.dumps(events, default=str)
    assert "sk-proj-FAKE-LEAK-12345" not in serialized
    assert "ghp_FAKELEAK12345678901234" not in serialized
    assert "Bearer ghp_FAKELEAK12345678901234" not in serialized


def test_iter_session_events_redacts_nested_lists(tmp_path: Path) -> None:
    rollout = tmp_path / "nested-secret-rollout.jsonl"
    rollout.write_text(
        json.dumps(
            {
                "type": "agent_message",
                "payload": {
                    "content": [[{"text": "nested sk-proj-FAKE-NESTED-SECRET"}]],
                    "metadata": ["safe", ["Bearer ghp_FAKELEAK12345678901234"]],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events = list(inspector.iter_session_events(rollout))
    serialized = json.dumps(events, default=str)

    assert "sk-proj-FAKE-NESTED-SECRET" not in serialized
    assert "ghp_FAKELEAK12345678901234" not in serialized
    assert serialized.count("[REDACTED]") >= 2


def test_iter_session_events_tolerates_partial_trailing_line(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    with fake_codex_home.recent_rollout.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "agent_message", "payload":')
    events = list(inspector.iter_session_events(fake_codex_home.recent_rollout))
    assert len(events) == 6


def test_iter_session_events_from_offset_uses_raw_offsets(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    rollout = fake_codex_home.recent_rollout
    first_line_len = len(rollout.read_text(encoding="utf-8").splitlines()[0]) + 1
    events = list(inspector.iter_session_events_from_offset(rollout, offset=first_line_len))
    assert events
    assert events[0][0]["type"] == "turn_start"
    assert events[-1][1] == rollout.stat().st_size


def test_iter_session_events_from_offset_skips_complete_bad_line(tmp_path: Path) -> None:
    rollout = tmp_path / "bad-midstream.jsonl"
    first = '{"type": "turn_start"}\n'
    rollout.write_text(first + "not json\n" + '{"type": "agent_message"}\n', encoding="utf-8")

    events = list(inspector.iter_session_events_from_offset(rollout, offset=len(first)))

    assert events[0][0] == {}
    assert events[1][0]["type"] == "agent_message"
    assert events[-1][1] == rollout.stat().st_size


def test_iter_session_events_from_offset_advances_non_dict_lines(tmp_path: Path) -> None:
    rollout = tmp_path / "non-dict.jsonl"
    rollout.write_text('["not", "an", "event"]\n{"type": "agent_message"}\n', encoding="utf-8")

    events = list(inspector.iter_session_events_from_offset(rollout, offset=0))

    assert events[0][0] == {}
    assert events[1][0]["type"] == "agent_message"
    assert events[-1][1] == rollout.stat().st_size


def test_iter_session_events_opt_out_returns_raw(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    events = list(inspector.iter_session_events(fake_codex_home.recent_rollout, redact=False))
    serialized = json.dumps(events, default=str)
    # Opt-out is honored — secrets pass through. This proves redaction is *doing*
    # something and is not a no-op.
    assert "sk-proj-FAKE-LEAK-12345" in serialized


# -- redacted briefing / prompt router ---------------------------------------


def _append_rollout_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def test_build_session_brief_redacts_raw_transcript_and_extracts_safe_tokens(
    fake_codex_home,
) -> None:  # type: ignore[no-untyped-def]
    _append_rollout_event(
        fake_codex_home.recent_rollout,
        {
            "timestamp": "2026-05-16T13:54:00.000Z",
            "type": "agent_message",
            "payload": {
                "role": "user",
                "content": (
                    "Please settle PR #7283 and inspect scripts/apply_operator_decisions.py "
                    "with sk-proj-FAKE-BRIEF-SECRET"
                ),
            },
        },
    )
    thread = inspector.find_thread(fake_codex_home.recent_thread_id)
    assert thread is not None

    brief = inspector.build_session_brief(
        thread,
        include_last_turns=0,
        repo_context={"open_pr_count": 14, "active_lanes": []},
    )
    payload = brief.to_dict()
    serialized = json.dumps(payload)

    assert payload["pr_mentions"] == [7283]
    assert "scripts/apply_operator_decisions.py" in payload["files_mentioned"]
    assert payload["router"]["category"] == "settle"
    assert "exact-head" in payload["router"]["recommended_next_prompt"]
    assert "Please settle PR" not in serialized
    assert "sk-proj-FAKE-BRIEF-SECRET" not in serialized


def test_build_session_brief_last_turns_are_safe_summaries(
    fake_codex_home,
) -> None:  # type: ignore[no-untyped-def]
    _append_rollout_event(
        fake_codex_home.recent_rollout,
        {
            "timestamp": "2026-05-16T13:54:30.000Z",
            "type": "agent_message",
            "payload": {
                "role": "assistant",
                "content": (
                    "I posted a review comment on #7285 and read "
                    "aragora/codex/desktop_inspector.py with ghp_FAKELEAK12345678901234"
                ),
            },
        },
    )
    thread = inspector.find_thread(fake_codex_home.recent_thread_id)
    assert thread is not None

    brief = inspector.build_session_brief(thread, include_last_turns=2)
    payload = brief.to_dict()
    serialized = json.dumps(payload)

    assert payload["recent_turns"]
    assert "assistant" in {turn["role"] for turn in payload["recent_turns"]}
    assert "review" in serialized
    assert "I posted a review comment" not in serialized
    assert "ghp_FAKELEAK12345678901234" not in serialized


def test_build_session_brief_redacts_extracted_file_and_branch_tokens(
    fake_codex_home,
) -> None:  # type: ignore[no-untyped-def]
    _append_rollout_event(
        fake_codex_home.recent_rollout,
        {
            "timestamp": "2026-05-16T13:54:45.000Z",
            "type": "agent_message",
            "payload": {
                "role": "user",
                "content": (
                    "Repair docs/sk-proj-FAKE-BRIEF-SECRET.md on "
                    "codex/sk-or-v1-abcdefghijklmnopqrstuvwxyz"
                ),
            },
        },
    )
    thread = inspector.find_thread(fake_codex_home.recent_thread_id)
    assert thread is not None

    brief = inspector.build_session_brief(thread, include_last_turns=2)
    payload = brief.to_dict()
    serialized = json.dumps(payload)

    assert "sk-proj-FAKE-BRIEF-SECRET" not in serialized
    assert "sk-or-v1-abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "[REDACTED]" in serialized


def test_build_session_brief_routes_ambiguous_session_to_paste_needed(
    fake_codex_home,
) -> None:  # type: ignore[no-untyped-def]
    thread = inspector.find_thread(fake_codex_home.secret_titled_thread_id)
    assert thread is not None

    brief = inspector.build_session_brief(thread, include_last_turns=2)

    assert brief.router.category == "paste-needed"
    assert "Paste the last 2-4 turns" in brief.router.recommended_next_prompt


def test_router_prefers_pause_for_broad_build_when_queue_pressure_is_high(
    fake_codex_home,
) -> None:  # type: ignore[no-untyped-def]
    _append_rollout_event(
        fake_codex_home.recent_rollout,
        {
            "timestamp": "2026-05-16T13:55:00.000Z",
            "type": "agent_message",
            "payload": {
                "role": "user",
                "content": "Please build a new broad feature and open another PR.",
            },
        },
    )
    thread = inspector.find_thread(fake_codex_home.recent_thread_id)
    assert thread is not None

    brief = inspector.build_session_brief(
        thread,
        include_last_turns=0,
        repo_context={"open_pr_count": 20, "active_lanes": []},
    )

    assert brief.router.category == "pause"
    assert "Stop new implementation" in brief.router.recommended_next_prompt


def test_build_session_brief_marks_assistant_final_as_prompt_needed(
    fake_codex_home,
) -> None:  # type: ignore[no-untyped-def]
    _append_rollout_event(
        fake_codex_home.recent_rollout,
        {
            "timestamp": "2026-05-16T13:55:15.000Z",
            "type": "agent_message",
            "payload": {
                "role": "assistant",
                "content": "I finished reviewing #7286 and am waiting for next instruction.",
            },
        },
    )
    thread = inspector.find_thread(fake_codex_home.recent_thread_id)
    assert thread is not None

    brief = inspector.build_session_brief(
        thread,
        include_last_turns=0,
        repo_context={"open_pr_count": 12, "active_lane_records": []},
    )

    assert brief.prompt_needed is True
    assert brief.prompt_needed_reason == "assistant_final_recent"


def test_build_session_brief_marks_matching_active_lane_not_prompt_needed(
    fake_codex_home,
) -> None:  # type: ignore[no-untyped-def]
    _append_rollout_event(
        fake_codex_home.recent_rollout,
        {
            "timestamp": "2026-05-16T13:55:30.000Z",
            "type": "agent_message",
            "payload": {
                "role": "assistant",
                "content": "Continuing the active lane for #7245.",
            },
        },
    )
    thread = inspector.find_thread(fake_codex_home.recent_thread_id)
    assert thread is not None

    brief = inspector.build_session_brief(
        thread,
        include_last_turns=0,
        repo_context={
            "open_pr_count": 12,
            "active_lane_records": [
                {
                    "lane_id": "Q02-repair-7245-conflict",
                    "owner_session": "codex-q02",
                    "status": "active",
                    "branch": "main",
                    "pr_number": 7245,
                }
            ],
        },
    )
    payload = brief.to_dict()

    assert payload["prompt_needed"] is False
    assert payload["prompt_needed_reason"] == "active_lane_owned"
    assert payload["active_lane"]["lane_id"] == "Q02-repair-7245-conflict"
    assert payload["conflict_risk"] == "active-lane-overlap"


# -- find_thread --------------------------------------------------------------


def test_find_thread_accepts_prefix(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    prefix = fake_codex_home.recent_thread_id[:13]
    thread = inspector.find_thread(prefix)
    assert thread is not None
    assert thread.id == fake_codex_home.recent_thread_id


def test_find_thread_rejects_ambiguous_prefix(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    assert inspector.find_thread(fake_codex_home.recent_thread_id[:8]) is None


def test_find_thread_treats_like_wildcards_as_literal(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    wildcard_prefix = (
        fake_codex_home.recent_thread_id[:3] + "_" + fake_codex_home.recent_thread_id[4:8]
    )
    assert inspector.find_thread(wildcard_prefix) is None


def test_find_thread_rejects_short_prefix(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    assert inspector.find_thread("abc") is None


def test_find_thread_returns_none_on_miss(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    assert inspector.find_thread("ffffffff-ffff-ffff-ffff-ffffffffffff") is None


# -- safety: no network imports in this package ------------------------------


def test_codex_package_has_no_network_or_provider_imports() -> None:
    package_root = Path(inspector.__file__).resolve().parent
    forbidden = (
        "import httpx",
        "import requests",
        "import openai",
        "import anthropic",
        "from openai",
        "from anthropic",
        "from urllib.request",
    )
    for source in package_root.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in text, f"{source.name} should not contain {term!r}"


# -- safety: read-only contract on home directory ----------------------------


def test_inspector_does_not_modify_home(fake_codex_home) -> None:  # type: ignore[no-untyped-def]
    pre_mtimes = {p: p.stat().st_mtime_ns for p in fake_codex_home.home.rglob("*") if p.is_file()}
    inspector.list_active_threads(since=timedelta(hours=4))
    inspector.summarize_session(fake_codex_home.recent_rollout)
    list(inspector.iter_session_events(fake_codex_home.recent_rollout))
    post_mtimes = {p: p.stat().st_mtime_ns for p in fake_codex_home.home.rglob("*") if p.is_file()}
    assert pre_mtimes == post_mtimes
