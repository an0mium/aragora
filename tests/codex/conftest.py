"""Shared fixtures for the aragora codex inspector tests.

Builds a synthetic Codex Desktop home (``state_5.sqlite`` + rollout JSONLs)
in a tmp dir so tests never touch the real ``~/.codex`` tree.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

# Thread schema mirrors the live Codex Desktop ``state_5.sqlite`` shape (the
# subset the inspector reads from). Times are seconds-since-epoch ints.
_CREATE_THREADS = """
CREATE TABLE threads (
    id TEXT PRIMARY KEY,
    rollout_path TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    source TEXT NOT NULL,
    model_provider TEXT NOT NULL,
    cwd TEXT NOT NULL,
    title TEXT NOT NULL,
    sandbox_policy TEXT NOT NULL,
    approval_mode TEXT NOT NULL,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    has_user_event INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    archived_at INTEGER,
    git_sha TEXT,
    git_branch TEXT,
    git_origin_url TEXT,
    cli_version TEXT NOT NULL DEFAULT '',
    first_user_message TEXT NOT NULL DEFAULT '',
    agent_nickname TEXT,
    agent_role TEXT,
    memory_mode TEXT NOT NULL DEFAULT 'enabled',
    model TEXT,
    reasoning_effort TEXT,
    agent_path TEXT,
    created_at_ms INTEGER,
    updated_at_ms INTEGER,
    thread_source TEXT,
    preview TEXT NOT NULL DEFAULT ''
);
"""


@dataclass(frozen=True)
class FakeCodexHome:
    home: Path
    recent_thread_id: str
    recent_rollout: Path
    archived_thread_id: str
    old_thread_id: str
    secret_titled_thread_id: str


def _write_rollout(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


@pytest.fixture
def fake_codex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeCodexHome]:
    home = tmp_path / "codex"
    home.mkdir()
    monkeypatch.setenv("ARAGORA_CODEX_HOME", str(home))

    db_path = home / "state_5.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_CREATE_THREADS)
        conn.commit()

        now = int(time.time())
        ten_minutes_ago = now - 600
        one_day_ago = now - 86400
        ten_days_ago = now - 10 * 86400

        recent_id = "019e3222-3e45-7920-90f4-203877f24690"
        recent_rollout = home / "sessions" / "2026" / "05" / "16" / f"rollout-{recent_id}.jsonl"
        _write_rollout(
            recent_rollout,
            events=[
                {
                    "timestamp": "2026-05-16T13:52:45.000Z",
                    "type": "session_meta",
                    "payload": {
                        "id": recent_id,
                        "cwd": "/Users/test/repo",
                        "model_provider": "openai",
                    },
                },
                {
                    "timestamp": "2026-05-16T13:53:00.000Z",
                    "type": "turn_start",
                    "payload": {"type": "turn_start", "turn_id": "t1"},
                },
                {
                    "timestamp": "2026-05-16T13:53:05.000Z",
                    "type": "agent_message",
                    "payload": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please review my code with sk-proj-FAKE-LEAK-12345 attached",
                            },
                        ],
                    },
                },
                {
                    "timestamp": "2026-05-16T13:53:10.000Z",
                    "type": "tool_call",
                    "payload": {"tool_name": "Read", "tool_call": {"name": "Read"}},
                },
                {
                    "timestamp": "2026-05-16T13:53:15.000Z",
                    "type": "tool_response",
                    "payload": {"tool_name": "Read"},
                },
                {
                    "timestamp": "2026-05-16T13:53:20.000Z",
                    "type": "agent_message",
                    "payload": {
                        "role": "user",
                        "content": "Authorization: Bearer ghp_FAKELEAK12345678901234",
                    },
                },
            ],
        )

        old_id = "019cccc0-0000-0000-0000-000000000001"
        old_rollout = home / "sessions" / "2026" / "03" / "01" / f"rollout-{old_id}.jsonl"
        _write_rollout(
            old_rollout,
            events=[
                {
                    "timestamp": "2026-03-01T00:00:00.000Z",
                    "type": "session_meta",
                    "payload": {"model_provider": "anthropic"},
                },
            ],
        )

        archived_id = "019cccc0-0000-0000-0000-000000000002"
        archived_rollout = home / "sessions" / "2026" / "04" / "01" / f"rollout-{archived_id}.jsonl"
        _write_rollout(
            archived_rollout,
            events=[
                {
                    "timestamp": "2026-04-01T00:00:00.000Z",
                    "type": "session_meta",
                    "payload": {"model_provider": "openai"},
                },
            ],
        )

        secret_titled_id = "019e3222-9999-7920-90f4-203877f24690"
        secret_rollout = (
            home
            / "sessions"
            / "sk-or-v1-abcdefghijklmnopqrstuvwxyz"
            / f"rollout-{secret_titled_id}.jsonl"
        )
        _write_rollout(
            secret_rollout,
            events=[
                {
                    "timestamp": "2026-05-16T14:00:00.000Z",
                    "type": "session_meta",
                    "payload": {"model_provider": "openai"},
                },
            ],
        )

        rows = [
            (
                recent_id,
                str(recent_rollout),
                ten_minutes_ago,
                ten_minutes_ago,
                "vscode",
                "openai",
                "/Users/test/repo",
                "Recent debugging thread",
                "workspace",
                "auto",
                1234,
                1,
                0,
                None,
                "abc1234",
                "main",
                "https://github.com/test/repo",
                "0.42.0",
                "Please look at this issue",
                None,
                None,
                "enabled",
                "gpt-5.4",
                "medium",
                None,
                ten_minutes_ago * 1000,
                ten_minutes_ago * 1000,
                "vscode",
                "Recent debugging thread preview",
            ),
            (
                secret_titled_id,
                str(secret_rollout),
                ten_minutes_ago,
                ten_minutes_ago,
                "vscode",
                "openai",
                "/Users/test/ghp_FAKELEAK1234567890ABCD/repo",
                "leaked sk-proj-FAKE-LEAK-XYZ thread",
                "workspace",
                "auto",
                500,
                1,
                0,
                None,
                "abc9999",
                "feature/sk-or-v1-abcdefghijklmnopqrstuvwxyz",
                None,
                "0.42.0",
                "User asked about Bearer ghp_FAKELEAK1234567890ABCD",
                None,
                None,
                "enabled",
                "claude-opus-4-7",
                None,
                None,
                ten_minutes_ago * 1000,
                ten_minutes_ago * 1000,
                "vscode",
                "secret titled preview",
            ),
            (
                archived_id,
                str(archived_rollout),
                one_day_ago,
                one_day_ago,
                "vscode",
                "openai",
                "/Users/test/archived",
                "Archived but recent thread",
                "workspace",
                "auto",
                42,
                1,
                1,
                one_day_ago,
                None,
                None,
                None,
                "0.42.0",
                "",
                None,
                None,
                "enabled",
                "gpt-5.4",
                None,
                None,
                one_day_ago * 1000,
                one_day_ago * 1000,
                "vscode",
                "",
            ),
            (
                old_id,
                str(old_rollout),
                ten_days_ago,
                ten_days_ago,
                "vscode",
                "anthropic",
                "/Users/test/repo",
                "Old thread out of window",
                "workspace",
                "auto",
                100,
                1,
                0,
                None,
                None,
                None,
                None,
                "0.40.0",
                "",
                None,
                None,
                "enabled",
                "claude-opus-4-6",
                None,
                None,
                ten_days_ago * 1000,
                ten_days_ago * 1000,
                "vscode",
                "",
            ),
        ]
        conn.executemany(
            """
            INSERT INTO threads (
                id, rollout_path, created_at, updated_at, source, model_provider,
                cwd, title, sandbox_policy, approval_mode, tokens_used,
                has_user_event, archived, archived_at, git_sha, git_branch,
                git_origin_url, cli_version, first_user_message, agent_nickname,
                agent_role, memory_mode, model, reasoning_effort, agent_path,
                created_at_ms, updated_at_ms, thread_source, preview
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    yield FakeCodexHome(
        home=home,
        recent_thread_id=recent_id,
        recent_rollout=recent_rollout,
        archived_thread_id=archived_id,
        old_thread_id=old_id,
        secret_titled_thread_id=secret_titled_id,
    )
