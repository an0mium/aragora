"""Unit tests for scripts/codex_worktree_autopilot.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def test_parse_worktree_porcelain_includes_branch_and_detached():
    import codex_worktree_autopilot as mod

    porcelain = (
        "worktree /repo\n"
        "HEAD abc\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /repo/.worktrees/codex-auto/s1\n"
        "HEAD def\n"
        "branch refs/heads/codex/s1\n"
        "\n"
        "worktree /repo/.worktrees/codex-auto/s2\n"
        "HEAD 123\n"
        "detached\n"
        "\n"
    )
    entries = mod._parse_worktree_porcelain(porcelain)
    assert len(entries) == 3
    assert entries[0].branch == "main"
    assert entries[1].branch == "codex/s1"
    assert entries[2].detached is True
    assert entries[2].branch is None


def test_prune_stale_state_removes_inactive_paths():
    import codex_worktree_autopilot as mod

    state = {
        "sessions": [
            {"session_id": "a", "path": "/repo/.worktrees/codex-auto/a"},
            {"session_id": "b", "path": "/repo/.worktrees/codex-auto/b"},
        ]
    }
    active = {"/repo/.worktrees/codex-auto/b"}
    pruned, removed = mod._prune_stale_state(state, active)
    assert removed == 1
    assert len(pruned["sessions"]) == 1
    assert pruned["sessions"][0]["session_id"] == "b"


def test_choose_reusable_session_prefers_latest_last_seen():
    import codex_worktree_autopilot as mod

    state = {
        "sessions": [
            {
                "agent": "codex",
                "session_id": "old",
                "path": "/repo/.worktrees/codex-auto/old",
                "last_seen_at": "2026-02-24T00:00:00+00:00",
            },
            {
                "agent": "codex",
                "session_id": "new",
                "path": "/repo/.worktrees/codex-auto/new",
                "last_seen_at": "2026-02-24T01:00:00+00:00",
            },
        ]
    }
    chosen = mod._choose_reusable_session(
        state,
        agent="codex",
        session_id=None,
        active_paths={
            "/repo/.worktrees/codex-auto/old",
            "/repo/.worktrees/codex-auto/new",
        },
    )
    assert chosen is not None
    assert chosen["session_id"] == "new"


def test_choose_reusable_session_honors_session_id_filter():
    import codex_worktree_autopilot as mod

    state = {
        "sessions": [
            {"agent": "codex", "session_id": "a", "path": "/repo/.worktrees/codex-auto/a"},
            {"agent": "codex", "session_id": "b", "path": "/repo/.worktrees/codex-auto/b"},
        ]
    }
    chosen = mod._choose_reusable_session(
        state,
        agent="codex",
        session_id="a",
        active_paths={
            "/repo/.worktrees/codex-auto/a",
            "/repo/.worktrees/codex-auto/b",
        },
    )
    assert chosen is not None
    assert chosen["session_id"] == "a"
