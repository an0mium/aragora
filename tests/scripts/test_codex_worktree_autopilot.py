"""Unit tests for scripts/codex_worktree_autopilot.py."""

from __future__ import annotations

import sys
from datetime import timezone
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


def test_prune_stale_state_removes_inactive_paths(tmp_path):
    import codex_worktree_autopilot as mod

    a_path = tmp_path / "a"
    b_path = tmp_path / "b"
    a_path.mkdir()
    b_path.mkdir()

    state = {
        "sessions": [
            {"session_id": "a", "path": str(a_path)},
            {"session_id": "b", "path": str(b_path)},
        ]
    }
    active = {str(b_path)}
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


def test_cleanup_parser_defaults_to_delete_branches():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["cleanup"])
    assert args.delete_branches is True


def test_ensure_parser_defaults_to_merge_strategy():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["ensure"])
    assert args.strategy == "merge"


def test_reconcile_parser_defaults_to_rebase_strategy():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["reconcile"])
    assert args.strategy == "rebase"


def test_cleanup_parser_allows_no_delete_branches_toggle():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["cleanup", "--no-delete-branches"])
    assert args.delete_branches is False


def test_maintain_parser_allows_no_delete_branches_toggle():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["maintain", "--no-delete-branches"])
    assert args.delete_branches is False


def test_maintain_parser_defaults_to_merge_strategy():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["maintain"])
    assert args.strategy == "merge"


def test_parse_ts_normalizes_naive_timestamp_to_utc():
    import codex_worktree_autopilot as mod

    parsed = mod._parse_ts("2026-02-24T12:00:00")
    assert parsed is not None
    assert parsed.tzinfo == timezone.utc
