"""Tests for aragora.swarm.worker_process module."""

import pytest

from aragora.swarm.worker_process import (
    LaunchConfig,
    WorkerProcess,
    is_ignored_changed_path,
)


# ---------------------------------------------------------------------------
# is_ignored_changed_path — session artifact names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "artifact",
    [
        ".codex_session_meta.json",
        ".codex_session.log",
        ".codex_session_active",
        ".swarm_worker_stdout.log",
        ".swarm_worker_stderr.log",
    ],
)
def test_session_artifact_names_are_ignored(artifact):
    """Each session artifact filename returns True."""
    assert is_ignored_changed_path(artifact) is True


# ---------------------------------------------------------------------------
# is_ignored_changed_path — node_modules paths
# ---------------------------------------------------------------------------


def test_node_modules_top_level_is_ignored():
    """node_modules/react/index.js is treated as runtime noise."""
    assert is_ignored_changed_path("node_modules/react/index.js") is True


def test_node_modules_nested_is_ignored():
    """Nested node_modules inside a subdirectory is treated as runtime noise."""
    assert is_ignored_changed_path("frontend/node_modules/lodash/index.js") is True


# ---------------------------------------------------------------------------
# is_ignored_changed_path — normal source files
# ---------------------------------------------------------------------------


def test_normal_source_file_is_not_ignored():
    """A regular source file returns False."""
    assert is_ignored_changed_path("aragora/swarm/boss_loop.py") is False


def test_empty_string_returns_false():
    """An empty string is not treated as ignored."""
    assert is_ignored_changed_path("") is False


def test_leading_dot_slash_normal_file_returns_false():
    """./aragora/swarm/boss_loop.py is stripped correctly and returns False."""
    assert is_ignored_changed_path("./aragora/swarm/boss_loop.py") is False


def test_leading_dot_slash_session_artifact_returns_true():
    """./.codex_session.log is recognised as a session artifact after stripping."""
    assert is_ignored_changed_path("./.codex_session.log") is True


# ---------------------------------------------------------------------------
# WorkerProcess dataclass
# ---------------------------------------------------------------------------


def test_worker_process_is_running_true_when_pid_set_and_no_exit_code():
    """is_running returns True when exit_code is None and pid is set."""
    wp = WorkerProcess(
        work_order_id="wo-1",
        agent="claude",
        worktree_path="/tmp/wt",
        branch="work/test",
        pid=12345,
    )
    assert wp.is_running is True


def test_worker_process_is_running_false_after_exit():
    """is_running returns False once exit_code is set."""
    wp = WorkerProcess(
        work_order_id="wo-2",
        agent="codex",
        worktree_path="/tmp/wt",
        branch="work/test",
        pid=9999,
        exit_code=0,
    )
    assert wp.is_running is False


def test_worker_process_is_running_false_without_pid():
    """is_running returns False when pid has not been set yet."""
    wp = WorkerProcess(
        work_order_id="wo-3",
        agent="claude",
        worktree_path="/tmp/wt",
        branch="work/test",
    )
    assert wp.is_running is False


# ---------------------------------------------------------------------------
# LaunchConfig defaults
# ---------------------------------------------------------------------------


def test_launch_config_default_construction():
    """LaunchConfig() constructs without error and has expected defaults."""
    cfg = LaunchConfig()
    assert cfg.claude_path == "claude"
    assert cfg.codex_path == "codex"
    assert cfg.timeout_seconds == 2400.0
    assert cfg.auto_commit is True
    assert cfg.allow_claude_dangerously_skip_permissions is False
    assert cfg.allow_codex_full_auto is False
