"""Tests for canonical worktree autopilot interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.worktree.autopilot import (
    AutopilotRequest,
    build_autopilot_command,
    resolve_repo_root,
    run_autopilot,
)


def test_resolve_repo_root_falls_back_to_input_path(tmp_path: Path) -> None:
    resolved = resolve_repo_root(tmp_path)
    assert resolved == tmp_path.resolve()


def test_build_ensure_command() -> None:
    repo = Path("/tmp/repo")
    request = AutopilotRequest(
        action="ensure",
        managed_dir=".worktrees/codex-auto-ci",
        base_branch="develop",
        agent="codex-ci",
        session_id="ci-1",
        force_new=True,
        strategy="rebase",
        reconcile=True,
        print_path=True,
        json_output=True,
    )
    cmd = build_autopilot_command(repo_root=repo, request=request, python_executable="python3")
    assert cmd[:2] == ["python3", "/tmp/repo/scripts/codex_worktree_autopilot.py"]
    assert "--managed-dir" in cmd
    assert ".worktrees/codex-auto-ci" in cmd
    assert "ensure" in cmd
    assert "--agent" in cmd and "codex-ci" in cmd
    assert "--base" in cmd and "develop" in cmd
    assert "--strategy" in cmd and "rebase" in cmd
    assert "--session-id" in cmd and "ci-1" in cmd
    assert "--force-new" in cmd
    assert "--reconcile" in cmd
    assert "--print-path" in cmd
    assert "--json" in cmd


def test_build_cleanup_command_delete_branches_false() -> None:
    repo = Path("/tmp/repo")
    request = AutopilotRequest(
        action="cleanup",
        base_branch="main",
        ttl_hours=48,
        force_unmerged=True,
        delete_branches=False,
        json_output=True,
    )
    cmd = build_autopilot_command(repo_root=repo, request=request)
    assert "cleanup" in cmd
    assert "--base" in cmd and "main" in cmd
    assert "--ttl-hours" in cmd and "48" in cmd
    assert "--force-unmerged" in cmd
    assert "--no-delete-branches" in cmd
    assert "--json" in cmd


def test_build_command_rejects_invalid_action() -> None:
    with pytest.raises(ValueError, match="Unsupported autopilot action"):
        build_autopilot_command(
            repo_root=Path("/tmp/repo"),
            request=AutopilotRequest(action="unknown"),
        )


def test_run_autopilot_raises_when_script_missing(tmp_path: Path) -> None:
    request = AutopilotRequest(action="status")
    with pytest.raises(FileNotFoundError):
        run_autopilot(repo_root=tmp_path, request=request)


@patch("aragora.worktree.autopilot.subprocess.run")
def test_run_autopilot_invokes_subprocess(mock_run, tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "codex_worktree_autopilot.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    mock_run.return_value = argparse.Namespace(stdout="ok", stderr="", returncode=0)

    result = run_autopilot(repo_root=tmp_path, request=AutopilotRequest(action="status"))

    assert result.stdout == "ok"
    call = mock_run.call_args
    cmd = call.args[0]
    assert str(script) in cmd
    assert call.kwargs["cwd"] == tmp_path
    assert call.kwargs["capture_output"] is True
    assert call.kwargs["text"] is True
    assert call.kwargs["check"] is False
