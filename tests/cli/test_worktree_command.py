"""Tests for ``aragora worktree`` CLI command."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from aragora.cli.commands.worktree import (
    _cmd_worktree_autopilot,
    add_worktree_parser,
    cmd_worktree,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    add_worktree_parser(subs)
    return parser


def _autopilot_args(**overrides: object) -> argparse.Namespace:
    base = {
        "managed_dir": ".worktrees/codex-auto",
        "auto_action": "status",
        "agent": "codex",
        "session_id": None,
        "force_new": False,
        "strategy": "merge",
        "reconcile": False,
        "all": False,
        "path": None,
        "ttl_hours": 24,
        "force_unmerged": False,
        "delete_branches": None,
        "json": False,
        "print_path": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestWorktreeParser:
    def test_autopilot_defaults(self):
        args = _parser().parse_args(["worktree", "autopilot"])
        assert args.command == "worktree"
        assert args.wt_action == "autopilot"
        assert args.auto_action == "status"
        assert args.managed_dir == ".worktrees/codex-auto"

    def test_autopilot_ensure_parse(self):
        args = _parser().parse_args(
            [
                "worktree",
                "--base",
                "develop",
                "autopilot",
                "ensure",
                "--managed-dir",
                ".worktrees/codex-auto-ci",
                "--agent",
                "codex-ci",
                "--session-id",
                "ci-123",
                "--force-new",
                "--strategy",
                "rebase",
                "--reconcile",
                "--print-path",
                "--json",
            ]
        )

        assert args.base == "develop"
        assert args.wt_action == "autopilot"
        assert args.auto_action == "ensure"
        assert args.managed_dir == ".worktrees/codex-auto-ci"
        assert args.agent == "codex-ci"
        assert args.session_id == "ci-123"
        assert args.force_new is True
        assert args.strategy == "rebase"
        assert args.reconcile is True
        assert args.print_path is True
        assert args.json is True


class TestWorktreeDispatch:
    @patch("aragora.cli.commands.worktree._cmd_worktree_autopilot")
    def test_dispatches_autopilot_before_branch_coordinator_import(self, mock_autopilot):
        args = argparse.Namespace(wt_action="autopilot", repo="/tmp/repo", base="main")
        cmd_worktree(args)
        mock_autopilot.assert_called_once()

        call = mock_autopilot.call_args
        assert call.kwargs["repo_path"] == Path("/tmp/repo").resolve()
        assert call.kwargs["base_branch"] == "main"


class TestWorktreeAutopilot:
    def test_missing_script_prints_error(self, capsys, tmp_path: Path):
        args = _autopilot_args()

        _cmd_worktree_autopilot(args, repo_path=tmp_path, base_branch="main")

        out = capsys.readouterr().out
        assert "autopilot script not found" in out

    @patch("aragora.cli.commands.worktree.subprocess.run")
    def test_runs_ensure_with_expected_flags(self, mock_run, tmp_path: Path):
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script = scripts_dir / "codex_worktree_autopilot.py"
        script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        mock_run.return_value = argparse.Namespace(stdout="/tmp/wt\n", stderr="", returncode=0)

        args = _autopilot_args(
            auto_action="ensure",
            agent="codex-ci",
            session_id="ci-123",
            force_new=True,
            strategy="rebase",
            reconcile=True,
            print_path=True,
            json=True,
        )

        _cmd_worktree_autopilot(args, repo_path=tmp_path, base_branch="develop")

        call = mock_run.call_args
        cmd = call.args[0]
        assert str(script) in cmd
        assert "--managed-dir" in cmd
        assert ".worktrees/codex-auto" in cmd
        assert "ensure" in cmd
        assert "--agent" in cmd and "codex-ci" in cmd
        assert "--session-id" in cmd and "ci-123" in cmd
        assert "--base" in cmd and "develop" in cmd
        assert "--strategy" in cmd and "rebase" in cmd
        assert "--force-new" in cmd
        assert "--reconcile" in cmd
        assert "--print-path" in cmd
        assert "--json" in cmd
        assert call.kwargs["cwd"] == tmp_path
        assert call.kwargs["capture_output"] is True
        assert call.kwargs["text"] is True
        assert call.kwargs["check"] is False

    @patch("aragora.cli.commands.worktree.subprocess.run")
    def test_nonzero_exit_reports_failure(self, mock_run, capsys, tmp_path: Path):
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script = scripts_dir / "codex_worktree_autopilot.py"
        script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        mock_run.return_value = argparse.Namespace(stdout="", stderr="boom", returncode=2)

        args = _autopilot_args(auto_action="status")
        _cmd_worktree_autopilot(args, repo_path=tmp_path, base_branch="main")

        captured = capsys.readouterr()
        assert "boom" in captured.err
        assert "Autopilot command failed with exit code 2" in captured.out
