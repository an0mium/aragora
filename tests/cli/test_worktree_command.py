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
        "auto_base": None,
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

    def test_autopilot_base_override_parse(self):
        args = _parser().parse_args(
            [
                "worktree",
                "autopilot",
                "status",
                "--base",
                "release",
            ]
        )
        assert args.base == "main"
        assert args.auto_base == "release"


class TestWorktreeDispatch:
    @patch("aragora.cli.commands.worktree._cmd_worktree_autopilot")
    def test_dispatches_autopilot_before_branch_coordinator_import(self, mock_autopilot):
        args = argparse.Namespace(
            wt_action="autopilot",
            repo="/tmp/repo",
            base="main",
            auto_base=None,
        )
        cmd_worktree(args)
        mock_autopilot.assert_called_once()

        call = mock_autopilot.call_args
        assert call.kwargs["repo_path"] == Path("/tmp/repo").resolve()
        assert call.kwargs["base_branch"] == "main"

    @patch("aragora.cli.commands.worktree._cmd_worktree_autopilot")
    def test_dispatches_autopilot_with_auto_base_override(self, mock_autopilot):
        args = argparse.Namespace(
            wt_action="autopilot",
            repo="/tmp/repo",
            base="main",
            auto_base="release",
        )
        cmd_worktree(args)
        call = mock_autopilot.call_args
        assert call.kwargs["base_branch"] == "release"


class TestWorktreeAutopilot:
    @patch("aragora.cli.commands.worktree.run_autopilot", side_effect=FileNotFoundError("/x/y/z"))
    def test_missing_script_prints_error(self, _mock_run, capsys, tmp_path: Path):
        args = _autopilot_args()

        _cmd_worktree_autopilot(args, repo_path=tmp_path, base_branch="main")

        out = capsys.readouterr().out
        assert "autopilot script not found" in out

    @patch("aragora.cli.commands.worktree.run_autopilot")
    def test_runs_ensure_with_expected_flags(self, mock_run, tmp_path: Path):
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
        request = call.kwargs["request"]
        assert call.kwargs["repo_root"] == tmp_path
        assert call.kwargs["python_executable"]
        assert request.action == "ensure"
        assert request.managed_dir == ".worktrees/codex-auto"
        assert request.agent == "codex-ci"
        assert request.session_id == "ci-123"
        assert request.base_branch == "develop"
        assert request.strategy == "rebase"
        assert request.force_new is True
        assert request.reconcile is True
        assert request.print_path is True
        assert request.json_output is True

    @patch("aragora.cli.commands.worktree.run_autopilot")
    def test_nonzero_exit_reports_failure(self, mock_run, capsys, tmp_path: Path):
        mock_run.return_value = argparse.Namespace(stdout="", stderr="boom", returncode=2)

        args = _autopilot_args(auto_action="status")
        _cmd_worktree_autopilot(args, repo_path=tmp_path, base_branch="main")

        captured = capsys.readouterr()
        assert "boom" in captured.err
        assert "Autopilot command failed with exit code 2" in captured.out
