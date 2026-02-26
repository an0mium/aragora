"""Tests for worktree maintainer CLI module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from aragora.worktree import maintainer


@patch("aragora.worktree.maintainer.WorktreeLifecycleService")
def test_main_success_json(mock_service_cls, capsys) -> None:
    mock_service = MagicMock()
    mock_service.maintain_managed_dirs.return_value = {
        "ok": True,
        "directories_total": 1,
        "processed": 1,
        "skipped_active": 0,
        "skipped_missing": 0,
        "failures": 0,
        "results": [],
    }
    mock_service_cls.return_value = mock_service

    old_argv = sys.argv
    try:
        sys.argv = ["maintainer", "--repo", "/tmp/repo", "--json"]
        exit_code = maintainer.main()
    finally:
        sys.argv = old_argv

    assert exit_code == 0
    out = capsys.readouterr().out
    assert '"ok": true' in out.lower()


@patch("aragora.worktree.maintainer.WorktreeLifecycleService")
def test_main_failure_exit_code(mock_service_cls, capsys) -> None:
    mock_service = MagicMock()
    mock_service.maintain_managed_dirs.return_value = {
        "ok": False,
        "directories_total": 1,
        "processed": 1,
        "skipped_active": 0,
        "skipped_missing": 0,
        "failures": 1,
        "results": [{"managed_dir": ".worktrees/codex-auto", "status": "failed"}],
    }
    mock_service_cls.return_value = mock_service

    old_argv = sys.argv
    try:
        sys.argv = ["maintainer", "--repo", "/tmp/repo"]
        exit_code = maintainer.main()
    finally:
        sys.argv = old_argv

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "failures=1" in out


def test_build_parser_defaults_to_ff_only_strategy() -> None:
    parser = maintainer.build_parser()
    args = parser.parse_args([])
    assert args.strategy == "ff-only"
