"""Unit tests for scripts/disk_recovery_coordinator.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def _args(tmp_path: Path, *, apply: bool) -> argparse.Namespace:
    return argparse.Namespace(
        apply=apply,
        command_timeout=5.0,
        inspect_timeout=5.0,
        remove_timeout=5.0,
        max_cleanup_per_cycle=5,
        max_inspect_per_cycle=25,
        quarantine_file=tmp_path / "quarantine.jsonl",
    )


def test_external_cleanup_quarantines_inspect_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import disk_recovery_coordinator as mod

    candidate = "/Users/armand/.codex/worktrees/abcd/aragora"
    monkeypatch.setattr(mod, "_git_worktree_paths", lambda _repo, _timeout: [candidate])
    monkeypatch.setattr(mod, "_active_external_worktrees", lambda _prefix: set())

    def fake_run_json(*_args, **_kwargs):
        return None, {"returncode": 124, "timed_out": True, "stderr": "timeout"}

    monkeypatch.setattr(mod, "_run_json", fake_run_json)

    result = mod._external_cleanup(tmp_path, _args(tmp_path, apply=True))

    assert result["removed"] == []
    assert result["blocked"] == [{"path": candidate, "reason": "inspect_timeout"}]
    records = [
        json.loads(line) for line in (tmp_path / "quarantine.jsonl").read_text().splitlines()
    ]
    assert records[0]["path"] == candidate
    assert records[0]["reason"] == "inspect_timeout"


def test_external_cleanup_dry_run_reports_would_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import disk_recovery_coordinator as mod

    candidate = "/Users/armand/.codex/worktrees/abcd/aragora"
    monkeypatch.setattr(mod, "_git_worktree_paths", lambda _repo, _timeout: [candidate])
    monkeypatch.setattr(mod, "_active_external_worktrees", lambda _prefix: set())
    monkeypatch.setattr(
        mod,
        "_run_json",
        lambda *_args, **_kwargs: ({"removable": True}, {"returncode": 0, "timed_out": False}),
    )

    result = mod._external_cleanup(tmp_path, _args(tmp_path, apply=False))

    assert result["would_remove"] == [candidate]
    assert result["removed"] == []


def test_external_cleanup_skips_quarantined_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import disk_recovery_coordinator as mod

    candidate = "/Users/armand/.codex/worktrees/abcd/aragora"
    (tmp_path / "quarantine.jsonl").write_text(
        json.dumps({"path": candidate, "reason": "inspect_timeout"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_git_worktree_paths", lambda _repo, _timeout: [candidate])
    monkeypatch.setattr(mod, "_active_external_worktrees", lambda _prefix: set())

    result = mod._external_cleanup(tmp_path, _args(tmp_path, apply=True))

    assert result["quarantined_skipped"] == 1
    assert result["inspected"] == 0
    assert result["removed"] == []


def test_external_cleanup_honors_inspect_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import disk_recovery_coordinator as mod

    candidates = [
        "/Users/armand/.codex/worktrees/aaaa/aragora",
        "/Users/armand/.codex/worktrees/bbbb/aragora",
    ]
    args = _args(tmp_path, apply=False)
    args.max_inspect_per_cycle = 1
    monkeypatch.setattr(mod, "_git_worktree_paths", lambda _repo, _timeout: candidates)
    monkeypatch.setattr(mod, "_active_external_worktrees", lambda _prefix: set())
    monkeypatch.setattr(
        mod,
        "_run_json",
        lambda *_args, **_kwargs: (
            {"removable": False, "blockers": ["open_pr"]},
            {"timed_out": False},
        ),
    )

    result = mod._external_cleanup(tmp_path, args)

    assert result["inspected"] == 1
    assert result["max_inspect_per_cycle"] == 1


def test_root_clean_current_allows_branch_ahead_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import disk_recovery_coordinator as mod

    def fake_run(argv, **_kwargs):
        stdout = ""
        returncode = 0
        if argv == ["git", "rev-parse", "HEAD"]:
            stdout = "feature-head\n"
        elif argv == ["git", "rev-parse", "origin/main"]:
            stdout = "main-head\n"
        elif argv == ["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"]:
            returncode = 0
        return {
            "argv": argv,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": "",
            "timed_out": False,
            "elapsed": 0,
        }

    monkeypatch.setattr(mod, "_run", fake_run)

    ok, payload = mod._root_clean_current(tmp_path, 5.0, allow_branch_ahead=True)

    assert ok is True
    assert payload["same_head"] is False
    assert payload["branch_ahead_allowed"] is True
