"""Tests for ``scripts/root_guarded_queue.py``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_module("root_guarded_queue.py")


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(["git", "add", "README.md"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)
    return repo


def test_dirty_root_blocks_command_without_running(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    sentinel = repo / "sentinel.txt"
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    report = guard.run_guard(
        guard.build_parser().parse_args(
            [
                "--cwd",
                str(repo),
                "--pr",
                "7466",
                "--",
                "bash",
                "-lc",
                f"touch {sentinel}",
            ]
        )
    )

    assert report["status"] == "blocked_dirty_root"
    assert "dirty.txt" in report["before"]["dirty_paths"]
    assert not sentinel.exists()
    assert "preserve/revert/switch authorization" in report["next_prompt"]
    assert "python3 scripts/build_next_prompt.py --pr 7466 --json" in report["next_prompt"]
    assert "clean_checkout.selected_path" in report["next_prompt"]


def test_detects_branch_drift_after_command(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "other"], cwd=repo)
    _run(["git", "switch", "main"], cwd=repo)

    report = guard.run_guard(
        guard.build_parser().parse_args(["--cwd", str(repo), "--", "git", "switch", "other"])
    )

    assert report["status"] == "blocked_root_drift"
    assert "branch drift: main -> other" in report["drift_reasons"]


def test_detects_dirty_state_after_command(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)

    report = guard.run_guard(
        guard.build_parser().parse_args(
            ["--cwd", str(repo), "--", "bash", "-lc", "printf dirty > generated.txt"]
        )
    )

    assert report["status"] == "blocked_root_drift"
    assert any("generated.txt" in reason for reason in report["drift_reasons"])
    assert "generated.txt" in report["after"]["dirty_paths"]


def test_clean_command_completes_and_emits_pr_sequence(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)

    report = guard.run_guard(
        guard.build_parser().parse_args(
            [
                "--cwd",
                str(repo),
                "--pr",
                "7466",
                "--expected-head",
                "abc123",
                "--",
                "python3",
                "-c",
                "print('ok')",
            ]
        )
    )

    assert report["status"] == "completed"
    assert report["command_result"]["stdout"] == "ok"
    assert "gh pr view 7466" in report["next_prompt"]
    assert "abc123" in report["next_prompt"]


def test_process_attribution_records_matching_process(monkeypatch: Any) -> None:
    def fake_run(args: list[str], *, cwd: Path, timeout: int = 120) -> Any:
        if args[:2] == ["ps", "-axo"]:
            return guard.CommandResult(
                command=args,
                returncode=0,
                stdout="123 1 00:01 S 0:00 0.0 python review-queue merge-packet --pr 7466",
                stderr="",
            )
        return guard.CommandResult(command=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(guard, "_run", fake_run)

    attribution = guard._process_attribution(["review-queue merge-packet"])

    assert attribution["matches"] == [
        "123 1 00:01 S 0:00 0.0 python review-queue merge-packet --pr 7466"
    ]


def test_process_attribution_reports_denied_process_census(monkeypatch: Any) -> None:
    def fake_run(args: list[str], *, cwd: Path, timeout: int = 120) -> Any:
        if args[:2] == ["ps", "-axo"]:
            raise PermissionError(1, "Operation not permitted", "ps")
        return guard.CommandResult(command=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(guard, "_run", fake_run)

    attribution = guard._process_attribution(["review-queue merge-packet"])

    assert attribution["available"] is False
    assert attribution["matches"] == []
    assert attribution["returncode"] is None
    assert "Operation not permitted" in attribution["reason"]
