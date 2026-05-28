"""Tests for ``scripts/assert_root_clean_on_main.py``."""

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


guard = _load_module("assert_root_clean_on_main.py")


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def _init_repo_with_origin(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"
    _run(["git", "init", "--bare", "-b", "main", str(origin)], cwd=tmp_path)
    repo.mkdir()
    _run(["git", "init", "-b", "main"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(["git", "add", "README.md"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)
    _run(["git", "remote", "add", "origin", str(origin)], cwd=repo)
    push = _run(["git", "push", "-u", "origin", "main"], cwd=repo)
    assert push.returncode == 0, push.stderr
    return repo, origin


def _commit(repo: Path, filename: str, content: str, message: str) -> None:
    (repo / filename).write_text(content, encoding="utf-8")
    _run(["git", "add", filename], cwd=repo)
    result = _run(["git", "commit", "-m", message], cwd=repo)
    assert result.returncode == 0, result.stderr


def test_clean_main_equal_to_origin_passes(tmp_path: Path) -> None:
    repo, _origin = _init_repo_with_origin(tmp_path)

    report = guard.check_root_clean_on_main(
        guard.build_parser().parse_args(["--canonical-root", str(repo)])
    )

    assert report["ok"] is True
    assert report["status"] == "ok_clean_on_main"
    assert report["canonical_root"] == str(repo.resolve())
    assert report["branch"] == "main"
    assert report["base_ref"] == "origin/main"
    assert report["reasons"] == []


def test_dirty_untracked_root_fails(tmp_path: Path) -> None:
    repo, _origin = _init_repo_with_origin(tmp_path)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    report = guard.check_root_clean_on_main(
        guard.build_parser().parse_args(["--canonical-root", str(repo)])
    )

    assert report["ok"] is False
    assert report["status"] == "blocked_root_not_clean_on_main"
    assert "dirty root: dirty.txt" in report["reasons"]


def test_feature_branch_fails(tmp_path: Path) -> None:
    repo, _origin = _init_repo_with_origin(tmp_path)
    _run(["git", "switch", "-c", "feature"], cwd=repo)

    report = guard.check_root_clean_on_main(
        guard.build_parser().parse_args(["--canonical-root", str(repo)])
    )

    assert report["ok"] is False
    assert "branch is feature, expected main" in report["reasons"]


def test_missing_origin_main_fails(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(["git", "add", "README.md"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)

    report = guard.check_root_clean_on_main(
        guard.build_parser().parse_args(["--canonical-root", str(repo)])
    )

    assert report["ok"] is False
    assert "missing ref origin/main" in report["reasons"]


def test_root_ahead_of_origin_main_fails(tmp_path: Path) -> None:
    repo, _origin = _init_repo_with_origin(tmp_path)
    _commit(repo, "local.txt", "local\n", "local change")

    report = guard.check_root_clean_on_main(
        guard.build_parser().parse_args(["--canonical-root", str(repo)])
    )

    assert report["ok"] is False
    assert "HEAD differs from origin/main: ahead=1 behind=0" in report["reasons"]


def test_root_behind_origin_main_fails(tmp_path: Path) -> None:
    repo, origin = _init_repo_with_origin(tmp_path)
    clone = tmp_path / "clone"
    _run(["git", "clone", str(origin), str(clone)], cwd=tmp_path)
    _run(["git", "config", "user.name", "Test User"], cwd=clone)
    _run(["git", "config", "user.email", "test@example.com"], cwd=clone)
    _commit(clone, "remote.txt", "remote\n", "remote change")
    push = _run(["git", "push", "origin", "main"], cwd=clone)
    assert push.returncode == 0, push.stderr
    _run(["git", "fetch", "origin", "main"], cwd=repo)

    report = guard.check_root_clean_on_main(
        guard.build_parser().parse_args(["--canonical-root", str(repo)])
    )

    assert report["ok"] is False
    assert "HEAD differs from origin/main: ahead=0 behind=1" in report["reasons"]


def test_merge_state_fails(tmp_path: Path) -> None:
    repo, _origin = _init_repo_with_origin(tmp_path)
    merge_head_path = _run(["git", "rev-parse", "--git-path", "MERGE_HEAD"], cwd=repo).stdout
    merge_head = Path(merge_head_path.strip())
    if not merge_head.is_absolute():
        merge_head = repo / merge_head
    merge_head.write_text("deadbeef\n", encoding="utf-8")

    report = guard.check_root_clean_on_main(
        guard.build_parser().parse_args(["--canonical-root", str(repo)])
    )

    assert report["ok"] is False
    assert "merge/rebase/cherry-pick state present: MERGE_HEAD" in report["reasons"]
