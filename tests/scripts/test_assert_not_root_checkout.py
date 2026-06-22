"""Tests for ``scripts/assert_not_root_checkout.py``."""

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


guard = _load_module("assert_not_root_checkout.py")


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


def test_primary_root_checkout_blocks(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)

    report = guard.check_not_root_checkout(
        guard.build_parser().parse_args(["--cwd", str(repo), "--canonical-root", str(repo)])
    )

    assert report["ok"] is False
    assert report["status"] == "blocked_shared_root"
    assert report["canonical_root"] == str(repo.resolve())
    assert report["cwd"] == str(repo.resolve())
    assert report["reasons"] == ["current git toplevel is the shared root checkout"]


def test_linked_worktree_checkout_passes(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    linked = tmp_path / "linked"
    result = _run(["git", "worktree", "add", str(linked), "-b", "feature"], cwd=repo)
    assert result.returncode == 0, result.stderr

    report = guard.check_not_root_checkout(
        guard.build_parser().parse_args(["--cwd", str(linked), "--canonical-root", str(repo)])
    )

    assert report["ok"] is True
    assert report["status"] == "ok_linked_worktree"
    assert report["canonical_root"] == str(repo.resolve())
    assert report["cwd"] == str(linked.resolve())
    assert report["reasons"] == []


def test_canonical_root_override_is_honored(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    other_root = tmp_path / "other-root"
    other_root.mkdir()

    report = guard.check_not_root_checkout(
        guard.build_parser().parse_args(["--cwd", str(repo), "--canonical-root", str(other_root)])
    )

    assert report["ok"] is True
    assert report["status"] == "ok_linked_worktree"
    assert report["canonical_root"] == str(other_root.resolve())
    assert report["cwd"] == str(repo.resolve())


def test_json_output_contains_stable_fields(tmp_path: Path, capsys: Any) -> None:
    repo = _init_repo(tmp_path)

    exit_code = guard.main(["--cwd", str(repo), "--canonical-root", str(repo), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert '"ok": false' in captured.out
    assert '"status": "blocked_shared_root"' in captured.out
    assert '"canonical_root":' in captured.out
    assert '"cwd":' in captured.out
    assert '"reasons":' in captured.out
