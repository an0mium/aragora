"""Tests for ``scripts/observer_truth_probe.py``.

The probe is exercised against an isolated tmp-path git repo. We build
a tiny bare-repo "origin" plus a working clone, point the clone at it
as ``origin``, and then drive the three canonical scenarios:

1. Clean checkout at ``origin/main`` -> exits 0, ``clean == True``.
2. Uncommitted modification         -> exits 1, ``uncommitted_modified_count > 0``.
3. Untracked file present           -> exits 1, ``untracked_count > 0``.

A small ``--no-fetch`` smoke also confirms the flag is honored (a
network-less invocation against the same fixture is identical to the
default path because the origin is local).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "observer_truth_probe.py"


@pytest.fixture(autouse=True)
def _setup_import_path() -> Generator[None, None, None]:
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def _git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )


@contextmanager
def _clean_git_env() -> Generator[None, None, None]:
    """Block ambient ``GIT_*`` env vars from polluting the tmp repo."""
    saved: dict[str, str | None] = {}
    for key in list(os.environ.keys()):
        if key.startswith("GIT_") and key != "GIT_EXEC_PATH":
            saved[key] = os.environ.pop(key)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def _build_clean_fixture(tmp_path: Path) -> Path:
    """Build a tmp repo whose HEAD is at ``origin/main`` and clean.

    Layout::

        tmp_path/
          origin.git/        -- bare repo, plays "origin"
          work/              -- working clone, the probed observer
    """
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"

    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(origin)],
        check=True,
        capture_output=True,
    )

    # Seed the bare repo via a transient clone so we can set up a real
    # initial commit on ``main`` before the observer clone touches it.
    seed = tmp_path / "seed"
    subprocess.run(
        ["git", "clone", str(origin), str(seed)],
        check=True,
        capture_output=True,
    )
    _git(["config", "user.email", "probe@example.com"], cwd=seed)
    _git(["config", "user.name", "probe"], cwd=seed)
    _git(["checkout", "-b", "main"], cwd=seed)
    (seed / "README.md").write_text("seed\n", encoding="utf-8")
    _git(["add", "README.md"], cwd=seed)
    _git(["commit", "-m", "seed"], cwd=seed)
    _git(["push", "-u", "origin", "main"], cwd=seed)

    subprocess.run(
        ["git", "clone", str(origin), str(work)],
        check=True,
        capture_output=True,
    )
    _git(["config", "user.email", "probe@example.com"], cwd=work)
    _git(["config", "user.name", "probe"], cwd=work)
    # Ensure local HEAD is on ``main`` (clone default may already be).
    _git(["checkout", "main"], cwd=work)
    return work


def _run_probe(
    work: Path,
    *,
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    args = [
        sys.executable,
        str(SCRIPT_PATH),
        "--repo-root",
        str(work),
        "--quiet",
        "--no-fetch",
    ]
    if extra:
        args.extend(extra)
    return subprocess.run(args, capture_output=True, text=True, check=False)


def test_clean_checkout_exits_zero(tmp_path: Path) -> None:
    with _clean_git_env():
        work = _build_clean_fixture(tmp_path)
        result = _run_probe(work)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["clean"] is True
    assert payload["untracked_count"] == 0
    assert payload["uncommitted_modified_count"] == 0
    assert payload["submodule_dirty"] is False
    assert payload["ahead"] == 0
    assert payload["behind"] == 0
    assert payload["head_sha"]
    assert payload["origin_main_sha"]
    assert payload["head_sha"] == payload["origin_main_sha"]
    assert payload["reasons"] == []


def test_uncommitted_change_exits_nonzero(tmp_path: Path) -> None:
    with _clean_git_env():
        work = _build_clean_fixture(tmp_path)
        (work / "README.md").write_text("seed\nmodified\n", encoding="utf-8")
        result = _run_probe(work)
    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["clean"] is False
    assert payload["uncommitted_modified_count"] >= 1
    assert payload["untracked_count"] == 0
    assert any(r.startswith("uncommitted_modified=") for r in payload["reasons"])


def test_untracked_file_exits_nonzero(tmp_path: Path) -> None:
    with _clean_git_env():
        work = _build_clean_fixture(tmp_path)
        (work / "scratch.txt").write_text("hello\n", encoding="utf-8")
        result = _run_probe(work)
    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["clean"] is False
    assert payload["untracked_count"] >= 1
    assert payload["uncommitted_modified_count"] == 0
    assert any(r.startswith("untracked_files=") for r in payload["reasons"])


def test_no_strict_mode_returns_zero_even_when_dirty(tmp_path: Path) -> None:
    with _clean_git_env():
        work = _build_clean_fixture(tmp_path)
        (work / "scratch.txt").write_text("hello\n", encoding="utf-8")
        result = _run_probe(work, extra=["--no-strict-mode"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["clean"] is False
    assert payload["untracked_count"] >= 1


def test_module_probe_pure_python(tmp_path: Path) -> None:
    """Direct module call returns a dict with all documented keys."""
    import observer_truth_probe as mod

    with _clean_git_env():
        work = _build_clean_fixture(tmp_path)
        result = mod.probe(work, fetch=False)
    expected_keys = {
        "clean",
        "head_sha",
        "origin_main_sha",
        "ahead",
        "behind",
        "untracked_count",
        "uncommitted_modified_count",
        "submodule_dirty",
        "reasons",
        "repo_root",
        "checked_at",
    }
    assert set(result.keys()) == expected_keys
    assert result["clean"] is True


def test_probe_fails_closed_when_untracked_status_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import observer_truth_probe as mod

    sha = "a" * 40

    def fake_run_git(
        args: list[str],
        *,
        repo_root: Path,
        check: bool = True,
        timeout: float = mod.GIT_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        if args == ["rev-parse", "HEAD"] or args == ["rev-parse", "origin/main"]:
            return subprocess.CompletedProcess(args, 0, stdout=f"{sha}\n", stderr="")
        if args == ["rev-list", "--left-right", "--count", "origin/main...HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="0 0\n", stderr="")
        if args == ["ls-files", "--others", "--exclude-standard"]:
            raise subprocess.CalledProcessError(128, args, stderr="status failed")
        if args[0] == "diff":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args == ["submodule", "status", "--recursive"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected git args: {args!r}")

    monkeypatch.setattr(mod, "_run_git", fake_run_git)

    result = mod.probe(tmp_path, fetch=False)

    assert result["clean"] is False
    assert result["untracked_count"] == 0
    assert "untracked_files_probe_failed" in result["reasons"]


def test_probe_fails_closed_when_diff_status_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import observer_truth_probe as mod

    sha = "b" * 40

    def fake_run_git(
        args: list[str],
        *,
        repo_root: Path,
        check: bool = True,
        timeout: float = mod.GIT_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        if args == ["rev-parse", "HEAD"] or args == ["rev-parse", "origin/main"]:
            return subprocess.CompletedProcess(args, 0, stdout=f"{sha}\n", stderr="")
        if args == ["rev-list", "--left-right", "--count", "origin/main...HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="0 0\n", stderr="")
        if args == ["ls-files", "--others", "--exclude-standard"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "diff":
            raise subprocess.TimeoutExpired(args, timeout=timeout)
        if args == ["submodule", "status", "--recursive"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected git args: {args!r}")

    monkeypatch.setattr(mod, "_run_git", fake_run_git)

    result = mod.probe(tmp_path, fetch=False)

    assert result["clean"] is False
    assert result["uncommitted_modified_count"] == 0
    assert "uncommitted_modified_probe_failed" in result["reasons"]


def test_probe_fails_closed_when_submodule_status_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import observer_truth_probe as mod

    sha = "c" * 40

    def fake_run_git(
        args: list[str],
        *,
        repo_root: Path,
        check: bool = True,
        timeout: float = mod.GIT_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        if args == ["rev-parse", "HEAD"] or args == ["rev-parse", "origin/main"]:
            return subprocess.CompletedProcess(args, 0, stdout=f"{sha}\n", stderr="")
        if args == ["rev-list", "--left-right", "--count", "origin/main...HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="0 0\n", stderr="")
        if args == ["ls-files", "--others", "--exclude-standard"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "diff":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args == ["submodule", "status", "--recursive"]:
            return subprocess.CompletedProcess(args, 128, stdout="", stderr="status failed")
        raise AssertionError(f"unexpected git args: {args!r}")

    monkeypatch.setattr(mod, "_run_git", fake_run_git)

    result = mod.probe(tmp_path, fetch=False)

    assert result["clean"] is False
    assert result["submodule_dirty"] is False
    assert "submodule_status_probe_failed" in result["reasons"]
