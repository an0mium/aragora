"""Tests for scripts/pristine_main_health.py (epic #9039, issue #9043)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "pristine_main_health.py"


@pytest.fixture()
def mod():
    spec = importlib.util.spec_from_file_location("pristine_main_health_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_worktree(mod, monkeypatch, sha="deadbeef" * 5):
    monkeypatch.setattr(mod, "refresh_pristine_worktree", lambda repo, pristine: sha)
    return sha


class _Proc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_existing_pristine_without_owner_marker_refuses_destructive_refresh(
    mod, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    pristine = tmp_path / "other-checkout"
    pristine.mkdir()
    (pristine / ".git").mkdir()
    commands: list[list[str]] = []

    def fake_run(cmd, *, cwd, timeout):
        commands.append(cmd)
        if cmd[:3] == ["git", "fetch", "origin"]:
            return _Proc(0)
        raise AssertionError(f"unexpected command after missing marker: {cmd}")

    monkeypatch.setattr(mod, "_run", fake_run)

    with pytest.raises(SystemExit) as exc:
        mod.refresh_pristine_worktree(repo, pristine)

    assert "unmarked --pristine-dir" in str(exc.value)
    assert not any(cmd[:3] == ["git", "reset", "--hard"] for cmd in commands)
    assert not any(cmd[:2] == ["git", "clean"] for cmd in commands)


def test_existing_marked_registered_pristine_refreshes(mod, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    pristine = tmp_path / "pristine"
    pristine.mkdir()
    (pristine / ".git").mkdir()
    mod._write_owner_marker(repo, pristine)
    commands: list[list[str]] = []

    def fake_run(cmd, *, cwd, timeout):
        commands.append(cmd)
        if cmd[:3] == ["git", "fetch", "origin"]:
            return _Proc(0)
        if cmd == ["git", "worktree", "list", "--porcelain"]:
            return _Proc(0, f"worktree {pristine}\nHEAD cafebabe\n")
        if cmd[0:3] == ["git", "checkout", "--detach"]:
            return _Proc(0)
        if cmd[:3] == ["git", "reset", "--hard"]:
            return _Proc(0)
        if cmd[:3] == ["git", "clean", "-fdx"]:
            return _Proc(0)
        if cmd == ["git", "rev-parse", "HEAD"]:
            return _Proc(0, "cafebabe\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "_run", fake_run)

    assert mod.refresh_pristine_worktree(repo, pristine) == "cafebabe"
    assert ["git", "reset", "--hard", "origin/main"] in commands
    assert ["git", "clean", "-fdx", "--quiet"] in commands


def test_green_run_writes_ledger_and_no_halt(mod, monkeypatch, tmp_path):
    _install_fake_worktree(mod, monkeypatch)
    monkeypatch.setattr(mod, "_run", lambda cmd, *, cwd, timeout: _Proc(0, "ok"))
    halt = tmp_path / "halt.json"
    rc = mod.main(
        [
            "--repo-root",
            str(tmp_path),
            "--pristine-dir",
            str(tmp_path / "pristine"),
            "--halt-file",
            str(halt),
            "--suite",
            "required",
        ]
    )
    assert rc == 0
    assert not halt.exists()
    from aragora.nomic.throughput import ThroughputLedger

    (record,) = ThroughputLedger(tmp_path).records()
    assert record.kind == "note"
    assert record.data["event"] == "pristine_main_health"
    assert record.data["green"] is True


def test_red_run_writes_merge_executor_compatible_halt_marker(mod, monkeypatch, tmp_path):
    sha = _install_fake_worktree(mod, monkeypatch)
    monkeypatch.setattr(
        mod, "_run", lambda cmd, *, cwd, timeout: _Proc(1, "FAILED tests/x.py::t - boom")
    )
    halt = tmp_path / "halt.json"
    rc = mod.main(
        [
            "--repo-root",
            str(tmp_path),
            "--pristine-dir",
            str(tmp_path / "pristine"),
            "--halt-file",
            str(halt),
            "--suite",
            "required",
        ]
    )
    assert rc == 1
    marker = json.loads(halt.read_text())
    assert marker["reason"] == "main_red"  # the field merge_executor keys on
    assert sha[:12] in marker["details"][0]
    assert "stdout:\nFAILED tests/x.py::t - boom" in marker["details"][1]
    assert "stderr:\n<empty>" in marker["details"][1]
    assert "human deletes" in marker["re_arm"]
    from aragora.nomic.throughput import ThroughputLedger

    (record,) = ThroughputLedger(tmp_path).records()
    assert record.data["green"] is False
    assert record.data["failures"]


def test_stderr_only_bootstrap_failure_is_preserved(mod, monkeypatch, tmp_path):
    _install_fake_worktree(mod, monkeypatch)
    monkeypatch.setattr(
        mod,
        "_run",
        lambda cmd, *, cwd, timeout: _Proc(1, stderr="python: No module named pytest"),
    )
    halt = tmp_path / "halt.json"

    assert (
        mod.main(
            [
                "--repo-root",
                str(tmp_path),
                "--pristine-dir",
                str(tmp_path / "pristine"),
                "--halt-file",
                str(halt),
                "--suite",
                "required",
            ]
        )
        == 1
    )

    failure = json.loads(halt.read_text())["details"][1]
    assert "stdout:\n<empty>" in failure
    assert "stderr:\npython: No module named pytest" in failure


def test_timeout_preserves_bounded_stdout_and_stderr(mod, monkeypatch, tmp_path):
    _install_fake_worktree(mod, monkeypatch)
    stdout = "\n".join(f"stdout-{index}-" + "x" * 200 for index in range(30))
    stderr = "\n".join(f"stderr-{index}-" + "y" * 200 for index in range(30))

    def timeout_run(cmd, *, cwd, timeout):
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=timeout,
            output=stdout,
            stderr=stderr,
        )

    monkeypatch.setattr(mod, "_run", timeout_run)
    halt = tmp_path / "halt.json"

    assert (
        mod.main(
            [
                "--repo-root",
                str(tmp_path),
                "--pristine-dir",
                str(tmp_path / "pristine"),
                "--halt-file",
                str(halt),
                "--suite",
                "required",
                "--timeout-minutes",
                "1",
            ]
        )
        == 1
    )

    failure = json.loads(halt.read_text())["details"][1]
    assert "TIMEOUT after 1m" in failure
    assert "stdout-29-" in failure and "stdout-0-" not in failure
    assert "stderr-29-" in failure and "stderr-0-" not in failure
    stdout_tail, stderr_tail = failure.split("stdout:\n", 1)[1].split("\nstderr:\n", 1)
    assert len(stdout_tail) <= mod.FAILURE_TAIL_CHARS
    assert len(stderr_tail) <= mod.FAILURE_TAIL_CHARS


def test_no_halt_file_flag_reports_only(mod, monkeypatch, tmp_path):
    _install_fake_worktree(mod, monkeypatch)
    monkeypatch.setattr(mod, "_run", lambda cmd, *, cwd, timeout: _Proc(2))
    halt = tmp_path / "halt.json"
    rc = mod.main(
        [
            "--repo-root",
            str(tmp_path),
            "--pristine-dir",
            str(tmp_path / "pristine"),
            "--halt-file",
            str(halt),
            "--suite",
            "required",
            "--no-halt-file",
        ]
    )
    assert rc == 1
    assert not halt.exists()


def test_full_suite_ignores_known_broken_collection(mod):
    (full_cmd,) = mod.SUITES["full"]
    assert "--ignore=tests/connectors" in full_cmd
