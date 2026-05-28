"""Tests for scripts/run_typecheck_gate.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import run_typecheck_gate  # noqa: E402


def test_build_typecheck_plan_targets_only_touched_aragora_python_files(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "aragora" / "cli").mkdir(parents=True)
    (repo_root / "aragora" / "cli" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_sample.py").write_text("def test_ok(): pass\n", encoding="utf-8")

    plan = run_typecheck_gate.build_typecheck_plan(
        repo_root=repo_root,
        changed_files=["aragora/cli/main.py", "tests/test_sample.py"],
    )

    assert plan.mode == "changed"
    assert plan.targets == ["aragora/cli/main.py"]
    assert "target:aragora/cli/main.py" in plan.reasons


def test_build_typecheck_plan_forces_full_on_config_changes(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "aragora").mkdir()
    (repo_root / "aragora" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")

    plan = run_typecheck_gate.build_typecheck_plan(
        repo_root=repo_root,
        changed_files=["pyproject.toml", "aragora/__init__.py"],
    )

    assert plan.mode == "full"
    assert plan.targets == []
    assert "force_full:pyproject.toml" in plan.reasons


def test_build_typecheck_plan_forces_full_on_deleted_aragora_target(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "aragora").mkdir()

    plan = run_typecheck_gate.build_typecheck_plan(
        repo_root=repo_root,
        changed_files=["aragora/deleted_module.py"],
    )

    assert plan.mode == "full"
    assert "deleted_target:aragora/deleted_module.py" in plan.reasons


def test_get_changed_files_uses_git_diff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(
        cmd: list[str],
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> SimpleNamespace:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        return SimpleNamespace(stdout="aragora/cli/main.py\ntests/test_cli.py\n")

    monkeypatch.setattr(run_typecheck_gate.subprocess, "run", _fake_run)

    changed = run_typecheck_gate.get_changed_files(
        repo_root=tmp_path,
        base_ref="main",
        head_ref="HEAD",
    )

    assert changed == ["aragora/cli/main.py", "tests/test_cli.py"]
    assert captured["cmd"] == ["git", "diff", "--name-only", "origin/main...HEAD"]
    assert captured["cwd"] == tmp_path


def test_get_changed_files_fails_closed_on_missing_merge_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> SimpleNamespace:
        calls.append(cmd)
        raise subprocess.CalledProcessError(
            128,
            cmd,
            stderr="fatal: no merge base",
        )

    monkeypatch.setattr(run_typecheck_gate.subprocess, "run", _fake_run)

    with pytest.raises(run_typecheck_gate.ChangedFilesError) as excinfo:
        run_typecheck_gate.get_changed_files(
            repo_root=tmp_path,
            base_ref="main",
            head_ref="HEAD",
        )

    assert "Refusing to fall back to origin/main..HEAD" in str(excinfo.value)
    assert "pass explicit changed files with --files" in str(excinfo.value)
    assert calls == [
        ["git", "diff", "--name-only", "origin/main...HEAD"],
    ]


def test_get_changed_files_uses_pr_head_parent_for_stale_pull_request_merge_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    calls: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> SimpleNamespace:
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "origin/main"]:
            return SimpleNamespace(stdout="current-base\n")
        if cmd == ["git", "rev-parse", "HEAD^1"]:
            return SimpleNamespace(stdout="stale-base\n")
        if cmd == ["git", "rev-parse", "HEAD^2"]:
            return SimpleNamespace(stdout="pr-head\n")
        if cmd == ["git", "diff", "--name-only", "origin/main...pr-head"]:
            return SimpleNamespace(
                stdout=(
                    "docs/specs/MODEL_LINEAGE_DISCLOSURE.md\n"
                    "tests/governance/test_model_lineage_disclosure_recognizer.py\n"
                )
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(run_typecheck_gate.subprocess, "run", _fake_run)

    changed = run_typecheck_gate.get_changed_files(
        repo_root=tmp_path,
        base_ref="main",
        head_ref="HEAD",
    )

    assert changed == [
        "docs/specs/MODEL_LINEAGE_DISCLOSURE.md",
        "tests/governance/test_model_lineage_disclosure_recognizer.py",
    ]
    assert ["git", "diff", "--name-only", "origin/main...HEAD"] not in calls


def test_get_changed_files_deepens_history_for_no_merge_base_pr_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_HEAD_REF", "codex/model-lineage-docs")
    calls: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> SimpleNamespace:
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "origin/main"]:
            return SimpleNamespace(stdout="current-base\n")
        if cmd == ["git", "rev-parse", "HEAD^1"]:
            return SimpleNamespace(stdout="stale-base\n")
        if cmd == ["git", "rev-parse", "HEAD^2"]:
            return SimpleNamespace(stdout="pr-head\n")
        if cmd == ["git", "fetch", "--no-tags", "--deepen=128", "origin", "main"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd == ["git", "diff", "--name-only", "origin/main...pr-head"]:
            if calls.count(["git", "diff", "--name-only", "origin/main...pr-head"]) == 1:
                raise subprocess.CalledProcessError(
                    128,
                    cmd,
                    stderr="fatal: origin/main...pr-head: no merge base",
                )
            return SimpleNamespace(
                stdout=(
                    "docs/specs/MODEL_LINEAGE_DISCLOSURE.md\n"
                    "tests/governance/test_model_lineage_disclosure_recognizer.py\n"
                )
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(run_typecheck_gate.subprocess, "run", _fake_run)

    changed = run_typecheck_gate.get_changed_files(
        repo_root=tmp_path,
        base_ref="main",
        head_ref="HEAD",
    )
    plan = run_typecheck_gate.build_typecheck_plan(
        repo_root=tmp_path,
        changed_files=changed,
    )

    assert changed == [
        "docs/specs/MODEL_LINEAGE_DISCLOSURE.md",
        "tests/governance/test_model_lineage_disclosure_recognizer.py",
    ]
    assert plan.mode == "skip"
    assert plan.targets == []
    assert "no_aragora_python_targets" in plan.reasons
    assert ["git", "diff", "--name-only", "origin/main..HEAD"] not in calls


def test_get_changed_files_raises_non_merge_base_git_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _fake_run(
        cmd: list[str],
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> SimpleNamespace:
        raise subprocess.CalledProcessError(2, cmd, stderr="fatal: bad revision")

    monkeypatch.setattr(run_typecheck_gate.subprocess, "run", _fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        run_typecheck_gate.get_changed_files(
            repo_root=tmp_path,
            base_ref="main",
            head_ref="HEAD",
        )


def test_main_writes_github_outputs_and_targets_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path
    (repo_root / "aragora" / "inbox").mkdir(parents=True)
    (repo_root / "aragora" / "inbox" / "triage_runner.py").write_text(
        "from __future__ import annotations\n",
        encoding="utf-8",
    )
    github_output = repo_root / "github_output.txt"
    targets_file = repo_root / "targets.txt"

    exit_code = run_typecheck_gate.main(
        [
            "--repo-root",
            str(repo_root),
            "--files",
            "aragora/inbox/triage_runner.py",
            "tests/inbox/test_triage_runner.py",
            "--github-output",
            str(github_output),
            "--targets-file",
            str(targets_file),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "changed"
    assert payload["targets"] == ["aragora/inbox/triage_runner.py"]
    assert targets_file.read_text(encoding="utf-8") == "aragora/inbox/triage_runner.py\n"
    output = github_output.read_text(encoding="utf-8")
    assert "mode=changed" in output
    assert "target_count=1" in output


def test_main_accepts_explicit_docs_test_files_without_git_diff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def _fake_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("explicit --files should not query git diff")

    monkeypatch.setattr(run_typecheck_gate.subprocess, "run", _fake_run)

    exit_code = run_typecheck_gate.main(
        [
            "--repo-root",
            str(tmp_path),
            "--files",
            "docs/specs/MODEL_LINEAGE_DISCLOSURE.md",
            "tests/governance/test_model_lineage_disclosure_recognizer.py",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "skip"
    assert payload["targets"] == []
    assert payload["changed_files"] == [
        "docs/specs/MODEL_LINEAGE_DISCLOSURE.md",
        "tests/governance/test_model_lineage_disclosure_recognizer.py",
    ]
    assert payload["reasons"] == ["no_aragora_python_targets"]


def test_main_exits_cleanly_when_changed_files_cannot_be_computed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _fake_run(
        cmd: list[str],
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> SimpleNamespace:
        raise subprocess.CalledProcessError(
            128,
            cmd,
            stderr="fatal: no merge base",
        )

    monkeypatch.setattr(run_typecheck_gate.subprocess, "run", _fake_run)

    with pytest.raises(SystemExit) as excinfo:
        run_typecheck_gate.main(
            [
                "--repo-root",
                str(tmp_path),
                "--base-ref",
                "main",
            ]
        )

    assert "Refusing to fall back to origin/main..HEAD" in str(excinfo.value)
