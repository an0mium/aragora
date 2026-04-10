from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "automation_pr_preflight.sh"


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _run(["git", "add", "README.md"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)
    _run(["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=repo)
    return repo


def test_automation_pr_preflight_accepts_docs_diff(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "codex/docs-update"], cwd=repo)
    (repo / "docs").mkdir()
    (repo / "docs" / "note.md").write_text("note\n", encoding="utf-8")
    _run(["git", "add", "docs/note.md"], cwd=repo)
    _run(["git", "commit", "-m", "docs: add note"], cwd=repo)

    proc = _run(["bash", str(SCRIPT), "origin/main", "HEAD"], cwd=repo)

    assert proc.returncode == 0
    assert "preflight: ok" in proc.stdout


def test_automation_pr_preflight_rejects_worker_artifacts(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "codex/bad-artifact"], cwd=repo)
    (repo / ".swarm_worker_stdout.log").write_text("worker log\n", encoding="utf-8")
    _run(["git", "add", ".swarm_worker_stdout.log"], cwd=repo)
    _run(["git", "commit", "-m", "bad: commit worker log"], cwd=repo)

    proc = _run(["bash", str(SCRIPT), "origin/main", "HEAD"], cwd=repo)

    assert proc.returncode == 1
    assert "automation/session artifacts" in proc.stderr
