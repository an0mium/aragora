from __future__ import annotations

import json
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


def test_automation_pr_preflight_json_accepts_docs_diff(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "codex/docs-json"], cwd=repo)
    (repo / "docs").mkdir()
    (repo / "docs" / "note.md").write_text("note\n", encoding="utf-8")
    _run(["git", "add", "docs/note.md"], cwd=repo)
    _run(["git", "commit", "-m", "docs: add json note"], cwd=repo)

    proc = _run(["bash", str(SCRIPT), "--json", "origin/main", "HEAD"], cwd=repo)

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["base_ref"] == "origin/main"
    assert payload["head_ref"] == "HEAD"
    assert payload["changed_files"] == ["docs/note.md"]
    assert payload["docs_only"] is True
    assert payload["source_without_tests"] is False
    assert payload["forbidden_files"] == []
    assert payload["rescue_publish_files"] == []
    assert payload["suggested_validation_commands"] == []


def test_automation_pr_preflight_rejects_worker_artifacts(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "codex/bad-artifact"], cwd=repo)
    (repo / ".swarm_worker_stdout.log").write_text("worker log\n", encoding="utf-8")
    _run(["git", "add", ".swarm_worker_stdout.log"], cwd=repo)
    _run(["git", "commit", "-m", "bad: commit worker log"], cwd=repo)

    proc = _run(["bash", str(SCRIPT), "origin/main", "HEAD"], cwd=repo)

    assert proc.returncode == 1
    assert "automation/session artifacts" in proc.stderr


def test_automation_pr_preflight_json_rejects_worker_artifacts(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "codex/bad-artifact-json"], cwd=repo)
    (repo / ".swarm_worker_stdout.log").write_text("worker log\n", encoding="utf-8")
    _run(["git", "add", ".swarm_worker_stdout.log"], cwd=repo)
    _run(["git", "commit", "-m", "bad: commit worker log"], cwd=repo)

    proc = _run(["bash", str(SCRIPT), "--json", "origin/main", "HEAD"], cwd=repo)

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["status"] == "failed"
    assert payload["forbidden_files"] == [".swarm_worker_stdout.log"]
    assert payload["error"] == "automation/session artifacts must not be committed"


def test_automation_pr_preflight_json_suggests_validation_for_source_without_tests(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "codex/source-json"], cwd=repo)
    source = repo / "scripts" / "tool.py"
    source.parent.mkdir()
    source.write_text("print('hi')\n", encoding="utf-8")
    _run(["git", "add", "scripts/tool.py"], cwd=repo)
    _run(["git", "commit", "-m", "feat: add tool"], cwd=repo)

    proc = _run(["bash", str(SCRIPT), "--json", "origin/main", "HEAD"], cwd=repo)

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["docs_only"] is False
    assert payload["source_without_tests"] is True
    assert payload["suggested_validation_commands"] == [
        "python3 scripts/nomic_ci_test_selector.py --changed-files scripts/tool.py --dry-run",
        "python3 -m ruff check scripts/tool.py",
    ]


def test_automation_pr_preflight_rejects_synthetic_preflight_commit_subject(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "codex/preflight-preflight-repro"], cwd=repo)
    workflow_dir = repo / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "benchmark-truth-publication.yml").write_text(
        "2026-04-14T19:38:09-0500\n",
        encoding="utf-8",
    )
    _run(["git", "add", ".github/workflows/benchmark-truth-publication.yml"], cwd=repo)
    _run(["git", "commit", "-m", "chore: preflight worker check"], cwd=repo)

    proc = _run(["bash", str(SCRIPT), "origin/main", "HEAD"], cwd=repo)

    assert proc.returncode == 1
    assert "synthetic preflight validation commits" in proc.stderr


def test_automation_pr_preflight_rejects_scratch_preflight_diff(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "codex/preflight-scratch-repro"], cwd=repo)
    scratch_dir = repo / "scratch"
    scratch_dir.mkdir()
    (scratch_dir / "preflight_worker_check.txt").write_text(
        "2026-04-15T03:10:00Z\n", encoding="utf-8"
    )
    _run(["git", "add", "scratch/preflight_worker_check.txt"], cwd=repo)
    _run(["git", "commit", "-m", "docs: publish scratch preflight artifact"], cwd=repo)

    proc = _run(["bash", str(SCRIPT), "origin/main", "HEAD"], cwd=repo)

    assert proc.returncode == 1
    assert "synthetic preflight validation scratch diffs" in proc.stderr


def test_automation_pr_preflight_rejects_aragora_coordination_artifacts(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "codex/bad-aragora-artifact"], cwd=repo)
    artifact = repo / ".aragora" / "automation-outbox" / "open-pr-demo.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"idempotency_key": "demo"}\n', encoding="utf-8")
    _run(["git", "add", ".aragora/automation-outbox/open-pr-demo.json"], cwd=repo)
    _run(["git", "commit", "-m", "bad: commit outbox artifact"], cwd=repo)

    proc = _run(["bash", str(SCRIPT), "origin/main", "HEAD"], cwd=repo)

    assert proc.returncode == 1
    assert "automation/session artifacts" in proc.stderr
    assert ".aragora/automation-outbox/open-pr-demo.json" in proc.stderr


def test_automation_pr_preflight_rejects_rescue_publish_artifacts(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "codex/bad-rescue-publish"], cwd=repo)
    artifact = (
        repo / "published" / "rescue_productization" / "rescue-productization-20260516T120000Z.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    _run(
        [
            "git",
            "add",
            "published/rescue_productization/rescue-productization-20260516T120000Z.json",
        ],
        cwd=repo,
    )
    _run(["git", "commit", "-m", "bad: commit rescue publish artifact"], cwd=repo)

    proc = _run(["bash", str(SCRIPT), "origin/main", "HEAD"], cwd=repo)

    assert proc.returncode == 1
    assert "rescue productization publish artifacts" in proc.stderr
    assert (
        "published/rescue_productization/rescue-productization-20260516T120000Z.json" in proc.stderr
    )


def test_automation_pr_preflight_accepts_unrelated_latest_json(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "switch", "-c", "codex/unrelated-latest-json"], cwd=repo)
    artifact = repo / "docs" / "fixtures" / "latest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    _run(["git", "add", "docs/fixtures/latest.json"], cwd=repo)
    _run(["git", "commit", "-m", "docs: add fixture latest json"], cwd=repo)

    proc = _run(["bash", str(SCRIPT), "origin/main", "HEAD"], cwd=repo)

    assert proc.returncode == 0
    assert "preflight: ok" in proc.stdout
