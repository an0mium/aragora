from __future__ import annotations

from pathlib import Path

import yaml


def _benchmark_truth_publication_run() -> str:
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "benchmark-truth-publication.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs", {})
    publish_job = jobs.get("publish-benchmark-truth", {})
    steps = publish_job.get("steps", [])
    for step in steps:
        if step.get("name") == "Verify runtime prerequisites":
            return str(step.get("run", ""))
    raise AssertionError("Verify runtime prerequisites step not found")


def _benchmark_truth_publication_steps() -> list[dict[str, object]]:
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "benchmark-truth-publication.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs", {})
    publish_job = jobs.get("publish-benchmark-truth", {})
    steps = publish_job.get("steps", [])
    if not isinstance(steps, list):
        raise AssertionError("publish-benchmark-truth steps not found")
    return [step for step in steps if isinstance(step, dict)]


def test_runtime_prereq_creates_metrics_dir_and_allows_fresh_recurrence() -> None:
    run = _benchmark_truth_publication_run()
    assert 'METRICS_PATH=".aragora/overnight/boss_metrics.jsonl"' in run
    assert 'mkdir -p "$(dirname "$METRICS_PATH")"' in run
    assert "recurrence will generate a fresh window" in run
    assert 'test -f "$METRICS_PATH"' not in run


def test_installs_dependencies_before_recurrence() -> None:
    steps = _benchmark_truth_publication_steps()
    names = [str(step.get("name", "")) for step in steps]
    install_index = names.index("Install dependencies")
    recurrence_index = names.index("Refresh recurring benchmark corpus metrics")
    assert install_index < recurrence_index
    install_run = str(steps[install_index].get("run", ""))
    assert 'python -m pip install -e ".[dev]" --quiet' in install_run


def test_installs_github_cli_before_runtime_prerequisites() -> None:
    steps = _benchmark_truth_publication_steps()
    names = [str(step.get("name", "")) for step in steps]
    gh_index = names.index("Install GitHub CLI")
    prereq_index = names.index("Verify runtime prerequisites")
    assert gh_index < prereq_index
    gh_run = str(steps[gh_index].get("run", ""))
    assert "https://api.github.com/repos/cli/cli/releases/latest" in gh_run
    assert "https://github.com/cli/cli/releases/download/" in gh_run
    assert 'echo "$gh_root/gh_${gh_version}_linux_${gh_arch}/bin" >> "$GITHUB_PATH"' in gh_run
