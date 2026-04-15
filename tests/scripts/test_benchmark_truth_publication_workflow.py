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


def test_runtime_prereq_creates_metrics_dir_and_allows_fresh_recurrence() -> None:
    run = _benchmark_truth_publication_run()
    assert 'METRICS_PATH=".aragora/overnight/boss_metrics.jsonl"' in run
    assert 'mkdir -p "$(dirname "$METRICS_PATH")"' in run
    assert "recurrence will generate a fresh window" in run
    assert 'test -f "$METRICS_PATH"' not in run
