"""Tests for scripts/phase0b_role_benchmark.py result capture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
import pytest

from aragora.swarm.campaign import CampaignManifest, CampaignProject, save_campaign_manifest
from aragora.swarm.spec import SwarmSpec

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import phase0b_role_benchmark  # noqa: E402


def _result_row(**overrides) -> dict[str, object]:
    row: dict[str, object] = {
        "recorded_at": "2026-03-17T00:00:00+00:00",
        "experiment_id": "exp-001",
        "config_id": "p-codex_w-claude_r-claude",
        "project_id": "B-6",
        "runtime_manifest_path": "/tmp/phase0b-runtime.yaml",
        "worker_branch": "codex/swarm-subtask-1",
        "worker_commit": "abc123",
        "worker_branch_count": 1,
        "worker_commit_count": 1,
        "worker_branches_json": json.dumps(["codex/swarm-subtask-1"]),
        "worker_commits_json": json.dumps(["abc123"]),
    }
    row.update(overrides)
    return row


def test_validate_result_row_accepts_consistent_worker_metadata() -> None:
    phase0b_role_benchmark.validate_result_row(_result_row())


def test_validate_result_row_rejects_branch_count_mismatch() -> None:
    row = _result_row(worker_branch_count=2)

    with pytest.raises(ValueError, match="worker_branch_count"):
        phase0b_role_benchmark.validate_result_row(row)


def test_validate_result_row_rejects_invalid_json_list() -> None:
    row = _result_row(worker_branches_json="not-json")

    with pytest.raises(ValueError, match="worker_branches_json"):
        phase0b_role_benchmark.validate_result_row(row)


def test_validate_result_row_rejects_primary_commit_outside_list() -> None:
    row = _result_row(worker_commit="missing")

    with pytest.raises(ValueError, match="worker_commit"):
        phase0b_role_benchmark.validate_result_row(row)


def test_load_json_returns_default_for_missing_file(tmp_path: Path) -> None:
    default = {"runs": []}

    assert phase0b_role_benchmark._load_json(tmp_path / "missing.json", default) == default


def test_load_json_rejects_malformed_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Malformed JSON file .*results\.json"):
        phase0b_role_benchmark._load_json(path, {"runs": []})


def test_upsert_result_rejects_invalid_rows_before_creating_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_dir = tmp_path / "phase0b_role_benchmark"
    monkeypatch.setattr(phase0b_role_benchmark, "EXPERIMENT_DIR", experiment_dir)
    monkeypatch.setattr(phase0b_role_benchmark, "RUNS_DIR", experiment_dir / "runs")
    monkeypatch.setattr(phase0b_role_benchmark, "ACTIVE_RUN_PATH", experiment_dir / "active.json")
    monkeypatch.setattr(
        phase0b_role_benchmark, "RESULTS_JSON_PATH", experiment_dir / "results.json"
    )
    monkeypatch.setattr(phase0b_role_benchmark, "RESULTS_CSV_PATH", experiment_dir / "results.csv")

    with pytest.raises(ValueError, match="worker_commit_count"):
        phase0b_role_benchmark._upsert_result(_result_row(worker_commit_count=2))

    assert not experiment_dir.exists()


def test_upsert_result_rejects_malformed_existing_results_without_overwrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_dir = tmp_path / "phase0b_role_benchmark"
    experiment_dir.mkdir()
    results_json = experiment_dir / "results.json"
    results_json.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(phase0b_role_benchmark, "EXPERIMENT_DIR", experiment_dir)
    monkeypatch.setattr(phase0b_role_benchmark, "RUNS_DIR", experiment_dir / "runs")
    monkeypatch.setattr(phase0b_role_benchmark, "ACTIVE_RUN_PATH", experiment_dir / "active.json")
    monkeypatch.setattr(phase0b_role_benchmark, "RESULTS_JSON_PATH", results_json)
    monkeypatch.setattr(phase0b_role_benchmark, "RESULTS_CSV_PATH", experiment_dir / "results.csv")

    with pytest.raises(ValueError, match=r"Malformed JSON file .*results\.json"):
        phase0b_role_benchmark._upsert_result(_result_row())

    assert results_json.read_text(encoding="utf-8") == "{not-json"
    assert not (experiment_dir / "results.csv").exists()


def test_build_result_row_includes_multi_branch_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    worktree = tmp_path / "bench-run"
    runtime_manifest_path = worktree / ".aragora" / "phase0b_runtime_manifest.yaml"
    receipt_path = tmp_path / "docs" / "receipts" / "phase0b-engine-hardening" / "B-6.yaml"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        yaml.safe_dump(
            {
                "worker_branch": "codex/swarm-subtask-1",
                "worker_commit": "def456",
                "worker_branches": [
                    "codex/swarm-subtask-1",
                    "codex/swarm-subtask-2",
                ],
                "worker_commits": ["abc123", "def456"],
                "changed_files": [
                    "docs/test.md",
                    "aragora/swarm/campaign.py",
                ],
                "duration_seconds": 321,
                "cost_usd": 3.0,
                "planner_strategy_requested": "model",
                "planner_strategy_used": "model",
                "planner_fallback_reason": None,
                "verification_missing_reason": None,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = CampaignManifest(
        campaign_id="phase0b-engine-hardening",
        created_at="2026-03-17T00:00:00+00:00",
        source_kind="test",
        source_ref="test",
        planner_model="codex",
        planner_strategy="model",
        worker_model="claude",
        review_model="claude",
        enforce_cross_model_review=False,
        experiment_id="exp-001",
        experiment_label="p-codex_w-claude_r-claude",
        projects=[
            CampaignProject(
                project_id="B-6",
                title="Engine hardening",
                spec=SwarmSpec(
                    raw_goal="goal",
                    refined_goal="goal",
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    file_scope_hints=["aragora/swarm/campaign.py"],
                ),
                status="completed",
                last_run_outcome="deliverable_created",
                receipt_id="docs/receipts/phase0b-engine-hardening/B-6.yaml",
            )
        ],
    )
    save_campaign_manifest(runtime_manifest_path, manifest)

    monkeypatch.setattr(phase0b_role_benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(phase0b_role_benchmark, "_lookup_pr", lambda branch: {})
    monkeypatch.setattr(phase0b_role_benchmark, "_lookup_ci_status", lambda pr_number: "")

    row = phase0b_role_benchmark.build_result_row(runtime_manifest_path)

    assert row["worker_branch"] == "codex/swarm-subtask-1"
    assert row["worker_commit"] == "def456"
    assert row["worker_branch_count"] == 2
    assert row["worker_commit_count"] == 2
    assert json.loads(row["worker_branches_json"]) == [
        "codex/swarm-subtask-1",
        "codex/swarm-subtask-2",
    ]
    assert json.loads(row["worker_commits_json"]) == ["abc123", "def456"]
    assert row["changed_files_count"] == 2
