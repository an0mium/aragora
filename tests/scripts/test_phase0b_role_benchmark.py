"""Tests for scripts/phase0b_role_benchmark.py result capture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from aragora.swarm.campaign import CampaignManifest, CampaignProject, save_campaign_manifest
from aragora.swarm.spec import SwarmSpec

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import phase0b_role_benchmark  # noqa: E402


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
