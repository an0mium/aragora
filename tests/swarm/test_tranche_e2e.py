from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from aragora.swarm.campaign import CampaignReviewGate
from aragora.swarm.pr_registry import PullRequestRegistry
from aragora.swarm.tranche import TrancheArtifactStore, TrancheLaneArtifact, TrancheManifest
from aragora.swarm.tranche_design_review import (
    DesignReviewRecord,
    load_design_review,
    run_design_review,
    save_design_review,
)
from aragora.swarm.tranche_integrate import (
    assess_lane_integration,
    discover_lane_pr,
    record_lane_integration,
    register_pr,
)
from aragora.swarm.tranche_review import review_lane
from aragora.swarm.tranche_submit import submit_intake_bundle
from aragora.swarm.tranche_watch import load_tranche_run_state, refresh_tranche_state


@pytest.mark.asyncio
async def test_tranche_lifecycle_e2e(tmp_path: Path) -> None:
    bundle = {
        "objective": "Test e2e flow",
        "candidate_lanes": [
            {
                "lane_id": "e2e_lane",
                "title": "E2E test lane",
                "prompt": "Do the thing",
                "owner_role": "engineer",
                "allowed_write_scope": ["tests/**"],
                "verification_commands": ["echo ok"],
            }
        ],
        "autonomy_mode": "checkpoint",
    }

    submit_result = submit_intake_bundle(
        bundle,
        repo_root=tmp_path,
        skip_github_resolution=True,
    )

    assert submit_result["inspection_status"] == "ok"
    assert submit_result["submission_status"] == "awaiting_confirmation"
    assert submit_result["recommended_action"] == "design-review"

    manifest_path = Path(submit_result["manifest_path"])
    tranche_dir = Path(submit_result["tranche_dir"])
    normalized_path = tranche_dir / "normalized_bundle.yaml"

    state = load_tranche_run_state(manifest_path)
    assert state.lane_states["e2e_lane"].status == "pending"

    manifest = TrancheManifest.from_text(manifest_path.read_text(encoding="utf-8"))
    normalized_bundle = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))

    design_review = await run_design_review(
        manifest=manifest,
        normalized_bundle=normalized_bundle,
        inspection={"preflight_status": submit_result["inspection_status"]},
    )

    assert design_review["recommendation"] == "approved"
    assert design_review["rounds_completed"] == 1

    design_review_path = tranche_dir / "design_review.yaml"
    save_design_review(
        design_review_path,
        DesignReviewRecord.from_dict(design_review["record"]),
    )
    saved_review = load_design_review(design_review_path)
    assert saved_review.recommendation == "approved"

    artifact = TrancheLaneArtifact(
        lane_id="e2e_lane",
        source_ref="lane:e2e_lane",
        status="completed",
        run_id="run-e2e",
        metadata={
            "receipt_id": "receipt-e2e",
            "lease_id": "lease-e2e",
            "deliverable": {
                "branch": "feat/e2e-lane",
                "pr_url": "https://github.com/org/repo/pull/42",
            },
        },
    )
    artifact_store = TrancheArtifactStore(repo_root=tmp_path)
    artifact_store.save(manifest.manifest_id, artifact)

    refreshed = refresh_tranche_state(state, artifact_store=artifact_store)
    lane_state = refreshed.lane_states["e2e_lane"]
    assert lane_state.status == "completed"
    assert lane_state.run_id == "run-e2e"
    assert lane_state.receipt_id == "receipt-e2e"
    assert lane_state.lease_id == "lease-e2e"

    reviewer = AsyncMock()
    reviewer.review.return_value = CampaignReviewGate(
        status="passed",
        findings=[],
        review_model="claude",
    )
    review_result = await review_lane(
        manifest=manifest,
        lane_id="e2e_lane",
        artifact=artifact,
        run_dict={
            "run_id": "run-e2e",
            "status": "completed",
            "work_orders": [
                {
                    "receipt_id": "receipt-e2e",
                    "lease_id": "lease-e2e",
                    "worktree_path": str(tmp_path / "worktrees" / "e2e_lane"),
                }
            ],
        },
        reviewer=reviewer,
        tier=1,
        repo_root=tmp_path,
    )

    assert review_result["status"] == "passed"
    reviewer.review.assert_awaited_once()

    registry = MagicMock(spec=PullRequestRegistry)
    pr_url = discover_lane_pr(artifact)
    assert pr_url == "https://github.com/org/repo/pull/42"

    register_pr(pr_url, "feat/e2e-lane", registry)
    registry.register.assert_called_once_with(
        "feat/e2e-lane",
        "https://github.com/org/repo/pull/42",
        creator="tranche-integrate",
    )

    integration = assess_lane_integration(
        artifact=artifact,
        checks="checks_passed",
        review_status=review_result["status"],
        merge_policy="auto",
        autonomy_mode="checkpoint",
    )

    assert integration["recommendation"] == "merge"
    assert integration["executed"] is False

    coordination_store = MagicMock()
    await record_lane_integration(
        artifact=artifact,
        decision="merge",
        rationale=integration["rationale"],
        decided_by="tranche-e2e",
        store=coordination_store,
        target_branch="main",
    )
    coordination_store.record_integration_decision.assert_called_once()
