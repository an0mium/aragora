from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aragora.swarm.tranche import TrancheManifest
from aragora.swarm.tranche_design_review import (
    DesignReviewRecord,
    load_design_review,
    run_design_review,
    save_design_review,
)


def _make_manifest() -> TrancheManifest:
    return TrancheManifest.from_dict(
        {
            "manifest_id": "pmf-tranche",
            "repo": {"name": "synaptent/aragora", "root": "/tmp/repo", "base_ref": "origin/main"},
            "references": {"source_refs": {}},
            "gates": {},
            "lanes": [
                {
                    "lane_id": "lane_a",
                    "owner_role": "engineer",
                    "title": "Build it",
                    "prompt": "Implement the feature",
                    "allowed_write_scope": ["aragora/server/**"],
                    "verification_commands": ["pytest -q"],
                }
            ],
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )


@pytest.mark.asyncio
async def test_design_review_runs_proposer_critic_and_synthesizer() -> None:
    proposer = AsyncMock(return_value={"proposal": {"objective": "ship it"}})
    critic = AsyncMock(return_value={"findings": ["Scope is too broad"], "grounded": True})
    synthesizer = AsyncMock(
        return_value={
            "recommendation": "awaiting_confirmation",
            "revised_manifest": {"manifest_id": "m1"},
            "unresolved_assumptions": ["Need narrower write scope"],
        }
    )

    result = await run_design_review(
        manifest=_make_manifest(),
        normalized_bundle={"objective": "ship it"},
        inspection={"preflight_status": "ok"},
        proposer_fn=proposer,
        critic_fn=critic,
        synthesizer_fn=synthesizer,
        max_rounds=2,
    )

    assert result["recommendation"] == "awaiting_confirmation"
    proposer.assert_awaited_once()
    critic.assert_awaited_once()
    synthesizer.assert_awaited_once()


@pytest.mark.asyncio
async def test_design_review_stops_after_two_rounds() -> None:
    proposer = AsyncMock(side_effect=[{"proposal": {"round": 1}}, {"proposal": {"round": 2}}])
    critic = AsyncMock(
        side_effect=[
            {"findings": ["Issue 1"], "grounded": True},
            {"findings": ["Issue 2"], "grounded": True},
        ]
    )
    synthesizer = AsyncMock(
        side_effect=[
            {
                "recommendation": "revise",
                "revised_manifest": {"round": 1},
                "unresolved_assumptions": [],
            },
            {
                "recommendation": "needs_human",
                "revised_manifest": {"round": 2},
                "unresolved_assumptions": ["Still disputed"],
            },
        ]
    )

    result = await run_design_review(
        manifest=_make_manifest(),
        normalized_bundle={"objective": "ship it"},
        inspection={"preflight_status": "ok"},
        proposer_fn=proposer,
        critic_fn=critic,
        synthesizer_fn=synthesizer,
        max_rounds=2,
    )

    assert result["rounds_completed"] == 2
    assert result["recommendation"] == "needs_human"


def test_design_review_record_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "design_review.yaml"
    record = DesignReviewRecord(
        manifest_id="pmf-tranche",
        status="awaiting_confirmation",
        rounds=[
            {
                "round": 1,
                "proposal": {"objective": "ship it"},
                "findings": ["Scope is too broad"],
            }
        ],
        proposed_manifest={"objective": "ship it"},
        critique_findings=["Scope is too broad"],
        revised_manifest={"objective": "ship it narrowly"},
        unresolved_assumptions=["Need narrower write scope"],
        recommendation="awaiting_confirmation",
    )

    save_design_review(path, record)
    loaded = load_design_review(path)

    assert loaded.manifest_id == "pmf-tranche"
    assert loaded.recommendation == "awaiting_confirmation"
    assert loaded.rounds[0]["round"] == 1
