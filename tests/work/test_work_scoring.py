from __future__ import annotations

from aragora.work.models import WorkItem
from aragora.work.scoring import build_recommendations, classify_work_item, score_work_item


def test_score_prefers_current_non_draft_pr_with_tests() -> None:
    item = WorkItem(
        id="pr:1",
        source="github_pr",
        item_type="pull_request",
        title="fix(queue): repair proof health",
        status="open",
        branch="codex/fix",
        updated_at="2026-05-16T00:00:00+00:00",
        metadata={
            "is_draft": False,
            "review_decision": "REVIEW_REQUIRED",
            "files": ["aragora/foo.py", "tests/test_foo.py"],
        },
    )

    score = score_work_item(item)

    assert score.readiness >= 0.6
    assert score.impact > 0.7
    assert score.test_obligation >= 0.8
    assert 0.0 <= score.total <= 1.0


def test_branch_only_has_less_owner_clarity_than_real_owner() -> None:
    branch_only = WorkItem(
        id="pr:1",
        source="github_pr",
        item_type="pull_request",
        title="fix(queue): repair proof health",
        status="open",
        branch="codex/fix",
    )
    owned = WorkItem(
        id="pr:2",
        source="github_pr",
        item_type="pull_request",
        title="fix(queue): repair proof health",
        status="open",
        branch="codex/fix",
        owner="codex-owner",
    )

    branch_score = score_work_item(branch_only)
    owner_score = score_work_item(owned)

    assert branch_score.owner_clarity == 0.62
    assert owner_score.owner_clarity == 0.9
    assert owner_score.owner_clarity > branch_score.owner_clarity
    assert "branch is explicit but owner is unknown" in branch_score.rationale
    assert "owner is explicit" in owner_score.rationale


def test_score_penalizes_stale_terminal_bead() -> None:
    item = WorkItem(
        id="bead:old",
        source="bead",
        item_type="bead",
        title="Old task",
        status="completed",
        updated_at="2026-01-01T00:00:00+00:00",
    )

    score = score_work_item(item)

    assert score.readiness < 0.3
    assert score.staleness < 0.2


def test_recommendations_are_ranked_by_total_score() -> None:
    low = WorkItem(id="mission:x", source="mission_file", item_type="mission", title="Context")
    high = WorkItem(
        id="automation-outbox:x",
        source="automation_outbox",
        item_type="handoff",
        title="repair queue health",
        status="pending",
        branch="codex/repair",
    )

    recommendations = build_recommendations([low, high])

    assert recommendations[0].item_id == "automation-outbox:x"
    assert recommendations[0].action == "publish_or_reconcile_handoff"
    assert recommendations[0].classification == "needs-polish"


def test_bead_missing_acceptance_criteria_needs_polish() -> None:
    item = WorkItem(
        id="bead:rough",
        source="bead",
        item_type="bead",
        title="Repair bridge session capture",
        status="pending",
        owner="claude",
        metadata={
            "objective": "Make broker sessions visible to operator-snapshot",
            "context": "Historical transcripts currently pollute live truth",
            "mutation_boundary": "read-only discovery code",
            "validation": "focused bridge tests",
        },
    )

    score = score_work_item(item)
    classification, blockers = classify_work_item(item, score=score)

    assert score.bead_quality < 0.7
    assert classification == "needs-polish"
    assert "acceptance criteria missing" in blockers


def test_complete_bounded_bead_is_ready() -> None:
    item = WorkItem(
        id="bead:polished",
        source="bead",
        item_type="bead",
        title="Repair bridge session capture",
        status="pending",
        owner="claude",
        dependencies=[],
        metadata={
            "objective": "Make broker sessions visible to operator-snapshot",
            "context": "Historical transcripts currently pollute live truth",
            "acceptance_criteria": ["operator-snapshot shows active broker runs"],
            "mutation_boundary": "read-only discovery code",
            "validation": "focused bridge tests",
            "dependencies_declared": True,
        },
    )

    score = score_work_item(item)
    classification, blockers = classify_work_item(item, score=score)

    assert score.bead_quality >= 0.8
    assert classification == "ready"
    assert blockers == []


def test_human_gated_pr_is_not_auto_ready() -> None:
    item = WorkItem(
        id="pr:99",
        source="github_pr",
        item_type="pull_request",
        title="feat: modify merge authority",
        status="open",
        branch="codex/gate-change",
        metadata={
            "is_draft": False,
            "review_decision": "APPROVED",
            "human_gate": True,
            "tier": 4,
        },
    )

    recommendations = build_recommendations([item])

    assert recommendations[0].classification == "human-gated"
    assert "human-gated risk surface" in recommendations[0].blockers
