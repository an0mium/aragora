from __future__ import annotations

import pytest

from aragora.swarm.proof_first_queue import classify_proof_first_queue_issue
from aragora.swarm.roadmap_priority import RoadmapPriorityPolicy


_ROADMAP_POLICY = RoadmapPriorityPolicy(
    do_now=frozenset({"CS-01", "CS-02", "TW-02"}),
    delay=frozenset({"BC-07", "CS-04"}),
    avoid=frozenset({"EX-99"}),
)


@pytest.mark.parametrize(
    ("title", "body", "expected_codes"),
    [
        (
            "[CS-01] Reconcile docs status surfaces",
            "Keep published claims narrower than measured proof.",
            ("CS-01",),
        ),
        (
            "Add queue governance tests",
            "This protects CS-02 while proof-first automation evolves.",
            ("CS-02",),
        ),
    ],
)
def test_classifies_roadmap_do_now_lane(
    title: str, body: str, expected_codes: tuple[str, ...]
) -> None:
    decision = classify_proof_first_queue_issue(
        title,
        body,
        roadmap_policy=_ROADMAP_POLICY,
    )

    assert decision.allowed is True
    assert decision.lane == "roadmap_do_now"
    assert decision.roadmap_codes == expected_codes
    assert decision.blocked_codes == ()


@pytest.mark.parametrize(
    ("title", "body", "expected_blocked"),
    [
        (
            "[BC-07] Expand broad billing cleanup",
            "Delay this roadmap lane until current proof gates settle.",
            ("BC-07",),
        ),
        (
            "Prototype speculative external proof layer",
            "EX-99 is explicitly avoided in this tranche.",
            ("EX-99",),
        ),
    ],
)
def test_classifies_blocked_roadmap_lane(
    title: str, body: str, expected_blocked: tuple[str, ...]
) -> None:
    decision = classify_proof_first_queue_issue(
        title,
        body,
        labels=("boss-ready",),
        roadmap_policy=_ROADMAP_POLICY,
    )

    assert decision.allowed is False
    assert decision.lane == "blocked_roadmap_lane"
    assert decision.blocked_codes == expected_blocked


@pytest.mark.parametrize(
    ("title", "body", "expected_terms"),
    [
        (
            "[TW-02] Refresh benchmark truth artifact",
            "The corpus scorecard is stale after the latest publication.",
            ("tw-02",),
        ),
        (
            "Repair benchmark freshness scorecard",
            "Truth artifact metrics drifted from the generated corpus.",
            ("benchmark", "corpus", "scorecard", "truth artifact", "freshness"),
        ),
    ],
)
def test_classifies_benchmark_regression_lane(
    title: str, body: str, expected_terms: tuple[str, ...]
) -> None:
    decision = classify_proof_first_queue_issue(title, body)

    assert decision.allowed is True
    assert decision.lane == "benchmark_regression"
    assert set(expected_terms).issubset(decision.matched_terms)


@pytest.mark.parametrize(
    ("title", "body", "expected_terms"),
    [
        (
            "[TW-03] Productize repeated rescue class",
            "Link the rescue productization report to issue drafts.",
            ("tw-03",),
        ),
        (
            "Productize recurring rescue follow-up",
            "A repeated rescue class needs a productization action.",
            ("rescue", "productization", "repeated rescue class"),
        ),
    ],
)
def test_classifies_rescue_productization_lane(
    title: str, body: str, expected_terms: tuple[str, ...]
) -> None:
    decision = classify_proof_first_queue_issue(title, body)

    assert decision.allowed is True
    assert decision.lane == "rescue_productization"
    assert set(expected_terms).issubset(decision.matched_terms)


@pytest.mark.parametrize(
    ("title", "body", "expected_terms"),
    [
        (
            "Reconcile docs proof drift",
            "Status docs must align with truth metrics after measured changes.",
            ("docs", "status", "proof", "truth", "drift", "reconcile", "align"),
        ),
        (
            "External claims outrun measured proof",
            "Commercial positioning should be narrower than the current gate.",
            (
                "positioning",
                "commercial",
                "external claims",
                "measured",
                "narrower",
                "outrun",
            ),
        ),
    ],
)
def test_classifies_docs_proof_drift_lane(
    title: str, body: str, expected_terms: tuple[str, ...]
) -> None:
    decision = classify_proof_first_queue_issue(title, body)

    assert decision.allowed is True
    assert decision.lane == "docs_proof_drift"
    assert set(expected_terms).issubset(decision.matched_terms)


@pytest.mark.parametrize(
    ("title", "body"),
    [
        (
            "Replace silent exception swallowing in postgres_store.py",
            "Tighten exception hygiene in one module.",
        ),
        (
            "Polish dashboard hover states",
            "Adjust spacing and button affordances without proof-first scope.",
        ),
    ],
)
def test_classifies_non_canonical_lane(title: str, body: str) -> None:
    decision = classify_proof_first_queue_issue(title, body)

    assert decision.allowed is False
    assert decision.lane == "non_canonical"
    assert decision.blocked_codes == ()
