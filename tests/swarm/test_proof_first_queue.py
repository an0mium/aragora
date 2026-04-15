from __future__ import annotations

from aragora.swarm.proof_first_queue import classify_proof_first_queue_issue
from aragora.swarm.roadmap_priority import RoadmapPriorityPolicy


def test_classify_proof_first_queue_issue_allows_do_now_roadmap_lane() -> None:
    decision = classify_proof_first_queue_issue(
        "[CS-01..03] Reconcile docs/status surfaces to current proof",
        "Keep roadmap, status, and positioning docs narrower than measured proof.",
        roadmap_policy=RoadmapPriorityPolicy(
            do_now=frozenset({"CS-01", "CS-02", "CS-03"}),
            delay=frozenset({"BC-07"}),
            avoid=frozenset({"CS-04"}),
        ),
    )

    assert decision.allowed is True
    assert decision.lane == "roadmap_do_now"


def test_classify_proof_first_queue_issue_allows_tw02_benchmark_follow_up() -> None:
    decision = classify_proof_first_queue_issue(
        "[TW-02] Restock stale issues in tw-01-bounded-execution-v1 rev-1",
        "Refresh benchmark corpus freshness after stale closed issues were detected in the truth artifact.",
    )

    assert decision.allowed is True
    assert decision.lane == "benchmark_regression"


def test_classify_proof_first_queue_issue_allows_tw03_rescue_follow_up() -> None:
    decision = classify_proof_first_queue_issue(
        "[TW-03] Productize repeated rescue class: blocked-auth-failure",
        "Repeated rescue class productization follow-up for rescue ledger drift.",
    )

    assert decision.allowed is True
    assert decision.lane == "rescue_productization"


def test_classify_proof_first_queue_issue_blocks_generic_cleanup() -> None:
    decision = classify_proof_first_queue_issue(
        "Replace silent exception swallowing in postgres_store.py",
        "Tighten exception hygiene in a single module.",
    )

    assert decision.allowed is False
    assert decision.lane == "non_canonical"
