from __future__ import annotations

import json

import pytest

from aragora.swarm.proof_first_queue import classify_proof_first_queue_issue
from aragora.swarm.roadmap_priority import RoadmapPriorityPolicy


_ROADMAP_POLICY = RoadmapPriorityPolicy(
    do_now=frozenset({"CS-01", "CS-02", "TW-02"}),
    delay=frozenset({"BC-07", "CS-04"}),
    avoid=frozenset({"EX-99"}),
)


def _write_strategy_mission_register(tmp_path, *, status: str = "queued") -> None:
    register_path = tmp_path / "docs" / "status" / "ROADMAP_INTAKE_REGISTER.md"
    register_path.parent.mkdir(parents=True, exist_ok=True)
    register_path.write_text(
        "\n".join(
            [
                "# Roadmap Intake Register",
                "",
                "## Strategy Mission Queue",
                "",
                "| id | title | tier | status | external-proof gate | tracking |",
                "|---|---|---|---|---|---|",
                (
                    "| M1 | ODR v1.0 GA (docs + verification only) | 0-1 | "
                    f"{status} | aragora-verify checks the example | Epic #8665 |"
                ),
                "",
                "## Maintenance",
                "",
            ]
        )
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


def test_subdecomposed_issue_uses_inherited_parent_roadmap_code() -> None:
    decision = classify_proof_first_queue_issue(
        "[CS-01b] Reconcile docs status surface",
        "## Decomposition Lineage\n- Inherited roadmap codes: CS-01\n",
        roadmap_policy=_ROADMAP_POLICY,
    )

    assert decision.allowed is True
    assert decision.lane == "roadmap_do_now"
    assert decision.roadmap_codes == ("CS-01",)


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
    ("title", "body", "expected_term"),
    [
        (
            "Strategy mission cadence: reconcile proof docs for Epic #8665",
            "Status docs must align with proof drift before the strategy mission queue moves.",
            "#8665",
        ),
        (
            "Open Strategy Mission Queue worker",
            "Roadmap proof drift should stay queued until the prior external gate verifies.",
            "strategy mission queue",
        ),
        (
            "Strategy mission intake for issues/8665",
            "Proof drift stays planning truth until the queue gate opens.",
            "8665",
        ),
        (
            "Epic 8665 proof-drift candidate",
            "Docs proof drift wording should not bypass the strategy-mission gate.",
            "8665",
        ),
    ],
)
def test_strategy_mission_tracking_stays_out_of_boss_ready(
    title: str, body: str, expected_term: str
) -> None:
    decision = classify_proof_first_queue_issue(
        title,
        body,
        labels=("boss-ready",),
    )

    assert decision.allowed is False
    assert decision.lane == "strategy_mission_gated"
    assert expected_term in decision.matched_terms
    assert "docs_proof_drift" != decision.lane


def test_strategy_mission_queue_row_stays_blocked_until_register_marks_active(tmp_path) -> None:
    _write_strategy_mission_register(tmp_path, status="queued")

    decision = classify_proof_first_queue_issue(
        "M1 ODR v1.0 GA proof drift worker for Epic #8665",
        "The strategy mission queue row is still queued, so boss-ready is premature.",
        labels=("boss-ready",),
        repo_root=tmp_path,
    )

    assert decision.allowed is False
    assert decision.lane == "strategy_mission_gated"


def test_strategy_mission_numeric_reference_requires_token_boundary() -> None:
    decision = classify_proof_first_queue_issue(
        "Investigate unrelated issue #18665 before promotion",
        "A substring-only numeric reference must not trip the strategy mission gate.",
        labels=("boss-ready",),
    )

    assert decision.allowed is False
    assert decision.lane == "non_canonical"
    assert "8665" not in decision.matched_terms


def test_strategy_mission_active_row_refreshes_after_register_edit(tmp_path) -> None:
    _write_strategy_mission_register(tmp_path, status="queued")

    first_decision = classify_proof_first_queue_issue(
        "M1 ODR v1.0 GA proof drift worker for Epic #8665",
        "The strategy mission queue row starts queued.",
        labels=("boss-ready",),
        repo_root=tmp_path,
    )
    assert first_decision.allowed is False
    assert first_decision.lane == "strategy_mission_gated"

    _write_strategy_mission_register(tmp_path, status="active")

    second_decision = classify_proof_first_queue_issue(
        "M1 ODR v1.0 GA proof drift worker for Epic #8665",
        "The same process must see the register row become active.",
        labels=("boss-ready",),
        repo_root=tmp_path,
    )
    assert second_decision.allowed is True
    assert second_decision.lane == "strategy_mission_active_row"


def test_strategy_mission_active_queue_row_can_enter_boss_ready(tmp_path) -> None:
    _write_strategy_mission_register(tmp_path, status="active")

    decision = classify_proof_first_queue_issue(
        "M1 ODR v1.0 GA proof drift worker for Epic #8665",
        "The specifically opened strategy mission queue row is active.",
        labels=("boss-ready",),
        repo_root=tmp_path,
    )

    assert decision.allowed is True
    assert decision.lane == "strategy_mission_active_row"
    assert "active:m1" in decision.matched_terms


def test_staged_rev4_corpus_precedence_beats_strategy_mission_gate(tmp_path) -> None:
    corpus_path = tmp_path / "tests" / "benchmarks" / "corpus_rev4.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(json.dumps({"issues": [{"issue_id": 5788}]}))

    decision = classify_proof_first_queue_issue(
        "Epic #8665 should not hide an explicit staged rev4 issue",
        "Single-file exception hygiene task.",
        labels=("autonomous", "boss-ready"),
        issue_number=5788,
        repo_root=tmp_path,
    )

    assert decision.allowed is True
    assert decision.lane == "staged_rev4_corpus"


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


def test_classifies_explicit_staged_rev4_corpus_issue(tmp_path) -> None:
    corpus_path = tmp_path / "tests" / "benchmarks" / "corpus_rev4.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(json.dumps({"issues": [{"issue_id": 5788}]}))

    decision = classify_proof_first_queue_issue(
        "Narrow broad except Exception in performance_monitor.py",
        "Single-file exception hygiene task.",
        labels=("autonomous", "boss-ready"),
        issue_number=5788,
        repo_root=tmp_path,
    )

    assert decision.allowed is True
    assert decision.lane == "staged_rev4_corpus"
    assert "corpus_rev4" in decision.matched_terms


def test_staged_rev4_corpus_issue_requires_explicit_boss_ready(tmp_path) -> None:
    corpus_path = tmp_path / "tests" / "benchmarks" / "corpus_rev4.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(json.dumps({"issues": [{"issue_id": 5788}]}))

    decision = classify_proof_first_queue_issue(
        "Narrow broad except Exception in performance_monitor.py",
        "Single-file exception hygiene task.",
        labels=("autonomous",),
        issue_number=5788,
        repo_root=tmp_path,
    )

    assert decision.allowed is False
    assert decision.lane == "non_canonical"


def test_staged_rev4_corpus_issue_requires_membership(tmp_path) -> None:
    corpus_path = tmp_path / "tests" / "benchmarks" / "corpus_rev4.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(json.dumps({"issues": [{"issue_id": 5788}]}))

    decision = classify_proof_first_queue_issue(
        "Narrow broad except Exception in unrelated.py",
        "Single-file exception hygiene task.",
        labels=("autonomous", "boss-ready"),
        issue_number=9999,
        repo_root=tmp_path,
    )

    assert decision.allowed is False
    assert decision.lane == "non_canonical"


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
