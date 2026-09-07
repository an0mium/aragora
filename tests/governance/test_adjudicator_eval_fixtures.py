"""Smoke tests for the adjudicator eval fixtures (#8856 item 3).

``tests/governance/fixtures/adjudicator_eval_cases.json`` distills REAL
multi-model review rounds (synaptent/aragora, 2026-07-02..05) into
EvidenceItem-shaped cases: verbatim reviewer bodies at an exact head SHA, the
adjudicator's expected CURRENT verdict, and the human-settled ground-truth
disposition per the #8748 SETTLE/BLOCK/ESCALATE taxonomy. Narrative and
receipts: ``docs/artifacts/2026-07-reviewer-failure-taxonomy.md``.

These tests pin two different things — deliberately:

1. **Current behavior** (``expected_adjudicator_verdict``): what M0
   ``adjudicate()`` returns today on each case. If the adjudicator changes,
   these assertions localize exactly which real-world case changed verdict.
2. **Safety invariants**: advisory-only dissent must never produce BLOCK, and
   an unrefuted [P1] must always produce BLOCK, regardless of scorer behavior.

Where ``expected_adjudicator_verdict`` differs from
``ground_truth.disposition`` (e.g. the verbatim-repeat case: ESCALATE today,
SETTLE per the operator record), the delta is the documented eval target for
adjudicator improvements — NOT a bug in these tests. Do not "fix" the fixture
to make the numbers agree; improve the adjudicator and update the expected
verdict with the receipts.

The default groundedness scorer imports ``aragora_debate`` (same hard test
dependency as ``tests/swarm/test_review_adjudicator.py``); only the
scorer-dependent cases require it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from aragora.swarm.review_adjudicator import AdjudicationVerdict, adjudicate

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "adjudicator_eval_cases.json"

# Cases whose expected verdict depends on the groundedness scorer (and thus the
# aragora_debate package). Hard-bar BLOCK and NOT_APPLICABLE verdicts are
# decided before the scorer is ever built, so they carry no such dependency.
SCORER_DEPENDENT_VERDICTS = {"adjudicated_settle", "adjudicated_escalate"}

VALID_DISPOSITIONS = {"settle", "block", "escalate"}
VALID_VERDICTS = {v.value for v in AdjudicationVerdict}


@dataclass
class _Item:
    """Minimal EvidenceItem-shaped stand-in (family, body, verdict, supportive)."""

    family: str
    body: str
    verdict: str

    @property
    def supportive(self) -> bool:
        return self.verdict == "pass"


def _load_cases() -> list[dict[str, Any]]:
    doc = json.loads(FIXTURE_PATH.read_text())
    assert doc["schema"] == "adjudicator_eval_cases.v1"
    return doc["cases"]


def _items(case: dict[str, Any]) -> list[_Item]:
    return [_Item(i["family"], i["body"], i["verdict"]) for i in case["items"]]


def _case(case_id: str) -> dict[str, Any]:
    matches = [c for c in _load_cases() if c["id"] == case_id]
    assert len(matches) == 1, f"expected exactly one case {case_id!r}"
    return matches[0]


class TestFixtureShape:
    def test_fixture_parses_and_case_fields_are_complete(self) -> None:
        cases = _load_cases()
        assert len(cases) >= 9
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids)), "case ids must be unique"
        for case in cases:
            assert case["pr"] > 0
            assert len(case["head_sha"]) == 40
            assert case["expected_adjudicator_verdict"] in VALID_VERDICTS
            assert case["ground_truth"]["disposition"] in VALID_DISPOSITIONS
            assert case["ground_truth"]["receipts"], case["id"]
            assert all(
                r.startswith("https://github.com/synaptent/aragora/")
                for r in case["ground_truth"]["receipts"]
            ), case["id"]
            assert len(case["items"]) >= 2
            for item in case["items"]:
                assert item["family"] in {"claude", "openai"}
                assert item["verdict"] in {"pass", "changes_requested"}
                assert item["body"].strip(), case["id"]

    def test_taxonomy_covers_failure_classes_and_controls(self) -> None:
        cases = _load_cases()
        classes = {fc for c in cases for fc in c["failure_classes"]}
        assert classes == {
            "diff_blind_grounding",
            "stale_external_world",
            "temporal_reasoning",
            "verbatim_repeat_dissent",
            "out_of_scope_carousel",
            "cross_family_contradiction",
        }
        assert sum(1 for c in cases if c["control"]) >= 3


class TestAdjudicatorVerdictsOnRealCases:
    """Pin the CURRENT M0 verdict for every fixture case."""

    @pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
    def test_expected_verdict_matches(self, case: dict[str, Any]) -> None:
        if case["expected_adjudicator_verdict"] in SCORER_DEPENDENT_VERDICTS:
            pytest.importorskip("aragora_debate")
        result = adjudicate(_items(case))
        assert result.verdict.value == case["expected_adjudicator_verdict"], case["id"]


class TestClearCaseInvariants:
    """The named clear cases from the taxonomy artifact, asserted individually."""

    def test_unrefuted_p1_blocks_without_scorer(self) -> None:
        # pr8824_r1: both dead-link findings were FALSE (the file existed on
        # main, blob 3c0458f1), but at this snapshot the [P1] was unrefuted —
        # BLOCK is the correct fail-safe verdict, reached via the hard bar with
        # the scorer never invoked (a raising scorer must not change it).
        def _raising_scorer(_body: str) -> float:
            raise AssertionError("hard-bar path must not invoke the scorer")

        result = adjudicate(_items(_case("pr8824_r1_diff_blind_grounding")), scorer=_raising_scorer)
        assert result.verdict is AdjudicationVerdict.BLOCK
        assert result.blocking_findings

    def test_real_convergent_p1_class_blocks(self) -> None:
        # pr8834_r1 carries the [P1] hard bar; same deterministic BLOCK path.
        result = adjudicate(_items(_case("pr8834_r1_temporal_and_stale_external")))
        assert result.verdict is AdjudicationVerdict.BLOCK

    def test_cross_family_contradiction_escalates(self) -> None:
        pytest.importorskip("aragora_debate")
        # pr8802_r7: claude's round-5 [P2] REQUIRED the verified_any-wins
        # precedence; openai's round-7 [P2] objected to exactly that ordering.
        # A genuine two-sided crux — ESCALATE, matching the human disposition.
        result = adjudicate(_items(_case("pr8802_r7_cross_family_contradiction")))
        assert result.verdict is AdjudicationVerdict.ESCALATE
        assert result.escalated_findings

    def test_verbatim_repeat_answered_never_blocks(self) -> None:
        pytest.importorskip("aragora_debate")
        # pr8800_r3: openai re-raised its answered round-1 finding; the operator
        # record settles it (ground truth = settle). The current heuristic
        # escalates (it cannot see that the finding was answered in-thread) —
        # fail-safe to a human, and the documented eval gap. The invariant that
        # must hold under any future scorer: advisory [P2] dissent with a
        # counting PASS present resolves toward settlement (SETTLE or a human
        # via ESCALATE), never an autonomous BLOCK.
        case = _case("pr8800_r3_verbatim_repeat_answered")
        result = adjudicate(_items(case))
        assert result.verdict in (AdjudicationVerdict.SETTLE, AdjudicationVerdict.ESCALATE)
        assert result.verdict is not AdjudicationVerdict.BLOCK
        assert result.verdict.value == case["expected_adjudicator_verdict"]
        assert case["ground_truth"]["disposition"] == "settle"

    def test_unanimous_rejection_and_clean_pass_are_not_stalls(self) -> None:
        # Control group: real convergent findings (both CR) and clean 2-0 PASS
        # are not stalls — the adjudicator abstains and the base gate decides.
        for case_id in (
            "pr8811_r1_convergent_real_finding",
            "pr8852_r1_convergent_real_findings",
            "pr8803_r1_clean_convergent_pass",
            "pr8824_r3_diff_blind_unanimous",
        ):
            result = adjudicate(_items(_case(case_id)))
            assert result.verdict is AdjudicationVerdict.NOT_APPLICABLE, case_id
