"""Tests for the severity-count surfacing (#6505 fix #1).

The brief-engine pipeline computes severity info on every ``top_finding``
but drops it when the brief is persisted as JSON. This module pins the
behavior introduced to expose aggregate counts at the brief level so
operators can triage without reading every finding.

Scope is narrow on purpose: the aggregator, the degraded-input
behavior, and the pass-through from ``build_brief``. The weighted-vote
policy is unchanged in this PR and keeps its own tests in
``tests/review/test_builder.py``.
"""

from __future__ import annotations

from aragora.brief_engine.protocol import (
    SlotFindingsResponse,
    _severity_counts_from_slot_findings,
)
from aragora.review.builder import PanelVote, build_brief
from aragora.review.protocol import (
    DissentPosition,
    Recommendation,
    ReviewRole,
    RoleFinding,
    SynthesisPolicy,
)
from aragora.swarm.pr_review_protocol import PRReviewFinding


def _finding(fid: str, severity: str) -> PRReviewFinding:
    return PRReviewFinding(
        finding_id=fid,
        category="correctness",
        severity=severity,
        summary="test finding",
        evidence=[],
    )


def _slot_response(
    slot_id: str,
    top_findings: tuple[PRReviewFinding, ...],
) -> SlotFindingsResponse:
    return SlotFindingsResponse(
        slot_id=slot_id,
        provider="mock",
        model="mock-model",
        position=DissentPosition.REQUEST_CHANGES,
        confidence=0.8,
        summary="s",
        top_findings=top_findings,
        contested_finding_ids=(),
        reason="r",
    )


class TestSeverityCountsAggregator:
    def test_empty_inputs_return_zero_counts(self) -> None:
        assert _severity_counts_from_slot_findings({}) == {
            "high": 0,
            "medium": 0,
            "low": 0,
        }

    def test_counts_aggregated_across_slots(self) -> None:
        findings_by_slot = {
            "s1": _slot_response(
                "s1",
                top_findings=(_finding("f1", "high"), _finding("f2", "low")),
            ),
            "s2": _slot_response(
                "s2",
                top_findings=(_finding("f3", "medium"), _finding("f4", "high")),
            ),
        }
        assert _severity_counts_from_slot_findings(findings_by_slot) == {
            "high": 2,
            "medium": 1,
            "low": 1,
        }

    def test_unknown_severity_strings_are_dropped_not_counted(self) -> None:
        # The severity field is LLM-parsed; defensive coercion drops
        # unknowns rather than spraying junk keys into the counts dict.
        findings_by_slot = {
            "s1": _slot_response(
                "s1",
                top_findings=(
                    _finding("f1", "critical"),
                    _finding("f2", ""),
                    _finding("f3", "MEDIUM"),
                ),
            ),
        }
        counts = _severity_counts_from_slot_findings(findings_by_slot)
        assert counts == {"high": 0, "medium": 1, "low": 0}
        assert "critical" not in counts

    def test_case_insensitive_severity_matching(self) -> None:
        findings_by_slot = {
            "s1": _slot_response(
                "s1",
                top_findings=(
                    _finding("f1", "HIGH"),
                    _finding("f2", " Medium "),
                    _finding("f3", "low"),
                ),
            ),
        }
        assert _severity_counts_from_slot_findings(findings_by_slot) == {
            "high": 1,
            "medium": 1,
            "low": 1,
        }


class TestBuildBriefSeverityPassThrough:
    def _votes(self) -> tuple[PanelVote, ...]:
        finding = RoleFinding(
            role=ReviewRole.LOGIC,
            agent="slot-a:claude",
            model="mock",
            confidence=0.9,
            finding_text="ok",
        )
        return (
            PanelVote(
                finding=finding,
                position=DissentPosition.APPROVE,
                reason="fine",
            ),
        )

    def test_build_brief_passes_severity_counts_through(self) -> None:
        counts = {"high": 1, "medium": 2, "low": 0}
        brief = build_brief(
            votes=self._votes(),
            pr_number=1,
            repo="o/r",
            head_sha="head",
            base_sha="base",
            top_line="t",
            validation_summary="v",
            generated_at="2026-04-24T00:00:00+00:00",
            synthesis_policy=SynthesisPolicy.MAJORITY,
            findings_severity_counts=counts,
        )
        assert brief.findings_severity_counts == counts
        assert brief.to_dict()["findings_severity_counts"] == counts
        # Recommendation path is unchanged by this field — confirm.
        assert brief.recommendation is Recommendation.APPROVE_CANDIDATE

    def test_build_brief_omits_severity_counts_defaults_to_empty(self) -> None:
        # Backwards compatibility: callers that don't supply the new
        # param (legacy, degraded, partial test fixtures) must still
        # produce a valid brief with an empty severity map.
        brief = build_brief(
            votes=self._votes(),
            pr_number=1,
            repo="o/r",
            head_sha="head",
            base_sha="base",
            top_line="t",
            validation_summary="v",
            generated_at="2026-04-24T00:00:00+00:00",
            synthesis_policy=SynthesisPolicy.MAJORITY,
        )
        assert brief.findings_severity_counts == {}
        assert brief.to_dict()["findings_severity_counts"] == {}
