"""Tests for reviewer-output execution schema validation."""

from __future__ import annotations

import json

import pytest

from aragora.review import (
    EvidenceKind,
    EvidenceRef,
    FindingCategory,
    FindingSeverity,
    Recommendation,
    REVIEWER_OUTPUT_SCHEMA_VERSION,
    ReviewerFinding,
    ReviewerOutput,
    validate_reviewer_outputs,
)


class TestEnums:
    def test_finding_category_values_match_execution_design(self) -> None:
        assert FindingCategory.LOGIC.value == "logic"
        assert FindingCategory.SECURITY.value == "security"
        assert FindingCategory.MAINTAINABILITY.value == "maintainability"
        assert FindingCategory.SKEPTIC.value == "skeptic"
        assert FindingCategory.VALIDATION.value == "validation"

    def test_finding_severity_values(self) -> None:
        assert FindingSeverity.LOW.value == "low"
        assert FindingSeverity.MEDIUM.value == "medium"
        assert FindingSeverity.HIGH.value == "high"


class TestReviewerFinding:
    def test_to_dict_serializes_enums(self) -> None:
        finding = ReviewerFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            claim="Potential auth bypass on missing org membership check.",
            evidence=("membership guard absent in handler",),
            files=("aragora/server/handlers/auth.py",),
        )
        payload = finding.to_dict()
        assert payload["category"] == "security"
        assert payload["severity"] == "high"
        assert payload["files"] == ["aragora/server/handlers/auth.py"]

    def test_from_dict_round_trip(self) -> None:
        finding = ReviewerFinding.from_dict(
            {
                "category": "logic",
                "severity": "medium",
                "claim": "Branch misses stale-head invalidation case.",
                "evidence": ["head_sha not compared during cache reuse"],
                "files": ["aragora/review/cache.py"],
            }
        )
        assert finding.category is FindingCategory.LOGIC
        assert finding.files == ("aragora/review/cache.py",)
        assert json.loads(json.dumps(finding.to_dict()))["evidence"] == [
            "head_sha not compared during cache reuse"
        ]

    def test_non_validation_findings_require_files(self) -> None:
        with pytest.raises(ValueError, match="files must contain at least one path"):
            ReviewerFinding.from_dict(
                {
                    "category": "security",
                    "severity": "high",
                    "claim": "Something is wrong.",
                    "evidence": ["trust boundary crossed"],
                    "files": [],
                }
            )

    def test_validation_findings_can_omit_files(self) -> None:
        finding = ReviewerFinding.from_dict(
            {
                "category": "validation",
                "severity": "medium",
                "claim": "CI still pending on current head.",
                "evidence": ["test-fast (server) pending"],
            }
        )
        assert finding.files == ()

    def test_claim_and_evidence_are_required(self) -> None:
        with pytest.raises(ValueError, match="claim is required"):
            ReviewerFinding.from_dict(
                {
                    "category": "logic",
                    "severity": "low",
                    "claim": "",
                    "evidence": ["something"],
                    "files": ["aragora/x.py"],
                }
            )
        with pytest.raises(ValueError, match="evidence must contain at least one item"):
            ReviewerFinding.from_dict(
                {
                    "category": "logic",
                    "severity": "low",
                    "claim": "Something.",
                    "evidence": [],
                    "files": ["aragora/x.py"],
                }
            )


class TestReviewerOutput:
    def _output(self, **overrides) -> ReviewerOutput:
        defaults = dict(
            reviewer_id="claude_core",
            slot_id="logic",
            provider="claude",
            lens="core",
            family="claude",
            recommendation_class=Recommendation.NEEDS_HUMAN_ATTENTION,
            confidence=0.62,
            summary="Flags one stale-cache risk and otherwise bounded changes.",
            top_findings=(
                ReviewerFinding(
                    category=FindingCategory.LOGIC,
                    severity=FindingSeverity.MEDIUM,
                    claim="Cache invalidation misses head SHA drift.",
                    evidence=("cache key omits head sha",),
                    files=("aragora/review/cache.py",),
                ),
            ),
            evidence_refs=(
                EvidenceRef(
                    kind=EvidenceKind.FILE,
                    path="aragora/review/cache.py",
                    line_range=(10, 24),
                    quote="cache key = repo/pr/base only",
                ),
            ),
            risk_flags=("stale_cache",),
            open_questions=("Should cache reuse require exact head SHA?",),
            round_index=1,
            latency_ms=1200,
            cost_usd=0.18,
        )
        defaults.update(overrides)
        return ReviewerOutput(**defaults)

    def test_to_dict_serializes_nested_contract(self) -> None:
        output = self._output()
        payload = output.to_dict()
        assert payload["schema_version"] == REVIEWER_OUTPUT_SCHEMA_VERSION
        assert payload["recommendation_class"] == "needs_human_attention"
        assert payload["top_findings"][0]["category"] == "logic"
        assert payload["evidence_refs"][0]["kind"] == "file"

    def test_from_dict_normalizes_and_validates(self) -> None:
        output = ReviewerOutput.from_dict(
            {
                "reviewer_id": "gpt_core",
                "slot_id": "security",
                "provider": "openai-api",
                "lens": "core",
                "family": "gpt",
                "recommendation_class": "repair_first",
                "confidence": 0.91,
                "summary": "Auth check missing on one route.",
                "top_findings": [
                    {
                        "category": "security",
                        "severity": "high",
                        "claim": "Route omits authorization guard.",
                        "evidence": ["guard decorator absent"],
                        "files": ["aragora/server/handlers/runs.py"],
                    }
                ],
                "evidence_refs": [
                    {
                        "kind": "file",
                        "path": "aragora/server/handlers/runs.py",
                        "line_range": [42, 65],
                        "quote": "@router.get('/runs')",
                    }
                ],
                "risk_flags": ["authz_gap", "authz_gap", ""],
                "open_questions": ["Should this route require org admin?"],
                "round_index": 2,
                "latency_ms": 880,
                "cost_usd": 0.22,
            }
        )
        assert output.recommendation_class is Recommendation.REPAIR_FIRST
        assert output.risk_flags == ("authz_gap",)
        assert output.round_index == 2

    def test_json_roundtrip(self) -> None:
        output = self._output()
        roundtrip = json.loads(json.dumps(output.to_dict()))
        assert roundtrip["reviewer_id"] == "claude_core"
        assert roundtrip["top_findings"][0]["claim"] == "Cache invalidation misses head SHA drift."
        assert roundtrip["evidence_refs"][0]["path"] == "aragora/review/cache.py"

    def test_validate_rejects_missing_required_fields(self) -> None:
        output = self._output(summary="")
        with pytest.raises(ValueError, match="missing required fields: summary"):
            output.validate()

    def test_validate_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            self._output(confidence=1.2).validate()

    def test_validate_rejects_missing_findings_or_evidence_refs(self) -> None:
        with pytest.raises(ValueError, match="top_findings must contain at least one item"):
            self._output(top_findings=()).validate()
        with pytest.raises(ValueError, match="evidence_refs must contain at least one item"):
            self._output(evidence_refs=()).validate()

    def test_validate_rejects_negative_execution_metrics(self) -> None:
        with pytest.raises(ValueError, match="round_index must be >= 1"):
            self._output(round_index=0).validate()
        with pytest.raises(ValueError, match="latency_ms must be >= 0"):
            self._output(latency_ms=-1).validate()
        with pytest.raises(ValueError, match="cost_usd must be >= 0"):
            self._output(cost_usd=-0.1).validate()

    def test_validate_rejects_wrong_schema_version(self) -> None:
        with pytest.raises(ValueError, match="schema_version must be"):
            self._output(schema_version="reviewer_output.v0").validate()

    def test_sequence_fields_are_immutable_tuples(self) -> None:
        output = self._output()
        assert isinstance(output.top_findings, tuple)
        assert isinstance(output.evidence_refs, tuple)
        assert isinstance(output.risk_flags, tuple)
        assert isinstance(output.open_questions, tuple)
        with pytest.raises(AttributeError):
            output.risk_flags.append("another")  # type: ignore[attr-defined]


class TestReviewerOutputBatchValidation:
    def test_batch_validation_requires_at_least_one_output(self) -> None:
        with pytest.raises(ValueError, match="at least one reviewer output is required"):
            validate_reviewer_outputs([])

    def test_batch_validation_rejects_duplicate_reviewer_round_pairs(self) -> None:
        output = ReviewerOutput.from_dict(
            {
                "reviewer_id": "claude_core",
                "slot_id": "logic",
                "provider": "claude",
                "lens": "core",
                "family": "claude",
                "recommendation_class": "approve_candidate",
                "confidence": 0.7,
                "summary": "Looks bounded.",
                "top_findings": [
                    {
                        "category": "logic",
                        "severity": "low",
                        "claim": "No blocking logic regressions found.",
                        "evidence": ["small bounded diff"],
                        "files": ["aragora/swarm/pr_review_protocol.py"],
                    }
                ],
                "evidence_refs": [
                    {
                        "kind": "file",
                        "path": "aragora/swarm/pr_review_protocol.py",
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="duplicate reviewer output"):
            validate_reviewer_outputs([output, output])
