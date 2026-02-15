"""
Tests for DecisionReceipt.from_review_result() factory method.

Verifies that PR review findings produced by ``extract_review_findings()``
in ``aragora/cli/review.py`` are correctly converted into cryptographic
decision receipts with deterministic hashing, provenance chains, and
consensus proofs.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from aragora.gauntlet.receipt_models import (
    ConsensusProof,
    DecisionReceipt,
    ProvenanceRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_review_result(**overrides) -> dict:
    """Build a realistic review-result dict matching extract_review_findings() output."""
    base = {
        "unanimous_critiques": [
            "SQL injection vulnerability in user search",
            "Missing input validation on file upload endpoint",
        ],
        "split_opinions": [
            (
                "Add request rate limiting",
                ["anthropic-api", "openai-api"],
                ["gemini-api"],
            ),
        ],
        "risk_areas": [
            "Error handling in payment flow may expose sensitive data",
        ],
        "agreement_score": 0.75,
        "agent_alignment": {
            "anthropic-api": {"openai-api": 0.8, "gemini-api": 0.6},
        },
        "critical_issues": [
            {
                "agent": "anthropic-api",
                "issue": "SQL injection in search_users()",
                "target": "api/users.py:45",
                "suggestions": ["Use parameterized queries"],
            },
        ],
        "high_issues": [
            {
                "agent": "openai-api",
                "issue": "Missing CSRF protection on POST endpoints",
                "target": "api/routes.py",
                "suggestions": ["Add CSRF middleware"],
            },
        ],
        "medium_issues": [
            {
                "agent": "gemini-api",
                "issue": "Unbounded query results - add pagination",
                "target": "api/products.py:102",
                "suggestions": ["Add LIMIT clause"],
            },
        ],
        "low_issues": [],
        "all_critiques": [],
        "final_summary": "Multi-agent review found 2 critical security issues.",
        "agents_used": ["anthropic-api", "openai-api", "gemini-api"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test: basic receipt creation
# ---------------------------------------------------------------------------


class TestFromReviewResult:
    """Test DecisionReceipt.from_review_result() factory."""

    def test_creates_receipt_from_review(self):
        """A receipt is created with correct identification fields."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)

        assert receipt.receipt_id  # non-empty UUID
        assert receipt.gauntlet_id.startswith("review-")
        assert receipt.timestamp  # ISO timestamp populated

    def test_verdict_fail_on_critical_issues(self):
        """Verdict is FAIL when critical issues exist."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.verdict == "FAIL"

    def test_verdict_pass_with_no_issues(self):
        """Verdict is PASS when no issues and agreement is high."""
        findings = _make_review_result(
            critical_issues=[],
            high_issues=[],
            medium_issues=[],
            low_issues=[],
            agreement_score=0.9,
        )
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.verdict == "PASS"

    def test_verdict_conditional_high_issues_good_agreement(self):
        """Verdict is CONDITIONAL when high issues exist but agreement is high."""
        findings = _make_review_result(
            critical_issues=[],
            agreement_score=0.8,
        )
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.verdict == "CONDITIONAL"

    def test_risk_summary_counts(self):
        """Risk summary correctly counts issues by severity."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)

        assert receipt.risk_summary["critical"] == 1
        assert receipt.risk_summary["high"] == 1
        assert receipt.risk_summary["medium"] == 1
        assert receipt.risk_summary["low"] == 0
        assert receipt.risk_summary["total"] == 3

    def test_confidence_from_agreement_score(self):
        """Confidence reflects the review agreement score."""
        findings = _make_review_result(agreement_score=0.82)
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.confidence == pytest.approx(0.82)

    def test_pr_url_in_metadata(self):
        """PR URL is stored in config_used and input_summary."""
        url = "https://github.com/owner/repo/pull/42"
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings, pr_url=url)

        assert receipt.config_used["pr_url"] == url
        assert url in receipt.input_summary

    def test_reviewer_agents_override(self):
        """Explicit reviewer_agents takes precedence over agents_used."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(
            findings,
            reviewer_agents=["custom-agent-a", "custom-agent-b"],
        )
        assert receipt.config_used["reviewer_agents"] == [
            "custom-agent-a",
            "custom-agent-b",
        ]
        assert receipt.probes_run == 2

    def test_agents_fallback_to_agents_used(self):
        """When reviewer_agents is None, agents_used from the result is used."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.config_used["reviewer_agents"] == [
            "anthropic-api",
            "openai-api",
            "gemini-api",
        ]


# ---------------------------------------------------------------------------
# Test: deterministic hashing
# ---------------------------------------------------------------------------


class TestHashDeterminism:
    """Receipt input_hash must be deterministic for the same review content."""

    def test_same_input_same_hash(self):
        """Two receipts from identical review results share the same input_hash."""
        findings = _make_review_result()
        receipt_a = DecisionReceipt.from_review_result(findings)
        receipt_b = DecisionReceipt.from_review_result(findings)

        assert receipt_a.input_hash == receipt_b.input_hash
        assert len(receipt_a.input_hash) == 64  # SHA-256 hex

    def test_different_input_different_hash(self):
        """Changing the review content changes the input_hash."""
        findings_a = _make_review_result(final_summary="Version A of the summary.")
        findings_b = _make_review_result(final_summary="Version B of the summary.")
        receipt_a = DecisionReceipt.from_review_result(findings_a)
        receipt_b = DecisionReceipt.from_review_result(findings_b)

        assert receipt_a.input_hash != receipt_b.input_hash

    def test_hash_matches_manual_computation(self):
        """input_hash matches a hand-rolled SHA-256 of the canonical content."""
        findings = _make_review_result(
            unanimous_critiques=["issue-1"],
            critical_issues=[],
            high_issues=[],
            medium_issues=[],
            low_issues=[],
            final_summary="clean",
        )
        receipt = DecisionReceipt.from_review_result(findings)

        canonical = json.dumps(
            {
                "unanimous_critiques": ["issue-1"],
                "critical_issues": [],
                "high_issues": [],
                "medium_issues": [],
                "low_issues": [],
                "final_summary": "clean",
            },
            sort_keys=True,
        )
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        assert receipt.input_hash == expected

    def test_artifact_hash_populated(self):
        """The artifact_hash (content-addressable) is auto-computed."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.artifact_hash
        assert len(receipt.artifact_hash) == 64

    def test_integrity_check_passes(self):
        """verify_integrity() returns True for an untampered receipt."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.verify_integrity()


# ---------------------------------------------------------------------------
# Test: provenance chain captures all agent assessments
# ---------------------------------------------------------------------------


class TestProvenanceChain:
    """Provenance chain must record every agent finding and the final verdict."""

    def test_provenance_includes_all_findings(self):
        """Each severity-level issue appears as a review_finding event."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)

        finding_events = [p for p in receipt.provenance_chain if p.event_type == "review_finding"]
        # 1 critical + 1 high + 1 medium + 0 low = 3
        assert len(finding_events) == 3

    def test_provenance_records_agent_name(self):
        """Finding events carry the originating agent's name."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)

        finding_events = [p for p in receipt.provenance_chain if p.event_type == "review_finding"]
        agents_in_provenance = {p.agent for p in finding_events}
        assert "anthropic-api" in agents_in_provenance
        assert "openai-api" in agents_in_provenance

    def test_provenance_includes_unanimous_critiques(self):
        """Unanimous critiques appear as unanimous_critique events."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)

        unanimous_events = [
            p for p in receipt.provenance_chain if p.event_type == "unanimous_critique"
        ]
        assert len(unanimous_events) == 2

    def test_provenance_includes_split_opinions(self):
        """Split opinions appear as split_opinion events."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)

        split_events = [p for p in receipt.provenance_chain if p.event_type == "split_opinion"]
        assert len(split_events) == 1

    def test_provenance_ends_with_verdict(self):
        """The last provenance record is always the verdict."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)

        assert receipt.provenance_chain[-1].event_type == "verdict"
        assert "FAIL" in receipt.provenance_chain[-1].description

    def test_evidence_hashes_present(self):
        """Every review_finding provenance record has a non-empty evidence_hash."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)

        finding_events = [p for p in receipt.provenance_chain if p.event_type == "review_finding"]
        for event in finding_events:
            assert event.evidence_hash, f"Missing hash on: {event.description}"


# ---------------------------------------------------------------------------
# Test: consensus proof
# ---------------------------------------------------------------------------


class TestConsensusProof:
    """Consensus proof must reflect agent agreement and dissent."""

    def test_consensus_proof_exists(self):
        """A ConsensusProof is always attached."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.consensus_proof is not None

    def test_consensus_method_is_multi_agent_review(self):
        """Method is 'multi_agent_review'."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.consensus_proof.method == "multi_agent_review"

    def test_dissenting_agents_from_split_opinions(self):
        """Agents in minority of split opinions are recorded as dissenters."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)
        assert "gemini-api" in receipt.consensus_proof.dissenting_agents

    def test_supporting_agents_exclude_dissenters(self):
        """Supporting agents are participants minus dissenters."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)
        assert "gemini-api" not in receipt.consensus_proof.supporting_agents
        assert "anthropic-api" in receipt.consensus_proof.supporting_agents

    def test_consensus_not_reached_with_critical(self):
        """Consensus is not reached when critical issues are present."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.consensus_proof.reached is False

    def test_consensus_reached_no_critical_good_agreement(self):
        """Consensus is reached when no critical issues and agreement >= 0.5."""
        findings = _make_review_result(
            critical_issues=[],
            agreement_score=0.8,
        )
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.consensus_proof.reached is True


# ---------------------------------------------------------------------------
# Test: missing optional fields
# ---------------------------------------------------------------------------


class TestMissingOptionalFields:
    """The factory must handle sparse / empty review results gracefully."""

    def test_empty_review_result(self):
        """An empty dict produces a valid receipt with safe defaults."""
        receipt = DecisionReceipt.from_review_result({})

        assert receipt.receipt_id
        assert receipt.verdict == "PASS"  # no issues found
        assert receipt.risk_summary["total"] == 0
        assert receipt.confidence == 0.0
        assert receipt.input_hash  # still computed (hash of empty canonical)
        assert receipt.verify_integrity()

    def test_no_agents_used(self):
        """Missing agents_used defaults to an empty list."""
        receipt = DecisionReceipt.from_review_result({"agreement_score": 0.5})
        assert receipt.config_used["reviewer_agents"] == []
        assert receipt.probes_run == 0

    def test_no_summary(self):
        """Missing final_summary falls back to a generated description."""
        findings = _make_review_result(final_summary="")
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.verdict_reasoning  # non-empty fallback

    def test_no_split_opinions(self):
        """Missing split_opinions does not break provenance or consensus."""
        findings = _make_review_result(split_opinions=[])
        receipt = DecisionReceipt.from_review_result(findings)

        split_events = [p for p in receipt.provenance_chain if p.event_type == "split_opinion"]
        assert len(split_events) == 0
        assert receipt.consensus_proof.dissenting_agents == []

    def test_no_risk_areas(self):
        """Missing risk_areas is fine; dissenting_views will still be populated from splits."""
        findings = _make_review_result(risk_areas=[])
        receipt = DecisionReceipt.from_review_result(findings)
        # split opinions still feed dissenting_views
        assert isinstance(receipt.dissenting_views, list)

    def test_pr_url_none(self):
        """When pr_url is None, input_summary falls back gracefully."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings, pr_url=None)
        assert receipt.config_used["pr_url"] is None
        assert receipt.input_summary  # non-empty fallback

    def test_issues_as_plain_strings(self):
        """Issues can be plain strings instead of dicts."""
        findings = _make_review_result(
            critical_issues=["raw string issue"],
            high_issues=[],
            medium_issues=[],
            low_issues=[],
        )
        receipt = DecisionReceipt.from_review_result(findings)
        assert receipt.risk_summary["critical"] == 1
        finding_events = [p for p in receipt.provenance_chain if p.event_type == "review_finding"]
        assert len(finding_events) == 1
        assert finding_events[0].agent is None  # no agent for plain strings


# ---------------------------------------------------------------------------
# Test: serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    """Receipt from review result can be serialized and deserialized."""

    def test_to_dict_and_back(self):
        """to_dict() -> from_dict() preserves key fields."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(
            findings, pr_url="https://github.com/owner/repo/pull/99"
        )
        data = receipt.to_dict()
        restored = DecisionReceipt.from_dict(data)

        assert restored.receipt_id == receipt.receipt_id
        assert restored.verdict == receipt.verdict
        assert restored.input_hash == receipt.input_hash
        assert restored.confidence == receipt.confidence
        assert restored.risk_summary == receipt.risk_summary
        assert len(restored.provenance_chain) == len(receipt.provenance_chain)

    def test_to_json_is_valid(self):
        """to_json() produces parseable JSON."""
        findings = _make_review_result()
        receipt = DecisionReceipt.from_review_result(findings)
        parsed = json.loads(receipt.to_json())
        assert parsed["verdict"] == "FAIL"
        assert parsed["input_hash"] == receipt.input_hash
