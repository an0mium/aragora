from __future__ import annotations

import json

import pytest

from aragora.review import (
    EvidenceKind,
    EvidenceRef,
    FindingCategory,
    FindingSeverity,
    Recommendation,
    ReviewerFinding,
    ReviewerOutput,
)
from aragora.review.provider_slots import ProviderSlotResolution
from aragora.swarm.pr_review_protocol import (
    EXECUTED_PROTOCOL_STATUS,
    FALLBACK_PROTOCOL_STATUS,
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
    PRReviewerExecutionFailure,
    RECOMMEND_APPROVE,
    RECOMMEND_ATTENTION,
    RECOMMEND_REPAIR,
    default_pr_review_protocol,
)


def _reviewer_output(
    *,
    slot_id: str,
    provider: str,
    family: str,
    recommendation: Recommendation,
    confidence: float,
    summary: str,
    claim: str,
    category: FindingCategory,
) -> ReviewerOutput:
    return ReviewerOutput(
        reviewer_id=f"{provider}:{slot_id}",
        slot_id=slot_id,
        provider=provider,
        lens="core" if slot_id in {"logic", "security"} else "heterodox",
        family=family,
        recommendation_class=recommendation,
        confidence=confidence,
        summary=summary,
        top_findings=(
            ReviewerFinding(
                category=category,
                severity=FindingSeverity.MEDIUM,
                claim=claim,
                evidence=(f"evidence for {slot_id}",),
                files=(f"aragora/{slot_id}.py",),
            ),
        ),
        evidence_refs=(
            EvidenceRef(
                kind=EvidenceKind.FILE,
                path=f"aragora/{slot_id}.py",
                line_range=(10, 18),
                quote=f"quoted evidence for {slot_id}",
            ),
        ),
        risk_flags=(f"{slot_id}_flag",),
        open_questions=(),
        round_index=1,
        latency_ms=1200,
        cost_usd=0.25,
    )


def test_resolve_provider_slots_prefers_available_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    def fake_which(binary: str) -> str | None:
        return {
            "claude": "/usr/bin/claude",
            "gemini": "/usr/bin/gemini",
        }.get(binary)

    monkeypatch.setattr("aragora.review.provider_slots.shutil.which", fake_which)

    protocol = default_pr_review_protocol()
    slots = protocol.resolve_provider_slots()

    assert [slot.selected_provider for slot in slots] == [
        "claude",
        "openai-api",
        "gemini-cli",
        None,
        "mistral-api",
    ]
    assert slots[3].status == "unavailable"
    assert "No configured provider available for grok" in slots[3].detail
    assert slots[4].status == "available"
    assert slots[4].selected_allowlisted is False
    assert slots[4].candidate_checks[0].provider == "mistral-api"


def test_build_packet_recommendation_and_binding() -> None:
    protocol = default_pr_review_protocol()
    packet = protocol.build_packet(
        repo="synaptent/aragora",
        pr_number=6306,
        title="Implement protocol",
        base_sha="base123",
        head_sha="head456",
        mergeable="MERGEABLE",
        review_decision="",
        checks_summary="1 failing / 3 total",
        has_failures=True,
        has_pending=False,
        additions=120,
        deletions=20,
        changed_files=4,
        labels=[],
        high_risk_paths=["aragora/security/policy.py"],
        validation_commands=["pytest -q tests/swarm/test_pr_review_protocol.py"],
        machine_recommendation=RECOMMEND_REPAIR,
        machine_recommendation_reason="checks failing or merge conflict — fix before review",
    )

    assert packet.protocol_version == PROTOCOL_VERSION
    assert packet.status == PROTOCOL_STATUS
    assert packet.binding.repo == "synaptent/aragora"
    assert packet.binding.pr_number == 6306
    assert packet.binding.base_sha == "base123"
    assert packet.binding.head_sha == "head456"
    assert packet.recommendation_class == RECOMMEND_REPAIR
    assert packet.validation_summary["has_failures"] is True
    assert packet.validation_summary["validation_commands"] == [
        "pytest -q tests/swarm/test_pr_review_protocol.py"
    ]
    assert packet.top_findings[0].finding_id == "validation-failing"
    assert packet.availability_summary.total_slots == 5
    assert packet.availability_summary.core_slots_total == 2
    assert packet.cost_estimate["low"] == pytest.approx(3.0)
    assert packet.cost_estimate["high"] == pytest.approx(5.0)


def test_build_packet_approve_candidate_has_empty_dissent_views() -> None:
    protocol = default_pr_review_protocol()
    packet = protocol.build_packet(
        repo="synaptent/aragora",
        pr_number=1,
        title="Small green diff",
        base_sha="base",
        head_sha="head",
        mergeable="MERGEABLE",
        review_decision="",
        checks_summary="3/3 green",
        has_failures=False,
        has_pending=False,
        additions=10,
        deletions=4,
        changed_files=2,
        labels=[],
        high_risk_paths=[],
        validation_commands=[],
        machine_recommendation=RECOMMEND_APPROVE,
        machine_recommendation_reason="all green, bounded diff, no high-risk paths",
    )

    assert packet.recommendation_class == RECOMMEND_APPROVE
    assert packet.confidence > 0.6
    assert packet.dissenting_views == []
    assert "No heterogeneous dissent recorded yet" in packet.dissent_summary
    assert packet.top_findings[0].severity == "low"


def test_build_packet_needs_attention_for_pending_and_large_diff() -> None:
    protocol = default_pr_review_protocol()
    packet = protocol.build_packet(
        repo="synaptent/aragora",
        pr_number=2,
        title="Large diff",
        base_sha="base",
        head_sha="head",
        mergeable="MERGEABLE",
        review_decision="REVIEW_REQUIRED",
        checks_summary="2 pending / 2 total",
        has_failures=False,
        has_pending=True,
        additions=700,
        deletions=40,
        changed_files=18,
        labels=["blocked"],
        high_risk_paths=[],
        validation_commands=[],
        machine_recommendation=RECOMMEND_ATTENTION,
        machine_recommendation_reason="high-risk paths touched or large diff — human should read it",
    )

    finding_ids = [finding.finding_id for finding in packet.top_findings]
    assert packet.recommendation_class == RECOMMEND_ATTENTION
    assert "large-diff" in finding_ids
    assert "checks-pending" in finding_ids
    assert "parked-label" in finding_ids
    assert packet.confidence < 0.6


def test_packet_to_dict_round_trips_to_json() -> None:
    protocol = default_pr_review_protocol()
    packet = protocol.build_packet(
        repo="synaptent/aragora",
        pr_number=5,
        title="Serialize me",
        base_sha="base",
        head_sha="head",
        mergeable="UNKNOWN",
        review_decision="",
        checks_summary="1/1 green",
        has_failures=False,
        has_pending=False,
        additions=1,
        deletions=1,
        changed_files=1,
        labels=[],
        high_risk_paths=[],
        validation_commands=[],
        machine_recommendation=RECOMMEND_APPROVE,
        machine_recommendation_reason="all green, bounded diff, no high-risk paths",
    )

    roundtrip = json.loads(json.dumps(packet.to_dict()))
    assert roundtrip["protocol_version"] == PROTOCOL_VERSION
    assert roundtrip["binding"]["repo"] == "synaptent/aragora"
    assert roundtrip["review_roles"][-1] == "synthesizer"
    assert roundtrip["availability_summary"]["total_slots"] == 5
    assert "candidate_checks" not in roundtrip["provider_slots"][0]
    assert "selected_allowlisted" not in roundtrip["provider_slots"][0]


def test_build_packet_upgrades_to_executed_status_and_preserves_dissent() -> None:
    protocol = default_pr_review_protocol()
    packet = protocol.build_packet(
        repo="synaptent/aragora",
        pr_number=77,
        title="Executed protocol",
        base_sha="base",
        head_sha="head",
        mergeable="MERGEABLE",
        review_decision="",
        checks_summary="5/5 green",
        has_failures=False,
        has_pending=False,
        additions=42,
        deletions=7,
        changed_files=3,
        labels=[],
        high_risk_paths=[],
        validation_commands=["pytest -q tests/swarm/test_pr_review_protocol.py"],
        machine_recommendation=RECOMMEND_APPROVE,
        machine_recommendation_reason="all green, bounded diff, no high-risk paths",
        reviewer_outputs=[
            _reviewer_output(
                slot_id="logic",
                provider="claude",
                family="claude",
                recommendation=Recommendation.APPROVE_CANDIDATE,
                confidence=0.72,
                summary="Logic review sees a bounded change.",
                claim="Logic path remains bounded.",
                category=FindingCategory.LOGIC,
            ),
            _reviewer_output(
                slot_id="security",
                provider="openai-api",
                family="gpt",
                recommendation=Recommendation.REPAIR_FIRST,
                confidence=0.44,
                summary="Security wants a tighter guard before merge.",
                claim="One auth boundary still needs direct review.",
                category=FindingCategory.SECURITY,
            ),
            _reviewer_output(
                slot_id="maintainability",
                provider="gemini-cli",
                family="gemini",
                recommendation=Recommendation.APPROVE_CANDIDATE,
                confidence=0.66,
                summary="Maintainability review is comfortable with the diff.",
                claim="Naming and structure stay within existing patterns.",
                category=FindingCategory.MAINTAINABILITY,
            ),
        ],
    )

    assert packet.status == EXECUTED_PROTOCOL_STATUS
    assert packet.recommendation_class == RECOMMEND_APPROVE
    assert packet.confidence_basis == EXECUTED_PROTOCOL_STATUS
    assert len(packet.dissenting_views) == 1
    assert packet.dissenting_views[0]["position"] == "request_changes"
    assert packet.validation_summary["reviewer_execution"]["reviewer_count"] == 3
    assert packet.top_findings[0].source in {
        "claude:logic",
        "openai-api:security",
        "gemini-cli:maintainability",
    }
    assert packet.cost_estimate["low"] == pytest.approx(0.75)
    assert packet.cost_estimate["high"] == pytest.approx(0.75)


def test_build_packet_falls_back_when_live_reviews_are_incomplete() -> None:
    protocol = default_pr_review_protocol()
    packet = protocol.build_packet(
        repo="synaptent/aragora",
        pr_number=78,
        title="Fallback protocol",
        base_sha="base",
        head_sha="head",
        mergeable="MERGEABLE",
        review_decision="",
        checks_summary="5/5 green",
        has_failures=False,
        has_pending=False,
        additions=18,
        deletions=4,
        changed_files=2,
        labels=[],
        high_risk_paths=[],
        validation_commands=[],
        machine_recommendation=RECOMMEND_APPROVE,
        machine_recommendation_reason="all green, bounded diff, no high-risk paths",
        reviewer_outputs=[
            _reviewer_output(
                slot_id="logic",
                provider="claude",
                family="claude",
                recommendation=Recommendation.APPROVE_CANDIDATE,
                confidence=0.7,
                summary="Logic is comfortable with the patch.",
                claim="Logic path is bounded.",
                category=FindingCategory.LOGIC,
            ),
            _reviewer_output(
                slot_id="security",
                provider="openai-api",
                family="gpt",
                recommendation=Recommendation.NEEDS_HUMAN_ATTENTION,
                confidence=0.5,
                summary="Security wants a human to verify one edge.",
                claim="Auth-adjacent change still needs human eyes.",
                category=FindingCategory.SECURITY,
            ),
        ],
        execution_failures=[
            PRReviewerExecutionFailure(
                slot_id="maintainability",
                review_role="maintainability_reviewer",
                provider="gemini-cli",
                reason="timed out",
            )
        ],
    )

    assert packet.status == f"{FALLBACK_PROTOCOL_STATUS}_insufficient_live_reviews"
    assert packet.recommendation_class == RECOMMEND_APPROVE
    assert packet.validation_summary["reviewer_execution"]["failure_count"] == 1
    assert len(packet.dissenting_views) == 1
    assert (
        "falling back" in packet.dissent_summary.lower()
        or "partial live reviewer" in packet.dissent_summary.lower()
    )


def test_execute_live_reviewers_parses_structured_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = default_pr_review_protocol()
    monkeypatch.setattr(
        "aragora.swarm.pr_review_protocol.PRReviewProtocol.resolve_provider_slots",
        lambda self: [
            ProviderSlotResolution(
                slot_id="logic",
                review_role="logic_reviewer",
                lens="core",
                family="claude",
                selected_provider="claude",
                status="available",
                detail="ready",
                candidates=["claude"],
            ),
            ProviderSlotResolution(
                slot_id="security",
                review_role="security_reviewer",
                lens="core",
                family="gpt",
                selected_provider="openai-api",
                status="available",
                detail="ready",
                candidates=["openai-api"],
            ),
            ProviderSlotResolution(
                slot_id="maintainability",
                review_role="maintainability_reviewer",
                lens="heterodox",
                family="gemini",
                selected_provider="gemini-cli",
                status="available",
                detail="ready",
                candidates=["gemini-cli"],
            ),
        ],
    )

    class _FakeAgent:
        def __init__(self, name: str) -> None:
            self.name = name

        async def generate(self, prompt: str, context=None) -> str:
            recommendation = "approve_candidate"
            if "security" in self.name:
                recommendation = "repair_first"
            return json.dumps(
                {
                    "schema_version": "reviewer_output.v1",
                    "reviewer_id": "placeholder",
                    "slot_id": "placeholder",
                    "provider": "placeholder",
                    "lens": "placeholder",
                    "family": "placeholder",
                    "recommendation_class": recommendation,
                    "confidence": 0.64,
                    "summary": f"review from {self.name}",
                    "top_findings": [
                        {
                            "category": "validation",
                            "severity": "medium",
                            "claim": f"{self.name} reviewed the diff",
                            "evidence": ["diff excerpt present"],
                        }
                    ],
                    "evidence_refs": [
                        {
                            "kind": "file",
                            "path": "aragora/example.py",
                            "line_range": [1, 2],
                            "quote": "example",
                        }
                    ],
                    "risk_flags": [],
                    "open_questions": [],
                    "round_index": 1,
                    "latency_ms": 0,
                    "cost_usd": 0.15,
                }
            )

    monkeypatch.setattr(
        "aragora.swarm.pr_review_protocol.create_agent",
        lambda model_type, name=None, role="critic", timeout=None: _FakeAgent(
            name or str(model_type)
        ),
    )

    outputs, failures = protocol.execute_live_reviewers(
        repo="synaptent/aragora",
        pr_number=88,
        title="Live execution",
        base_sha="base",
        head_sha="head",
        checks_summary="5/5 green",
        changed_files=["aragora/example.py"],
        diff_text="diff --git a/aragora/example.py b/aragora/example.py\n+print('hi')\n",
        machine_recommendation=RECOMMEND_APPROVE,
        machine_recommendation_reason="all green, bounded diff, no high-risk paths",
    )

    assert failures == []
    assert [output.reviewer_id for output in outputs] == [
        "claude:logic",
        "openai-api:security",
        "gemini-cli:maintainability",
    ]
    assert outputs[1].recommendation_class is Recommendation.REPAIR_FIRST
