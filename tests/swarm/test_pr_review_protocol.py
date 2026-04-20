from __future__ import annotations

import json

import pytest

from aragora.swarm.pr_review_protocol import (
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
    RECOMMEND_APPROVE,
    RECOMMEND_ATTENTION,
    RECOMMEND_REPAIR,
    default_pr_review_protocol,
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

    monkeypatch.setattr("aragora.swarm.pr_review_protocol.shutil.which", fake_which)

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
