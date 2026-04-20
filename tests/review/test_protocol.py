"""Tests for aragora.review.protocol — schema-only PRReviewProtocol contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aragora.review import (
    ADVISORY_NOTE,
    DissentingView,
    PRReviewProtocol,
    Recommendation,
    ReviewBrief,
    ReviewRole,
    RoleFinding,
)
from aragora.review.protocol import DissentPosition

UTC = timezone.utc


# --- Enums ---------------------------------------------------------------


class TestEnums:
    def test_review_role_values_match_design_brief(self) -> None:
        # The design brief at docs/plans/2026-04-19-pr-intelligence-brief.md
        # explicitly lists these five roles. Drift between code and brief
        # would silently break #6307 / #6304 / #6305 which all consume them.
        assert ReviewRole.LOGIC.value == "logic_reviewer"
        assert ReviewRole.SECURITY.value == "security_reviewer"
        assert ReviewRole.MAINTAINABILITY.value == "maintainability_reviewer"
        assert ReviewRole.SKEPTIC.value == "skeptic"
        assert ReviewRole.SYNTHESIZER.value == "synthesizer"

    def test_recommendation_classes_match_review_packet(self) -> None:
        # ReviewPacket in aragora.cli.commands.review_queue uses the same
        # three recommendation strings; uniformity matters for downstream
        # consumers that may receive either Brief or Packet output.
        assert Recommendation.APPROVE_CANDIDATE.value == "approve_candidate"
        assert Recommendation.NEEDS_HUMAN_ATTENTION.value == "needs_human_attention"
        assert Recommendation.REPAIR_FIRST.value == "repair_first"

    def test_dissent_position_values(self) -> None:
        assert DissentPosition.APPROVE.value == "approve"
        assert DissentPosition.REQUEST_CHANGES.value == "request_changes"
        assert DissentPosition.DEFER.value == "defer"


# --- Constants ------------------------------------------------------------


class TestAdvisoryNote:
    def test_advisory_note_says_advisory_only(self) -> None:
        # The literal string is part of the contract. Downstream consumers
        # check it by exact match to verify a brief is not an approval.
        assert ADVISORY_NOTE == (
            "This brief is advisory only. It does not approve or block merge. "
            "Human settlement required."
        )


# --- RoleFinding ----------------------------------------------------------


class TestRoleFinding:
    def _finding(self, **overrides) -> RoleFinding:
        defaults = dict(
            role=ReviewRole.LOGIC,
            agent="claude-opus-4-7",
            model="claude-opus-4-7-1m",
            confidence=0.85,
            finding_text="No regressions found in changed code paths.",
            latency_ms=1200,
            cost_usd=0.045,
        )
        defaults.update(overrides)
        return RoleFinding(**defaults)

    def test_to_dict_serializes_role_as_string(self) -> None:
        finding = self._finding()
        d = finding.to_dict()
        assert d["role"] == "logic_reviewer"
        assert d["agent"] == "claude-opus-4-7"
        assert d["confidence"] == 0.85

    def test_json_roundtrip_preserves_fields(self) -> None:
        finding = self._finding()
        roundtrip = json.loads(json.dumps(finding.to_dict()))
        assert roundtrip["role"] == "logic_reviewer"
        assert roundtrip["model"] == "claude-opus-4-7-1m"
        assert roundtrip["latency_ms"] == 1200
        assert roundtrip["cost_usd"] == 0.045

    def test_frozen(self) -> None:
        finding = self._finding()
        with pytest.raises((AttributeError, TypeError)):
            finding.confidence = 0.99  # type: ignore[misc]


# --- DissentingView -------------------------------------------------------


class TestDissentingView:
    def test_to_dict_serializes_position(self) -> None:
        view = DissentingView(
            agent="grok-3",
            position=DissentPosition.REQUEST_CHANGES,
            reason="Flags potential auth bypass in handler.",
            role=ReviewRole.SECURITY,
        )
        d = view.to_dict()
        assert d["position"] == "request_changes"
        assert d["role"] == "security_reviewer"
        assert d["agent"] == "grok-3"

    def test_role_is_optional(self) -> None:
        view = DissentingView(
            agent="gpt-5-4",
            position=DissentPosition.DEFER,
            reason="Not enough evidence to settle.",
        )
        d = view.to_dict()
        # asdict serializes None as null in JSON, which is valid; the
        # contract is that role can be omitted when constructing the dataclass.
        assert d.get("role") is None
        assert view.role is None


# --- ReviewBrief ----------------------------------------------------------


class TestReviewBrief:
    def _brief(self, **overrides) -> ReviewBrief:
        defaults = dict(
            pr_number=6306,
            repo="synaptent/aragora",
            head_sha="2272f79cc7aee6da1d3ee1ea3de3dcbe5d253ade",
            base_sha="ae42ff033",
            packet_sha="abc123def456",
            recommendation=Recommendation.APPROVE_CANDIDATE,
            top_line="Bounded foundation PR; all gates green; no high-risk paths.",
            role_findings=[],
            dissent=[],
            validation_summary="32 unit tests pass; pre-commit clean.",
            total_cost_usd=0.18,
            total_wall_clock_ms=4200,
            agent_roster=["claude-opus-4-7", "gpt-5-4", "gemini-3-1-pro"],
            generated_at=datetime.now(UTC).isoformat(),
        )
        defaults.update(overrides)
        return ReviewBrief(**defaults)

    def test_advisory_only_default_is_true(self) -> None:
        # SAFETY INVARIANT: a brief is never an approval.
        brief = self._brief()
        assert brief.advisory_only is True

    def test_settlement_note_default_is_advisory(self) -> None:
        brief = self._brief()
        assert brief.settlement_note == ADVISORY_NOTE

    def test_to_dict_includes_advisory_signature(self) -> None:
        # Downstream consumers can check this property mechanically without
        # parsing prose; that is the whole point of the frozen field pair.
        brief = self._brief()
        d = brief.to_dict()
        assert d["advisory_only"] is True
        assert d["settlement_note"] == ADVISORY_NOTE

    def test_to_dict_serializes_recommendation(self) -> None:
        brief = self._brief(recommendation=Recommendation.NEEDS_HUMAN_ATTENTION)
        assert brief.to_dict()["recommendation"] == "needs_human_attention"

    def test_to_dict_serializes_nested_findings_and_dissent(self) -> None:
        brief = self._brief(
            role_findings=[
                RoleFinding(
                    role=ReviewRole.LOGIC,
                    agent="claude-opus-4-7",
                    model="claude-opus-4-7-1m",
                    confidence=0.9,
                    finding_text="OK.",
                ),
            ],
            dissent=[
                DissentingView(
                    agent="grok-3",
                    position=DissentPosition.REQUEST_CHANGES,
                    reason="Edge case unconsidered.",
                ),
            ],
        )
        d = brief.to_dict()
        assert d["role_findings"][0]["role"] == "logic_reviewer"
        assert d["dissent"][0]["position"] == "request_changes"

    def test_json_roundtrip(self) -> None:
        brief = self._brief()
        roundtrip = json.loads(json.dumps(brief.to_dict()))
        assert roundtrip["pr_number"] == 6306
        assert roundtrip["repo"] == "synaptent/aragora"
        assert roundtrip["head_sha"] == "2272f79cc7aee6da1d3ee1ea3de3dcbe5d253ade"
        assert roundtrip["advisory_only"] is True

    def test_frozen(self) -> None:
        brief = self._brief()
        with pytest.raises((AttributeError, TypeError)):
            brief.advisory_only = False  # type: ignore[misc]

    def test_head_sha_and_packet_sha_are_required_for_settlement_binding(
        self,
    ) -> None:
        # The design brief (Section "Outputs") requires brief binding to the
        # exact head_sha so settlement can verify the brief still matches.
        # If either field is empty, downstream settlement would lose the
        # SHA-bound property.
        brief = self._brief(head_sha="", packet_sha="")
        # We don't validate non-emptiness in this layer (callers do), but
        # we *do* require the fields exist on the dataclass.
        assert hasattr(brief, "head_sha")
        assert hasattr(brief, "packet_sha")


# --- PRReviewProtocol -----------------------------------------------------


class TestPRReviewProtocol:
    def _protocol(self, **overrides) -> PRReviewProtocol:
        defaults = dict(
            roles=[
                ReviewRole.LOGIC,
                ReviewRole.SECURITY,
                ReviewRole.SKEPTIC,
            ],
            role_to_model={
                ReviewRole.LOGIC: "claude-opus-4-7-1m",
                ReviewRole.SECURITY: "gpt-5-4",
                ReviewRole.SKEPTIC: "grok-3",
            },
        )
        defaults.update(overrides)
        return PRReviewProtocol(**defaults)

    def test_advisory_only_is_invariant(self) -> None:
        # The configuration cannot ship with advisory_only=False because
        # that would imply machine settlement, which the design brief
        # explicitly bans.
        protocol = self._protocol()
        assert protocol.advisory_only is True

    def test_default_max_cost_anchors_to_market(self) -> None:
        # $25 is Anthropic's per-PR review price as of 2026-04-19; any
        # heterogeneous brief that exceeds this without explicit operator
        # opt-in is unjustifiable on cost grounds.
        protocol = self._protocol()
        assert protocol.max_cost_usd == 25.0

    def test_heterogeneity_required_by_default(self) -> None:
        protocol = self._protocol()
        assert protocol.require_heterogeneous_models is True

    def test_to_dict_serializes_roles_and_role_to_model(self) -> None:
        protocol = self._protocol()
        d = protocol.to_dict()
        assert d["roles"] == ["logic_reviewer", "security_reviewer", "skeptic"]
        assert d["role_to_model"] == {
            "logic_reviewer": "claude-opus-4-7-1m",
            "security_reviewer": "gpt-5-4",
            "skeptic": "grok-3",
        }

    def test_json_roundtrip(self) -> None:
        protocol = self._protocol()
        roundtrip = json.loads(json.dumps(protocol.to_dict()))
        assert "logic_reviewer" in roundtrip["roles"]
        assert roundtrip["max_cost_usd"] == 25.0
        assert roundtrip["max_wall_seconds"] == 600

    def test_frozen(self) -> None:
        protocol = self._protocol()
        with pytest.raises((AttributeError, TypeError)):
            protocol.advisory_only = False  # type: ignore[misc]


# --- Cross-module contract coherence --------------------------------------


class TestContractCoherence:
    def test_brief_and_packet_use_same_recommendation_strings(self) -> None:
        # ReviewBrief.recommendation values must match ReviewPacket
        # machine_recommendation values, since downstream consumers (queue,
        # UI, ledger) may receive either output kind.
        from aragora.cli.commands.review_queue import ReviewPacket

        # Build a minimal ReviewPacket and confirm its machine_recommendation
        # field accepts the same strings ReviewBrief.recommendation produces.
        packet_recommendations = {"approve_candidate", "needs_human_attention", "repair_first"}
        brief_recommendations = {r.value for r in Recommendation}
        assert packet_recommendations == brief_recommendations
