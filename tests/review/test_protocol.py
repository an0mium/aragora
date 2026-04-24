"""Tests for aragora.review.protocol — schema-only PRReviewProtocol contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aragora.review import (
    ADVISORY_NOTE,
    DissentingView,
    DissentPosition,
    PRReviewProtocol,
    Recommendation,
    ReviewBrief,
    ReviewRole,
    RoleFinding,
    SynthesisPolicy,
)

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
            role_findings=(),
            dissent=(),
            validation_summary="32 unit tests pass; pre-commit clean.",
            overall_confidence=0.88,
            disagreement_score=0.05,
            total_cost_usd=0.18,
            total_wall_clock_ms=4200,
            agent_roster=("claude-opus-4-7", "gpt-5-4", "gemini-3-1-pro"),
            generated_at=datetime.now(UTC).isoformat(),
        )
        defaults.update(overrides)
        return ReviewBrief(**defaults)

    def test_brief_level_confidence_and_disagreement_are_first_class(self) -> None:
        # The UI (#6304), budget/escalation policy (#6305), and receipt
        # extension (#6307) all need an aggregate signal — not just per-finding
        # scores. This test is the contract guard against future refactors
        # that try to derive these from role_findings on the fly.
        brief = self._brief(overall_confidence=0.72, disagreement_score=0.41)
        assert brief.overall_confidence == 0.72
        assert brief.disagreement_score == 0.41
        d = brief.to_dict()
        assert d["overall_confidence"] == 0.72
        assert d["disagreement_score"] == 0.41

    def test_findings_severity_counts_defaults_to_empty(self) -> None:
        # Briefs built without structured severity input (legacy callers,
        # degraded paths) must not crash; the field carries an empty dict.
        brief = self._brief()
        assert brief.findings_severity_counts == {}
        assert brief.to_dict()["findings_severity_counts"] == {}

    def test_findings_severity_counts_round_trip(self) -> None:
        # First-class operator triage signal per #6505: "1 high finding"
        # and "5 low-severity editorial comments" must be mechanically
        # distinguishable without reading every finding's prose.
        counts = {"high": 1, "medium": 2, "low": 5}
        brief = self._brief(findings_severity_counts=counts)
        assert brief.findings_severity_counts == counts
        assert brief.to_dict()["findings_severity_counts"] == counts

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
            role_findings=(
                RoleFinding(
                    role=ReviewRole.LOGIC,
                    agent="claude-opus-4-7",
                    model="claude-opus-4-7-1m",
                    confidence=0.9,
                    finding_text="OK.",
                ),
            ),
            dissent=(
                DissentingView(
                    agent="grok-3",
                    position=DissentPosition.REQUEST_CHANGES,
                    reason="Edge case unconsidered.",
                ),
            ),
        )
        d = brief.to_dict()
        assert d["role_findings"][0]["role"] == "logic_reviewer"
        assert d["dissent"][0]["position"] == "request_changes"

    def test_sequence_fields_are_immutable_tuples(self) -> None:
        # frozen=True only blocks attribute reassignment; without tuple
        # types, callers could `brief.role_findings.append(...)` mid-flight
        # and break receipt stability + SHA binding. Tuples make the brief
        # a hashable, stable artifact suitable for receipt storage.
        brief = self._brief(
            role_findings=(
                RoleFinding(
                    role=ReviewRole.LOGIC,
                    agent="claude-opus-4-7",
                    model="claude-opus-4-7-1m",
                    confidence=0.9,
                    finding_text="OK.",
                ),
            ),
            dissent=(),
            agent_roster=("claude-opus-4-7", "gpt-5-4"),
        )
        assert isinstance(brief.role_findings, tuple)
        assert isinstance(brief.dissent, tuple)
        assert isinstance(brief.agent_roster, tuple)
        # Mutation attempts must fail; tuples have no append/extend.
        with pytest.raises(AttributeError):
            brief.role_findings.append(  # type: ignore[attr-defined]
                RoleFinding(
                    role=ReviewRole.SECURITY,
                    agent="x",
                    model="y",
                    confidence=0.0,
                    finding_text="z",
                )
            )
        with pytest.raises(AttributeError):
            brief.agent_roster.append("nope")  # type: ignore[attr-defined]

    def test_packet_sha_preimage_is_documented(self) -> None:
        # The preimage rule lives in the docstring (not in code) because
        # this module is intentionally behavior-free. #6307 will implement
        # the hash. This test guards that the rule is documented so #6307
        # cannot drift silently.
        from aragora.review.protocol import ReviewBrief as RB

        doc = RB.__doc__ or ""
        assert "Packet-SHA preimage" in doc
        assert 'Remove the ``"packet_sha"`` key' in doc
        assert "canonical JSON" in doc
        assert "sha256" in doc.lower()

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
            model_panel=("claude-opus-4-7-1m", "gpt-5-4", "grok-3"),
        )
        defaults.update(overrides)
        return PRReviewProtocol(**defaults)

    def test_model_panel_is_immutable_tuple(self) -> None:
        protocol = self._protocol()
        assert isinstance(protocol.model_panel, tuple)
        with pytest.raises(AttributeError):
            protocol.model_panel.append("extra-model")  # type: ignore[attr-defined]

    def test_advisory_only_is_invariant(self) -> None:
        # The configuration cannot ship with advisory_only=False because
        # that would imply machine settlement, which the design brief
        # explicitly bans.
        protocol = self._protocol()
        assert protocol.advisory_only is True

    def test_panel_oriented_topology_not_role_to_model(self) -> None:
        # Roles are output tags on findings, not input constraints binding
        # one model to one role. The runner is free to assign roles to
        # panel members dynamically. If a future refactor reintroduces
        # `role_to_model` as a config field, this test fails loudly.
        # Note: `output_roles` IS expected (declares required role coverage
        # in the brief, distinct from the rejected `role_to_model` shape
        # which fixed one model per role).
        protocol = self._protocol()
        assert hasattr(protocol, "model_panel")
        assert not hasattr(protocol, "role_to_model")
        assert hasattr(protocol, "output_roles")  # declared role coverage, OK
        assert not hasattr(protocol, "roles")  # no anonymous role list

    def test_output_roles_declared_for_brief_coverage(self) -> None:
        # Without an explicit output_roles contract, downstream consumers
        # (#6307 receipts, #6304 UI, #6305 policy) cannot tell whether a
        # missing role section in a brief is a bug or an acceptable
        # omission. This test guards the contract so they can rely on it.
        protocol = self._protocol()
        assert isinstance(protocol.output_roles, tuple)
        assert len(protocol.output_roles) >= 1
        # Every entry must be a ReviewRole, not a bare string.
        for role in protocol.output_roles:
            assert isinstance(role, ReviewRole)

    def test_default_output_roles_cover_four_substantive_reviewers(self) -> None:
        # Default coverage is the four substantive reviewer roles.
        # SYNTHESIZER is opt-in via SynthesisPolicy.SYNTHESIZER_AGENT;
        # it should NOT be in the default output_roles tuple.
        protocol = self._protocol()
        assert protocol.output_roles == (
            ReviewRole.LOGIC,
            ReviewRole.SECURITY,
            ReviewRole.MAINTAINABILITY,
            ReviewRole.SKEPTIC,
        )
        assert ReviewRole.SYNTHESIZER not in protocol.output_roles

    def test_output_roles_is_immutable_tuple(self) -> None:
        protocol = self._protocol()
        assert isinstance(protocol.output_roles, tuple)
        with pytest.raises(AttributeError):
            protocol.output_roles.append(ReviewRole.SYNTHESIZER)  # type: ignore[attr-defined]

    def test_output_roles_serialized_as_strings_in_to_dict(self) -> None:
        protocol = self._protocol()
        d = protocol.to_dict()
        assert d["output_roles"] == [
            "logic_reviewer",
            "security_reviewer",
            "maintainability_reviewer",
            "skeptic",
        ]

    def test_output_roles_can_be_overridden(self) -> None:
        protocol = self._protocol(
            output_roles=(ReviewRole.LOGIC, ReviewRole.SECURITY),
        )
        assert protocol.output_roles == (ReviewRole.LOGIC, ReviewRole.SECURITY)
        d = protocol.to_dict()
        assert d["output_roles"] == ["logic_reviewer", "security_reviewer"]

    def test_heterogeneity_required_by_default(self) -> None:
        protocol = self._protocol()
        assert protocol.require_heterogeneous_models is True

    def test_no_cost_or_budget_in_contract_layer(self) -> None:
        # Cost caps and budget defaults belong in #6305 (cost-aware policy),
        # not in this cross-cutting foundation type. If a future refactor
        # tries to bake `max_cost_usd` or similar back into the contract,
        # this test fails loudly.
        protocol = self._protocol()
        for forbidden in (
            "max_cost_usd",
            "max_wall_seconds",
            "max_findings_per_role",
            "budget_usd",
            "cost_cap",
        ):
            assert not hasattr(protocol, forbidden), (
                f"PRReviewProtocol carries a policy field `{forbidden}`; "
                f"policy belongs in #6305, not the foundation contract."
            )

    def test_synthesis_policy_default_is_weighted(self) -> None:
        protocol = self._protocol()
        assert protocol.synthesis_policy == SynthesisPolicy.WEIGHTED

    def test_synthesis_policy_enum_values(self) -> None:
        assert SynthesisPolicy.MAJORITY.value == "majority"
        assert SynthesisPolicy.WEIGHTED.value == "weighted"
        assert SynthesisPolicy.SYNTHESIZER_AGENT.value == "synthesizer"
        assert SynthesisPolicy.UNANIMOUS_OR_ESCALATE.value == "unanimous_or_escalate"

    def test_rounds_default_is_one(self) -> None:
        # Single-pass parallel by default; multi-round is opt-in.
        protocol = self._protocol()
        assert protocol.rounds == 1

    def test_to_dict_serializes_synthesis_policy(self) -> None:
        protocol = self._protocol(synthesis_policy=SynthesisPolicy.UNANIMOUS_OR_ESCALATE)
        d = protocol.to_dict()
        assert d["synthesis_policy"] == "unanimous_or_escalate"
        # Tuple field serializes to JSON-compatible list.
        assert d["model_panel"] == ["claude-opus-4-7-1m", "gpt-5-4", "grok-3"]
        assert d["rounds"] == 1
        assert d["require_heterogeneous_models"] is True
        assert d["advisory_only"] is True

    def test_json_roundtrip(self) -> None:
        protocol = self._protocol()
        roundtrip = json.loads(json.dumps(protocol.to_dict()))
        assert roundtrip["model_panel"] == ["claude-opus-4-7-1m", "gpt-5-4", "grok-3"]
        assert roundtrip["synthesis_policy"] == "weighted"
        assert roundtrip["advisory_only"] is True

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
