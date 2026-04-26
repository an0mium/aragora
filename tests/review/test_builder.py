"""Tests for aragora.review.builder — deterministic ReviewBrief assembly.

The builder is a pure function that maps panel votes + a synthesis
policy into a settlement-ready ReviewBrief. These tests pin the
contract on:

  - per-policy recommendation logic (majority / weighted / synthesizer /
    unanimous_or_escalate), including ties and escalation
  - dissent emission (any vote whose position differs from the brief
    recommendation must appear in DissentingView)
  - confidence + disagreement aggregation
  - packet_sha determinism (same inputs ⇒ same SHA; different inputs ⇒
    different SHA)
  - input validation (empty panel, malformed synthesizer policy)
"""

from __future__ import annotations

import pytest

from aragora.review import (
    DissentPosition,
    PanelVote,
    Recommendation,
    ReviewBrief,
    ReviewRole,
    RoleFinding,
    SynthesisPolicy,
    build_brief,
    compute_packet_sha,
)


# --- helpers -------------------------------------------------------------


def _vote(
    *,
    role: ReviewRole,
    agent: str,
    position: DissentPosition,
    confidence: float = 0.8,
    text: str = "",
    reason: str = "",
) -> PanelVote:
    return PanelVote(
        finding=RoleFinding(
            role=role,
            agent=agent,
            model=f"{agent}-model",
            confidence=confidence,
            finding_text=text or f"{role.value} finding",
            latency_ms=100,
            cost_usd=0.05,
        ),
        position=position,
        reason=reason or f"{agent} reason",
    )


def _build(
    votes: list[PanelVote],
    *,
    policy: SynthesisPolicy = SynthesisPolicy.MAJORITY,
    head_sha: str = "deadbeef",
    base_sha: str = "feedface",
    findings_severity_counts: dict[str, int] | None = None,
) -> ReviewBrief:
    return build_brief(
        votes=votes,
        pr_number=6306,
        repo="synaptent/aragora",
        head_sha=head_sha,
        base_sha=base_sha,
        top_line="One-line top.",
        validation_summary="checks green",
        generated_at="2026-04-20T12:00:00+00:00",
        synthesis_policy=policy,
        findings_severity_counts=findings_severity_counts,
    )


# --- MAJORITY policy ------------------------------------------------------


class TestMajorityPolicy:
    def test_unanimous_approve_yields_approve_candidate_no_dissent(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.MAINTAINABILITY, agent="a3", position=DissentPosition.APPROVE),
        ]
        brief = _build(votes, policy=SynthesisPolicy.MAJORITY)
        assert brief.recommendation is Recommendation.APPROVE_CANDIDATE
        assert brief.dissent == ()

    def test_plurality_wins_with_dissent_for_minority(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.MAINTAINABILITY, agent="a3", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SKEPTIC, agent="a4", position=DissentPosition.REQUEST_CHANGES),
        ]
        brief = _build(votes, policy=SynthesisPolicy.MAJORITY)
        assert brief.recommendation is Recommendation.APPROVE_CANDIDATE
        assert len(brief.dissent) == 1
        d = brief.dissent[0]
        assert d.agent == "a4"
        assert d.position is DissentPosition.REQUEST_CHANGES
        assert d.role is ReviewRole.SKEPTIC

    def test_tie_escalates_to_needs_human_attention(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.APPROVE),
            _vote(
                role=ReviewRole.MAINTAINABILITY,
                agent="a3",
                position=DissentPosition.REQUEST_CHANGES,
            ),
            _vote(role=ReviewRole.SKEPTIC, agent="a4", position=DissentPosition.REQUEST_CHANGES),
        ]
        brief = _build(votes, policy=SynthesisPolicy.MAJORITY)
        # 2-vs-2 tie → escalation. Both APPROVE and REQUEST_CHANGES voters
        # disagree with NEEDS_HUMAN_ATTENTION, so all four are dissent
        # — that's the operator-visible signal that the panel split.
        assert brief.recommendation is Recommendation.NEEDS_HUMAN_ATTENTION
        assert len(brief.dissent) == 4


# --- WEIGHTED policy ------------------------------------------------------


class TestWeightedPolicy:
    def test_high_confidence_minority_overrides_low_confidence_majority(self) -> None:
        # Three low-confidence APPROVE votes (sum 0.3) vs. one
        # high-confidence REQUEST_CHANGES (0.95): WEIGHTED chooses the
        # heavier side, so the brief recommends REPAIR_FIRST and the
        # three APPROVE voters become dissent.
        votes = [
            _vote(
                role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE, confidence=0.1
            ),
            _vote(
                role=ReviewRole.SECURITY,
                agent="a2",
                position=DissentPosition.APPROVE,
                confidence=0.1,
            ),
            _vote(
                role=ReviewRole.MAINTAINABILITY,
                agent="a3",
                position=DissentPosition.APPROVE,
                confidence=0.1,
            ),
            _vote(
                role=ReviewRole.SKEPTIC,
                agent="a4",
                position=DissentPosition.REQUEST_CHANGES,
                confidence=0.95,
            ),
        ]
        brief = _build(votes, policy=SynthesisPolicy.WEIGHTED)
        assert brief.recommendation is Recommendation.REPAIR_FIRST
        assert len(brief.dissent) == 3
        assert {d.agent for d in brief.dissent} == {"a1", "a2", "a3"}

    def test_weighted_tie_escalates(self) -> None:
        votes = [
            _vote(
                role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE, confidence=0.5
            ),
            _vote(
                role=ReviewRole.SECURITY,
                agent="a2",
                position=DissentPosition.REQUEST_CHANGES,
                confidence=0.5,
            ),
        ]
        brief = _build(votes, policy=SynthesisPolicy.WEIGHTED)
        assert brief.recommendation is Recommendation.NEEDS_HUMAN_ATTENTION

    def test_negative_confidence_is_clamped_so_does_not_subtract_weight(self) -> None:
        # Defensive: a buggy upstream that emits negative confidence
        # must not be able to flip an outcome by subtracting weight from
        # its own bucket.
        votes = [
            _vote(
                role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE, confidence=-1.0
            ),
            _vote(
                role=ReviewRole.SECURITY,
                agent="a2",
                position=DissentPosition.APPROVE,
                confidence=0.1,
            ),
        ]
        brief = _build(votes, policy=SynthesisPolicy.WEIGHTED)
        assert brief.recommendation is Recommendation.APPROVE_CANDIDATE

    def test_above_one_confidence_is_capped_so_cannot_overpower_in_range_vote(self) -> None:
        # Codex C rev 1 finding P2: _weighted previously only clamped the
        # lower bound, so a 1.5 confidence vote could overpower a 1.0
        # opposite vote — inconsistent with the brief's [0,1] clamping.
        # With per-input clamping in place, both votes now compete at 1.0
        # weight and the tie escalates to NEEDS_HUMAN_ATTENTION.
        votes = [
            _vote(
                role=ReviewRole.LOGIC,
                agent="a1",
                position=DissentPosition.APPROVE,
                confidence=1.5,
            ),
            _vote(
                role=ReviewRole.SECURITY,
                agent="a2",
                position=DissentPosition.REQUEST_CHANGES,
                confidence=1.0,
            ),
        ]
        brief = _build(votes, policy=SynthesisPolicy.WEIGHTED)
        assert brief.recommendation is Recommendation.NEEDS_HUMAN_ATTENTION


# --- SYNTHESIZER_AGENT policy --------------------------------------------


class TestSynthesizerPolicy:
    def test_designated_synthesizer_position_wins_regardless_of_others(self) -> None:
        votes = [
            _vote(
                role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE, confidence=0.95
            ),
            _vote(
                role=ReviewRole.SECURITY,
                agent="a2",
                position=DissentPosition.APPROVE,
                confidence=0.95,
            ),
            _vote(
                role=ReviewRole.SYNTHESIZER,
                agent="syn",
                position=DissentPosition.DEFER,
                confidence=0.4,
            ),
        ]
        brief = _build(votes, policy=SynthesisPolicy.SYNTHESIZER_AGENT)
        assert brief.recommendation is Recommendation.NEEDS_HUMAN_ATTENTION
        # The two APPROVE voters now dissent (they disagree with the
        # synthesizer-driven NEEDS_HUMAN_ATTENTION recommendation).
        assert {d.agent for d in brief.dissent} == {"a1", "a2"}

    def test_no_synthesizer_in_panel_raises(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.APPROVE),
        ]
        with pytest.raises(ValueError, match="exactly one panel member"):
            _build(votes, policy=SynthesisPolicy.SYNTHESIZER_AGENT)

    def test_multiple_synthesizers_raise(self) -> None:
        votes = [
            _vote(role=ReviewRole.SYNTHESIZER, agent="s1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SYNTHESIZER, agent="s2", position=DissentPosition.APPROVE),
        ]
        with pytest.raises(ValueError, match="exactly one panel member"):
            _build(votes, policy=SynthesisPolicy.SYNTHESIZER_AGENT)


# --- UNANIMOUS_OR_ESCALATE policy ----------------------------------------


class TestUnanimousOrEscalatePolicy:
    def test_unanimous_yields_that_recommendation(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.REQUEST_CHANGES),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.REQUEST_CHANGES),
            _vote(
                role=ReviewRole.MAINTAINABILITY,
                agent="a3",
                position=DissentPosition.REQUEST_CHANGES,
            ),
        ]
        brief = _build(votes, policy=SynthesisPolicy.UNANIMOUS_OR_ESCALATE)
        assert brief.recommendation is Recommendation.REPAIR_FIRST
        assert brief.dissent == ()

    def test_any_disagreement_escalates(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.MAINTAINABILITY, agent="a3", position=DissentPosition.DEFER),
        ]
        brief = _build(votes, policy=SynthesisPolicy.UNANIMOUS_OR_ESCALATE)
        assert brief.recommendation is Recommendation.NEEDS_HUMAN_ATTENTION
        # The two APPROVE voters dissent; the DEFER voter aligns with
        # the NEEDS_HUMAN_ATTENTION recommendation.
        assert {d.agent for d in brief.dissent} == {"a1", "a2"}


# --- aggregation: confidence + disagreement -------------------------------


class TestAggregation:
    def test_overall_confidence_is_arithmetic_mean(self) -> None:
        votes = [
            _vote(
                role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE, confidence=0.8
            ),
            _vote(
                role=ReviewRole.SECURITY,
                agent="a2",
                position=DissentPosition.APPROVE,
                confidence=0.6,
            ),
        ]
        brief = _build(votes, policy=SynthesisPolicy.MAJORITY)
        assert brief.overall_confidence == pytest.approx(0.7)

    def test_disagreement_zero_when_unanimous(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.APPROVE),
        ]
        brief = _build(votes, policy=SynthesisPolicy.MAJORITY)
        assert brief.disagreement_score == 0.0

    def test_disagreement_half_when_split_two_two(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.APPROVE),
            _vote(
                role=ReviewRole.MAINTAINABILITY,
                agent="a3",
                position=DissentPosition.REQUEST_CHANGES,
            ),
            _vote(role=ReviewRole.SKEPTIC, agent="a4", position=DissentPosition.REQUEST_CHANGES),
        ]
        brief = _build(votes, policy=SynthesisPolicy.MAJORITY)
        assert brief.disagreement_score == 0.5

    def test_disagreement_two_thirds_when_three_way_split(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.REQUEST_CHANGES),
            _vote(role=ReviewRole.MAINTAINABILITY, agent="a3", position=DissentPosition.DEFER),
        ]
        brief = _build(votes, policy=SynthesisPolicy.MAJORITY)
        # Largest faction = 1 of 3 → disagreement = 1 - 1/3 ≈ 0.6667.
        assert brief.disagreement_score == pytest.approx(2 / 3, abs=1e-4)


# --- packet_sha determinism ----------------------------------------------


class TestPacketSha:
    def _stable_votes(self) -> list[PanelVote]:
        return [
            _vote(
                role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE, confidence=0.8
            ),
            _vote(
                role=ReviewRole.SECURITY,
                agent="a2",
                position=DissentPosition.APPROVE,
                confidence=0.7,
            ),
        ]

    def test_packet_sha_is_deterministic(self) -> None:
        b1 = _build(self._stable_votes())
        b2 = _build(self._stable_votes())
        assert b1.packet_sha == b2.packet_sha
        assert len(b1.packet_sha) == 64  # sha256 hex

    def test_packet_sha_changes_when_recommendation_changes(self) -> None:
        approving = self._stable_votes()
        rejecting = [
            _vote(
                role=ReviewRole.LOGIC,
                agent="a1",
                position=DissentPosition.REQUEST_CHANGES,
                confidence=0.8,
            ),
            _vote(
                role=ReviewRole.SECURITY,
                agent="a2",
                position=DissentPosition.REQUEST_CHANGES,
                confidence=0.7,
            ),
        ]
        assert _build(approving).packet_sha != _build(rejecting).packet_sha

    def test_packet_sha_changes_when_head_sha_changes(self) -> None:
        # SHA-bound: changing head_sha must produce a different packet_sha
        # so settlement on a stale head fails the merge_arbiter check.
        b1 = _build(self._stable_votes(), head_sha="aaa")
        b2 = _build(self._stable_votes(), head_sha="bbb")
        assert b1.packet_sha != b2.packet_sha

    def test_packet_sha_excludes_itself_from_preimage(self) -> None:
        # Re-computing on the brief's own to_dict() (which now contains
        # packet_sha) must yield the same value the builder set.
        brief = _build(self._stable_votes())
        recomputed = compute_packet_sha(brief)
        assert recomputed == brief.packet_sha


# --- ordering + advisory invariants ---------------------------------------


class TestStructure:
    def test_role_findings_and_agent_roster_preserve_input_order(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="zeta", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="alpha", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SKEPTIC, agent="mu", position=DissentPosition.APPROVE),
        ]
        brief = _build(votes)
        assert brief.agent_roster == ("zeta", "alpha", "mu")
        assert tuple(f.role for f in brief.role_findings) == (
            ReviewRole.LOGIC,
            ReviewRole.SECURITY,
            ReviewRole.SKEPTIC,
        )

    def test_advisory_only_remains_true(self) -> None:
        # Brief defaults to advisory_only=True; the builder must not flip
        # it. This is the safety boundary from the design brief.
        brief = _build(
            [_vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE)],
        )
        assert brief.advisory_only is True


# --- input validation ----------------------------------------------------


class TestValidation:
    def test_empty_votes_raise(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _build([])


# --- output_roles coverage (codex rev 1 finding P1) ----------------------


class TestOutputRoleCoverage:
    """``output_roles`` enforces ``PRReviewProtocol.output_roles`` contract:
    each declared role must appear in exactly one vote. Panel members with
    roles outside ``output_roles`` are tolerated as extras (e.g., a
    SYNTHESIZER panelist not declared as an output section).
    """

    def test_default_none_skips_coverage_check(self) -> None:
        # Backwards-compatible: existing callers that don't pass
        # output_roles get no enforcement (matches pre-revision behavior).
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
        ]
        brief = _build(votes)
        assert brief.recommendation is Recommendation.APPROVE_CANDIDATE

    def test_full_coverage_passes(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.MAINTAINABILITY, agent="a3", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SKEPTIC, agent="a4", position=DissentPosition.APPROVE),
        ]
        brief = build_brief(
            votes=votes,
            pr_number=6306,
            repo="synaptent/aragora",
            head_sha="x",
            base_sha="y",
            top_line="",
            validation_summary="",
            generated_at="2026-04-20T12:00:00+00:00",
            synthesis_policy=SynthesisPolicy.MAJORITY,
            output_roles=(
                ReviewRole.LOGIC,
                ReviewRole.SECURITY,
                ReviewRole.MAINTAINABILITY,
                ReviewRole.SKEPTIC,
            ),
        )
        assert brief.recommendation is Recommendation.APPROVE_CANDIDATE

    def test_missing_required_role_raises(self) -> None:
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.APPROVE),
        ]
        with pytest.raises(ValueError, match="missing roles.*maintainability_reviewer"):
            build_brief(
                votes=votes,
                pr_number=6306,
                repo="synaptent/aragora",
                head_sha="x",
                base_sha="y",
                top_line="",
                validation_summary="",
                generated_at="2026-04-20T12:00:00+00:00",
                synthesis_policy=SynthesisPolicy.MAJORITY,
                output_roles=(
                    ReviewRole.LOGIC,
                    ReviewRole.SECURITY,
                    ReviewRole.MAINTAINABILITY,
                ),
            )

    def test_duplicated_required_role_raises(self) -> None:
        # Two LOGIC votes in the panel; the brief can't decide which one to
        # render in the LOGIC section.
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.LOGIC, agent="a2", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a3", position=DissentPosition.APPROVE),
        ]
        with pytest.raises(ValueError, match="duplicated roles.*logic_reviewer"):
            build_brief(
                votes=votes,
                pr_number=6306,
                repo="synaptent/aragora",
                head_sha="x",
                base_sha="y",
                top_line="",
                validation_summary="",
                generated_at="2026-04-20T12:00:00+00:00",
                synthesis_policy=SynthesisPolicy.MAJORITY,
                output_roles=(ReviewRole.LOGIC, ReviewRole.SECURITY),
            )

    def test_extra_panel_role_tolerated(self) -> None:
        # SYNTHESIZER is in the panel (e.g., for SYNTHESIZER_AGENT policy)
        # but NOT in output_roles. Should not raise.
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SYNTHESIZER, agent="syn", position=DissentPosition.APPROVE),
        ]
        brief = build_brief(
            votes=votes,
            pr_number=6306,
            repo="synaptent/aragora",
            head_sha="x",
            base_sha="y",
            top_line="",
            validation_summary="",
            generated_at="2026-04-20T12:00:00+00:00",
            synthesis_policy=SynthesisPolicy.SYNTHESIZER_AGENT,
            output_roles=(ReviewRole.LOGIC, ReviewRole.SECURITY),
        )
        assert brief.recommendation is Recommendation.APPROVE_CANDIDATE
        # The SYNTHESIZER finding still appears in role_findings — extras
        # are not stripped, only validated against output_roles.
        assert any(f.role is ReviewRole.SYNTHESIZER for f in brief.role_findings)


# --- confidence clamping (codex rev 1 finding P2) ------------------------


class TestConfidenceClamping:
    """``overall_confidence`` is documented as 0.0..1.0. Per-input clamping
    in ``_aggregate_confidence`` mirrors ``_weighted``'s defensive treatment
    so the brief stays within contract even if upstream emits malformed
    confidence values.
    """

    def test_negative_confidence_clamped_to_zero(self) -> None:
        votes = [
            _vote(
                role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE, confidence=-0.5
            ),
            _vote(
                role=ReviewRole.SECURITY,
                agent="a2",
                position=DissentPosition.APPROVE,
                confidence=0.8,
            ),
        ]
        brief = _build(votes)
        # -0.5 clamps to 0.0, so mean = (0.0 + 0.8) / 2 = 0.4 (not -0.35 / 2 = +0.15).
        assert brief.overall_confidence == pytest.approx(0.4)
        assert 0.0 <= brief.overall_confidence <= 1.0

    def test_above_one_confidence_clamped_to_one(self) -> None:
        votes = [
            _vote(
                role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE, confidence=1.5
            ),
            _vote(
                role=ReviewRole.SECURITY,
                agent="a2",
                position=DissentPosition.APPROVE,
                confidence=0.6,
            ),
        ]
        brief = _build(votes)
        # 1.5 clamps to 1.0, so mean = (1.0 + 0.6) / 2 = 0.8 (not 1.05).
        assert brief.overall_confidence == pytest.approx(0.8)
        assert 0.0 <= brief.overall_confidence <= 1.0

    def test_in_range_confidence_unchanged(self) -> None:
        # Clamping must be a no-op for well-formed input.
        votes = [
            _vote(
                role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE, confidence=0.7
            ),
            _vote(
                role=ReviewRole.SECURITY,
                agent="a2",
                position=DissentPosition.APPROVE,
                confidence=0.5,
            ),
        ]
        brief = _build(votes)
        assert brief.overall_confidence == pytest.approx(0.6)


# --- Severity-gated verdict (#6505) --------------------------------------


class TestSeverityGate:
    """Gate downgrades REPAIR_FIRST → APPROVE_WITH_FOLLOWUPS when no high findings.

    Core behavioral fix from the Mode 3 rubric calibration epic. The
    primary verdict logic is unchanged; only the specific
    ``REPAIR_FIRST`` → ``APPROVE_WITH_FOLLOWUPS`` downgrade is added.
    """

    def _request_changes_votes(self) -> list[PanelVote]:
        # Majority request_changes → primary verdict is REPAIR_FIRST.
        return [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.REQUEST_CHANGES),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.REQUEST_CHANGES),
            _vote(
                role=ReviewRole.MAINTAINABILITY,
                agent="a3",
                position=DissentPosition.REQUEST_CHANGES,
            ),
        ]

    def test_no_high_severity_downgrades_to_approve_with_followups(self) -> None:
        votes = self._request_changes_votes()
        brief = _build(
            votes,
            findings_severity_counts={"high": 0, "medium": 2, "low": 5},
        )
        assert brief.recommendation is Recommendation.APPROVE_WITH_FOLLOWUPS

    def test_high_severity_preserves_repair_first(self) -> None:
        votes = self._request_changes_votes()
        brief = _build(
            votes,
            findings_severity_counts={"high": 1, "medium": 0, "low": 0},
        )
        assert brief.recommendation is Recommendation.REPAIR_FIRST

    def test_legacy_caller_without_severity_preserves_repair_first(self) -> None:
        # Backwards compatibility: briefs built without severity counts
        # (legacy path, degraded runs, older callers) must keep the old
        # three-class behavior.
        votes = self._request_changes_votes()
        brief = _build(votes)  # no findings_severity_counts passed
        assert brief.recommendation is Recommendation.REPAIR_FIRST

    def test_malformed_severity_map_missing_high_key_preserves_repair_first(self) -> None:
        # Defensive: if upstream ever emits a counts dict without the
        # canonical key, do not guess. Preserve the conservative verdict.
        votes = self._request_changes_votes()
        brief = _build(
            votes,
            findings_severity_counts={"medium": 3, "low": 1},
        )
        assert brief.recommendation is Recommendation.REPAIR_FIRST

    def test_approve_primary_unchanged_by_gate(self) -> None:
        # The gate only ever fires on REPAIR_FIRST. APPROVE_CANDIDATE
        # cannot be downgraded (and should not be). Confirm regardless
        # of severity input.
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.APPROVE),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.APPROVE),
        ]
        for counts in (None, {"high": 0, "medium": 0, "low": 0}, {"high": 5}):
            brief = _build(votes, findings_severity_counts=counts)
            assert brief.recommendation is Recommendation.APPROVE_CANDIDATE

    def test_defer_primary_unchanged_by_gate(self) -> None:
        # DEFER → NEEDS_HUMAN_ATTENTION is never downgraded; operator
        # attention is a signal in its own right, not a symptom of
        # finding severity.
        votes = [
            _vote(role=ReviewRole.LOGIC, agent="a1", position=DissentPosition.DEFER),
            _vote(role=ReviewRole.SECURITY, agent="a2", position=DissentPosition.DEFER),
        ]
        brief = _build(
            votes,
            findings_severity_counts={"high": 0, "medium": 0, "low": 0},
        )
        assert brief.recommendation is Recommendation.NEEDS_HUMAN_ATTENTION
