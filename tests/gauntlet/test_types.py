"""Comprehensive tests for aragora.gauntlet.types module."""

from __future__ import annotations

from datetime import datetime

import pytest

from aragora.gauntlet.types import (
    BaseFinding,
    GauntletPhase,
    GauntletSeverity,
    InputType,
    RiskSummary,
    SeverityLevel,
    Verdict,
)


# ---------------------------------------------------------------------------
# InputType enum
# ---------------------------------------------------------------------------


class TestInputType:
    """Tests for the InputType enum."""

    def test_has_seven_members(self) -> None:
        assert len(InputType) == 7

    @pytest.mark.parametrize(
        "member, expected_value",
        [
            (InputType.SPEC, "spec"),
            (InputType.ARCHITECTURE, "architecture"),
            (InputType.POLICY, "policy"),
            (InputType.CODE, "code"),
            (InputType.STRATEGY, "strategy"),
            (InputType.CONTRACT, "contract"),
            (InputType.CUSTOM, "custom"),
        ],
    )
    def test_string_values(self, member: InputType, expected_value: str) -> None:
        assert member.value == expected_value

    def test_members_are_unique(self) -> None:
        values = [m.value for m in InputType]
        assert len(values) == len(set(values))

    def test_lookup_by_value(self) -> None:
        assert InputType("spec") is InputType.SPEC
        assert InputType("code") is InputType.CODE


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------


class TestVerdict:
    """Tests for the Verdict enum."""

    def test_has_seven_members(self) -> None:
        assert len(Verdict) == 7

    @pytest.mark.parametrize(
        "member, expected_value",
        [
            (Verdict.PASS, "pass"),
            (Verdict.APPROVED, "approved"),
            (Verdict.CONDITIONAL, "conditional"),
            (Verdict.APPROVED_WITH_CONDITIONS, "approved_with_conditions"),
            (Verdict.NEEDS_REVIEW, "needs_review"),
            (Verdict.FAIL, "fail"),
            (Verdict.REJECTED, "rejected"),
        ],
    )
    def test_string_values(self, member: Verdict, expected_value: str) -> None:
        assert member.value == expected_value

    # -- is_passing ---------------------------------------------------------

    def test_pass_is_passing(self) -> None:
        assert Verdict.PASS.is_passing is True

    def test_approved_is_passing(self) -> None:
        assert Verdict.APPROVED.is_passing is True

    @pytest.mark.parametrize(
        "verdict",
        [
            Verdict.CONDITIONAL,
            Verdict.APPROVED_WITH_CONDITIONS,
            Verdict.NEEDS_REVIEW,
            Verdict.FAIL,
            Verdict.REJECTED,
        ],
    )
    def test_non_passing_verdicts(self, verdict: Verdict) -> None:
        assert verdict.is_passing is False

    # -- is_conditional -----------------------------------------------------

    def test_conditional_is_conditional(self) -> None:
        assert Verdict.CONDITIONAL.is_conditional is True

    def test_approved_with_conditions_is_conditional(self) -> None:
        assert Verdict.APPROVED_WITH_CONDITIONS.is_conditional is True

    def test_needs_review_is_conditional(self) -> None:
        assert Verdict.NEEDS_REVIEW.is_conditional is True

    @pytest.mark.parametrize(
        "verdict",
        [Verdict.PASS, Verdict.APPROVED, Verdict.FAIL, Verdict.REJECTED],
    )
    def test_non_conditional_verdicts(self, verdict: Verdict) -> None:
        assert verdict.is_conditional is False

    # -- is_failing ---------------------------------------------------------

    def test_fail_is_failing(self) -> None:
        assert Verdict.FAIL.is_failing is True

    def test_rejected_is_failing(self) -> None:
        assert Verdict.REJECTED.is_failing is True

    @pytest.mark.parametrize(
        "verdict",
        [
            Verdict.PASS,
            Verdict.APPROVED,
            Verdict.CONDITIONAL,
            Verdict.APPROVED_WITH_CONDITIONS,
            Verdict.NEEDS_REVIEW,
        ],
    )
    def test_non_failing_verdicts(self, verdict: Verdict) -> None:
        assert verdict.is_failing is False

    # -- mutual exclusivity -------------------------------------------------

    def test_mutual_exclusivity_passing_and_failing(self) -> None:
        """No verdict should be both passing and failing."""
        for verdict in Verdict:
            assert not (verdict.is_passing and verdict.is_failing), (
                f"{verdict} is both passing and failing"
            )

    def test_mutual_exclusivity_passing_and_conditional(self) -> None:
        """No verdict should be both passing and conditional."""
        for verdict in Verdict:
            assert not (verdict.is_passing and verdict.is_conditional), (
                f"{verdict} is both passing and conditional"
            )

    def test_mutual_exclusivity_conditional_and_failing(self) -> None:
        """No verdict should be both conditional and failing."""
        for verdict in Verdict:
            assert not (verdict.is_conditional and verdict.is_failing), (
                f"{verdict} is both conditional and failing"
            )

    def test_every_verdict_falls_into_exactly_one_category(self) -> None:
        """Every verdict must be in exactly one of: passing, conditional, failing."""
        for verdict in Verdict:
            categories = sum([verdict.is_passing, verdict.is_conditional, verdict.is_failing])
            assert categories == 1, f"{verdict} belongs to {categories} categories (expected 1)"


# ---------------------------------------------------------------------------
# SeverityLevel enum
# ---------------------------------------------------------------------------


class TestSeverityLevel:
    """Tests for the SeverityLevel enum."""

    def test_has_five_members(self) -> None:
        assert len(SeverityLevel) == 5

    @pytest.mark.parametrize(
        "member, expected_value",
        [
            (SeverityLevel.CRITICAL, "critical"),
            (SeverityLevel.HIGH, "high"),
            (SeverityLevel.MEDIUM, "medium"),
            (SeverityLevel.LOW, "low"),
            (SeverityLevel.INFO, "info"),
        ],
    )
    def test_string_values(self, member: SeverityLevel, expected_value: str) -> None:
        assert member.value == expected_value

    # -- numeric_value property ---------------------------------------------

    @pytest.mark.parametrize(
        "member, expected_numeric",
        [
            (SeverityLevel.CRITICAL, 0.95),
            (SeverityLevel.HIGH, 0.75),
            (SeverityLevel.MEDIUM, 0.50),
            (SeverityLevel.LOW, 0.25),
            (SeverityLevel.INFO, 0.10),
        ],
    )
    def test_numeric_value(self, member: SeverityLevel, expected_numeric: float) -> None:
        assert member.numeric_value == pytest.approx(expected_numeric)

    def test_numeric_values_are_descending(self) -> None:
        """Higher severity should have higher numeric value."""
        ordered = [
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
            SeverityLevel.MEDIUM,
            SeverityLevel.LOW,
            SeverityLevel.INFO,
        ]
        for i in range(len(ordered) - 1):
            assert ordered[i].numeric_value > ordered[i + 1].numeric_value

    # -- from_numeric classmethod -------------------------------------------

    @pytest.mark.parametrize(
        "value, expected",
        [
            # CRITICAL boundary: >= 0.9
            (1.0, SeverityLevel.CRITICAL),
            (0.95, SeverityLevel.CRITICAL),
            (0.9, SeverityLevel.CRITICAL),
            # HIGH boundary: >= 0.7
            (0.89, SeverityLevel.HIGH),
            (0.75, SeverityLevel.HIGH),
            (0.7, SeverityLevel.HIGH),
            # MEDIUM boundary: >= 0.4
            (0.69, SeverityLevel.MEDIUM),
            (0.5, SeverityLevel.MEDIUM),
            (0.4, SeverityLevel.MEDIUM),
            # LOW boundary: >= 0.2
            (0.39, SeverityLevel.LOW),
            (0.25, SeverityLevel.LOW),
            (0.2, SeverityLevel.LOW),
            # INFO: < 0.2
            (0.19, SeverityLevel.INFO),
            (0.1, SeverityLevel.INFO),
            (0.0, SeverityLevel.INFO),
        ],
    )
    def test_from_numeric(self, value: float, expected: SeverityLevel) -> None:
        assert SeverityLevel.from_numeric(value) is expected

    def test_from_numeric_negative(self) -> None:
        """Negative values should return INFO."""
        assert SeverityLevel.from_numeric(-0.5) is SeverityLevel.INFO

    def test_from_numeric_above_one(self) -> None:
        """Values above 1.0 should still return CRITICAL."""
        assert SeverityLevel.from_numeric(1.5) is SeverityLevel.CRITICAL


# ---------------------------------------------------------------------------
# GauntletPhase enum
# ---------------------------------------------------------------------------


class TestGauntletPhase:
    """Tests for the GauntletPhase enum."""

    def test_has_eleven_members(self) -> None:
        assert len(GauntletPhase) == 11

    @pytest.mark.parametrize(
        "member, expected_value",
        [
            (GauntletPhase.NOT_STARTED, "not_started"),
            (GauntletPhase.INITIALIZATION, "initialization"),
            (GauntletPhase.RISK_ASSESSMENT, "risk_assessment"),
            (GauntletPhase.SCENARIO_ANALYSIS, "scenario_analysis"),
            (GauntletPhase.RED_TEAM, "red_team"),
            (GauntletPhase.ADVERSARIAL_PROBING, "adversarial_probing"),
            (GauntletPhase.DEEP_AUDIT, "deep_audit"),
            (GauntletPhase.FORMAL_VERIFICATION, "formal_verification"),
            (GauntletPhase.SYNTHESIS, "synthesis"),
            (GauntletPhase.COMPLETE, "complete"),
            (GauntletPhase.FAILED, "failed"),
        ],
    )
    def test_string_values(self, member: GauntletPhase, expected_value: str) -> None:
        assert member.value == expected_value

    def test_members_are_unique(self) -> None:
        values = [m.value for m in GauntletPhase]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# BaseFinding dataclass
# ---------------------------------------------------------------------------


class TestBaseFinding:
    """Tests for the BaseFinding dataclass."""

    def _make_finding(self, **overrides) -> BaseFinding:
        """Helper to create a BaseFinding with default values."""
        defaults = {
            "id": "F-001",
            "title": "SQL Injection",
            "description": "Unparameterized query in login handler",
            "severity": SeverityLevel.HIGH,
            "category": "security",
            "source": "red_team",
        }
        defaults.update(overrides)
        return BaseFinding(**defaults)

    def test_creation_with_required_fields(self) -> None:
        finding = self._make_finding()
        assert finding.id == "F-001"
        assert finding.title == "SQL Injection"
        assert finding.description == "Unparameterized query in login handler"
        assert finding.severity is SeverityLevel.HIGH
        assert finding.category == "security"
        assert finding.source == "red_team"

    def test_default_evidence(self) -> None:
        finding = self._make_finding()
        assert finding.evidence == ""

    def test_default_mitigation(self) -> None:
        finding = self._make_finding()
        assert finding.mitigation is None

    def test_default_is_verified(self) -> None:
        finding = self._make_finding()
        assert finding.is_verified is False

    def test_default_verification_method(self) -> None:
        finding = self._make_finding()
        assert finding.verification_method is None

    def test_created_at_is_auto_generated(self) -> None:
        finding = self._make_finding()
        assert finding.created_at is not None
        # Should be a valid ISO-format string
        parsed = datetime.fromisoformat(finding.created_at)
        assert isinstance(parsed, datetime)

    def test_custom_optional_fields(self) -> None:
        finding = self._make_finding(
            evidence="Found in /login endpoint",
            mitigation="Use parameterized queries",
            is_verified=True,
            verification_method="manual_review",
        )
        assert finding.evidence == "Found in /login endpoint"
        assert finding.mitigation == "Use parameterized queries"
        assert finding.is_verified is True
        assert finding.verification_method == "manual_review"

    # -- severity_numeric property ------------------------------------------

    @pytest.mark.parametrize(
        "severity, expected_numeric",
        [
            (SeverityLevel.CRITICAL, 0.95),
            (SeverityLevel.HIGH, 0.75),
            (SeverityLevel.MEDIUM, 0.50),
            (SeverityLevel.LOW, 0.25),
            (SeverityLevel.INFO, 0.10),
        ],
    )
    def test_severity_numeric(self, severity: SeverityLevel, expected_numeric: float) -> None:
        finding = self._make_finding(severity=severity)
        assert finding.severity_numeric == pytest.approx(expected_numeric)

    # -- to_dict method -----------------------------------------------------

    def test_to_dict_keys(self) -> None:
        finding = self._make_finding()
        d = finding.to_dict()
        expected_keys = {
            "id",
            "title",
            "description",
            "severity",
            "severity_numeric",
            "category",
            "source",
            "evidence",
            "mitigation",
            "is_verified",
            "verification_method",
            "created_at",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_severity_is_string(self) -> None:
        finding = self._make_finding(severity=SeverityLevel.CRITICAL)
        d = finding.to_dict()
        assert d["severity"] == "critical"
        assert isinstance(d["severity"], str)

    def test_to_dict_severity_numeric_included(self) -> None:
        finding = self._make_finding(severity=SeverityLevel.MEDIUM)
        d = finding.to_dict()
        assert d["severity_numeric"] == pytest.approx(0.50)

    def test_to_dict_values_match_fields(self) -> None:
        finding = self._make_finding(
            evidence="proof",
            mitigation="fix it",
            is_verified=True,
            verification_method="static_analysis",
        )
        d = finding.to_dict()
        assert d["id"] == "F-001"
        assert d["title"] == "SQL Injection"
        assert d["description"] == "Unparameterized query in login handler"
        assert d["category"] == "security"
        assert d["source"] == "red_team"
        assert d["evidence"] == "proof"
        assert d["mitigation"] == "fix it"
        assert d["is_verified"] is True
        assert d["verification_method"] == "static_analysis"
        assert d["created_at"] == finding.created_at

    def test_to_dict_defaults(self) -> None:
        finding = self._make_finding()
        d = finding.to_dict()
        assert d["evidence"] == ""
        assert d["mitigation"] is None
        assert d["is_verified"] is False
        assert d["verification_method"] is None

    def test_two_findings_have_different_created_at_or_same_instant(self) -> None:
        """Each finding gets its own timestamp via default_factory."""
        f1 = self._make_finding(id="F-001")
        f2 = self._make_finding(id="F-002")
        # Both should be valid ISO timestamps
        datetime.fromisoformat(f1.created_at)
        datetime.fromisoformat(f2.created_at)


# ---------------------------------------------------------------------------
# RiskSummary dataclass
# ---------------------------------------------------------------------------


class TestRiskSummary:
    """Tests for the RiskSummary dataclass."""

    def test_default_all_zeros(self) -> None:
        summary = RiskSummary()
        assert summary.critical == 0
        assert summary.high == 0
        assert summary.medium == 0
        assert summary.low == 0
        assert summary.info == 0

    def test_total_empty(self) -> None:
        summary = RiskSummary()
        assert summary.total == 0

    def test_weighted_score_empty(self) -> None:
        summary = RiskSummary()
        assert summary.weighted_score == pytest.approx(0.0)

    # -- add_finding --------------------------------------------------------

    def test_add_finding_critical(self) -> None:
        summary = RiskSummary()
        summary.add_finding(SeverityLevel.CRITICAL)
        assert summary.critical == 1
        assert summary.high == 0
        assert summary.medium == 0
        assert summary.low == 0
        assert summary.info == 0

    def test_add_finding_high(self) -> None:
        summary = RiskSummary()
        summary.add_finding(SeverityLevel.HIGH)
        assert summary.high == 1

    def test_add_finding_medium(self) -> None:
        summary = RiskSummary()
        summary.add_finding(SeverityLevel.MEDIUM)
        assert summary.medium == 1

    def test_add_finding_low(self) -> None:
        summary = RiskSummary()
        summary.add_finding(SeverityLevel.LOW)
        assert summary.low == 1

    def test_add_finding_info(self) -> None:
        summary = RiskSummary()
        summary.add_finding(SeverityLevel.INFO)
        assert summary.info == 1

    def test_add_multiple_findings_same_severity(self) -> None:
        summary = RiskSummary()
        for _ in range(5):
            summary.add_finding(SeverityLevel.HIGH)
        assert summary.high == 5
        assert summary.total == 5

    def test_add_findings_mixed_severities(self) -> None:
        summary = RiskSummary()
        summary.add_finding(SeverityLevel.CRITICAL)
        summary.add_finding(SeverityLevel.HIGH)
        summary.add_finding(SeverityLevel.MEDIUM)
        summary.add_finding(SeverityLevel.LOW)
        summary.add_finding(SeverityLevel.INFO)
        assert summary.critical == 1
        assert summary.high == 1
        assert summary.medium == 1
        assert summary.low == 1
        assert summary.info == 1

    # -- total property -----------------------------------------------------

    def test_total_after_adding(self) -> None:
        summary = RiskSummary()
        summary.add_finding(SeverityLevel.CRITICAL)
        summary.add_finding(SeverityLevel.CRITICAL)
        summary.add_finding(SeverityLevel.HIGH)
        summary.add_finding(SeverityLevel.LOW)
        assert summary.total == 4

    def test_total_matches_sum_of_fields(self) -> None:
        summary = RiskSummary(critical=3, high=2, medium=5, low=1, info=4)
        assert summary.total == 3 + 2 + 5 + 1 + 4

    # -- weighted_score property --------------------------------------------

    def test_weighted_score_single_critical(self) -> None:
        summary = RiskSummary(critical=1)
        assert summary.weighted_score == pytest.approx(10.0)

    def test_weighted_score_single_high(self) -> None:
        summary = RiskSummary(high=1)
        assert summary.weighted_score == pytest.approx(5.0)

    def test_weighted_score_single_medium(self) -> None:
        summary = RiskSummary(medium=1)
        assert summary.weighted_score == pytest.approx(2.0)

    def test_weighted_score_single_low(self) -> None:
        summary = RiskSummary(low=1)
        assert summary.weighted_score == pytest.approx(1.0)

    def test_weighted_score_info_not_counted(self) -> None:
        """Info findings have zero weight in the score."""
        summary = RiskSummary(info=100)
        assert summary.weighted_score == pytest.approx(0.0)

    def test_weighted_score_formula(self) -> None:
        """critical*10 + high*5 + medium*2 + low*1"""
        summary = RiskSummary(critical=2, high=3, medium=4, low=5, info=10)
        expected = 2 * 10 + 3 * 5 + 4 * 2 + 5 * 1
        assert summary.weighted_score == pytest.approx(expected)
        assert summary.weighted_score == pytest.approx(48.0)

    # -- to_dict method -----------------------------------------------------

    def test_to_dict_keys(self) -> None:
        summary = RiskSummary()
        d = summary.to_dict()
        expected_keys = {
            "critical",
            "high",
            "medium",
            "low",
            "info",
            "total",
            "weighted_score",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_includes_total_and_weighted_score(self) -> None:
        summary = RiskSummary(critical=1, high=2, medium=3, low=4, info=5)
        d = summary.to_dict()
        assert d["total"] == 1 + 2 + 3 + 4 + 5
        expected_score = 1 * 10 + 2 * 5 + 3 * 2 + 4 * 1
        assert d["weighted_score"] == pytest.approx(expected_score)

    def test_to_dict_empty(self) -> None:
        summary = RiskSummary()
        d = summary.to_dict()
        assert d["critical"] == 0
        assert d["high"] == 0
        assert d["medium"] == 0
        assert d["low"] == 0
        assert d["info"] == 0
        assert d["total"] == 0
        assert d["weighted_score"] == pytest.approx(0.0)

    def test_to_dict_values_match_fields(self) -> None:
        summary = RiskSummary(critical=7, high=3, medium=1, low=0, info=2)
        d = summary.to_dict()
        assert d["critical"] == 7
        assert d["high"] == 3
        assert d["medium"] == 1
        assert d["low"] == 0
        assert d["info"] == 2

    def test_to_dict_after_add_finding(self) -> None:
        summary = RiskSummary()
        summary.add_finding(SeverityLevel.CRITICAL)
        summary.add_finding(SeverityLevel.MEDIUM)
        d = summary.to_dict()
        assert d["critical"] == 1
        assert d["medium"] == 1
        assert d["total"] == 2
        assert d["weighted_score"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# GauntletSeverity alias
# ---------------------------------------------------------------------------


class TestGauntletSeverityAlias:
    """Tests for the GauntletSeverity type alias."""

    def test_alias_is_severity_level(self) -> None:
        assert GauntletSeverity is SeverityLevel

    def test_alias_members_accessible(self) -> None:
        assert GauntletSeverity.CRITICAL is SeverityLevel.CRITICAL
        assert GauntletSeverity.HIGH is SeverityLevel.HIGH
        assert GauntletSeverity.MEDIUM is SeverityLevel.MEDIUM
        assert GauntletSeverity.LOW is SeverityLevel.LOW
        assert GauntletSeverity.INFO is SeverityLevel.INFO

    def test_alias_numeric_value(self) -> None:
        assert GauntletSeverity.CRITICAL.numeric_value == pytest.approx(0.95)

    def test_alias_from_numeric(self) -> None:
        result = GauntletSeverity.from_numeric(0.8)
        assert result is SeverityLevel.HIGH
