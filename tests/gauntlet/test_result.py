"""Tests for aragora.gauntlet.result module.

Covers:
- Vulnerability creation, defaults, risk_score, to_dict
- RiskSummary (total, weighted_score, add_finding, to_dict)
- AttackSummary to_dict with success_rate (including division by zero)
- ProbeSummary to_dict
- ScenarioSummary to_dict with defaults
- GauntletResult creation, add_vulnerability, calculate_verdict, get_critical_vulnerabilities, to_dict
"""

from __future__ import annotations

import pytest

from aragora.gauntlet.result import (
    AttackSummary,
    GauntletResult,
    ProbeSummary,
    RiskSummary,
    ScenarioSummary,
    SeverityLevel,
    Verdict,
    Vulnerability,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vulnerability(
    *,
    id: str = "VULN-001",
    title: str = "Test Vulnerability",
    description: str = "A test vulnerability",
    severity: SeverityLevel = SeverityLevel.MEDIUM,
    category: str = "injection",
    source: str = "red_team",
    exploitability: float = 0.5,
    impact: float = 0.5,
    **kwargs,
) -> Vulnerability:
    """Create a Vulnerability with sensible defaults for testing."""
    return Vulnerability(
        id=id,
        title=title,
        description=description,
        severity=severity,
        category=category,
        source=source,
        exploitability=exploitability,
        impact=impact,
        **kwargs,
    )


def _make_gauntlet_result(**kwargs) -> GauntletResult:
    """Create a GauntletResult with sensible defaults for testing."""
    defaults = {
        "gauntlet_id": "gauntlet-001",
        "input_hash": "abc123hash",
        "input_summary": "Test input summary",
        "started_at": "2025-01-01T00:00:00+00:00",
    }
    defaults.update(kwargs)
    return GauntletResult(**defaults)


# ===========================================================================
# Vulnerability tests
# ===========================================================================


class TestVulnerability:
    """Tests for the Vulnerability dataclass."""

    def test_creation_with_required_fields(self):
        vuln = Vulnerability(
            id="V-1",
            title="SQL Injection",
            description="Input not sanitized",
            severity=SeverityLevel.HIGH,
            category="injection",
            source="scanner",
        )
        assert vuln.id == "V-1"
        assert vuln.title == "SQL Injection"
        assert vuln.description == "Input not sanitized"
        assert vuln.severity == SeverityLevel.HIGH
        assert vuln.category == "injection"
        assert vuln.source == "scanner"

    def test_default_values(self):
        vuln = _make_vulnerability()
        assert vuln.evidence == ""
        assert vuln.exploit_scenario == ""
        assert vuln.mitigation == ""
        assert vuln.exploitability == 0.5
        assert vuln.impact == 0.5
        assert vuln.agent_name is None
        assert vuln.round_number is None
        assert vuln.scenario_id is None
        # created_at should be auto-generated as a non-empty ISO string
        assert isinstance(vuln.created_at, str)
        assert len(vuln.created_at) > 0

    def test_risk_score_calculation(self):
        vuln = _make_vulnerability(exploitability=0.8, impact=0.6)
        assert vuln.risk_score == pytest.approx(0.48)

    def test_risk_score_zero_exploitability(self):
        vuln = _make_vulnerability(exploitability=0.0, impact=1.0)
        assert vuln.risk_score == pytest.approx(0.0)

    def test_risk_score_zero_impact(self):
        vuln = _make_vulnerability(exploitability=1.0, impact=0.0)
        assert vuln.risk_score == pytest.approx(0.0)

    def test_risk_score_max(self):
        vuln = _make_vulnerability(exploitability=1.0, impact=1.0)
        assert vuln.risk_score == pytest.approx(1.0)

    def test_risk_score_default(self):
        vuln = _make_vulnerability()
        assert vuln.risk_score == pytest.approx(0.25)

    def test_to_dict_structure(self):
        vuln = _make_vulnerability(
            id="V-99",
            title="XSS",
            description="Cross-site scripting",
            severity=SeverityLevel.CRITICAL,
            category="xss",
            source="probe",
            evidence="<script>alert(1)</script>",
            exploit_scenario="Inject script via comment field",
            mitigation="Escape HTML output",
            exploitability=0.9,
            impact=0.7,
            agent_name="claude",
            round_number=2,
            scenario_id="scenario-42",
        )
        d = vuln.to_dict()
        assert d["id"] == "V-99"
        assert d["title"] == "XSS"
        assert d["description"] == "Cross-site scripting"
        assert d["severity"] == "critical"
        assert d["category"] == "xss"
        assert d["source"] == "probe"
        assert d["evidence"] == "<script>alert(1)</script>"
        assert d["exploit_scenario"] == "Inject script via comment field"
        assert d["mitigation"] == "Escape HTML output"
        assert d["exploitability"] == 0.9
        assert d["impact"] == 0.7
        assert d["risk_score"] == pytest.approx(0.63)
        assert d["agent_name"] == "claude"
        assert d["round_number"] == 2
        assert d["scenario_id"] == "scenario-42"
        assert "created_at" in d

    def test_to_dict_default_optional_fields(self):
        vuln = _make_vulnerability()
        d = vuln.to_dict()
        assert d["evidence"] == ""
        assert d["exploit_scenario"] == ""
        assert d["mitigation"] == ""
        assert d["agent_name"] is None
        assert d["round_number"] is None
        assert d["scenario_id"] is None

    def test_created_at_auto_generated(self):
        """Each Vulnerability gets its own created_at timestamp."""
        v1 = _make_vulnerability(id="V-A")
        v2 = _make_vulnerability(id="V-B")
        # Both should have timestamps (may or may not be identical depending on speed)
        assert isinstance(v1.created_at, str)
        assert isinstance(v2.created_at, str)


# ===========================================================================
# RiskSummary tests
# ===========================================================================


class TestRiskSummary:
    """Tests for the RiskSummary dataclass (aliased from types)."""

    def test_default_values(self):
        rs = RiskSummary()
        assert rs.critical == 0
        assert rs.high == 0
        assert rs.medium == 0
        assert rs.low == 0
        assert rs.info == 0

    def test_total_property(self):
        rs = RiskSummary(critical=1, high=2, medium=3, low=4, info=5)
        assert rs.total == 15

    def test_total_zero(self):
        rs = RiskSummary()
        assert rs.total == 0

    def test_weighted_score(self):
        rs = RiskSummary(critical=1, high=2, medium=3, low=4, info=5)
        # 1*10 + 2*5 + 3*2 + 4*1 = 10 + 10 + 6 + 4 = 30
        assert rs.weighted_score == 30

    def test_weighted_score_ignores_info(self):
        rs = RiskSummary(info=100)
        assert rs.weighted_score == 0

    def test_add_finding_critical(self):
        rs = RiskSummary()
        rs.add_finding(SeverityLevel.CRITICAL)
        assert rs.critical == 1
        assert rs.total == 1

    def test_add_finding_high(self):
        rs = RiskSummary()
        rs.add_finding(SeverityLevel.HIGH)
        assert rs.high == 1

    def test_add_finding_medium(self):
        rs = RiskSummary()
        rs.add_finding(SeverityLevel.MEDIUM)
        assert rs.medium == 1

    def test_add_finding_low(self):
        rs = RiskSummary()
        rs.add_finding(SeverityLevel.LOW)
        assert rs.low == 1

    def test_add_finding_info(self):
        rs = RiskSummary()
        rs.add_finding(SeverityLevel.INFO)
        assert rs.info == 1

    def test_to_dict(self):
        rs = RiskSummary(critical=2, high=3, medium=1, low=0, info=4)
        d = rs.to_dict()
        assert d == {
            "critical": 2,
            "high": 3,
            "medium": 1,
            "low": 0,
            "info": 4,
            "total": 10,
            "weighted_score": 2 * 10 + 3 * 5 + 1 * 2 + 0 * 1,
        }


# ===========================================================================
# AttackSummary tests
# ===========================================================================


class TestAttackSummary:
    """Tests for the AttackSummary dataclass."""

    def test_default_values(self):
        a = AttackSummary()
        assert a.total_attacks == 0
        assert a.successful_attacks == 0
        assert a.by_category == {}
        assert a.robustness_score == 1.0
        assert a.coverage_score == 0.0

    def test_to_dict_with_attacks(self):
        a = AttackSummary(
            total_attacks=10,
            successful_attacks=3,
            by_category={"injection": 2, "evasion": 1},
            robustness_score=0.7,
            coverage_score=0.8,
        )
        d = a.to_dict()
        assert d["total_attacks"] == 10
        assert d["successful_attacks"] == 3
        assert d["success_rate"] == pytest.approx(0.3)
        assert d["by_category"] == {"injection": 2, "evasion": 1}
        assert d["robustness_score"] == 0.7
        assert d["coverage_score"] == 0.8

    def test_to_dict_success_rate_division_by_zero(self):
        """When total_attacks is 0, success_rate should be 0 (not raise ZeroDivisionError)."""
        a = AttackSummary(total_attacks=0, successful_attacks=0)
        d = a.to_dict()
        assert d["success_rate"] == 0

    def test_to_dict_success_rate_all_successful(self):
        a = AttackSummary(total_attacks=5, successful_attacks=5)
        d = a.to_dict()
        assert d["success_rate"] == pytest.approx(1.0)

    def test_to_dict_by_category_isolated(self):
        """by_category dict should not be shared across instances."""
        a1 = AttackSummary()
        a2 = AttackSummary()
        a1.by_category["injection"] = 5
        assert a2.by_category == {}


# ===========================================================================
# ProbeSummary tests
# ===========================================================================


class TestProbeSummary:
    """Tests for the ProbeSummary dataclass."""

    def test_default_values(self):
        p = ProbeSummary()
        assert p.probes_run == 0
        assert p.vulnerabilities_found == 0
        assert p.by_category == {}
        assert p.vulnerability_rate == 0.0
        assert p.elo_penalty == 0.0

    def test_to_dict(self):
        p = ProbeSummary(
            probes_run=20,
            vulnerabilities_found=4,
            by_category={"logic": 2, "data": 2},
            vulnerability_rate=0.2,
            elo_penalty=-15.5,
        )
        d = p.to_dict()
        assert d == {
            "probes_run": 20,
            "vulnerabilities_found": 4,
            "vulnerability_rate": 0.2,
            "by_category": {"logic": 2, "data": 2},
            "elo_penalty": -15.5,
        }

    def test_to_dict_defaults(self):
        d = ProbeSummary().to_dict()
        assert d["probes_run"] == 0
        assert d["vulnerabilities_found"] == 0
        assert d["vulnerability_rate"] == 0.0
        assert d["by_category"] == {}
        assert d["elo_penalty"] == 0.0

    def test_by_category_isolated(self):
        """by_category dict should not be shared across instances."""
        p1 = ProbeSummary()
        p2 = ProbeSummary()
        p1.by_category["xss"] = 3
        assert p2.by_category == {}


# ===========================================================================
# ScenarioSummary tests
# ===========================================================================


class TestScenarioSummary:
    """Tests for the ScenarioSummary dataclass."""

    def test_default_values(self):
        s = ScenarioSummary()
        assert s.scenarios_run == 0
        assert s.outcome_category == "inconclusive"
        assert s.avg_similarity == 0.0
        assert s.universal_conclusions == []
        assert s.conditional_patterns == {}

    def test_to_dict_with_data(self):
        s = ScenarioSummary(
            scenarios_run=5,
            outcome_category="consistent",
            avg_similarity=0.85,
            universal_conclusions=["System is robust", "No critical flaws"],
            conditional_patterns={"high_load": ["Latency spikes"]},
        )
        d = s.to_dict()
        assert d == {
            "scenarios_run": 5,
            "outcome_category": "consistent",
            "avg_similarity": 0.85,
            "universal_conclusions": ["System is robust", "No critical flaws"],
            "conditional_patterns": {"high_load": ["Latency spikes"]},
        }

    def test_to_dict_defaults(self):
        d = ScenarioSummary().to_dict()
        assert d["scenarios_run"] == 0
        assert d["outcome_category"] == "inconclusive"
        assert d["avg_similarity"] == 0.0
        assert d["universal_conclusions"] == []
        assert d["conditional_patterns"] == {}

    def test_lists_isolated(self):
        """Mutable default lists should not be shared across instances."""
        s1 = ScenarioSummary()
        s2 = ScenarioSummary()
        s1.universal_conclusions.append("test")
        assert s2.universal_conclusions == []

    def test_dicts_isolated(self):
        """Mutable default dicts should not be shared across instances."""
        s1 = ScenarioSummary()
        s2 = ScenarioSummary()
        s1.conditional_patterns["key"] = ["val"]
        assert s2.conditional_patterns == {}


# ===========================================================================
# GauntletResult tests
# ===========================================================================


class TestGauntletResultCreation:
    """Tests for GauntletResult creation and defaults."""

    def test_creation_with_required_fields(self):
        result = GauntletResult(
            gauntlet_id="g-001",
            input_hash="hash123",
            input_summary="Summary text",
            started_at="2025-01-01T00:00:00+00:00",
        )
        assert result.gauntlet_id == "g-001"
        assert result.input_hash == "hash123"
        assert result.input_summary == "Summary text"
        assert result.started_at == "2025-01-01T00:00:00+00:00"

    def test_default_values(self):
        result = _make_gauntlet_result()
        assert result.completed_at == ""
        assert result.duration_seconds == 0.0
        assert result.verdict == Verdict.CONDITIONAL
        assert result.confidence == 0.5
        assert result.verdict_reasoning == ""
        assert result.vulnerabilities == []
        assert isinstance(result.risk_summary, RiskSummary)
        assert result.risk_summary.total == 0
        assert isinstance(result.attack_summary, AttackSummary)
        assert isinstance(result.probe_summary, ProbeSummary)
        assert isinstance(result.scenario_summary, ScenarioSummary)
        assert result.dissenting_views == []
        assert result.consensus_points == []
        assert result.config_used == {}
        assert result.agents_used == []

    def test_mutable_defaults_isolated(self):
        """Mutable defaults should not be shared across instances."""
        r1 = _make_gauntlet_result(gauntlet_id="r1")
        r2 = _make_gauntlet_result(gauntlet_id="r2")
        r1.vulnerabilities.append(_make_vulnerability())
        r1.dissenting_views.append("view")
        r1.consensus_points.append("point")
        r1.agents_used.append("agent")
        assert r2.vulnerabilities == []
        assert r2.dissenting_views == []
        assert r2.consensus_points == []
        assert r2.agents_used == []


class TestGauntletResultAddVulnerability:
    """Tests for GauntletResult.add_vulnerability()."""

    def test_add_critical_vulnerability(self):
        result = _make_gauntlet_result()
        vuln = _make_vulnerability(severity=SeverityLevel.CRITICAL)
        result.add_vulnerability(vuln)
        assert len(result.vulnerabilities) == 1
        assert result.vulnerabilities[0] is vuln
        assert result.risk_summary.critical == 1
        assert result.risk_summary.total == 1

    def test_add_high_vulnerability(self):
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.HIGH))
        assert result.risk_summary.high == 1
        assert result.risk_summary.total == 1

    def test_add_medium_vulnerability(self):
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.MEDIUM))
        assert result.risk_summary.medium == 1

    def test_add_low_vulnerability(self):
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.LOW))
        assert result.risk_summary.low == 1

    def test_add_info_vulnerability(self):
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.INFO))
        assert result.risk_summary.info == 1

    def test_add_multiple_vulnerabilities(self):
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(id="V-1", severity=SeverityLevel.CRITICAL))
        result.add_vulnerability(_make_vulnerability(id="V-2", severity=SeverityLevel.CRITICAL))
        result.add_vulnerability(_make_vulnerability(id="V-3", severity=SeverityLevel.HIGH))
        result.add_vulnerability(_make_vulnerability(id="V-4", severity=SeverityLevel.MEDIUM))
        result.add_vulnerability(_make_vulnerability(id="V-5", severity=SeverityLevel.LOW))
        result.add_vulnerability(_make_vulnerability(id="V-6", severity=SeverityLevel.INFO))

        assert len(result.vulnerabilities) == 6
        assert result.risk_summary.critical == 2
        assert result.risk_summary.high == 1
        assert result.risk_summary.medium == 1
        assert result.risk_summary.low == 1
        assert result.risk_summary.info == 1
        assert result.risk_summary.total == 6

    def test_add_vulnerability_updates_weighted_score(self):
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.CRITICAL))
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.HIGH))
        # critical=10, high=5 -> weighted_score=15
        assert result.risk_summary.weighted_score == 15


class TestGauntletResultCalculateVerdict:
    """Tests for GauntletResult.calculate_verdict()."""

    def test_fail_on_critical_above_threshold(self):
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.CRITICAL))
        result.calculate_verdict(critical_threshold=0)
        assert result.verdict == Verdict.FAIL
        assert result.confidence == 0.9
        assert "Critical vulnerabilities" in result.verdict_reasoning
        assert "(1)" in result.verdict_reasoning
        assert "(0)" in result.verdict_reasoning

    def test_fail_on_critical_custom_threshold(self):
        """With a higher threshold, a single critical should not trigger FAIL."""
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.CRITICAL))
        result.calculate_verdict(critical_threshold=1)
        # 1 critical does NOT exceed threshold of 1, so should not FAIL on critical
        assert result.verdict != Verdict.FAIL or "Critical" not in result.verdict_reasoning

    def test_fail_on_high_above_threshold(self):
        result = _make_gauntlet_result()
        for i in range(3):
            result.add_vulnerability(_make_vulnerability(id=f"V-{i}", severity=SeverityLevel.HIGH))
        result.calculate_verdict(high_threshold=2)
        assert result.verdict == Verdict.FAIL
        assert result.confidence == 0.8
        assert "High-severity" in result.verdict_reasoning
        assert "(3)" in result.verdict_reasoning

    def test_no_fail_on_high_at_threshold(self):
        """Exactly at threshold should not trigger FAIL (uses > not >=)."""
        result = _make_gauntlet_result()
        for i in range(2):
            result.add_vulnerability(_make_vulnerability(id=f"V-{i}", severity=SeverityLevel.HIGH))
        result.calculate_verdict(high_threshold=2)
        # 2 high does NOT exceed threshold of 2
        assert result.verdict != Verdict.FAIL or "High-severity" not in result.verdict_reasoning

    def test_conditional_on_high_vulnerability_rate(self):
        result = _make_gauntlet_result()
        result.probe_summary.vulnerability_rate = 0.3
        result.calculate_verdict(vulnerability_rate_threshold=0.2)
        assert result.verdict == Verdict.CONDITIONAL
        assert result.confidence == 0.7
        assert "Vulnerability rate" in result.verdict_reasoning

    def test_no_conditional_on_vulnerability_rate_at_threshold(self):
        """Exactly at threshold should not trigger (uses > not >=)."""
        result = _make_gauntlet_result()
        result.probe_summary.vulnerability_rate = 0.2
        result.calculate_verdict(vulnerability_rate_threshold=0.2)
        # 0.2 is NOT > 0.2, so this check should not trigger
        assert "Vulnerability rate" not in result.verdict_reasoning

    def test_conditional_on_low_robustness(self):
        result = _make_gauntlet_result()
        result.attack_summary.robustness_score = 0.4
        result.calculate_verdict(robustness_threshold=0.6)
        assert result.verdict == Verdict.CONDITIONAL
        assert result.confidence == 0.7
        assert "Robustness score" in result.verdict_reasoning

    def test_no_conditional_on_robustness_at_threshold(self):
        """Exactly at threshold should not trigger (uses < not <=)."""
        result = _make_gauntlet_result()
        result.attack_summary.robustness_score = 0.6
        result.calculate_verdict(robustness_threshold=0.6)
        assert "Robustness" not in result.verdict_reasoning

    def test_conditional_on_divergent_scenarios(self):
        result = _make_gauntlet_result()
        result.scenario_summary.outcome_category = "divergent"
        result.calculate_verdict()
        assert result.verdict == Verdict.CONDITIONAL
        assert result.confidence == 0.6
        assert "diverge" in result.verdict_reasoning.lower()

    def test_pass_when_all_thresholds_met(self):
        result = _make_gauntlet_result()
        result.attack_summary.robustness_score = 0.9
        result.scenario_summary.outcome_category = "consistent"
        result.calculate_verdict()
        assert result.verdict == Verdict.PASS
        assert "No vulnerabilities found" in result.verdict_reasoning
        assert "Strong robustness" in result.verdict_reasoning
        assert "Consistent conclusions" in result.verdict_reasoning

    def test_pass_with_vulnerabilities_within_thresholds(self):
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.MEDIUM))
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.LOW, id="V-2"))
        result.attack_summary.robustness_score = 0.8
        result.scenario_summary.outcome_category = "consistent"
        result.calculate_verdict()
        assert result.verdict == Verdict.PASS
        assert "within thresholds" in result.verdict_reasoning
        assert "(2 total)" in result.verdict_reasoning

    def test_pass_confidence_calculation(self):
        result = _make_gauntlet_result()
        result.attack_summary.robustness_score = 0.9
        result.calculate_verdict()
        assert result.verdict == Verdict.PASS
        # confidence = min(0.95, 0.6 + 0.9 * 0.3) = min(0.95, 0.87) = 0.87
        assert result.confidence == pytest.approx(0.87)

    def test_pass_confidence_capped_at_095(self):
        result = _make_gauntlet_result()
        result.attack_summary.robustness_score = 1.5  # unrealistic but tests cap
        result.calculate_verdict()
        assert result.verdict == Verdict.PASS
        # confidence = min(0.95, 0.6 + 1.5 * 0.3) = min(0.95, 1.05) = 0.95
        assert result.confidence == pytest.approx(0.95)

    def test_verdict_priority_critical_over_high(self):
        """Critical check runs before high check."""
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(id="V-1", severity=SeverityLevel.CRITICAL))
        for i in range(3):
            result.add_vulnerability(
                _make_vulnerability(id=f"V-H-{i}", severity=SeverityLevel.HIGH)
            )
        result.calculate_verdict(critical_threshold=0, high_threshold=2)
        assert result.verdict == Verdict.FAIL
        assert result.confidence == 0.9
        assert "Critical" in result.verdict_reasoning

    def test_verdict_priority_high_over_vuln_rate(self):
        """High check runs before vulnerability_rate check."""
        result = _make_gauntlet_result()
        for i in range(3):
            result.add_vulnerability(_make_vulnerability(id=f"V-{i}", severity=SeverityLevel.HIGH))
        result.probe_summary.vulnerability_rate = 0.5
        result.calculate_verdict(high_threshold=2, vulnerability_rate_threshold=0.2)
        assert result.verdict == Verdict.FAIL
        assert result.confidence == 0.8
        assert "High-severity" in result.verdict_reasoning

    def test_verdict_priority_vuln_rate_over_robustness(self):
        """Vulnerability rate check runs before robustness check."""
        result = _make_gauntlet_result()
        result.probe_summary.vulnerability_rate = 0.5
        result.attack_summary.robustness_score = 0.3
        result.calculate_verdict()
        assert result.verdict == Verdict.CONDITIONAL
        assert "Vulnerability rate" in result.verdict_reasoning

    def test_verdict_priority_robustness_over_divergent(self):
        """Robustness check runs before divergent scenario check."""
        result = _make_gauntlet_result()
        result.attack_summary.robustness_score = 0.3
        result.scenario_summary.outcome_category = "divergent"
        result.calculate_verdict()
        assert result.verdict == Verdict.CONDITIONAL
        assert "Robustness" in result.verdict_reasoning

    def test_pass_without_consistent_scenarios(self):
        """PASS verdict should work even with inconclusive scenarios."""
        result = _make_gauntlet_result()
        result.attack_summary.robustness_score = 0.9
        result.scenario_summary.outcome_category = "inconclusive"
        result.calculate_verdict()
        assert result.verdict == Verdict.PASS
        assert "Consistent conclusions" not in result.verdict_reasoning

    def test_calculate_verdict_with_custom_thresholds(self):
        """All thresholds can be customized."""
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.CRITICAL))
        # With a very high critical threshold, this should PASS
        result.attack_summary.robustness_score = 0.8
        result.calculate_verdict(
            critical_threshold=5,
            high_threshold=10,
            vulnerability_rate_threshold=0.9,
            robustness_threshold=0.1,
        )
        assert result.verdict == Verdict.PASS


class TestGauntletResultGetCriticalVulnerabilities:
    """Tests for GauntletResult.get_critical_vulnerabilities()."""

    def test_returns_critical_and_high(self):
        result = _make_gauntlet_result()
        v_crit = _make_vulnerability(id="V-C", severity=SeverityLevel.CRITICAL)
        v_high = _make_vulnerability(id="V-H", severity=SeverityLevel.HIGH)
        v_med = _make_vulnerability(id="V-M", severity=SeverityLevel.MEDIUM)
        v_low = _make_vulnerability(id="V-L", severity=SeverityLevel.LOW)
        v_info = _make_vulnerability(id="V-I", severity=SeverityLevel.INFO)
        result.add_vulnerability(v_crit)
        result.add_vulnerability(v_high)
        result.add_vulnerability(v_med)
        result.add_vulnerability(v_low)
        result.add_vulnerability(v_info)
        critical = result.get_critical_vulnerabilities()
        assert len(critical) == 2
        assert v_crit in critical
        assert v_high in critical
        assert v_med not in critical
        assert v_low not in critical
        assert v_info not in critical

    def test_empty_when_no_vulnerabilities(self):
        result = _make_gauntlet_result()
        assert result.get_critical_vulnerabilities() == []

    def test_empty_when_only_low_severity(self):
        result = _make_gauntlet_result()
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.MEDIUM))
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.LOW, id="V-2"))
        result.add_vulnerability(_make_vulnerability(severity=SeverityLevel.INFO, id="V-3"))
        assert result.get_critical_vulnerabilities() == []

    def test_returns_multiple_critical(self):
        result = _make_gauntlet_result()
        for i in range(5):
            result.add_vulnerability(
                _make_vulnerability(id=f"V-{i}", severity=SeverityLevel.CRITICAL)
            )
        assert len(result.get_critical_vulnerabilities()) == 5


class TestGauntletResultToDict:
    """Tests for GauntletResult.to_dict()."""

    def test_to_dict_structure(self):
        result = _make_gauntlet_result(
            gauntlet_id="g-test",
            input_hash="hashval",
            input_summary="My input",
            started_at="2025-01-01T00:00:00+00:00",
            completed_at="2025-01-01T00:01:00+00:00",
            duration_seconds=60.0,
        )
        result.dissenting_views = ["View A"]
        result.consensus_points = ["Point B"]
        result.config_used = {"rounds": 3}
        result.agents_used = ["claude", "gpt"]

        d = result.to_dict()

        assert d["gauntlet_id"] == "g-test"
        assert d["input_hash"] == "hashval"
        assert d["input_summary"] == "My input"
        assert d["started_at"] == "2025-01-01T00:00:00+00:00"
        assert d["completed_at"] == "2025-01-01T00:01:00+00:00"
        assert d["duration_seconds"] == 60.0
        assert d["verdict"] == "conditional"
        assert d["confidence"] == 0.5
        assert d["verdict_reasoning"] == ""
        assert d["vulnerabilities"] == []
        assert d["dissenting_views"] == ["View A"]
        assert d["consensus_points"] == ["Point B"]
        assert d["config_used"] == {"rounds": 3}
        assert d["agents_used"] == ["claude", "gpt"]

    def test_to_dict_includes_risk_summary(self):
        result = _make_gauntlet_result()
        d = result.to_dict()
        assert "risk_summary" in d
        assert d["risk_summary"]["total"] == 0

    def test_to_dict_includes_attack_summary(self):
        result = _make_gauntlet_result()
        d = result.to_dict()
        assert "attack_summary" in d
        assert d["attack_summary"]["success_rate"] == 0

    def test_to_dict_includes_probe_summary(self):
        result = _make_gauntlet_result()
        d = result.to_dict()
        assert "probe_summary" in d
        assert d["probe_summary"]["probes_run"] == 0

    def test_to_dict_includes_scenario_summary(self):
        result = _make_gauntlet_result()
        d = result.to_dict()
        assert "scenario_summary" in d
        assert d["scenario_summary"]["outcome_category"] == "inconclusive"

    def test_to_dict_serializes_vulnerabilities(self):
        result = _make_gauntlet_result()
        vuln = _make_vulnerability(
            id="V-1",
            severity=SeverityLevel.HIGH,
            exploitability=0.8,
            impact=0.9,
        )
        result.add_vulnerability(vuln)
        d = result.to_dict()
        assert len(d["vulnerabilities"]) == 1
        assert d["vulnerabilities"][0]["id"] == "V-1"
        assert d["vulnerabilities"][0]["severity"] == "high"
        assert d["vulnerabilities"][0]["risk_score"] == pytest.approx(0.72)

    def test_to_dict_verdict_after_calculate(self):
        result = _make_gauntlet_result()
        result.attack_summary.robustness_score = 0.9
        result.calculate_verdict()
        d = result.to_dict()
        assert d["verdict"] == "pass"
        assert d["confidence"] > 0.5

    def test_to_dict_all_expected_keys(self):
        """Verify all expected top-level keys are present."""
        result = _make_gauntlet_result()
        d = result.to_dict()
        expected_keys = {
            "gauntlet_id",
            "input_hash",
            "input_summary",
            "started_at",
            "completed_at",
            "duration_seconds",
            "verdict",
            "confidence",
            "verdict_reasoning",
            "vulnerabilities",
            "risk_summary",
            "attack_summary",
            "probe_summary",
            "scenario_summary",
            "dissenting_views",
            "consensus_points",
            "config_used",
            "agents_used",
        }
        assert set(d.keys()) == expected_keys


# ===========================================================================
# Enum re-export tests
# ===========================================================================


class TestEnumReexports:
    """Verify that SeverityLevel and Verdict are properly re-exported from result module."""

    def test_severity_level_values(self):
        assert SeverityLevel.CRITICAL.value == "critical"
        assert SeverityLevel.HIGH.value == "high"
        assert SeverityLevel.MEDIUM.value == "medium"
        assert SeverityLevel.LOW.value == "low"
        assert SeverityLevel.INFO.value == "info"

    def test_verdict_values(self):
        assert Verdict.PASS.value == "pass"
        assert Verdict.APPROVED.value == "approved"
        assert Verdict.CONDITIONAL.value == "conditional"
        assert Verdict.APPROVED_WITH_CONDITIONS.value == "approved_with_conditions"
        assert Verdict.NEEDS_REVIEW.value == "needs_review"
        assert Verdict.FAIL.value == "fail"
        assert Verdict.REJECTED.value == "rejected"

    def test_risk_summary_is_from_types(self):
        """RiskSummary in result module is aliased from types.RiskSummary."""
        from aragora.gauntlet.types import RiskSummary as TypesRiskSummary

        assert RiskSummary is TypesRiskSummary
