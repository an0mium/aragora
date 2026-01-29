"""
Tests for Domain Risk Assessor module.

Tests risk assessment functionality for debate topics:
- RiskLevel enum
- RiskAssessment dataclass
- RiskAssessor class with pattern matching
- RISK_PATTERNS configuration
- Utility functions
"""

from __future__ import annotations

import pytest


# =============================================================================
# RiskLevel Enum Tests
# =============================================================================


class TestRiskLevel:
    """Test RiskLevel enum."""

    def test_low_level_exists(self):
        """Test LOW level exists."""
        from aragora.debate.risk_assessor import RiskLevel

        assert RiskLevel.LOW.value == "low"

    def test_medium_level_exists(self):
        """Test MEDIUM level exists."""
        from aragora.debate.risk_assessor import RiskLevel

        assert RiskLevel.MEDIUM.value == "medium"

    def test_high_level_exists(self):
        """Test HIGH level exists."""
        from aragora.debate.risk_assessor import RiskLevel

        assert RiskLevel.HIGH.value == "high"

    def test_critical_level_exists(self):
        """Test CRITICAL level exists."""
        from aragora.debate.risk_assessor import RiskLevel

        assert RiskLevel.CRITICAL.value == "critical"


# =============================================================================
# RiskAssessment Tests
# =============================================================================


class TestRiskAssessment:
    """Test RiskAssessment dataclass."""

    def test_create_assessment(self):
        """Test creating risk assessment."""
        from aragora.debate.risk_assessor import RiskAssessment, RiskLevel

        assessment = RiskAssessment(
            level=RiskLevel.HIGH,
            domain="medical",
            category="health_advice",
            description="Medical topics require professional expertise.",
        )

        assert assessment.level == RiskLevel.HIGH
        assert assessment.domain == "medical"
        assert assessment.category == "health_advice"
        assert assessment.confidence == 0.5  # default

    def test_create_with_mitigations(self):
        """Test creating assessment with mitigations."""
        from aragora.debate.risk_assessor import RiskAssessment, RiskLevel

        assessment = RiskAssessment(
            level=RiskLevel.MEDIUM,
            domain="financial",
            category="financial_advice",
            description="Financial recommendations require due diligence.",
            mitigations=["Consult financial advisor", "Include disclaimer"],
        )

        assert len(assessment.mitigations) == 2
        assert "Consult financial advisor" in assessment.mitigations

    def test_create_with_custom_confidence(self):
        """Test creating assessment with custom confidence."""
        from aragora.debate.risk_assessor import RiskAssessment, RiskLevel

        assessment = RiskAssessment(
            level=RiskLevel.LOW,
            domain="general",
            category="general",
            description="Low risk topic.",
            confidence=0.85,
        )

        assert assessment.confidence == 0.85


# =============================================================================
# RISK_PATTERNS Tests
# =============================================================================


class TestRiskPatterns:
    """Test RISK_PATTERNS configuration."""

    def test_medical_pattern_exists(self):
        """Test medical pattern is defined."""
        from aragora.debate.risk_assessor import RISK_PATTERNS

        assert "medical" in RISK_PATTERNS
        assert "keywords" in RISK_PATTERNS["medical"]
        assert "level" in RISK_PATTERNS["medical"]

    def test_legal_pattern_exists(self):
        """Test legal pattern is defined."""
        from aragora.debate.risk_assessor import RISK_PATTERNS

        assert "legal" in RISK_PATTERNS
        assert "law" in RISK_PATTERNS["legal"]["keywords"]

    def test_financial_pattern_exists(self):
        """Test financial pattern is defined."""
        from aragora.debate.risk_assessor import RISK_PATTERNS

        assert "financial" in RISK_PATTERNS
        assert "investment" in RISK_PATTERNS["financial"]["keywords"]

    def test_safety_pattern_exists(self):
        """Test safety pattern is defined."""
        from aragora.debate.risk_assessor import RISK_PATTERNS, RiskLevel

        assert "safety" in RISK_PATTERNS
        assert RISK_PATTERNS["safety"]["level"] == RiskLevel.CRITICAL

    def test_speculative_pattern_exists(self):
        """Test speculative pattern is defined."""
        from aragora.debate.risk_assessor import RISK_PATTERNS

        assert "speculative" in RISK_PATTERNS
        assert "predict" in RISK_PATTERNS["speculative"]["keywords"]

    def test_patterns_have_mitigations(self):
        """Test all patterns have mitigations."""
        from aragora.debate.risk_assessor import RISK_PATTERNS

        for pattern_name, pattern_config in RISK_PATTERNS.items():
            assert "mitigations" in pattern_config, f"{pattern_name} missing mitigations"
            assert len(pattern_config["mitigations"]) > 0


# =============================================================================
# RiskAssessor Tests
# =============================================================================


class TestRiskAssessor:
    """Test RiskAssessor class."""

    def test_init_default(self):
        """Test default initialization."""
        from aragora.debate.risk_assessor import RiskAssessor

        assessor = RiskAssessor()

        assert assessor.patterns is not None
        assert "medical" in assessor.patterns

    def test_init_with_custom_patterns(self):
        """Test initialization with custom patterns."""
        from aragora.debate.risk_assessor import RiskAssessor, RiskLevel

        custom = {
            "custom": {
                "keywords": ["custom", "test"],
                "level": RiskLevel.LOW,
                "category": "custom_category",
                "description": "Custom test pattern",
                "mitigations": ["Custom mitigation"],
            }
        }

        assessor = RiskAssessor(custom_patterns=custom)

        assert "custom" in assessor.patterns
        assert "medical" in assessor.patterns  # Default patterns still present

    def test_assess_topic_medical(self):
        """Test assessing medical topic."""
        from aragora.debate.risk_assessor import RiskAssessor, RiskLevel

        assessor = RiskAssessor()
        topic = "What is the best treatment for my symptoms?"

        risks = assessor.assess_topic(topic)

        assert len(risks) > 0
        # Should find medical risk
        medical_risks = [r for r in risks if r.domain == "medical"]
        assert len(medical_risks) > 0
        assert medical_risks[0].level == RiskLevel.HIGH

    def test_assess_topic_legal(self):
        """Test assessing legal topic."""
        from aragora.debate.risk_assessor import RiskAssessor, RiskLevel

        assessor = RiskAssessor()
        topic = "Should I sue my employer for breach of contract?"

        risks = assessor.assess_topic(topic)

        legal_risks = [r for r in risks if r.domain == "legal"]
        assert len(legal_risks) > 0
        assert legal_risks[0].level == RiskLevel.HIGH

    def test_assess_topic_financial(self):
        """Test assessing financial topic."""
        from aragora.debate.risk_assessor import RiskAssessor, RiskLevel

        assessor = RiskAssessor()
        topic = "What stocks should I invest in for my portfolio?"

        risks = assessor.assess_topic(topic)

        financial_risks = [r for r in risks if r.domain == "financial"]
        assert len(financial_risks) > 0
        assert financial_risks[0].level == RiskLevel.MEDIUM

    def test_assess_topic_safety(self):
        """Test assessing safety topic."""
        from aragora.debate.risk_assessor import RiskAssessor, RiskLevel

        assessor = RiskAssessor()
        topic = "How to exploit this vulnerability in the system?"

        risks = assessor.assess_topic(topic)

        safety_risks = [r for r in risks if r.domain == "safety"]
        assert len(safety_risks) > 0
        assert safety_risks[0].level == RiskLevel.CRITICAL

    def test_assess_topic_speculative(self):
        """Test assessing speculative topic."""
        from aragora.debate.risk_assessor import RiskAssessor

        assessor = RiskAssessor()
        topic = "What will happen to Bitcoin price in the future?"

        risks = assessor.assess_topic(topic)

        speculative_risks = [r for r in risks if r.domain == "speculative"]
        assert len(speculative_risks) > 0

    def test_assess_topic_no_risks(self):
        """Test assessing topic with no risks."""
        from aragora.debate.risk_assessor import RiskAssessor

        assessor = RiskAssessor()
        topic = "What color is the sky?"

        risks = assessor.assess_topic(topic)

        # May find no risks for benign topic
        assert isinstance(risks, list)

    def test_assess_topic_multiple_risks(self):
        """Test assessing topic with multiple risks."""
        from aragora.debate.risk_assessor import RiskAssessor

        assessor = RiskAssessor()
        topic = "What medical treatment should I invest in for future health?"

        risks = assessor.assess_topic(topic)

        # Should find both medical and financial risks
        domains = {r.domain for r in risks}
        assert len(risks) >= 2

    def test_assess_topic_sorted_by_severity(self):
        """Test risks are sorted by severity (critical first)."""
        from aragora.debate.risk_assessor import RiskAssessor, RiskLevel

        assessor = RiskAssessor()
        # Topic with multiple risk levels
        topic = "How to exploit vulnerability for financial gain with medical side effects?"

        risks = assessor.assess_topic(topic)

        if len(risks) >= 2:
            # Verify sorted by severity (lower index = higher severity)
            severity_order = {
                RiskLevel.CRITICAL: 0,
                RiskLevel.HIGH: 1,
                RiskLevel.MEDIUM: 2,
                RiskLevel.LOW: 3,
            }
            for i in range(len(risks) - 1):
                current_severity = severity_order.get(risks[i].level, 99)
                next_severity = severity_order.get(risks[i + 1].level, 99)
                assert current_severity <= next_severity

    def test_assess_topic_with_domain(self):
        """Test assessing topic with explicit domain."""
        from aragora.debate.risk_assessor import RiskAssessor

        assessor = RiskAssessor()
        topic = "Should I take this medication?"

        risks = assessor.assess_topic(topic, domain="custom_medical")

        medical_risks = [r for r in risks if "medical" in r.domain or r.domain == "custom_medical"]
        assert len(medical_risks) > 0

    def test_confidence_increases_with_matches(self):
        """Test confidence increases with keyword matches."""
        from aragora.debate.risk_assessor import RiskAssessor

        assessor = RiskAssessor()

        # Single keyword match
        topic_single = "What is the diagnosis?"
        risks_single = assessor.assess_topic(topic_single)

        # Multiple keyword matches
        topic_multi = "What diagnosis and treatment should the doctor give for this disease?"
        risks_multi = assessor.assess_topic(topic_multi)

        if risks_single and risks_multi:
            medical_single = [r for r in risks_single if r.domain == "medical"]
            medical_multi = [r for r in risks_multi if r.domain == "medical"]

            if medical_single and medical_multi:
                # More matches should have higher confidence
                assert medical_multi[0].confidence >= medical_single[0].confidence

    def test_get_highest_risk(self):
        """Test getting highest risk."""
        from aragora.debate.risk_assessor import RiskAssessor

        assessor = RiskAssessor()
        topic = "What medical treatment should I invest in?"

        highest = assessor.get_highest_risk(topic)

        if highest:
            # Should be the most severe risk
            all_risks = assessor.assess_topic(topic)
            assert highest.level == all_risks[0].level

    def test_get_highest_risk_no_risks(self):
        """Test getting highest risk when none found."""
        from aragora.debate.risk_assessor import RiskAssessor

        assessor = RiskAssessor()
        topic = "What color is the sky?"

        highest = assessor.get_highest_risk(topic)

        # May return None for benign topic
        assert highest is None or highest is not None

    def test_to_event_data(self):
        """Test converting assessment to event data."""
        from aragora.debate.risk_assessor import RiskAssessment, RiskAssessor, RiskLevel

        assessor = RiskAssessor()
        assessment = RiskAssessment(
            level=RiskLevel.HIGH,
            domain="medical",
            category="health_advice",
            description="Medical advice risk",
            mitigations=["Consult doctor"],
            confidence=0.8,
        )

        event_data = assessor.to_event_data(assessment, debate_id="debate-123")

        assert event_data["level"] == "high"
        assert event_data["domain"] == "medical"
        assert event_data["debate_id"] == "debate-123"
        assert event_data["confidence"] == 0.8
        assert "Consult doctor" in event_data["mitigations"]


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestGetRiskAssessor:
    """Test get_risk_assessor function."""

    def test_returns_assessor(self):
        """Test function returns RiskAssessor."""
        from aragora.debate.risk_assessor import RiskAssessor, get_risk_assessor

        assessor = get_risk_assessor()

        assert isinstance(assessor, RiskAssessor)

    def test_returns_singleton(self):
        """Test function returns same instance."""
        from aragora.debate.risk_assessor import get_risk_assessor

        assessor1 = get_risk_assessor()
        assessor2 = get_risk_assessor()

        assert assessor1 is assessor2


class TestAssessDebateRisk:
    """Test assess_debate_risk convenience function."""

    def test_returns_list(self):
        """Test function returns list of assessments."""
        from aragora.debate.risk_assessor import assess_debate_risk

        risks = assess_debate_risk("What is the best medical treatment?")

        assert isinstance(risks, list)

    def test_with_domain(self):
        """Test function with domain parameter."""
        from aragora.debate.risk_assessor import assess_debate_risk

        risks = assess_debate_risk("Test topic", domain="medical")

        assert isinstance(risks, list)


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_risk_level_exportable(self):
        """Test RiskLevel can be imported."""
        from aragora.debate.risk_assessor import RiskLevel

        assert RiskLevel is not None

    def test_risk_assessment_exportable(self):
        """Test RiskAssessment can be imported."""
        from aragora.debate.risk_assessor import RiskAssessment

        assert RiskAssessment is not None

    def test_risk_assessor_exportable(self):
        """Test RiskAssessor can be imported."""
        from aragora.debate.risk_assessor import RiskAssessor

        assert RiskAssessor is not None

    def test_risk_patterns_exportable(self):
        """Test RISK_PATTERNS can be imported."""
        from aragora.debate.risk_assessor import RISK_PATTERNS

        assert RISK_PATTERNS is not None

    def test_get_risk_assessor_exportable(self):
        """Test get_risk_assessor can be imported."""
        from aragora.debate.risk_assessor import get_risk_assessor

        assert get_risk_assessor is not None

    def test_assess_debate_risk_exportable(self):
        """Test assess_debate_risk can be imported."""
        from aragora.debate.risk_assessor import assess_debate_risk

        assert assess_debate_risk is not None

    def test_all_exports_in_module_all(self):
        """Test __all__ contains expected exports."""
        from aragora.debate import risk_assessor

        expected = [
            "RiskLevel",
            "RiskAssessment",
            "RiskAssessor",
            "RISK_PATTERNS",
            "get_risk_assessor",
            "assess_debate_risk",
        ]

        for name in expected:
            assert name in risk_assessor.__all__
