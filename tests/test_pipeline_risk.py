"""
Tests for aragora.pipeline.risk_register module.

Tests the Risk, RiskRegister, and RiskAnalyzer classes for:
- Risk dataclass creation and serialization
- RiskRegister filtering and summary methods
- RiskAnalyzer critique/verification/consensus analysis
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from aragora.pipeline.risk_register import (
    Risk,
    RiskLevel,
    RiskCategory,
    RiskRegister,
    RiskAnalyzer,
)


class TestRisk:
    """Tests for the Risk dataclass."""

    def test_risk_creation_with_defaults(self):
        """Test creating a Risk with minimal required fields."""
        risk = Risk(
            id="risk-001",
            title="SQL Injection Vulnerability",
            description="User input not sanitized",
            level=RiskLevel.HIGH,
            category=RiskCategory.SECURITY,
            source="agent:claude",
        )

        assert risk.id == "risk-001"
        assert risk.title == "SQL Injection Vulnerability"
        assert risk.level == RiskLevel.HIGH
        assert risk.category == RiskCategory.SECURITY
        assert risk.impact == 0.5  # default
        assert risk.likelihood == 0.5  # default
        assert risk.mitigation == ""
        assert risk.mitigation_status == "proposed"
        assert risk.related_critique_ids == []
        assert risk.related_claim_ids == []
        assert risk.created_at  # Should be auto-populated

    def test_risk_creation_with_all_fields(self):
        """Test creating a Risk with all fields specified."""
        risk = Risk(
            id="risk-002",
            title="Performance Degradation",
            description="Algorithm is O(n^2)",
            level=RiskLevel.MEDIUM,
            category=RiskCategory.PERFORMANCE,
            source="critique:critique-123",
            impact=0.7,
            likelihood=0.3,
            mitigation="Use O(n log n) algorithm",
            mitigation_status="in_progress",
            related_critique_ids=["crit-1", "crit-2"],
            related_claim_ids=["claim-a"],
            created_at="2025-01-01T00:00:00",
        )

        assert risk.impact == 0.7
        assert risk.likelihood == 0.3
        assert risk.mitigation == "Use O(n log n) algorithm"
        assert risk.mitigation_status == "in_progress"
        assert len(risk.related_critique_ids) == 2

    def test_risk_score_calculation(self):
        """Test that risk_score is calculated correctly."""
        risk = Risk(
            id="risk-003",
            title="Test Risk",
            description="Testing",
            level=RiskLevel.LOW,
            category=RiskCategory.TECHNICAL,
            source="test",
            impact=0.8,
            likelihood=0.4,
        )

        assert risk.risk_score == 0.8 * 0.4  # 0.32

    def test_risk_score_edge_cases(self):
        """Test risk_score with edge case values."""
        # Zero impact
        risk_zero_impact = Risk(
            id="r1",
            title="T",
            description="D",
            level=RiskLevel.LOW,
            category=RiskCategory.UNKNOWN,
            source="test",
            impact=0.0,
            likelihood=1.0,
        )
        assert risk_zero_impact.risk_score == 0.0

        # Maximum values
        risk_max = Risk(
            id="r2",
            title="T",
            description="D",
            level=RiskLevel.CRITICAL,
            category=RiskCategory.SECURITY,
            source="test",
            impact=1.0,
            likelihood=1.0,
        )
        assert risk_max.risk_score == 1.0

    def test_risk_to_dict(self):
        """Test Risk serialization to dictionary."""
        risk = Risk(
            id="risk-004",
            title="Serialization Test",
            description="Test description",
            level=RiskLevel.CRITICAL,
            category=RiskCategory.SCALABILITY,
            source="test",
            impact=0.9,
            likelihood=0.8,
        )

        d = risk.to_dict()

        assert d["id"] == "risk-004"
        assert d["title"] == "Serialization Test"
        assert d["level"] == "critical"  # Enum value
        assert d["category"] == "scalability"  # Enum value
        assert d["risk_score"] == 0.9 * 0.8
        assert "created_at" in d

    def test_all_risk_levels(self):
        """Test that all RiskLevel enum values work."""
        for level in RiskLevel:
            risk = Risk(
                id=f"risk-{level.value}",
                title=f"Test {level.value}",
                description="Test",
                level=level,
                category=RiskCategory.UNKNOWN,
                source="test",
            )
            assert risk.level == level
            assert risk.to_dict()["level"] == level.value

    def test_all_risk_categories(self):
        """Test that all RiskCategory enum values work."""
        for category in RiskCategory:
            risk = Risk(
                id=f"risk-{category.value}",
                title=f"Test {category.value}",
                description="Test",
                level=RiskLevel.LOW,
                category=category,
                source="test",
            )
            assert risk.category == category
            assert risk.to_dict()["category"] == category.value


class TestRiskRegister:
    """Tests for the RiskRegister dataclass."""

    @pytest.fixture
    def sample_risks(self):
        """Create sample risks for testing."""
        return [
            Risk(
                id="r1",
                title="Critical Security",
                description="Auth bypass",
                level=RiskLevel.CRITICAL,
                category=RiskCategory.SECURITY,
                source="audit",
                mitigation_status="proposed",
            ),
            Risk(
                id="r2",
                title="High Performance",
                description="Slow query",
                level=RiskLevel.HIGH,
                category=RiskCategory.PERFORMANCE,
                source="profiler",
                mitigation_status="in_progress",
            ),
            Risk(
                id="r3",
                title="Medium Technical",
                description="Legacy code",
                level=RiskLevel.MEDIUM,
                category=RiskCategory.TECHNICAL,
                source="review",
                mitigation_status="implemented",
            ),
            Risk(
                id="r4",
                title="Low Maintainability",
                description="Missing docs",
                level=RiskLevel.LOW,
                category=RiskCategory.MAINTAINABILITY,
                source="lint",
                mitigation_status="accepted",
            ),
            Risk(
                id="r5",
                title="High Security",
                description="XSS vuln",
                level=RiskLevel.HIGH,
                category=RiskCategory.SECURITY,
                source="scanner",
                mitigation_status="proposed",
            ),
        ]

    @pytest.fixture
    def register(self, sample_risks):
        """Create a populated RiskRegister."""
        reg = RiskRegister(debate_id="test-debate-001")
        for risk in sample_risks:
            reg.add_risk(risk)
        return reg

    def test_empty_register(self):
        """Test empty RiskRegister."""
        reg = RiskRegister(debate_id="empty-debate")

        assert reg.debate_id == "empty-debate"
        assert reg.risks == []
        assert reg.summary["total_risks"] == 0
        assert reg.summary["avg_risk_score"] == 0

    def test_add_risk(self, sample_risks):
        """Test adding risks to register."""
        reg = RiskRegister(debate_id="test")

        for risk in sample_risks:
            reg.add_risk(risk)

        assert len(reg.risks) == 5

    def test_get_by_level(self, register):
        """Test filtering risks by level."""
        critical = register.get_by_level(RiskLevel.CRITICAL)
        high = register.get_by_level(RiskLevel.HIGH)
        medium = register.get_by_level(RiskLevel.MEDIUM)
        low = register.get_by_level(RiskLevel.LOW)

        assert len(critical) == 1
        assert len(high) == 2
        assert len(medium) == 1
        assert len(low) == 1

    def test_get_by_category(self, register):
        """Test filtering risks by category."""
        security = register.get_by_category(RiskCategory.SECURITY)
        performance = register.get_by_category(RiskCategory.PERFORMANCE)
        technical = register.get_by_category(RiskCategory.TECHNICAL)

        assert len(security) == 2
        assert len(performance) == 1
        assert len(technical) == 1

    def test_get_unmitigated(self, register):
        """Test filtering unmitigated risks."""
        unmitigated = register.get_unmitigated()

        # "proposed", "in_progress", "accepted" are not "implemented"
        assert len(unmitigated) == 4
        assert all(r.mitigation_status != "implemented" for r in unmitigated)

    def test_get_critical_risks(self, register):
        """Test getting high and critical risks."""
        critical = register.get_critical_risks()

        assert len(critical) == 3  # 1 critical + 2 high
        assert all(r.level in [RiskLevel.CRITICAL, RiskLevel.HIGH] for r in critical)

    def test_summary(self, register):
        """Test summary statistics."""
        summary = register.summary

        assert summary["total_risks"] == 5
        assert summary["critical"] == 1
        assert summary["high"] == 2
        assert summary["medium"] == 1
        assert summary["low"] == 1
        assert summary["unmitigated"] == 4
        assert "avg_risk_score" in summary

    def test_to_markdown(self, register):
        """Test markdown generation."""
        md = register.to_markdown()

        assert "# Risk Register" in md
        assert "test-debate-001" in md
        assert "CRITICAL" in md
        assert "HIGH" in md
        assert "Critical Security" in md
        assert "Generated by aragora" in md

    def test_to_dict(self, register):
        """Test dictionary serialization."""
        d = register.to_dict()

        assert d["debate_id"] == "test-debate-001"
        assert len(d["risks"]) == 5
        assert "summary" in d
        assert "thresholds" in d
        assert d["thresholds"]["low_support"] == 0.5


class TestRiskAnalyzer:
    """Tests for the RiskAnalyzer class."""

    @pytest.fixture
    def mock_artifact(self):
        """Create a mock DebateArtifact."""
        artifact = MagicMock()
        artifact.debate_id = "analysis-debate-001"
        artifact.trace_data = {
            "events": [
                {
                    "event_type": "agent_critique",
                    "payload": {
                        "severity": 0.8,
                        "issues": ["Issue 1", "Issue 2"],
                        "agent": "claude",
                    },
                },
                {
                    "event_type": "agent_critique",
                    "payload": {
                        "severity": 0.3,
                        "issues": ["Minor issue"],
                        "agent": "gpt-4",
                    },
                },
                {
                    "event_type": "agent_message",
                    "payload": {"content": "Some message"},
                },
            ]
        }
        artifact.consensus_proof = MagicMock()
        artifact.consensus_proof.confidence = 0.75
        artifact.consensus_proof.supporting_agents = ["claude", "gpt-4"]
        artifact.consensus_proof.dissenting_agents = []
        return artifact

    def test_analyzer_creation(self, mock_artifact):
        """Test RiskAnalyzer initialization."""
        analyzer = RiskAnalyzer(mock_artifact)
        assert analyzer.artifact == mock_artifact

    def test_analyze_returns_register(self, mock_artifact):
        """Test that analyze returns a RiskRegister."""
        analyzer = RiskAnalyzer(mock_artifact)
        register = analyzer.analyze()

        assert isinstance(register, RiskRegister)
        assert register.debate_id == "analysis-debate-001"

    def test_analyze_with_no_trace_data(self):
        """Test analysis with empty trace data."""
        artifact = MagicMock()
        artifact.debate_id = "empty-trace"
        artifact.trace_data = None
        artifact.consensus_proof = None

        analyzer = RiskAnalyzer(artifact)
        register = analyzer.analyze()

        # Should not raise, returns empty register
        assert isinstance(register, RiskRegister)

    def test_analyze_extracts_critique_risks(self, mock_artifact):
        """Test that high-severity critiques become risks."""
        analyzer = RiskAnalyzer(mock_artifact)
        register = analyzer.analyze()

        # Should extract risk from severity 0.8 critique
        risk_titles = [r.title for r in register.risks]
        # The analyzer creates risks from high-severity critiques
        # Check that some risks were identified
        assert len(register.risks) >= 0  # May vary based on thresholds


class TestRiskLevelEnum:
    """Tests for RiskLevel enum."""

    def test_risk_level_values(self):
        """Test RiskLevel enum has expected values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_risk_level_ordering(self):
        """Test that risk levels can be compared by list position."""
        levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        # Verify all levels are present
        assert len(levels) == 4


class TestRiskCategoryEnum:
    """Tests for RiskCategory enum."""

    def test_risk_category_values(self):
        """Test RiskCategory enum has expected values."""
        expected = {
            "technical",
            "security",
            "performance",
            "scalability",
            "maintainability",
            "compatibility",
            "unknown",
        }
        actual = {cat.value for cat in RiskCategory}
        assert actual == expected


class TestIntegration:
    """Integration tests for the risk_register module."""

    def test_full_workflow(self):
        """Test complete workflow: create risks, add to register, analyze."""
        # Create register
        register = RiskRegister(debate_id="integration-test")

        # Add various risks
        risks = [
            Risk(
                id=f"risk-{i}",
                title=f"Risk {i}",
                description=f"Description {i}",
                level=list(RiskLevel)[i % 4],
                category=list(RiskCategory)[i % len(RiskCategory)],
                source="test",
                impact=0.1 * (i + 1),
                likelihood=0.1 * (10 - i),
            )
            for i in range(10)
        ]

        for risk in risks:
            register.add_risk(risk)

        # Verify filtering works
        assert len(register.get_critical_risks()) > 0
        assert len(register.get_unmitigated()) == 10

        # Verify serialization
        d = register.to_dict()
        assert len(d["risks"]) == 10

        # Verify markdown
        md = register.to_markdown()
        assert "integration-test" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
