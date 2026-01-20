"""Tests for Deep Audit mode - intensive debate protocol."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.modes.deep_audit import (
    AuditFinding,
    DeepAuditConfig,
    DeepAuditOrchestrator,
    DeepAuditVerdict,
    STRATEGY_AUDIT,
    CONTRACT_AUDIT,
    CODE_ARCHITECTURE_AUDIT,
)
from aragora.debate.roles import CognitiveRole


class TestDeepAuditConfig:
    """Tests for DeepAuditConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = DeepAuditConfig()
        assert config.rounds == 6
        assert config.enable_research is True
        assert config.synthesizer_final_round is True
        assert config.cross_examination_depth == 3
        assert config.require_citations is True
        assert config.risk_threshold == 0.7

    def test_default_roles(self):
        """Default roles include key cognitive perspectives."""
        config = DeepAuditConfig()
        assert CognitiveRole.ANALYST in config.roles
        assert CognitiveRole.SKEPTIC in config.roles
        assert CognitiveRole.LATERAL_THINKER in config.roles
        assert CognitiveRole.ADVOCATE in config.roles

    def test_custom_config(self):
        """Config can be customized."""
        config = DeepAuditConfig(
            rounds=4,
            enable_research=False,
            risk_threshold=0.5,
            cross_examination_depth=5,
        )
        assert config.rounds == 4
        assert config.enable_research is False
        assert config.risk_threshold == 0.5
        assert config.cross_examination_depth == 5


class TestAuditFinding:
    """Tests for AuditFinding dataclass."""

    def test_create_finding(self):
        """Finding can be created with category and summary."""
        finding = AuditFinding(
            category="risk",
            summary="Potential security vulnerability",
            details="The code does not validate input",
        )
        assert finding.category == "risk"
        assert finding.summary == "Potential security vulnerability"
        assert finding.details == "The code does not validate input"

    def test_default_values(self):
        """Finding has sensible defaults."""
        finding = AuditFinding(
            category="insight",
            summary="Test",
            details="Details",
        )
        assert finding.agents_agree == []
        assert finding.agents_disagree == []
        assert finding.confidence == 0.0
        assert finding.citations == []
        assert finding.severity == 0.0

    def test_finding_with_agents(self):
        """Finding can track agreeing/disagreeing agents."""
        finding = AuditFinding(
            category="split",
            summary="Disputed recommendation",
            details="Agents disagree on approach",
            agents_agree=["Claude", "Gemini"],
            agents_disagree=["GPT"],
            confidence=0.6,
        )
        assert "Claude" in finding.agents_agree
        assert "GPT" in finding.agents_disagree
        assert finding.confidence == 0.6


class TestDeepAuditVerdict:
    """Tests for DeepAuditVerdict dataclass."""

    def test_create_verdict(self):
        """Verdict can be created with recommendation."""
        verdict = DeepAuditVerdict(
            recommendation="Proceed with caution",
            confidence=0.75,
        )
        assert verdict.recommendation == "Proceed with caution"
        assert verdict.confidence == 0.75

    def test_default_values(self):
        """Verdict has sensible defaults."""
        verdict = DeepAuditVerdict(
            recommendation="Test",
            confidence=0.5,
        )
        assert verdict.findings == []
        assert verdict.unanimous_issues == []
        assert verdict.split_opinions == []
        assert verdict.risk_areas == []
        assert verdict.citations == []
        assert verdict.cross_examination_notes == ""

    def test_verdict_summary(self):
        """summary() generates readable report."""
        verdict = DeepAuditVerdict(
            recommendation="Approve with modifications",
            confidence=0.85,
            unanimous_issues=["Address security concern"],
            split_opinions=["Database choice divided"],
            risk_areas=["Scalability needs attention"],
            citations=["https://example.com/ref1"],
        )
        summary = verdict.summary()

        assert "DEEP AUDIT VERDICT" in summary
        assert "Approve with modifications" in summary
        assert "85%" in summary
        assert "UNANIMOUS ISSUES" in summary
        assert "Address security concern" in summary
        assert "SPLIT OPINIONS" in summary
        assert "RISK AREAS" in summary
        assert "Citations" in summary

    def test_verdict_summary_truncation(self):
        """summary() truncates long values."""
        verdict = DeepAuditVerdict(
            recommendation="A" * 600,  # Longer than 500 chars
            confidence=0.5,
        )
        summary = verdict.summary()
        # Recommendation should be truncated
        assert len([line for line in summary.split("\n") if "Recommendation:" in line][0]) < 600


class TestPreConfiguredConfigs:
    """Tests for pre-configured audit configs."""

    def test_strategy_audit_config(self):
        """STRATEGY_AUDIT has strategic analysis settings."""
        assert STRATEGY_AUDIT.rounds == 6
        assert STRATEGY_AUDIT.enable_research is True
        assert CognitiveRole.DEVIL_ADVOCATE in STRATEGY_AUDIT.roles
        assert STRATEGY_AUDIT.cross_examination_depth == 4
        assert STRATEGY_AUDIT.require_citations is True

    def test_contract_audit_config(self):
        """CONTRACT_AUDIT focuses on document analysis."""
        assert CONTRACT_AUDIT.rounds == 4
        assert CONTRACT_AUDIT.enable_research is False  # Focus on document
        assert CONTRACT_AUDIT.risk_threshold == 0.5  # Lower threshold
        assert CONTRACT_AUDIT.cross_examination_depth == 5
        assert CONTRACT_AUDIT.require_citations is True

    def test_code_architecture_audit_config(self):
        """CODE_ARCHITECTURE_AUDIT has code review settings."""
        assert CODE_ARCHITECTURE_AUDIT.rounds == 5
        assert CODE_ARCHITECTURE_AUDIT.enable_research is True
        assert CODE_ARCHITECTURE_AUDIT.require_citations is False
        assert CODE_ARCHITECTURE_AUDIT.cross_examination_depth == 3


class TestDeepAuditOrchestrator:
    """Tests for DeepAuditOrchestrator class."""

    def test_orchestrator_creation(self):
        """Orchestrator can be created with agents."""
        agent = MagicMock(name="TestAgent")
        orchestrator = DeepAuditOrchestrator(agents=[agent])

        assert orchestrator.agents == [agent]
        assert orchestrator.config is not None
        assert orchestrator.findings == []
        assert orchestrator.round_summaries == []

    def test_orchestrator_with_config(self):
        """Orchestrator accepts custom config."""
        agent = MagicMock(name="TestAgent")
        config = DeepAuditConfig(rounds=3, enable_research=False)
        orchestrator = DeepAuditOrchestrator(agents=[agent], config=config)

        assert orchestrator.config.rounds == 3
        assert orchestrator.config.enable_research is False

    def test_orchestrator_with_research_fn(self):
        """Orchestrator accepts research function."""
        agent = MagicMock(name="TestAgent")

        async def research(query: str) -> str:
            return f"Research results for: {query}"

        orchestrator = DeepAuditOrchestrator(
            agents=[agent],
            research_fn=research,
        )
        assert orchestrator.research_fn is not None

    def test_role_rotator_initialized(self):
        """Orchestrator initializes role rotator."""
        agent = MagicMock(name="TestAgent")
        config = DeepAuditConfig(
            roles=[CognitiveRole.ANALYST, CognitiveRole.SKEPTIC],
            synthesizer_final_round=True,
        )
        orchestrator = DeepAuditOrchestrator(agents=[agent], config=config)

        assert orchestrator.role_rotator is not None

    def test_build_verdict_basic(self):
        """_build_verdict creates verdict from result."""
        agent = MagicMock(name="TestAgent")
        orchestrator = DeepAuditOrchestrator(agents=[agent])

        # Mock debate result
        result = MagicMock()
        result.final_answer = "Test recommendation"
        result.confidence = 0.8
        result.disagreement_report = None
        result.critiques = []

        verdict = orchestrator._build_verdict(result, "Cross exam notes")

        assert verdict.recommendation == "Test recommendation"
        assert verdict.confidence == 0.8
        assert verdict.cross_examination_notes == "Cross exam notes"

    def test_build_verdict_with_disagreement(self):
        """_build_verdict extracts disagreement info."""
        agent = MagicMock(name="TestAgent")
        orchestrator = DeepAuditOrchestrator(agents=[agent])

        # Mock disagreement report
        disagreement_report = MagicMock()
        disagreement_report.unanimous_critiques = ["Issue 1", "Issue 2"]
        disagreement_report.risk_areas = ["Risk 1"]
        disagreement_report.split_opinions = [
            ("Topic1", ["Agent1"], ["Agent2"]),
        ]

        result = MagicMock()
        result.final_answer = "Recommendation"
        result.confidence = 0.7
        result.disagreement_report = disagreement_report
        result.critiques = []

        verdict = orchestrator._build_verdict(result, "")

        assert len(verdict.unanimous_issues) == 2
        assert "Issue 1" in verdict.unanimous_issues
        assert len(verdict.risk_areas) == 1
        assert len(verdict.split_opinions) == 1
        assert "Topic1" in verdict.split_opinions[0]

    def test_build_verdict_with_high_severity_critiques(self):
        """_build_verdict captures high-severity critiques as findings."""
        agent = MagicMock(name="TestAgent")
        config = DeepAuditConfig(risk_threshold=0.6)
        orchestrator = DeepAuditOrchestrator(agents=[agent], config=config)

        # Mock critique
        critique = MagicMock()
        critique.severity = 0.8  # Above threshold
        critique.issues = ["Critical issue found"]
        critique.reasoning = "Detailed reasoning"
        critique.agent = "Agent1"

        result = MagicMock()
        result.final_answer = "Recommendation"
        result.confidence = 0.7
        result.disagreement_report = None
        result.critiques = [critique]

        verdict = orchestrator._build_verdict(result, "")

        assert len(verdict.findings) == 1
        assert verdict.findings[0].category == "risk"
        assert verdict.findings[0].severity == 0.8

    def test_build_verdict_filters_low_severity(self):
        """_build_verdict ignores low-severity critiques."""
        agent = MagicMock(name="TestAgent")
        config = DeepAuditConfig(risk_threshold=0.7)
        orchestrator = DeepAuditOrchestrator(agents=[agent], config=config)

        critique = MagicMock()
        critique.severity = 0.5  # Below threshold
        critique.issues = ["Minor issue"]
        critique.reasoning = "Minor reasoning"
        critique.agent = "Agent1"

        result = MagicMock()
        result.final_answer = "Recommendation"
        result.confidence = 0.7
        result.disagreement_report = None
        result.critiques = [critique]

        verdict = orchestrator._build_verdict(result, "")

        assert len(verdict.findings) == 0
