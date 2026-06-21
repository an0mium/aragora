"""
Tests for Deep Audit mode.

Tests the Heavy3.ai-inspired intensive debate protocol with:
- Cognitive role rotation
- Cross-examination
- Verdict generation with findings
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import fields

from aragora.modes.deep_audit import (
    DeepAuditConfig,
    AuditFinding,
    DeepAuditVerdict,
    DeepAuditOrchestrator,
    run_deep_audit,
    STRATEGY_AUDIT,
    CONTRACT_AUDIT,
    CODE_ARCHITECTURE_AUDIT,
)
from aragora.debate.roles import CognitiveRole


class TestDeepAuditConfig:
    """Tests for DeepAuditConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeepAuditConfig()

        assert config.rounds == 6
        assert config.enable_research is True
        assert len(config.roles) == 4
        assert CognitiveRole.ANALYST in config.roles
        assert CognitiveRole.SKEPTIC in config.roles
        assert CognitiveRole.LATERAL_THINKER in config.roles
        assert CognitiveRole.ADVOCATE in config.roles
        assert config.synthesizer_final_round is True
        assert config.cross_examination_depth == 3
        assert config.require_citations is True
        assert config.risk_threshold == 0.7

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DeepAuditConfig(
            rounds=4,
            enable_research=False,
            roles=[CognitiveRole.ANALYST, CognitiveRole.SKEPTIC],
            synthesizer_final_round=False,
            cross_examination_depth=5,
            require_citations=False,
            risk_threshold=0.5,
        )

        assert config.rounds == 4
        assert config.enable_research is False
        assert len(config.roles) == 2
        assert config.synthesizer_final_round is False
        assert config.cross_examination_depth == 5
        assert config.require_citations is False
        assert config.risk_threshold == 0.5

    def test_config_is_dataclass(self):
        """Test that DeepAuditConfig is a valid dataclass."""
        config = DeepAuditConfig()
        config_fields = {f.name for f in fields(config)}

        expected_fields = {
            "rounds",
            "enable_research",
            "roles",
            "synthesizer_final_round",
            "cross_examination_depth",
            "require_citations",
            "risk_threshold",
        }
        assert config_fields == expected_fields


class TestAuditFinding:
    """Tests for AuditFinding dataclass."""

    def test_finding_creation(self):
        """Test creating an audit finding."""
        finding = AuditFinding(
            category="unanimous",
            summary="Critical security issue found",
            details="The authentication module has a bypass vulnerability",
        )

        assert finding.category == "unanimous"
        assert finding.summary == "Critical security issue found"
        assert finding.details == "The authentication module has a bypass vulnerability"

    def test_finding_default_values(self):
        """Test default values for optional fields."""
        finding = AuditFinding(
            category="risk",
            summary="Performance concern",
            details="Database queries are not optimized",
        )

        assert finding.agents_agree == []
        assert finding.agents_disagree == []
        assert finding.confidence == 0.0
        assert finding.citations == []
        assert finding.severity == 0.0

    def test_finding_with_agents(self):
        """Test finding with agent agreement/disagreement."""
        finding = AuditFinding(
            category="split",
            summary="Architecture choice",
            details="Debate over microservices vs monolith",
            agents_agree=["claude", "gpt-4"],
            agents_disagree=["gemini"],
            confidence=0.7,
        )

        assert finding.agents_agree == ["claude", "gpt-4"]
        assert finding.agents_disagree == ["gemini"]
        assert finding.confidence == 0.7

    def test_finding_with_citations(self):
        """Test finding with citations."""
        finding = AuditFinding(
            category="insight",
            summary="Best practice identified",
            details="Using immutable data structures",
            citations=["https://example.com/best-practices", "RFC 1234"],
        )

        assert len(finding.citations) == 2
        assert "https://example.com/best-practices" in finding.citations

    def test_finding_categories(self):
        """Test all valid finding categories."""
        categories = ["unanimous", "split", "risk", "insight"]

        for category in categories:
            finding = AuditFinding(
                category=category,
                summary=f"Test {category}",
                details="Details",
            )
            assert finding.category == category

    def test_finding_severity_range(self):
        """Test severity values at boundaries."""
        finding_low = AuditFinding(category="risk", summary="Low", details="", severity=0.0)
        finding_high = AuditFinding(category="risk", summary="High", details="", severity=1.0)

        assert finding_low.severity == 0.0
        assert finding_high.severity == 1.0


class TestDeepAuditVerdict:
    """Tests for DeepAuditVerdict dataclass."""

    def test_verdict_creation(self):
        """Test creating a verdict."""
        verdict = DeepAuditVerdict(
            recommendation="Proceed with implementation",
            confidence=0.85,
        )

        assert verdict.recommendation == "Proceed with implementation"
        assert verdict.confidence == 0.85

    def test_verdict_default_values(self):
        """Test default values."""
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

    def test_verdict_with_findings(self):
        """Test verdict with findings."""
        findings = [
            AuditFinding(category="risk", summary="Issue 1", details="Details 1"),
            AuditFinding(category="unanimous", summary="Issue 2", details="Details 2"),
        ]
        verdict = DeepAuditVerdict(
            recommendation="Address issues before proceeding",
            confidence=0.6,
            findings=findings,
        )

        assert len(verdict.findings) == 2
        assert verdict.findings[0].category == "risk"

    def test_verdict_summary_basic(self):
        """Test summary generation with basic verdict."""
        verdict = DeepAuditVerdict(
            recommendation="Proceed with caution",
            confidence=0.75,
        )

        summary = verdict.summary()

        assert "DEEP AUDIT VERDICT" in summary
        assert "Proceed with caution" in summary
        assert "75%" in summary

    def test_verdict_summary_with_unanimous_issues(self):
        """Test summary with unanimous issues."""
        verdict = DeepAuditVerdict(
            recommendation="Address critical issues",
            confidence=0.9,
            unanimous_issues=[
                "Security vulnerability in auth module",
                "Missing input validation",
            ],
        )

        summary = verdict.summary()

        assert "UNANIMOUS ISSUES" in summary
        assert "address immediately" in summary.lower()
        assert "Security vulnerability" in summary
        assert "Missing input validation" in summary

    def test_verdict_summary_with_split_opinions(self):
        """Test summary with split opinions."""
        verdict = DeepAuditVerdict(
            recommendation="Review and decide",
            confidence=0.65,
            split_opinions=[
                "Use microservices architecture",
                "Prefer PostgreSQL over MongoDB",
            ],
        )

        summary = verdict.summary()

        assert "SPLIT OPINIONS" in summary
        assert "review carefully" in summary.lower()

    def test_verdict_summary_with_risk_areas(self):
        """Test summary with risk areas."""
        verdict = DeepAuditVerdict(
            recommendation="Monitor closely",
            confidence=0.7,
            risk_areas=[
                "Database scaling issues",
                "Third-party API dependency",
            ],
        )

        summary = verdict.summary()

        assert "RISK AREAS" in summary
        assert "monitor" in summary.lower()

    def test_verdict_summary_with_citations(self):
        """Test summary with citations."""
        verdict = DeepAuditVerdict(
            recommendation="See references",
            confidence=0.8,
            citations=[
                "https://example.com/doc1",
                "https://example.com/doc2",
            ],
        )

        summary = verdict.summary()

        assert "Citations" in summary
        assert "example.com" in summary

    def test_verdict_summary_truncates_long_content(self):
        """Test that summary truncates long content."""
        long_recommendation = "A" * 1000
        verdict = DeepAuditVerdict(
            recommendation=long_recommendation,
            confidence=0.5,
        )

        summary = verdict.summary()

        # Should truncate to 500 chars
        assert len(verdict.recommendation) == 1000
        assert "A" * 500 in summary
        assert "A" * 600 not in summary

    def test_verdict_summary_limits_items(self):
        """Test that summary limits number of items displayed."""
        verdict = DeepAuditVerdict(
            recommendation="Test",
            confidence=0.5,
            unanimous_issues=[f"Issue {i}" for i in range(10)],
            split_opinions=[f"Opinion {i}" for i in range(10)],
            risk_areas=[f"Risk {i}" for i in range(10)],
            citations=[f"Citation {i}" for i in range(10)],
        )

        summary = verdict.summary()

        # Should show counts but limit displayed items to 5
        assert "10 UNANIMOUS ISSUES" in summary
        assert "Issue 0" in summary
        assert "Issue 4" in summary
        # Item 5 should not be shown (0-indexed, so items 0-4 are shown)
        # The count still shows 10 though


class TestDeepAuditOrchestrator:
    """Tests for DeepAuditOrchestrator class."""

    def create_mock_agent(self, name: str = "test_agent") -> MagicMock:
        """Create a mock agent for testing."""
        agent = MagicMock()
        agent.name = name
        agent.generate = AsyncMock(return_value="Test response")
        return agent

    def test_orchestrator_initialization_default(self):
        """Test orchestrator with default config."""
        agents = [self.create_mock_agent("agent1")]
        orchestrator = DeepAuditOrchestrator(agents)

        assert orchestrator.agents == agents
        assert orchestrator.config.rounds == 6
        assert orchestrator.research_fn is None
        assert orchestrator.findings == []
        assert orchestrator.round_summaries == []
        assert orchestrator.citations == []

    def test_orchestrator_initialization_custom_config(self):
        """Test orchestrator with custom config."""
        agents = [self.create_mock_agent("agent1")]
        config = DeepAuditConfig(rounds=4, enable_research=False)

        orchestrator = DeepAuditOrchestrator(agents, config)

        assert orchestrator.config.rounds == 4
        assert orchestrator.config.enable_research is False

    def test_orchestrator_initialization_with_research_fn(self):
        """Test orchestrator with research function."""
        agents = [self.create_mock_agent("agent1")]

        async def mock_research(query: str) -> str:
            return f"Research results for: {query}"

        orchestrator = DeepAuditOrchestrator(agents, research_fn=mock_research)

        assert orchestrator.research_fn is not None

    def test_orchestrator_role_rotator_initialized(self):
        """Test that role rotator is properly initialized."""
        agents = [self.create_mock_agent("agent1")]
        config = DeepAuditConfig(
            roles=[CognitiveRole.ANALYST, CognitiveRole.SKEPTIC],
            synthesizer_final_round=True,
        )

        orchestrator = DeepAuditOrchestrator(agents, config)

        assert orchestrator.role_rotator is not None
        assert orchestrator.role_rotator.config.enabled is True
        assert orchestrator.role_rotator.config.synthesizer_final_round is True

    def test_orchestrator_multiple_agents(self):
        """Test orchestrator with multiple agents."""
        agents = [
            self.create_mock_agent("claude"),
            self.create_mock_agent("gpt-4"),
            self.create_mock_agent("gemini"),
        ]

        orchestrator = DeepAuditOrchestrator(agents)

        assert len(orchestrator.agents) == 3
        assert orchestrator.agents[0].name == "claude"

    @pytest.mark.asyncio
    async def test_build_verdict_basic(self):
        """Test _build_verdict with basic result."""
        agents = [self.create_mock_agent("agent1")]
        orchestrator = DeepAuditOrchestrator(agents)

        # Create mock debate result
        mock_result = MagicMock()
        mock_result.final_answer = "Proceed with implementation"
        mock_result.confidence = 0.8
        mock_result.disagreement_report = None
        mock_result.critiques = []

        verdict = orchestrator._build_verdict(mock_result, "Cross-exam notes")

        assert verdict.recommendation == "Proceed with implementation"
        assert verdict.confidence == 0.8
        assert verdict.cross_examination_notes == "Cross-exam notes"

    @pytest.mark.asyncio
    async def test_build_verdict_with_disagreement_report(self):
        """Test _build_verdict with disagreement report."""
        agents = [self.create_mock_agent("agent1")]
        orchestrator = DeepAuditOrchestrator(agents)

        # Create mock disagreement report
        mock_report = MagicMock()
        mock_report.unanimous_critiques = ["Issue 1", "Issue 2"]
        mock_report.risk_areas = ["Risk A"]
        mock_report.split_opinions = [
            ("Topic 1", ["claude"], ["gpt-4"]),
        ]

        mock_result = MagicMock()
        mock_result.final_answer = "Test"
        mock_result.confidence = 0.7
        mock_result.disagreement_report = mock_report
        mock_result.critiques = []

        verdict = orchestrator._build_verdict(mock_result, "")

        assert verdict.unanimous_issues == ["Issue 1", "Issue 2"]
        assert verdict.risk_areas == ["Risk A"]
        assert len(verdict.split_opinions) == 1
        assert "Topic 1" in verdict.split_opinions[0]
        assert "Agree: claude" in verdict.split_opinions[0]
        assert "Disagree: gpt-4" in verdict.split_opinions[0]

    @pytest.mark.asyncio
    async def test_build_verdict_with_high_severity_critiques(self):
        """Test _build_verdict adds high severity critiques to findings."""
        agents = [self.create_mock_agent("agent1")]
        config = DeepAuditConfig(risk_threshold=0.7)
        orchestrator = DeepAuditOrchestrator(agents, config)

        # Create mock critique with high severity
        mock_critique = MagicMock()
        mock_critique.severity = 0.9
        mock_critique.issues = ["Critical security flaw"]
        mock_critique.reasoning = "Authentication bypass possible"
        mock_critique.agent = "claude"

        mock_result = MagicMock()
        mock_result.final_answer = "Test"
        mock_result.confidence = 0.5
        mock_result.disagreement_report = None
        mock_result.critiques = [mock_critique]

        verdict = orchestrator._build_verdict(mock_result, "")

        assert len(verdict.findings) == 1
        assert verdict.findings[0].category == "risk"
        assert verdict.findings[0].summary == "Critical security flaw"
        assert verdict.findings[0].severity == 0.9

    @pytest.mark.asyncio
    async def test_build_verdict_ignores_low_severity_critiques(self):
        """Test _build_verdict ignores critiques below threshold."""
        agents = [self.create_mock_agent("agent1")]
        config = DeepAuditConfig(risk_threshold=0.7)
        orchestrator = DeepAuditOrchestrator(agents, config)

        # Create mock critique with low severity
        mock_critique = MagicMock()
        mock_critique.severity = 0.5  # Below threshold
        mock_critique.issues = ["Minor issue"]

        mock_result = MagicMock()
        mock_result.final_answer = "Test"
        mock_result.confidence = 0.8
        mock_result.disagreement_report = None
        mock_result.critiques = [mock_critique]

        verdict = orchestrator._build_verdict(mock_result, "")

        assert len(verdict.findings) == 0

    @pytest.mark.asyncio
    async def test_run_cross_examination(self):
        """Test _run_cross_examination method."""
        mock_agent = self.create_mock_agent("synthesizer")
        mock_agent.generate = AsyncMock(return_value="Cross-examination complete")

        agents = [mock_agent]
        orchestrator = DeepAuditOrchestrator(agents)

        mock_result = MagicMock()
        mock_result.final_answer = "Test answer"
        mock_result.critiques = []

        notes = await orchestrator._run_cross_examination(
            task="Test task",
            result=mock_result,
            findings=None,
        )

        assert notes == "Cross-examination complete"
        mock_agent.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cross_examination_handles_failure(self):
        """Test _run_cross_examination handles agent failure."""
        mock_agent = self.create_mock_agent("synthesizer")
        mock_agent.generate = AsyncMock(side_effect=Exception("API error"))

        agents = [mock_agent]
        orchestrator = DeepAuditOrchestrator(agents)

        mock_result = MagicMock()
        mock_result.final_answer = "Test"
        mock_result.critiques = []

        notes = await orchestrator._run_cross_examination(
            task="Test task",
            result=mock_result,
            findings=None,
        )

        assert "Cross-examination failed" in notes
        assert "API error" in notes


class TestPreConfiguredAudits:
    """Tests for pre-configured audit configurations."""

    def test_strategy_audit_config(self):
        """Test STRATEGY_AUDIT configuration."""
        assert STRATEGY_AUDIT.rounds == 6
        assert STRATEGY_AUDIT.enable_research is True
        assert CognitiveRole.ANALYST in STRATEGY_AUDIT.roles
        assert CognitiveRole.SKEPTIC in STRATEGY_AUDIT.roles
        assert CognitiveRole.LATERAL_THINKER in STRATEGY_AUDIT.roles
        assert CognitiveRole.DEVIL_ADVOCATE in STRATEGY_AUDIT.roles
        assert STRATEGY_AUDIT.cross_examination_depth == 4
        assert STRATEGY_AUDIT.require_citations is True

    def test_contract_audit_config(self):
        """Test CONTRACT_AUDIT configuration."""
        assert CONTRACT_AUDIT.rounds == 4
        assert CONTRACT_AUDIT.enable_research is False  # Focus on document
        assert CognitiveRole.ANALYST in CONTRACT_AUDIT.roles
        assert CognitiveRole.SKEPTIC in CONTRACT_AUDIT.roles
        assert CognitiveRole.ADVOCATE in CONTRACT_AUDIT.roles
        assert CONTRACT_AUDIT.cross_examination_depth == 5
        assert CONTRACT_AUDIT.require_citations is True
        assert CONTRACT_AUDIT.risk_threshold == 0.5  # Lower for contracts

    def test_code_architecture_audit_config(self):
        """Test CODE_ARCHITECTURE_AUDIT configuration."""
        assert CODE_ARCHITECTURE_AUDIT.rounds == 5
        assert CODE_ARCHITECTURE_AUDIT.enable_research is True
        assert CognitiveRole.ANALYST in CODE_ARCHITECTURE_AUDIT.roles
        assert CognitiveRole.SKEPTIC in CODE_ARCHITECTURE_AUDIT.roles
        assert CognitiveRole.LATERAL_THINKER in CODE_ARCHITECTURE_AUDIT.roles
        assert CODE_ARCHITECTURE_AUDIT.cross_examination_depth == 3
        assert CODE_ARCHITECTURE_AUDIT.require_citations is False


class TestRunDeepAudit:
    """Tests for run_deep_audit convenience function."""

    def create_mock_agent(self, name: str = "test_agent") -> MagicMock:
        """Create a mock agent for testing."""
        agent = MagicMock()
        agent.name = name
        agent.generate = AsyncMock(return_value="Test response")
        return agent

    @pytest.mark.asyncio
    async def test_run_deep_audit_creates_orchestrator(self):
        """Test that run_deep_audit creates proper orchestrator."""
        agents = [self.create_mock_agent("agent1")]

        with patch.object(DeepAuditOrchestrator, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = DeepAuditVerdict(recommendation="Test", confidence=0.8)

            result = await run_deep_audit(
                task="Test task",
                agents=agents,
                context="Test context",
            )

            mock_run.assert_called_once_with("Test task", "Test context")
            assert result.recommendation == "Test"

    @pytest.mark.asyncio
    async def test_run_deep_audit_with_custom_config(self):
        """Test run_deep_audit with custom config."""
        agents = [self.create_mock_agent("agent1")]
        config = DeepAuditConfig(rounds=3)

        with patch.object(DeepAuditOrchestrator, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = DeepAuditVerdict(recommendation="Custom", confidence=0.9)

            await run_deep_audit(
                task="Test",
                agents=agents,
                config=config,
            )

            mock_run.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_verdict_no_recommendation(self):
        """Test verdict with empty recommendation."""
        verdict = DeepAuditVerdict(
            recommendation="",
            confidence=0.0,
        )

        summary = verdict.summary()
        assert "DEEP AUDIT VERDICT" in summary

    def test_finding_empty_details(self):
        """Test finding with empty details."""
        finding = AuditFinding(
            category="risk",
            summary="Issue",
            details="",
        )

        assert finding.details == ""

    def test_config_single_role(self):
        """Test config with single role."""
        config = DeepAuditConfig(
            roles=[CognitiveRole.ANALYST],
        )

        assert len(config.roles) == 1
        assert config.roles[0] == CognitiveRole.ANALYST

    def test_config_extreme_rounds(self):
        """Test config with extreme round values."""
        config_low = DeepAuditConfig(rounds=1)
        config_high = DeepAuditConfig(rounds=100)

        assert config_low.rounds == 1
        assert config_high.rounds == 100

    def test_config_extreme_risk_threshold(self):
        """Test config with extreme risk threshold values."""
        config_low = DeepAuditConfig(risk_threshold=0.0)
        config_high = DeepAuditConfig(risk_threshold=1.0)

        assert config_low.risk_threshold == 0.0
        assert config_high.risk_threshold == 1.0

    def test_verdict_with_all_lists_populated(self):
        """Test verdict summary with all lists populated."""
        verdict = DeepAuditVerdict(
            recommendation="Complex verdict",
            confidence=0.75,
            findings=[
                AuditFinding(category="risk", summary="F1", details="D1"),
            ],
            unanimous_issues=["U1", "U2"],
            split_opinions=["S1"],
            risk_areas=["R1"],
            citations=["C1"],
            cross_examination_notes="Extensive notes here",
        )

        summary = verdict.summary()

        assert "UNANIMOUS ISSUES" in summary
        assert "SPLIT OPINIONS" in summary
        assert "RISK AREAS" in summary
        assert "Citations" in summary

    def test_finding_confidence_boundary(self):
        """Test finding confidence at boundaries."""
        finding_zero = AuditFinding(category="insight", summary="Test", details="", confidence=0.0)
        finding_one = AuditFinding(category="insight", summary="Test", details="", confidence=1.0)

        assert finding_zero.confidence == 0.0
        assert finding_one.confidence == 1.0
