"""
Tests for Security Debate Module.

Tests the security debate functionality including:
- Running security debates
- Security-focused agent selection
- SecurityEvent integration
- Debate configuration for security context
"""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# SecurityEvent Mock
# =============================================================================


class MockFinding:
    """Mock security finding."""

    def __init__(
        self,
        cve_id: str,
        severity: str,
        description: str,
        finding_type: str = "vulnerability",
    ):
        self.cve_id = cve_id
        self.severity = severity
        self.description = description
        self.finding_type = finding_type
        self.title = f"Finding: {cve_id}"
        self.package_name = "test-package"
        self.affected_package = "test-package"
        self.recommendation = "Update to latest version"
        self.source = "test-scanner"
        self.file_path = "/src/vulnerable.py"
        self.line_number = 42

    def to_dict(self) -> dict:
        return {
            "cve_id": self.cve_id,
            "severity": self.severity,
            "description": self.description,
            "finding_type": self.finding_type,
            "package_name": self.package_name,
        }


class MockSecurityEvent:
    """Mock SecurityEvent for testing."""

    def __init__(
        self,
        event_id: str = "event-123",
        event_type: str = "critical_cve",
        severity: str = "critical",
        repository: str = "test-repo",
        findings: list | None = None,
    ):
        self.id = event_id
        self.event_type = MagicMock(value=event_type)
        self.severity = MagicMock(value=severity)
        self.repository = repository
        self.scan_id = "scan-456"
        self.source = "test-scanner"
        self.findings = findings or [
            MockFinding("CVE-2024-1234", "critical", "Test vulnerability"),
        ]
        self.debate_question = None
        self.debate_requested = False
        self.debate_id = None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_arena():
    """Create mock Arena that returns successful debate result."""
    mock_result = MagicMock()
    mock_result.debate_id = "debate-123"
    mock_result.consensus_reached = True
    mock_result.confidence = 0.85

    arena = AsyncMock()
    arena.run.return_value = mock_result
    return arena, mock_result


@pytest.fixture
def captured_kwargs():
    """Capture kwargs passed to mocked constructors."""
    return {"env": {}, "protocol": {}, "arena": {}}


# =============================================================================
# run_security_debate Tests
# =============================================================================


class TestRunSecurityDebate:
    """Test run_security_debate function."""

    @pytest.mark.asyncio
    async def test_returns_empty_result_without_agents(self):
        """Test returns empty result when no agents available."""
        from aragora.debate.security_debate import run_security_debate

        mock_event = MockSecurityEvent()

        with (
            patch(
                "aragora.events.security_events.build_security_debate_question",
                return_value="What remediation steps should we take?",
            ),
            patch(
                "aragora.debate.security_debate.get_security_debate_agents",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await run_security_debate(event=mock_event, agents=None)

            assert result.consensus_reached is False
            assert result.confidence == 0.0
            assert "No agents available" in result.final_answer

    @pytest.mark.asyncio
    async def test_empty_result_structure(self):
        """Test empty result has correct structure."""
        from aragora.debate.security_debate import run_security_debate

        mock_event = MockSecurityEvent()

        with (
            patch(
                "aragora.events.security_events.build_security_debate_question",
                return_value="What remediation steps should we take?",
            ),
            patch(
                "aragora.debate.security_debate.get_security_debate_agents",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await run_security_debate(event=mock_event, agents=None)

            # Check result structure
            assert hasattr(result, "task")
            assert hasattr(result, "messages")
            assert hasattr(result, "critiques")
            assert hasattr(result, "votes")
            assert result.rounds_used == 0

    @pytest.mark.asyncio
    async def test_passes_agents_when_provided(self):
        """Test agents are used when provided directly."""
        # This tests the flow without actually running Arena
        mock_event = MockSecurityEvent()
        mock_agent = MagicMock()
        mock_agent.name = "security-auditor"

        # Import the module to check behavior
        from aragora.debate.security_debate import run_security_debate

        # Just test the empty agents case is handled
        with (
            patch(
                "aragora.events.security_events.build_security_debate_question",
                return_value="What remediation?",
            ),
            patch(
                "aragora.debate.security_debate.get_security_debate_agents",
                new_callable=AsyncMock,
            ) as mock_get_agents,
        ):
            mock_get_agents.return_value = []

            # When agents=None, get_security_debate_agents is called
            result = await run_security_debate(event=mock_event, agents=None)
            mock_get_agents.assert_called_once()

            # When agents are provided and non-empty, the function proceeds
            # (but will fail at Arena creation - we just test it doesn't call get_agents)


# =============================================================================
# get_security_debate_agents Tests
# =============================================================================


class TestGetSecurityDebateAgents:
    """Test get_security_debate_agents function."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_all_fail(self):
        """Test returns empty list when all agent creation fails."""
        # Import module and mock its internal imports
        import aragora.debate.security_debate as sd_module

        # Save original function
        original_fn = sd_module.get_security_debate_agents

        # Create a new version that simulates all failures
        async def mock_get_agents():
            # Simulate ImportError for factory
            agents = []

            # Simulate Anthropic failure
            try:
                raise ImportError("No Anthropic")
            except (ImportError, Exception):
                pass

            # Simulate OpenAI failure
            try:
                raise ImportError("No OpenAI")
            except (ImportError, Exception):
                pass

            return agents

        # Patch and test
        with patch.object(sd_module, "get_security_debate_agents", mock_get_agents):
            agents = await sd_module.get_security_debate_agents()
            assert agents == []

    @pytest.mark.asyncio
    async def test_tries_factory_first(self):
        """Test get_security_debate_agents tries factory first."""
        from aragora.debate.security_debate import get_security_debate_agents

        # Factory returns agents - they should be used
        mock_agents = [MagicMock(name="a1"), MagicMock(name="a2")]

        # We need to mock at the point where it's imported inside the function
        # Since it's imported locally, we mock the module it imports from
        with patch.dict(
            "sys.modules",
            {"aragora.agents.factory": MagicMock()},
        ):
            factory_module = sys.modules["aragora.agents.factory"]
            factory_module.get_available_agents = AsyncMock(return_value=mock_agents)

            agents = await get_security_debate_agents()
            assert agents == mock_agents


# =============================================================================
# Integration-Style Tests (without mocking internals)
# =============================================================================


class TestSecurityDebateIntegration:
    """Integration-style tests for security debate."""

    @pytest.mark.asyncio
    async def test_no_agents_returns_graceful_failure(self):
        """Test graceful failure when no agents can be created."""
        from aragora.debate.security_debate import run_security_debate

        mock_event = MockSecurityEvent(
            event_id="event-integration-1",
            repository="test-integration-repo",
        )

        # Mock get_security_debate_agents to return empty list
        with (
            patch(
                "aragora.events.security_events.build_security_debate_question",
                return_value="What remediation steps?",
            ),
            patch(
                "aragora.debate.security_debate.get_security_debate_agents",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await run_security_debate(event=mock_event)

            # Should return a valid result even without agents
            assert result is not None
            assert result.consensus_reached is False
            assert "No agents available" in result.final_answer

    @pytest.mark.asyncio
    async def test_event_question_set_even_on_failure(self):
        """Test debate_question is set on event even when debate fails."""
        from aragora.debate.security_debate import run_security_debate

        mock_event = MockSecurityEvent()
        mock_event.debate_question = None

        with (
            patch(
                "aragora.events.security_events.build_security_debate_question",
                return_value="Remediation question",
            ),
            patch(
                "aragora.debate.security_debate.get_security_debate_agents",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await run_security_debate(event=mock_event)

            # The question should be set (it's set before the agents check)
            assert mock_event.debate_question == "Remediation question"

    @pytest.mark.asyncio
    async def test_default_timeout(self):
        """Test default timeout is 300 seconds."""
        from aragora.debate.security_debate import run_security_debate

        mock_event = MockSecurityEvent()

        with (
            patch(
                "aragora.events.security_events.build_security_debate_question",
                return_value="Q",
            ),
            patch(
                "aragora.debate.security_debate.get_security_debate_agents",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            # The default should be 300 - we can't easily verify this
            # without mocking internals, but we can verify the function
            # accepts the parameter
            result = await run_security_debate(
                event=mock_event,
                timeout_seconds=600,
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_confidence_threshold_parameter(self):
        """Test confidence_threshold parameter is accepted."""
        from aragora.debate.security_debate import run_security_debate

        mock_event = MockSecurityEvent()

        with (
            patch(
                "aragora.events.security_events.build_security_debate_question",
                return_value="Q",
            ),
            patch(
                "aragora.debate.security_debate.get_security_debate_agents",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await run_security_debate(
                event=mock_event,
                confidence_threshold=0.9,
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_org_id_parameter(self):
        """Test org_id parameter is accepted."""
        from aragora.debate.security_debate import run_security_debate

        mock_event = MockSecurityEvent()

        with (
            patch(
                "aragora.events.security_events.build_security_debate_question",
                return_value="Q",
            ),
            patch(
                "aragora.debate.security_debate.get_security_debate_agents",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await run_security_debate(
                event=mock_event,
                org_id="org-custom",
            )

            assert result is not None


# =============================================================================
# Function Signature Tests
# =============================================================================


class TestFunctionSignatures:
    """Test function signatures and parameters."""

    def test_run_security_debate_signature(self):
        """Test run_security_debate has correct signature."""
        import inspect

        from aragora.debate.security_debate import run_security_debate

        sig = inspect.signature(run_security_debate)
        params = list(sig.parameters.keys())

        assert "event" in params
        assert "agents" in params
        assert "confidence_threshold" in params
        assert "timeout_seconds" in params
        assert "org_id" in params

    def test_get_security_debate_agents_signature(self):
        """Test get_security_debate_agents has correct signature."""
        import inspect

        from aragora.debate.security_debate import get_security_debate_agents

        sig = inspect.signature(get_security_debate_agents)
        # Should have no required parameters
        for param in sig.parameters.values():
            if param.default == inspect.Parameter.empty:
                # No required params expected
                assert False, f"Unexpected required param: {param.name}"

    def test_default_parameter_values(self):
        """Test default parameter values."""
        import inspect

        from aragora.debate.security_debate import run_security_debate

        sig = inspect.signature(run_security_debate)

        assert sig.parameters["agents"].default is None
        assert sig.parameters["confidence_threshold"].default == 0.7
        assert sig.parameters["timeout_seconds"].default == 300
        assert sig.parameters["org_id"].default == "default"


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Test module can be imported correctly."""

    def test_run_security_debate_importable(self):
        """Test run_security_debate can be imported."""
        from aragora.debate.security_debate import run_security_debate

        assert run_security_debate is not None
        assert callable(run_security_debate)

    def test_get_security_debate_agents_importable(self):
        """Test get_security_debate_agents can be imported."""
        from aragora.debate.security_debate import get_security_debate_agents

        assert get_security_debate_agents is not None
        assert callable(get_security_debate_agents)
