"""Tests for content moderation integration in debate orchestrator.

Covers:
- Moderation check runs when enable_content_moderation=True and blocks spam
- Moderation check is skipped when enable_content_moderation=False (default)
- Graceful degradation when moderation module import fails
- Non-blocking: runtime errors in moderation don't crash the debate
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent, Critique, Environment, Vote
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol


class MockAgent(Agent):
    """Minimal mock agent for content moderation tests."""

    def __init__(self, name: str = "mock-agent"):
        super().__init__(name=name, model="mock-model", role="proposer")
        self.agent_type = "mock"

    async def generate(self, prompt: str, context: list = None) -> str:
        return "Test response"

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=[],
            suggestions=[],
            severity=0.0,
            reasoning="OK",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Test vote",
            confidence=0.8,
            continue_debate=False,
        )


@pytest.fixture
def env():
    """Create a test environment."""
    return Environment(task="Should we adopt microservices?", context="Engineering review")


@pytest.fixture
def agents():
    """Create mock agents."""
    return [MockAgent(name="agent1"), MockAgent(name="agent2")]


@pytest.fixture
def protocol_moderation_enabled():
    """Protocol with content moderation enabled."""
    return DebateProtocol(
        rounds=2,
        consensus="majority",
        enable_content_moderation=True,
    )


@pytest.fixture
def protocol_moderation_disabled():
    """Protocol with content moderation disabled (default)."""
    return DebateProtocol(
        rounds=2,
        consensus="majority",
        enable_content_moderation=False,
    )


class TestContentModerationBlocking:
    """Tests that content moderation blocks spam when enabled."""

    @pytest.mark.asyncio
    async def test_blocks_spam_content(self, env, agents, protocol_moderation_enabled):
        """When moderation is enabled and content is spam, debate should be blocked."""
        from aragora.moderation import ContentModerationError
        from aragora.moderation.spam_integration import SpamCheckResult, SpamVerdict

        blocked_result = SpamCheckResult(
            verdict=SpamVerdict.SPAM,
            confidence=0.95,
            should_block=True,
            reasons=["Detected spam patterns"],
        )

        arena = Arena(env, agents, protocol_moderation_enabled)

        with patch(
            "aragora.moderation.check_debate_content",
            new_callable=AsyncMock,
            return_value=blocked_result,
        ) as mock_check:
            with pytest.raises(ContentModerationError, match="Debate content blocked: spam"):
                await arena._run_inner()

            mock_check.assert_awaited_once_with(
                env.task, context=env.context
            )

    @pytest.mark.asyncio
    async def test_allows_clean_content(self, env, agents, protocol_moderation_enabled):
        """When moderation is enabled and content is clean, debate should proceed."""
        from aragora.moderation.spam_integration import SpamCheckResult, SpamVerdict

        clean_result = SpamCheckResult(
            verdict=SpamVerdict.CLEAN,
            confidence=0.1,
            should_block=False,
        )

        arena = Arena(env, agents, protocol_moderation_enabled)

        with (
            patch(
                "aragora.moderation.check_debate_content",
                new_callable=AsyncMock,
                return_value=clean_result,
            ),
            patch(
                "aragora.debate.orchestrator._runner_setup_debate_infrastructure",
                new_callable=AsyncMock,
            ) as mock_setup,
            patch(
                "aragora.debate.orchestrator._runner_execute_debate_phases",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_record_debate_metrics",
            ),
            patch(
                "aragora.debate.orchestrator._runner_handle_debate_completion",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_cleanup_debate_resources",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            # Should not raise, debate proceeds
            await arena._run_inner()
            mock_setup.assert_awaited_once()


class TestContentModerationSkipped:
    """Tests that moderation is skipped when disabled."""

    @pytest.mark.asyncio
    async def test_skipped_when_disabled(self, env, agents, protocol_moderation_disabled):
        """When enable_content_moderation=False, check_debate_content should not be called."""
        arena = Arena(env, agents, protocol_moderation_disabled)

        with (
            patch(
                "aragora.moderation.check_debate_content",
                new_callable=AsyncMock,
            ) as mock_check,
            patch(
                "aragora.debate.orchestrator._runner_setup_debate_infrastructure",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_execute_debate_phases",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_record_debate_metrics",
            ),
            patch(
                "aragora.debate.orchestrator._runner_handle_debate_completion",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_cleanup_debate_resources",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            await arena._run_inner()
            mock_check.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_default_protocol_skips_moderation(self, env, agents):
        """Default DebateProtocol has enable_content_moderation=False."""
        protocol = DebateProtocol(rounds=2, consensus="majority")
        assert protocol.enable_content_moderation is False


class TestContentModerationGracefulDegradation:
    """Tests for graceful degradation when moderation module is unavailable."""

    @pytest.mark.asyncio
    async def test_import_error_continues_debate(self, env, agents, protocol_moderation_enabled):
        """When aragora.moderation import fails, debate should proceed with a warning."""
        arena = Arena(env, agents, protocol_moderation_enabled)

        with (
            patch(
                "builtins.__import__",
                side_effect=_import_blocker("aragora.moderation"),
            ),
            patch(
                "aragora.debate.orchestrator._runner_setup_debate_infrastructure",
                new_callable=AsyncMock,
            ) as mock_setup,
            patch(
                "aragora.debate.orchestrator._runner_execute_debate_phases",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_record_debate_metrics",
            ),
            patch(
                "aragora.debate.orchestrator._runner_handle_debate_completion",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_cleanup_debate_resources",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            # Should not raise, debate proceeds despite import error
            await arena._run_inner()
            mock_setup.assert_awaited_once()


class TestContentModerationRuntimeErrors:
    """Tests that runtime errors in moderation don't crash the debate."""

    @pytest.mark.asyncio
    async def test_runtime_error_continues_debate(self, env, agents, protocol_moderation_enabled):
        """When check_debate_content raises a runtime error, debate should proceed."""
        arena = Arena(env, agents, protocol_moderation_enabled)

        with (
            patch(
                "aragora.moderation.check_debate_content",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Classifier unavailable"),
            ),
            patch(
                "aragora.debate.orchestrator._runner_setup_debate_infrastructure",
                new_callable=AsyncMock,
            ) as mock_setup,
            patch(
                "aragora.debate.orchestrator._runner_execute_debate_phases",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_record_debate_metrics",
            ),
            patch(
                "aragora.debate.orchestrator._runner_handle_debate_completion",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_cleanup_debate_resources",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            await arena._run_inner()
            mock_setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_runtime_error_logs_warning(
        self, env, agents, protocol_moderation_enabled, caplog
    ):
        """Runtime errors during moderation should be logged as warnings."""
        arena = Arena(env, agents, protocol_moderation_enabled)

        with (
            patch(
                "aragora.moderation.check_debate_content",
                new_callable=AsyncMock,
                side_effect=ValueError("Bad config"),
            ),
            patch(
                "aragora.debate.orchestrator._runner_setup_debate_infrastructure",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_execute_debate_phases",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_record_debate_metrics",
            ),
            patch(
                "aragora.debate.orchestrator._runner_handle_debate_completion",
                new_callable=AsyncMock,
            ),
            patch(
                "aragora.debate.orchestrator._runner_cleanup_debate_resources",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            caplog.at_level(logging.WARNING),
        ):
            await arena._run_inner()
            assert any(
                "Content moderation check failed" in record.message
                for record in caplog.records
            )

    @pytest.mark.asyncio
    async def test_content_moderation_error_is_not_swallowed(
        self, env, agents, protocol_moderation_enabled
    ):
        """ContentModerationError should propagate, not be caught by generic handler."""
        from aragora.moderation import ContentModerationError

        arena = Arena(env, agents, protocol_moderation_enabled)

        with patch(
            "aragora.moderation.check_debate_content",
            new_callable=AsyncMock,
            side_effect=ContentModerationError("Blocked: spam detected"),
        ):
            with pytest.raises(ContentModerationError, match="Blocked: spam detected"):
                await arena._run_inner()


def _import_blocker(blocked_module: str):
    """Create an __import__ side_effect that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocked_import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Simulated: {name} not available")
        return real_import(name, *args, **kwargs)

    return _blocked_import
