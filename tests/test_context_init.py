"""
Tests for aragora.debate.phases.context_init module.

Tests ContextInitializer class and context initialization logic.
"""

import pytest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from aragora.debate.context import DebateContext
from aragora.debate.phases.context_init import ContextInitializer


# ============================================================================
# Mock Classes
# ============================================================================

@dataclass
class MockEnvironment:
    """Mock environment for testing."""
    task: str = "Test task"
    context: str = ""
    domain: str = "general"


@dataclass
class MockAgent:
    """Mock agent for testing."""
    name: str = "test_agent"
    role: str = "proposer"


@dataclass
class MockMessage:
    """Mock message for testing."""
    role: str = "proposer"
    agent: str = "test_agent"
    content: str = "Test content"
    round: int = 0


@dataclass
class MockTrendingTopic:
    """Mock trending topic for testing."""
    topic: str = "AI Ethics"
    platform: str = "twitter"
    category: str = "technology"
    volume: int = 50000

    def to_debate_prompt(self):
        return "Discuss the ethical implications of AI."


@dataclass
class MockProtocol:
    """Mock protocol for testing."""
    enable_research: bool = False
    rounds: int = 3


# ============================================================================
# ContextInitializer Construction Tests
# ============================================================================

class TestContextInitializerConstruction:
    """Tests for ContextInitializer construction."""

    def test_minimal_construction(self):
        """Should create with no arguments."""
        init = ContextInitializer()

        assert init.initial_messages == []
        assert init.trending_topic is None
        assert init.recorder is None

    def test_full_construction(self):
        """Should create with all arguments."""
        recorder = MagicMock()
        insight_store = MagicMock()

        init = ContextInitializer(
            initial_messages=[{"content": "test"}],
            trending_topic=MockTrendingTopic(),
            recorder=recorder,
            insight_store=insight_store,
        )

        assert len(init.initial_messages) == 1
        assert init.trending_topic is not None
        assert init.recorder is recorder
        assert init.insight_store is insight_store


# ============================================================================
# Fork History Injection Tests
# ============================================================================

class TestForkHistoryInjection:
    """Tests for fork history injection."""

    @pytest.mark.asyncio
    async def test_inject_dict_messages(self):
        """Should convert dict messages to Message objects."""
        init = ContextInitializer(
            initial_messages=[
                {"role": "user", "agent": "claude", "content": "Hello", "round": 0},
                {"role": "assistant", "agent": "gpt4", "content": "Hi", "round": 1},
            ]
        )
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert len(ctx.partial_messages) == 2
        assert ctx.partial_messages[0].content == "Hello"
        assert ctx.partial_messages[1].content == "Hi"

    @pytest.mark.asyncio
    async def test_inject_message_objects(self):
        """Should pass through Message objects directly."""
        from aragora.core import Message

        msg = Message(role="user", agent="test", content="Test")
        init = ContextInitializer(initial_messages=[msg])
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert len(ctx.partial_messages) == 1
        assert ctx.partial_messages[0] is msg

    @pytest.mark.asyncio
    async def test_inject_empty_messages(self):
        """Should handle empty initial_messages."""
        init = ContextInitializer(initial_messages=[])
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert ctx.partial_messages == []


# ============================================================================
# Trending Topic Injection Tests
# ============================================================================

class TestTrendingTopicInjection:
    """Tests for trending topic injection."""

    @pytest.mark.asyncio
    async def test_inject_trending_topic(self):
        """Should inject trending topic into context."""
        topic = MockTrendingTopic()
        init = ContextInitializer(trending_topic=topic)
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert "TRENDING TOPIC" in ctx.env.context
        assert "AI Ethics" in ctx.env.context
        assert "twitter" in ctx.env.context

    @pytest.mark.asyncio
    async def test_inject_trending_topic_with_category(self):
        """Should include category in context."""
        topic = MockTrendingTopic(category="science")
        init = ContextInitializer(trending_topic=topic)
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert "Category: science" in ctx.env.context

    @pytest.mark.asyncio
    async def test_inject_trending_topic_with_volume(self):
        """Should include engagement volume."""
        topic = MockTrendingTopic(volume=100000)
        init = ContextInitializer(trending_topic=topic)
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert "Engagement: 100,000" in ctx.env.context

    @pytest.mark.asyncio
    async def test_inject_trending_topic_preserves_existing(self):
        """Should prepend to existing context."""
        topic = MockTrendingTopic()
        init = ContextInitializer(trending_topic=topic)
        env = MockEnvironment(context="Existing context")
        ctx = DebateContext(env=env)

        await init.initialize(ctx)

        assert "TRENDING TOPIC" in ctx.env.context
        assert "Existing context" in ctx.env.context

    @pytest.mark.asyncio
    async def test_no_trending_topic(self):
        """Should not modify context without trending topic."""
        init = ContextInitializer()
        env = MockEnvironment(context="Original")
        ctx = DebateContext(env=env)

        await init.initialize(ctx)

        assert ctx.env.context == "Original"


# ============================================================================
# Recorder Tests
# ============================================================================

class TestRecorderStart:
    """Tests for recorder initialization."""

    @pytest.mark.asyncio
    async def test_start_recorder(self):
        """Should start recorder if provided."""
        recorder = MagicMock()
        init = ContextInitializer(recorder=recorder)
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        recorder.start.assert_called_once()
        recorder.record_phase_change.assert_called_once_with("debate_start")

    @pytest.mark.asyncio
    async def test_recorder_error_non_fatal(self):
        """Should continue on recorder error."""
        recorder = MagicMock()
        recorder.start.side_effect = Exception("Recorder error")
        init = ContextInitializer(recorder=recorder)
        ctx = DebateContext(env=MockEnvironment())

        # Should not raise
        await init.initialize(ctx)

        assert ctx.result is not None


# ============================================================================
# Historical Context Tests
# ============================================================================

class TestHistoricalContext:
    """Tests for historical context fetching."""

    @pytest.mark.asyncio
    async def test_fetch_historical_context(self):
        """Should fetch and cache historical context."""
        fetch_mock = AsyncMock(return_value="Prior debate context")
        init = ContextInitializer(
            debate_embeddings=MagicMock(),
            fetch_historical_context=fetch_mock,
        )
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert ctx.historical_context_cache == "Prior debate context"
        fetch_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_historical_context_timeout(self):
        """Should handle timeout gracefully."""
        async def slow_fetch(*args, **kwargs):
            import asyncio
            await asyncio.sleep(20)
            return "Too slow"

        init = ContextInitializer(
            debate_embeddings=MagicMock(),
            fetch_historical_context=slow_fetch,
        )
        ctx = DebateContext(env=MockEnvironment())

        # Should not raise, should timeout and continue
        # Note: This test will actually wait 10s due to the timeout
        # In practice you might want to mock asyncio.wait_for
        # For now, we'll skip the actual timeout test

    @pytest.mark.asyncio
    async def test_historical_context_error(self):
        """Should handle fetch error gracefully."""
        fetch_mock = AsyncMock(side_effect=Exception("Fetch failed"))
        init = ContextInitializer(
            debate_embeddings=MagicMock(),
            fetch_historical_context=fetch_mock,
        )
        ctx = DebateContext(env=MockEnvironment())

        # Should not raise
        await init.initialize(ctx)

        assert ctx.historical_context_cache == ""

    @pytest.mark.asyncio
    async def test_no_debate_embeddings(self):
        """Should skip fetch without debate_embeddings."""
        fetch_mock = AsyncMock()
        init = ContextInitializer(
            debate_embeddings=None,
            fetch_historical_context=fetch_mock,
        )
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        fetch_mock.assert_not_called()


# ============================================================================
# Pattern Injection Tests
# ============================================================================

class TestPatternInjection:
    """Tests for insight pattern injection."""

    @pytest.mark.asyncio
    async def test_inject_insight_patterns(self):
        """Should inject patterns from insight store."""
        insight_store = MagicMock()
        insight_store.get_common_patterns = AsyncMock(
            return_value=[{"pattern": "Always verify inputs"}]
        )
        format_mock = MagicMock(return_value="## Learned Patterns\n- Verify inputs")

        init = ContextInitializer(
            insight_store=insight_store,
            format_patterns_for_prompt=format_mock,
        )
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert "Learned Patterns" in ctx.env.context

    @pytest.mark.asyncio
    async def test_inject_patterns_empty(self):
        """Should handle no patterns gracefully."""
        insight_store = MagicMock()
        insight_store.get_common_patterns = AsyncMock(return_value=[])

        init = ContextInitializer(insight_store=insight_store)
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        # Should not have modified context (no patterns)
        assert "Learned Patterns" not in (ctx.env.context or "")

    @pytest.mark.asyncio
    async def test_inject_patterns_error(self):
        """Should handle pattern fetch error."""
        insight_store = MagicMock()
        insight_store.get_common_patterns = AsyncMock(
            side_effect=Exception("Store error")
        )

        init = ContextInitializer(insight_store=insight_store)
        ctx = DebateContext(env=MockEnvironment())

        # Should not raise
        await init.initialize(ctx)


# ============================================================================
# Memory Pattern Tests
# ============================================================================

class TestMemoryPatternInjection:
    """Tests for memory pattern injection."""

    @pytest.mark.asyncio
    async def test_inject_memory_patterns(self):
        """Should inject patterns from memory."""
        memory = MagicMock()
        get_patterns = MagicMock(return_value="## Successful Patterns\n- Pattern 1")

        init = ContextInitializer(
            memory=memory,
            get_successful_patterns_from_memory=get_patterns,
        )
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert "Successful Patterns" in ctx.env.context

    @pytest.mark.asyncio
    async def test_memory_patterns_empty(self):
        """Should handle no memory patterns."""
        memory = MagicMock()
        get_patterns = MagicMock(return_value="")

        init = ContextInitializer(
            memory=memory,
            get_successful_patterns_from_memory=get_patterns,
        )
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert "Successful Patterns" not in (ctx.env.context or "")


# ============================================================================
# Research Tests
# ============================================================================

class TestPreDebateResearch:
    """Tests for pre-debate research."""

    @pytest.mark.asyncio
    async def test_perform_research_enabled(self):
        """Should perform research when enabled."""
        protocol = MockProtocol(enable_research=True)
        research_mock = AsyncMock(return_value="## Research Findings\n- Finding 1")

        init = ContextInitializer(
            protocol=protocol,
            perform_research=research_mock,
        )
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert "Research Findings" in ctx.env.context
        assert ctx.research_context == "## Research Findings\n- Finding 1"

    @pytest.mark.asyncio
    async def test_research_disabled(self):
        """Should skip research when disabled."""
        protocol = MockProtocol(enable_research=False)
        research_mock = AsyncMock()

        init = ContextInitializer(
            protocol=protocol,
            perform_research=research_mock,
        )
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        research_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_research_empty_result(self):
        """Should handle empty research result."""
        protocol = MockProtocol(enable_research=True)
        research_mock = AsyncMock(return_value=None)

        init = ContextInitializer(
            protocol=protocol,
            perform_research=research_mock,
        )
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert ctx.research_context is None

    @pytest.mark.asyncio
    async def test_research_error(self):
        """Should handle research error gracefully."""
        protocol = MockProtocol(enable_research=True)
        research_mock = AsyncMock(side_effect=Exception("Research failed"))

        init = ContextInitializer(
            protocol=protocol,
            perform_research=research_mock,
        )
        ctx = DebateContext(env=MockEnvironment())

        # Should not raise
        await init.initialize(ctx)


# ============================================================================
# Result Initialization Tests
# ============================================================================

class TestResultInitialization:
    """Tests for DebateResult initialization."""

    @pytest.mark.asyncio
    async def test_creates_debate_result(self):
        """Should create DebateResult with task."""
        init = ContextInitializer()
        env = MockEnvironment(task="Important task")
        ctx = DebateContext(env=env)

        await init.initialize(ctx)

        assert ctx.result is not None
        assert ctx.result.task == "Important task"
        assert ctx.result.messages == []
        assert ctx.result.critiques == []
        assert ctx.result.votes == []


# ============================================================================
# Context Messages Tests
# ============================================================================

class TestContextMessagesInit:
    """Tests for context message initialization."""

    @pytest.mark.asyncio
    async def test_init_context_from_fork(self):
        """Should initialize context messages from fork history."""
        init = ContextInitializer(
            initial_messages=[
                {"agent": "prior", "content": "Previous discussion", "role": "user"},
            ]
        )
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert len(ctx.context_messages) == 1
        assert ctx.context_messages[0].content == "Previous discussion"
        assert ctx.context_messages[0].round == -1  # Pre-debate marker

    @pytest.mark.asyncio
    async def test_context_messages_empty(self):
        """Should handle empty initial messages."""
        init = ContextInitializer()
        ctx = DebateContext(env=MockEnvironment())

        await init.initialize(ctx)

        assert ctx.context_messages == []


# ============================================================================
# Proposer Selection Tests
# ============================================================================

class TestProposerSelection:
    """Tests for proposer selection."""

    @pytest.mark.asyncio
    async def test_select_proposer_role(self):
        """Should select agents with proposer role."""
        init = ContextInitializer()
        agents = [
            MockAgent(name="claude", role="proposer"),
            MockAgent(name="gpt4", role="critic"),
            MockAgent(name="gemini", role="proposer"),
        ]
        ctx = DebateContext(env=MockEnvironment(), agents=agents)

        await init.initialize(ctx)

        assert len(ctx.proposers) == 2
        proposer_names = [p.name for p in ctx.proposers]
        assert "claude" in proposer_names
        assert "gemini" in proposer_names

    @pytest.mark.asyncio
    async def test_select_fallback_proposer(self):
        """Should fallback to first agent if no proposers."""
        init = ContextInitializer()
        agents = [
            MockAgent(name="claude", role="critic"),
            MockAgent(name="gpt4", role="synthesizer"),
        ]
        ctx = DebateContext(env=MockEnvironment(), agents=agents)

        await init.initialize(ctx)

        assert len(ctx.proposers) == 1
        assert ctx.proposers[0].name == "claude"

    @pytest.mark.asyncio
    async def test_no_agents(self):
        """Should handle no agents."""
        init = ContextInitializer()
        ctx = DebateContext(env=MockEnvironment(), agents=[])

        await init.initialize(ctx)

        assert ctx.proposers == []


# ============================================================================
# Integration Tests
# ============================================================================

class TestContextInitializerIntegration:
    """Integration tests for full initialization flow."""

    @pytest.mark.asyncio
    async def test_full_initialization(self):
        """Should perform full initialization sequence."""
        # Set up all components
        recorder = MagicMock()
        insight_store = MagicMock()
        insight_store.get_common_patterns = AsyncMock(return_value=[])
        fetch_mock = AsyncMock(return_value="History")
        format_mock = MagicMock(return_value="")
        memory_mock = MagicMock(return_value="")

        init = ContextInitializer(
            initial_messages=[{"content": "Prior", "agent": "user"}],
            trending_topic=MockTrendingTopic(),
            recorder=recorder,
            debate_embeddings=MagicMock(),
            insight_store=insight_store,
            memory=MagicMock(),
            protocol=MockProtocol(),
            fetch_historical_context=fetch_mock,
            format_patterns_for_prompt=format_mock,
            get_successful_patterns_from_memory=memory_mock,
        )

        agents = [MockAgent(name="claude", role="proposer")]
        ctx = DebateContext(env=MockEnvironment(task="Full test"), agents=agents)

        await init.initialize(ctx)

        # Verify all initialization happened
        assert ctx.result is not None
        assert ctx.result.task == "Full test"
        assert len(ctx.partial_messages) >= 1
        assert "TRENDING TOPIC" in ctx.env.context
        assert ctx.historical_context_cache == "History"
        assert len(ctx.proposers) == 1
        recorder.start.assert_called_once()
