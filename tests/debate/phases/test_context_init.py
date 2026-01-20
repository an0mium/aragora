"""
Tests for context initialization module.

Tests cover:
- ContextInitializer class
- Fork debate history injection
- Trending topic context
- Historical context fetching
- Pattern injection
- Proposer selection
- Background task management
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.phases.context_init import ContextInitializer


@dataclass
class MockEnv:
    """Mock environment."""

    task: str = "What is the best approach?"
    context: str = ""


@dataclass
class MockAgent:
    """Mock agent."""

    name: str
    role: str = "proposer"


@dataclass
class MockTrendingTopic:
    """Mock trending topic."""

    topic: str = "AI Safety"
    platform: str = "twitter"
    category: str = "technology"
    volume: int = 10000

    def to_debate_prompt(self):
        return f"Discuss: {self.topic}"


@dataclass
class MockDebateContext:
    """Mock debate context."""

    env: MockEnv = field(default_factory=MockEnv)
    agents: list = field(default_factory=list)
    proposers: list = field(default_factory=list)
    partial_messages: list = field(default_factory=list)
    context_messages: list = field(default_factory=list)
    result: Any = None
    historical_context_cache: str = ""
    research_context: str = ""
    evidence_pack: Any = None
    applied_insight_ids: list = field(default_factory=list)
    background_research_task: Any = None
    background_evidence_task: Any = None
    debate_id: str = "test-debate-123"
    domain: str = "general"
    rlm_context: Any = None


class TestContextInitializerInit:
    """Tests for ContextInitializer initialization."""

    def test_default_init(self):
        """Default initialization sets correct defaults."""
        init = ContextInitializer()

        assert init.initial_messages == []
        assert init.trending_topic is None
        assert init.recorder is None
        assert init.auto_fetch_trending is False
        assert init.enable_knowledge_retrieval is True

    def test_custom_init(self):
        """Custom initialization stores all parameters."""
        recorder = MagicMock()
        topic = MockTrendingTopic()
        messages = [{"content": "Hello"}]

        init = ContextInitializer(
            initial_messages=messages,
            trending_topic=topic,
            recorder=recorder,
            auto_fetch_trending=True,
        )

        assert init.initial_messages == messages
        assert init.trending_topic is topic
        assert init.recorder is recorder
        assert init.auto_fetch_trending is True


class TestSelectProposers:
    """Tests for proposer selection."""

    def test_selects_proposer_role_agents(self):
        """Selects agents with proposer role."""
        ctx = MockDebateContext()
        ctx.agents = [
            MockAgent("claude", "proposer"),
            MockAgent("gpt4", "critic"),
            MockAgent("gemini", "proposer"),
        ]

        init = ContextInitializer()
        init._select_proposers(ctx)

        assert len(ctx.proposers) == 2
        names = [p.name for p in ctx.proposers]
        assert "claude" in names
        assert "gemini" in names

    def test_defaults_to_first_agent(self):
        """Defaults to first agent if no proposers."""
        ctx = MockDebateContext()
        ctx.agents = [
            MockAgent("claude", "critic"),
            MockAgent("gpt4", "judge"),
        ]

        init = ContextInitializer()
        init._select_proposers(ctx)

        assert len(ctx.proposers) == 1
        assert ctx.proposers[0].name == "claude"

    def test_empty_agents_empty_proposers(self):
        """Empty agents results in empty proposers."""
        ctx = MockDebateContext()
        ctx.agents = []

        init = ContextInitializer()
        init._select_proposers(ctx)

        assert ctx.proposers == []


class TestInjectForkHistory:
    """Tests for fork history injection."""

    def test_injects_message_objects(self):
        """Injects Message objects directly."""
        from aragora.core import Message

        msg = Message(role="assistant", agent="claude", content="Previous response")
        ctx = MockDebateContext()

        init = ContextInitializer(initial_messages=[msg])
        init._inject_fork_history(ctx)

        assert len(ctx.partial_messages) == 1
        assert ctx.partial_messages[0].content == "Previous response"

    def test_injects_dict_messages(self):
        """Converts dict messages to Message objects."""
        ctx = MockDebateContext()
        messages = [
            {"role": "user", "agent": "user", "content": "Question"},
            {"role": "assistant", "agent": "claude", "content": "Answer"},
        ]

        init = ContextInitializer(initial_messages=messages)
        init._inject_fork_history(ctx)

        assert len(ctx.partial_messages) == 2
        assert ctx.partial_messages[0].content == "Question"

    def test_no_injection_without_messages(self):
        """No injection when no initial messages."""
        ctx = MockDebateContext()

        init = ContextInitializer()
        init._inject_fork_history(ctx)

        assert ctx.partial_messages == []


class TestInjectTrendingTopic:
    """Tests for trending topic injection."""

    def test_injects_topic_context(self):
        """Injects trending topic context."""
        ctx = MockDebateContext()
        topic = MockTrendingTopic()

        init = ContextInitializer(trending_topic=topic)
        init._inject_trending_topic(ctx)

        assert "AI Safety" in ctx.env.context
        assert "twitter" in ctx.env.context
        assert "technology" in ctx.env.context

    def test_appends_to_existing_context(self):
        """Appends topic to existing context."""
        ctx = MockDebateContext()
        ctx.env.context = "Existing context"
        topic = MockTrendingTopic()

        init = ContextInitializer(trending_topic=topic)
        init._inject_trending_topic(ctx)

        assert "Existing context" in ctx.env.context
        assert "AI Safety" in ctx.env.context

    def test_no_injection_without_topic(self):
        """No injection when no trending topic."""
        ctx = MockDebateContext()
        ctx.env.context = "Original"

        init = ContextInitializer()
        init._inject_trending_topic(ctx)

        assert ctx.env.context == "Original"


class TestStartRecorder:
    """Tests for recorder startup."""

    def test_starts_and_records_phase(self):
        """Starts recorder and records debate_start phase."""
        recorder = MagicMock()

        init = ContextInitializer(recorder=recorder)
        init._start_recorder()

        recorder.start.assert_called_once()
        recorder.record_phase_change.assert_called_once_with("debate_start")

    def test_handles_recorder_error(self):
        """Handles recorder errors gracefully."""
        recorder = MagicMock()
        recorder.start.side_effect = RuntimeError("Start failed")

        init = ContextInitializer(recorder=recorder)

        # Should not raise
        init._start_recorder()

    def test_no_op_without_recorder(self):
        """No-op when no recorder provided."""
        init = ContextInitializer()

        # Should not raise
        init._start_recorder()


class TestFetchHistorical:
    """Tests for historical context fetching."""

    @pytest.mark.asyncio
    async def test_fetches_historical_context(self):
        """Fetches and caches historical context."""
        ctx = MockDebateContext()

        async def fetch(task, limit):
            return "Historical debate context"

        embeddings = MagicMock()
        init = ContextInitializer(
            debate_embeddings=embeddings,
            fetch_historical_context=fetch,
        )

        await init._fetch_historical(ctx)

        assert ctx.historical_context_cache == "Historical debate context"

    @pytest.mark.asyncio
    async def test_handles_timeout(self):
        """Handles fetch timeout."""
        ctx = MockDebateContext()

        async def slow_fetch(task, limit):
            await asyncio.sleep(20)
            return "Result"

        embeddings = MagicMock()
        init = ContextInitializer(
            debate_embeddings=embeddings,
            fetch_historical_context=slow_fetch,
        )

        # Mock timeout by patching wait_for
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            await init._fetch_historical(ctx)

        assert ctx.historical_context_cache == ""

    @pytest.mark.asyncio
    async def test_skips_without_embeddings(self):
        """Skips when no debate embeddings."""
        ctx = MockDebateContext()

        init = ContextInitializer()
        await init._fetch_historical(ctx)

        assert ctx.historical_context_cache == ""


class TestInjectInsightPatterns:
    """Tests for insight pattern injection."""

    @pytest.mark.asyncio
    async def test_injects_common_patterns(self):
        """Injects common patterns from InsightStore."""
        ctx = MockDebateContext()

        patterns = [
            MagicMock(pattern="Pattern 1", count=5),
            MagicMock(pattern="Pattern 2", count=3),
        ]

        store = AsyncMock()
        store.get_common_patterns.return_value = patterns
        store.get_relevant_insights.return_value = []

        def format_patterns(p):
            return "## Patterns\n" + "\n".join(x.pattern for x in p)

        init = ContextInitializer(
            insight_store=store,
            format_patterns_for_prompt=format_patterns,
        )

        await init._inject_insight_patterns(ctx)

        assert "Pattern 1" in ctx.env.context
        assert "Pattern 2" in ctx.env.context

    @pytest.mark.asyncio
    async def test_injects_relevant_insights(self):
        """Injects high-confidence insights."""
        ctx = MockDebateContext()
        ctx.domain = "testing"

        insight = MagicMock()
        insight.id = "insight-1"
        insight.title = "Test Insight"
        insight.description = "Important insight"
        insight.confidence = 0.85

        store = AsyncMock()
        store.get_common_patterns.return_value = []
        store.get_relevant_insights.return_value = [insight]

        init = ContextInitializer(insight_store=store)

        await init._inject_insight_patterns(ctx)

        assert "Test Insight" in ctx.env.context
        assert "insight-1" in ctx.applied_insight_ids

    @pytest.mark.asyncio
    async def test_handles_errors(self):
        """Handles insight store errors gracefully."""
        ctx = MockDebateContext()

        store = AsyncMock()
        store.get_common_patterns.side_effect = RuntimeError("Store error")

        init = ContextInitializer(insight_store=store)

        # Should not raise
        await init._inject_insight_patterns(ctx)


class TestInjectMemoryPatterns:
    """Tests for memory pattern injection."""

    def test_injects_memory_patterns(self):
        """Injects patterns from CritiqueStore."""
        ctx = MockDebateContext()

        def get_patterns(limit):
            return "## Memory Patterns\nPattern A"

        init = ContextInitializer(
            memory=MagicMock(),
            get_successful_patterns_from_memory=get_patterns,
        )

        init._inject_memory_patterns(ctx)

        assert "Memory Patterns" in ctx.env.context
        assert "Pattern A" in ctx.env.context

    def test_skips_without_memory(self):
        """Skips when no memory system."""
        ctx = MockDebateContext()
        ctx.env.context = "Original"

        init = ContextInitializer()
        init._inject_memory_patterns(ctx)

        assert ctx.env.context == "Original"


class TestInitialize:
    """Tests for full initialization flow."""

    @pytest.mark.asyncio
    async def test_initializes_debate_result(self):
        """Creates DebateResult on context."""
        ctx = MockDebateContext()
        ctx.agents = [MockAgent("claude")]

        init = ContextInitializer()

        await init.initialize(ctx)

        assert ctx.result is not None
        assert ctx.result.task == "What is the best approach?"

    @pytest.mark.asyncio
    async def test_selects_proposers(self):
        """Selects proposers during initialization."""
        ctx = MockDebateContext()
        ctx.agents = [MockAgent("claude", "proposer"), MockAgent("gpt4", "critic")]

        init = ContextInitializer()

        await init.initialize(ctx)

        assert len(ctx.proposers) == 1
        assert ctx.proposers[0].name == "claude"

    @pytest.mark.asyncio
    async def test_starts_background_research(self):
        """Starts background research task when enabled."""
        ctx = MockDebateContext()
        ctx.agents = [MockAgent("claude")]

        protocol = MagicMock()
        protocol.enable_research = True

        async def research(task):
            return "Research results"

        init = ContextInitializer(
            protocol=protocol,
            perform_research=research,
        )

        await init.initialize(ctx)

        assert ctx.background_research_task is not None


class TestAwaitBackgroundContext:
    """Tests for background task completion."""

    @pytest.mark.asyncio
    async def test_awaits_running_tasks(self):
        """Awaits incomplete background tasks."""
        ctx = MockDebateContext()

        async def slow_task():
            await asyncio.sleep(0.1)
            return "Done"

        ctx.background_research_task = asyncio.create_task(slow_task())

        init = ContextInitializer()
        await init.await_background_context(ctx)

        assert ctx.background_research_task is None

    @pytest.mark.asyncio
    async def test_handles_task_timeout(self):
        """Handles task timeout and cancels."""
        ctx = MockDebateContext()

        async def very_slow_task():
            await asyncio.sleep(100)
            return "Done"

        ctx.background_research_task = asyncio.create_task(very_slow_task())

        init = ContextInitializer()

        # Mock timeout
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            await init.await_background_context(ctx)

        assert ctx.background_research_task is None

    @pytest.mark.asyncio
    async def test_no_op_without_tasks(self):
        """No-op when no background tasks."""
        ctx = MockDebateContext()

        init = ContextInitializer()

        # Should not raise
        await init.await_background_context(ctx)


class TestInjectPulseContext:
    """Tests for Pulse trending topic auto-fetch."""

    @pytest.mark.asyncio
    async def test_fetches_from_pulse(self):
        """Fetches trending topics from Pulse manager."""
        ctx = MockDebateContext()
        topic = MockTrendingTopic()

        pulse = MagicMock()
        # get_trending_topics is async, but select_topic_for_debate is sync
        pulse.get_trending_topics = AsyncMock(return_value=[topic])
        pulse.select_topic_for_debate = MagicMock(return_value=topic)

        init = ContextInitializer(
            pulse_manager=pulse,
            auto_fetch_trending=True,
        )

        await init._inject_pulse_context(ctx)

        assert init.trending_topic is topic

    @pytest.mark.asyncio
    async def test_handles_pulse_timeout(self):
        """Handles Pulse fetch timeout."""
        ctx = MockDebateContext()

        pulse = AsyncMock()
        pulse.get_trending_topics.side_effect = asyncio.TimeoutError

        init = ContextInitializer(pulse_manager=pulse)

        # Should not raise
        await init._inject_pulse_context(ctx)

    @pytest.mark.asyncio
    async def test_skips_without_pulse(self):
        """Skips when no Pulse manager."""
        ctx = MockDebateContext()

        init = ContextInitializer()

        # Should not raise
        await init._inject_pulse_context(ctx)


class TestInjectKnowledgeContext:
    """Tests for Knowledge Mound context injection."""

    @pytest.mark.asyncio
    async def test_injects_knowledge_context(self):
        """Injects knowledge from Knowledge Mound."""
        ctx = MockDebateContext()

        async def fetch_knowledge(task, limit):
            return "## Relevant Knowledge\nPreviously learned: X is true"

        mound = MagicMock()
        init = ContextInitializer(
            knowledge_mound=mound,
            enable_knowledge_retrieval=True,
            fetch_knowledge_context=fetch_knowledge,
        )

        await init._inject_knowledge_context(ctx)

        assert "Relevant Knowledge" in ctx.env.context

    @pytest.mark.asyncio
    async def test_handles_knowledge_timeout(self):
        """Handles knowledge fetch timeout."""
        # Clear the module-level knowledge cache to ensure fresh fetch
        from aragora.debate.phases import context_init
        context_init._knowledge_cache.clear()

        ctx = MockDebateContext()

        async def slow_fetch(task, limit):
            await asyncio.sleep(100)
            return "Knowledge"

        mound = MagicMock()
        init = ContextInitializer(
            knowledge_mound=mound,
            fetch_knowledge_context=slow_fetch,
        )

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            await init._inject_knowledge_context(ctx)

        # Context unchanged
        assert ctx.env.context == ""

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        """Skips when knowledge retrieval disabled."""
        ctx = MockDebateContext()

        init = ContextInitializer(
            knowledge_mound=MagicMock(),
            enable_knowledge_retrieval=False,
        )

        await init._inject_knowledge_context(ctx)

        assert ctx.env.context == ""
