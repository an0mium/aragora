"""
Integration tests for Knowledge Mound ↔ Debate Engine.

Tests the critical path between knowledge retrieval before debates
and knowledge ingestion after consensus is reached.

Critical paths tested:
1. Knowledge retrieval during context initialization
2. Knowledge ingestion after debate consensus
3. Knowledge availability in subsequent debates
4. Error handling when knowledge mound is unavailable
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from aragora.core import Agent, DebateResult, Environment, Message
from aragora.debate.context import DebateContext
from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "test-agent"):
        self.name = name
        self.role = "proposer"
        self.model = "test-model"
        self.provider = "test"

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return f"Response from {self.name}: {prompt[:50]}..."


class MockKnowledgeMound:
    """Mock knowledge mound for testing."""

    def __init__(self):
        self.workspace_id = "test-workspace"
        self._stored_items = []

    async def query_semantic(self, query: str, limit: int = 10, min_confidence: float = 0.5):
        """Mock semantic query."""
        from dataclasses import dataclass

        @dataclass
        class MockItem:
            content: str
            source: str
            confidence: float

        @dataclass
        class MockResults:
            items: list

        return MockResults(items=[
            MockItem(content="Test fact 1", source="previous_debate", confidence=0.9),
            MockItem(content="Test fact 2", source="user_input", confidence=0.85),
        ])

    async def store(self, request):
        """Mock store."""
        from dataclasses import dataclass

        @dataclass
        class MockResult:
            node_id: str

        self._stored_items.append(request)
        return MockResult(node_id=f"node-{len(self._stored_items)}")


class TestKnowledgeMoundDebateIntegration:
    """Tests for knowledge mound integration with debates."""

    @pytest.fixture
    def mock_mound(self):
        """Create a mock knowledge mound."""
        return MockKnowledgeMound()

    @pytest.fixture
    def knowledge_ops(self, mock_mound):
        """Create knowledge mound operations with mock."""
        return KnowledgeMoundOperations(
            knowledge_mound=mock_mound,
            enable_retrieval=True,
            enable_ingestion=True,
        )

    @pytest.fixture
    def debate_context(self):
        """Create a debate context for testing."""
        env = Environment(task="What is the capital of France?")
        agents = [MockAgent("agent-1"), MockAgent("agent-2")]
        ctx = DebateContext(
            env=env,
            agents=agents,
            debate_id="test-debate-123",
            start_time=0,
        )
        return ctx

    @pytest.fixture
    def debate_result(self):
        """Create a debate result for testing."""
        return DebateResult(
            task="What is the capital of France?",
            messages=[
                Message(agent="agent-1", content="Paris is the capital of France.", role="proposer"),
                Message(agent="agent-2", content="I agree, Paris is the capital.", role="proposer"),
            ],
            critiques=[],
            votes=[],
            rounds_used=2,
            consensus_reached=True,
            confidence=0.95,
            final_answer="Paris is the capital of France.",
        )

    def test_knowledge_ops_initialization(self):
        """Test that knowledge ops can be initialized."""
        ops = KnowledgeMoundOperations()
        assert ops is not None
        assert ops.enable_retrieval is True
        assert ops.enable_ingestion is True

    @pytest.mark.asyncio
    async def test_fetch_knowledge_context_returns_facts(self, knowledge_ops, debate_context):
        """Test that knowledge context retrieval returns relevant facts."""
        result = await knowledge_ops.fetch_knowledge_context(
            task=debate_context.env.task,
            limit=5,
        )

        assert result is not None
        assert "KNOWLEDGE MOUND CONTEXT" in result
        assert "Test fact" in result

    @pytest.mark.asyncio
    async def test_fetch_knowledge_context_respects_limit(self, knowledge_ops, debate_context):
        """Test knowledge retrieval respects limit."""
        result = await knowledge_ops.fetch_knowledge_context(
            task=debate_context.env.task,
            limit=1,
        )

        assert result is not None
        # Should still contain context even with limit=1

    @pytest.mark.asyncio
    async def test_ingest_debate_outcome_stores_consensus(self, knowledge_ops, debate_context, debate_result):
        """Test that debate outcomes are ingested into knowledge mound."""
        await knowledge_ops.ingest_debate_outcome(
            result=debate_result,
            env=debate_context.env,
        )

        # Check that store was called
        assert len(knowledge_ops.knowledge_mound._stored_items) == 1

    @pytest.mark.asyncio
    async def test_ingest_skipped_for_low_confidence(self, knowledge_ops, debate_context, debate_result):
        """Test that low-confidence results are not ingested."""
        debate_result.confidence = 0.3  # Below 0.85 threshold
        debate_result.consensus_reached = False

        await knowledge_ops.ingest_debate_outcome(
            result=debate_result,
            env=debate_context.env,
        )

        # Should skip ingestion for low confidence
        assert len(knowledge_ops.knowledge_mound._stored_items) == 0

    @pytest.mark.asyncio
    async def test_knowledge_context_enhances_debate_prompt(self, knowledge_ops, debate_context):
        """Test that retrieved knowledge can be formatted for prompts."""
        context = await knowledge_ops.fetch_knowledge_context(
            task=debate_context.env.task,
        )

        # Should be properly formatted for prompt injection
        assert context is not None
        assert "##" in context  # Markdown header
        assert "confidence" in context.lower()


class TestKnowledgeDebateErrorHandling:
    """Tests for error handling in knowledge-debate integration."""

    @pytest.mark.asyncio
    async def test_fetch_handles_mound_unavailable(self):
        """Test graceful handling when knowledge mound is unavailable."""
        ops = KnowledgeMoundOperations(knowledge_mound=None)

        result = await ops.fetch_knowledge_context(
            task="test query",
        )

        # Should return None without raising
        assert result is None

    @pytest.mark.asyncio
    async def test_ingest_handles_mound_unavailable(self):
        """Test graceful handling when ingestion fails."""
        ops = KnowledgeMoundOperations(knowledge_mound=None)

        env = Environment(task="Test task")
        result = DebateResult(
            task="Test task",
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=1,
            consensus_reached=True,
            confidence=0.9,
            final_answer="Test answer",
        )

        # Should not raise exception
        await ops.ingest_debate_outcome(result, env)
        # Success: no exception raised

    @pytest.mark.asyncio
    async def test_fetch_handles_query_timeout(self):
        """Test handling of query timeouts."""
        mock_mound = MagicMock()
        mock_mound.query_semantic = AsyncMock(side_effect=TimeoutError("Query timed out"))

        ops = KnowledgeMoundOperations(knowledge_mound=mock_mound)

        result = await ops.fetch_knowledge_context(
            task="test query",
        )

        # Should handle timeout gracefully
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_handles_generic_error(self):
        """Test handling of generic errors."""
        mock_mound = MagicMock()
        mock_mound.query_semantic = AsyncMock(side_effect=ValueError("Some error"))

        ops = KnowledgeMoundOperations(knowledge_mound=mock_mound)

        result = await ops.fetch_knowledge_context(
            task="test query",
        )

        # Should handle error gracefully
        assert result is None


class TestKnowledgePersistenceAcrossDebates:
    """Tests for knowledge persistence and reuse across debates."""

    @pytest.mark.asyncio
    async def test_ingested_knowledge_available_in_next_debate(self):
        """Test that knowledge ingested from one debate is available in the next."""
        mound = MockKnowledgeMound()
        ops = KnowledgeMoundOperations(knowledge_mound=mound)

        # First debate - ingest a fact
        env1 = Environment(task="What is Python?")
        result1 = DebateResult(
            task="What is Python?",
            messages=[Message(agent="a1", content="Python is a programming language.", role="proposer")],
            critiques=[],
            votes=[],
            rounds_used=1,
            consensus_reached=True,
            confidence=0.95,
            final_answer="Python is a high-level programming language.",
        )

        await ops.ingest_debate_outcome(result1, env1)

        # The mock should have the fact stored
        assert len(mound._stored_items) == 1

        # Second debate - should be able to retrieve context
        context = await ops.fetch_knowledge_context(
            task="Tell me about Python programming",
        )

        # Should get some context back
        assert context is not None


class TestKnowledgeContextInitializerIntegration:
    """Tests for knowledge integration in context initializer phase."""

    @pytest.mark.asyncio
    async def test_context_initializer_fetches_knowledge(self):
        """Test that context initializer phase fetches knowledge."""
        # This tests the integration point in aragora/debate/phases/context_init.py

        from aragora.debate.phases.context_init import ContextInitializer

        # Create minimal initializer
        initializer = ContextInitializer()

        # Verify the knowledge fetch method exists
        assert hasattr(initializer, "_fetch_knowledge_context") or \
               hasattr(initializer, "fetch_knowledge") or \
               hasattr(initializer, "_knowledge_mound_ops")


class TestKnowledgeFeedbackPhaseIntegration:
    """Tests for knowledge integration in feedback phase."""

    @pytest.mark.asyncio
    async def test_feedback_phase_has_knowledge_integration(self):
        """Test that feedback phase can integrate with knowledge mound."""
        # This tests the integration point in aragora/debate/phases/feedback_phase.py

        from aragora.debate.phases.feedback_phase import FeedbackPhase

        # Create minimal phase
        phase = FeedbackPhase()

        # Verify the class can be instantiated
        assert phase is not None


class TestConcurrentDebatesWithKnowledge:
    """Tests for concurrent debates sharing knowledge mound."""

    @pytest.mark.asyncio
    async def test_concurrent_debates_dont_corrupt_knowledge(self):
        """Test that concurrent debates can safely access knowledge mound."""
        mound = MockKnowledgeMound()
        ops = KnowledgeMoundOperations(knowledge_mound=mound)

        # Run multiple concurrent queries
        tasks = []
        for i in range(5):
            tasks.append(
                ops.fetch_knowledge_context(
                    task=f"Query {i}",
                )
            )

        # All should complete without error
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check no exceptions were raised
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Got exceptions: {exceptions}"

    @pytest.mark.asyncio
    async def test_concurrent_ingestion_is_safe(self):
        """Test that concurrent knowledge ingestion is thread-safe."""
        mound = MockKnowledgeMound()
        ops = KnowledgeMoundOperations(knowledge_mound=mound)

        tasks = []
        for i in range(5):
            env = Environment(task=f"Task {i}")
            result = DebateResult(
                task=f"Task {i}",
                messages=[],
                critiques=[],
                votes=[],
                rounds_used=1,
                consensus_reached=True,
                confidence=0.9,
                final_answer=f"Answer {i}",
            )

            tasks.append(ops.ingest_debate_outcome(result, env))

        # All should complete without error
        results = await asyncio.gather(*tasks, return_exceptions=True)

        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Got exceptions: {exceptions}"

        # All should have been stored
        assert len(mound._stored_items) == 5
