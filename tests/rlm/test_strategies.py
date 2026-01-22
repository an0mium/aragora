"""
Tests for RLM decomposition strategies.

Tests cover:
- PeekStrategy: Initial section inspection
- GrepStrategy: Regex-based keyword search
- PartitionMapStrategy: Chunk and parallel processing
- SummarizeStrategy: Recursive summarization
- AutoStrategy: Automatic strategy selection
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.rlm.strategies import (
    BaseStrategy,
    PeekStrategy,
    GrepStrategy,
    PartitionMapStrategy,
    SummarizeStrategy,
    AutoStrategy,
    StrategyResult,
    get_strategy,
)
from aragora.rlm.types import (
    AbstractionLevel,
    DecompositionStrategy,
    RLMConfig,
    RLMContext,
    RLMQuery,
    AbstractionNode,
)


@pytest.fixture
def rlm_config():
    """Create a default RLM config."""
    return RLMConfig()


@pytest.fixture
def sample_content():
    """Create sample document content for testing."""
    return """
# Introduction
This is a comprehensive technical document about software architecture.
It covers microservices, API design, and database patterns.

# Chapter 1: Microservices
Microservices are an architectural style that structures an application
as a collection of loosely coupled services. Each service is fine-grained
and the protocols are lightweight.

## Benefits
- Scalability
- Technology diversity
- Fault isolation

## Challenges
- Network complexity
- Data consistency
- Service discovery

# Chapter 2: API Design
REST APIs follow specific principles for resource-oriented design.

## Best Practices
- Use proper HTTP methods
- Version your APIs
- Handle errors gracefully

# Chapter 3: Database Patterns
Different database patterns suit different use cases.

## CQRS Pattern
Command Query Responsibility Segregation separates read and write operations.

## Event Sourcing
Event sourcing stores all changes as a sequence of events.
"""


@pytest.fixture
def simple_context(sample_content):
    """Create a simple RLM context with content at different levels."""
    # Create nodes at different abstraction levels
    abstract_node = AbstractionNode(
        id="abstract_1",
        content="Document about software architecture: microservices, APIs, databases",
        level=AbstractionLevel.ABSTRACT,
        token_count=10,
    )

    summary_node = AbstractionNode(
        id="summary_1",
        content="Covers microservices benefits and challenges, REST API design principles, and database patterns like CQRS and Event Sourcing.",
        level=AbstractionLevel.SUMMARY,
        token_count=25,
    )

    detailed_node = AbstractionNode(
        id="detailed_1",
        content=sample_content,
        level=AbstractionLevel.DETAILED,
        token_count=200,
    )

    return RLMContext(
        original_content=sample_content,
        original_tokens=235,
        levels={
            AbstractionLevel.ABSTRACT: [abstract_node],
            AbstractionLevel.SUMMARY: [summary_node],
            AbstractionLevel.DETAILED: [detailed_node],
        },
    )


@pytest.fixture
def empty_context():
    """Create an empty RLM context."""
    return RLMContext(
        original_content="Simple document content here.",
        original_tokens=10,
        levels={},
    )


# =============================================================================
# PeekStrategy Tests
# =============================================================================


class TestPeekStrategy:
    """Tests for the PeekStrategy class."""

    def test_strategy_type(self, rlm_config):
        """Test that strategy returns correct type."""
        strategy = PeekStrategy(config=rlm_config)
        assert strategy.strategy_type == DecompositionStrategy.PEEK

    @pytest.mark.asyncio
    async def test_execute_returns_strategy_result(self, rlm_config, simple_context):
        """Test that execute returns a StrategyResult."""
        strategy = PeekStrategy(config=rlm_config)
        query = RLMQuery(query="What is this document about?")

        result = await strategy.execute(query, simple_context)

        assert isinstance(result, StrategyResult)
        assert result.answer is not None
        assert 0 <= result.confidence <= 1
        assert result.tokens_examined >= 0
        assert result.sub_calls == 0  # Peek doesn't make sub-calls

    @pytest.mark.asyncio
    async def test_peek_with_empty_context(self, rlm_config, empty_context):
        """Test peek strategy handles empty context gracefully."""
        strategy = PeekStrategy(config=rlm_config)
        query = RLMQuery(query="What is this?")

        result = await strategy.execute(query, empty_context)

        # Should fall back to original content preview
        assert "Simple document" in result.answer
        assert result.tokens_examined > 0

    @pytest.mark.asyncio
    async def test_peek_limits_token_examination(self, rlm_config, simple_context):
        """Test that peek strategy limits tokens examined per node."""
        strategy = PeekStrategy(config=rlm_config)
        query = RLMQuery(query="Overview please")

        result = await strategy.execute(query, simple_context)

        # Should not examine entire document
        assert result.tokens_examined < simple_context.original_tokens

    @pytest.mark.asyncio
    async def test_peek_confidence_is_moderate(self, rlm_config, simple_context):
        """Test that peek gives moderate confidence (structural view, not answers)."""
        strategy = PeekStrategy(config=rlm_config)
        query = RLMQuery(query="Overview")

        result = await strategy.execute(query, simple_context)

        # Peek should give moderate confidence
        assert result.confidence <= 0.7


# =============================================================================
# GrepStrategy Tests
# =============================================================================


class TestGrepStrategy:
    """Tests for the GrepStrategy class."""

    def test_strategy_type(self, rlm_config):
        """Test that strategy returns correct type."""
        strategy = GrepStrategy(config=rlm_config)
        assert strategy.strategy_type == DecompositionStrategy.GREP

    @pytest.mark.asyncio
    async def test_execute_finds_matching_content(self, rlm_config, simple_context):
        """Test that grep finds content matching search terms."""
        strategy = GrepStrategy(config=rlm_config)
        query = RLMQuery(query="What are the benefits of microservices?")

        result = await strategy.execute(query, simple_context)

        assert isinstance(result, StrategyResult)
        # Should find microservices-related content
        answer_lower = result.answer.lower()
        assert "microservices" in answer_lower or "scalability" in answer_lower

    @pytest.mark.asyncio
    async def test_grep_with_no_matches(self, rlm_config, sample_content):
        """Test grep behavior when no matches are found."""
        strategy = GrepStrategy(config=rlm_config)
        context = RLMContext(
            original_content=sample_content,
            original_tokens=200,
            levels={},
        )
        query = RLMQuery(query="quantum computing details")  # Not in document

        result = await strategy.execute(query, context)

        # Should still return a result
        assert isinstance(result, StrategyResult)

    @pytest.mark.asyncio
    async def test_grep_handles_regex_special_chars(self, rlm_config, sample_content):
        """Test grep handles regex special characters in query safely."""
        strategy = GrepStrategy(config=rlm_config)
        context = RLMContext(
            original_content=sample_content,
            original_tokens=200,
            levels={},
        )
        # Query with regex special chars
        query = RLMQuery(query="What about (APIs) and [patterns]?")

        # Should not raise
        result = await strategy.execute(query, context)
        assert isinstance(result, StrategyResult)


# =============================================================================
# PartitionMapStrategy Tests
# =============================================================================


class TestPartitionMapStrategy:
    """Tests for the PartitionMapStrategy class."""

    def test_strategy_type(self, rlm_config):
        """Test that strategy returns correct type."""
        strategy = PartitionMapStrategy(config=rlm_config)
        assert strategy.strategy_type == DecompositionStrategy.PARTITION_MAP

    @pytest.mark.asyncio
    async def test_execute_partitions_content(self, rlm_config, simple_context):
        """Test that partition+map chunks content for processing."""
        mock_agent = AsyncMock(return_value="Chunk result")
        strategy = PartitionMapStrategy(config=rlm_config, agent_call=mock_agent)
        query = RLMQuery(query="Summarize all topics")

        result = await strategy.execute(query, simple_context)

        assert isinstance(result, StrategyResult)


# =============================================================================
# SummarizeStrategy Tests
# =============================================================================


class TestSummarizeStrategy:
    """Tests for the SummarizeStrategy class."""

    def test_strategy_type(self, rlm_config):
        """Test that strategy returns correct type."""
        strategy = SummarizeStrategy(config=rlm_config)
        assert strategy.strategy_type == DecompositionStrategy.SUMMARIZE

    @pytest.mark.asyncio
    async def test_execute_produces_summary(self, rlm_config, simple_context):
        """Test that summarize strategy produces a summary."""

        async def mock_agent(prompt, context_str, query):
            return "Summary: The document covers architecture topics."

        strategy = SummarizeStrategy(config=rlm_config, agent_call=mock_agent)
        query = RLMQuery(query="Summarize the document")

        result = await strategy.execute(query, simple_context)

        assert isinstance(result, StrategyResult)
        assert len(result.answer) > 0


# =============================================================================
# AutoStrategy Tests
# =============================================================================


class TestAutoStrategy:
    """Tests for automatic strategy selection."""

    def test_strategy_type(self, rlm_config):
        """Test that auto strategy returns correct type."""
        strategy = AutoStrategy(config=rlm_config)
        assert strategy.strategy_type == DecompositionStrategy.AUTO

    @pytest.mark.asyncio
    async def test_auto_executes_query(self, rlm_config, simple_context):
        """Test auto strategy can execute queries."""
        strategy = AutoStrategy(config=rlm_config)
        query = RLMQuery(query="What is the document structure?")

        result = await strategy.execute(query, simple_context)

        assert isinstance(result, StrategyResult)


# =============================================================================
# Strategy Selection Tests
# =============================================================================


class TestStrategySelection:
    """Tests for the get_strategy function."""

    def test_get_strategy_returns_valid_strategy(self, rlm_config):
        """Test that get_strategy returns a valid strategy."""
        strategies = [
            DecompositionStrategy.PEEK,
            DecompositionStrategy.GREP,
            DecompositionStrategy.PARTITION_MAP,
            DecompositionStrategy.SUMMARIZE,
            DecompositionStrategy.AUTO,
        ]

        for strategy_type in strategies:
            strategy = get_strategy(strategy_type, rlm_config)
            assert isinstance(strategy, BaseStrategy)
            assert strategy.strategy_type == strategy_type

    def test_get_strategy_with_agent_call(self, rlm_config):
        """Test that get_strategy passes agent_call to strategy."""
        mock_agent = AsyncMock()
        strategy = get_strategy(
            DecompositionStrategy.PARTITION_MAP,
            rlm_config,
            agent_call=mock_agent,
        )

        assert strategy.agent_call == mock_agent


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestStrategyEdgeCases:
    """Test edge cases and error handling for strategies."""

    @pytest.mark.asyncio
    async def test_empty_query(self, rlm_config, simple_context):
        """Test strategies handle empty queries gracefully."""
        strategies = [
            PeekStrategy(config=rlm_config),
            GrepStrategy(config=rlm_config),
        ]

        for strategy in strategies:
            query = RLMQuery(query="")
            result = await strategy.execute(query, simple_context)
            assert isinstance(result, StrategyResult)

    @pytest.mark.asyncio
    async def test_very_long_query(self, rlm_config, simple_context):
        """Test strategies handle very long queries."""
        strategy = GrepStrategy(config=rlm_config)
        long_query = "Find information about " + "topic " * 100
        query = RLMQuery(query=long_query)

        result = await strategy.execute(query, simple_context)
        assert isinstance(result, StrategyResult)

    @pytest.mark.asyncio
    async def test_unicode_content(self, rlm_config):
        """Test strategies handle unicode content correctly."""
        strategy = GrepStrategy(config=rlm_config)
        unicode_content = "Unicode test: \u4e2d\u6587 \u65e5\u672c\u8a9e \ud83d\ude00"
        context = RLMContext(
            original_content=unicode_content,
            original_tokens=20,
            levels={},
        )
        query = RLMQuery(query="\u4e2d\u6587")

        result = await strategy.execute(query, context)
        assert isinstance(result, StrategyResult)

    @pytest.mark.asyncio
    async def test_strategy_with_partial_levels(self, rlm_config):
        """Test strategies work with partial abstraction levels."""
        # Context with only summary level
        summary_node = AbstractionNode(
            id="summary_only",
            content="Just a summary",
            level=AbstractionLevel.SUMMARY,
            token_count=5,
        )
        context = RLMContext(
            original_content="Original content here",
            original_tokens=10,
            levels={AbstractionLevel.SUMMARY: [summary_node]},
        )

        strategies = [
            PeekStrategy(config=rlm_config),
            GrepStrategy(config=rlm_config),
        ]

        for strategy in strategies:
            query = RLMQuery(query="Test query")
            result = await strategy.execute(query, context)
            assert isinstance(result, StrategyResult)
