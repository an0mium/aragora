"""Tests for SupermemoryAdapter."""

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.adapters.supermemory_adapter import (
    SupermemoryAdapter,
    ContextInjectionResult,
    SyncOutcomeResult,
    SupermemorySearchResult,
)


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    debate_id: str = "test_debate_123"
    conclusion: str = "The rate limiter should use sliding window"
    confidence: float = 0.85
    consensus_type: str = "strong"
    round_count: int = 3
    key_claims: list = None

    def __post_init__(self):
        if self.key_claims is None:
            self.key_claims = ["Use redis", "Sliding window", "10 req/sec"]


@dataclass
class MockSearchResult:
    """Mock search result from Supermemory client."""

    content: str
    similarity: float
    memory_id: str = "mem_123"
    container_tag: str = "aragora_debates"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@pytest.fixture
def mock_client():
    """Create a mock Supermemory client."""
    client = AsyncMock()

    # Mock add_memory
    client.add_memory.return_value = MagicMock(
        success=True,
        memory_id="mem_new_123",
        error=None,
    )

    # Mock search
    client.search.return_value = MagicMock(
        results=[
            MockSearchResult(
                content="Previous debate: use rate limiting",
                similarity=0.92,
            ),
            MockSearchResult(
                content="Earlier insight: sliding window best",
                similarity=0.85,
            ),
        ]
    )

    # Mock health_check
    client.health_check.return_value = {"healthy": True, "latency_ms": 50}

    return client


@pytest.fixture
def mock_config():
    """Create a mock config."""
    config = MagicMock()
    config.get_container_tag.return_value = "aragora_debates"
    config.sync_threshold = 0.7
    return config


@pytest.fixture
def adapter(mock_client, mock_config):
    """Create an adapter with mock client."""
    return SupermemoryAdapter(
        client=mock_client,
        config=mock_config,
        enable_resilience=False,  # Disable for unit tests
    )


class TestSupermemoryAdapterInit:
    """Test adapter initialization."""

    def test_adapter_name(self):
        """Test adapter has correct name."""
        adapter = SupermemoryAdapter(enable_resilience=False)
        assert adapter.adapter_name == "supermemory"

    def test_init_with_client(self, mock_client, mock_config):
        """Test initialization with client."""
        adapter = SupermemoryAdapter(
            client=mock_client,
            config=mock_config,
            enable_resilience=False,
        )
        assert adapter._client == mock_client
        assert adapter._config == mock_config

    def test_init_defaults(self):
        """Test default initialization values."""
        adapter = SupermemoryAdapter(enable_resilience=False)
        assert adapter._min_importance == 0.7
        assert adapter._max_context_items == 10
        assert adapter._enable_privacy_filter is True


class TestSupermemoryAdapterContextInjection:
    """Test context injection."""

    @pytest.mark.asyncio
    async def test_inject_context_success(self, adapter, mock_client):
        """Test successful context injection."""
        result = await adapter.inject_context(debate_topic="rate limiting")

        assert isinstance(result, ContextInjectionResult)
        assert result.memories_injected == 2
        assert len(result.context_content) == 2
        assert result.source == "supermemory"
        assert result.search_time_ms >= 0

        mock_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_context_with_container(self, adapter, mock_client):
        """Test context injection with container filter."""
        await adapter.inject_context(
            debate_topic="rate limiting",
            container_tag="custom_tag",
        )

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["container_tag"] == "custom_tag"

    @pytest.mark.asyncio
    async def test_inject_context_no_client(self):
        """Test context injection without client returns empty result."""
        adapter = SupermemoryAdapter(enable_resilience=False)
        result = await adapter.inject_context(debate_topic="test")

        assert result.memories_injected == 0
        assert result.context_content == []

    @pytest.mark.asyncio
    async def test_inject_context_with_limit(self, adapter, mock_client):
        """Test context injection with custom limit."""
        await adapter.inject_context(
            debate_topic="test",
            limit=5,
        )

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["limit"] == 5


class TestSupermemoryAdapterSyncOutcome:
    """Test debate outcome syncing."""

    @pytest.mark.asyncio
    async def test_sync_outcome_success(self, adapter, mock_client):
        """Test successful outcome sync."""
        debate_result = MockDebateResult()
        result = await adapter.sync_debate_outcome(debate_result)

        assert isinstance(result, SyncOutcomeResult)
        assert result.success is True
        assert result.memory_id == "mem_new_123"
        assert result.synced_at is not None

        mock_client.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_outcome_below_threshold(self, adapter):
        """Test outcome not synced when below threshold."""
        debate_result = MockDebateResult(confidence=0.5)  # Below 0.7 threshold
        result = await adapter.sync_debate_outcome(debate_result)

        assert result.success is True
        assert "threshold" in result.error.lower()

    @pytest.mark.asyncio
    async def test_sync_outcome_no_client(self):
        """Test sync outcome without client (or package not installed)."""
        adapter = SupermemoryAdapter(enable_resilience=False)
        debate_result = MockDebateResult()

        result = await adapter.sync_debate_outcome(debate_result)

        assert result.success is False
        # Error message varies: "not available" or "not installed"
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_sync_outcome_with_privacy_filter(self, adapter, mock_client):
        """Test privacy filter is applied."""
        debate_result = MockDebateResult(
            conclusion="Use API key sk-test1234567890123456789012345678"
        )
        adapter._enable_privacy_filter = True

        # Mock privacy filter
        mock_filter = MagicMock()
        mock_filter.filter.return_value = "Use API key [REDACTED_SK_KEY]"
        adapter._privacy_filter = mock_filter

        await adapter.sync_debate_outcome(debate_result)

        mock_filter.filter.assert_called()

    @pytest.mark.asyncio
    async def test_sync_outcome_error_handling(self, adapter, mock_client):
        """Test error handling during sync."""
        mock_client.add_memory.return_value = MagicMock(
            success=False,
            error="Connection timeout",
        )

        result = await adapter.sync_debate_outcome(MockDebateResult())

        assert result.success is False
        assert result.error == "Connection timeout"


class TestSupermemoryAdapterSearch:
    """Test search operations."""

    @pytest.mark.asyncio
    async def test_search_memories(self, adapter, mock_client):
        """Test searching memories."""
        results = await adapter.search_memories(
            query="rate limiting",
            limit=10,
        )

        assert len(results) == 2
        assert all(isinstance(r, SupermemorySearchResult) for r in results)
        assert results[0].similarity == 0.92
        assert results[0].content == "Previous debate: use rate limiting"

    @pytest.mark.asyncio
    async def test_search_memories_with_similarity_filter(self, adapter, mock_client):
        """Test search with similarity filter."""
        results = await adapter.search_memories(
            query="test",
            min_similarity=0.9,
        )

        # Only one result should pass 0.9 threshold
        assert len(results) == 1
        assert results[0].similarity >= 0.9

    @pytest.mark.asyncio
    async def test_search_memories_no_client(self):
        """Test search without client returns empty."""
        adapter = SupermemoryAdapter(enable_resilience=False)
        results = await adapter.search_memories("test")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_cross_session_insights(self, adapter, mock_client):
        """Test getting cross-session insights."""
        insights = await adapter.get_cross_session_insights(topic="caching")

        assert len(insights) == 2
        assert all("content" in i for i in insights)
        assert all(i["source"] == "supermemory" for i in insights)


class TestSupermemoryAdapterHealth:
    """Test health and stats."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, adapter, mock_client):
        """Test health check when healthy."""
        health = await adapter.health_check()

        assert health["healthy"] is True
        assert health["adapter"] == "supermemory"
        assert "latency_ms" in health

    @pytest.mark.asyncio
    async def test_health_check_no_client(self):
        """Test health check without client (or package not installed)."""
        adapter = SupermemoryAdapter(enable_resilience=False)
        health = await adapter.health_check()

        assert health["healthy"] is False
        # Error is either at top level or nested in 'external'
        has_error = "error" in health or (
            "external" in health and "error" in health.get("external", {})
        )
        assert has_error

    @pytest.mark.asyncio
    async def test_health_check_error(self, adapter, mock_client):
        """Test health check on error."""
        mock_client.health_check.side_effect = Exception("Connection failed")

        health = await adapter.health_check()

        assert health["healthy"] is False
        assert "error" in health

    def test_get_stats(self, adapter):
        """Test getting adapter stats."""
        stats = adapter.get_stats()

        assert stats["adapter"] == "supermemory"
        assert stats["client_initialized"] is True
        assert "min_importance_threshold" in stats
        assert "reverse_flow" in stats


class TestContextInjectionResult:
    """Test ContextInjectionResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = ContextInjectionResult()

        assert result.memories_injected == 0
        assert result.context_content == []
        assert result.total_tokens_estimate == 0
        assert result.source == "supermemory"

    def test_with_values(self):
        """Test with custom values."""
        result = ContextInjectionResult(
            memories_injected=5,
            context_content=["a", "b", "c"],
            total_tokens_estimate=100,
            search_time_ms=50,
        )

        assert result.memories_injected == 5
        assert len(result.context_content) == 3
        assert result.total_tokens_estimate == 100


class TestSyncOutcomeResult:
    """Test SyncOutcomeResult dataclass."""

    def test_success_result(self):
        """Test successful sync result."""
        result = SyncOutcomeResult(
            success=True,
            memory_id="mem_123",
            synced_at=datetime.utcnow(),
        )

        assert result.success is True
        assert result.memory_id == "mem_123"
        assert result.error is None

    def test_failure_result(self):
        """Test failed sync result."""
        result = SyncOutcomeResult(
            success=False,
            error="Connection failed",
        )

        assert result.success is False
        assert result.memory_id is None
        assert result.error == "Connection failed"
