"""Tests for ClaudeMemAdapter."""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.adapters._base import ADAPTER_CIRCUIT_CONFIGS
from aragora.knowledge.mound.adapters.claude_mem_adapter import (
    ClaudeMemAdapter,
    ClaudeMemContextResult,
)


@dataclass
class MockEvidence:
    """Mock Evidence object returned by ClaudeMemConnector.search()."""

    id: str
    content: str
    title: str = ""
    created_at: str | None = None
    metadata: dict = field(default_factory=dict)


@pytest.fixture
def mock_connector():
    """Create a mock ClaudeMemConnector."""
    connector = AsyncMock()
    connector.search.return_value = [
        MockEvidence(
            id="obs_101",
            content="Rate limiting should use sliding window",
            title="Rate Limiting Observation",
            created_at="2026-02-15T10:00:00Z",
            metadata={"source": "claude-mem", "project": "aragora"},
        ),
        MockEvidence(
            id="obs_102",
            content="Deploy with blue-green strategy",
            title="Deployment Note",
            created_at="2026-02-15T11:00:00Z",
            metadata={"source": "claude-mem", "project": "aragora", "files_read": ["deploy.py"]},
        ),
    ]
    return connector


@pytest.fixture
def adapter(mock_connector):
    """Create an adapter with mock connector."""
    return ClaudeMemAdapter(
        connector=mock_connector,
        project="aragora",
        enable_resilience=False,
    )


class TestClaudeMemAdapterInit:
    """Test adapter initialization."""

    def test_adapter_name(self):
        """Test adapter has correct name."""
        adapter = ClaudeMemAdapter(enable_resilience=False)
        assert adapter.adapter_name == "claude_mem"

    def test_init_with_connector(self, mock_connector):
        """Test initialization with connector."""
        adapter = ClaudeMemAdapter(
            connector=mock_connector,
            project="test_project",
            enable_resilience=False,
        )
        assert adapter._connector is mock_connector
        assert adapter._project == "test_project"

    def test_init_without_connector(self):
        """Test initialization without connector (lazy init)."""
        adapter = ClaudeMemAdapter(enable_resilience=False)
        assert adapter._connector is None
        assert adapter._project is None
        assert adapter._synced_ids == set()

    def test_circuit_breaker_config_registered(self):
        """Test circuit breaker config is registered for claude_mem."""
        assert "claude_mem" in ADAPTER_CIRCUIT_CONFIGS
        config = ADAPTER_CIRCUIT_CONFIGS["claude_mem"]
        assert config.failure_threshold == 5
        assert config.timeout_seconds == 30.0


class TestSearchObservations:
    """Test search_observations method."""

    @pytest.mark.asyncio
    async def test_search_delegates_to_connector(self, adapter, mock_connector):
        """Test search_observations delegates to connector.search."""
        results = await adapter.search_observations("rate limiting", limit=5)

        assert len(results) == 2
        mock_connector.search.assert_called_once_with(
            query="rate limiting",
            limit=5,
            project="aragora",
        )

    @pytest.mark.asyncio
    async def test_search_returns_evidence_dicts(self, adapter):
        """Test search results contain expected fields."""
        results = await adapter.search_observations("rate limiting")

        assert len(results) == 2
        first = results[0]
        assert first["id"] == "obs_101"
        assert first["content"] == "Rate limiting should use sliding window"
        assert first["title"] == "Rate Limiting Observation"
        assert first["source"] == "claude_mem"
        assert first["created_at"] == "2026-02-15T10:00:00Z"

    @pytest.mark.asyncio
    async def test_search_empty_on_no_connector(self):
        """Test search returns empty list when connector unavailable."""
        adapter = ClaudeMemAdapter(enable_resilience=False)
        with patch(
            "aragora.knowledge.mound.adapters.claude_mem_adapter.ClaudeMemAdapter._get_connector",
            return_value=None,
        ):
            results = await adapter.search_observations("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_connector_failure(self, adapter, mock_connector):
        """Test search returns empty on connector error."""
        mock_connector.search.side_effect = RuntimeError("Connection refused")
        results = await adapter.search_observations("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_uses_project_override(self, adapter, mock_connector):
        """Test search with explicit project parameter."""
        await adapter.search_observations("test", project="other_project")
        mock_connector.search.assert_called_once_with(
            query="test",
            limit=10,
            project="other_project",
        )


class TestInjectContext:
    """Test inject_context method."""

    @pytest.mark.asyncio
    async def test_inject_context_returns_observations(self, adapter):
        """Test inject_context returns observations as context."""
        result = await adapter.inject_context(topic="rate limiting")

        assert isinstance(result, ClaudeMemContextResult)
        assert result.observations_injected == 2
        assert len(result.context_content) == 2
        assert result.source == "claude_mem"
        assert result.search_time_ms >= 0

    @pytest.mark.asyncio
    async def test_inject_context_no_topic_returns_empty(self, adapter):
        """Test inject_context with no topic returns empty result."""
        result = await adapter.inject_context(topic=None)

        assert result.observations_injected == 0
        assert result.context_content == []
        assert result.total_tokens_estimate == 0

    @pytest.mark.asyncio
    async def test_inject_context_empty_topic_returns_empty(self, adapter):
        """Test inject_context with empty string topic returns empty result."""
        result = await adapter.inject_context(topic="")

        assert result.observations_injected == 0
        assert result.context_content == []

    @pytest.mark.asyncio
    async def test_inject_context_estimates_tokens(self, adapter):
        """Test token estimation in inject_context."""
        result = await adapter.inject_context(topic="deployment")

        # Token estimate = sum of len(content) // 4 for each observation
        expected_tokens = sum(len(c) // 4 for c in result.context_content)
        assert result.total_tokens_estimate == expected_tokens


class TestEvidenceToKnowledgeItem:
    """Test evidence_to_knowledge_item conversion."""

    def test_converts_correctly(self, adapter):
        """Test evidence dict converts to KM item."""
        evidence = {
            "id": "obs_101",
            "content": "Rate limiting observation",
            "title": "Rate Limiting",
            "created_at": "2026-02-15T10:00:00Z",
            "metadata": {
                "project": "aragora",
                "files_read": ["server.py"],
                "files_modified": ["limiter.py"],
            },
        }

        item = adapter.evidence_to_knowledge_item(evidence)

        assert item["id"] == "cm_obs_101"
        assert item["content"] == "Rate limiting observation"
        assert item["source_type"] == "claude_mem"
        assert item["source_id"] == "obs_101"
        assert item["confidence"] == 0.6
        assert item["metadata"]["source"] == "claude_mem"
        assert item["metadata"]["title"] == "Rate Limiting"
        assert item["metadata"]["project"] == "aragora"
        assert item["metadata"]["files_read"] == ["server.py"]
        assert item["metadata"]["files_modified"] == ["limiter.py"]

    def test_generates_correct_id_prefix(self, adapter):
        """Test generated IDs have cm_ prefix."""
        evidence = {"id": "obs_999", "content": "test"}
        item = adapter.evidence_to_knowledge_item(evidence)
        assert item["id"].startswith("cm_")

    def test_generates_content_hash(self, adapter):
        """Test content hash is generated."""
        evidence = {"id": "obs_1", "content": "test content"}
        item = adapter.evidence_to_knowledge_item(evidence)
        assert "content_hash" in item
        assert len(item["content_hash"]) == 16

    def test_handles_missing_fields(self, adapter):
        """Test conversion handles missing optional fields."""
        evidence = {"content": "minimal"}
        item = adapter.evidence_to_knowledge_item(evidence)
        assert item["id"] == "cm_unknown"
        assert item["source_id"] == ""
        assert item["metadata"]["title"] == ""
        assert item["metadata"]["project"] is None


class TestSyncToKM:
    """Test sync_to_km method."""

    @pytest.mark.asyncio
    async def test_sync_observations_to_mound(self, adapter):
        """Test syncing observations to mound."""
        mock_mound = AsyncMock()
        mock_mound.store_knowledge = AsyncMock()

        result = await adapter.sync_to_km(mound=mock_mound)

        assert result.records_synced == 2
        assert result.records_skipped == 0
        assert result.records_failed == 0
        assert result.duration_ms >= 0
        assert mock_mound.store_knowledge.call_count == 2

    @pytest.mark.asyncio
    async def test_sync_skips_already_synced(self, adapter):
        """Test sync skips already-synced observations."""
        mock_mound = AsyncMock()
        mock_mound.store_knowledge = AsyncMock()

        # First sync
        result1 = await adapter.sync_to_km(mound=mock_mound)
        assert result1.records_synced == 2

        # Second sync should skip all
        result2 = await adapter.sync_to_km(mound=mock_mound)
        assert result2.records_synced == 0
        assert result2.records_skipped == 2

    @pytest.mark.asyncio
    async def test_sync_without_mound(self, adapter):
        """Test sync without mound still tracks synced IDs."""
        result = await adapter.sync_to_km(mound=None)

        assert result.records_synced == 2
        assert len(adapter._synced_ids) == 2

    @pytest.mark.asyncio
    async def test_sync_handles_mound_errors(self, adapter):
        """Test sync handles mound store errors gracefully."""
        mock_mound = AsyncMock()
        mock_mound.store_knowledge = AsyncMock(side_effect=RuntimeError("DB error"))

        result = await adapter.sync_to_km(mound=mock_mound)

        assert result.records_failed == 2
        assert result.records_synced == 0
        assert len(result.errors) == 2


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_with_connector(self, adapter):
        """Test health check returns connector status."""
        health = adapter.health_check()

        assert health["adapter"] == "claude_mem"
        assert health["connector_available"] is True
        assert health["synced_observation_count"] == 0
        assert health["project"] == "aragora"

    def test_health_check_without_connector(self):
        """Test health check without connector."""
        adapter = ClaudeMemAdapter(enable_resilience=False)
        with patch(
            "aragora.knowledge.mound.adapters.claude_mem_adapter.ClaudeMemAdapter._get_connector",
            return_value=None,
        ):
            health = adapter.health_check()

        assert health["adapter"] == "claude_mem"
        assert health["connector_available"] is False

    def test_health_check_tracks_synced_count(self, adapter):
        """Test health check tracks synced observation count."""
        adapter._synced_ids.add("obs_1")
        adapter._synced_ids.add("obs_2")

        health = adapter.health_check()
        assert health["synced_observation_count"] == 2


class TestClaudeMemContextResult:
    """Test ClaudeMemContextResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = ClaudeMemContextResult()

        assert result.observations_injected == 0
        assert result.context_content == []
        assert result.total_tokens_estimate == 0
        assert result.search_time_ms == 0
        assert result.source == "claude_mem"

    def test_with_values(self):
        """Test with custom values."""
        result = ClaudeMemContextResult(
            observations_injected=3,
            context_content=["a", "b", "c"],
            total_tokens_estimate=75,
            search_time_ms=42,
        )

        assert result.observations_injected == 3
        assert len(result.context_content) == 3
        assert result.total_tokens_estimate == 75
        assert result.search_time_ms == 42
        assert result.source == "claude_mem"


class TestEventEmission:
    """Test event emission during operations."""

    @pytest.mark.asyncio
    async def test_inject_context_emits_event(self, mock_connector):
        """Test inject_context emits event callback."""
        callback = MagicMock()
        adapter = ClaudeMemAdapter(
            connector=mock_connector,
            project="aragora",
            enable_resilience=False,
            event_callback=callback,
        )

        await adapter.inject_context(topic="test topic")

        callback.assert_called_once()
        event_type = callback.call_args[0][0]
        event_data = callback.call_args[0][1]
        assert event_type == "claude_mem_context_injected"
        assert event_data["topic"] == "test topic"
        assert event_data["observations_injected"] == 2

    @pytest.mark.asyncio
    async def test_sync_emits_event(self, mock_connector):
        """Test sync_to_km emits event callback."""
        callback = MagicMock()
        adapter = ClaudeMemAdapter(
            connector=mock_connector,
            project="aragora",
            enable_resilience=False,
            event_callback=callback,
        )

        await adapter.sync_to_km()

        callback.assert_called_once()
        event_type = callback.call_args[0][0]
        event_data = callback.call_args[0][1]
        assert event_type == "claude_mem_sync_complete"
        assert event_data["synced"] == 2
        assert event_data["errors"] == 0
