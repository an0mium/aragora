"""Tests for Supermemory client."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.connectors.supermemory.client import (
    SupermemoryClient,
    SupermemoryConfig,
    SupermemoryError,
    SupermemoryConnectionError,
    SupermemoryRateLimitError,
    MemoryAddResult,
    SearchResult,
    SearchResponse,
    get_client,
    clear_client,
)


@pytest.fixture
def mock_config():
    """Create a mock config."""
    return SupermemoryConfig(
        api_key="sm_test_api_key_for_testing",
        privacy_filter_enabled=True,
    )


@pytest.fixture
def mock_sdk():
    """Create a mock Supermemory SDK."""
    mock = MagicMock()
    mock.memories.add.return_value = MagicMock(id="mem_123")
    mock.search.execute.return_value = MagicMock(
        results=[
            MagicMock(
                content="Test memory content",
                similarity=0.95,
                id="mem_456",
                container_tag="test_tag",
                metadata={},
            )
        ]
    )
    return mock


@pytest.fixture
def client_with_mock(mock_config, mock_sdk):
    """Create a client with mocked SDK."""
    client = SupermemoryClient(mock_config)
    # Directly inject the mock SDK to avoid import issues
    client._sdk_client = mock_sdk
    client._initialized = True
    return client


class TestSupermemoryClient:
    """Test SupermemoryClient class."""

    def test_client_initialization(self, mock_config):
        """Test client initializes correctly."""
        client = SupermemoryClient(mock_config)

        assert client.config == mock_config
        assert client._privacy_filter is not None
        assert client._initialized is False

    def test_client_no_privacy_filter(self):
        """Test client without privacy filter."""
        config = SupermemoryConfig(
            api_key="sm_test_key",
            privacy_filter_enabled=False,
        )
        client = SupermemoryClient(config)

        assert client._privacy_filter is None

    def test_filter_content(self, client_with_mock):
        """Test content filtering."""
        # Should filter sensitive content
        content = "API key: sk-abcdefghijklmnopqrstuvwxyz123456"
        filtered = client_with_mock._filter_content(content)

        assert "sk-abcdef" not in filtered
        assert "[REDACTED" in filtered

    def test_filter_content_no_filter(self, mock_sdk):
        """Test content not filtered when disabled."""
        config = SupermemoryConfig(
            api_key="sm_test_key",
            privacy_filter_enabled=False,
        )
        client = SupermemoryClient(config)

        content = "API key: sk-abcdefghijklmnopqrstuvwxyz123456"
        assert client._filter_content(content) == content


class TestSupermemoryClientAddMemory:
    """Test add_memory method."""

    @pytest.mark.asyncio
    async def test_add_memory_success(self, client_with_mock, mock_sdk):
        """Test successful memory addition."""
        result = await client_with_mock.add_memory(
            content="Test memory content",
            container_tag="test_tag",
            metadata={"key": "value"},
        )

        assert result.success is True
        assert result.memory_id == "mem_123"
        assert result.container_tag == "test_tag"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_add_memory_default_container(self, client_with_mock, mock_sdk):
        """Test memory uses default container tag."""
        result = await client_with_mock.add_memory(
            content="Test content",
        )

        assert result.success is True
        assert result.container_tag == "aragora"

    @pytest.mark.asyncio
    async def test_add_memory_filters_content(self, client_with_mock, mock_sdk):
        """Test memory content is filtered before sending."""
        content = "Secret: sk-abcdefghijklmnopqrstuvwxyz123456"
        await client_with_mock.add_memory(content=content)

        # Check that the SDK was called with filtered content
        call_args = mock_sdk.memories.add.call_args
        assert "sk-abcdef" not in call_args.kwargs.get("content", "")

    @pytest.mark.asyncio
    async def test_add_memory_general_error(self, client_with_mock, mock_sdk):
        """Test handling general errors (non-retryable)."""
        mock_sdk.memories.add.side_effect = Exception("Invalid request format")

        result = await client_with_mock.add_memory(
            content="Test content",
        )

        assert result.success is False
        assert "Invalid request format" in result.error


class TestSupermemoryClientSearch:
    """Test search method."""

    @pytest.mark.asyncio
    async def test_search_success(self, client_with_mock, mock_sdk):
        """Test successful search."""
        response = await client_with_mock.search(
            query="test query",
            limit=10,
        )

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 1
        assert response.results[0].content == "Test memory content"
        assert response.results[0].similarity == 0.95
        assert response.query == "test query"

    @pytest.mark.asyncio
    async def test_search_with_container(self, client_with_mock, mock_sdk):
        """Test search with container tag filter."""
        await client_with_mock.search(
            query="test query",
            container_tag="specific_tag",
        )

        call_kwargs = mock_sdk.search.execute.call_args.kwargs
        assert call_kwargs.get("container_tag") == "specific_tag"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, client_with_mock, mock_sdk):
        """Test search with no results."""
        mock_sdk.search.execute.return_value = MagicMock(results=[])

        response = await client_with_mock.search(query="no matches")

        assert response.results == []
        assert response.total_found == 0


class TestSupermemoryClientHealthCheck:
    """Test health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client_with_mock, mock_sdk):
        """Test health check when healthy."""
        health = await client_with_mock.health_check()

        assert health["healthy"] is True
        assert health["service"] == "supermemory"
        assert "latency_ms" in health

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, client_with_mock, mock_sdk):
        """Test health check when unhealthy."""
        mock_sdk.search.execute.side_effect = Exception("Connection failed")

        health = await client_with_mock.health_check()

        assert health["healthy"] is False
        assert "error" in health


class TestMemoryAddResult:
    """Test MemoryAddResult dataclass."""

    def test_memory_add_result_success(self):
        """Test successful result."""
        result = MemoryAddResult(
            memory_id="mem_123",
            container_tag="test",
            success=True,
        )

        assert result.memory_id == "mem_123"
        assert result.success is True
        assert result.error is None

    def test_memory_add_result_failure(self):
        """Test failed result."""
        result = MemoryAddResult(
            memory_id="",
            container_tag="test",
            success=False,
            error="Connection failed",
        )

        assert result.success is False
        assert result.error == "Connection failed"


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_basic(self):
        """Test basic search result."""
        result = SearchResult(
            content="Test content",
            similarity=0.95,
            memory_id="mem_123",
        )

        assert result.content == "Test content"
        assert result.similarity == 0.95
        assert result.memory_id == "mem_123"

    def test_search_result_with_metadata(self):
        """Test search result with metadata."""
        result = SearchResult(
            content="Test",
            similarity=0.8,
            metadata={"key": "value"},
        )

        assert result.metadata == {"key": "value"}


class TestSearchResponse:
    """Test SearchResponse dataclass."""

    def test_search_response(self):
        """Test search response."""
        results = [
            SearchResult(content="Result 1", similarity=0.9),
            SearchResult(content="Result 2", similarity=0.8),
        ]
        response = SearchResponse(
            results=results,
            query="test query",
            total_found=2,
            search_time_ms=50,
        )

        assert len(response.results) == 2
        assert response.query == "test query"
        assert response.total_found == 2
        assert response.search_time_ms == 50


class TestSupermemoryExceptions:
    """Test Supermemory exceptions."""

    def test_supermemory_error_recoverable(self):
        """Test recoverable error."""
        error = SupermemoryError("Test error", recoverable=True)

        assert str(error) == "Test error"
        assert error.recoverable is True

    def test_supermemory_error_not_recoverable(self):
        """Test non-recoverable error."""
        error = SupermemoryError("Fatal error", recoverable=False)

        assert error.recoverable is False

    def test_connection_error(self):
        """Test connection error."""
        error = SupermemoryConnectionError("Connection failed")

        assert str(error) == "Connection failed"
        assert isinstance(error, SupermemoryError)

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = SupermemoryRateLimitError("Rate limited", retry_after=60)

        assert str(error) == "Rate limited"
        assert error.retry_after == 60
        assert error.recoverable is True


class TestGetClient:
    """Test get_client function."""

    def test_get_client_no_config(self):
        """Test get_client returns None when no config."""
        clear_client()
        with patch.dict("os.environ", {}, clear=True):
            client = get_client()
            assert client is None

    def test_get_client_with_config(self, mock_config):
        """Test get_client returns client with config."""
        clear_client()
        client = get_client(mock_config)

        assert client is not None
        assert isinstance(client, SupermemoryClient)

    def test_get_client_singleton(self, mock_config):
        """Test get_client returns same instance."""
        clear_client()
        client1 = get_client(mock_config)
        client2 = get_client()  # Should return cached instance

        assert client1 is client2

    def test_clear_client(self, mock_config):
        """Test clear_client clears the instance."""
        clear_client()
        client1 = get_client(mock_config)
        clear_client()
        client2 = get_client(mock_config)

        assert client1 is not client2
