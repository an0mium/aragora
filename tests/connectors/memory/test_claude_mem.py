"""
Tests for the Claude-Mem connector.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.connectors.memory.claude_mem import ClaudeMemConfig, ClaudeMemConnector
from aragora.connectors.base import ConnectorAPIError
from aragora.reasoning.provenance import SourceType


class TestClaudeMemConfig:
    """Tests for ClaudeMemConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ClaudeMemConfig()
        assert config.base_url == "http://localhost:37777"
        assert config.timeout_seconds == 10.0
        assert config.project is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ClaudeMemConfig(
            base_url="http://custom:8080",
            timeout_seconds=30.0,
            project="my-project",
        )
        assert config.base_url == "http://custom:8080"
        assert config.timeout_seconds == 30.0
        assert config.project == "my-project"

    def test_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("ARAGORA_CLAUDE_MEM_BASE_URL", "http://env-host:9999")
        monkeypatch.setenv("ARAGORA_CLAUDE_MEM_TIMEOUT", "15")
        monkeypatch.setenv("ARAGORA_CLAUDE_MEM_PROJECT", "env-project")

        config = ClaudeMemConfig.from_env()
        assert config.base_url == "http://env-host:9999"
        assert config.timeout_seconds == 15.0
        assert config.project == "env-project"

    def test_from_env_defaults(self, monkeypatch):
        """Test from_env uses defaults when env vars not set."""
        monkeypatch.delenv("ARAGORA_CLAUDE_MEM_BASE_URL", raising=False)
        monkeypatch.delenv("ARAGORA_CLAUDE_MEM_TIMEOUT", raising=False)
        monkeypatch.delenv("ARAGORA_CLAUDE_MEM_PROJECT", raising=False)

        config = ClaudeMemConfig.from_env()
        assert config.base_url == "http://localhost:37777"
        assert config.timeout_seconds == 10.0
        assert config.project is None


class TestClaudeMemConnector:
    """Tests for ClaudeMemConnector."""

    @pytest.fixture
    def connector(self):
        """Create a connector with test config."""
        config = ClaudeMemConfig(
            base_url="http://test:37777",
            timeout_seconds=5.0,
            project="test-project",
        )
        return ClaudeMemConnector(config)

    def test_connector_properties(self, connector):
        """Test connector property values."""
        assert connector.name == "Claude-Mem"
        assert connector.source_type == SourceType.EXTERNAL_API

    def test_connector_default_config(self, monkeypatch):
        """Test connector uses default config when none provided."""
        monkeypatch.setenv("ARAGORA_CLAUDE_MEM_BASE_URL", "http://default:37777")
        connector = ClaudeMemConnector()
        assert connector.config.base_url == "http://default:37777"

    @pytest.mark.asyncio
    async def test_search_success(self, connector):
        """Test successful search returns evidence."""
        mock_response = {
            "observations": [
                {
                    "id": "obs-123",
                    "text": "Test observation content",
                    "title": "Test Title",
                    "created_at": "2026-01-01T00:00:00Z",
                    "project": "test-project",
                    "type": "observation",
                    "files_read": ["file1.py"],
                    "files_modified": [],
                },
                {
                    "id": "obs-456",
                    "narrative": "Narrative content",
                    "title": "Narrative Title",
                },
            ]
        }

        with patch.object(connector, "_get_json", return_value=mock_response):
            results = await connector.search("test query", limit=10)

        assert len(results) == 2
        assert results[0].id == "obs_obs-123"
        assert results[0].content == "Test observation content"
        assert results[0].title == "Test Title"
        assert results[0].source_type == SourceType.EXTERNAL_API
        assert results[0].metadata["source"] == "claude-mem"
        assert results[0].metadata["project"] == "test-project"

        assert results[1].id == "obs_obs-456"
        assert results[1].content == "Narrative content"

    @pytest.mark.asyncio
    async def test_search_empty_observations(self, connector):
        """Test search with empty observations returns empty list."""
        with patch.object(connector, "_get_json", return_value={"observations": []}):
            results = await connector.search("test query")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_skips_empty_content(self, connector):
        """Test search skips observations with no content."""
        mock_response = {
            "observations": [
                {"id": "obs-1", "text": "Has content"},
                {"id": "obs-2"},  # No content fields
                {"id": "obs-3", "text": ""},  # Empty content
            ]
        }

        with patch.object(connector, "_get_json", return_value=mock_response):
            results = await connector.search("test query")

        assert len(results) == 1
        assert results[0].id == "obs_obs-1"

    @pytest.mark.asyncio
    async def test_search_with_project_override(self, connector):
        """Test search with project override in kwargs."""
        with patch.object(connector, "_get_json", return_value={"observations": []}) as mock_get:
            await connector.search("test query", project="override-project")

        # Verify the URL includes the project parameter
        call_url = mock_get.call_args[0][0]
        assert "project=override-project" in call_url

    @pytest.mark.asyncio
    async def test_fetch_success(self, connector):
        """Test successful fetch returns evidence."""
        mock_response = {
            "id": "123",
            "text": "Fetched content",
            "title": "Fetched Title",
            "created_at": "2026-01-01T00:00:00Z",
            "project": "test-project",
            "type": "observation",
            "files_read": ["file.py"],
            "files_modified": ["other.py"],
        }

        with patch.object(connector, "_get_json", return_value=mock_response):
            result = await connector.fetch("obs_123")

        assert result is not None
        assert result.id == "obs_123"
        assert result.content == "Fetched content"
        assert result.title == "Fetched Title"
        assert result.metadata["files_read"] == ["file.py"]
        assert result.metadata["files_modified"] == ["other.py"]

    @pytest.mark.asyncio
    async def test_fetch_empty_response(self, connector):
        """Test fetch with empty response returns None."""
        with patch.object(connector, "_get_json", return_value={}):
            result = await connector.fetch("obs_123")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_no_content(self, connector):
        """Test fetch with no content fields returns None."""
        with patch.object(connector, "_get_json", return_value={"id": "123"}):
            result = await connector.fetch("obs_123")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_json_success(self, connector):
        """Test _get_json parses JSON response."""
        mock_data = {"key": "value"}
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = await connector._get_json("http://test/api")

        assert result == mock_data

    @pytest.mark.asyncio
    async def test_get_json_empty_response(self, connector):
        """Test _get_json handles empty response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b""
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = await connector._get_json("http://test/api")

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_json_error(self, connector):
        """Test _get_json raises ConnectorAPIError on failure."""
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            with pytest.raises(ConnectorAPIError) as exc_info:
                await connector._get_json("http://test/api")

        assert "Claude-Mem request failed" in str(exc_info.value)
        assert exc_info.value.connector_name == "Claude-Mem"


class TestClaudeMemIntegration:
    """Integration-style tests (still mocked but test full flow)."""

    @pytest.mark.asyncio
    async def test_search_and_fetch_flow(self):
        """Test complete search and fetch workflow."""
        config = ClaudeMemConfig(base_url="http://test:37777")
        connector = ClaudeMemConnector(config)

        search_response = {
            "observations": [
                {
                    "id": "found-obs",
                    "text": "Found observation",
                    "title": "Found Title",
                }
            ]
        }

        fetch_response = {
            "id": "found-obs",
            "text": "Full observation details",
            "title": "Found Title",
            "project": "test",
        }

        with patch.object(connector, "_get_json") as mock_get:
            mock_get.return_value = search_response
            results = await connector.search("query")

            assert len(results) == 1
            assert results[0].id == "obs_found-obs"

            mock_get.return_value = fetch_response
            detail = await connector.fetch(results[0].id)

            assert detail is not None
            assert detail.content == "Full observation details"
