"""
Tests for the Generic Tax Connector.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.tax.generic import (
    GenericTaxConnector,
    _extract_results,
    _item_to_evidence,
)
from aragora.reasoning.provenance import SourceType


class TestGenericTaxConnector:
    """Tests for GenericTaxConnector."""

    @pytest.fixture
    def connector(self, monkeypatch):
        """Create a connector with test config."""
        monkeypatch.setenv("TAX_US_API_BASE", "http://test-tax-api.local")
        monkeypatch.setenv("TAX_US_API_KEY", "test-api-key")
        return GenericTaxConnector(jurisdiction="US")

    @pytest.fixture
    def unconfigured_connector(self, monkeypatch):
        """Create an unconfigured connector."""
        monkeypatch.delenv("TAX_UK_API_BASE", raising=False)
        monkeypatch.delenv("TAX_UK_SEARCH_URL", raising=False)
        return GenericTaxConnector(jurisdiction="UK")

    def test_connector_properties(self, connector):
        """Test connector property values."""
        assert connector.name == "Tax:US"
        assert connector.source_type == SourceType.DOCUMENT
        assert connector.jurisdiction == "US"

    def test_connector_lowercase_jurisdiction(self):
        """Test jurisdiction is uppercased."""
        connector = GenericTaxConnector(jurisdiction="uk")
        assert connector.jurisdiction == "UK"
        assert connector.name == "Tax:UK"

    def test_is_configured(self, connector, unconfigured_connector):
        """Test is_configured property."""
        assert connector.is_configured is True
        assert unconfigured_connector.is_configured is False

    def test_env_prefix(self, connector):
        """Test environment prefix generation."""
        assert connector._env_prefix() == "TAX_US_"

    def test_base_url(self, connector, monkeypatch):
        """Test base URL retrieval from env."""
        assert connector._base_url() == "http://test-tax-api.local"

        monkeypatch.delenv("TAX_US_API_BASE")
        assert connector._base_url() is None

    def test_search_url(self, monkeypatch):
        """Test search URL retrieval and resolution."""
        monkeypatch.setenv("TAX_CA_SEARCH_URL", "http://search.local/search")
        connector = GenericTaxConnector(jurisdiction="CA")
        assert connector._search_url() == "http://search.local/search"
        assert connector._resolve_search_url() == "http://search.local/search"

    def test_resolve_search_url_from_base(self, connector):
        """Test search URL resolution from base URL."""
        assert connector._resolve_search_url() == "http://test-tax-api.local/search"

    def test_resolve_method(self, connector, monkeypatch):
        """Test HTTP method resolution."""
        assert connector._resolve_method() == "GET"

        monkeypatch.setenv("TAX_US_SEARCH_METHOD", "POST")
        assert connector._resolve_method() == "POST"

        monkeypatch.setenv("TAX_US_SEARCH_METHOD", "post")
        assert connector._resolve_method() == "POST"

    def test_resolve_param_names(self, connector, monkeypatch):
        """Test query parameter name resolution."""
        query_key, limit_key = connector._resolve_param_names()
        assert query_key == "q"
        assert limit_key == "limit"

        monkeypatch.setenv("TAX_US_SEARCH_QUERY_PARAM", "query")
        monkeypatch.setenv("TAX_US_SEARCH_LIMIT_PARAM", "max_results")
        query_key, limit_key = connector._resolve_param_names()
        assert query_key == "query"
        assert limit_key == "max_results"

    def test_headers(self, connector, monkeypatch):
        """Test header generation."""
        headers = connector._headers()
        assert headers["Accept"] == "application/json"
        assert headers["User-Agent"] == "Aragora/1.0"
        assert headers["Authorization"] == "Bearer test-api-key"

        monkeypatch.delenv("TAX_US_API_KEY")
        headers = connector._headers()
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_search_empty_query(self, connector):
        """Test search with empty query returns empty list."""
        results = await connector.search("")
        assert results == []

        results = await connector.search("   ")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_unconfigured(self, unconfigured_connector):
        """Test search with unconfigured connector returns empty list."""
        results = await unconfigured_connector.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_success_get(self, connector, monkeypatch):
        """Test successful GET search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "doc-123",
                    "title": "Tax Guide",
                    "summary": "Summary text",
                    "url": "http://example.com/doc",
                    "issued": "2026-01-01",
                    "type": "regulation",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("aragora.connectors.tax.generic.HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient", return_value=mock_client):
                results = await connector.search("tax deductions", limit=10)

        assert len(results) == 1
        assert results[0].title == "Tax Guide"
        assert results[0].metadata["jurisdiction"] == "US"

    @pytest.mark.asyncio
    async def test_search_success_post(self, connector, monkeypatch):
        """Test successful POST search."""
        monkeypatch.setenv("TAX_US_SEARCH_METHOD", "POST")

        mock_response = MagicMock()
        mock_response.json.return_value = {"items": [{"title": "Doc"}]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("aragora.connectors.tax.generic.HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient", return_value=mock_client):
                results = await connector.search("query")

        mock_client.post.assert_called_once()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_httpx_unavailable(self, connector, monkeypatch):
        """Test search returns empty when httpx unavailable."""
        with patch("aragora.connectors.tax.generic.HTTPX_AVAILABLE", False):
            results = await connector.search("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_api_error(self, connector):
        """Test search handles API errors gracefully."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Network error")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("aragora.connectors.tax.generic.HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient", return_value=mock_client):
                results = await connector.search("query")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_limit_capped(self, connector):
        """Test search limit is capped at 50."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("aragora.connectors.tax.generic.HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient", return_value=mock_client):
                await connector.search("query", limit=100)

        # Verify limit was capped to 50
        call_kwargs = mock_client.get.call_args
        assert call_kwargs[1]["params"]["limit"] == 50

    @pytest.mark.asyncio
    async def test_fetch_returns_none(self, connector):
        """Test fetch always returns None (not implemented)."""
        result = await connector.fetch("any-id")
        assert result is None


class TestExtractResults:
    """Tests for _extract_results helper."""

    def test_list_input(self):
        """Test extraction from list input."""
        data = [{"title": "Doc1"}, {"title": "Doc2"}]
        results = _extract_results(data)
        assert len(results) == 2

    def test_list_with_non_dict_items(self):
        """Test list filtering non-dict items."""
        data = [{"title": "Doc"}, "string", 123, None]
        results = _extract_results(data)
        assert len(results) == 1

    def test_dict_with_results_key(self):
        """Test extraction from dict with 'results' key."""
        data = {"results": [{"title": "Doc"}]}
        results = _extract_results(data)
        assert len(results) == 1

    def test_dict_with_items_key(self):
        """Test extraction from dict with 'items' key."""
        data = {"items": [{"title": "Doc"}]}
        results = _extract_results(data)
        assert len(results) == 1

    def test_dict_with_documents_key(self):
        """Test extraction from dict with 'documents' key."""
        data = {"documents": [{"title": "Doc"}]}
        results = _extract_results(data)
        assert len(results) == 1

    def test_dict_no_recognized_key(self):
        """Test extraction from dict without recognized keys."""
        data = {"unknown": [{"title": "Doc"}]}
        results = _extract_results(data)
        assert len(results) == 0

    def test_other_types(self):
        """Test extraction from other types returns empty."""
        assert _extract_results("string") == []
        assert _extract_results(123) == []
        assert _extract_results(None) == []


class TestItemToEvidence:
    """Tests for _item_to_evidence helper."""

    def test_full_item(self):
        """Test conversion with all fields present."""
        item = {
            "id": "doc-123",
            "title": "Tax Guide",
            "summary": "Summary text",
            "url": "http://example.com",
            "issued": "2026-01-01",
            "type": "regulation",
        }
        evidence = _item_to_evidence(item, "US")

        assert evidence.id == "tax:US:doc-123"
        assert evidence.title == "Tax Guide"
        assert "Tax Guide" in evidence.content
        assert "Summary text" in evidence.content
        assert evidence.url == "http://example.com"
        assert evidence.author == "US"
        assert evidence.created_at == "2026-01-01"
        assert evidence.metadata["doc_type"] == "regulation"
        assert evidence.metadata["jurisdiction"] == "US"

    def test_minimal_item(self):
        """Test conversion with minimal fields."""
        item = {"title": "Basic Doc"}
        evidence = _item_to_evidence(item, "UK")

        assert evidence.title == "Basic Doc"
        assert evidence.content == "Basic Doc"
        assert evidence.author == "UK"
        assert evidence.url is None

    def test_alternative_field_names(self):
        """Test conversion with alternative field names."""
        item = {
            "name": "Doc Name",
            "abstract": "Abstract text",
            "link": "http://link.com",
            "published": "2026-02-01",
            "document_type": "guidance",
            "doc_id": "alt-123",
        }
        evidence = _item_to_evidence(item, "AU")

        assert evidence.title == "Doc Name"
        assert "Abstract text" in evidence.content
        assert evidence.url == "http://link.com"
        assert evidence.created_at == "2026-02-01"
        assert evidence.id == "tax:AU:alt-123"

    def test_snippet_as_summary(self):
        """Test conversion using snippet as summary."""
        item = {"title": "Doc", "snippet": "Snippet text"}
        evidence = _item_to_evidence(item, "NZ")
        assert "Snippet text" in evidence.content

    def test_empty_item_generates_hash_id(self):
        """Test empty item generates hash-based ID."""
        item = {}
        evidence = _item_to_evidence(item, "JP")

        # Should generate a hash-based ID
        assert evidence.id.startswith("tax:JP:") or len(evidence.id) == 16
        assert evidence.content == "Tax guidance result"
