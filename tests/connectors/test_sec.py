"""Tests for SEC EDGAR connector."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from aragora.connectors.sec import SECConnector, FORM_TYPES
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


# Sample SEC API responses
SAMPLE_COMPANY_TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
    "2": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon.com, Inc."},
}

SAMPLE_SUBMISSIONS = {
    "cik": "0000320193",
    "name": "Apple Inc.",
    "tickers": ["AAPL"],
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-24-000001",
                "0000320193-23-000095",
                "0000320193-23-000080",
            ],
            "form": ["10-K", "10-Q", "8-K"],
            "filingDate": ["2024-11-01", "2024-08-02", "2024-07-15"],
            "primaryDocument": ["aapl-20240928.htm", "aapl-20240629.htm", "aapl-8k.htm"],
            "primaryDocDescription": [
                "Annual Report",
                "Quarterly Report",
                "Current Report",
            ],
        }
    },
}


class TestSECConnector:
    """Tests for SECConnector."""

    @pytest.fixture
    def connector(self):
        """Create an SEC connector for testing."""
        # Clear CIK cache before each test
        SECConnector._cik_cache.clear()
        return SECConnector(rate_limit_delay=0.0)

    def test_connector_properties(self, connector):
        """Test connector property methods."""
        assert connector.source_type == SourceType.DOCUMENT
        assert connector.name == "SEC EDGAR"
        assert connector.is_available

    def test_form_types(self, connector):
        """Test form types dictionary."""
        form_types = connector.get_form_types()

        assert "10-K" in form_types
        assert "10-Q" in form_types
        assert "8-K" in form_types
        assert form_types["10-K"] == "Annual report"

    @pytest.mark.asyncio
    async def test_resolve_cik_from_ticker(self, connector):
        """Test CIK resolution from ticker."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_COMPANY_TICKERS
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            cik = await connector._resolve_to_cik("AAPL")

            assert cik == "0000320193"

    @pytest.mark.asyncio
    async def test_resolve_cik_from_cik(self, connector):
        """Test CIK resolution when already a CIK."""
        # No HTTP call needed for pure numeric input
        cik = await connector._resolve_to_cik("320193")
        assert cik == "0000320193"

    @pytest.mark.asyncio
    async def test_search_by_ticker(self, connector):
        """Test search with ticker symbol."""
        with patch("httpx.AsyncClient") as mock_client:
            # Mock both CIK lookup and submissions fetch
            mock_tickers_response = MagicMock()
            mock_tickers_response.json.return_value = SAMPLE_COMPANY_TICKERS
            mock_tickers_response.raise_for_status = MagicMock()

            mock_submissions_response = MagicMock()
            mock_submissions_response.json.return_value = SAMPLE_SUBMISSIONS
            mock_submissions_response.raise_for_status = MagicMock()
            mock_submissions_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(
                side_effect=[mock_tickers_response, mock_submissions_response]
            )
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("AAPL", limit=5)

            assert len(results) == 3
            assert all(isinstance(r, Evidence) for r in results)

            # Check first result is 10-K
            first = results[0]
            assert "10-K" in first.title
            assert first.metadata["form_type"] == "10-K"
            assert first.metadata["ticker"] == "AAPL"
            assert first.authority == 0.95  # Regulatory filings

    @pytest.mark.asyncio
    async def test_search_with_form_filter(self, connector):
        """Test search with form type filter."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_tickers_response = MagicMock()
            mock_tickers_response.json.return_value = SAMPLE_COMPANY_TICKERS
            mock_tickers_response.raise_for_status = MagicMock()

            mock_submissions_response = MagicMock()
            mock_submissions_response.json.return_value = SAMPLE_SUBMISSIONS
            mock_submissions_response.raise_for_status = MagicMock()
            mock_submissions_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(
                side_effect=[mock_tickers_response, mock_submissions_response]
            )
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("AAPL", form_type="10-K", limit=5)

            assert len(results) == 1
            assert results[0].metadata["form_type"] == "10-K"

    @pytest.mark.asyncio
    async def test_get_recent_filings(self, connector):
        """Test get_recent_filings helper."""
        with patch.object(connector, "search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            await connector.get_recent_filings("AAPL", form_type="10-Q", limit=3)

            mock_search.assert_called_once_with("AAPL", limit=3, form_type="10-Q")

    @pytest.mark.asyncio
    async def test_get_8k_filings(self, connector):
        """Test get_8k_filings helper."""
        with patch.object(connector, "search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            await connector.get_8k_filings("MSFT", days=30, limit=10)

            # Verify it was called with 8-K form type
            call_args = mock_search.call_args
            assert call_args[1]["form_type"] == "8-K"
            assert "date_from" in call_args[1]

    @pytest.mark.asyncio
    async def test_fetch_cached(self, connector):
        """Test fetch returns cached evidence."""
        cached = Evidence(
            id="sec:cached-accession",
            source_type=SourceType.DOCUMENT,
            source_id="0000320193-24-000001",
            content="Cached filing content",
            title="AAPL: 10-K - 2024-11-01",
        )
        connector._cache_put("sec:cached-accession", cached)

        result = await connector.fetch("sec:cached-accession")

        assert result is not None
        assert result.title == "AAPL: 10-K - 2024-11-01"

    @pytest.mark.asyncio
    async def test_fetch_uncached_returns_none(self, connector):
        """Test fetch returns None for uncached evidence."""
        result = await connector.fetch("sec:nonexistent")
        assert result is None

    def test_create_filing_evidence(self, connector):
        """Test filing evidence creation."""
        evidence = connector._create_filing_evidence(
            company_name="Apple Inc.",
            ticker="AAPL",
            cik="0000320193",
            form="10-K",
            filing_date="2024-11-01",
            accession="0000320193-24-000001",
            primary_doc="aapl-20240928.htm",
            description="Annual Report",
        )

        assert evidence is not None
        assert evidence.id == "sec:0000320193-24-000001"
        assert "AAPL" in evidence.title
        assert "10-K" in evidence.title
        assert evidence.metadata["form_type"] == "10-K"
        assert evidence.authority == 0.95
        assert "sec.gov" in evidence.url

    def test_create_filing_evidence_missing_data(self, connector):
        """Test filing evidence creation with missing data."""
        evidence = connector._create_filing_evidence(
            company_name="Test Corp",
            ticker="",
            cik="123",
            form="",  # Missing form
            filing_date="",  # Missing date
            accession="123",
            primary_doc="",
            description="",
        )

        assert evidence is None

    @pytest.mark.asyncio
    async def test_cik_caching(self, connector):
        """Test that CIK lookups are cached."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_COMPANY_TICKERS
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            # First lookup
            cik1 = await connector._resolve_to_cik("AAPL")

            # Second lookup should use cache
            cik2 = await connector._resolve_to_cik("AAPL")

            assert cik1 == cik2
            # Should only have made one HTTP call
            assert mock_client_instance.get.call_count == 1

    @pytest.mark.asyncio
    async def test_search_not_found(self, connector):
        """Test search when company not found."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_tickers_response = MagicMock()
            mock_tickers_response.json.return_value = {}  # No companies
            mock_tickers_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_tickers_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("NONEXISTENT")

            # Should return empty list (falls through to fulltext search)
            assert isinstance(results, list)


class TestFormTypes:
    """Tests for FORM_TYPES constant."""

    def test_common_forms_present(self):
        """Test that common form types are defined."""
        assert "10-K" in FORM_TYPES
        assert "10-Q" in FORM_TYPES
        assert "8-K" in FORM_TYPES
        assert "S-1" in FORM_TYPES
        assert "DEF 14A" in FORM_TYPES

    def test_form_descriptions(self):
        """Test form descriptions are meaningful."""
        assert "Annual" in FORM_TYPES["10-K"]
        assert "Quarterly" in FORM_TYPES["10-Q"]
        assert "Current" in FORM_TYPES["8-K"] or "material" in FORM_TYPES["8-K"]
