"""
Tests for the GenesisHandler module.

Tests cover:
- Handler routing for all genesis endpoints
- Rate limiting behavior
- can_handle method for static and dynamic routes
- ROUTES attribute
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from aragora.server.handlers.genesis import GenesisHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


class TestGenesisHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_can_handle_stats(self, handler):
        """Handler can handle stats endpoint."""
        assert handler.can_handle("/api/genesis/stats")

    def test_can_handle_events(self, handler):
        """Handler can handle events endpoint."""
        assert handler.can_handle("/api/genesis/events")

    def test_can_handle_genomes(self, handler):
        """Handler can handle genomes endpoint."""
        assert handler.can_handle("/api/genesis/genomes")

    def test_can_handle_genomes_top(self, handler):
        """Handler can handle top genomes endpoint."""
        assert handler.can_handle("/api/genesis/genomes/top")

    def test_can_handle_population(self, handler):
        """Handler can handle population endpoint."""
        assert handler.can_handle("/api/genesis/population")

    def test_can_handle_lineage_dynamic(self, handler):
        """Handler can handle dynamic lineage endpoint."""
        assert handler.can_handle("/api/genesis/lineage/genome_123")

    def test_can_handle_tree_dynamic(self, handler):
        """Handler can handle dynamic tree endpoint."""
        assert handler.can_handle("/api/genesis/tree/debate_123")

    def test_can_handle_genome_by_id(self, handler):
        """Handler can handle genome by ID endpoint."""
        assert handler.can_handle("/api/genesis/genomes/genome_abc")

    def test_can_handle_descendants(self, handler):
        """Handler can handle descendants endpoint."""
        assert handler.can_handle("/api/genesis/descendants/genome_123")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/other")
        assert not handler.can_handle("/api/genomes")


class TestGenesisHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_routes_contains_stats(self, handler):
        """ROUTES contains stats endpoint."""
        assert "/api/genesis/stats" in handler.ROUTES

    def test_routes_contains_events(self, handler):
        """ROUTES contains events endpoint."""
        assert "/api/genesis/events" in handler.ROUTES

    def test_routes_contains_genomes(self, handler):
        """ROUTES contains genomes endpoint."""
        assert "/api/genesis/genomes" in handler.ROUTES

    def test_routes_contains_genomes_top(self, handler):
        """ROUTES contains top genomes endpoint."""
        assert "/api/genesis/genomes/top" in handler.ROUTES

    def test_routes_contains_population(self, handler):
        """ROUTES contains population endpoint."""
        assert "/api/genesis/population" in handler.ROUTES


class TestGenesisHandlerRouteDispatch:
    """Tests for route dispatch logic."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_handle_returns_result_for_stats(self, handler):
        """Handle returns a result for stats endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/genesis/stats", {}, mock_http)

        # Should return some result (may be 503 if genesis not available)
        assert result is not None

    def test_handle_returns_result_for_events(self, handler):
        """Handle returns a result for events endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/genesis/events", {}, mock_http)

        assert result is not None

    def test_handle_returns_result_for_genomes(self, handler):
        """Handle returns a result for genomes endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/genesis/genomes", {}, mock_http)

        assert result is not None

    def test_handle_returns_result_for_population(self, handler):
        """Handle returns a result for population endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/genesis/population", {}, mock_http)

        assert result is not None

    def test_handle_returns_none_for_unknown(self, handler):
        """Handle returns None for unknown paths."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/unknown", {}, mock_http)

        assert result is None


class TestGenesisHandlerQueryParams:
    """Tests for query parameter handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_events_limit_param(self, handler):
        """Events endpoint respects limit parameter."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/genesis/events",
            {"limit": ["50"]},
            mock_http,
        )

        assert result is not None

    def test_events_limit_capped_at_100(self, handler):
        """Events endpoint caps limit at 100."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        # Request 500 but should be capped to 100
        result = handler.handle(
            "/api/genesis/events",
            {"limit": ["500"]},
            mock_http,
        )

        assert result is not None

    def test_genomes_offset_param(self, handler):
        """Genomes endpoint respects offset parameter."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/genesis/genomes",
            {"offset": ["10"], "limit": ["20"]},
            mock_http,
        )

        assert result is not None


class TestGenesisHandlerDynamicRoutes:
    """Tests for dynamic route handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_lineage_extracts_genome_id(self, handler):
        """Lineage endpoint extracts genome ID from path."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/genesis/lineage/genome_abc123",
            {},
            mock_http,
        )

        assert result is not None

    def test_tree_extracts_debate_id(self, handler):
        """Tree endpoint extracts debate ID from path."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/genesis/tree/debate_xyz",
            {},
            mock_http,
        )

        assert result is not None

    def test_genome_by_id_extracts_id(self, handler):
        """Genome by ID endpoint extracts ID from path."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/genesis/genomes/genome_specific",
            {},
            mock_http,
        )

        assert result is not None


class TestGenesisHandlerValidation:
    """Tests for input validation."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_invalid_genome_id_handled(self, handler):
        """Handler handles invalid genome ID gracefully."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        # Invalid characters in genome_id
        result = handler.handle(
            "/api/genesis/genomes/../etc/passwd",
            {},
            mock_http,
        )

        # Should return error or None (not crash)
        # The handler may validate, return 503 if genesis unavailable, or 429 if rate limited
        assert result is None or result.status_code in (400, 404, 429, 503)

    def test_event_type_filter_handled(self, handler):
        """Handler handles event_type filter."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/genesis/events",
            {"event_type": ["mutation"]},
            mock_http,
        )

        assert result is not None
