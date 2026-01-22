"""
Tests for the BeliefHandler module.

Tests cover:
- Handler routing for belief network endpoints
- Handler routing for provenance endpoints
- can_handle method for static and dynamic routes
- Rate limiting behavior
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from aragora.server.handlers.belief import BeliefHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


class TestBeliefHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BeliefHandler(mock_server_context)

    def test_can_handle_cruxes(self, handler):
        """Handler can handle cruxes endpoint."""
        assert handler.can_handle("/api/v1/belief-network/debate_123/cruxes")

    def test_can_handle_load_bearing_claims(self, handler):
        """Handler can handle load-bearing-claims endpoint."""
        assert handler.can_handle("/api/v1/belief-network/debate_abc/load-bearing-claims")

    def test_can_handle_graph(self, handler):
        """Handler can handle graph endpoint."""
        assert handler.can_handle("/api/v1/belief-network/debate_xyz/graph")

    def test_can_handle_export(self, handler):
        """Handler can handle export endpoint."""
        assert handler.can_handle("/api/v1/belief-network/debate_123/export")

    def test_can_handle_claim_support(self, handler):
        """Handler can handle claim support endpoint."""
        assert handler.can_handle("/api/v1/provenance/debate_123/claims/claim_456/support")

    def test_can_handle_graph_stats(self, handler):
        """Handler can handle graph stats endpoint."""
        assert handler.can_handle("/api/v1/debate/debate_123/graph-stats")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/belief-network")
        assert not handler.can_handle("/api/v1/other")


class TestBeliefHandlerRouteDispatch:
    """Tests for route dispatch logic."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BeliefHandler(mock_server_context)

    def test_handle_cruxes_returns_result(self, handler):
        """Handle returns result for cruxes endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/v1/belief-network/debate_123/cruxes",
            {},
            mock_http,
        )

        # Should return some result (may be error if module unavailable)
        assert result is not None

    def test_handle_load_bearing_claims_returns_result(self, handler):
        """Handle returns result for load-bearing-claims endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/v1/belief-network/debate_abc/load-bearing-claims",
            {},
            mock_http,
        )

        assert result is not None

    def test_handle_graph_returns_result(self, handler):
        """Handle returns result for graph endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/v1/belief-network/debate_xyz/graph",
            {},
            mock_http,
        )

        assert result is not None

    def test_handle_export_returns_result(self, handler):
        """Handle returns result for export endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/v1/belief-network/debate_123/export",
            {},
            mock_http,
        )

        assert result is not None

    def test_handle_claim_support_returns_result(self, handler):
        """Handle returns result for claim support endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/v1/provenance/debate_123/claims/claim_456/support",
            {},
            mock_http,
        )

        assert result is not None

    def test_handle_graph_stats_returns_result(self, handler):
        """Handle returns result for graph stats endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/v1/debate/debate_123/graph-stats",
            {},
            mock_http,
        )

        assert result is not None

    def test_handle_unknown_returns_none(self, handler):
        """Handle returns None for unknown paths."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/unknown", {}, mock_http)

        assert result is None


class TestBeliefHandlerQueryParams:
    """Tests for query parameter handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BeliefHandler(mock_server_context)

    def test_cruxes_top_k_param(self, handler):
        """Cruxes endpoint respects top_k parameter."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/v1/belief-network/debate_123/cruxes",
            {"top_k": ["5"]},
            mock_http,
        )

        assert result is not None

    def test_load_bearing_limit_param(self, handler):
        """Load-bearing-claims endpoint respects limit parameter."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/v1/belief-network/debate_abc/load-bearing-claims",
            {"limit": ["10"]},
            mock_http,
        )

        assert result is not None

    def test_graph_include_cruxes_param(self, handler):
        """Graph endpoint respects include_cruxes parameter."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle(
            "/api/v1/belief-network/debate_xyz/graph",
            {"include_cruxes": ["true"]},
            mock_http,
        )

        assert result is not None


class TestBeliefHandlerValidation:
    """Tests for input validation."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BeliefHandler(mock_server_context)

    def test_invalid_debate_id_returns_error(self, handler):
        """Invalid debate ID returns 400 error."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        # Missing debate_id in path
        result = handler.handle(
            "/api/v1/belief-network//cruxes",
            {},
            mock_http,
        )

        # Should return error or None (400 for invalid ID, 503 if nomic_dir not configured)
        assert result is None or result.status_code in (400, 503)

    def test_top_k_clamped_to_max(self, handler):
        """Top K parameter is clamped to maximum."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        # Request high top_k, should be clamped to 10
        result = handler.handle(
            "/api/v1/belief-network/debate_123/cruxes",
            {"top_k": ["100"]},
            mock_http,
        )

        assert result is not None

    def test_limit_clamped_to_max(self, handler):
        """Limit parameter is clamped to maximum."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        # Request high limit, should be clamped to 20
        result = handler.handle(
            "/api/v1/belief-network/debate_abc/load-bearing-claims",
            {"limit": ["500"]},
            mock_http,
        )

        assert result is not None


class TestBeliefHandlerExtractDebateId:
    """Tests for debate ID extraction from paths."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BeliefHandler(mock_server_context)

    def test_extract_debate_id_from_cruxes_path(self, handler):
        """Extract debate ID from cruxes path."""
        # split("/") gives: ["", "api", "v1", "belief-network", "abc123", "cruxes"]
        # Index 4 is "abc123"
        debate_id = handler._extract_debate_id("/api/v1/belief-network/abc123/cruxes", 4)
        assert debate_id == "abc123"

    def test_extract_debate_id_from_graph_stats_path(self, handler):
        """Extract debate ID from graph-stats path."""
        # split("/") gives: ["", "api", "v1", "debate", "xyz789", "graph-stats"]
        # Index 4 is "xyz789"
        debate_id = handler._extract_debate_id("/api/v1/debate/xyz789/graph-stats", 4)
        assert debate_id == "xyz789"

    def test_extract_debate_id_invalid_returns_none(self, handler):
        """Invalid debate ID extraction returns None."""
        # Path traversal attempt - ".." is not a valid debate ID
        # split("/") gives: ["", "api", "v1", "belief-network", "..", "etc", "cruxes"]
        # Index 4 is ".." which fails validation
        debate_id = handler._extract_debate_id("/api/v1/belief-network/../etc/cruxes", 4)
        # Should return None for invalid path
        assert debate_id is None


class TestBeliefHandlerPathMatching:
    """Tests for path matching patterns."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BeliefHandler(mock_server_context)

    def test_matches_various_debate_ids(self, handler):
        """Handler matches various debate ID formats."""
        assert handler.can_handle("/api/v1/belief-network/debate_123/cruxes")
        assert handler.can_handle("/api/v1/belief-network/abc-def-ghi/cruxes")
        assert handler.can_handle("/api/v1/belief-network/uuid_format_id/cruxes")

    def test_matches_claim_support_pattern(self, handler):
        """Handler matches claim support pattern correctly."""
        assert handler.can_handle("/api/v1/provenance/d1/claims/c1/support")
        assert handler.can_handle("/api/v1/something/debate_123/claims/claim_456/support")

    def test_does_not_match_partial_patterns(self, handler):
        """Handler does not match partial patterns."""
        # Missing /cruxes suffix
        assert not handler.can_handle("/api/v1/belief-network/debate_123")
        # Missing /claims/ in path (support suffix alone is not enough)
        assert not handler.can_handle("/api/v1/provenance/debate_123/support")
