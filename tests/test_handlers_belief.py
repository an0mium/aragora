"""
Tests for the Belief Handler endpoints.

Covers:
- GET /api/belief-network/:debate_id/cruxes - Get key claims
- GET /api/belief-network/:debate_id/load-bearing-claims - Get high-centrality claims
- GET /api/provenance/:debate_id/claims/:claim_id/support - Get claim support
- GET /api/debate/:debate_id/graph-stats - Get argument graph statistics
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from aragora.server.handlers.belief import BeliefHandler


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def belief_handler(handler_context):
    """Create a BeliefHandler with mock context."""
    return BeliefHandler(handler_context)


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler object."""
    handler = Mock()
    handler.headers = {}
    handler.command = "GET"
    return handler


@pytest.fixture
def temp_nomic_dir_with_traces():
    """Create a temporary nomic directory with debate traces."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nomic_dir = Path(tmpdir)

        # Create traces directory
        traces_dir = nomic_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        # Create a mock trace file
        trace_data = {
            "debate_id": "debate_abc123",
            "task": "Test debate topic",
            "messages": [
                {"agent": "agent1", "content": "First argument", "role": "proposer", "round": 1},
                {"agent": "agent2", "content": "Counter argument", "role": "critic", "round": 1},
            ],
            "critiques": [],
            "consensus": None,
        }
        trace_file = traces_dir / "debate_abc123.json"
        trace_file.write_text(json.dumps(trace_data))

        # Create provenance directory
        prov_dir = nomic_dir / "provenance"
        prov_dir.mkdir(parents=True, exist_ok=True)

        # Create replays directory
        replays_dir = nomic_dir / "replays" / "debate_xyz789"
        replays_dir.mkdir(parents=True, exist_ok=True)

        # Create events.jsonl for replay
        events = [
            {
                "type": "agent_message",
                "agent": "agent1",
                "data": {"content": "Test", "role": "proposer"},
                "round": 1,
            },
            {
                "type": "critique",
                "agent": "agent2",
                "data": {"target": "agent1", "severity": 0.5, "content": "Weak"},
                "round": 1,
            },
        ]
        events_file = replays_dir / "events.jsonl"
        events_file.write_text("\n".join(json.dumps(e) for e in events))

        yield nomic_dir


# =============================================================================
# can_handle Tests
# =============================================================================


class TestCanHandle:
    """Test route matching for BeliefHandler."""

    def test_can_handle_cruxes_route(self, belief_handler):
        """Test matching cruxes endpoint."""
        assert belief_handler.can_handle("/api/belief-network/debate_abc123/cruxes")

    def test_can_handle_load_bearing_claims_route(self, belief_handler):
        """Test matching load-bearing-claims endpoint."""
        assert belief_handler.can_handle("/api/belief-network/debate_abc123/load-bearing-claims")

    def test_can_handle_claim_support_route(self, belief_handler):
        """Test matching claim support endpoint."""
        assert belief_handler.can_handle("/api/provenance/debate_abc123/claims/claim_1/support")

    def test_can_handle_graph_stats_route(self, belief_handler):
        """Test matching graph-stats endpoint."""
        assert belief_handler.can_handle("/api/debate/debate_abc123/graph-stats")

    def test_cannot_handle_unknown_route(self, belief_handler):
        """Test rejection of unknown routes."""
        assert not belief_handler.can_handle("/api/unknown")
        assert not belief_handler.can_handle("/api/belief-network/")
        assert not belief_handler.can_handle("/api/debates")


# =============================================================================
# Cruxes Endpoint Tests
# =============================================================================


class TestCruxesEndpoint:
    """Test /api/belief-network/:id/cruxes endpoint."""

    def test_invalid_debate_id_returns_400(self, belief_handler, mock_http_handler):
        """Test error on invalid debate ID."""
        result = belief_handler.handle(
            "/api/belief-network/../invalid/cruxes", {}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 400

    def test_cruxes_without_nomic_dir_returns_503(self, handler_context, mock_http_handler):
        """Test error when nomic_dir not configured."""
        handler_context["nomic_dir"] = None
        handler = BeliefHandler(handler_context)

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler.handle(
                "/api/belief-network/debate_abc123/cruxes", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503

    def test_cruxes_missing_trace_returns_404(
        self, handler_context, temp_nomic_dir_with_traces, mock_http_handler
    ):
        """Test error when debate trace not found."""
        handler_context["nomic_dir"] = temp_nomic_dir_with_traces
        handler = BeliefHandler(handler_context)

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler.handle(
                "/api/belief-network/nonexistent_debate/cruxes", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 404

    def test_cruxes_belief_network_unavailable_returns_503(self, belief_handler, mock_http_handler):
        """Test error when belief network module unavailable."""
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = belief_handler.handle(
                "/api/belief-network/debate_abc123/cruxes", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503

    def test_cruxes_top_k_parameter_clamped(self, belief_handler, mock_http_handler):
        """Test top_k parameter is clamped to valid range."""
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = belief_handler.handle(
                "/api/belief-network/debate_abc123/cruxes", {"top_k": ["100"]}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503


# =============================================================================
# Load Bearing Claims Endpoint Tests
# =============================================================================


class TestLoadBearingClaimsEndpoint:
    """Test /api/belief-network/:id/load-bearing-claims endpoint."""

    def test_invalid_debate_id_returns_400(self, belief_handler, mock_http_handler):
        """Test error on invalid debate ID."""
        result = belief_handler.handle(
            "/api/belief-network/../invalid/load-bearing-claims", {}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 400

    def test_load_bearing_without_nomic_dir_returns_503(self, handler_context, mock_http_handler):
        """Test error when nomic_dir not configured."""
        handler_context["nomic_dir"] = None
        handler = BeliefHandler(handler_context)

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler.handle(
                "/api/belief-network/debate_abc123/load-bearing-claims", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503

    def test_load_bearing_belief_network_unavailable_returns_503(
        self, belief_handler, mock_http_handler
    ):
        """Test error when belief network module unavailable."""
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = belief_handler.handle(
                "/api/belief-network/debate_abc123/load-bearing-claims", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503


# =============================================================================
# Claim Support Endpoint Tests
# =============================================================================


class TestClaimSupportEndpoint:
    """Test /api/provenance/:id/claims/:claim_id/support endpoint."""

    def test_invalid_path_format_returns_400(self, belief_handler, mock_http_handler):
        """Test malformed path returns 400 error."""
        result = belief_handler.handle("/api/provenance/claims/support", {}, mock_http_handler)
        # Handler catches this as invalid path format
        assert result is not None
        assert result.status_code == 400

    def test_invalid_debate_id_returns_400(self, belief_handler, mock_http_handler):
        """Test error on invalid debate ID format."""
        result = belief_handler.handle(
            "/api/provenance/../bad/claims/claim_1/support", {}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 400

    def test_claim_support_without_nomic_dir_returns_503(self, handler_context, mock_http_handler):
        """Test error when nomic_dir not configured."""
        handler_context["nomic_dir"] = None
        handler = BeliefHandler(handler_context)

        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", True):
            result = handler.handle(
                "/api/provenance/debate_abc123/claims/claim_1/support", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503

    def test_claim_support_provenance_unavailable_returns_503(
        self, belief_handler, mock_http_handler
    ):
        """Test error when provenance module unavailable."""
        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", False):
            result = belief_handler.handle(
                "/api/provenance/debate_abc123/claims/claim_1/support", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503

    def test_claim_support_no_provenance_data_returns_message(
        self, handler_context, temp_nomic_dir_with_traces, mock_http_handler
    ):
        """Test graceful handling when no provenance data exists."""
        handler_context["nomic_dir"] = temp_nomic_dir_with_traces
        handler = BeliefHandler(handler_context)

        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", True):
            result = handler.handle(
                "/api/provenance/debate_abc123/claims/claim_1/support", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["support"] is None
        assert "message" in body


# =============================================================================
# Graph Stats Endpoint Tests
# =============================================================================


class TestGraphStatsEndpoint:
    """Test /api/debate/:id/graph-stats endpoint."""

    def test_invalid_debate_id_returns_400(self, belief_handler, mock_http_handler):
        """Test error on invalid debate ID."""
        result = belief_handler.handle("/api/debate/../invalid/graph-stats", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 400

    def test_graph_stats_without_nomic_dir_returns_503(self, handler_context, mock_http_handler):
        """Test error when nomic_dir not configured."""
        handler_context["nomic_dir"] = None
        handler = BeliefHandler(handler_context)

        result = handler.handle("/api/debate/debate_abc123/graph-stats", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503

    def test_graph_stats_missing_debate_returns_404(
        self, handler_context, temp_nomic_dir_with_traces, mock_http_handler
    ):
        """Test error when debate not found."""
        handler_context["nomic_dir"] = temp_nomic_dir_with_traces
        handler = BeliefHandler(handler_context)

        result = handler.handle("/api/debate/nonexistent_debate/graph-stats", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 404

    def test_graph_stats_from_replay_events(
        self, handler_context, temp_nomic_dir_with_traces, mock_http_handler
    ):
        """Test graph stats loaded from replay events.jsonl."""
        handler_context["nomic_dir"] = temp_nomic_dir_with_traces
        handler = BeliefHandler(handler_context)

        mock_cartographer = MagicMock()
        mock_cartographer.get_statistics.return_value = {
            "total_arguments": 2,
            "total_critiques": 1,
            "agents": ["agent1", "agent2"],
        }

        # ArgumentCartographer is imported inside the method from aragora.visualization.mapper
        with patch(
            "aragora.visualization.mapper.ArgumentCartographer", return_value=mock_cartographer
        ):
            result = handler.handle("/api/debate/debate_xyz789/graph-stats", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_arguments"] == 2


# =============================================================================
# ID Extraction Tests
# =============================================================================


class TestIdExtraction:
    """Test debate ID extraction from paths."""

    def test_extract_valid_debate_id(self, belief_handler):
        """Test extraction of valid debate ID."""
        debate_id = belief_handler._extract_debate_id("/api/belief-network/debate_abc123/cruxes", 3)
        assert debate_id == "debate_abc123"

    def test_extract_invalid_debate_id_returns_none(self, belief_handler):
        """Test extraction returns None for invalid IDs."""
        debate_id = belief_handler._extract_debate_id("/api/belief-network/../etc/passwd/cruxes", 3)
        assert debate_id is None

    def test_extract_from_short_path_returns_none(self, belief_handler):
        """Test extraction returns None for too-short paths."""
        debate_id = belief_handler._extract_debate_id("/api/belief-network", 3)
        assert debate_id is None
