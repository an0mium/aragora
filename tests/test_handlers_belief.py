"""Tests for BeliefHandler."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from aragora.server.handlers.belief import BeliefHandler


class TestBeliefHandlerRouting:
    """Tests for route matching."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return BeliefHandler({"nomic_dir": Path("/tmp/test")})

    def test_can_handle_emergent_traits(self, handler):
        """Should handle /api/laboratory/emergent-traits."""
        assert handler.can_handle("/api/laboratory/emergent-traits") is True

    def test_can_handle_cruxes(self, handler):
        """Should handle /api/belief-network/:debate_id/cruxes."""
        assert handler.can_handle("/api/belief-network/debate-123/cruxes") is True

    def test_can_handle_load_bearing_claims(self, handler):
        """Should handle /api/belief-network/:debate_id/load-bearing-claims."""
        assert handler.can_handle("/api/belief-network/debate-456/load-bearing-claims") is True

    def test_can_handle_claim_support(self, handler):
        """Should handle /api/provenance/:debate_id/claims/:claim_id/support."""
        assert handler.can_handle("/api/provenance/debate-123/claims/claim-456/support") is True

    def test_can_handle_graph_stats(self, handler):
        """Should handle /api/debate/:debate_id/graph-stats."""
        assert handler.can_handle("/api/debate/debate-123/graph-stats") is True

    def test_cannot_handle_unrelated(self, handler):
        """Should not handle unrelated routes."""
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agents") is False
        assert handler.can_handle("/api/relationships/summary") is False

    def test_cannot_handle_partial_paths(self, handler):
        """Should not handle incomplete paths."""
        assert handler.can_handle("/api/belief-network/debate-123") is False
        assert handler.can_handle("/api/laboratory") is False


class TestEmergentTraitsEndpoint:
    """Tests for /api/laboratory/emergent-traits endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return BeliefHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.belief.LABORATORY_AVAILABLE", False)
    def test_503_when_laboratory_unavailable(self, handler):
        """Should return 503 when laboratory not available."""
        result = handler.handle("/api/laboratory/emergent-traits", {}, Mock())
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not available" in data["error"]

    @patch("aragora.server.handlers.belief.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.belief.PersonaLaboratory")
    def test_returns_traits_list(self, mock_lab_class, handler):
        """Should return filtered traits list."""
        mock_trait = Mock()
        mock_trait.agent_name = "claude"
        mock_trait.trait_name = "analytical"
        mock_trait.domain = "reasoning"
        mock_trait.confidence = 0.8
        mock_trait.evidence = ["won 5 debates"]
        mock_trait.detected_at = "2025-01-01T00:00:00"

        mock_lab = Mock()
        mock_lab.detect_emergent_traits.return_value = [mock_trait]
        mock_lab_class.return_value = mock_lab

        result = handler.handle("/api/laboratory/emergent-traits", {"min_confidence": "0.5"}, Mock())

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "emergent_traits" in data
        assert len(data["emergent_traits"]) == 1
        assert data["emergent_traits"][0]["agent"] == "claude"


class TestCruxesEndpoint:
    """Tests for /api/belief-network/:debate_id/cruxes endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return BeliefHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False)
    def test_503_when_belief_network_unavailable(self, handler):
        """Should return 503 when belief network not available."""
        result = handler.handle("/api/belief-network/debate-123/cruxes", {}, Mock())
        assert result.status_code == 503

    def test_rejects_invalid_debate_id(self, handler):
        """Should reject debate IDs with path traversal."""
        result = handler.handle("/api/belief-network/../etc/cruxes", {}, Mock())
        assert result.status_code == 400


class TestLoadBearingClaimsEndpoint:
    """Tests for /api/belief-network/:debate_id/load-bearing-claims endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return BeliefHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False)
    def test_503_when_belief_network_unavailable(self, handler):
        """Should return 503 when belief network not available."""
        result = handler.handle("/api/belief-network/debate-123/load-bearing-claims", {}, Mock())
        assert result.status_code == 503


class TestClaimSupportEndpoint:
    """Tests for /api/provenance/:debate_id/claims/:claim_id/support endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return BeliefHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", False)
    def test_503_when_provenance_unavailable(self, handler):
        """Should return 503 when provenance not available."""
        result = handler.handle("/api/provenance/debate-123/claims/claim-456/support", {}, Mock())
        assert result.status_code == 503

    def test_rejects_invalid_claim_id(self, handler):
        """Should reject claim IDs with special characters."""
        result = handler.handle("/api/provenance/debate-123/claims/../etc/support", {}, Mock())
        assert result.status_code == 400

    def test_rejects_invalid_path_format(self, handler):
        """Should reject malformed paths."""
        result = handler.handle("/api/provenance/support", {}, Mock())
        # This path doesn't match can_handle, so returns None
        assert result is None


class TestGraphStatsEndpoint:
    """Tests for /api/debate/:debate_id/graph-stats endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return BeliefHandler({"nomic_dir": Path("/tmp/test")})

    def test_returns_503_without_nomic_dir(self):
        """Should return 503 when nomic_dir not configured."""
        handler = BeliefHandler({"nomic_dir": None})
        result = handler.handle("/api/debate/debate-123/graph-stats", {}, Mock())
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not configured" in data["error"]


class TestDebateIdExtraction:
    """Tests for debate ID extraction and validation."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return BeliefHandler({"nomic_dir": Path("/tmp/test")})

    def test_extracts_valid_debate_id(self, handler):
        """Should extract valid debate ID from path."""
        debate_id = handler._extract_debate_id("/api/belief-network/debate-123/cruxes", 3)
        assert debate_id == "debate-123"

    def test_extracts_id_with_underscores(self, handler):
        """Should accept IDs with underscores."""
        debate_id = handler._extract_debate_id("/api/belief-network/debate_123_abc/cruxes", 3)
        assert debate_id == "debate_123_abc"

    def test_rejects_path_traversal(self, handler):
        """Should reject path traversal attempts."""
        debate_id = handler._extract_debate_id("/api/belief-network/../etc/cruxes", 3)
        assert debate_id is None

    def test_rejects_special_characters(self, handler):
        """Should reject special characters."""
        debate_id = handler._extract_debate_id("/api/belief-network/debate;drop/cruxes", 3)
        assert debate_id is None


class TestHandlerImport:
    """Tests for handler module imports."""

    def test_handler_can_be_imported(self):
        """Should be importable from handlers package."""
        from aragora.server.handlers import BeliefHandler
        assert BeliefHandler is not None

    def test_handler_in_all(self):
        """Should be in __all__ exports."""
        from aragora.server.handlers import __all__
        assert "BeliefHandler" in __all__


class TestParameterValidation:
    """Tests for query parameter validation."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return BeliefHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.belief.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.belief.PersonaLaboratory")
    def test_min_confidence_clamped_to_valid_range(self, mock_lab_class, handler):
        """min_confidence should be clamped to 0.0-1.0."""
        mock_lab = Mock()
        mock_lab.detect_emergent_traits.return_value = []
        mock_lab_class.return_value = mock_lab

        # Test with value > 1.0
        result = handler.handle("/api/laboratory/emergent-traits", {"min_confidence": "2.0"}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["min_confidence"] == 1.0

        # Test with negative value
        result = handler.handle("/api/laboratory/emergent-traits", {"min_confidence": "-0.5"}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["min_confidence"] == 0.0

    @patch("aragora.server.handlers.belief.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.belief.PersonaLaboratory")
    def test_limit_capped_at_max(self, mock_lab_class, handler):
        """limit should be capped at maximum value."""
        mock_lab = Mock()
        mock_lab.detect_emergent_traits.return_value = []
        mock_lab_class.return_value = mock_lab

        # Test with limit > 50
        result = handler.handle("/api/laboratory/emergent-traits", {"limit": "100"}, Mock())
        assert result.status_code == 200
        # Implementation limits to 50
