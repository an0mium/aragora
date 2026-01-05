"""
Tests for modular HTTP endpoint handlers.

Tests the new handler architecture including:
- ConsensusHandler
- BeliefHandler
- Handler routing and response formatting
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from aragora.server.handlers import (
    ConsensusHandler,
    BeliefHandler,
    HandlerResult,
    json_response,
    error_response,
)
from aragora.server.handlers.base import (
    get_int_param,
    get_float_param,
    get_bool_param,
    BoundedTTLCache,
)


# ============================================================================
# Base Handler Utilities
# ============================================================================

class TestGetIntParam:
    """Tests for get_int_param utility."""

    def test_returns_default_when_missing(self):
        """Test default is returned when key missing."""
        assert get_int_param({}, 'limit', 10) == 10

    def test_parses_valid_integer(self):
        """Test parsing valid integer."""
        assert get_int_param({'limit': '25'}, 'limit', 10) == 25

    def test_parses_negative_value(self):
        """Test parsing negative integer."""
        assert get_int_param({'limit': '-5'}, 'limit', 10) == -5

    def test_handles_invalid_string(self):
        """Test handling invalid string."""
        assert get_int_param({'limit': 'abc'}, 'limit', 10) == 10

    def test_handles_none(self):
        """Test handling None value."""
        assert get_int_param({'limit': None}, 'limit', 10) == 10


class TestGetFloatParam:
    """Tests for get_float_param utility."""

    def test_returns_default_when_missing(self):
        """Test default is returned when key missing."""
        assert get_float_param({}, 'threshold', 0.5) == 0.5

    def test_parses_valid_float(self):
        """Test parsing valid float."""
        assert get_float_param({'threshold': '0.75'}, 'threshold', 0.5) == 0.75

    def test_handles_invalid_string(self):
        """Test handling invalid string."""
        assert get_float_param({'threshold': 'abc'}, 'threshold', 0.5) == 0.5


class TestGetBoolParam:
    """Tests for get_bool_param utility."""

    def test_true_values(self):
        """Test various true representations."""
        assert get_bool_param({'flag': 'true'}, 'flag', False) is True
        assert get_bool_param({'flag': '1'}, 'flag', False) is True
        assert get_bool_param({'flag': 'yes'}, 'flag', False) is True
        assert get_bool_param({'flag': 'on'}, 'flag', False) is True

    def test_false_values(self):
        """Test various false representations."""
        assert get_bool_param({'flag': 'false'}, 'flag', True) is False
        assert get_bool_param({'flag': '0'}, 'flag', True) is False
        assert get_bool_param({'flag': 'no'}, 'flag', True) is False


class TestBoundedTTLCache:
    """Tests for BoundedTTLCache."""

    def test_basic_get_set(self):
        """Test basic cache operations."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        hit, value = cache.get("key1", ttl_seconds=60)
        assert hit is True
        assert value == "value1"

    def test_cache_miss(self):
        """Test cache miss."""
        cache = BoundedTTLCache(max_entries=10)
        hit, value = cache.get("nonexistent", ttl_seconds=60)
        assert hit is False
        assert value is None

    def test_eviction(self):
        """Test LRU eviction when full."""
        cache = BoundedTTLCache(max_entries=3, evict_percent=0.5)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Should trigger eviction of oldest
        cache.set("d", 4)
        # 'a' should be evicted
        hit, _ = cache.get("a", ttl_seconds=60)
        assert hit is False

    def test_clear(self):
        """Test cache clear."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        count = cache.clear()
        assert count == 2
        assert len(cache) == 0

    def test_clear_with_prefix(self):
        """Test clearing with prefix filter."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("user:1", "a")
        cache.set("user:2", "b")
        cache.set("post:1", "c")
        count = cache.clear("user:")
        assert count == 2
        assert len(cache) == 1

    def test_stats(self):
        """Test cache statistics."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        cache.get("key1", ttl_seconds=60)  # Hit
        cache.get("nonexistent", ttl_seconds=60)  # Miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestJsonResponse:
    """Tests for json_response utility."""

    def test_basic_response(self):
        """Test basic JSON response creation."""
        result = json_response({"message": "hello"})
        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert b'"message"' in result.body

    def test_custom_status(self):
        """Test response with custom status."""
        result = json_response({"error": "not found"}, status=404)
        assert result.status_code == 404

    def test_serializes_complex_types(self):
        """Test serialization of complex types."""
        from datetime import datetime
        result = json_response({"time": datetime(2024, 1, 1)})
        assert b"2024-01-01" in result.body


class TestErrorResponse:
    """Tests for error_response utility."""

    def test_error_format(self):
        """Test error response format."""
        result = error_response("Something went wrong", 500)
        assert result.status_code == 500
        body = json.loads(result.body)
        assert body["error"] == "Something went wrong"


# ============================================================================
# ConsensusHandler Tests
# ============================================================================

class TestConsensusHandler:
    """Tests for ConsensusHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        ctx = {
            "storage": Mock(),
            "nomic_dir": Path("/tmp/test"),
        }
        return ConsensusHandler(ctx)

    def test_can_handle_similar(self, handler):
        """Test can_handle for similar endpoint."""
        assert handler.can_handle("/api/consensus/similar") is True

    def test_can_handle_settled(self, handler):
        """Test can_handle for settled endpoint."""
        assert handler.can_handle("/api/consensus/settled") is True

    def test_can_handle_stats(self, handler):
        """Test can_handle for stats endpoint."""
        assert handler.can_handle("/api/consensus/stats") is True

    def test_can_handle_dissents(self, handler):
        """Test can_handle for dissents endpoint."""
        assert handler.can_handle("/api/consensus/dissents") is True

    def test_can_handle_contrarian(self, handler):
        """Test can_handle for contrarian-views endpoint."""
        assert handler.can_handle("/api/consensus/contrarian-views") is True

    def test_can_handle_risk_warnings(self, handler):
        """Test can_handle for risk-warnings endpoint."""
        assert handler.can_handle("/api/consensus/risk-warnings") is True

    def test_can_handle_domain(self, handler):
        """Test can_handle for domain-specific endpoint."""
        assert handler.can_handle("/api/consensus/domain/security") is True
        assert handler.can_handle("/api/consensus/domain/performance") is True

    def test_cannot_handle_unrelated(self, handler):
        """Test rejects unrelated endpoints."""
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agents") is False

    def test_similar_requires_topic(self, handler):
        """Test similar endpoint requires topic parameter."""
        result = handler.handle("/api/consensus/similar", {}, Mock())
        assert result.status_code == 400
        assert b"Topic required" in result.body

    def test_similar_validates_topic_length(self, handler):
        """Test topic length validation."""
        long_topic = "x" * 600
        result = handler.handle("/api/consensus/similar", {"topic": long_topic}, Mock())
        assert result.status_code == 400

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", False)
    def test_returns_503_when_unavailable(self, handler):
        """Test returns 503 when consensus memory unavailable."""
        result = handler.handle("/api/consensus/stats", {}, Mock())
        assert result.status_code == 503


# ============================================================================
# BeliefHandler Tests
# ============================================================================

class TestBeliefHandler:
    """Tests for BeliefHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        ctx = {
            "nomic_dir": Path("/tmp/test"),
            "persona_manager": Mock(),
        }
        return BeliefHandler(ctx)

    def test_can_handle_emergent_traits(self, handler):
        """Test can_handle for emergent-traits endpoint."""
        assert handler.can_handle("/api/laboratory/emergent-traits") is True

    def test_can_handle_cruxes(self, handler):
        """Test can_handle for cruxes endpoint."""
        assert handler.can_handle("/api/belief-network/debate-123/cruxes") is True

    def test_can_handle_load_bearing(self, handler):
        """Test can_handle for load-bearing-claims endpoint."""
        assert handler.can_handle("/api/belief-network/debate-456/load-bearing-claims") is True

    def test_can_handle_claim_support(self, handler):
        """Test can_handle for claim support endpoint."""
        assert handler.can_handle("/api/provenance/debate-123/claims/claim-456/support") is True

    def test_can_handle_graph_stats(self, handler):
        """Test can_handle for graph-stats endpoint."""
        assert handler.can_handle("/api/debate/debate-789/graph-stats") is True

    def test_cannot_handle_unrelated(self, handler):
        """Test rejects unrelated endpoints."""
        assert handler.can_handle("/api/consensus/stats") is False
        assert handler.can_handle("/api/debates") is False

    @patch("aragora.server.handlers.belief.LABORATORY_AVAILABLE", False)
    def test_emergent_traits_503_when_unavailable(self, handler):
        """Test returns 503 when laboratory unavailable."""
        result = handler.handle("/api/laboratory/emergent-traits", {}, Mock())
        assert result.status_code == 503

    @patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False)
    def test_cruxes_503_when_unavailable(self, handler):
        """Test returns 503 when belief network unavailable."""
        result = handler.handle("/api/belief-network/debate-123/cruxes", {}, Mock())
        assert result.status_code == 503

    def test_cruxes_validates_debate_id(self, handler):
        """Test cruxes validates debate_id format."""
        # Invalid characters in debate_id
        result = handler.handle("/api/belief-network/../etc/passwd/cruxes", {}, Mock())
        assert result is None or result.status_code == 400

    def test_claim_support_parses_path(self, handler):
        """Test claim support endpoint parses path correctly."""
        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", True):
            with patch("aragora.server.handlers.belief.ProvenanceTracker") as mock_pt:
                mock_pt.load.return_value = Mock(get_claim_support=Mock(return_value=None))
                # Should parse debate-123 and claim-456 from path
                result = handler.handle(
                    "/api/provenance/debate-123/claims/claim-456/support",
                    {},
                    Mock()
                )
                # Will fail due to nomic_dir not existing, but validates parsing
                assert result is not None


# ============================================================================
# Handler Integration Tests
# ============================================================================

class TestHandlerRouting:
    """Tests for handler routing in unified server."""

    def test_handlers_import(self):
        """Test all handlers can be imported."""
        from aragora.server.handlers import (
            DebatesHandler,
            AgentsHandler,
            SystemHandler,
            PulseHandler,
            AnalyticsHandler,
            MetricsHandler,
            ConsensusHandler,
            BeliefHandler,
        )
        # All imports should succeed
        assert DebatesHandler is not None
        assert ConsensusHandler is not None
        assert BeliefHandler is not None

    def test_unified_server_registers_handlers(self):
        """Test unified server registers all handlers."""
        from aragora.server.unified_server import UnifiedHandler

        # Check handler class variables exist
        assert hasattr(UnifiedHandler, '_consensus_handler')
        assert hasattr(UnifiedHandler, '_belief_handler')

    def test_handler_result_structure(self):
        """Test HandlerResult has required fields."""
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"test": true}',
        )
        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert result.headers == {}  # Default empty dict


# ============================================================================
# Security Tests
# ============================================================================

class TestHandlerSecurity:
    """Tests for handler security measures."""

    @pytest.fixture
    def belief_handler(self):
        """Create belief handler."""
        return BeliefHandler({"nomic_dir": Path("/tmp")})

    def test_path_traversal_blocked(self, belief_handler):
        """Test path traversal attempts are blocked."""
        # Attempt path traversal in debate_id
        assert belief_handler.can_handle("/api/belief-network/../etc/passwd/cruxes") is True
        # But the handler should reject invalid IDs
        result = belief_handler.handle(
            "/api/belief-network/../etc/passwd/cruxes",
            {},
            Mock()
        )
        # Should either return None (not handling) or 400 (bad request)
        assert result is None or result.status_code in (400, 503)

    def test_safe_id_pattern_enforced(self, belief_handler):
        """Test safe ID pattern is enforced."""
        # Valid patterns
        assert belief_handler._extract_debate_id("/api/test/debate-123/end", 3) == "debate-123"
        assert belief_handler._extract_debate_id("/api/test/abc_def/end", 3) == "abc_def"

        # Invalid patterns (should return None)
        assert belief_handler._extract_debate_id("/api/test/../../etc/end", 3) is None
        assert belief_handler._extract_debate_id("/api/test/a b c/end", 3) is None


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in handlers."""

    def test_safe_error_message_hides_paths(self):
        """Test error messages don't leak file paths."""
        from aragora.server.handlers.consensus import _safe_error_message

        error = FileNotFoundError("/etc/passwd not found")
        msg = _safe_error_message(error, "test")
        assert "/etc" not in msg
        assert "Resource not found" in msg

    def test_safe_error_message_handles_timeout(self):
        """Test timeout errors are handled."""
        from aragora.server.handlers.consensus import _safe_error_message

        error = TimeoutError("Operation timed out after 30s")
        msg = _safe_error_message(error, "test")
        assert "timed out" in msg.lower()

    def test_generic_errors_sanitized(self):
        """Test generic errors are sanitized."""
        from aragora.server.handlers.consensus import _safe_error_message

        error = Exception("Internal database password: secret123")
        msg = _safe_error_message(error, "test")
        assert "secret123" not in msg
        assert "password" not in msg


# ============================================================================
# CritiqueHandler Tests
# ============================================================================

class TestCritiqueHandler:
    """Tests for CritiqueHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        from aragora.server.handlers import CritiqueHandler
        ctx = {
            "nomic_dir": Path("/tmp/test"),
        }
        return CritiqueHandler(ctx)

    def test_can_handle_patterns(self, handler):
        """Test can_handle for patterns endpoint."""
        assert handler.can_handle("/api/critiques/patterns") is True

    def test_can_handle_archive(self, handler):
        """Test can_handle for archive endpoint."""
        assert handler.can_handle("/api/critiques/archive") is True

    def test_can_handle_all_reputations(self, handler):
        """Test can_handle for all reputations endpoint."""
        assert handler.can_handle("/api/reputation/all") is True

    def test_can_handle_agent_reputation(self, handler):
        """Test can_handle for agent-specific reputation."""
        assert handler.can_handle("/api/agent/claude/reputation") is True
        assert handler.can_handle("/api/agent/gpt-4/reputation") is True

    def test_cannot_handle_unrelated(self, handler):
        """Test rejects unrelated endpoints."""
        assert handler.can_handle("/api/consensus/stats") is False
        assert handler.can_handle("/api/debates") is False

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", False)
    def test_returns_503_when_unavailable(self, handler):
        """Test returns 503 when critique store unavailable."""
        result = handler.handle("/api/critiques/patterns", {}, Mock())
        assert result.status_code == 503

    def test_extract_agent_name_valid(self, handler):
        """Test agent name extraction."""
        assert handler._extract_agent_name("/api/agent/claude/reputation") == "claude"
        assert handler._extract_agent_name("/api/agent/gpt-4/reputation") == "gpt-4"

    def test_extract_agent_name_invalid(self, handler):
        """Test agent name extraction blocks invalid patterns."""
        assert handler._extract_agent_name("/api/agent/../etc/reputation") is None
        assert handler._extract_agent_name("/api/agent/a b c/reputation") is None


# ============================================================================
# GenesisHandler Tests
# ============================================================================

class TestGenesisHandler:
    """Tests for GenesisHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        from aragora.server.handlers import GenesisHandler
        ctx = {
            "nomic_dir": Path("/tmp/test"),
        }
        return GenesisHandler(ctx)

    def test_can_handle_stats(self, handler):
        """Test can_handle for stats endpoint."""
        assert handler.can_handle("/api/genesis/stats") is True

    def test_can_handle_events(self, handler):
        """Test can_handle for events endpoint."""
        assert handler.can_handle("/api/genesis/events") is True

    def test_can_handle_lineage(self, handler):
        """Test can_handle for lineage endpoint."""
        assert handler.can_handle("/api/genesis/lineage/genome-123") is True

    def test_can_handle_tree(self, handler):
        """Test can_handle for tree endpoint."""
        assert handler.can_handle("/api/genesis/tree/debate-456") is True

    def test_cannot_handle_unrelated(self, handler):
        """Test rejects unrelated endpoints."""
        assert handler.can_handle("/api/consensus/stats") is False
        assert handler.can_handle("/api/debates") is False

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False)
    def test_returns_503_when_unavailable(self, handler):
        """Test returns 503 when genesis module unavailable."""
        result = handler.handle("/api/genesis/stats", {}, Mock())
        assert result.status_code == 503

    def test_lineage_validates_genome_id(self, handler):
        """Test lineage validates genome_id format."""
        # Valid genome ID
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = handler.handle("/api/genesis/lineage/genome-123", {}, Mock())
            assert result.status_code == 503  # Would fail due to unavailable module

        # Invalid genome ID (path traversal)
        result = handler.handle("/api/genesis/lineage/../etc/passwd", {}, Mock())
        assert result.status_code == 400

    def test_tree_validates_debate_id(self, handler):
        """Test tree validates debate_id format."""
        # Invalid debate ID
        result = handler.handle("/api/genesis/tree/../etc/passwd", {}, Mock())
        assert result.status_code == 400
