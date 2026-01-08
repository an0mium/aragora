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
    ReplaysHandler,
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


class TestPathValidation:
    """Tests for path validation utilities."""

    def test_validate_agent_name_valid(self):
        """Test valid agent names."""
        from aragora.server.handlers.base import validate_agent_name

        is_valid, err = validate_agent_name("claude-3-opus")
        assert is_valid is True
        assert err is None

        is_valid, err = validate_agent_name("agent_123")
        assert is_valid is True
        assert err is None

    def test_validate_agent_name_invalid_chars(self):
        """Test invalid characters in agent name."""
        from aragora.server.handlers.base import validate_agent_name

        is_valid, err = validate_agent_name("agent$bad")
        assert is_valid is False
        assert "invalid" in err.lower()  # Error message includes "Invalid agent name"

    def test_validate_agent_name_path_traversal(self):
        """Test path traversal in agent name."""
        from aragora.server.handlers.base import validate_agent_name

        is_valid, err = validate_agent_name("../etc/passwd")
        assert is_valid is False
        assert "invalid" in err.lower()  # Pattern blocks path traversal characters

    def test_validate_agent_name_too_long(self):
        """Test agent name too long (max 32 chars)."""
        from aragora.server.handlers.base import validate_agent_name

        is_valid, err = validate_agent_name("a" * 33)
        assert is_valid is False
        assert "invalid" in err.lower()  # Pattern enforces max length

    def test_validate_agent_name_empty(self):
        """Test empty agent name."""
        from aragora.server.handlers.base import validate_agent_name

        is_valid, err = validate_agent_name("")
        assert is_valid is False
        assert "missing" in err.lower()

    def test_validate_debate_id_valid(self):
        """Test valid debate IDs."""
        from aragora.server.handlers.base import validate_debate_id

        is_valid, err = validate_debate_id("debate-2024-01-15-abc123")
        assert is_valid is True
        assert err is None

    def test_validate_debate_id_invalid(self):
        """Test invalid debate IDs."""
        from aragora.server.handlers.base import validate_debate_id

        is_valid, err = validate_debate_id("debate/with/slashes")
        assert is_valid is False

    def test_validate_debate_id_too_long(self):
        """Test debate ID too long (max 128 chars)."""
        from aragora.server.handlers.base import validate_debate_id

        is_valid, err = validate_debate_id("a" * 129)
        assert is_valid is False


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

    def test_cannot_handle_emergent_traits(self, handler):
        """Test can_handle for emergent-traits (moved to LaboratoryHandler)."""
        assert handler.can_handle("/api/laboratory/emergent-traits") is False

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

    def test_emergent_traits_not_handled(self, handler):
        """Test BeliefHandler does not handle emergent-traits (moved to LaboratoryHandler)."""
        result = handler.handle("/api/laboratory/emergent-traits", {}, Mock())
        assert result is None  # Route handled by LaboratoryHandler now

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

# ============================================================================
# AgentsHandler Tests
# ============================================================================

class TestAgentsHandler:
    """Tests for AgentsHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        from aragora.server.handlers import AgentsHandler
        from aragora.server.handlers.base import clear_cache
        clear_cache()
        ctx = {
            "storage": Mock(),
            "elo_system": Mock(),
            "nomic_dir": Path("/tmp/test"),
        }
        return AgentsHandler(ctx)

    def test_can_handle_leaderboard(self, handler):
        """Test can_handle for leaderboard endpoint."""
        assert handler.can_handle("/api/leaderboard") is True
        assert handler.can_handle("/api/rankings") is True

    def test_cannot_handle_calibration(self, handler):
        """Test can_handle for calibration (moved to CalibrationHandler)."""
        assert handler.can_handle("/api/calibration/leaderboard") is False

    def test_can_handle_matches(self, handler):
        """Test can_handle for matches endpoint."""
        assert handler.can_handle("/api/matches/recent") is True

    def test_can_handle_agent_profile(self, handler):
        """Test can_handle for agent profile endpoint."""
        assert handler.can_handle("/api/agent/claude/profile") is True
        assert handler.can_handle("/api/agent/gpt-4/profile") is True

    def test_can_handle_agent_history(self, handler):
        """Test can_handle for agent history endpoint."""
        assert handler.can_handle("/api/agent/claude/history") is True

    def test_can_handle_head_to_head(self, handler):
        """Test can_handle for head-to-head endpoint."""
        assert handler.can_handle("/api/agent/claude/head-to-head/gpt-4") is True

    def test_can_handle_compare(self, handler):
        """Test can_handle for compare endpoint."""
        assert handler.can_handle("/api/agent/compare") is True

    def test_cannot_handle_unrelated(self, handler):
        """Test rejects unrelated endpoints."""
        assert handler.can_handle("/api/consensus/stats") is False
        assert handler.can_handle("/api/debates") is False

    def test_leaderboard_returns_503_when_no_elo(self):
        """Test returns 503 when ELO system unavailable."""
        from aragora.server.handlers import AgentsHandler
        from aragora.server.handlers.base import clear_cache
        clear_cache()
        ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
        handler = AgentsHandler(ctx)
        result = handler.handle("/api/leaderboard", {}, Mock())
        assert result.status_code == 503

    def test_compare_requires_two_agents(self, handler):
        """Test compare requires at least 2 agents."""
        result = handler.handle("/api/agent/compare", {"agents": ["claude"]}, Mock())
        assert result.status_code == 400


# ============================================================================
# AnalyticsHandler Tests
# ============================================================================

class TestAnalyticsHandler:
    """Tests for AnalyticsHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        from aragora.server.handlers import AnalyticsHandler
        from aragora.server.handlers.base import clear_cache
        clear_cache()
        ctx = {
            "storage": Mock(),
            "elo_system": Mock(),
            "nomic_dir": Path("/tmp/test"),
        }
        return AnalyticsHandler(ctx)

    def test_can_handle_disagreements(self, handler):
        """Test can_handle for disagreements endpoint."""
        assert handler.can_handle("/api/analytics/disagreements") is True

    def test_can_handle_role_rotation(self, handler):
        """Test can_handle for role-rotation endpoint."""
        assert handler.can_handle("/api/analytics/role-rotation") is True

    def test_can_handle_early_stops(self, handler):
        """Test can_handle for early-stops endpoint."""
        assert handler.can_handle("/api/analytics/early-stops") is True

    def test_can_handle_ranking_stats(self, handler):
        """Test can_handle for ranking stats endpoint."""
        assert handler.can_handle("/api/ranking/stats") is True

    def test_can_handle_memory_stats(self, handler):
        """Test can_handle for memory stats endpoint."""
        assert handler.can_handle("/api/memory/stats") is True
        # Note: /api/memory/tier-stats moved to MemoryHandler
        assert handler.can_handle("/api/memory/tier-stats") is False

    def test_cannot_handle_unrelated(self, handler):
        """Test rejects unrelated endpoints."""
        assert handler.can_handle("/api/consensus/stats") is False
        assert handler.can_handle("/api/debates") is False


# ============================================================================
# DebatesHandler Tests
# ============================================================================

class TestDebatesHandler:
    """Tests for DebatesHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        from aragora.server.handlers import DebatesHandler
        ctx = {
            "storage": Mock(),
            "nomic_dir": Path("/tmp/test"),
        }
        return DebatesHandler(ctx)

    def test_can_handle_debates_list(self, handler):
        """Test can_handle for debates list endpoint."""
        assert handler.can_handle("/api/debates") is True

    def test_can_handle_debate_by_slug(self, handler):
        """Test can_handle for debate by slug endpoint."""
        assert handler.can_handle("/api/debates/test-debate-123") is True

    def test_can_handle_export(self, handler):
        """Test can_handle for export endpoint."""
        assert handler.can_handle("/api/debates/test-debate/export") is True

    def test_can_handle_impasse(self, handler):
        """Test can_handle for impasse endpoint."""
        assert handler.can_handle("/api/debates/test-debate/impasse") is True

    def test_cannot_handle_unrelated(self, handler):
        """Test rejects unrelated endpoints."""
        assert handler.can_handle("/api/consensus/stats") is False
        assert handler.can_handle("/api/agents") is False


# ============================================================================
# SystemHandler Tests
# ============================================================================

class TestSystemHandler:
    """Tests for SystemHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        from aragora.server.handlers import SystemHandler
        ctx = {
            "storage": Mock(),
            "nomic_dir": Path("/tmp/test"),
        }
        return SystemHandler(ctx)

    def test_can_handle_health(self, handler):
        """Test can_handle for health endpoint."""
        assert handler.can_handle("/api/health") is True

    def test_can_handle_nomic_state(self, handler):
        """Test can_handle for nomic state endpoint."""
        assert handler.can_handle("/api/nomic/state") is True

    def test_can_handle_modes(self, handler):
        """Test can_handle for modes endpoint."""
        assert handler.can_handle("/api/modes") is True

    def test_can_handle_history(self, handler):
        """Test can_handle for history endpoints."""
        assert handler.can_handle("/api/history/cycles") is True
        assert handler.can_handle("/api/history/events") is True
        assert handler.can_handle("/api/history/debates") is True
        assert handler.can_handle("/api/history/summary") is True

    def test_cannot_handle_unrelated(self, handler):
        """Test rejects unrelated endpoints."""
        assert handler.can_handle("/api/consensus/stats") is False
        assert handler.can_handle("/api/debates") is False


# ============================================================================
# PulseHandler Tests
# ============================================================================

class TestPulseHandler:
    """Tests for PulseHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        from aragora.server.handlers import PulseHandler
        ctx = {
            "nomic_dir": Path("/tmp/test"),
        }
        return PulseHandler(ctx)

    def test_can_handle_trending(self, handler):
        """Test can_handle for trending endpoint."""
        assert handler.can_handle("/api/pulse/trending") is True

    def test_can_handle_suggest(self, handler):
        """Test can_handle for suggest endpoint."""
        assert handler.can_handle("/api/pulse/suggest") is True

    def test_cannot_handle_unrelated(self, handler):
        """Test rejects unrelated endpoints."""
        assert handler.can_handle("/api/consensus/stats") is False
        assert handler.can_handle("/api/debates") is False


# ============================================================================
# MetricsHandler Tests
# ============================================================================

class TestMetricsHandler:
    """Tests for MetricsHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        from aragora.server.handlers import MetricsHandler
        ctx = {
            "storage": Mock(),
            "elo_system": Mock(),
            "nomic_dir": Path("/tmp/test"),
        }
        return MetricsHandler(ctx)

    def test_can_handle_metrics(self, handler):
        """Test can_handle for metrics endpoint."""
        assert handler.can_handle("/api/metrics") is True

    def test_can_handle_health(self, handler):
        """Test can_handle for health endpoint."""
        assert handler.can_handle("/api/metrics/health") is True

    def test_can_handle_cache_stats(self, handler):
        """Test can_handle for cache stats endpoint."""
        assert handler.can_handle("/api/metrics/cache") is True

    def test_can_handle_system_info(self, handler):
        """Test can_handle for system info endpoint."""
        assert handler.can_handle("/api/metrics/system") is True

    def test_cannot_handle_unrelated(self, handler):
        """Test rejects unrelated endpoints."""
        assert handler.can_handle("/api/consensus/stats") is False
        assert handler.can_handle("/api/debates") is False


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


# ============================================================================
# ReplaysHandler Tests
# ============================================================================

class TestReplaysHandler:
    """Tests for ReplaysHandler."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create a ReplaysHandler with test directory."""
        ctx = {"nomic_dir": tmp_path}
        return ReplaysHandler(ctx)

    @pytest.fixture
    def handler_no_nomic(self):
        """Create a ReplaysHandler without nomic_dir."""
        ctx = {"nomic_dir": None}
        return ReplaysHandler(ctx)

    def test_can_handle_replays_list(self, handler):
        """Test can_handle for replays list endpoint."""
        assert handler.can_handle("/api/replays") is True

    def test_can_handle_replay_detail(self, handler):
        """Test can_handle for specific replay endpoint."""
        assert handler.can_handle("/api/replays/replay-123") is True
        assert handler.can_handle("/api/replays/my_replay") is True

    def test_can_handle_learning_evolution(self, handler):
        """Test can_handle for learning evolution endpoint."""
        assert handler.can_handle("/api/learning/evolution") is True

    def test_cannot_handle_unrelated(self, handler):
        """Test rejects unrelated endpoints."""
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agents") is False
        assert handler.can_handle("/api/replays/foo/bar/baz") is False  # Too deep

    def test_list_replays_empty(self, handler, tmp_path):
        """Test listing replays when no replays exist."""
        result = handler.handle("/api/replays", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data == []

    def test_list_replays_with_data(self, handler, tmp_path):
        """Test listing replays with actual replay data."""
        # Create replays directory with test data
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir()

        # Create a replay
        replay_dir = replays_dir / "test-replay-001"
        replay_dir.mkdir()
        meta = {
            "topic": "Test Topic",
            "agents": [{"name": "Agent1"}, {"name": "Agent2"}],
            "schema_version": "2.0"
        }
        (replay_dir / "meta.json").write_text(json.dumps(meta))

        # Need to clear cache to get fresh results
        from aragora.server.handlers.base import clear_cache
        clear_cache()

        result = handler.handle("/api/replays", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data) == 1
        assert data[0]["id"] == "test-replay-001"
        assert data[0]["topic"] == "Test Topic"
        assert data[0]["agents"] == ["Agent1", "Agent2"]

    def test_list_replays_no_nomic_dir(self, handler_no_nomic):
        """Test listing replays without nomic_dir configured."""
        result = handler_no_nomic.handle("/api/replays", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data == []

    def test_get_replay_not_found(self, handler, tmp_path):
        """Test getting non-existent replay."""
        (tmp_path / "replays").mkdir()
        result = handler.handle("/api/replays/nonexistent", {}, Mock())
        assert result.status_code == 404

    def test_get_replay_with_events(self, handler, tmp_path):
        """Test getting replay with events."""
        # Create replay with events
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir()
        replay_dir = replays_dir / "test-replay"
        replay_dir.mkdir()

        meta = {"topic": "Test", "agents": []}
        (replay_dir / "meta.json").write_text(json.dumps(meta))

        events = [
            {"type": "start", "timestamp": 0},
            {"type": "argument", "content": "Hello"},
            {"type": "end", "timestamp": 100}
        ]
        (replay_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events)
        )

        from aragora.server.handlers.base import clear_cache
        clear_cache()

        result = handler.handle("/api/replays/test-replay", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["id"] == "test-replay"
        assert data["event_count"] == 3
        assert len(data["events"]) == 3

    def test_get_replay_no_nomic_dir(self, handler_no_nomic):
        """Test getting replay without nomic_dir configured."""
        result = handler_no_nomic.handle("/api/replays/any-id", {}, Mock())
        assert result.status_code == 503

    def test_replay_id_path_traversal_blocked(self, handler):
        """Test path traversal is blocked in replay ID."""
        result = handler.handle("/api/replays/../etc/passwd", {}, Mock())
        assert result.status_code == 400

    def test_replay_id_invalid_chars_blocked(self, handler):
        """Test invalid characters in replay ID are blocked."""
        result = handler.handle("/api/replays/foo$bar", {}, Mock())
        assert result.status_code == 400
        result = handler.handle("/api/replays/foo/bar", {}, Mock())
        # This would be caught by can_handle returning False
        assert handler.can_handle("/api/replays/foo/bar") is False

    def test_learning_evolution_no_db(self, handler, tmp_path):
        """Test learning evolution when no database exists."""
        result = handler.handle("/api/learning/evolution", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["patterns"] == []
        assert data["count"] == 0

    def test_learning_evolution_no_nomic_dir(self, handler_no_nomic):
        """Test learning evolution without nomic_dir."""
        result = handler_no_nomic.handle("/api/learning/evolution", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["patterns"] == []

    def test_learning_evolution_with_limit(self, handler):
        """Test learning evolution respects limit parameter."""
        result = handler.handle("/api/learning/evolution", {"limit": "5"}, Mock())
        assert result.status_code == 200
