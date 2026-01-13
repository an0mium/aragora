"""Tests for MemoryHandler - continuum memory endpoints."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from enum import Enum

from aragora.server.handlers.memory import MemoryHandler, CONTINUUM_AVAILABLE


class MockMemoryTier(Enum):
    """Mock MemoryTier for testing."""

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    GLACIAL = "glacial"


class MockMemory:
    """Mock memory object for testing."""

    def __init__(self, id="mem1", tier=None, content="Test content", importance=0.5):
        self.id = id
        self.tier = tier or MockMemoryTier.FAST
        self.content = content
        self.importance = importance
        self.surprise_score = 0.3
        self.consolidation_score = 0.7
        self.update_count = 2
        self.created_at = "2024-01-01T00:00:00"
        self.updated_at = "2024-01-02T00:00:00"


@pytest.fixture
def mock_ctx():
    """Create mock context for handler."""
    return {
        "nomic_dir": Path("/tmp/test_nomic"),
        "continuum_memory": None,
    }


@pytest.fixture
def handler(mock_ctx):
    """Create MemoryHandler with mock context."""
    return MemoryHandler(mock_ctx)


class TestMemoryHandlerRouting:
    """Test route matching for MemoryHandler."""

    def test_can_handle_retrieve_endpoint(self, handler):
        """Handler should match /api/memory/continuum/retrieve."""
        assert handler.can_handle("/api/memory/continuum/retrieve")

    def test_can_handle_consolidate_endpoint(self, handler):
        """Handler should match /api/memory/continuum/consolidate."""
        assert handler.can_handle("/api/memory/continuum/consolidate")

    def test_cannot_handle_invalid_path(self, handler):
        """Handler should not match invalid paths."""
        assert not handler.can_handle("/api/memory")
        assert not handler.can_handle("/api/memory/continuum")
        assert not handler.can_handle("/api/memory/other")

    def test_cannot_handle_partial_paths(self, handler):
        """Handler should not match partial paths."""
        assert not handler.can_handle("/api/memory/continuum/retrieve/extra")


class TestRetrieveEndpoint:
    """Test /api/memory/continuum/retrieve endpoint."""

    def test_retrieve_no_continuum(self, handler):
        """Returns 503 when continuum memory not configured."""
        result = handler.handle("/api/memory/continuum/retrieve", {}, None)
        assert result is not None
        assert result.status_code == 503

    def test_retrieve_with_continuum(self, mock_ctx):
        """Returns memories when continuum is configured."""
        mock_memory = MockMemory()
        mock_continuum = Mock()
        mock_continuum.retrieve.return_value = [mock_memory]

        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        with patch("aragora.server.handlers.memory.MemoryTier", MockMemoryTier):
            result = handler.handle("/api/memory/continuum/retrieve", {}, None)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "memories" in body
        assert body["count"] == 1

    def test_retrieve_with_query(self, mock_ctx):
        """Query parameter is passed to retrieve."""
        mock_continuum = Mock()
        mock_continuum.retrieve.return_value = []
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        with patch("aragora.server.handlers.memory.MemoryTier", MockMemoryTier):
            result = handler.handle(
                "/api/memory/continuum/retrieve", {"query": ["test query"]}, None
            )

        mock_continuum.retrieve.assert_called_once()
        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["query"] == "test query"

    def test_retrieve_with_limit(self, mock_ctx):
        """Limit parameter is passed to retrieve."""
        mock_continuum = Mock()
        mock_continuum.retrieve.return_value = []
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        with patch("aragora.server.handlers.memory.MemoryTier", MockMemoryTier):
            result = handler.handle(
                "/api/memory/continuum/retrieve",
                {"limit": "5"},  # parse_query_params converts single-value lists to strings
                None,
            )

        mock_continuum.retrieve.assert_called_once()
        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["limit"] == 5

    def test_retrieve_with_min_importance(self, mock_ctx):
        """min_importance parameter is passed to retrieve."""
        mock_continuum = Mock()
        mock_continuum.retrieve.return_value = []
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        with patch("aragora.server.handlers.memory.MemoryTier", MockMemoryTier):
            result = handler.handle(
                "/api/memory/continuum/retrieve", {"min_importance": ["0.7"]}, None
            )

        mock_continuum.retrieve.assert_called_once()
        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["min_importance"] == 0.7

    def test_retrieve_with_tiers(self, mock_ctx):
        """Tiers parameter filters memory tiers."""
        mock_continuum = Mock()
        mock_continuum.retrieve.return_value = []
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        with patch("aragora.server.handlers.memory.MemoryTier", MockMemoryTier):
            result = handler.handle(
                "/api/memory/continuum/retrieve", {"tiers": ["fast,slow"]}, None
            )

        mock_continuum.retrieve.assert_called_once()
        call_kwargs = mock_continuum.retrieve.call_args[1]
        tier_names = [t.name for t in call_kwargs["tiers"]]
        assert "FAST" in tier_names
        assert "SLOW" in tier_names

    def test_retrieve_truncates_long_content(self, mock_ctx):
        """Long memory content is truncated in response."""
        long_content = "x" * 600
        mock_memory = MockMemory(content=long_content)
        mock_continuum = Mock()
        mock_continuum.retrieve.return_value = [mock_memory]
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        with patch("aragora.server.handlers.memory.MemoryTier", MockMemoryTier):
            result = handler.handle("/api/memory/continuum/retrieve", {}, None)

        body = json.loads(result.body)
        assert len(body["memories"][0]["content"]) <= 503  # 500 + "..."


class TestConsolidateEndpoint:
    """Test /api/memory/continuum/consolidate endpoint."""

    def _make_auth_handler(self, authenticated=True):
        """Create mock request handler with auth headers."""
        mock_handler = Mock()
        mock_handler.headers = {"Authorization": "Bearer test_token"} if authenticated else {}
        return mock_handler

    def test_consolidate_returns_405_on_get(self, handler):
        """GET request returns 405 Method Not Allowed."""
        result = handler.handle("/api/memory/continuum/consolidate", {}, None)
        assert result is not None
        assert result.status_code == 405
        body = json.loads(result.body)
        assert "POST" in body.get("error", "")

    def test_consolidate_requires_auth(self, mock_ctx):
        """Returns 401 when not authenticated."""
        mock_ctx["user_store"] = None
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler(authenticated=False)

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = False
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_post("/api/memory/continuum/consolidate", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401

    def test_consolidate_no_continuum(self, mock_ctx):
        """Returns 503 when continuum memory not configured."""
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_post("/api/memory/continuum/consolidate", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    def test_consolidate_success(self, mock_ctx):
        """Returns success when consolidation completes."""
        mock_continuum = Mock()
        mock_continuum.consolidate.return_value = {
            "processed": 10,
            "promoted": 3,
            "consolidated": 2,
        }
        mock_ctx["continuum_memory"] = mock_continuum
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_post("/api/memory/continuum/consolidate", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["entries_processed"] == 10
        assert body["entries_promoted"] == 3
        assert body["entries_consolidated"] == 2
        assert "duration_seconds" in body

    def test_consolidate_tracks_duration(self, mock_ctx):
        """Consolidation response includes duration."""
        mock_continuum = Mock()
        mock_continuum.consolidate.return_value = {}
        mock_ctx["continuum_memory"] = mock_continuum
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_post("/api/memory/continuum/consolidate", {}, mock_handler)

        body = json.loads(result.body)
        assert "duration_seconds" in body
        assert isinstance(body["duration_seconds"], float)


class TestMemoryNotConfigured:
    """Test error handling when memory is not configured."""

    def test_retrieve_unavailable_system(self, mock_ctx):
        """Returns 503 when continuum system unavailable."""
        mock_ctx["continuum_memory"] = None
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/continuum/retrieve", {}, None)
        assert result.status_code == 503

    def test_consolidate_unavailable_system(self, mock_ctx):
        """Returns 503 when continuum system unavailable (via POST with auth)."""
        mock_ctx["continuum_memory"] = None
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = Mock()
        mock_handler.headers = {"Authorization": "Bearer test"}

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_post("/api/memory/continuum/consolidate", {}, mock_handler)

        assert result.status_code == 503


class TestMemoryHandlerImport:
    """Test MemoryHandler import and export."""

    def test_handler_importable(self):
        """MemoryHandler can be imported from handlers package."""
        from aragora.server.handlers import MemoryHandler

        assert MemoryHandler is not None

    def test_handler_in_all_exports(self):
        """MemoryHandler is in __all__ exports."""
        from aragora.server.handlers import __all__

        assert "MemoryHandler" in __all__

    def test_continuum_available_flag(self):
        """CONTINUUM_AVAILABLE flag is defined."""
        from aragora.server.handlers.memory import CONTINUUM_AVAILABLE

        assert isinstance(CONTINUUM_AVAILABLE, bool)


class TestErrorHandling:
    """Test error handling in MemoryHandler."""

    def test_handle_returns_none_for_unmatched(self, handler):
        """Handle returns None for unmatched paths."""
        result = handler.handle("/api/unmatched", {}, None)
        assert result is None

    def test_retrieve_handles_exception(self, mock_ctx):
        """Retrieve handles exceptions gracefully."""
        mock_continuum = Mock()
        mock_continuum.retrieve.side_effect = Exception("Test error")
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        with patch("aragora.server.handlers.memory.MemoryTier", MockMemoryTier):
            result = handler.handle("/api/memory/continuum/retrieve", {}, None)

        assert result.status_code == 500

    def test_consolidate_handles_exception(self, mock_ctx):
        """Consolidate handles exceptions gracefully (via POST with auth)."""
        mock_continuum = Mock()
        mock_continuum.consolidate.side_effect = Exception("Test error")
        mock_ctx["continuum_memory"] = mock_continuum
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = Mock()
        mock_handler.headers = {"Authorization": "Bearer test"}

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_post("/api/memory/continuum/consolidate", {}, mock_handler)

        assert result.status_code == 500


class TestMemoryPressureEndpoint:
    """Tests for the memory pressure monitoring endpoint."""

    def test_can_handle_pressure_endpoint(self, handler):
        """Handler recognizes pressure endpoint."""
        assert handler.can_handle("/api/memory/pressure")

    def test_pressure_returns_503_when_not_configured(self, mock_ctx):
        """Returns 503 when continuum memory not configured."""
        mock_ctx["continuum_memory"] = None
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/pressure", {}, None)

        assert result.status_code == 503

    def test_pressure_returns_normal_status(self, mock_ctx):
        """Returns normal status when pressure is low."""
        mock_continuum = Mock()
        mock_continuum.get_memory_pressure.return_value = 0.3
        mock_continuum.get_stats.return_value = {
            "by_tier": {
                "FAST": {"count": 10},
                "MEDIUM": {"count": 50},
            },
            "total_memories": 60,
        }
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/pressure", {}, None)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "normal"
        assert body["pressure"] == 0.3
        assert body["cleanup_recommended"] is False

    def test_pressure_returns_elevated_status(self, mock_ctx):
        """Returns elevated status when pressure is moderate."""
        mock_continuum = Mock()
        mock_continuum.get_memory_pressure.return_value = 0.65
        mock_continuum.get_stats.return_value = {"by_tier": {}, "total_memories": 100}
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/pressure", {}, None)

        body = json.loads(result.body)
        assert body["status"] == "elevated"

    def test_pressure_returns_high_status(self, mock_ctx):
        """Returns high status when pressure is significant."""
        mock_continuum = Mock()
        mock_continuum.get_memory_pressure.return_value = 0.85
        mock_continuum.get_stats.return_value = {"by_tier": {}, "total_memories": 200}
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/pressure", {}, None)

        body = json.loads(result.body)
        assert body["status"] == "high"

    def test_pressure_recommends_cleanup_when_critical(self, mock_ctx):
        """Recommends cleanup when pressure > 0.9 (no auto-trigger for idempotent GET)."""
        mock_continuum = Mock()
        mock_continuum.get_memory_pressure.return_value = 0.95
        mock_continuum.get_stats.return_value = {"by_tier": {}, "total_memories": 500}
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/pressure", {}, None)

        body = json.loads(result.body)
        assert body["status"] == "critical"
        assert body["cleanup_recommended"] is True
        # GET endpoint should be idempotent - no cleanup call
        mock_continuum.cleanup_expired_memories.assert_not_called()

    def test_pressure_includes_tier_utilization(self, mock_ctx):
        """Response includes per-tier utilization breakdown."""
        mock_continuum = Mock()
        mock_continuum.get_memory_pressure.return_value = 0.4
        mock_continuum.get_stats.return_value = {
            "by_tier": {
                "FAST": {"count": 50},
                "MEDIUM": {"count": 250},
                "SLOW": {"count": 500},
                "GLACIAL": {"count": 1000},
            },
            "total_memories": 1800,
        }
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/pressure", {}, None)

        body = json.loads(result.body)
        assert "tier_utilization" in body
        assert body["tier_utilization"]["FAST"]["count"] == 50
        assert body["tier_utilization"]["FAST"]["limit"] == 100
        assert body["tier_utilization"]["FAST"]["utilization"] == 0.5

    def test_pressure_does_not_recommend_cleanup_when_normal(self, mock_ctx):
        """Does not recommend cleanup when pressure is below threshold."""
        mock_continuum = Mock()
        mock_continuum.get_memory_pressure.return_value = 0.5
        mock_continuum.get_stats.return_value = {"by_tier": {}, "total_memories": 100}
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/pressure", {}, None)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "elevated"
        assert body["cleanup_recommended"] is False


class TestDeleteMemoryEndpoint:
    """Tests for DELETE /api/memory/continuum/{id} endpoint."""

    def _make_auth_handler(self, authenticated=True, client_ip="127.0.0.1"):
        """Create mock request handler with auth headers."""
        mock_handler = Mock()
        mock_handler.headers = {"Authorization": "Bearer test_token"} if authenticated else {}
        mock_handler.client_address = (client_ip, 12345)
        return mock_handler

    def test_delete_requires_auth(self, mock_ctx):
        """Returns 401 when not authenticated."""
        mock_ctx["user_store"] = None
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler(authenticated=False)

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = False
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_delete("/api/memory/continuum/mem123", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401

    def test_delete_returns_503_when_not_configured(self, mock_ctx):
        """Returns 503 when continuum memory not configured."""
        mock_ctx["continuum_memory"] = None
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_delete("/api/memory/continuum/mem123", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    def test_delete_success(self, mock_ctx):
        """Returns 200 when memory deleted successfully."""
        mock_continuum = Mock()
        mock_continuum.delete.return_value = True
        mock_ctx["continuum_memory"] = mock_continuum
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_delete("/api/memory/continuum/mem123", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert "mem123" in body["message"]
        mock_continuum.delete.assert_called_once_with("mem123")

    def test_delete_not_found(self, mock_ctx):
        """Returns 404 when memory not found."""
        mock_continuum = Mock()
        mock_continuum.delete.return_value = False
        mock_ctx["continuum_memory"] = mock_continuum
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_delete("/api/memory/continuum/nonexistent", {}, mock_handler)

        assert result is not None
        assert result.status_code == 404

    def test_delete_no_method_on_continuum(self, mock_ctx):
        """Returns 501 when continuum doesn't support delete."""
        mock_continuum = Mock(spec=[])  # Empty spec = no delete method
        mock_ctx["continuum_memory"] = mock_continuum
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_delete("/api/memory/continuum/mem123", {}, mock_handler)

        assert result is not None
        assert result.status_code == 501

    def test_delete_handles_exception(self, mock_ctx):
        """Returns 500 when delete raises exception."""
        mock_continuum = Mock()
        mock_continuum.delete.side_effect = Exception("Database error")
        mock_ctx["continuum_memory"] = mock_continuum
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_delete("/api/memory/continuum/mem123", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500

    def test_delete_validates_memory_id(self, mock_ctx):
        """Should validate memory ID format."""
        mock_continuum = Mock()
        mock_ctx["continuum_memory"] = mock_continuum
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            # Path traversal attempt
            result = handler.handle_delete(
                "/api/memory/continuum/../../../etc/passwd", {}, mock_handler
            )

        # Should either reject with 400 or not match the route at all (None)
        assert result is None or result.status_code == 400

    def test_delete_rate_limiting(self, mock_ctx):
        """Should rate limit DELETE requests."""
        mock_continuum = Mock()
        mock_continuum.delete.return_value = True
        mock_ctx["continuum_memory"] = mock_continuum
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)

        # Make more than 10 requests from same IP
        for i in range(15):
            mock_handler = self._make_auth_handler(client_ip="192.168.1.100")

            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_auth_ctx = Mock()
                mock_auth_ctx.is_authenticated = True
                mock_extract.return_value = mock_auth_ctx
                result = handler.handle_delete(f"/api/memory/continuum/mem{i}", {}, mock_handler)

            # After 10 requests, should get rate limited
            if i >= 10:
                if result is not None:
                    assert result.status_code == 429

    def test_delete_returns_none_for_wrong_path(self, mock_ctx):
        """Should return None for non-matching paths."""
        mock_ctx["user_store"] = Mock()
        handler = MemoryHandler(mock_ctx)
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = handler.handle_delete("/api/memory/wrong/path", {}, mock_handler)

        assert result is None
