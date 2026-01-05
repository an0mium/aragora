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
                "/api/memory/continuum/retrieve",
                {"query": ["test query"]},
                None
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
                None
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
                "/api/memory/continuum/retrieve",
                {"min_importance": ["0.7"]},
                None
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
                "/api/memory/continuum/retrieve",
                {"tiers": ["fast,slow"]},
                None
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

    def test_consolidate_no_continuum(self, handler):
        """Returns 503 when continuum memory not configured."""
        result = handler.handle("/api/memory/continuum/consolidate", {}, None)
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
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/continuum/consolidate", {}, None)

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
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/continuum/consolidate", {}, None)

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
        """Returns 503 when continuum system unavailable."""
        mock_ctx["continuum_memory"] = None
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/continuum/consolidate", {}, None)
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
        """Consolidate handles exceptions gracefully."""
        mock_continuum = Mock()
        mock_continuum.consolidate.side_effect = Exception("Test error")
        mock_ctx["continuum_memory"] = mock_continuum
        handler = MemoryHandler(mock_ctx)

        result = handler.handle("/api/memory/continuum/consolidate", {}, None)

        assert result.status_code == 500
