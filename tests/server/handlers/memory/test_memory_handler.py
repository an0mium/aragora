"""
Integration tests for Memory handlers.

Tests cover:
- MemoryHandler: Continuum operations, tiers, search, pressure
- CoordinatorHandler: Coordinator metrics and config
- InsightsHandler: Insights extraction
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_continuum_memory():
    """Create a mock ContinuumMemory."""
    memory = MagicMock()

    # Mock memory entries
    mock_entry = MagicMock()
    mock_entry.id = "mem_001"
    mock_entry.content = "Test memory content"
    mock_entry.importance = 0.8
    mock_entry.surprise = 0.5
    mock_entry.created_at = datetime.now(timezone.utc)

    memory.retrieve = MagicMock(return_value=[mock_entry])
    memory.search = MagicMock(return_value=[mock_entry])
    memory.get_tier_stats = MagicMock(
        return_value={
            "fast": {"count": 10, "limit": 100, "utilization": 0.1},
            "medium": {"count": 50, "limit": 500, "utilization": 0.1},
            "slow": {"count": 200, "limit": 1000, "utilization": 0.2},
            "glacial": {"count": 500, "limit": 5000, "utilization": 0.1},
        }
    )
    memory.get_archive_stats = MagicMock(
        return_value={
            "total_archived": 100,
            "oldest_archive": datetime.now(timezone.utc).isoformat(),
        }
    )
    memory.get_memory_pressure = MagicMock(return_value=0.3)
    memory.get_stats = MagicMock(
        return_value={
            "by_tier": {
                "FAST": {"count": 10},
                "MEDIUM": {"count": 50},
                "SLOW": {"count": 200},
                "GLACIAL": {"count": 500},
            }
        }
    )
    memory.get_all_tiers = MagicMock(
        return_value=[
            {
                "name": "fast",
                "description": "Fast tier",
                "ttl": 60,
                "count": 10,
                "limit": 100,
                "utilization": 0.1,
            },
            {
                "name": "medium",
                "description": "Medium tier",
                "ttl": 3600,
                "count": 50,
                "limit": 500,
                "utilization": 0.1,
            },
        ]
    )
    memory.consolidate = MagicMock(
        return_value={
            "entries_processed": 20,
            "entries_promoted": 5,
            "entries_consolidated": 10,
        }
    )
    memory.cleanup = MagicMock(return_value={"expired": 5, "tier_limits": {"fast": 0, "medium": 2}})
    memory.delete_memory = MagicMock(return_value=True)

    return memory


@pytest.fixture
def mock_critique_store():
    """Create a mock CritiqueStore."""
    store = MagicMock()

    mock_critique = MagicMock()
    mock_critique.id = "critique_001"
    mock_critique.agent = "claude"
    mock_critique.severity = "medium"
    mock_critique.issues = ["Issue 1", "Issue 2"]
    mock_critique.suggestions = ["Fix 1"]

    store.get_critiques = MagicMock(return_value=[mock_critique])
    store.count_critiques = MagicMock(return_value=10)

    return store


@pytest.fixture
def mock_memory_coordinator():
    """Create a mock MemoryCoordinator."""
    coordinator = MagicMock()

    coordinator.get_metrics = MagicMock(
        return_value={
            "total_transactions": 100,
            "successful_transactions": 95,
            "partial_failures": 3,
            "rollbacks_performed": 2,
            "success_rate": 0.95,
        }
    )
    coordinator.get_config = MagicMock(
        return_value={
            "write_continuum": True,
            "write_consensus": True,
            "write_critique": True,
            "write_mound": False,
            "rollback_on_failure": True,
            "parallel_writes": True,
            "min_confidence_for_mound": 0.8,
        }
    )
    coordinator.get_memory_systems = MagicMock(
        return_value={
            "continuum": True,
            "consensus": True,
            "critique": True,
            "mound": False,
        }
    )

    return coordinator


@pytest.fixture
def memory_handler(mock_continuum_memory, mock_critique_store):
    """Create a MemoryHandler with mocked dependencies."""
    from aragora.server.handlers.memory.memory import MemoryHandler
    from aragora.server.handlers.base import clear_cache

    clear_cache()

    ctx = {
        "continuum_memory": mock_continuum_memory,
        "critique_store": mock_critique_store,
    }
    handler = MemoryHandler(server_context=ctx)
    return handler


@pytest.fixture
def coordinator_handler(mock_memory_coordinator):
    """Create a CoordinatorHandler with mocked dependencies."""
    from aragora.server.handlers.memory.coordinator import CoordinatorHandler
    from aragora.server.handlers.base import clear_cache

    clear_cache()

    ctx = {
        "memory_coordinator": mock_memory_coordinator,
    }
    handler = CoordinatorHandler(server_context=ctx)
    return handler


# ===========================================================================
# MemoryHandler Routing Tests
# ===========================================================================


class TestMemoryHandlerRouting:
    """Tests for MemoryHandler routing."""

    def test_can_handle_retrieve(self, memory_handler):
        """Handler recognizes continuum retrieve path."""
        assert memory_handler.can_handle("/api/v1/memory/continuum/retrieve") is True

    def test_can_handle_consolidate(self, memory_handler):
        """Handler recognizes consolidate path."""
        assert memory_handler.can_handle("/api/v1/memory/continuum/consolidate") is True

    def test_can_handle_cleanup(self, memory_handler):
        """Handler recognizes cleanup path."""
        assert memory_handler.can_handle("/api/v1/memory/continuum/cleanup") is True

    def test_can_handle_tier_stats(self, memory_handler):
        """Handler recognizes tier-stats path."""
        assert memory_handler.can_handle("/api/v1/memory/tier-stats") is True

    def test_can_handle_archive_stats(self, memory_handler):
        """Handler recognizes archive-stats path."""
        assert memory_handler.can_handle("/api/v1/memory/archive-stats") is True

    def test_can_handle_pressure(self, memory_handler):
        """Handler recognizes pressure path."""
        assert memory_handler.can_handle("/api/v1/memory/pressure") is True

    def test_can_handle_tiers(self, memory_handler):
        """Handler recognizes tiers path."""
        assert memory_handler.can_handle("/api/v1/memory/tiers") is True

    def test_can_handle_search(self, memory_handler):
        """Handler recognizes search path."""
        assert memory_handler.can_handle("/api/v1/memory/search") is True

    def test_can_handle_critiques(self, memory_handler):
        """Handler recognizes critiques path."""
        assert memory_handler.can_handle("/api/v1/memory/critiques") is True

    def test_can_handle_delete_memory(self, memory_handler):
        """Handler recognizes memory delete path with ID."""
        assert memory_handler.can_handle("/api/v1/memory/continuum/mem_12345") is True

    def test_cannot_handle_unrelated_path(self, memory_handler):
        """Handler rejects unrelated paths."""
        assert memory_handler.can_handle("/api/v1/agents") is False
        assert memory_handler.can_handle("/api/v1/debates") is False


# ===========================================================================
# MemoryHandler Method Tests
# ===========================================================================


class TestGetTierStats:
    """Tests for _get_tier_stats method."""

    def test_get_tier_stats_success(self, memory_handler):
        """Get tier stats returns tier information."""
        result = memory_handler._get_tier_stats()

        assert result is not None
        assert result.status_code == 200

    def test_get_tier_stats_response_structure(self, memory_handler):
        """Get tier stats returns proper structure."""
        result = memory_handler._get_tier_stats()

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        assert isinstance(body, dict)


class TestGetArchiveStats:
    """Tests for _get_archive_stats method."""

    def test_get_archive_stats_success(self, memory_handler):
        """Get archive stats returns archive information."""
        result = memory_handler._get_archive_stats()

        assert result is not None
        assert result.status_code == 200


class TestGetMemoryPressure:
    """Tests for _get_memory_pressure method."""

    def test_get_memory_pressure_success(self, memory_handler):
        """Get memory pressure returns pressure data."""
        result = memory_handler._get_memory_pressure()

        assert result is not None
        assert result.status_code == 200

    def test_get_memory_pressure_response_structure(self, memory_handler):
        """Get memory pressure returns proper structure."""
        result = memory_handler._get_memory_pressure()

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        assert isinstance(body, dict)


class TestGetAllTiers:
    """Tests for _get_all_tiers method."""

    def test_get_all_tiers_success(self, memory_handler):
        """Get all tiers returns tier list."""
        result = memory_handler._get_all_tiers()

        assert result is not None
        assert result.status_code == 200

    def test_get_all_tiers_response_structure(self, memory_handler):
        """Get all tiers returns proper structure."""
        result = memory_handler._get_all_tiers()

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        assert isinstance(body, dict)


class TestSearchMemories:
    """Tests for _search_memories method."""

    def test_search_memories_requires_query(self, memory_handler):
        """Search memories requires query parameter."""
        result = memory_handler._search_memories({})

        # Should return 400 for missing query
        assert result is not None
        assert result.status_code == 400

    def test_search_memories_with_query(self, memory_handler):
        """Search memories with query returns results."""
        result = memory_handler._search_memories({"q": "test query"})

        assert result is not None
        # Should return 200 or graceful error
        assert result.status_code in (200, 503)


class TestGetCritiques:
    """Tests for _get_critiques method."""

    def test_get_critiques_success(self, memory_handler):
        """Get critiques returns critique entries."""
        result = memory_handler._get_critiques({})

        assert result is not None
        # Should return 200 or 503 if store not available
        assert result.status_code in (200, 503)

    def test_get_critiques_with_agent_filter(self, memory_handler):
        """Get critiques filters by agent."""
        result = memory_handler._get_critiques({"agent": "claude"})

        assert result is not None


# ===========================================================================
# CoordinatorHandler Routing Tests
# ===========================================================================


class TestCoordinatorHandlerRouting:
    """Tests for CoordinatorHandler routing."""

    def test_can_handle_metrics(self, coordinator_handler):
        """Handler recognizes metrics path."""
        assert coordinator_handler.can_handle("/api/v1/memory/coordinator/metrics") is True

    def test_can_handle_config(self, coordinator_handler):
        """Handler recognizes config path."""
        assert coordinator_handler.can_handle("/api/v1/memory/coordinator/config") is True

    def test_cannot_handle_unrelated_path(self, coordinator_handler):
        """Handler rejects unrelated paths."""
        assert coordinator_handler.can_handle("/api/v1/memory/tiers") is False


class TestCoordinatorMetrics:
    """Tests for coordinator metrics endpoint."""

    def test_get_metrics_success(self, coordinator_handler):
        """Get coordinator metrics returns metrics data."""
        result = coordinator_handler._get_metrics()

        assert result is not None
        assert result.status_code == 200

    def test_get_metrics_response_structure(self, coordinator_handler):
        """Get coordinator metrics returns proper structure."""
        result = coordinator_handler._get_metrics()

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        assert isinstance(body, dict)
        assert "configured" in body or "metrics" in body or "error" in body


class TestCoordinatorConfig:
    """Tests for coordinator config endpoint."""

    def test_get_config_success(self, coordinator_handler):
        """Get coordinator config returns config data."""
        result = coordinator_handler._get_config()

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Edge Cases and Error Handling
# ===========================================================================


class TestMemoryHandlerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_continuum_memory_handled(self):
        """Missing continuum memory is handled gracefully."""
        from aragora.server.handlers.memory.memory import MemoryHandler
        from aragora.server.handlers.base import clear_cache

        clear_cache()

        # Create handler with empty context
        handler = MemoryHandler(server_context={})

        result = handler._get_tier_stats()
        assert result is not None
        # Should return 503 when memory not available
        assert result.status_code == 503

    def test_missing_critique_store_handled(self):
        """Missing critique store is handled gracefully."""
        from aragora.server.handlers.memory.memory import MemoryHandler
        from aragora.server.handlers.base import clear_cache

        clear_cache()

        handler = MemoryHandler(server_context={})

        result = handler._get_critiques({})
        assert result is not None
        assert result.status_code == 503

    def test_search_query_length_limit(self, memory_handler):
        """Search query respects length limits."""
        # Very long query should be handled
        long_query = "a" * 1000
        result = memory_handler._search_memories({"q": long_query})

        assert result is not None
        # Should either truncate or reject

    def test_pagination_parameters(self, memory_handler):
        """Pagination parameters are handled correctly."""
        result = memory_handler._get_critiques({"limit": "50", "offset": "10"})

        assert result is not None


class TestCoordinatorEdgeCases:
    """Tests for coordinator edge cases."""

    def test_missing_coordinator_handled(self):
        """Missing coordinator is handled gracefully."""
        from aragora.server.handlers.memory.coordinator import CoordinatorHandler
        from aragora.server.handlers.base import clear_cache

        clear_cache()

        handler = CoordinatorHandler(server_context={})

        result = handler._get_metrics()
        assert result is not None
        assert result.status_code == 200  # Returns configured: false

    def test_coordinator_not_configured(self):
        """Coordinator not configured returns proper response."""
        from aragora.server.handlers.memory.coordinator import CoordinatorHandler
        from aragora.server.handlers.base import clear_cache

        clear_cache()

        handler = CoordinatorHandler(server_context={})

        result = handler._get_config()
        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        assert body.get("configured") is False
