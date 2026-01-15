"""Tests for cache invalidation system.

Tests the CACHE_INVALIDATION_MAP registry and invalidate_cache() function
that ensures stale data is cleared from caches after mutation operations.
"""

import pytest
from unittest.mock import patch, MagicMock

from aragora.server.handlers.base import (
    BoundedTTLCache,
    CACHE_INVALIDATION_MAP,
    invalidate_cache,
    clear_cache,
    ttl_cache,
)


class TestCacheInvalidationMap:
    """Tests for CACHE_INVALIDATION_MAP registry.

    CACHE_INVALIDATION_MAP uses event-based keys (e.g., 'elo_updated') not
    data-source keys (e.g., 'elo'). The invalidate_cache() function maps
    data-source names to event names internally.
    """

    def test_contains_elo_updated_event(self) -> None:
        """Test elo_updated event is defined."""
        assert "elo_updated" in CACHE_INVALIDATION_MAP
        prefixes = CACHE_INVALIDATION_MAP["elo_updated"]
        assert isinstance(prefixes, list)
        assert len(prefixes) > 0

    def test_contains_memory_updated_event(self) -> None:
        """Test memory_updated event is defined."""
        assert "memory_updated" in CACHE_INVALIDATION_MAP
        prefixes = CACHE_INVALIDATION_MAP["memory_updated"]
        assert isinstance(prefixes, list)
        assert len(prefixes) > 0

    def test_contains_consensus_reached_event(self) -> None:
        """Test consensus_reached event is defined."""
        assert "consensus_reached" in CACHE_INVALIDATION_MAP
        prefixes = CACHE_INVALIDATION_MAP["consensus_reached"]
        assert isinstance(prefixes, list)
        assert len(prefixes) > 0

    def test_contains_debate_completed_event(self) -> None:
        """Test debate_completed event is defined."""
        assert "debate_completed" in CACHE_INVALIDATION_MAP
        prefixes = CACHE_INVALIDATION_MAP["debate_completed"]
        assert isinstance(prefixes, list)
        assert len(prefixes) > 0

    def test_contains_agent_updated_event(self) -> None:
        """Test agent_updated event is defined."""
        assert "agent_updated" in CACHE_INVALIDATION_MAP
        prefixes = CACHE_INVALIDATION_MAP["agent_updated"]
        assert isinstance(prefixes, list)
        assert len(prefixes) > 0

    def test_elo_updated_prefixes_include_expected(self) -> None:
        """Test elo_updated prefixes include expected cache keys."""
        prefixes = CACHE_INVALIDATION_MAP["elo_updated"]
        # These are the main caches used by leaderboard endpoints
        assert "lb_rankings" in prefixes
        assert "leaderboard" in prefixes
        assert "agent_profile" in prefixes

    def test_consensus_reached_prefixes_include_expected(self) -> None:
        """Test consensus_reached prefixes include expected cache keys."""
        prefixes = CACHE_INVALIDATION_MAP["consensus_reached"]
        assert "consensus_similar" in prefixes
        assert "consensus_settled" in prefixes
        assert "consensus_stats" in prefixes

    def test_memory_updated_prefixes_include_expected(self) -> None:
        """Test memory_updated prefixes include expected cache keys."""
        prefixes = CACHE_INVALIDATION_MAP["memory_updated"]
        # Memory events clear analytics and critique caches
        assert "analytics_memory" in prefixes
        assert "critique_patterns" in prefixes

    def test_all_prefixes_are_strings(self) -> None:
        """Test all prefixes in all data sources are strings."""
        for data_source, prefixes in CACHE_INVALIDATION_MAP.items():
            for prefix in prefixes:
                assert isinstance(prefix, str), f"Prefix in {data_source} is not a string: {prefix}"
                assert len(prefix) > 0, f"Empty prefix in {data_source}"


class TestInvalidateCache:
    """Tests for invalidate_cache() function."""

    def setup_method(self) -> None:
        """Clear global cache before each test."""
        clear_cache()

    def teardown_method(self) -> None:
        """Clear global cache after each test."""
        clear_cache()

    def test_clears_matching_prefixes(self) -> None:
        """Test invalidate_cache clears entries with matching prefixes."""

        # Create some cached entries with ELO-related prefixes
        @ttl_cache(ttl_seconds=300, key_prefix="lb_rankings", skip_first=False)
        def get_rankings():
            return ["agent1", "agent2"]

        @ttl_cache(ttl_seconds=300, key_prefix="leaderboard", skip_first=False)
        def get_leaderboard():
            return {"entries": []}

        # Populate caches
        get_rankings()
        get_leaderboard()

        # Invalidate ELO caches
        cleared = invalidate_cache("elo")

        # Should have cleared entries
        assert cleared >= 2

    def test_does_not_clear_unrelated_prefixes(self) -> None:
        """Test invalidate_cache does not affect unrelated caches."""

        # Create cached entries for different data sources
        @ttl_cache(ttl_seconds=300, key_prefix="consensus_stats", skip_first=False)
        def get_consensus():
            return {"count": 10}

        @ttl_cache(ttl_seconds=300, key_prefix="lb_rankings", skip_first=False)
        def get_rankings():
            return ["agent1"]

        # Populate caches
        get_consensus()
        get_rankings()

        # Invalidate only ELO caches
        invalidate_cache("elo")

        # Consensus cache should still hit
        call_count = 0

        @ttl_cache(ttl_seconds=300, key_prefix="consensus_stats", skip_first=False)
        def get_consensus_again():
            nonlocal call_count
            call_count += 1
            return {"count": 20}

        # The original consensus entry uses different function name in key,
        # so we test by creating a fresh entry and checking it exists
        # Actually let's just verify clear_cache with prefix works correctly
        from aragora.server.handlers.base import _cache

        _cache.set("consensus_stats:test", {"data": 1})
        _cache.set("lb_rankings:test", {"data": 2})

        invalidate_cache("elo")

        # Consensus entry should still exist
        hit, value = _cache.get("consensus_stats:test", 300)
        assert hit is True
        assert value == {"data": 1}

    def test_unknown_data_source_returns_zero(self) -> None:
        """Test invalidate_cache with unknown data source clears nothing."""
        from aragora.server.handlers.base import _cache

        _cache.set("lb_rankings:test", {"data": 1})

        cleared = invalidate_cache("unknown_source")

        assert cleared == 0
        # Original entry should still exist
        hit, value = _cache.get("lb_rankings:test", 300)
        assert hit is True

    def test_returns_total_cleared_count(self) -> None:
        """Test invalidate_cache returns count of cleared entries."""
        from aragora.server.handlers.base import _cache

        # Add entries with consensus-related prefixes
        _cache.set("consensus_similar:key1", "value1")
        _cache.set("consensus_settled:key2", "value2")
        _cache.set("consensus_stats:key3", "value3")

        cleared = invalidate_cache("consensus")

        assert cleared == 3

    def test_empty_cache_returns_zero(self) -> None:
        """Test invalidate_cache on empty cache returns zero."""
        clear_cache()  # Ensure empty

        cleared = invalidate_cache("elo")

        assert cleared == 0


class TestCacheInvalidationIntegration:
    """Integration tests for cache invalidation in mutation functions.

    These tests verify that the invalidation code exists in the right places,
    without actually running the full mutation operations.
    """

    def test_elo_imports_invalidate_cache(self) -> None:
        """Test elo.py can import invalidate_cache."""
        # This verifies the import path is correct
        try:
            from aragora.server.handlers.base import invalidate_cache

            assert callable(invalidate_cache)
        except ImportError:
            pytest.fail("Could not import invalidate_cache from base")

    def test_invalidate_cache_is_exported(self) -> None:
        """Test invalidate_cache is in __all__ exports."""
        from aragora.server.handlers import base

        assert "invalidate_cache" in base.__all__

    def test_cache_invalidation_map_is_exported(self) -> None:
        """Test CACHE_INVALIDATION_MAP is in __all__ exports."""
        from aragora.server.handlers import base

        assert "CACHE_INVALIDATION_MAP" in base.__all__


class TestCacheClearByPrefix:
    """Tests for clear_cache with prefix filtering."""

    def setup_method(self) -> None:
        """Clear global cache before each test."""
        clear_cache()

    def teardown_method(self) -> None:
        """Clear global cache after each test."""
        clear_cache()

    def test_clears_exact_prefix_match(self) -> None:
        """Test clear_cache clears entries starting with prefix."""
        from aragora.server.handlers.base import _cache

        _cache.set("user:1", "alice")
        _cache.set("user:2", "bob")
        _cache.set("item:1", "widget")

        cleared = clear_cache("user:")

        assert cleared == 2
        # Verify item entry still exists
        hit, value = _cache.get("item:1", 300)
        assert hit is True

    def test_clears_all_with_no_prefix(self) -> None:
        """Test clear_cache with no prefix clears everything."""
        from aragora.server.handlers.base import _cache

        _cache.set("key1", "value1")
        _cache.set("key2", "value2")
        _cache.set("key3", "value3")

        cleared = clear_cache(None)

        assert cleared == 3
        assert len(_cache) == 0

    def test_prefix_is_case_sensitive(self) -> None:
        """Test prefix matching is case-sensitive."""
        from aragora.server.handlers.base import _cache

        _cache.set("User:1", "alice")
        _cache.set("user:2", "bob")

        cleared = clear_cache("user:")

        assert cleared == 1
        # Uppercase entry should still exist
        hit, value = _cache.get("User:1", 300)
        assert hit is True


class TestInvalidateCacheLogging:
    """Tests for cache invalidation logging."""

    def setup_method(self) -> None:
        """Clear global cache before each test."""
        clear_cache()

    def teardown_method(self) -> None:
        """Clear global cache after each test."""
        clear_cache()

    def test_logs_when_entries_cleared(self) -> None:
        """Test debug log is emitted when entries are cleared."""
        from aragora.server.handlers.base import _cache

        _cache.set("lb_rankings:test", "value")

        with patch("aragora.server.handlers.admin.cache.logger") as mock_logger:
            invalidate_cache("elo")
            # Should have logged debug message
            mock_logger.debug.assert_called()

    def test_no_log_when_nothing_cleared(self) -> None:
        """Test no log when no entries cleared."""
        clear_cache()  # Ensure empty

        with patch("aragora.server.handlers.admin.cache.logger") as mock_logger:
            invalidate_cache("elo")
            # Should not have logged (nothing cleared)
            mock_logger.debug.assert_not_called()


class TestAllDataSourcePrefixes:
    """Verify all data sources have valid prefix configurations."""

    def test_all_data_sources_have_prefixes(self) -> None:
        """Test every data source in the map has at least one prefix."""
        for data_source, prefixes in CACHE_INVALIDATION_MAP.items():
            assert len(prefixes) > 0, f"Data source '{data_source}' has no prefixes"

    def test_no_duplicate_prefixes_within_source(self) -> None:
        """Test no duplicate prefixes within a single data source."""
        for data_source, prefixes in CACHE_INVALIDATION_MAP.items():
            unique = set(prefixes)
            assert len(unique) == len(prefixes), f"Duplicate prefixes in '{data_source}'"

    def test_prefixes_follow_naming_convention(self) -> None:
        """Test prefixes follow expected naming conventions."""
        for data_source, prefixes in CACHE_INVALIDATION_MAP.items():
            for prefix in prefixes:
                # Prefixes should be lowercase with underscores
                assert prefix == prefix.lower(), f"Prefix '{prefix}' should be lowercase"
                assert " " not in prefix, f"Prefix '{prefix}' should not contain spaces"
