"""Tests for PerformanceRouterBridge."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock

from aragora.ml.performance_router_bridge import (
    PerformanceRouterBridge,
    PerformanceRouterBridgeConfig,
    SyncResult,
    create_performance_router_bridge,
)


@dataclass
class MockAgentStats:
    """Mock agent stats for testing."""

    name: str = "test-agent"
    total_calls: int = 100
    success_rate: float = 85.0
    avg_duration_ms: float = 3000.0
    min_duration_ms: float = 1000.0
    max_duration_ms: float = 8000.0
    timeout_rate: float = 5.0


class MockPerformanceMonitor:
    """Mock performance monitor."""

    def __init__(self):
        self.agent_stats: Dict[str, MockAgentStats] = {}

    def add_agent(self, name: str, **kwargs) -> None:
        """Add agent stats."""
        stats = MockAgentStats(name=name, **kwargs)
        self.agent_stats[name] = stats


class MockAgentCapabilities:
    """Mock agent capabilities."""

    def __init__(self):
        self.strengths = []
        self.speed_tier = 2


class MockAgentRouter:
    """Mock agent router."""

    def __init__(self):
        self._capabilities: Dict[str, MockAgentCapabilities] = {}
        self._historical_performance: Dict[str, List[bool]] = {}

    def record_performance(self, agent_name: str, task_type: str, success: bool):
        """Record performance."""
        key = f"{agent_name}:{task_type}"
        if key not in self._historical_performance:
            self._historical_performance[key] = []
        self._historical_performance[key].append(success)


class TestPerformanceRouterBridge:
    """Tests for PerformanceRouterBridge."""

    def test_create_bridge(self):
        """Test bridge creation."""
        bridge = PerformanceRouterBridge()
        assert bridge.performance_monitor is None
        assert bridge.agent_router is None
        assert bridge.config is not None

    def test_create_with_config(self):
        """Test bridge creation with custom config."""
        config = PerformanceRouterBridgeConfig(
            min_calls_for_sync=10,
            success_rate_weight=0.5,
        )
        bridge = PerformanceRouterBridge(config=config)
        assert bridge.config.min_calls_for_sync == 10
        assert bridge.config.success_rate_weight == 0.5

    def test_sync_no_monitor(self):
        """Test sync with no monitor returns empty result."""
        bridge = PerformanceRouterBridge()
        result = bridge.sync_performance()
        assert result.agents_synced == 0
        assert result.records_added == 0

    def test_sync_no_router(self):
        """Test sync with no router returns empty result."""
        monitor = MockPerformanceMonitor()
        monitor.add_agent("claude", total_calls=100)

        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        result = bridge.sync_performance()
        assert result.agents_synced == 0

    def test_sync_performance(self):
        """Test syncing performance data."""
        monitor = MockPerformanceMonitor()
        monitor.add_agent("claude", total_calls=100, success_rate=90.0)
        monitor.add_agent("gpt-4", total_calls=50, success_rate=85.0)

        router = MockAgentRouter()
        router._capabilities["claude"] = MockAgentCapabilities()
        router._capabilities["gpt-4"] = MockAgentCapabilities()

        bridge = PerformanceRouterBridge(
            performance_monitor=monitor,
            agent_router=router,
            config=PerformanceRouterBridgeConfig(min_calls_for_sync=5),
        )

        result = bridge.sync_performance(force=True)

        assert result.agents_synced == 2
        assert result.records_added > 0
        assert "claude" in result.agents_updated

    def test_compute_agent_score(self):
        """Test computing agent score."""
        monitor = MockPerformanceMonitor()
        monitor.add_agent(
            "claude",
            total_calls=100,
            success_rate=90.0,
            avg_duration_ms=2000.0,
            timeout_rate=2.0,
            min_duration_ms=1000.0,
            max_duration_ms=4000.0,
        )

        bridge = PerformanceRouterBridge(
            performance_monitor=monitor,
            config=PerformanceRouterBridgeConfig(min_calls_for_sync=5),
        )

        score = bridge.compute_agent_score("claude")
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Good agent should score above neutral

    def test_compute_agent_score_no_data(self):
        """Test agent score with no data returns neutral."""
        bridge = PerformanceRouterBridge()
        score = bridge.compute_agent_score("unknown")
        assert score == 0.5

    def test_get_agent_scores(self):
        """Test getting scores for all agents."""
        monitor = MockPerformanceMonitor()
        monitor.add_agent("claude", total_calls=100, success_rate=90.0)
        monitor.add_agent("gpt-4", total_calls=50, success_rate=70.0)

        bridge = PerformanceRouterBridge(
            performance_monitor=monitor,
            config=PerformanceRouterBridgeConfig(min_calls_for_sync=5),
        )

        scores = bridge.get_agent_scores()
        assert "claude" in scores
        assert "gpt-4" in scores
        assert scores["claude"] > scores["gpt-4"]

    def test_auto_sync(self):
        """Test auto-sync mechanism."""
        monitor = MockPerformanceMonitor()
        monitor.add_agent("claude", total_calls=100)

        router = MockAgentRouter()
        router._capabilities["claude"] = MockAgentCapabilities()

        bridge = PerformanceRouterBridge(
            performance_monitor=monitor,
            agent_router=router,
            config=PerformanceRouterBridgeConfig(auto_sync_interval=5),
        )

        bridge.enable_auto_sync()
        assert bridge._auto_sync_enabled

        # Should not sync yet
        for _ in range(4):
            result = bridge.maybe_auto_sync()
            assert result is None

        # Should sync now
        bridge._last_sync_counts["claude"] = 50  # Simulate old sync
        result = bridge.maybe_auto_sync()
        assert result is not None

    def test_disable_auto_sync(self):
        """Test disabling auto-sync."""
        bridge = PerformanceRouterBridge()
        bridge.enable_auto_sync()
        bridge.disable_auto_sync()
        assert not bridge._auto_sync_enabled

    def test_get_sync_history(self):
        """Test getting sync history."""
        bridge = PerformanceRouterBridge()
        history = bridge.get_sync_history()
        assert isinstance(history, list)

    def test_get_stats(self):
        """Test getting bridge stats."""
        bridge = PerformanceRouterBridge()
        stats = bridge.get_stats()
        assert "auto_sync_enabled" in stats
        assert "calls_since_sync" in stats
        assert "total_syncs" in stats

    def test_factory_function(self):
        """Test factory function."""
        bridge = create_performance_router_bridge(
            auto_sync=True,
            min_calls_for_sync=10,
        )
        assert bridge._auto_sync_enabled
        assert bridge.config.min_calls_for_sync == 10

    def test_speed_tier_update(self):
        """Test speed tier update based on latency."""
        monitor = MockPerformanceMonitor()
        monitor.add_agent("fast-agent", total_calls=100, avg_duration_ms=1500.0)
        monitor.add_agent("slow-agent", total_calls=100, avg_duration_ms=15000.0)

        router = MockAgentRouter()
        router._capabilities["fast-agent"] = MockAgentCapabilities()
        router._capabilities["slow-agent"] = MockAgentCapabilities()

        bridge = PerformanceRouterBridge(
            performance_monitor=monitor,
            agent_router=router,
        )

        bridge.sync_performance(force=True)

        assert router._capabilities["fast-agent"].speed_tier == 1  # Fast
        assert router._capabilities["slow-agent"].speed_tier == 3  # Slow


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_sync_result_defaults(self):
        """Test SyncResult default values."""
        result = SyncResult(agents_synced=5, records_added=50)
        assert result.agents_synced == 5
        assert result.records_added == 50
        assert result.timestamp is not None
        assert result.agents_updated == []
