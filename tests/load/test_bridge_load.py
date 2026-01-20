"""Load Tests for Phase 9 Cross-Pollination Bridges.

Tests bridge performance under simulated high-load conditions to ensure
they can handle production workloads without degradation.

Run with:
    pytest tests/load/test_bridge_load.py -v --benchmark-enable

Or with stress testing:
    pytest tests/load/test_bridge_load.py -v -k stress --benchmark-disable
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# =============================================================================
# Mock Subsystems for Load Testing
# =============================================================================


@dataclass
class LoadTestMetrics:
    """Performance metrics for an agent."""

    avg_response_time: float = 1.0
    quality_score: float = 0.8
    consistency_score: float = 0.9
    total_calls: int = 10


class LoadTestPerformanceMonitor:
    """High-performance mock PerformanceMonitor for load testing."""

    def __init__(self, agent_count: int = 100):
        self._metrics: Dict[str, LoadTestMetrics] = {}
        # Pre-populate with agents
        for i in range(agent_count):
            self._metrics[f"agent_{i}"] = LoadTestMetrics(
                avg_response_time=0.5 + (i % 10) * 0.1,
                quality_score=0.6 + (i % 5) * 0.08,
                consistency_score=0.7 + (i % 4) * 0.05,
                total_calls=10 + i,
            )

    def get_agent_metrics(self, agent_name: str) -> Optional[LoadTestMetrics]:
        return self._metrics.get(agent_name)


@dataclass
class LoadTestRelationshipMetrics:
    """Relationship metrics between agents."""

    alliance_score: float = 0.5
    rivalry_score: float = 0.3
    agreement_rate: float = 0.6
    debate_count: int = 10


class LoadTestRelationshipTracker:
    """High-performance mock RelationshipTracker for load testing."""

    def __init__(self, agent_count: int = 100):
        self._relationships: Dict[tuple, LoadTestRelationshipMetrics] = {}
        # Pre-populate relationships between agents
        agents = [f"agent_{i}" for i in range(agent_count)]
        for i, a in enumerate(agents):
            for j, b in enumerate(agents):
                if i < j:
                    key = tuple(sorted([a, b]))
                    self._relationships[key] = LoadTestRelationshipMetrics(
                        alliance_score=0.3 + ((i + j) % 7) * 0.1,
                        agreement_rate=0.4 + ((i * j) % 5) * 0.1,
                    )

    def compute_metrics(self, agent_a: str, agent_b: str) -> Optional[LoadTestRelationshipMetrics]:
        key = tuple(sorted([agent_a, agent_b]))
        return self._relationships.get(key)


@dataclass
class LoadTestCalibrationSummary:
    """Calibration summary for an agent."""

    agent: str = ""
    total_predictions: int = 100
    total_correct: int = 85
    ece: float = 0.08
    brier_score: float = 0.1
    is_overconfident: bool = False
    is_underconfident: bool = False

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_correct / self.total_predictions


class LoadTestCalibrationTracker:
    """High-performance mock CalibrationTracker for load testing."""

    def __init__(self, agent_count: int = 100):
        self._summaries: Dict[str, LoadTestCalibrationSummary] = {}
        for i in range(agent_count):
            self._summaries[f"agent_{i}"] = LoadTestCalibrationSummary(
                agent=f"agent_{i}",
                total_predictions=50 + i * 2,
                total_correct=40 + i,
                ece=0.05 + (i % 10) * 0.01,
            )

    def get_calibration_summary(self, agent_name: str) -> Optional[LoadTestCalibrationSummary]:
        return self._summaries.get(agent_name)

    def get_all_agents(self) -> List[str]:
        return list(self._summaries.keys())


# =============================================================================
# Load Test Fixtures
# =============================================================================


@pytest.fixture
def performance_monitor():
    """Create a high-performance mock PerformanceMonitor."""
    return LoadTestPerformanceMonitor(agent_count=100)


@pytest.fixture
def relationship_tracker():
    """Create a high-performance mock RelationshipTracker."""
    return LoadTestRelationshipTracker(agent_count=100)


@pytest.fixture
def calibration_tracker():
    """Create a high-performance mock CalibrationTracker."""
    return LoadTestCalibrationTracker(agent_count=100)


@pytest.fixture
def available_agents():
    """List of available agents for testing."""
    return [f"agent_{i}" for i in range(100)]


# =============================================================================
# PerformanceRouterBridge Load Tests
# =============================================================================


class TestPerformanceRouterBridgeLoad:
    """Load tests for PerformanceRouterBridge."""

    def test_routing_score_computation_throughput(self, performance_monitor, benchmark):
        """Test routing score computation throughput."""
        from aragora.debate.performance_router_bridge import PerformanceRouterBridge

        bridge = PerformanceRouterBridge(performance_monitor=performance_monitor)

        def compute_scores():
            for i in range(100):
                bridge.compute_routing_score(f"agent_{i}", "balanced")

        # Benchmark the computation
        result = benchmark(compute_scores)

        # Should complete in reasonable time
        assert result is None  # benchmark returns None

    def test_best_agent_selection_throughput(
        self, performance_monitor, available_agents, benchmark
    ):
        """Test best agent selection throughput."""
        from aragora.debate.performance_router_bridge import PerformanceRouterBridge

        bridge = PerformanceRouterBridge(performance_monitor=performance_monitor)

        def select_best():
            return bridge.get_best_agent_for_task(available_agents, "speed")

        result = benchmark(select_best)
        assert result is None or result.startswith("agent_")

    def test_agent_ranking_throughput(self, performance_monitor, available_agents, benchmark):
        """Test agent ranking throughput."""
        from aragora.debate.performance_router_bridge import PerformanceRouterBridge

        bridge = PerformanceRouterBridge(performance_monitor=performance_monitor)

        def rank_agents():
            return bridge.rank_agents_for_task(available_agents, "precision")

        result = benchmark(rank_agents)
        assert len(result) == 100

    def test_concurrent_score_computation(self, performance_monitor, available_agents):
        """Test concurrent routing score computation."""
        from aragora.debate.performance_router_bridge import PerformanceRouterBridge

        bridge = PerformanceRouterBridge(performance_monitor=performance_monitor)

        results = []
        errors = []

        def compute_score(agent_name: str):
            try:
                score = bridge.compute_routing_score(agent_name, "balanced")
                results.append(score)
            except Exception as e:
                errors.append(e)

        # Run concurrent computations
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(compute_score, agent) for agent in available_agents]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 100

    def test_stress_routing_decisions(self, performance_monitor):
        """Stress test with many rapid routing decisions."""
        from aragora.debate.performance_router_bridge import PerformanceRouterBridge

        bridge = PerformanceRouterBridge(performance_monitor=performance_monitor)
        agents = [f"agent_{i}" for i in range(10)]

        start_time = time.perf_counter()
        iterations = 1000

        for _ in range(iterations):
            bridge.get_best_agent_for_task(agents, "speed")

        elapsed = time.perf_counter() - start_time
        ops_per_second = iterations / elapsed

        # Should handle at least 500 operations per second
        assert ops_per_second > 500, f"Only {ops_per_second:.0f} ops/sec, expected > 500"


# =============================================================================
# RelationshipBiasBridge Load Tests
# =============================================================================


class TestRelationshipBiasBridgeLoad:
    """Load tests for RelationshipBiasBridge."""

    def test_echo_chamber_detection_throughput(self, relationship_tracker, benchmark):
        """Test echo chamber detection throughput."""
        from aragora.debate.relationship_bias_bridge import RelationshipBiasBridge

        bridge = RelationshipBiasBridge(relationship_tracker=relationship_tracker)

        team = ["agent_0", "agent_1", "agent_2", "agent_3"]

        def detect_echo_chamber():
            return bridge.compute_team_echo_risk(team)

        result = benchmark(detect_echo_chamber)
        assert result.overall_risk >= 0

    def test_concurrent_echo_chamber_detection(self, relationship_tracker):
        """Test concurrent echo chamber detection."""
        from aragora.debate.relationship_bias_bridge import RelationshipBiasBridge

        bridge = RelationshipBiasBridge(relationship_tracker=relationship_tracker)

        results = []
        errors = []

        def detect_risk(team_id: int):
            try:
                team = [f"agent_{i}" for i in range(team_id, team_id + 4)]
                risk = bridge.compute_team_echo_risk(team)
                results.append(risk)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(detect_risk, i) for i in range(50)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 50

    def test_stress_bias_detection(self, relationship_tracker):
        """Stress test with rapid bias detection calls."""
        from aragora.debate.relationship_bias_bridge import RelationshipBiasBridge

        bridge = RelationshipBiasBridge(relationship_tracker=relationship_tracker)
        team = ["agent_0", "agent_1", "agent_2"]

        start_time = time.perf_counter()
        iterations = 500

        for _ in range(iterations):
            bridge.compute_team_echo_risk(team)

        elapsed = time.perf_counter() - start_time
        ops_per_second = iterations / elapsed

        assert ops_per_second > 100, f"Only {ops_per_second:.0f} ops/sec, expected > 100"


# =============================================================================
# CalibrationCostBridge Load Tests
# =============================================================================


class TestCalibrationCostBridgeLoad:
    """Load tests for CalibrationCostBridge."""

    def test_cost_efficiency_throughput(self, calibration_tracker, benchmark):
        """Test cost efficiency calculation throughput."""
        from aragora.billing.calibration_cost_bridge import CalibrationCostBridge

        bridge = CalibrationCostBridge(calibration_tracker=calibration_tracker)

        def compute_efficiency():
            for i in range(100):
                bridge.compute_cost_efficiency(f"agent_{i}")

        benchmark(compute_efficiency)

    def test_budget_filtering_throughput(self, calibration_tracker, available_agents, benchmark):
        """Test budget-aware filtering throughput."""
        from aragora.billing.calibration_cost_bridge import CalibrationCostBridge

        bridge = CalibrationCostBridge(calibration_tracker=calibration_tracker)

        def filter_by_budget():
            return bridge.get_budget_aware_selection(
                available_agents=available_agents[:20],
                budget_remaining=Decimal("1.00"),
            )

        result = benchmark(filter_by_budget)
        assert isinstance(result, list)

    def test_concurrent_efficiency_calculation(self, calibration_tracker, available_agents):
        """Test concurrent cost efficiency calculations."""
        from aragora.billing.calibration_cost_bridge import CalibrationCostBridge

        bridge = CalibrationCostBridge(calibration_tracker=calibration_tracker)

        results = []
        errors = []

        def calculate_efficiency(agent_name: str):
            try:
                eff = bridge.compute_cost_efficiency(agent_name)
                results.append(eff)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(calculate_efficiency, agent) for agent in available_agents]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 100


# =============================================================================
# Bridge Telemetry Load Tests
# =============================================================================


class TestBridgeTelemetryLoad:
    """Load tests for bridge telemetry system."""

    def test_telemetry_recording_throughput(self, benchmark):
        """Test telemetry recording throughput."""
        from aragora.debate.bridge_telemetry import (
            record_bridge_operation,
            reset_bridge_telemetry,
        )

        reset_bridge_telemetry()

        def record_operations():
            for i in range(100):
                record_bridge_operation(
                    bridge_name=f"bridge_{i % 7}",
                    operation="sync",
                    success=True,
                    duration_ms=10.0 + i * 0.1,
                )

        benchmark(record_operations)

    def test_concurrent_telemetry_recording(self):
        """Test concurrent telemetry recording."""
        from aragora.debate.bridge_telemetry import (
            record_bridge_operation,
            get_bridge_telemetry_stats,
            reset_bridge_telemetry,
        )

        reset_bridge_telemetry()

        errors = []

        def record_ops(thread_id: int):
            try:
                for i in range(50):
                    record_bridge_operation(
                        bridge_name=f"bridge_{thread_id}",
                        operation="op",
                        success=True,
                        duration_ms=5.0,
                    )
            except Exception as e:
                errors.append(e)

        # Run 20 threads, each recording 50 operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(record_ops, i) for i in range(20)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors: {errors}"

        stats = get_bridge_telemetry_stats()
        assert stats["total_operations"] == 1000  # 20 threads * 50 operations

    def test_stats_retrieval_under_load(self):
        """Test stats retrieval while operations are being recorded."""
        from aragora.debate.bridge_telemetry import (
            record_bridge_operation,
            get_bridge_telemetry_stats,
            reset_bridge_telemetry,
        )

        reset_bridge_telemetry()

        stop_event = threading.Event()
        stats_results = []
        errors = []

        def record_loop():
            i = 0
            while not stop_event.is_set():
                record_bridge_operation(
                    bridge_name="load_bridge",
                    operation="op",
                    success=True,
                    duration_ms=1.0,
                )
                i += 1

        def read_stats():
            while not stop_event.is_set():
                try:
                    stats = get_bridge_telemetry_stats()
                    stats_results.append(stats)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.01)

        # Start recording and reading threads
        record_thread = threading.Thread(target=record_loop)
        read_thread = threading.Thread(target=read_stats)

        record_thread.start()
        read_thread.start()

        # Let it run for a short time
        time.sleep(0.5)
        stop_event.set()

        record_thread.join(timeout=1.0)
        read_thread.join(timeout=1.0)

        assert len(errors) == 0, f"Errors during stats retrieval: {errors}"
        assert len(stats_results) > 0, "No stats were retrieved"


# =============================================================================
# SubsystemCoordinator Load Tests
# =============================================================================


class TestSubsystemCoordinatorLoad:
    """Load tests for SubsystemCoordinator with multiple bridges."""

    def test_coordinator_initialization_time(
        self, performance_monitor, relationship_tracker, calibration_tracker, benchmark
    ):
        """Test coordinator initialization time with all bridges enabled."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator

        def init_coordinator():
            return SubsystemCoordinator(
                performance_monitor=performance_monitor,
                relationship_tracker=relationship_tracker,
                calibration_tracker=calibration_tracker,
                enable_performance_router=True,
                enable_relationship_bias=True,
                enable_calibration_cost=True,
            )

        result = benchmark(init_coordinator)
        assert result.active_bridges_count == 3

    def test_coordinator_status_throughput(
        self, performance_monitor, relationship_tracker, calibration_tracker, benchmark
    ):
        """Test coordinator status retrieval throughput."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator

        coordinator = SubsystemCoordinator(
            performance_monitor=performance_monitor,
            relationship_tracker=relationship_tracker,
            calibration_tracker=calibration_tracker,
            enable_performance_router=True,
            enable_relationship_bias=True,
            enable_calibration_cost=True,
        )

        def get_status():
            return coordinator.get_status()

        result = benchmark(get_status)
        assert result["active_bridges_count"] == 3


# =============================================================================
# Memory and Resource Tests
# =============================================================================


class TestBridgeMemoryUsage:
    """Tests for bridge memory usage under load."""

    def test_telemetry_memory_bounds(self):
        """Test that telemetry memory is bounded."""
        from aragora.debate.bridge_telemetry import (
            record_bridge_operation,
            get_recent_bridge_operations,
            reset_bridge_telemetry,
        )

        reset_bridge_telemetry()

        # Record many operations
        for i in range(1000):
            record_bridge_operation(
                bridge_name=f"bridge_{i % 10}",
                operation="op",
                success=True,
                duration_ms=1.0,
            )

        # Recent operations should be bounded
        recent = get_recent_bridge_operations(limit=100)
        assert len(recent) == 100

    def test_routing_cache_bounds(self, performance_monitor):
        """Test that routing score cache doesn't grow unbounded."""
        from aragora.debate.performance_router_bridge import PerformanceRouterBridge

        bridge = PerformanceRouterBridge(performance_monitor=performance_monitor)

        # Compute scores for many agents
        for i in range(1000):
            bridge.compute_routing_score(f"agent_{i % 100}", "balanced")

        # Cache should be bounded by unique agents
        assert len(bridge.get_all_scores()) <= 100
