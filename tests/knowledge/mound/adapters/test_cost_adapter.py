"""
Tests for CostAdapter - Bridges Billing/Cost Tracking to Knowledge Mound.

Tests cover:
- CostAnomaly dataclass
- Adapter initialization
- Alert storage with level thresholds
- Anomaly storage with variance thresholds
- Cost snapshot storage
- Pattern analysis and anomaly detection
- Workspace queries
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from aragora.knowledge.mound.adapters.cost_adapter import (
    CostAdapter,
    CostAnomaly,
    AlertSearchResult,
)


# =============================================================================
# CostAnomaly Dataclass Tests
# =============================================================================


class TestCostAnomaly:
    """Tests for CostAnomaly dataclass."""

    def test_create_anomaly(self):
        """Should create a cost anomaly with all fields."""
        anomaly = CostAnomaly(
            id="anomaly-001",
            workspace_id="ws-123",
            agent_id="agent-456",
            anomaly_type="spike",
            severity=0.8,
            description="Cost spike detected",
            expected_value=100.0,
            actual_value=300.0,
            variance_ratio=3.0,
            detected_at=datetime.now(timezone.utc),
        )

        assert anomaly.id == "anomaly-001"
        assert anomaly.workspace_id == "ws-123"
        assert anomaly.agent_id == "agent-456"
        assert anomaly.anomaly_type == "spike"
        assert anomaly.severity == 0.8
        assert anomaly.variance_ratio == 3.0

    def test_anomaly_with_metadata(self):
        """Should include optional metadata."""
        anomaly = CostAnomaly(
            id="anomaly-002",
            workspace_id="ws-123",
            agent_id=None,
            anomaly_type="unusual_model",
            severity=0.5,
            description="Unusual model usage",
            expected_value=50.0,
            actual_value=100.0,
            variance_ratio=2.0,
            detected_at=datetime.now(timezone.utc),
            metadata={"model": "gpt-4", "reason": "first_use"},
        )

        assert anomaly.metadata == {"model": "gpt-4", "reason": "first_use"}

    def test_anomaly_default_metadata(self):
        """Should default metadata to empty dict."""
        anomaly = CostAnomaly(
            id="anomaly-003",
            workspace_id="ws-123",
            agent_id=None,
            anomaly_type="spike",
            severity=0.3,
            description="Minor spike",
            expected_value=10.0,
            actual_value=25.0,
            variance_ratio=2.5,
            detected_at=datetime.now(timezone.utc),
        )

        assert anomaly.metadata == {}

    def test_anomaly_to_dict(self):
        """Should serialize to dictionary."""
        now = datetime.now(timezone.utc)
        anomaly = CostAnomaly(
            id="anomaly-004",
            workspace_id="ws-123",
            agent_id="agent-789",
            anomaly_type="call_spike",
            severity=0.6,
            description="API call spike",
            expected_value=1000,
            actual_value=3000,
            variance_ratio=3.0,
            detected_at=now,
            metadata={"period": "hourly"},
        )

        d = anomaly.to_dict()

        assert d["id"] == "anomaly-004"
        assert d["workspace_id"] == "ws-123"
        assert d["agent_id"] == "agent-789"
        assert d["anomaly_type"] == "call_spike"
        assert d["severity"] == 0.6
        assert d["expected_value"] == 1000
        assert d["actual_value"] == 3000
        assert d["variance_ratio"] == 3.0
        assert d["detected_at"] == now.isoformat()
        assert d["metadata"] == {"period": "hourly"}


# =============================================================================
# AlertSearchResult Tests
# =============================================================================


class TestAlertSearchResult:
    """Tests for AlertSearchResult dataclass."""

    def test_create_result(self):
        """Should create alert search result."""
        result = AlertSearchResult(
            alert={"id": "alert-001", "level": "warning"},
            relevance_score=0.85,
        )

        assert result.alert["id"] == "alert-001"
        assert result.relevance_score == 0.85

    def test_default_relevance(self):
        """Should default relevance to 0."""
        result = AlertSearchResult(alert={"id": "alert-002"})

        assert result.relevance_score == 0.0


# =============================================================================
# CostAdapter Initialization Tests
# =============================================================================


class TestCostAdapterInit:
    """Tests for CostAdapter initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        adapter = CostAdapter()

        assert adapter._cost_tracker is None
        assert adapter._enable_dual_write is False
        assert adapter._event_callback is None
        assert adapter.ID_PREFIX == "ct_"
        assert adapter.adapter_name == "cost"

    def test_init_with_cost_tracker(self):
        """Should accept cost tracker."""
        mock_tracker = MagicMock()
        adapter = CostAdapter(cost_tracker=mock_tracker)

        assert adapter._cost_tracker is mock_tracker
        assert adapter.cost_tracker is mock_tracker

    def test_init_with_dual_write(self):
        """Should accept dual write flag."""
        adapter = CostAdapter(enable_dual_write=True)

        assert adapter._enable_dual_write is True

    def test_init_with_event_callback(self):
        """Should accept event callback."""
        callback = MagicMock()
        adapter = CostAdapter(event_callback=callback)

        assert adapter._event_callback is callback

    def test_set_event_callback(self):
        """Should set event callback after init."""
        adapter = CostAdapter()
        callback = MagicMock()

        adapter.set_event_callback(callback)

        assert adapter._event_callback is callback


# =============================================================================
# Alert Level Threshold Tests
# =============================================================================


class TestAlertLevelThreshold:
    """Tests for alert level threshold checking."""

    def test_warning_meets_threshold(self):
        """Warning level should meet threshold."""
        adapter = CostAdapter()
        assert adapter._alert_level_meets_threshold("warning") is True

    def test_critical_meets_threshold(self):
        """Critical level should meet threshold."""
        adapter = CostAdapter()
        assert adapter._alert_level_meets_threshold("critical") is True

    def test_exceeded_meets_threshold(self):
        """Exceeded level should meet threshold."""
        adapter = CostAdapter()
        assert adapter._alert_level_meets_threshold("exceeded") is True

    def test_info_below_threshold(self):
        """Info level should be below threshold."""
        adapter = CostAdapter()
        assert adapter._alert_level_meets_threshold("info") is False

    def test_invalid_level(self):
        """Invalid level should return False."""
        adapter = CostAdapter()
        assert adapter._alert_level_meets_threshold("unknown") is False


# =============================================================================
# Alert Storage Tests
# =============================================================================


class TestStoreAlert:
    """Tests for storing budget alerts."""

    def test_store_warning_alert(self):
        """Should store warning level alert."""
        adapter = CostAdapter()

        # Create mock alert with warning level
        mock_alert = MagicMock()
        mock_alert.id = "alert-001"
        mock_alert.budget_id = "budget-123"
        mock_alert.workspace_id = "ws-456"
        mock_alert.org_id = "org-789"
        mock_alert.level = MagicMock(value="warning")
        mock_alert.message = "80% of budget used"
        mock_alert.current_spend = 80.0
        mock_alert.limit = 100.0
        mock_alert.percentage = 80.0
        mock_alert.created_at = datetime.now(timezone.utc)
        mock_alert.acknowledged = False

        result = adapter.store_alert(mock_alert)

        assert result is not None
        assert result.startswith("ct_alert_")
        assert "ct_alert_alert-001" in adapter._alerts

    def test_skip_info_alert(self):
        """Should skip info level alert (below threshold)."""
        adapter = CostAdapter()

        mock_alert = MagicMock()
        mock_alert.id = "alert-002"
        mock_alert.level = MagicMock(value="info")

        result = adapter.store_alert(mock_alert)

        assert result is None
        assert len(adapter._alerts) == 0

    def test_store_critical_alert(self):
        """Should store critical level alert."""
        adapter = CostAdapter()

        mock_alert = MagicMock()
        mock_alert.id = "alert-003"
        mock_alert.budget_id = "budget-123"
        mock_alert.workspace_id = "ws-456"
        mock_alert.org_id = "org-789"
        mock_alert.level = MagicMock(value="critical")
        mock_alert.message = "95% of budget used"
        mock_alert.current_spend = 95.0
        mock_alert.limit = 100.0
        mock_alert.percentage = 95.0
        mock_alert.created_at = None
        mock_alert.acknowledged = False

        result = adapter.store_alert(mock_alert)

        assert result is not None
        alert_data = adapter._alerts[result]
        assert alert_data["level"] == "critical"

    def test_alert_updates_workspace_index(self):
        """Should update workspace index."""
        adapter = CostAdapter()

        mock_alert = MagicMock()
        mock_alert.id = "alert-004"
        mock_alert.budget_id = "budget-123"
        mock_alert.workspace_id = "ws-test"
        mock_alert.org_id = "org-789"
        mock_alert.level = MagicMock(value="warning")
        mock_alert.message = "Test"
        mock_alert.current_spend = 50.0
        mock_alert.limit = 100.0
        mock_alert.percentage = 50.0
        mock_alert.created_at = None
        mock_alert.acknowledged = False

        adapter.store_alert(mock_alert)

        assert "ws-test" in adapter._workspace_alerts
        assert len(adapter._workspace_alerts["ws-test"]) == 1


# =============================================================================
# Anomaly Storage Tests
# =============================================================================


class TestStoreAnomaly:
    """Tests for storing cost anomalies."""

    def test_store_high_variance_anomaly(self):
        """Should store anomaly with high variance."""
        adapter = CostAdapter()

        anomaly = CostAnomaly(
            id="anomaly-001",
            workspace_id="ws-123",
            agent_id=None,
            anomaly_type="spike",
            severity=0.8,
            description="Cost spike",
            expected_value=100.0,
            actual_value=300.0,
            variance_ratio=3.0,
            detected_at=datetime.now(timezone.utc),
        )

        result = adapter.store_anomaly(anomaly)

        assert result is not None
        assert result.startswith("ct_anomaly_")
        assert result in adapter._anomalies

    def test_skip_low_variance_anomaly(self):
        """Should skip anomaly with low variance (< 2x)."""
        adapter = CostAdapter()

        anomaly = CostAnomaly(
            id="anomaly-002",
            workspace_id="ws-123",
            agent_id=None,
            anomaly_type="spike",
            severity=0.3,
            description="Minor variance",
            expected_value=100.0,
            actual_value=150.0,
            variance_ratio=1.5,  # Below MIN_ANOMALY_VARIANCE (2.0)
            detected_at=datetime.now(timezone.utc),
        )

        result = adapter.store_anomaly(anomaly)

        assert result is None
        assert len(adapter._anomalies) == 0

    def test_anomaly_updates_workspace_index(self):
        """Should update workspace anomaly index."""
        adapter = CostAdapter()

        anomaly = CostAnomaly(
            id="anomaly-003",
            workspace_id="ws-test",
            agent_id=None,
            anomaly_type="spike",
            severity=0.7,
            description="Spike",
            expected_value=100.0,
            actual_value=250.0,
            variance_ratio=2.5,
            detected_at=datetime.now(timezone.utc),
        )

        adapter.store_anomaly(anomaly)

        assert "ws-test" in adapter._workspace_anomalies
        assert len(adapter._workspace_anomalies["ws-test"]) == 1


# =============================================================================
# Cost Snapshot Tests
# =============================================================================


class TestStoreCostSnapshot:
    """Tests for storing cost snapshots."""

    def test_store_snapshot(self):
        """Should store cost snapshot."""
        adapter = CostAdapter()

        result = adapter.store_cost_snapshot(
            workspace_id="ws-123",
            agent_id="agent-456",
            total_cost_usd=50.0,
            tokens_in=10000,
            tokens_out=5000,
            api_calls=100,
            period="daily",
        )

        assert result.startswith("ct_snap_")
        assert result in adapter._cost_snapshots

    def test_snapshot_updates_agent_index(self):
        """Should update agent costs index."""
        adapter = CostAdapter()

        adapter.store_cost_snapshot(
            workspace_id="ws-123",
            agent_id="agent-test",
            total_cost_usd=25.0,
            tokens_in=5000,
            tokens_out=2500,
            api_calls=50,
        )

        assert "agent-test" in adapter._agent_costs
        assert len(adapter._agent_costs["agent-test"]) == 1


# =============================================================================
# Query Tests
# =============================================================================


class TestQueries:
    """Tests for query methods."""

    def test_get_alert(self):
        """Should get alert by ID."""
        adapter = CostAdapter()

        mock_alert = MagicMock()
        mock_alert.id = "alert-query"
        mock_alert.budget_id = "budget-123"
        mock_alert.workspace_id = "ws-456"
        mock_alert.org_id = "org-789"
        mock_alert.level = MagicMock(value="warning")
        mock_alert.message = "Test"
        mock_alert.current_spend = 50.0
        mock_alert.limit = 100.0
        mock_alert.percentage = 50.0
        mock_alert.created_at = None
        mock_alert.acknowledged = False

        alert_id = adapter.store_alert(mock_alert)
        result = adapter.get_alert(alert_id)

        assert result is not None
        assert result["original_id"] == "alert-query"

    def test_get_alert_not_found(self):
        """Should return None for missing alert."""
        adapter = CostAdapter()
        result = adapter.get_alert("nonexistent")
        assert result is None

    def test_get_anomaly(self):
        """Should get anomaly by ID."""
        adapter = CostAdapter()

        anomaly = CostAnomaly(
            id="anomaly-query",
            workspace_id="ws-123",
            agent_id=None,
            anomaly_type="spike",
            severity=0.8,
            description="Test",
            expected_value=100.0,
            actual_value=300.0,
            variance_ratio=3.0,
            detected_at=datetime.now(timezone.utc),
        )

        anomaly_id = adapter.store_anomaly(anomaly)
        result = adapter.get_anomaly(anomaly_id)

        assert result is not None
        assert result["original_id"] == "anomaly-query"

    def test_get_workspace_alerts(self):
        """Should get alerts for workspace."""
        adapter = CostAdapter()

        # Store multiple alerts
        for i, level in enumerate(["warning", "critical", "warning"]):
            mock_alert = MagicMock()
            mock_alert.id = f"alert-ws-{i}"
            mock_alert.budget_id = "budget-123"
            mock_alert.workspace_id = "ws-query"
            mock_alert.org_id = "org-789"
            mock_alert.level = MagicMock(value=level)
            mock_alert.message = f"Alert {i}"
            mock_alert.current_spend = 50.0 + i * 10
            mock_alert.limit = 100.0
            mock_alert.percentage = 50.0 + i * 10
            mock_alert.created_at = datetime.now(timezone.utc)
            mock_alert.acknowledged = False
            adapter.store_alert(mock_alert)

        results = adapter.get_workspace_alerts("ws-query")

        assert len(results) == 3

    def test_get_workspace_anomalies(self):
        """Should get anomalies for workspace."""
        adapter = CostAdapter()

        for i in range(3):
            anomaly = CostAnomaly(
                id=f"anomaly-ws-{i}",
                workspace_id="ws-query",
                agent_id=None,
                anomaly_type="spike",
                severity=0.5 + i * 0.1,
                description=f"Anomaly {i}",
                expected_value=100.0,
                actual_value=250.0 + i * 50,
                variance_ratio=2.5 + i * 0.5,
                detected_at=datetime.now(timezone.utc),
            )
            adapter.store_anomaly(anomaly)

        results = adapter.get_workspace_anomalies("ws-query")

        assert len(results) == 3


# =============================================================================
# Pattern Analysis Tests
# =============================================================================


class TestPatternAnalysis:
    """Tests for cost pattern analysis."""

    def test_get_cost_patterns_empty(self):
        """Should return empty patterns for workspace with no data."""
        adapter = CostAdapter()

        patterns = adapter.get_cost_patterns("ws-empty")

        assert patterns["sample_size"] == 0
        assert patterns["avg_cost"] == 0.0

    def test_get_cost_patterns_with_data(self):
        """Should calculate patterns from snapshots."""
        adapter = CostAdapter()

        # Store multiple snapshots
        for i in range(5):
            adapter.store_cost_snapshot(
                workspace_id="ws-patterns",
                agent_id=None,
                total_cost_usd=10.0 + i,
                tokens_in=1000 + i * 100,
                tokens_out=500 + i * 50,
                api_calls=10 + i,
            )

        patterns = adapter.get_cost_patterns("ws-patterns")

        assert patterns["sample_size"] == 5
        assert patterns["avg_cost"] == 12.0  # Average of 10, 11, 12, 13, 14
        assert patterns["min_cost"] == 10.0
        assert patterns["max_cost"] == 14.0

    def test_detect_anomalies_insufficient_data(self):
        """Should return empty for insufficient data."""
        adapter = CostAdapter()

        # Only 3 snapshots (below threshold of 5)
        for i in range(3):
            adapter.store_cost_snapshot(
                workspace_id="ws-detect",
                agent_id=None,
                total_cost_usd=10.0,
                tokens_in=1000,
                tokens_out=500,
                api_calls=10,
            )

        anomalies = adapter.detect_anomalies(
            workspace_id="ws-detect",
            current_cost=30.0,
            current_tokens=3000,
            current_calls=30,
        )

        assert len(anomalies) == 0

    def test_detect_cost_spike(self):
        """Should detect cost spike anomaly."""
        adapter = CostAdapter()

        # Store baseline data
        for i in range(10):
            adapter.store_cost_snapshot(
                workspace_id="ws-spike",
                agent_id=None,
                total_cost_usd=10.0,
                tokens_in=1000,
                tokens_out=500,
                api_calls=10,
            )

        # Detect with 3x spike
        anomalies = adapter.detect_anomalies(
            workspace_id="ws-spike",
            current_cost=30.0,  # 3x normal
            current_tokens=1500,
            current_calls=10,
        )

        assert len(anomalies) >= 1
        spike_anomaly = next(a for a in anomalies if a.anomaly_type == "cost_spike")
        assert spike_anomaly.variance_ratio >= 2.0


# =============================================================================
# Agent Cost History Tests
# =============================================================================


class TestAgentCostHistory:
    """Tests for agent cost history."""

    def test_get_agent_cost_history(self):
        """Should get cost history for agent."""
        adapter = CostAdapter()

        for i in range(3):
            adapter.store_cost_snapshot(
                workspace_id="ws-123",
                agent_id="agent-history",
                total_cost_usd=10.0 + i,
                tokens_in=1000 + i * 100,
                tokens_out=500 + i * 50,
                api_calls=10 + i,
            )

        history = adapter.get_agent_cost_history("agent-history")

        assert len(history) == 3

    def test_get_agent_cost_history_empty(self):
        """Should return empty list for unknown agent."""
        adapter = CostAdapter()
        history = adapter.get_agent_cost_history("unknown-agent")
        assert history == []


# =============================================================================
# Stats Tests
# =============================================================================


class TestStats:
    """Tests for adapter statistics."""

    def test_get_stats_empty(self):
        """Should return stats for empty adapter."""
        adapter = CostAdapter()

        stats = adapter.get_stats()

        assert stats["total_alerts"] == 0
        assert stats["total_anomalies"] == 0
        assert stats["total_snapshots"] == 0
        assert stats["workspaces_with_alerts"] == 0
        assert stats["agents_tracked"] == 0

    def test_get_stats_with_data(self):
        """Should return accurate stats with data."""
        adapter = CostAdapter()

        # Add alert
        mock_alert = MagicMock()
        mock_alert.id = "alert-stats"
        mock_alert.budget_id = "budget-123"
        mock_alert.workspace_id = "ws-stats"
        mock_alert.org_id = "org-789"
        mock_alert.level = MagicMock(value="warning")
        mock_alert.message = "Test"
        mock_alert.current_spend = 50.0
        mock_alert.limit = 100.0
        mock_alert.percentage = 50.0
        mock_alert.created_at = None
        mock_alert.acknowledged = False
        adapter.store_alert(mock_alert)

        # Add anomaly
        anomaly = CostAnomaly(
            id="anomaly-stats",
            workspace_id="ws-stats",
            agent_id=None,
            anomaly_type="spike",
            severity=0.8,
            description="Test",
            expected_value=100.0,
            actual_value=300.0,
            variance_ratio=3.0,
            detected_at=datetime.now(timezone.utc),
        )
        adapter.store_anomaly(anomaly)

        # Add snapshot
        adapter.store_cost_snapshot(
            workspace_id="ws-stats",
            agent_id="agent-stats",
            total_cost_usd=25.0,
            tokens_in=5000,
            tokens_out=2500,
            api_calls=50,
        )

        stats = adapter.get_stats()

        assert stats["total_alerts"] == 1
        assert stats["total_anomalies"] == 1
        assert stats["total_snapshots"] == 1
        assert stats["agents_tracked"] == 1
        assert "warning" in stats["alert_levels"]


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission."""

    def test_emit_event_with_callback(self):
        """Should call callback when emitting events."""
        callback = MagicMock()
        adapter = CostAdapter(event_callback=callback)

        adapter._emit_event("test_event", {"key": "value"})

        callback.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_event_without_callback(self):
        """Should not raise without callback."""
        adapter = CostAdapter()
        adapter._emit_event("test_event", {"key": "value"})  # Should not raise

    def test_emit_event_handles_error(self):
        """Should catch callback errors."""
        callback = MagicMock(side_effect=RuntimeError("Callback failed"))
        adapter = CostAdapter(event_callback=callback)

        adapter._emit_event("test_event", {"key": "value"})  # Should not raise
