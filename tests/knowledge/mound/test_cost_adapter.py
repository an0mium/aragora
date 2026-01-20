"""Tests for the CostAdapter."""

import pytest
from unittest.mock import Mock
from datetime import datetime
from decimal import Decimal

from aragora.knowledge.mound.adapters.cost_adapter import (
    CostAdapter,
    CostAnomaly,
    AlertSearchResult,
)
from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource


class TestCostAnomaly:
    """Tests for CostAnomaly dataclass."""

    def test_basic_creation(self):
        """Create a basic anomaly."""
        anomaly = CostAnomaly(
            id="anom_123",
            workspace_id="ws1",
            agent_id="agent1",
            anomaly_type="cost_spike",
            severity=0.8,
            description="Cost spike detected",
            expected_value=10.0,
            actual_value=50.0,
            variance_ratio=5.0,
            detected_at=datetime.utcnow(),
        )
        assert anomaly.id == "anom_123"
        assert anomaly.variance_ratio == 5.0
        assert anomaly.metadata == {}

    def test_to_dict(self):
        """Convert anomaly to dict."""
        anomaly = CostAnomaly(
            id="anom_123",
            workspace_id="ws1",
            agent_id=None,
            anomaly_type="call_spike",
            severity=0.6,
            description="API call spike",
            expected_value=100,
            actual_value=300,
            variance_ratio=3.0,
            detected_at=datetime.utcnow(),
            metadata={"source": "billing"},
        )

        data = anomaly.to_dict()

        assert data["id"] == "anom_123"
        assert data["anomaly_type"] == "call_spike"
        assert data["variance_ratio"] == 3.0


class TestAlertSearchResult:
    """Tests for AlertSearchResult dataclass."""

    def test_basic_creation(self):
        """Create a basic search result."""
        result = AlertSearchResult(
            alert={"id": "ct_1", "level": "warning"},
            relevance_score=0.8,
        )
        assert result.alert["id"] == "ct_1"


class TestCostAdapterInit:
    """Tests for CostAdapter initialization."""

    def test_init_without_tracker(self):
        """Initialize without tracker."""
        adapter = CostAdapter()
        assert adapter.cost_tracker is None

    def test_init_with_tracker(self):
        """Initialize with tracker."""
        mock_tracker = Mock()
        adapter = CostAdapter(cost_tracker=mock_tracker)
        assert adapter.cost_tracker is mock_tracker

    def test_constants(self):
        """Verify adapter constants."""
        assert CostAdapter.ID_PREFIX == "ct_"
        assert CostAdapter.MIN_ALERT_LEVEL == "warning"
        assert CostAdapter.MIN_ANOMALY_VARIANCE == 2.0
        assert CostAdapter.ALERT_LEVELS == ["info", "warning", "critical", "exceeded"]


class TestCostAdapterAlertLevelThreshold:
    """Tests for _alert_level_meets_threshold method."""

    def test_warning_meets_threshold(self):
        """Warning level meets threshold."""
        adapter = CostAdapter()
        assert adapter._alert_level_meets_threshold("warning") is True

    def test_critical_meets_threshold(self):
        """Critical level meets threshold."""
        adapter = CostAdapter()
        assert adapter._alert_level_meets_threshold("critical") is True

    def test_info_below_threshold(self):
        """Info level below threshold."""
        adapter = CostAdapter()
        assert adapter._alert_level_meets_threshold("info") is False

    def test_exceeded_meets_threshold(self):
        """Exceeded level meets threshold."""
        adapter = CostAdapter()
        assert adapter._alert_level_meets_threshold("exceeded") is True


class TestCostAdapterStoreAlert:
    """Tests for store_alert method."""

    def test_store_warning_alert(self):
        """Store a warning-level alert."""
        adapter = CostAdapter()

        mock_alert = Mock()
        mock_alert.id = "alert_123"
        mock_alert.budget_id = "budget_456"
        mock_alert.workspace_id = "ws1"
        mock_alert.org_id = None
        mock_alert.level = Mock(value="warning")
        mock_alert.message = "Budget at 75%"
        mock_alert.current_spend = Decimal("75.00")
        mock_alert.limit = Decimal("100.00")
        mock_alert.percentage = 75.0
        mock_alert.created_at = datetime.utcnow()
        mock_alert.acknowledged = False

        alert_id = adapter.store_alert(mock_alert)

        assert alert_id is not None
        assert alert_id.startswith("ct_alert_")

    def test_skip_info_alert(self):
        """Don't store info-level alerts."""
        adapter = CostAdapter()

        mock_alert = Mock()
        mock_alert.id = "alert_123"
        mock_alert.level = Mock(value="info")

        alert_id = adapter.store_alert(mock_alert)
        assert alert_id is None

    def test_updates_workspace_index(self):
        """Verify workspace index is updated."""
        adapter = CostAdapter()

        mock_alert = Mock()
        mock_alert.id = "alert_123"
        mock_alert.budget_id = "b1"
        mock_alert.workspace_id = "ws1"
        mock_alert.org_id = None
        mock_alert.level = Mock(value="critical")
        mock_alert.message = "Budget critical"
        mock_alert.current_spend = Decimal("90.00")
        mock_alert.limit = Decimal("100.00")
        mock_alert.percentage = 90.0
        mock_alert.created_at = datetime.utcnow()
        mock_alert.acknowledged = False

        adapter.store_alert(mock_alert)

        assert "ws1" in adapter._workspace_alerts


class TestCostAdapterStoreAnomaly:
    """Tests for store_anomaly method."""

    def test_store_high_variance_anomaly(self):
        """Store anomaly with high variance."""
        adapter = CostAdapter()

        anomaly = CostAnomaly(
            id="anom_123",
            workspace_id="ws1",
            agent_id="agent1",
            anomaly_type="cost_spike",
            severity=0.8,
            description="5x cost spike",
            expected_value=10.0,
            actual_value=50.0,
            variance_ratio=5.0,
            detected_at=datetime.utcnow(),
        )

        anomaly_id = adapter.store_anomaly(anomaly)

        assert anomaly_id is not None
        assert anomaly_id.startswith("ct_anomaly_")

    def test_skip_low_variance_anomaly(self):
        """Don't store anomaly with low variance."""
        adapter = CostAdapter()

        anomaly = CostAnomaly(
            id="anom_123",
            workspace_id="ws1",
            agent_id=None,
            anomaly_type="minor_spike",
            severity=0.3,
            description="Minor variation",
            expected_value=10.0,
            actual_value=15.0,
            variance_ratio=1.5,  # Below 2.0
            detected_at=datetime.utcnow(),
        )

        anomaly_id = adapter.store_anomaly(anomaly)
        assert anomaly_id is None


class TestCostAdapterStoreCostSnapshot:
    """Tests for store_cost_snapshot method."""

    def test_store_snapshot(self):
        """Store a cost snapshot."""
        adapter = CostAdapter()

        snapshot_id = adapter.store_cost_snapshot(
            workspace_id="ws1",
            agent_id="agent1",
            total_cost_usd=25.50,
            tokens_in=100000,
            tokens_out=50000,
            api_calls=500,
            period="daily",
        )

        assert snapshot_id is not None
        assert snapshot_id.startswith("ct_snap_")
        assert "agent1" in adapter._agent_costs


class TestCostAdapterGetWorkspaceAlerts:
    """Tests for get_workspace_alerts method."""

    def test_get_alerts(self):
        """Get alerts for a workspace."""
        adapter = CostAdapter()

        adapter._alerts["ct_a1"] = {
            "id": "ct_a1",
            "level": "warning",
            "created_at": "2024-01-01T00:00:00Z",
        }
        adapter._alerts["ct_a2"] = {
            "id": "ct_a2",
            "level": "critical",
            "created_at": "2024-01-02T00:00:00Z",
        }
        adapter._workspace_alerts["ws1"] = ["ct_a1", "ct_a2"]

        results = adapter.get_workspace_alerts("ws1")

        assert len(results) == 2
        # Should be sorted newest first
        assert results[0]["level"] == "critical"


class TestCostAdapterGetCostPatterns:
    """Tests for get_cost_patterns method."""

    def test_get_patterns_with_data(self):
        """Get patterns from historical data."""
        adapter = CostAdapter()

        # Add some snapshots
        for i in range(10):
            adapter._cost_snapshots[f"snap_{i}"] = {
                "workspace_id": "ws1",
                "agent_id": None,
                "total_cost_usd": 10.0 + i,
                "tokens_in": 10000,
                "tokens_out": 5000,
                "api_calls": 100,
                "created_at": f"2024-01-{10+i:02d}T00:00:00Z",
            }

        patterns = adapter.get_cost_patterns("ws1")

        assert patterns["sample_size"] == 10
        assert patterns["avg_cost"] > 0
        assert patterns["cost_stddev"] > 0

    def test_get_patterns_no_data(self):
        """Handle no data gracefully."""
        adapter = CostAdapter()

        patterns = adapter.get_cost_patterns("ws1")

        assert patterns["sample_size"] == 0
        assert patterns["avg_cost"] == 0.0


class TestCostAdapterDetectAnomalies:
    """Tests for detect_anomalies method."""

    def test_detect_cost_spike(self):
        """Detect a cost spike anomaly."""
        adapter = CostAdapter()

        # Add historical data
        for i in range(10):
            adapter._cost_snapshots[f"snap_{i}"] = {
                "workspace_id": "ws1",
                "total_cost_usd": 10.0,
                "tokens_in": 10000,
                "tokens_out": 5000,
                "api_calls": 100,
                "created_at": f"2024-01-{10+i:02d}T00:00:00Z",
            }

        # Detect anomalies with 5x spike
        anomalies = adapter.detect_anomalies(
            workspace_id="ws1",
            current_cost=50.0,  # 5x normal
            current_tokens=10000,
            current_calls=100,
        )

        assert len(anomalies) >= 1
        assert any(a.anomaly_type == "cost_spike" for a in anomalies)

    def test_detect_call_spike(self):
        """Detect an API call spike anomaly."""
        adapter = CostAdapter()

        # Add historical data
        for i in range(10):
            adapter._cost_snapshots[f"snap_{i}"] = {
                "workspace_id": "ws1",
                "total_cost_usd": 10.0,
                "tokens_in": 10000,
                "tokens_out": 5000,
                "api_calls": 100,
                "created_at": f"2024-01-{10+i:02d}T00:00:00Z",
            }

        # Detect anomalies with 3x call spike
        anomalies = adapter.detect_anomalies(
            workspace_id="ws1",
            current_cost=10.0,
            current_tokens=10000,
            current_calls=300,  # 3x normal
        )

        assert len(anomalies) >= 1
        assert any(a.anomaly_type == "call_spike" for a in anomalies)

    def test_no_anomalies_insufficient_data(self):
        """No anomalies when insufficient historical data."""
        adapter = CostAdapter()

        # Only 2 snapshots (need 5)
        for i in range(2):
            adapter._cost_snapshots[f"snap_{i}"] = {
                "workspace_id": "ws1",
                "total_cost_usd": 10.0,
                "tokens_in": 10000,
                "tokens_out": 5000,
                "api_calls": 100,
                "created_at": f"2024-01-{10+i:02d}T00:00:00Z",
            }

        anomalies = adapter.detect_anomalies(
            workspace_id="ws1",
            current_cost=100.0,
            current_tokens=10000,
            current_calls=100,
        )

        assert len(anomalies) == 0


class TestCostAdapterToKnowledgeItem:
    """Tests for to_knowledge_item method."""

    def test_convert_exceeded_alert(self):
        """Convert exceeded-level alert."""
        adapter = CostAdapter()

        alert = {
            "id": "ct_alert_123",
            "original_id": "alert_123",
            "level": "exceeded",
            "message": "Budget exceeded by 10%",
            "percentage": 110.0,
            "current_spend": "110.00",
            "limit": "100.00",
            "workspace_id": "ws1",
            "created_at": "2024-01-01T00:00:00Z",
        }

        item = adapter.to_knowledge_item(alert)

        assert item.id == "ct_alert_123"
        assert "Budget exceeded" in item.content
        assert item.source == KnowledgeSource.COST
        assert item.confidence == ConfidenceLevel.VERIFIED

    def test_convert_warning_alert(self):
        """Convert warning-level alert."""
        adapter = CostAdapter()

        alert = {
            "id": "ct_alert_456",
            "level": "warning",
            "message": "Budget at 75%",
            "percentage": 75.0,
            "created_at": "2024-01-01T00:00:00Z",
        }

        item = adapter.to_knowledge_item(alert)

        assert item.confidence == ConfidenceLevel.MEDIUM


class TestCostAdapterGetStats:
    """Tests for get_stats method."""

    def test_get_stats(self):
        """Get adapter statistics."""
        adapter = CostAdapter()

        adapter._alerts["a1"] = {"level": "warning"}
        adapter._alerts["a2"] = {"level": "critical"}
        adapter._anomalies["an1"] = {}
        adapter._cost_snapshots["s1"] = {}
        adapter._workspace_alerts["ws1"] = ["a1", "a2"]
        adapter._agent_costs["agent1"] = ["s1"]

        stats = adapter.get_stats()

        assert stats["total_alerts"] == 2
        assert stats["total_anomalies"] == 1
        assert stats["total_snapshots"] == 1
        assert stats["workspaces_with_alerts"] == 1
        assert stats["agents_tracked"] == 1
        assert stats["alert_levels"]["warning"] == 1
        assert stats["alert_levels"]["critical"] == 1
