"""Tests for CostAdvisory and anomaly-to-advisory pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.billing.cost_tracker import CostAdvisory, CostTracker


class TestCostAdvisory:
    """Test CostAdvisory dataclass and severity mapping."""

    def test_no_action_factory(self):
        """no_action() creates an advisory with 'none' action and severity."""
        advisory = CostAdvisory.no_action("ws1")
        assert advisory.recommended_action == "none"
        assert advisory.severity == "none"
        assert advisory.workspace_id == "ws1"
        assert advisory.anomaly_count == 0

    def test_severity_to_action_critical(self):
        assert CostAdvisory.severity_to_action("critical") == "downgrade_tier"

    def test_severity_to_action_high(self):
        assert CostAdvisory.severity_to_action("high") == "reduce_rounds"

    def test_severity_to_action_warning(self):
        assert CostAdvisory.severity_to_action("warning") == "pause_workspace"

    def test_severity_to_action_info(self):
        assert CostAdvisory.severity_to_action("info") == "none"

    def test_severity_to_action_unknown(self):
        assert CostAdvisory.severity_to_action("unknown_level") == "none"

    def test_to_dict(self):
        advisory = CostAdvisory(
            recommended_action="reduce_rounds",
            severity="high",
            reason="Cost spike detected",
            workspace_id="ws1",
            anomaly_count=2,
        )
        d = advisory.to_dict()
        assert d["recommended_action"] == "reduce_rounds"
        assert d["severity"] == "high"
        assert d["reason"] == "Cost spike detected"
        assert d["workspace_id"] == "ws1"
        assert d["anomaly_count"] == 2
        assert "timestamp" in d


class TestDetectAndStoreAnomaliesAdvisory:
    """Test that detect_and_store_anomalies returns advisories."""

    @pytest.mark.asyncio
    async def test_no_adapter_returns_no_action(self):
        """No KM adapter returns empty list and no-action advisory."""
        tracker = CostTracker(usage_tracker=None, km_adapter=None)
        anomalies, advisory = await tracker.detect_and_store_anomalies("ws1")
        assert anomalies == []
        assert advisory.recommended_action == "none"
        assert advisory.severity == "none"

    @pytest.mark.asyncio
    async def test_no_stats_returns_no_action(self):
        """No workspace stats returns no-action advisory."""
        adapter = MagicMock()
        tracker = CostTracker(usage_tracker=None, km_adapter=adapter)
        anomalies, advisory = await tracker.detect_and_store_anomalies("ws_empty")
        assert anomalies == []
        assert advisory.recommended_action == "none"

    @pytest.mark.asyncio
    async def test_anomaly_detected_returns_advisory_with_action(self):
        """Detected anomalies produce an advisory with a recommended action."""
        mock_anomaly = MagicMock()
        mock_anomaly.to_dict.return_value = {
            "type": "cost_spike",
            "severity": "high",
            "actual": 50.0,
            "expected": 10.0,
            "description": "Cost 5x above normal",
        }

        adapter = MagicMock()
        adapter.detect_anomalies.return_value = [mock_anomaly]
        adapter.store_anomaly.return_value = "anomaly-123"

        tracker = CostTracker(usage_tracker=None, km_adapter=adapter)
        # Seed workspace stats so the method doesn't short-circuit
        tracker._workspace_stats["ws1"] = {
            "total_cost": 50.0,
            "tokens_in": 1000,
            "tokens_out": 500,
            "api_calls": 10,
        }

        anomalies, advisory = await tracker.detect_and_store_anomalies("ws1")

        assert len(anomalies) == 1
        assert advisory.recommended_action == "reduce_rounds"
        assert advisory.severity == "high"
        assert advisory.anomaly_count == 1
        assert "5x above normal" in advisory.reason

    @pytest.mark.asyncio
    async def test_critical_anomaly_maps_to_downgrade(self):
        """Critical severity maps to downgrade_tier action."""
        mock_anomaly = MagicMock()
        mock_anomaly.to_dict.return_value = {
            "type": "budget_breach",
            "severity": "critical",
            "actual": 200.0,
            "expected": 20.0,
            "description": "Budget breached by 10x",
        }

        adapter = MagicMock()
        adapter.detect_anomalies.return_value = [mock_anomaly]
        adapter.store_anomaly.return_value = "anomaly-456"

        tracker = CostTracker(usage_tracker=None, km_adapter=adapter)
        tracker._workspace_stats["ws2"] = {
            "total_cost": 200.0,
            "tokens_in": 5000,
            "tokens_out": 2000,
            "api_calls": 50,
        }

        anomalies, advisory = await tracker.detect_and_store_anomalies("ws2")

        assert advisory.recommended_action == "downgrade_tier"
        assert advisory.severity == "critical"


class TestGetWorkspaceCostAdvisory:
    """Test cached advisory retrieval."""

    def test_returns_cached_advisory(self):
        """get_workspace_cost_advisory returns the last stored advisory."""
        tracker = CostTracker(usage_tracker=None)
        cached = CostAdvisory(
            recommended_action="reduce_rounds",
            severity="high",
            reason="Cached advisory",
            workspace_id="ws1",
            anomaly_count=3,
        )
        tracker._last_advisory["ws1"] = cached

        result = tracker.get_workspace_cost_advisory("ws1")
        assert result is cached
        assert result.recommended_action == "reduce_rounds"

    def test_returns_no_action_when_no_cache(self):
        """Returns no-action advisory when no cached state exists."""
        tracker = CostTracker(usage_tracker=None)
        result = tracker.get_workspace_cost_advisory("ws_new")
        assert result.recommended_action == "none"
        assert result.severity == "none"
        assert result.workspace_id == "ws_new"
