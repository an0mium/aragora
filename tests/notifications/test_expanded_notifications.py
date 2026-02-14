"""
Tests for expanded notification lifecycle functions.

Covers:
- notify_budget_alert
- notify_cost_anomaly
- notify_compliance_finding
- notify_workflow_progress
- Cost tracker anomaly → notification wiring
- Graceful failure handling (notifications never propagate errors)
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.notifications.models import (
    Notification,
    NotificationChannel,
    NotificationPriority,
    NotificationResult,
)
from aragora.notifications.service import (
    _severity_to_priority,
    notify_budget_alert,
    notify_compliance_finding,
    notify_cost_anomaly,
    notify_workflow_progress,
)


@pytest.fixture
def mock_service():
    """Mock notification service for all tests."""
    service = MagicMock()
    service.notify = AsyncMock(return_value=[])
    service.notify_all_webhooks = AsyncMock(return_value=[])
    return service


@pytest.fixture(autouse=True)
def patch_get_service(mock_service):
    """Patch get_notification_service to return our mock."""
    with patch(
        "aragora.notifications.service.get_notification_service",
        return_value=mock_service,
    ):
        yield mock_service


# =============================================================================
# notify_budget_alert tests
# =============================================================================


class TestNotifyBudgetAlert:
    """Tests for budget alert notifications."""

    @pytest.mark.asyncio
    async def test_budget_alert_basic(self, mock_service):
        """Test basic budget alert sends notification."""
        results = await notify_budget_alert(
            budget_id="budget-1",
            current_spend=75.0,
            limit=100.0,
            threshold_pct=75.0,
            workspace_id="ws-1",
        )

        mock_service.notify.assert_awaited_once()
        notification = mock_service.notify.call_args[0][0]
        assert notification.resource_type == "budget"
        assert notification.resource_id == "budget-1"
        assert notification.workspace_id == "ws-1"
        assert notification.severity == "warning"
        assert "$75.00" in notification.message
        assert "$100.00" in notification.message

    @pytest.mark.asyncio
    async def test_budget_alert_exceeded(self, mock_service):
        """Test budget exceeded sets critical severity."""
        await notify_budget_alert(
            budget_id="b-2",
            current_spend=110.0,
            limit=100.0,
            threshold_pct=110.0,
        )

        notification = mock_service.notify.call_args[0][0]
        assert notification.severity == "critical"
        assert "Exceeded" in notification.title

    @pytest.mark.asyncio
    async def test_budget_alert_90pct(self, mock_service):
        """Test 90% threshold sets critical severity."""
        await notify_budget_alert(
            budget_id="b-3",
            current_spend=90.0,
            limit=100.0,
            threshold_pct=90.0,
        )

        notification = mock_service.notify.call_args[0][0]
        assert notification.severity == "critical"
        assert "90%" in notification.title

    @pytest.mark.asyncio
    async def test_budget_alert_50pct(self, mock_service):
        """Test 50% threshold sets info severity."""
        await notify_budget_alert(
            budget_id="b-4",
            current_spend=50.0,
            limit=100.0,
            threshold_pct=50.0,
        )

        notification = mock_service.notify.call_args[0][0]
        assert notification.severity == "info"

    @pytest.mark.asyncio
    async def test_budget_alert_metadata(self, mock_service):
        """Test budget alert includes correct metadata."""
        await notify_budget_alert(
            budget_id="b-5",
            current_spend=80.0,
            limit=100.0,
            threshold_pct=80.0,
            budget_name="Production Budget",
        )

        notification = mock_service.notify.call_args[0][0]
        assert notification.metadata["budget_id"] == "b-5"
        assert notification.metadata["current_spend"] == 80.0
        assert notification.metadata["limit"] == 100.0
        assert notification.metadata["threshold_pct"] == 80.0
        assert "Production Budget" in notification.title

    @pytest.mark.asyncio
    async def test_budget_alert_sends_webhook(self, mock_service):
        """Test budget alert triggers webhook delivery."""
        await notify_budget_alert(
            budget_id="b-6",
            current_spend=75.0,
            limit=100.0,
            threshold_pct=75.0,
        )

        mock_service.notify_all_webhooks.assert_awaited_once()
        args = mock_service.notify_all_webhooks.call_args
        assert args[0][1] == "budget.alert"

    @pytest.mark.asyncio
    async def test_budget_alert_remaining_calculation(self, mock_service):
        """Test remaining budget is calculated correctly."""
        await notify_budget_alert(
            budget_id="b-7",
            current_spend=85.0,
            limit=100.0,
            threshold_pct=85.0,
        )

        notification = mock_service.notify.call_args[0][0]
        assert "$15.00" in notification.message  # remaining


# =============================================================================
# notify_cost_anomaly tests
# =============================================================================


class TestNotifyCostAnomaly:
    """Tests for cost anomaly notifications."""

    @pytest.mark.asyncio
    async def test_cost_anomaly_basic(self, mock_service):
        """Test basic cost anomaly notification."""
        results = await notify_cost_anomaly(
            anomaly_type="spike",
            severity="warning",
            amount=50.0,
            expected=10.0,
            workspace_id="ws-1",
        )

        mock_service.notify.assert_awaited_once()
        notification = mock_service.notify.call_args[0][0]
        assert notification.resource_type == "cost_anomaly"
        assert notification.resource_id == "spike"
        assert notification.severity == "warning"
        assert "$50.0000" in notification.message
        assert "$10.0000" in notification.message

    @pytest.mark.asyncio
    async def test_cost_anomaly_deviation_calc(self, mock_service):
        """Test deviation percentage calculation."""
        await notify_cost_anomaly(
            anomaly_type="spike",
            severity="critical",
            amount=30.0,
            expected=10.0,
        )

        notification = mock_service.notify.call_args[0][0]
        assert notification.metadata["deviation_pct"] == pytest.approx(200.0)
        assert "+200.0%" in notification.message

    @pytest.mark.asyncio
    async def test_cost_anomaly_zero_expected(self, mock_service):
        """Test deviation with zero expected cost."""
        await notify_cost_anomaly(
            anomaly_type="unexpected",
            severity="warning",
            amount=5.0,
            expected=0.0,
        )

        notification = mock_service.notify.call_args[0][0]
        assert notification.metadata["deviation_pct"] == 0

    @pytest.mark.asyncio
    async def test_cost_anomaly_with_agent(self, mock_service):
        """Test anomaly includes agent info."""
        await notify_cost_anomaly(
            anomaly_type="unusual_agent",
            severity="warning",
            amount=100.0,
            expected=20.0,
            agent_id="claude-3",
        )

        notification = mock_service.notify.call_args[0][0]
        assert "claude-3" in notification.message
        assert notification.metadata["agent_id"] == "claude-3"

    @pytest.mark.asyncio
    async def test_cost_anomaly_with_details(self, mock_service):
        """Test anomaly includes extra details."""
        await notify_cost_anomaly(
            anomaly_type="model_drift",
            severity="info",
            amount=15.0,
            expected=12.0,
            details="Model pricing changed on 2026-02-01",
        )

        notification = mock_service.notify.call_args[0][0]
        assert "Model pricing changed" in notification.message

    @pytest.mark.asyncio
    async def test_cost_anomaly_sends_webhook(self, mock_service):
        """Test cost anomaly triggers webhook."""
        await notify_cost_anomaly(
            anomaly_type="spike",
            severity="critical",
            amount=200.0,
            expected=50.0,
        )

        mock_service.notify_all_webhooks.assert_awaited_once()
        assert mock_service.notify_all_webhooks.call_args[0][1] == "cost.anomaly"


# =============================================================================
# notify_compliance_finding tests
# =============================================================================


class TestNotifyComplianceFinding:
    """Tests for compliance finding notifications."""

    @pytest.mark.asyncio
    async def test_compliance_finding_basic(self, mock_service):
        """Test basic compliance finding notification."""
        results = await notify_compliance_finding(
            finding_id="cf-1",
            severity="critical",
            description="Missing encryption at rest",
            framework="SOC2",
            workspace_id="ws-1",
        )

        mock_service.notify.assert_awaited_once()
        notification = mock_service.notify.call_args[0][0]
        assert notification.resource_type == "compliance_finding"
        assert notification.resource_id == "cf-1"
        assert notification.severity == "critical"
        assert "SOC2" in notification.message
        assert "Missing encryption" in notification.message

    @pytest.mark.asyncio
    async def test_compliance_finding_with_control(self, mock_service):
        """Test compliance finding includes control ID."""
        await notify_compliance_finding(
            finding_id="cf-2",
            severity="warning",
            description="Weak password policy",
            framework="HIPAA",
            control_id="AC-7",
        )

        notification = mock_service.notify.call_args[0][0]
        assert "AC-7" in notification.message
        assert notification.metadata["control_id"] == "AC-7"

    @pytest.mark.asyncio
    async def test_compliance_finding_with_remediation(self, mock_service):
        """Test compliance finding includes remediation guidance."""
        await notify_compliance_finding(
            finding_id="cf-3",
            severity="info",
            description="Audit log retention below recommended",
            framework="GDPR",
            remediation="Increase retention to 90 days",
        )

        notification = mock_service.notify.call_args[0][0]
        assert "Increase retention" in notification.message

    @pytest.mark.asyncio
    async def test_compliance_finding_title_truncation(self, mock_service):
        """Test long description is truncated in title."""
        long_desc = "A" * 100
        await notify_compliance_finding(
            finding_id="cf-4",
            severity="warning",
            description=long_desc,
            framework="SOX",
        )

        notification = mock_service.notify.call_args[0][0]
        # Title should contain first 60 chars of description
        assert long_desc[:60] in notification.title
        assert len(notification.title) < 200

    @pytest.mark.asyncio
    async def test_compliance_finding_metadata(self, mock_service):
        """Test compliance finding metadata fields."""
        await notify_compliance_finding(
            finding_id="cf-5",
            severity="critical",
            description="Data breach detected",
            framework="GDPR",
            control_id="A.12.4",
        )

        notification = mock_service.notify.call_args[0][0]
        assert notification.metadata["finding_id"] == "cf-5"
        assert notification.metadata["framework"] == "GDPR"
        assert notification.metadata["control_id"] == "A.12.4"

    @pytest.mark.asyncio
    async def test_compliance_finding_sends_webhook(self, mock_service):
        """Test compliance finding triggers webhook."""
        await notify_compliance_finding(
            finding_id="cf-6",
            severity="warning",
            description="Test",
            framework="SOC2",
        )

        mock_service.notify_all_webhooks.assert_awaited_once()
        assert mock_service.notify_all_webhooks.call_args[0][1] == "compliance.finding"


# =============================================================================
# notify_workflow_progress tests
# =============================================================================


class TestNotifyWorkflowProgress:
    """Tests for workflow progress notifications."""

    @pytest.mark.asyncio
    async def test_workflow_progress_basic(self, mock_service):
        """Test basic workflow progress notification."""
        results = await notify_workflow_progress(
            workflow_id="wf-12345678901234567890",
            step_name="data_validation",
            status="completed",
            progress_pct=50.0,
            workspace_id="ws-1",
        )

        mock_service.notify.assert_awaited_once()
        notification = mock_service.notify.call_args[0][0]
        assert notification.resource_type == "workflow"
        assert notification.resource_id == "wf-12345678901234567890"
        assert notification.severity == "info"
        assert "50%" in notification.title

    @pytest.mark.asyncio
    async def test_workflow_progress_failed_step(self, mock_service):
        """Test failed workflow step sets error severity."""
        await notify_workflow_progress(
            workflow_id="wf-abc",
            step_name="deploy",
            status="failed",
            progress_pct=75.0,
        )

        notification = mock_service.notify.call_args[0][0]
        assert notification.severity == "error"
        assert notification.priority == NotificationPriority.HIGH
        assert "Failed" in notification.title

    @pytest.mark.asyncio
    async def test_workflow_progress_complete(self, mock_service):
        """Test 100% progress shows complete."""
        await notify_workflow_progress(
            workflow_id="wf-done",
            step_name="final_step",
            status="completed",
            progress_pct=100.0,
        )

        notification = mock_service.notify.call_args[0][0]
        assert "Complete" in notification.title

    @pytest.mark.asyncio
    async def test_workflow_progress_metadata(self, mock_service):
        """Test workflow progress includes correct metadata."""
        await notify_workflow_progress(
            workflow_id="wf-meta",
            step_name="analysis",
            status="started",
            progress_pct=25.0,
        )

        notification = mock_service.notify.call_args[0][0]
        assert notification.metadata["workflow_id"] == "wf-meta"
        assert notification.metadata["step_name"] == "analysis"
        assert notification.metadata["status"] == "started"
        assert notification.metadata["progress_pct"] == 25.0

    @pytest.mark.asyncio
    async def test_workflow_progress_with_details(self, mock_service):
        """Test workflow progress includes details."""
        await notify_workflow_progress(
            workflow_id="wf-detail",
            step_name="processing",
            status="completed",
            progress_pct=60.0,
            details="Processed 150 of 250 records",
        )

        notification = mock_service.notify.call_args[0][0]
        assert "Processed 150" in notification.message

    @pytest.mark.asyncio
    async def test_workflow_progress_sends_webhook(self, mock_service):
        """Test workflow progress triggers webhook."""
        await notify_workflow_progress(
            workflow_id="wf-hook",
            step_name="test",
            status="completed",
            progress_pct=100.0,
        )

        mock_service.notify_all_webhooks.assert_awaited_once()
        assert mock_service.notify_all_webhooks.call_args[0][1] == "workflow.progress"


# =============================================================================
# Graceful failure tests
# =============================================================================


class TestGracefulFailure:
    """Test that notification failures never propagate."""

    @pytest.mark.asyncio
    async def test_budget_alert_failure_returns_empty(self, mock_service):
        """Budget alert swallows service errors."""
        mock_service.notify.side_effect = RuntimeError("Connection lost")

        results = await notify_budget_alert(
            budget_id="b-fail",
            current_spend=50.0,
            limit=100.0,
            threshold_pct=50.0,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_cost_anomaly_failure_returns_empty(self, mock_service):
        """Cost anomaly swallows service errors."""
        mock_service.notify.side_effect = RuntimeError("Timeout")

        results = await notify_cost_anomaly(
            anomaly_type="spike",
            severity="warning",
            amount=100.0,
            expected=50.0,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_compliance_finding_failure_returns_empty(self, mock_service):
        """Compliance finding swallows service errors."""
        mock_service.notify.side_effect = RuntimeError("Network error")

        results = await notify_compliance_finding(
            finding_id="cf-fail",
            severity="critical",
            description="Test failure",
            framework="SOC2",
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_workflow_progress_failure_returns_empty(self, mock_service):
        """Workflow progress swallows service errors."""
        mock_service.notify.side_effect = RuntimeError("Service unavailable")

        results = await notify_workflow_progress(
            workflow_id="wf-fail",
            step_name="deploy",
            status="started",
            progress_pct=0.0,
        )

        assert results == []


# =============================================================================
# Cost tracker integration tests
# =============================================================================


class TestCostTrackerIntegration:
    """Test that cost tracker wires notify_cost_anomaly on anomaly detection."""

    @pytest.mark.asyncio
    async def test_detect_anomalies_calls_notify(self):
        """detect_and_store_anomalies sends notification for each anomaly."""
        from aragora.billing.cost_tracker import CostTracker

        # Create mock KM adapter
        mock_adapter = MagicMock()

        # Create a mock anomaly object
        mock_anomaly = MagicMock()
        mock_anomaly.to_dict.return_value = {
            "type": "spike",
            "severity": "warning",
            "actual": 100.0,
            "expected": 20.0,
            "description": "Cost spike detected",
        }
        mock_adapter.detect_anomalies.return_value = [mock_anomaly]
        mock_adapter.store_anomaly.return_value = "anomaly-id-1"

        tracker = CostTracker(km_adapter=mock_adapter)
        # Add some stats so detection has data
        tracker._workspace_stats["ws-test"] = {
            "total_cost": Decimal("100"),
            "tokens_in": 5000,
            "tokens_out": 1000,
            "api_calls": 10,
        }

        with patch(
            "aragora.notifications.service.notify_cost_anomaly",
            new_callable=AsyncMock,
        ) as mock_notify:
            anomalies = await tracker.detect_and_store_anomalies("ws-test")

            assert len(anomalies) == 1
            mock_notify.assert_awaited_once_with(
                anomaly_type="spike",
                severity="warning",
                amount=100.0,
                expected=20.0,
                workspace_id="ws-test",
                details="Cost spike detected",
            )

    @pytest.mark.asyncio
    async def test_detect_anomalies_notify_failure_doesnt_propagate(self):
        """Notification failure during anomaly detection doesn't break detection."""
        from aragora.billing.cost_tracker import CostTracker

        mock_adapter = MagicMock()
        mock_anomaly = MagicMock()
        mock_anomaly.to_dict.return_value = {
            "type": "spike",
            "severity": "critical",
            "actual": 500.0,
            "expected": 100.0,
        }
        mock_adapter.detect_anomalies.return_value = [mock_anomaly]
        mock_adapter.store_anomaly.return_value = "anomaly-id-2"

        tracker = CostTracker(km_adapter=mock_adapter)
        tracker._workspace_stats["ws-fail"] = {
            "total_cost": Decimal("500"),
            "tokens_in": 10000,
            "tokens_out": 2000,
            "api_calls": 20,
        }

        with patch(
            "aragora.notifications.service.notify_cost_anomaly",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Notification service down"),
        ):
            # Should still return anomalies despite notification failure
            anomalies = await tracker.detect_and_store_anomalies("ws-fail")
            assert len(anomalies) == 1

    @pytest.mark.asyncio
    async def test_detect_anomalies_no_adapter_skips_notify(self):
        """Without KM adapter, no notification is sent."""
        from aragora.billing.cost_tracker import CostTracker

        tracker = CostTracker()
        anomalies = await tracker.detect_and_store_anomalies("ws-none")
        assert anomalies == []


# =============================================================================
# Event type tests
# =============================================================================


class TestEventTypes:
    """Test that new event types exist in StreamEventType."""

    def test_cost_anomaly_event_type(self):
        from aragora.events.types import StreamEventType

        assert StreamEventType.COST_ANOMALY.value == "cost_anomaly"

    def test_budget_alert_event_type(self):
        from aragora.events.types import StreamEventType

        assert StreamEventType.BUDGET_ALERT.value == "budget_alert"

    def test_compliance_finding_event_type(self):
        from aragora.events.types import StreamEventType

        assert StreamEventType.COMPLIANCE_FINDING.value == "compliance_finding"
