"""
Tests for autonomous operations handlers (Phase 5).

Tests:
- ApprovalHandler - Approval flow operations
- TriggerHandler - Scheduled trigger operations
- AlertHandler - Alert management operations
- MonitoringHandler - Trend and anomaly monitoring
- LearningHandler - Continuous learning operations
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from aiohttp import web
from aiohttp.test_utils import make_mocked_request


# ============================================================================
# ApprovalHandler Tests
# ============================================================================


class TestApprovalHandlerListPending:
    """Tests for ApprovalHandler.list_pending."""

    @pytest.fixture
    def mock_approval_flow(self):
        """Create mock ApprovalFlow."""
        mock = MagicMock()
        mock.list_pending.return_value = []
        return mock

    @pytest.fixture
    def mock_request(self):
        """Create mock request with auth context."""
        request = MagicMock()
        request.app = {}
        return request

    @pytest.mark.asyncio
    async def test_list_pending_returns_empty_list(self, mock_approval_flow, mock_request):
        """Should return empty list when no pending approvals."""
        from aragora.server.handlers.autonomous.approvals import (
            ApprovalHandler,
            set_approval_flow,
        )

        set_approval_flow(mock_approval_flow)
        mock_approval_flow.list_pending.return_value = []

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_auth_context",
            new_callable=AsyncMock,
        ) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test-user")

            response = await ApprovalHandler.list_pending(mock_request)

            assert response.status == 200
            data = response.body
            assert b"success" in data
            assert b"pending" in data

    @pytest.mark.asyncio
    async def test_list_pending_returns_pending_requests(self, mock_approval_flow, mock_request):
        """Should return list of pending approval requests."""
        from aragora.server.handlers.autonomous.approvals import (
            ApprovalHandler,
            set_approval_flow,
        )

        # Create mock pending request
        mock_pending = MagicMock()
        mock_pending.id = "req-001"
        mock_pending.title = "Deploy to production"
        mock_pending.description = "Deploy v2.0 to production"
        mock_pending.changes = {"version": "2.0"}
        mock_pending.risk_level = "medium"
        mock_pending.requested_at = datetime.now()
        mock_pending.requested_by = "agent-1"
        mock_pending.timeout_seconds = 3600
        mock_pending.metadata = {}

        mock_approval_flow.list_pending.return_value = [mock_pending]
        set_approval_flow(mock_approval_flow)

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_auth_context",
            new_callable=AsyncMock,
        ) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test-user")

            response = await ApprovalHandler.list_pending(mock_request)

            assert response.status == 200
            mock_approval_flow.list_pending.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_pending_requires_auth(self, mock_approval_flow, mock_request):
        """Should require authentication."""
        from aragora.server.handlers.autonomous.approvals import (
            ApprovalHandler,
            set_approval_flow,
            UnauthorizedError,
        )

        set_approval_flow(mock_approval_flow)

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_auth_context",
            new_callable=AsyncMock,
        ) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            response = await ApprovalHandler.list_pending(mock_request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_list_pending_handles_errors(self, mock_approval_flow, mock_request):
        """Should handle errors gracefully."""
        from aragora.server.handlers.autonomous.approvals import (
            ApprovalHandler,
            set_approval_flow,
        )

        mock_approval_flow.list_pending.side_effect = Exception("Database error")
        set_approval_flow(mock_approval_flow)

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_auth_context",
            new_callable=AsyncMock,
        ) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test-user")

            response = await ApprovalHandler.list_pending(mock_request)

            assert response.status == 500


class TestApprovalHandlerApprove:
    """Tests for ApprovalHandler.approve."""

    def test_approve_method_exists(self):
        """ApprovalHandler should have approve method."""
        from aragora.server.handlers.autonomous.approvals import ApprovalHandler

        assert hasattr(ApprovalHandler, "approve")
        assert callable(ApprovalHandler.approve)

    def test_approve_is_async(self):
        """approve should be an async method."""
        from aragora.server.handlers.autonomous.approvals import ApprovalHandler
        import asyncio

        assert asyncio.iscoroutinefunction(ApprovalHandler.approve)


class TestApprovalHandlerReject:
    """Tests for ApprovalHandler.reject."""

    def test_reject_method_exists(self):
        """ApprovalHandler should have reject method."""
        from aragora.server.handlers.autonomous.approvals import ApprovalHandler

        assert hasattr(ApprovalHandler, "reject")
        assert callable(ApprovalHandler.reject)

    def test_reject_is_async(self):
        """reject should be an async method."""
        from aragora.server.handlers.autonomous.approvals import ApprovalHandler
        import asyncio

        assert asyncio.iscoroutinefunction(ApprovalHandler.reject)


# ============================================================================
# TriggerHandler Tests
# ============================================================================


class TestTriggerHandlerListTriggers:
    """Tests for TriggerHandler.list_triggers."""

    @pytest.fixture
    def mock_scheduled_trigger(self):
        """Create mock ScheduledTrigger."""
        mock = MagicMock()
        mock.list_triggers.return_value = []
        return mock

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_list_triggers_returns_empty_list(self, mock_scheduled_trigger, mock_request):
        """Should return empty list when no triggers."""
        from aragora.server.handlers.autonomous.triggers import (
            TriggerHandler,
            set_scheduled_trigger,
        )

        set_scheduled_trigger(mock_scheduled_trigger)

        response = await TriggerHandler.list_triggers(mock_request)

        assert response.status == 200
        data = response.body
        assert b"success" in data
        assert b"triggers" in data

    @pytest.mark.asyncio
    async def test_list_triggers_returns_triggers(self, mock_scheduled_trigger, mock_request):
        """Should return list of triggers."""
        from aragora.server.handlers.autonomous.triggers import (
            TriggerHandler,
            set_scheduled_trigger,
        )

        mock_trigger = MagicMock()
        mock_trigger.id = "trig-001"
        mock_trigger.name = "Daily report"
        mock_trigger.interval_seconds = 86400
        mock_trigger.cron_expression = None
        mock_trigger.enabled = True
        mock_trigger.last_run = None
        mock_trigger.next_run = datetime.now()
        mock_trigger.run_count = 0
        mock_trigger.max_runs = None
        mock_trigger.metadata = {}

        mock_scheduled_trigger.list_triggers.return_value = [mock_trigger]
        set_scheduled_trigger(mock_scheduled_trigger)

        response = await TriggerHandler.list_triggers(mock_request)

        assert response.status == 200
        mock_scheduled_trigger.list_triggers.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_triggers_handles_errors(self, mock_scheduled_trigger, mock_request):
        """Should handle errors gracefully."""
        from aragora.server.handlers.autonomous.triggers import (
            TriggerHandler,
            set_scheduled_trigger,
        )

        mock_scheduled_trigger.list_triggers.side_effect = Exception("Database error")
        set_scheduled_trigger(mock_scheduled_trigger)

        response = await TriggerHandler.list_triggers(mock_request)

        assert response.status == 500


class TestTriggerHandlerAddTrigger:
    """Tests for TriggerHandler.add_trigger."""

    @pytest.fixture
    def mock_scheduled_trigger(self):
        """Create mock ScheduledTrigger."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def mock_request(self):
        """Create mock request with valid data."""
        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "trigger_id": "trig-001",
                "name": "Daily report",
                "interval_seconds": 86400,
                "enabled": True,
            }
        )
        return request

    @pytest.mark.asyncio
    async def test_add_trigger_success(self, mock_scheduled_trigger, mock_request):
        """Should add trigger successfully."""
        from aragora.server.handlers.autonomous.triggers import (
            TriggerHandler,
            set_scheduled_trigger,
        )

        mock_trigger = MagicMock()
        mock_trigger.id = "trig-001"
        mock_trigger.name = "Daily report"
        mock_trigger.interval_seconds = 86400
        mock_trigger.cron_expression = None
        mock_trigger.enabled = True
        mock_trigger.last_run = None
        mock_trigger.next_run = datetime.now()
        mock_trigger.run_count = 0
        mock_trigger.max_runs = None
        mock_trigger.metadata = {}

        mock_scheduled_trigger.add_trigger.return_value = mock_trigger
        set_scheduled_trigger(mock_scheduled_trigger)

        response = await TriggerHandler.add_trigger(mock_request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_add_trigger_missing_required_fields(self, mock_scheduled_trigger, mock_request):
        """Should return 400 when required fields missing."""
        from aragora.server.handlers.autonomous.triggers import (
            TriggerHandler,
            set_scheduled_trigger,
        )

        mock_request.json = AsyncMock(return_value={})
        set_scheduled_trigger(mock_scheduled_trigger)

        response = await TriggerHandler.add_trigger(mock_request)

        assert response.status == 400


class TestTriggerHandlerRemoveTrigger:
    """Tests for TriggerHandler.remove_trigger."""

    @pytest.fixture
    def mock_scheduled_trigger(self):
        """Create mock ScheduledTrigger."""
        mock = MagicMock()
        mock.remove_trigger.return_value = True
        return mock

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.match_info = {"trigger_id": "trig-001"}
        return request

    @pytest.mark.asyncio
    async def test_remove_trigger_success(self, mock_scheduled_trigger, mock_request):
        """Should remove trigger successfully."""
        from aragora.server.handlers.autonomous.triggers import (
            TriggerHandler,
            set_scheduled_trigger,
        )

        set_scheduled_trigger(mock_scheduled_trigger)

        response = await TriggerHandler.remove_trigger(mock_request)

        assert response.status == 200
        mock_scheduled_trigger.remove_trigger.assert_called_once_with("trig-001")

    @pytest.mark.asyncio
    async def test_remove_trigger_not_found(self, mock_scheduled_trigger, mock_request):
        """Should return 404 when trigger not found."""
        from aragora.server.handlers.autonomous.triggers import (
            TriggerHandler,
            set_scheduled_trigger,
        )

        mock_scheduled_trigger.remove_trigger.return_value = False
        set_scheduled_trigger(mock_scheduled_trigger)

        response = await TriggerHandler.remove_trigger(mock_request)

        assert response.status == 404


# ============================================================================
# AlertHandler Tests
# ============================================================================


class TestAlertHandlerListActive:
    """Tests for AlertHandler.list_active."""

    @pytest.fixture
    def mock_alert_analyzer(self):
        """Create mock AlertAnalyzer."""
        mock = MagicMock()
        mock.get_active_alerts.return_value = []
        return mock

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_list_active_returns_empty_list(self, mock_alert_analyzer, mock_request):
        """Should return empty list when no active alerts."""
        from aragora.server.handlers.autonomous.alerts import (
            AlertHandler,
            set_alert_analyzer,
        )

        set_alert_analyzer(mock_alert_analyzer)

        response = await AlertHandler.list_active(mock_request)

        assert response.status == 200
        data = response.body
        assert b"success" in data
        assert b"alerts" in data

    @pytest.mark.asyncio
    async def test_list_active_returns_alerts(self, mock_alert_analyzer, mock_request):
        """Should return list of active alerts."""
        from aragora.server.handlers.autonomous.alerts import (
            AlertHandler,
            set_alert_analyzer,
        )
        from enum import Enum

        class Severity(Enum):
            HIGH = "high"

        mock_alert = MagicMock()
        mock_alert.id = "alert-001"
        mock_alert.severity = Severity.HIGH
        mock_alert.title = "High CPU usage"
        mock_alert.description = "CPU above 90%"
        mock_alert.source = "system-monitor"
        mock_alert.timestamp = datetime.now()
        mock_alert.acknowledged = False
        mock_alert.acknowledged_by = None
        mock_alert.debate_triggered = False
        mock_alert.debate_id = None
        mock_alert.metadata = {}

        mock_alert_analyzer.get_active_alerts.return_value = [mock_alert]
        set_alert_analyzer(mock_alert_analyzer)

        response = await AlertHandler.list_active(mock_request)

        assert response.status == 200
        mock_alert_analyzer.get_active_alerts.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_active_handles_errors(self, mock_alert_analyzer, mock_request):
        """Should handle errors gracefully."""
        from aragora.server.handlers.autonomous.alerts import (
            AlertHandler,
            set_alert_analyzer,
        )

        mock_alert_analyzer.get_active_alerts.side_effect = Exception("Database error")
        set_alert_analyzer(mock_alert_analyzer)

        response = await AlertHandler.list_active(mock_request)

        assert response.status == 500


class TestAlertHandlerAcknowledge:
    """Tests for AlertHandler.acknowledge."""

    @pytest.fixture
    def mock_alert_analyzer(self):
        """Create mock AlertAnalyzer."""
        mock = MagicMock()
        mock.acknowledge_alert.return_value = True
        return mock

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.match_info = {"alert_id": "alert-001"}
        request.json = AsyncMock(return_value={"user_id": "user-001"})
        return request

    @pytest.mark.asyncio
    async def test_acknowledge_success(self, mock_alert_analyzer, mock_request):
        """Should acknowledge alert successfully."""
        from aragora.server.handlers.autonomous.alerts import (
            AlertHandler,
            set_alert_analyzer,
        )

        set_alert_analyzer(mock_alert_analyzer)

        response = await AlertHandler.acknowledge(mock_request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_acknowledge_not_found(self, mock_alert_analyzer, mock_request):
        """Should return 404 when alert not found."""
        from aragora.server.handlers.autonomous.alerts import (
            AlertHandler,
            set_alert_analyzer,
        )

        mock_alert_analyzer.acknowledge_alert.return_value = False
        set_alert_analyzer(mock_alert_analyzer)

        response = await AlertHandler.acknowledge(mock_request)

        assert response.status == 404


# ============================================================================
# MonitoringHandler Tests
# ============================================================================


class TestMonitoringHandlerRecordMetric:
    """Tests for MonitoringHandler.record_metric."""

    @pytest.fixture
    def mock_trend_monitor(self):
        """Create mock TrendMonitor."""
        mock = MagicMock()
        mock.get_trend.return_value = MagicMock(
            direction="up",
            magnitude=0.05,
            confidence=0.8,
        )
        return mock

    @pytest.fixture
    def mock_anomaly_detector(self):
        """Create mock AnomalyDetector."""
        mock = MagicMock()
        mock.record.return_value = None  # No anomaly
        return mock

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.json = AsyncMock(return_value={"metric_name": "cpu_usage", "value": 75.5})
        return request

    @pytest.mark.asyncio
    async def test_record_metric_success(
        self, mock_trend_monitor, mock_anomaly_detector, mock_request
    ):
        """Should record metric successfully."""
        from aragora.server.handlers.autonomous.monitoring import (
            MonitoringHandler,
            set_trend_monitor,
            set_anomaly_detector,
        )

        set_trend_monitor(mock_trend_monitor)
        set_anomaly_detector(mock_anomaly_detector)

        response = await MonitoringHandler.record_metric(mock_request)

        assert response.status == 200
        mock_trend_monitor.record.assert_called_once_with("cpu_usage", 75.5)
        mock_anomaly_detector.record.assert_called_once_with("cpu_usage", 75.5)

    @pytest.mark.asyncio
    async def test_record_metric_missing_fields(
        self, mock_trend_monitor, mock_anomaly_detector, mock_request
    ):
        """Should return 400 when required fields missing."""
        from aragora.server.handlers.autonomous.monitoring import (
            MonitoringHandler,
            set_trend_monitor,
            set_anomaly_detector,
        )

        mock_request.json = AsyncMock(return_value={})
        set_trend_monitor(mock_trend_monitor)
        set_anomaly_detector(mock_anomaly_detector)

        response = await MonitoringHandler.record_metric(mock_request)

        assert response.status == 400

    def test_record_metric_method_exists(self):
        """MonitoringHandler should have record_metric method."""
        from aragora.server.handlers.autonomous.monitoring import MonitoringHandler

        assert hasattr(MonitoringHandler, "record_metric")
        assert callable(MonitoringHandler.record_metric)


class TestMonitoringHandlerGetTrends:
    """Tests for MonitoringHandler.get_trends."""

    def test_get_trends_method_exists(self):
        """MonitoringHandler should have get_trends method."""
        from aragora.server.handlers.autonomous.monitoring import MonitoringHandler

        assert hasattr(MonitoringHandler, "get_trends")
        assert callable(MonitoringHandler.get_trends)


class TestMonitoringHandlerGetAnomalies:
    """Tests for MonitoringHandler.get_anomalies."""

    @pytest.fixture
    def mock_anomaly_detector(self):
        """Create mock AnomalyDetector."""
        mock = MagicMock()
        mock.get_recent_anomalies.return_value = []
        return mock

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_get_anomalies_returns_empty(self, mock_anomaly_detector, mock_request):
        """Should return empty list when no anomalies."""
        from aragora.server.handlers.autonomous.monitoring import (
            MonitoringHandler,
            set_anomaly_detector,
        )

        set_anomaly_detector(mock_anomaly_detector)

        response = await MonitoringHandler.get_anomalies(mock_request)

        assert response.status == 200
        data = response.body
        assert b"success" in data
        assert b"anomalies" in data


# ============================================================================
# LearningHandler Tests
# ============================================================================


class TestLearningHandlerGetRatings:
    """Tests for LearningHandler.get_agent_ratings."""

    @pytest.fixture
    def mock_continuous_learner(self):
        """Create mock ContinuousLearner."""
        mock = MagicMock()
        mock.elo_updater.get_all_ratings.return_value = {}
        return mock

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_get_ratings_returns_empty(self, mock_continuous_learner, mock_request):
        """Should return empty ratings when no agents."""
        from aragora.server.handlers.autonomous.learning import (
            LearningHandler,
            set_continuous_learner,
        )

        set_continuous_learner(mock_continuous_learner)

        response = await LearningHandler.get_agent_ratings(mock_request)

        assert response.status == 200
        data = response.body
        assert b"success" in data
        assert b"ratings" in data

    @pytest.mark.asyncio
    async def test_get_ratings_returns_ratings(self, mock_continuous_learner, mock_request):
        """Should return agent ratings."""
        from aragora.server.handlers.autonomous.learning import (
            LearningHandler,
            set_continuous_learner,
        )

        mock_continuous_learner.elo_updater.get_all_ratings.return_value = {
            "claude": 1200,
            "gpt-4": 1150,
            "gemini": 1100,
        }
        set_continuous_learner(mock_continuous_learner)

        response = await LearningHandler.get_agent_ratings(mock_request)

        assert response.status == 200
        mock_continuous_learner.elo_updater.get_all_ratings.assert_called_once()


class TestLearningHandlerGetCalibration:
    """Tests for LearningHandler.get_agent_calibration."""

    def test_get_calibration_method_exists(self):
        """LearningHandler should have get_agent_calibration method."""
        from aragora.server.handlers.autonomous.learning import LearningHandler

        assert hasattr(LearningHandler, "get_agent_calibration")
        assert callable(LearningHandler.get_agent_calibration)

    def test_get_calibration_is_async(self):
        """get_agent_calibration should be an async method."""
        from aragora.server.handlers.autonomous.learning import LearningHandler
        import asyncio

        assert asyncio.iscoroutinefunction(LearningHandler.get_agent_calibration)


class TestLearningHandlerRecordDebateOutcome:
    """Tests for LearningHandler.record_debate_outcome."""

    def test_record_outcome_method_exists(self):
        """LearningHandler should have record_debate_outcome method."""
        from aragora.server.handlers.autonomous.learning import LearningHandler

        assert hasattr(LearningHandler, "record_debate_outcome")
        assert callable(LearningHandler.record_debate_outcome)

    def test_record_outcome_is_async(self):
        """record_debate_outcome should be an async method."""
        from aragora.server.handlers.autonomous.learning import LearningHandler
        import asyncio

        assert asyncio.iscoroutinefunction(LearningHandler.record_debate_outcome)


# ============================================================================
# Integration Tests
# ============================================================================


class TestAutonomousHandlerIntegration:
    """Integration tests for autonomous handlers."""

    def test_all_handlers_have_routes(self):
        """All autonomous handlers should define their routes."""
        from aragora.server.handlers.autonomous import (
            ApprovalHandler,
            AlertHandler,
            TriggerHandler,
            MonitoringHandler,
            LearningHandler,
        )

        # All handlers should be importable
        assert ApprovalHandler is not None
        assert AlertHandler is not None
        assert TriggerHandler is not None
        assert MonitoringHandler is not None
        assert LearningHandler is not None

    def test_approval_handler_methods(self):
        """ApprovalHandler should have all required methods."""
        from aragora.server.handlers.autonomous import ApprovalHandler

        assert hasattr(ApprovalHandler, "list_pending")
        assert hasattr(ApprovalHandler, "approve")
        assert hasattr(ApprovalHandler, "reject")
        assert callable(ApprovalHandler.list_pending)
        assert callable(ApprovalHandler.approve)
        assert callable(ApprovalHandler.reject)

    def test_trigger_handler_methods(self):
        """TriggerHandler should have all required methods."""
        from aragora.server.handlers.autonomous import TriggerHandler

        assert hasattr(TriggerHandler, "list_triggers")
        assert hasattr(TriggerHandler, "add_trigger")
        assert hasattr(TriggerHandler, "remove_trigger")
        assert callable(TriggerHandler.list_triggers)
        assert callable(TriggerHandler.add_trigger)
        assert callable(TriggerHandler.remove_trigger)

    def test_alert_handler_methods(self):
        """AlertHandler should have all required methods."""
        from aragora.server.handlers.autonomous import AlertHandler

        assert hasattr(AlertHandler, "list_active")
        assert hasattr(AlertHandler, "acknowledge")
        assert callable(AlertHandler.list_active)
        assert callable(AlertHandler.acknowledge)

    def test_monitoring_handler_methods(self):
        """MonitoringHandler should have all required methods."""
        from aragora.server.handlers.autonomous import MonitoringHandler

        assert hasattr(MonitoringHandler, "record_metric")
        assert callable(MonitoringHandler.record_metric)

    def test_learning_handler_methods(self):
        """LearningHandler should have all required methods."""
        from aragora.server.handlers.autonomous import LearningHandler

        assert hasattr(LearningHandler, "get_agent_ratings")
        assert hasattr(LearningHandler, "get_agent_calibration")
        assert callable(LearningHandler.get_agent_ratings)
        assert callable(LearningHandler.get_agent_calibration)
