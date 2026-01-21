"""Integration tests for autonomous operation handlers.

Tests the HTTP API endpoints for:
- Approval flows (human-in-the-loop)
- Scheduled triggers
- Alert management
- Trend monitoring
- Continuous learning
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from aragora.server.handlers.autonomous import (
    ApprovalHandler,
    AlertHandler,
    TriggerHandler,
    MonitoringHandler,
    LearningHandler,
)


class TestApprovalHandler:
    """Tests for ApprovalHandler endpoints."""

    @pytest.fixture
    def mock_approval_flow(self):
        """Create a mock ApprovalFlow."""
        from enum import Enum

        class MockStatus(Enum):
            APPROVED = "approved"
            REJECTED = "rejected"
            PENDING = "pending"

        flow = MagicMock()
        flow.list_pending.return_value = []
        flow._load_request.return_value = None

        # approve/reject return the updated request object with proper status mock
        class MockApprovedStatus:
            value = "approved"

        mock_approved_request = MagicMock()
        mock_approved_request.id = "req-123"
        mock_approved_request.status = MockApprovedStatus()
        mock_approved_request.approved_by = "test-user"
        mock_approved_request.approved_at = datetime.now()
        flow.approve.return_value = mock_approved_request

        class MockRejectedStatus:
            value = "rejected"

        mock_rejected_request = MagicMock()
        mock_rejected_request.id = "req-123"
        mock_rejected_request.status = MockRejectedStatus()
        mock_rejected_request.approved_by = None
        mock_rejected_request.rejected_by = "test-user"
        mock_rejected_request.rejected_at = datetime.now()
        mock_rejected_request.rejection_reason = "Not needed"
        flow.reject.return_value = mock_rejected_request

        return flow

    @pytest.mark.asyncio
    async def test_list_pending_empty(self, mock_approval_flow):
        """Test listing pending approvals when empty."""
        with patch(
            "aragora.server.handlers.autonomous.approvals.get_approval_flow",
            return_value=mock_approval_flow,
        ):
            request = MagicMock()
            response = await ApprovalHandler.list_pending(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert data["pending"] == []
            assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_list_pending_with_requests(self, mock_approval_flow):
        """Test listing pending approvals with some requests."""
        mock_request = MagicMock()
        mock_request.id = "req-123"
        mock_request.title = "Test Approval"
        mock_request.description = "Test description"
        mock_request.changes = {"file": "test.py", "action": "modify"}
        mock_request.risk_level = "low"
        mock_request.requested_at = datetime.now()
        mock_request.requested_by = "test-agent"
        mock_request.timeout_seconds = 3600
        mock_request.metadata = {}

        mock_approval_flow.list_pending.return_value = [mock_request]

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_approval_flow",
            return_value=mock_approval_flow,
        ):
            request = MagicMock()
            response = await ApprovalHandler.list_pending(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert len(data["pending"]) == 1
            assert data["pending"][0]["id"] == "req-123"
            assert data["pending"][0]["title"] == "Test Approval"

    @pytest.mark.asyncio
    async def test_get_request_not_found(self, mock_approval_flow):
        """Test getting a non-existent approval request."""
        mock_approval_flow._load_request.return_value = None

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_approval_flow",
            return_value=mock_approval_flow,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "nonexistent-id"

            response = await ApprovalHandler.get_request(request)

            assert response.status == 404
            data = await self._parse_json_response(response)
            assert data["success"] is False
            assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_approve_request(self, mock_approval_flow):
        """Test approving a request."""
        with patch(
            "aragora.server.handlers.autonomous.approvals.get_approval_flow",
            return_value=mock_approval_flow,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-123"
            request.json = AsyncMock(return_value={"approved_by": "test-user"})

            response = await ApprovalHandler.approve(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            mock_approval_flow.approve.assert_called_once()

    @pytest.mark.asyncio
    async def test_reject_request(self, mock_approval_flow):
        """Test rejecting a request."""
        with patch(
            "aragora.server.handlers.autonomous.approvals.get_approval_flow",
            return_value=mock_approval_flow,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-123"
            request.json = AsyncMock(
                return_value={"rejected_by": "test-user", "reason": "Not needed"}
            )

            response = await ApprovalHandler.reject(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True

    async def _parse_json_response(self, response):
        """Parse JSON from aiohttp response."""
        import json

        return json.loads(response.body)


class TestAlertHandler:
    """Tests for AlertHandler endpoints."""

    @pytest.fixture
    def mock_alert_analyzer(self):
        """Create a mock AlertAnalyzer."""
        from enum import Enum

        class MockSeverity(Enum):
            CRITICAL = "critical"
            HIGH = "high"
            MEDIUM = "medium"
            LOW = "low"
            INFO = "info"

        analyzer = MagicMock()
        analyzer.get_active_alerts.return_value = []
        analyzer.acknowledge_alert.return_value = True
        analyzer.resolve_alert.return_value = True
        analyzer.set_threshold.return_value = None
        # check_metric is async in the handler
        analyzer.check_metric = AsyncMock(return_value=None)
        return analyzer, MockSeverity

    @pytest.mark.asyncio
    async def test_list_active_alerts_empty(self, mock_alert_analyzer):
        """Test listing active alerts when empty."""
        analyzer, _ = mock_alert_analyzer
        with patch(
            "aragora.server.handlers.autonomous.alerts.get_alert_analyzer",
            return_value=analyzer,
        ):
            request = MagicMock()
            response = await AlertHandler.list_active(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert data["alerts"] == []

    @pytest.mark.asyncio
    async def test_list_active_alerts_with_alerts(self, mock_alert_analyzer):
        """Test listing active alerts with some alerts."""
        analyzer, MockSeverity = mock_alert_analyzer

        mock_alert = MagicMock()
        mock_alert.id = "alert-123"
        mock_alert.severity = MockSeverity.HIGH
        mock_alert.title = "High Error Rate"
        mock_alert.description = "Error rate exceeded threshold"
        mock_alert.source = "monitoring"
        mock_alert.timestamp = datetime.now()
        mock_alert.acknowledged = False
        mock_alert.acknowledged_by = None
        mock_alert.debate_triggered = False
        mock_alert.debate_id = None
        mock_alert.metadata = {}

        analyzer.get_active_alerts.return_value = [mock_alert]

        with patch(
            "aragora.server.handlers.autonomous.alerts.get_alert_analyzer",
            return_value=analyzer,
        ):
            request = MagicMock()
            response = await AlertHandler.list_active(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert len(data["alerts"]) == 1
            assert data["alerts"][0]["id"] == "alert-123"
            assert data["alerts"][0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, mock_alert_analyzer):
        """Test acknowledging an alert."""
        analyzer, _ = mock_alert_analyzer
        with patch(
            "aragora.server.handlers.autonomous.alerts.get_alert_analyzer",
            return_value=analyzer,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "alert-123"
            request.json = AsyncMock(return_value={"acknowledged_by": "test-user"})

            response = await AlertHandler.acknowledge(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_resolve_alert(self, mock_alert_analyzer):
        """Test resolving an alert."""
        analyzer, _ = mock_alert_analyzer
        with patch(
            "aragora.server.handlers.autonomous.alerts.get_alert_analyzer",
            return_value=analyzer,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "alert-123"

            response = await AlertHandler.resolve(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_set_threshold(self, mock_alert_analyzer):
        """Test setting an alert threshold."""
        analyzer, _ = mock_alert_analyzer
        with patch(
            "aragora.server.handlers.autonomous.alerts.get_alert_analyzer",
            return_value=analyzer,
        ):
            request = MagicMock()
            request.json = AsyncMock(
                return_value={
                    "metric_name": "error_rate",
                    "critical_threshold": 0.05,
                    "comparison": "gt",
                    "enabled": True,
                }
            )

            response = await AlertHandler.set_threshold(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_check_metric(self, mock_alert_analyzer):
        """Test checking a metric against thresholds."""
        analyzer, _ = mock_alert_analyzer
        analyzer.check_anomalies.return_value = []

        with patch(
            "aragora.server.handlers.autonomous.alerts.get_alert_analyzer",
            return_value=analyzer,
        ):
            request = MagicMock()
            request.json = AsyncMock(
                return_value={"metric_name": "error_rate", "value": 0.03}
            )

            response = await AlertHandler.check_metric(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True

    async def _parse_json_response(self, response):
        """Parse JSON from aiohttp response."""
        import json

        return json.loads(response.body)


class TestTriggerHandler:
    """Tests for TriggerHandler endpoints."""

    @pytest.fixture
    def mock_scheduled_trigger(self):
        """Create a mock ScheduledTrigger."""
        trigger = MagicMock()
        trigger.list_triggers.return_value = []

        # add_trigger returns a config object with attributes
        mock_config = MagicMock()
        mock_config.id = "trigger-new"
        mock_config.name = "Daily Summary"
        mock_config.interval_seconds = 86400
        mock_config.enabled = True
        mock_config.next_run = datetime.now() + timedelta(days=1)
        trigger.add_trigger.return_value = mock_config

        trigger.remove_trigger.return_value = True
        trigger.enable_trigger.return_value = True
        trigger.disable_trigger.return_value = True

        # start/stop are async in the handler
        trigger.start = AsyncMock()
        trigger.stop = AsyncMock()
        trigger.running = False
        return trigger

    @pytest.mark.asyncio
    async def test_list_triggers_empty(self, mock_scheduled_trigger):
        """Test listing triggers when empty."""
        with patch(
            "aragora.server.handlers.autonomous.triggers.get_scheduled_trigger",
            return_value=mock_scheduled_trigger,
        ):
            request = MagicMock()
            response = await TriggerHandler.list_triggers(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert data["triggers"] == []

    @pytest.mark.asyncio
    async def test_list_triggers_with_triggers(self, mock_scheduled_trigger):
        """Test listing triggers with some triggers."""
        mock_trigger = MagicMock()
        mock_trigger.id = "trigger-123"
        mock_trigger.name = "Hourly Analysis"
        mock_trigger.interval_seconds = 3600
        mock_trigger.cron_expression = None
        mock_trigger.enabled = True
        mock_trigger.last_run = None
        mock_trigger.next_run = datetime.now() + timedelta(hours=1)
        mock_trigger.run_count = 0
        mock_trigger.max_runs = None
        mock_trigger.metadata = {}

        mock_scheduled_trigger.list_triggers.return_value = [mock_trigger]

        with patch(
            "aragora.server.handlers.autonomous.triggers.get_scheduled_trigger",
            return_value=mock_scheduled_trigger,
        ):
            request = MagicMock()
            response = await TriggerHandler.list_triggers(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert len(data["triggers"]) == 1
            assert data["triggers"][0]["name"] == "Hourly Analysis"

    @pytest.mark.asyncio
    async def test_add_trigger(self, mock_scheduled_trigger):
        """Test adding a new trigger."""
        with patch(
            "aragora.server.handlers.autonomous.triggers.get_scheduled_trigger",
            return_value=mock_scheduled_trigger,
        ):
            request = MagicMock()
            request.json = AsyncMock(
                return_value={
                    "trigger_id": "trigger-new",
                    "name": "Daily Summary",
                    "interval_seconds": 86400,
                    "enabled": True,
                }
            )

            response = await TriggerHandler.add_trigger(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert "trigger" in data
            assert data["trigger"]["name"] == "Daily Summary"

    @pytest.mark.asyncio
    async def test_remove_trigger(self, mock_scheduled_trigger):
        """Test removing a trigger."""
        with patch(
            "aragora.server.handlers.autonomous.triggers.get_scheduled_trigger",
            return_value=mock_scheduled_trigger,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "trigger-123"

            response = await TriggerHandler.remove_trigger(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_enable_trigger(self, mock_scheduled_trigger):
        """Test enabling a trigger."""
        with patch(
            "aragora.server.handlers.autonomous.triggers.get_scheduled_trigger",
            return_value=mock_scheduled_trigger,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "trigger-123"

            response = await TriggerHandler.enable_trigger(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_disable_trigger(self, mock_scheduled_trigger):
        """Test disabling a trigger."""
        with patch(
            "aragora.server.handlers.autonomous.triggers.get_scheduled_trigger",
            return_value=mock_scheduled_trigger,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "trigger-123"

            response = await TriggerHandler.disable_trigger(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_start_scheduler(self, mock_scheduled_trigger):
        """Test starting the scheduler."""
        with patch(
            "aragora.server.handlers.autonomous.triggers.get_scheduled_trigger",
            return_value=mock_scheduled_trigger,
        ):
            request = MagicMock()
            response = await TriggerHandler.start_scheduler(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_stop_scheduler(self, mock_scheduled_trigger):
        """Test stopping the scheduler."""
        with patch(
            "aragora.server.handlers.autonomous.triggers.get_scheduled_trigger",
            return_value=mock_scheduled_trigger,
        ):
            request = MagicMock()
            response = await TriggerHandler.stop_scheduler(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True

    async def _parse_json_response(self, response):
        """Parse JSON from aiohttp response."""
        import json

        return json.loads(response.body)


class TestMonitoringHandler:
    """Tests for MonitoringHandler endpoints."""

    @pytest.fixture
    def mock_monitors(self):
        """Create mock TrendMonitor and AnomalyDetector."""
        from enum import Enum

        class MockDirection(Enum):
            UP = "up"
            DOWN = "down"
            STABLE = "stable"

        class MockSeverity(Enum):
            HIGH = "high"
            MEDIUM = "medium"
            LOW = "low"

        trend_monitor = MagicMock()
        trend_monitor.record.return_value = None
        trend_monitor.get_trend.return_value = None
        trend_monitor.get_all_trends.return_value = {}

        anomaly_detector = MagicMock()
        anomaly_detector.record.return_value = None
        anomaly_detector.get_recent_anomalies.return_value = []
        anomaly_detector.get_baseline_stats.return_value = None

        return trend_monitor, anomaly_detector, MockDirection, MockSeverity

    @pytest.mark.asyncio
    async def test_record_metric(self, mock_monitors):
        """Test recording a metric."""
        trend_monitor, anomaly_detector, _, _ = mock_monitors
        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_trend_monitor",
            return_value=trend_monitor,
        ), patch(
            "aragora.server.handlers.autonomous.monitoring.get_anomaly_detector",
            return_value=anomaly_detector,
        ):
            request = MagicMock()
            request.json = AsyncMock(
                return_value={
                    "metric_name": "response_time",
                    "value": 150.5,
                }
            )

            response = await MonitoringHandler.record_metric(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert data["metric_name"] == "response_time"
            assert data["value"] == 150.5

    @pytest.mark.asyncio
    async def test_get_trend(self, mock_monitors):
        """Test getting a trend for a metric."""
        trend_monitor, _, MockDirection, _ = mock_monitors

        mock_trend = MagicMock()
        mock_trend.metric_name = "response_time"
        mock_trend.direction = MockDirection.UP
        mock_trend.current_value = 150.5
        mock_trend.previous_value = 140.0
        mock_trend.change_percent = 7.5
        mock_trend.period_start = datetime.now() - timedelta(hours=1)
        mock_trend.period_end = datetime.now()
        mock_trend.data_points = 60
        mock_trend.confidence = 0.95
        trend_monitor.get_trend.return_value = mock_trend

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_trend_monitor",
            return_value=trend_monitor,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "response_time"
            request.query = {"period_seconds": "3600"}

            response = await MonitoringHandler.get_trend(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert "trend" in data

    @pytest.mark.asyncio
    async def test_get_all_trends(self, mock_monitors):
        """Test getting all trends."""
        trend_monitor, _, _, _ = mock_monitors
        trend_monitor.get_all_trends.return_value = {}

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_trend_monitor",
            return_value=trend_monitor,
        ):
            request = MagicMock()
            response = await MonitoringHandler.get_all_trends(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert "trends" in data

    @pytest.mark.asyncio
    async def test_get_anomalies(self, mock_monitors):
        """Test getting anomalies."""
        _, anomaly_detector, _, _ = mock_monitors
        anomaly_detector.get_recent_anomalies.return_value = []

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_anomaly_detector",
            return_value=anomaly_detector,
        ):
            request = MagicMock()
            request.query = {"hours": "24"}

            response = await MonitoringHandler.get_anomalies(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert "anomalies" in data

    @pytest.mark.asyncio
    async def test_get_baseline_stats(self, mock_monitors):
        """Test getting baseline stats."""
        _, anomaly_detector, _, _ = mock_monitors
        anomaly_detector.get_baseline_stats.return_value = {
            "mean": 100.0,
            "std": 10.0,
            "min": 80.0,
            "max": 120.0,
            "median": 100.0,
        }

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_anomaly_detector",
            return_value=anomaly_detector,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "response_time"

            response = await MonitoringHandler.get_baseline_stats(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert "stats" in data

    async def _parse_json_response(self, response):
        """Parse JSON from aiohttp response."""
        import json

        return json.loads(response.body)


class TestLearningHandler:
    """Tests for LearningHandler endpoints."""

    @pytest.fixture
    def mock_continuous_learner(self):
        """Create a mock ContinuousLearner with proper sub-objects."""
        from enum import Enum

        class MockEventType(Enum):
            DEBATE_OUTCOME = "debate_outcome"
            USER_FEEDBACK = "user_feedback"

        learner = MagicMock()

        # Mock elo_updater for get_agent_ratings
        learner.elo_updater = MagicMock()
        learner.elo_updater.get_all_ratings.return_value = {}
        learner.elo_updater.get_rating.return_value = 1500

        # Mock pattern_extractor for get_patterns
        learner.pattern_extractor = MagicMock()
        learner.pattern_extractor.get_patterns.return_value = []

        # Calibration methods
        learner.get_calibration.return_value = None
        learner.get_all_calibrations.return_value = {}

        # Async methods for recording
        mock_event = MagicMock()
        mock_event.id = "event-123"
        mock_event.event_type = MockEventType.DEBATE_OUTCOME
        mock_event.applied = True
        learner.on_debate_completed = AsyncMock(return_value=mock_event)
        learner.on_user_feedback = AsyncMock(return_value=mock_event)
        learner.run_periodic_learning = AsyncMock(return_value={"updated": 5})

        return learner, MockEventType

    @pytest.mark.asyncio
    async def test_get_agent_ratings(self, mock_continuous_learner):
        """Test getting agent ratings."""
        learner, _ = mock_continuous_learner
        learner.elo_updater.get_all_ratings.return_value = {
            "claude": 1500,
            "gpt-4": 1450,
        }

        with patch(
            "aragora.server.handlers.autonomous.learning.get_continuous_learner",
            return_value=learner,
        ):
            request = MagicMock()
            response = await LearningHandler.get_agent_ratings(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert "ratings" in data

    @pytest.mark.asyncio
    async def test_get_agent_calibration(self, mock_continuous_learner):
        """Test getting calibration for a specific agent."""
        learner, _ = mock_continuous_learner

        mock_calibration = MagicMock()
        mock_calibration.agent_id = "claude"
        mock_calibration.elo_rating = 1500
        mock_calibration.confidence_accuracy = 0.85
        mock_calibration.topic_strengths = ["coding", "math"]
        mock_calibration.topic_weaknesses = ["history"]
        mock_calibration.last_updated = datetime.now()
        mock_calibration.total_debates = 100
        mock_calibration.win_rate = 0.65
        learner.get_calibration.return_value = mock_calibration

        with patch(
            "aragora.server.handlers.autonomous.learning.get_continuous_learner",
            return_value=learner,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "claude"

            response = await LearningHandler.get_agent_calibration(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert "calibration" in data

    @pytest.mark.asyncio
    async def test_get_agent_calibration_not_found(self, mock_continuous_learner):
        """Test getting calibration for non-existent agent."""
        learner, _ = mock_continuous_learner
        learner.get_calibration.return_value = None

        with patch(
            "aragora.server.handlers.autonomous.learning.get_continuous_learner",
            return_value=learner,
        ):
            request = MagicMock()
            request.match_info.get.return_value = "unknown-agent"

            response = await LearningHandler.get_agent_calibration(request)

            # Handler returns 200 with null calibration, not 404
            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert data["calibration"] is None

    @pytest.mark.asyncio
    async def test_get_all_calibrations(self, mock_continuous_learner):
        """Test getting all calibrations."""
        learner, _ = mock_continuous_learner
        learner.get_all_calibrations.return_value = {}

        with patch(
            "aragora.server.handlers.autonomous.learning.get_continuous_learner",
            return_value=learner,
        ):
            request = MagicMock()
            response = await LearningHandler.get_all_calibrations(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert "calibrations" in data

    @pytest.mark.asyncio
    async def test_record_debate_outcome(self, mock_continuous_learner):
        """Test recording a debate outcome."""
        learner, _ = mock_continuous_learner

        with patch(
            "aragora.server.handlers.autonomous.learning.get_continuous_learner",
            return_value=learner,
        ):
            request = MagicMock()
            request.json = AsyncMock(
                return_value={
                    "debate_id": "debate-123",
                    "agents": ["claude", "gpt-4"],
                    "winner": "claude",
                    "votes": {"claude": 3, "gpt-4": 1},
                    "consensus_reached": True,
                    "topics": ["coding"],
                }
            )

            response = await LearningHandler.record_debate_outcome(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert "event" in data

    @pytest.mark.asyncio
    async def test_record_user_feedback(self, mock_continuous_learner):
        """Test recording user feedback."""
        learner, _ = mock_continuous_learner

        with patch(
            "aragora.server.handlers.autonomous.learning.get_continuous_learner",
            return_value=learner,
        ):
            request = MagicMock()
            request.json = AsyncMock(
                return_value={
                    "debate_id": "debate-123",
                    "agent_id": "claude",
                    "feedback_type": "helpful",
                    "score": 0.8,
                }
            )

            response = await LearningHandler.record_user_feedback(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert "event" in data

    @pytest.mark.asyncio
    async def test_get_patterns(self, mock_continuous_learner):
        """Test getting learned patterns."""
        learner, _ = mock_continuous_learner
        learner.pattern_extractor.get_patterns.return_value = []

        with patch(
            "aragora.server.handlers.autonomous.learning.get_continuous_learner",
            return_value=learner,
        ):
            request = MagicMock()
            request.query = {}

            response = await LearningHandler.get_patterns(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True
            assert "patterns" in data

    @pytest.mark.asyncio
    async def test_run_periodic_learning(self, mock_continuous_learner):
        """Test running periodic learning."""
        learner, _ = mock_continuous_learner

        with patch(
            "aragora.server.handlers.autonomous.learning.get_continuous_learner",
            return_value=learner,
        ):
            request = MagicMock()
            response = await LearningHandler.run_periodic_learning(request)

            assert response.status == 200
            data = await self._parse_json_response(response)
            assert data["success"] is True

    async def _parse_json_response(self, response):
        """Parse JSON from aiohttp response."""
        import json

        return json.loads(response.body)


class TestErrorHandling:
    """Tests for error handling across all autonomous handlers."""

    @pytest.mark.asyncio
    async def test_approval_handler_error(self):
        """Test error handling in ApprovalHandler."""
        with patch(
            "aragora.server.handlers.autonomous.approvals.get_approval_flow",
            side_effect=Exception("Test error"),
        ):
            request = MagicMock()
            response = await ApprovalHandler.list_pending(request)

            assert response.status == 500
            data = await self._parse_json_response(response)
            assert data["success"] is False
            assert "error" in data

    @pytest.mark.asyncio
    async def test_alert_handler_error(self):
        """Test error handling in AlertHandler."""
        with patch(
            "aragora.server.handlers.autonomous.alerts.get_alert_analyzer",
            side_effect=Exception("Test error"),
        ):
            request = MagicMock()
            response = await AlertHandler.list_active(request)

            assert response.status == 500
            data = await self._parse_json_response(response)
            assert data["success"] is False
            assert "error" in data

    @pytest.mark.asyncio
    async def test_trigger_handler_error(self):
        """Test error handling in TriggerHandler."""
        with patch(
            "aragora.server.handlers.autonomous.triggers.get_scheduled_trigger",
            side_effect=Exception("Test error"),
        ):
            request = MagicMock()
            response = await TriggerHandler.list_triggers(request)

            assert response.status == 500
            data = await self._parse_json_response(response)
            assert data["success"] is False
            assert "error" in data

    @pytest.mark.asyncio
    async def test_monitoring_handler_error(self):
        """Test error handling in MonitoringHandler."""
        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_trend_monitor",
            side_effect=Exception("Test error"),
        ):
            request = MagicMock()
            request.json = AsyncMock(return_value={"metric_name": "test", "value": 1.0})

            response = await MonitoringHandler.record_metric(request)

            assert response.status == 500
            data = await self._parse_json_response(response)
            assert data["success"] is False
            assert "error" in data

    @pytest.mark.asyncio
    async def test_learning_handler_error(self):
        """Test error handling in LearningHandler."""
        with patch(
            "aragora.server.handlers.autonomous.learning.get_continuous_learner",
            side_effect=Exception("Test error"),
        ):
            request = MagicMock()
            response = await LearningHandler.get_agent_ratings(request)

            assert response.status == 500
            data = await self._parse_json_response(response)
            assert data["success"] is False
            assert "error" in data

    async def _parse_json_response(self, response):
        """Parse JSON from aiohttp response."""
        import json

        return json.loads(response.body)
