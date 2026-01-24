"""Tests for InboxSLAMonitor."""

import pytest
from datetime import datetime, timezone, timedelta

from aragora.services.inbox_sla_monitor import (
    InboxSLAMonitor,
    SLAConfig,
    SLAViolation,
    SLAViolationType,
    AtRiskMessage,
    SLAMetrics,
    EscalationRule,
    EscalationLevel,
    reset_sla_monitor,
)


@pytest.fixture
def monitor():
    """Create a fresh SLA monitor for testing."""
    reset_sla_monitor()
    return InboxSLAMonitor()


@pytest.fixture
def sample_config():
    """Create a sample SLA config."""
    return SLAConfig(
        inbox_id="inbox_123",
        org_id="org_456",
        response_time_minutes=30,
        resolution_time_minutes=240,
        escalation_rules=[
            EscalationRule(
                id="rule_1",
                level=EscalationLevel.WARNING,
                threshold_minutes=15,
                notify_channels=["slack"],
                notify_users=["user_1"],
            ),
            EscalationRule(
                id="rule_2",
                level=EscalationLevel.BREACH,
                threshold_minutes=0,
                notify_channels=["email", "slack"],
                notify_users=["user_1", "user_2"],
                reassign_to="manager_1",
            ),
        ],
    )


class TestSLAConfig:
    """Tests for SLAConfig."""

    def test_create_config(self, sample_config):
        """Test creating an SLA config."""
        assert sample_config.inbox_id == "inbox_123"
        assert sample_config.response_time_minutes == 30
        assert sample_config.resolution_time_minutes == 240
        assert len(sample_config.escalation_rules) == 2

    def test_config_to_dict(self, sample_config):
        """Test converting config to dict."""
        d = sample_config.to_dict()
        assert d["inbox_id"] == "inbox_123"
        assert d["response_time_minutes"] == 30
        assert d["enabled"] is True
        assert len(d["escalation_rules"]) == 2

    def test_config_from_dict(self):
        """Test creating config from dict."""
        data = {
            "inbox_id": "inbox_abc",
            "org_id": "org_xyz",
            "response_time_minutes": 60,
            "resolution_time_minutes": 480,
            "enabled": True,
        }
        config = SLAConfig.from_dict(data)
        assert config.inbox_id == "inbox_abc"
        assert config.response_time_minutes == 60

    def test_config_defaults(self):
        """Test config default values."""
        config = SLAConfig(inbox_id="test", org_id="org")
        assert config.response_time_minutes == 60
        assert config.resolution_time_minutes == 480
        assert config.enabled is True
        assert config.business_hours_only is False


class TestSLAMonitorConfig:
    """Tests for SLA monitor config management."""

    def test_set_and_get_config(self, monitor, sample_config):
        """Test setting and retrieving config."""
        monitor.set_config(sample_config)
        retrieved = monitor.get_config("inbox_123")

        assert retrieved is not None
        assert retrieved.inbox_id == "inbox_123"
        assert retrieved.response_time_minutes == 30

    def test_get_nonexistent_config(self, monitor):
        """Test getting config that doesn't exist."""
        result = monitor.get_config("nonexistent")
        assert result is None

    def test_delete_config(self, monitor, sample_config):
        """Test deleting config."""
        monitor.set_config(sample_config)
        deleted = monitor.delete_config("inbox_123")

        assert deleted is True
        assert monitor.get_config("inbox_123") is None

    def test_delete_nonexistent_config(self, monitor):
        """Test deleting config that doesn't exist."""
        deleted = monitor.delete_config("nonexistent")
        assert deleted is False


class TestSLACompliance:
    """Tests for SLA compliance checking."""

    @pytest.mark.asyncio
    async def test_check_response_sla_breach(self, monitor, sample_config):
        """Test detecting first response SLA breach."""
        monitor.set_config(sample_config)

        now = datetime.now(timezone.utc)
        messages = [
            {
                "id": "msg_1",
                "received_at": now - timedelta(minutes=60),  # 60 min ago
                "status": "open",
                # No first_response_at - breached 30 min SLA
            }
        ]

        violations = await monitor.check_sla_compliance("inbox_123", messages)

        assert len(violations) >= 1
        response_violations = [
            v for v in violations if v.violation_type == SLAViolationType.FIRST_RESPONSE
        ]
        assert len(response_violations) == 1
        assert response_violations[0].sla_minutes == 30
        assert response_violations[0].actual_minutes >= 60

    @pytest.mark.asyncio
    async def test_check_resolution_sla_breach(self, monitor, sample_config):
        """Test detecting resolution SLA breach."""
        monitor.set_config(sample_config)

        now = datetime.now(timezone.utc)
        messages = [
            {
                "id": "msg_1",
                "received_at": now - timedelta(minutes=300),  # 5 hours ago
                "first_response_at": now - timedelta(minutes=290),  # Responded
                "status": "in_progress",
                # Not resolved - breached 4 hour SLA
            }
        ]

        violations = await monitor.check_sla_compliance("inbox_123", messages)

        resolution_violations = [
            v for v in violations if v.violation_type == SLAViolationType.RESOLUTION
        ]
        assert len(resolution_violations) == 1
        assert resolution_violations[0].sla_minutes == 240

    @pytest.mark.asyncio
    async def test_no_violations_when_compliant(self, monitor, sample_config):
        """Test no violations when SLAs are met."""
        monitor.set_config(sample_config)

        now = datetime.now(timezone.utc)
        messages = [
            {
                "id": "msg_1",
                "received_at": now - timedelta(minutes=10),
                "first_response_at": now - timedelta(minutes=5),  # Within SLA
                "status": "in_progress",
            }
        ]

        violations = await monitor.check_sla_compliance("inbox_123", messages)

        # Should have no response violations (responded in time)
        response_violations = [
            v for v in violations if v.violation_type == SLAViolationType.FIRST_RESPONSE
        ]
        assert len(response_violations) == 0

    @pytest.mark.asyncio
    async def test_resolved_messages_no_violations(self, monitor, sample_config):
        """Test that resolved messages don't trigger violations."""
        monitor.set_config(sample_config)

        now = datetime.now(timezone.utc)
        messages = [
            {
                "id": "msg_1",
                "received_at": now - timedelta(hours=24),  # Very old
                "resolved_at": now - timedelta(hours=23),
                "status": "resolved",
            }
        ]

        violations = await monitor.check_sla_compliance("inbox_123", messages)
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_disabled_config_no_check(self, monitor, sample_config):
        """Test that disabled configs don't check SLAs."""
        sample_config.enabled = False
        monitor.set_config(sample_config)

        now = datetime.now(timezone.utc)
        messages = [
            {
                "id": "msg_1",
                "received_at": now - timedelta(hours=48),  # Very breached
                "status": "open",
            }
        ]

        violations = await monitor.check_sla_compliance("inbox_123", messages)
        assert len(violations) == 0


class TestAtRiskMessages:
    """Tests for at-risk message detection."""

    @pytest.mark.asyncio
    async def test_get_at_risk_response(self, monitor, sample_config):
        """Test detecting messages approaching response SLA breach."""
        monitor.set_config(sample_config)

        now = datetime.now(timezone.utc)
        messages = [
            {
                "id": "msg_1",
                "subject": "Urgent request",
                "received_at": now - timedelta(minutes=20),  # 20 min ago
                "status": "open",
                # 30 min SLA, 10 min remaining - at risk
            }
        ]

        at_risk = await monitor.get_at_risk_messages(
            "inbox_123", threshold_minutes=15, messages=messages
        )

        assert len(at_risk) == 1
        assert at_risk[0].message_id == "msg_1"
        assert at_risk[0].minutes_remaining <= 10
        assert at_risk[0].risk_type == SLAViolationType.FIRST_RESPONSE

    @pytest.mark.asyncio
    async def test_at_risk_sorted_by_urgency(self, monitor, sample_config):
        """Test at-risk messages are sorted by urgency."""
        monitor.set_config(sample_config)

        now = datetime.now(timezone.utc)
        messages = [
            {
                "id": "msg_less_urgent",
                "subject": "Less urgent",
                "received_at": now - timedelta(minutes=18),  # 12 min remaining
                "status": "open",
            },
            {
                "id": "msg_more_urgent",
                "subject": "More urgent",
                "received_at": now - timedelta(minutes=25),  # 5 min remaining
                "status": "open",
            },
        ]

        at_risk = await monitor.get_at_risk_messages(
            "inbox_123", threshold_minutes=15, messages=messages
        )

        # More urgent should be first
        if len(at_risk) >= 2:
            assert at_risk[0].minutes_remaining <= at_risk[1].minutes_remaining


class TestSLAMetrics:
    """Tests for SLA metrics calculation."""

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, monitor, sample_config):
        """Test calculating SLA metrics."""
        monitor.set_config(sample_config)

        now = datetime.now(timezone.utc)
        messages = [
            # Met both SLAs
            {
                "id": "msg_1",
                "received_at": now - timedelta(hours=1),
                "first_response_at": now - timedelta(hours=1) + timedelta(minutes=10),
                "resolved_at": now - timedelta(hours=1) + timedelta(minutes=60),
            },
            # Met response, breached resolution
            {
                "id": "msg_2",
                "received_at": now - timedelta(hours=10),
                "first_response_at": now - timedelta(hours=10) + timedelta(minutes=15),
                "resolved_at": now - timedelta(hours=10) + timedelta(minutes=300),
            },
            # Breached both
            {
                "id": "msg_3",
                "received_at": now - timedelta(hours=12),
                "first_response_at": now - timedelta(hours=12) + timedelta(minutes=45),
                "resolved_at": now - timedelta(hours=12) + timedelta(minutes=360),
            },
        ]

        metrics = await monitor.get_sla_metrics("inbox_123", period_days=7, messages=messages)

        assert metrics.total_messages == 3
        assert metrics.response_sla_met == 2  # msg_1 and msg_2
        assert metrics.response_sla_breached == 1  # msg_3
        assert metrics.resolution_sla_met == 1  # msg_1
        assert metrics.resolution_sla_breached == 2  # msg_2 and msg_3
        assert 0 <= metrics.response_compliance_rate <= 1
        assert 0 <= metrics.resolution_compliance_rate <= 1

    @pytest.mark.asyncio
    async def test_metrics_empty_inbox(self, monitor, sample_config):
        """Test metrics for empty inbox."""
        monitor.set_config(sample_config)

        metrics = await monitor.get_sla_metrics("inbox_123", messages=[])

        assert metrics.total_messages == 0
        assert metrics.response_compliance_rate == 1.0  # Default to compliant
        assert metrics.resolution_compliance_rate == 1.0

    def test_metrics_to_dict(self):
        """Test converting metrics to dict."""
        metrics = SLAMetrics(
            inbox_id="inbox_123",
            period_start=datetime.now(timezone.utc) - timedelta(days=7),
            period_end=datetime.now(timezone.utc),
            total_messages=100,
            response_sla_met=90,
            response_sla_breached=10,
            resolution_sla_met=85,
            resolution_sla_breached=15,
            avg_response_time_minutes=25.5,
            avg_resolution_time_minutes=180.3,
            response_compliance_rate=0.9,
            resolution_compliance_rate=0.85,
        )

        d = metrics.to_dict()
        assert d["total_messages"] == 100
        assert d["response_compliance_rate"] == 0.9
        assert d["avg_response_time_minutes"] == 25.5


class TestEscalation:
    """Tests for escalation triggering."""

    @pytest.mark.asyncio
    async def test_trigger_escalation(self, monitor, sample_config):
        """Test triggering escalation for violation."""
        monitor.set_config(sample_config)

        # Track if handler was called
        handler_called = []

        def test_handler(violation, config):
            handler_called.append((violation, config))

        monitor.register_escalation_handler(test_handler)

        violation = SLAViolation(
            id="viol_1",
            inbox_id="inbox_123",
            message_id="msg_1",
            violation_type=SLAViolationType.FIRST_RESPONSE,
            sla_minutes=30,
            actual_minutes=45,
            breached_at=datetime.now(timezone.utc),
            escalation_level=EscalationLevel.BREACH,
        )

        result = await monitor.trigger_escalation(violation)

        assert result is True
        assert violation.escalation_triggered is True
        assert len(handler_called) == 1

    @pytest.mark.asyncio
    async def test_escalation_no_matching_rules(self, monitor):
        """Test escalation with no matching rules."""
        config = SLAConfig(
            inbox_id="inbox_123",
            org_id="org_456",
            escalation_rules=[],  # No rules
        )
        monitor.set_config(config)

        violation = SLAViolation(
            id="viol_1",
            inbox_id="inbox_123",
            message_id="msg_1",
            violation_type=SLAViolationType.FIRST_RESPONSE,
            sla_minutes=30,
            actual_minutes=45,
            breached_at=datetime.now(timezone.utc),
            escalation_level=EscalationLevel.BREACH,
        )

        result = await monitor.trigger_escalation(violation)
        assert result is False


class TestEscalationLevel:
    """Tests for escalation level determination."""

    def test_warning_level(self, monitor):
        """Test warning level (at or below SLA)."""
        level = monitor._determine_escalation_level(29, 30)
        assert level == EscalationLevel.WARNING

    def test_breach_level(self, monitor):
        """Test breach level (up to 50% over)."""
        level = monitor._determine_escalation_level(40, 30)
        assert level == EscalationLevel.BREACH

    def test_critical_level(self, monitor):
        """Test critical level (more than 50% over)."""
        level = monitor._determine_escalation_level(60, 30)
        assert level == EscalationLevel.CRITICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
