"""
Tests for the alerting module.

Tests AlertRule, AlertManager, notification channels, and critical alert rules.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.observability.alerting import (
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertState,
    EmailNotificationChannel,
    MetricsCollector,
    MetricsSnapshot,
    PrometheusAlertManagerChannel,
    SlackNotificationChannel,
    create_critical_alert_rules,
    get_alert_manager,
    init_alerting,
    shutdown_alerting,
)


# =============================================================================
# Test AlertRule
# =============================================================================


class TestAlertRule:
    """Tests for AlertRule dataclass."""

    def test_create_basic_rule(self):
        """Basic rule creation works."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
        )

        assert rule.name == "test_rule"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.notification_channels == ["slack"]  # Default
        assert rule.for_duration_seconds == 0.0

    def test_create_full_rule(self):
        """Rule with all options works."""
        rule = AlertRule(
            name="full_rule",
            condition=lambda m: m.get("value", 0) > 100,
            severity=AlertSeverity.CRITICAL,
            notification_channels=["slack", "email", "prometheus"],
            description="Test description",
            runbook_url="https://example.com/runbook",
            labels={"team": "platform", "category": "test"},
            cooldown_seconds=600.0,
            for_duration_seconds=120.0,
        )

        assert rule.name == "full_rule"
        assert rule.severity == AlertSeverity.CRITICAL
        assert len(rule.notification_channels) == 3
        assert rule.description == "Test description"
        assert rule.runbook_url == "https://example.com/runbook"
        assert rule.labels["team"] == "platform"
        assert rule.cooldown_seconds == 600.0
        assert rule.for_duration_seconds == 120.0

    def test_rule_condition_callable(self):
        """Rule condition is properly callable."""
        rule = AlertRule(
            name="condition_test",
            condition=lambda m: m.get("count", 0) >= 5,
            severity=AlertSeverity.WARNING,
        )

        assert rule.condition({"count": 5}) is True
        assert rule.condition({"count": 4}) is False
        assert rule.condition({}) is False

    def test_rule_equality(self):
        """Rules with same name are equal."""
        rule1 = AlertRule(
            name="test",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
        )
        rule2 = AlertRule(
            name="test",
            condition=lambda m: False,
            severity=AlertSeverity.CRITICAL,
        )

        assert rule1 == rule2
        assert hash(rule1) == hash(rule2)

    def test_rule_inequality(self):
        """Rules with different names are not equal."""
        rule1 = AlertRule(
            name="rule1",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
        )
        rule2 = AlertRule(
            name="rule2",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
        )

        assert rule1 != rule2


# =============================================================================
# Test Alert
# =============================================================================


class TestAlert:
    """Tests for Alert dataclass."""

    def test_create_alert(self):
        """Basic alert creation works."""
        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            message="Test alert message",
        )

        assert alert.rule_name == "test_rule"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.state == AlertState.FIRING
        assert alert.message == "Test alert message"
        assert alert.notification_count == 0
        assert alert.id.startswith("alert-")

    def test_alert_timestamps(self):
        """Alert timestamps are set correctly."""
        before = datetime.now(timezone.utc)
        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.INFO,
            state=AlertState.PENDING,
            message="Test",
        )
        after = datetime.now(timezone.utc)

        assert before <= alert.first_triggered <= after
        assert before <= alert.last_triggered <= after
        assert alert.resolved_at is None

    def test_alert_to_dict(self):
        """Alert serialization works."""
        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.CRITICAL,
            state=AlertState.FIRING,
            message="Critical alert",
            labels={"env": "production"},
            annotations={"dashboard": "https://grafana/d/123"},
        )

        data = alert.to_dict()

        assert data["rule_name"] == "test_rule"
        assert data["severity"] == "critical"
        assert data["state"] == "firing"
        assert data["message"] == "Critical alert"
        assert data["labels"] == {"env": "production"}
        assert data["annotations"]["dashboard"] == "https://grafana/d/123"
        assert "first_triggered" in data
        assert "last_triggered" in data


# =============================================================================
# Test MetricsSnapshot
# =============================================================================


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot dataclass."""

    def test_create_empty_snapshot(self):
        """Empty snapshot has default values."""
        snapshot = MetricsSnapshot()

        assert snapshot.active_debates == 0
        assert snapshot.queue_size == 0
        assert snapshot.queue_capacity == 1000
        assert snapshot.memory_eviction_rate == 0.0
        assert snapshot.agent_failures == {}
        assert snapshot.circuit_breaker_states == {}

    def test_snapshot_with_values(self):
        """Snapshot with values works."""
        snapshot = MetricsSnapshot(
            agent_failures={"claude": 2, "gpt-4": 3},
            circuit_breaker_states={"claude": "open"},
            queue_size=850,
            queue_capacity=1000,
            memory_eviction_rate=0.08,
        )

        assert snapshot.agent_failures["claude"] == 2
        assert snapshot.agent_failures["gpt-4"] == 3
        assert snapshot.circuit_breaker_states["claude"] == "open"
        assert snapshot.queue_size == 850
        assert snapshot.memory_eviction_rate == 0.08


# =============================================================================
# Test AlertSeverity Enum
# =============================================================================


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_values(self):
        """Severity values are correct strings."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.EMERGENCY.value == "emergency"

    def test_severity_comparison(self):
        """Severities can be compared by name."""
        assert AlertSeverity.INFO != AlertSeverity.WARNING
        assert AlertSeverity.CRITICAL == AlertSeverity.CRITICAL


# =============================================================================
# Test AlertState Enum
# =============================================================================


class TestAlertState:
    """Tests for AlertState enum."""

    def test_state_values(self):
        """State values are correct strings."""
        assert AlertState.PENDING.value == "pending"
        assert AlertState.FIRING.value == "firing"
        assert AlertState.RESOLVED.value == "resolved"


# =============================================================================
# Test MetricsCollector
# =============================================================================


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_create_collector(self):
        """Collector can be created."""
        collector = MetricsCollector()
        assert collector is not None

    @pytest.mark.asyncio
    async def test_collect_returns_snapshot(self):
        """Collect returns a MetricsSnapshot."""
        collector = MetricsCollector()
        snapshot = await collector.collect()

        assert isinstance(snapshot, MetricsSnapshot)
        assert snapshot.timestamp is not None

    def test_record_agent_failure(self):
        """Recording agent failures works."""
        collector = MetricsCollector()

        collector.record_agent_failure("claude")
        collector.record_agent_failure("claude")
        collector.record_agent_failure("gpt-4")

        assert len(collector._agent_failure_window["claude"]) == 2
        assert len(collector._agent_failure_window["gpt-4"]) == 1

    @pytest.mark.asyncio
    async def test_collect_includes_agent_failures(self):
        """Collected snapshot includes recorded failures."""
        collector = MetricsCollector()

        collector.record_agent_failure("test-agent")
        collector.record_agent_failure("test-agent")

        snapshot = await collector.collect()

        assert snapshot.agent_failures.get("test-agent", 0) == 2


# =============================================================================
# Test Notification Channels
# =============================================================================


class TestSlackNotificationChannel:
    """Tests for SlackNotificationChannel."""

    def test_create_channel(self):
        """Channel can be created."""
        channel = SlackNotificationChannel("https://hooks.slack.com/test")

        assert channel.webhook_url == "https://hooks.slack.com/test"
        assert channel.get_name() == "slack"

    def test_get_severity_color(self):
        """Severity colors are correct."""
        channel = SlackNotificationChannel("https://hooks.slack.com/test")

        assert channel._get_severity_color(AlertSeverity.INFO) == "#36a64f"
        assert channel._get_severity_color(AlertSeverity.WARNING) == "#ffa500"
        assert channel._get_severity_color(AlertSeverity.CRITICAL) == "#ff0000"
        assert channel._get_severity_color(AlertSeverity.EMERGENCY) == "#8b0000"

    def test_build_slack_blocks(self):
        """Slack blocks are built correctly."""
        channel = SlackNotificationChannel("https://hooks.slack.com/test")

        rule = AlertRule(
            name="test_rule",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
            description="Test description",
            runbook_url="https://example.com/runbook",
        )

        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            message="Test message",
        )

        blocks = channel._build_slack_blocks(alert, rule)

        assert len(blocks) >= 2
        assert blocks[0]["type"] == "header"
        assert "WARNING" in blocks[0]["text"]["text"]
        assert blocks[1]["type"] == "section"

    @pytest.mark.asyncio
    async def test_send_handles_errors(self):
        """Send handles errors gracefully."""
        channel = SlackNotificationChannel("https://invalid.url")

        rule = AlertRule(
            name="test",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
        )

        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            message="Test",
        )

        # Should not raise, returns False on error
        with patch.object(channel, "_post_webhook", side_effect=Exception("Network error")):
            result = await channel.send(alert, rule)
            assert result is False


class TestEmailNotificationChannel:
    """Tests for EmailNotificationChannel."""

    def test_create_channel(self):
        """Channel can be created."""
        channel = EmailNotificationChannel(
            smtp_host="smtp.example.com",
            recipients=["alerts@example.com"],
        )

        assert channel.smtp_host == "smtp.example.com"
        assert channel.get_name() == "email"
        assert "alerts@example.com" in channel.recipients

    def test_build_text_content(self):
        """Text email content is built correctly."""
        channel = EmailNotificationChannel(
            smtp_host="smtp.example.com",
            recipients=["test@example.com"],
        )

        rule = AlertRule(
            name="test_rule",
            condition=lambda m: True,
            severity=AlertSeverity.CRITICAL,
            description="Critical test",
        )

        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.CRITICAL,
            state=AlertState.FIRING,
            message="Critical alert triggered",
        )

        content = channel._build_text_content(alert, rule)

        assert "test_rule" in content
        assert "CRITICAL" in content
        assert "Critical alert triggered" in content

    def test_build_html_content(self):
        """HTML email content is built correctly."""
        channel = EmailNotificationChannel(
            smtp_host="smtp.example.com",
            recipients=["test@example.com"],
        )

        rule = AlertRule(
            name="test_rule",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
        )

        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            message="Warning alert",
        )

        content = channel._build_html_content(alert, rule)

        assert "<html>" in content
        assert "test_rule" in content
        assert "Warning alert" in content

    @pytest.mark.asyncio
    async def test_send_without_recipients(self):
        """Send without recipients returns False."""
        channel = EmailNotificationChannel(
            smtp_host="smtp.example.com",
            recipients=[],
        )

        rule = AlertRule(
            name="test",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
        )

        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            message="Test",
        )

        result = await channel.send(alert, rule)
        assert result is False


class TestPrometheusAlertManagerChannel:
    """Tests for PrometheusAlertManagerChannel."""

    def test_create_channel(self):
        """Channel can be created."""
        channel = PrometheusAlertManagerChannel("http://alertmanager:9093")

        assert channel.alertmanager_url == "http://alertmanager:9093"
        assert channel.get_name() == "prometheus"

    def test_build_alertmanager_payload(self):
        """AlertManager payload is built correctly."""
        channel = PrometheusAlertManagerChannel("http://alertmanager:9093")

        rule = AlertRule(
            name="test_alert",
            condition=lambda m: True,
            severity=AlertSeverity.CRITICAL,
            description="Test description",
            runbook_url="https://example.com/runbook",
            labels={"team": "platform"},
        )

        alert = Alert(
            rule_name="test_alert",
            severity=AlertSeverity.CRITICAL,
            state=AlertState.FIRING,
            message="Test alert message",
            labels={"env": "production"},
        )

        payload = channel._build_alertmanager_payload(alert, rule)

        assert len(payload) == 1
        assert payload[0]["labels"]["alertname"] == "test_alert"
        assert payload[0]["labels"]["severity"] == "critical"
        assert payload[0]["labels"]["team"] == "platform"
        assert payload[0]["labels"]["env"] == "production"
        assert payload[0]["annotations"]["summary"] == "Test alert message"
        assert payload[0]["annotations"]["runbook_url"] == "https://example.com/runbook"


# =============================================================================
# Test AlertManager
# =============================================================================


class TestAlertManager:
    """Tests for AlertManager."""

    def test_create_manager(self):
        """Manager can be created."""
        manager = AlertManager()

        assert manager is not None
        assert manager.check_interval == 30.0
        assert manager.default_cooldown == 300.0

    def test_create_manager_with_options(self):
        """Manager can be created with custom options."""
        manager = AlertManager(
            check_interval_seconds=60.0,
            default_cooldown_seconds=600.0,
        )

        assert manager.check_interval == 60.0
        assert manager.default_cooldown == 600.0

    def test_add_rule(self):
        """Rules can be added."""
        manager = AlertManager()

        rule = AlertRule(
            name="test_rule",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
        )

        manager.add_rule(rule)

        assert "test_rule" in manager._rules
        assert len(manager.get_rules()) == 1

    def test_remove_rule(self):
        """Rules can be removed."""
        manager = AlertManager()

        rule = AlertRule(
            name="test_rule",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
        )

        manager.add_rule(rule)
        assert manager.remove_rule("test_rule") is True
        assert manager.remove_rule("test_rule") is False  # Already removed
        assert len(manager.get_rules()) == 0

    def test_add_channel(self):
        """Channels can be added."""
        manager = AlertManager()

        channel = SlackNotificationChannel("https://hooks.slack.com/test")
        manager.add_channel(channel)

        assert "slack" in manager._channels

    def test_remove_channel(self):
        """Channels can be removed."""
        manager = AlertManager()

        channel = SlackNotificationChannel("https://hooks.slack.com/test")
        manager.add_channel(channel)

        assert manager.remove_channel("slack") is True
        assert manager.remove_channel("slack") is False
        assert "slack" not in manager._channels

    @pytest.mark.asyncio
    async def test_evaluate_rules_no_rules(self):
        """Evaluating with no rules returns empty list."""
        manager = AlertManager()

        alerts = await manager.evaluate_rules()

        assert alerts == []

    @pytest.mark.asyncio
    async def test_evaluate_rules_condition_not_met(self):
        """Rules with unmet conditions don't fire."""
        manager = AlertManager()

        rule = AlertRule(
            name="never_fires",
            condition=lambda m: False,
            severity=AlertSeverity.WARNING,
        )
        manager.add_rule(rule)

        alerts = await manager.evaluate_rules()

        assert len(alerts) == 0
        assert len(manager.get_active_alerts()) == 0

    @pytest.mark.asyncio
    async def test_evaluate_rules_condition_met(self):
        """Rules with met conditions fire."""
        manager = AlertManager(default_cooldown_seconds=0)  # No cooldown for test

        rule = AlertRule(
            name="always_fires",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
            for_duration_seconds=0,  # Immediate firing
        )
        manager.add_rule(rule)

        alerts = await manager.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].rule_name == "always_fires"
        assert alerts[0].state == AlertState.FIRING

    @pytest.mark.asyncio
    async def test_evaluate_rules_for_duration(self):
        """Rules respect for_duration_seconds."""
        manager = AlertManager()

        rule = AlertRule(
            name="delayed_fire",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
            for_duration_seconds=1.0,  # Must be true for 1 second
        )
        manager.add_rule(rule)

        # First evaluation - should be pending
        alerts1 = await manager.evaluate_rules()
        assert len(alerts1) == 0  # Not yet firing
        assert len(manager.get_all_alerts()) == 1
        assert manager.get_all_alerts()[0].state == AlertState.PENDING

        # Wait and evaluate again
        await asyncio.sleep(1.1)
        alerts2 = await manager.evaluate_rules()

        assert len(alerts2) == 1
        assert alerts2[0].state == AlertState.FIRING

    @pytest.mark.asyncio
    async def test_alert_resolution(self):
        """Alerts are resolved when condition clears."""
        manager = AlertManager(default_cooldown_seconds=0)

        condition_value = {"fire": True}

        rule = AlertRule(
            name="resolvable",
            condition=lambda m: condition_value["fire"],
            severity=AlertSeverity.WARNING,
            for_duration_seconds=0,
        )
        manager.add_rule(rule)

        # Fire the alert
        await manager.evaluate_rules()
        assert len(manager.get_active_alerts()) == 1

        # Clear the condition
        condition_value["fire"] = False
        await manager.evaluate_rules()

        # Alert should be resolved and removed
        assert len(manager.get_active_alerts()) == 0

    @pytest.mark.asyncio
    async def test_manual_resolve_alert(self):
        """Alerts can be manually resolved."""
        manager = AlertManager(default_cooldown_seconds=0)

        rule = AlertRule(
            name="manual_resolve",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
            for_duration_seconds=0,
        )
        manager.add_rule(rule)

        await manager.evaluate_rules()
        assert len(manager.get_active_alerts()) == 1

        result = manager.resolve_alert("manual_resolve")
        assert result is True
        assert len(manager.get_active_alerts()) == 0

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self):
        """Alerts can be acknowledged."""
        manager = AlertManager(default_cooldown_seconds=0)

        rule = AlertRule(
            name="ack_test",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
            for_duration_seconds=0,
        )
        manager.add_rule(rule)

        await manager.evaluate_rules()
        alert = manager.get_active_alerts()[0]

        result = manager.acknowledge_alert(alert.id)
        assert result is True
        assert alert.notification_count == 999999  # Silenced

    @pytest.mark.asyncio
    async def test_notification_cooldown(self):
        """Notifications respect cooldown period."""
        manager = AlertManager(default_cooldown_seconds=60.0)

        mock_channel = MagicMock()
        mock_channel.get_name.return_value = "mock"
        mock_channel.send = AsyncMock(return_value=True)
        manager.add_channel(mock_channel)

        rule = AlertRule(
            name="cooldown_test",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
            notification_channels=["mock"],
            for_duration_seconds=0,
        )
        manager.add_rule(rule)

        # First evaluation - should notify
        await manager.evaluate_rules()
        assert mock_channel.send.call_count == 1

        # Second evaluation - should not notify (cooldown)
        await manager.evaluate_rules()
        assert mock_channel.send.call_count == 1  # Still 1

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Monitoring can be started and stopped."""
        manager = AlertManager(check_interval_seconds=0.1)

        # Start monitoring
        await manager.start_monitoring()
        assert manager._running is True
        assert manager._task is not None

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop monitoring
        await manager.stop_monitoring()
        assert manager._running is False
        assert manager._task is None

    def test_get_collector(self):
        """Collector is accessible."""
        manager = AlertManager()
        collector = manager.get_collector()

        assert isinstance(collector, MetricsCollector)


# =============================================================================
# Test Critical Alert Rules
# =============================================================================


class TestCriticalAlertRules:
    """Tests for built-in critical alert rules."""

    def test_create_critical_rules(self):
        """All 8 critical rules are created."""
        rules = create_critical_alert_rules()

        assert len(rules) == 8

        rule_names = {r.name for r in rules}
        assert "agent_cascade_failure" in rule_names
        assert "debate_stalling" in rule_names
        assert "queue_saturation_critical" in rule_names
        assert "consensus_failure_pattern" in rule_names
        assert "memory_pressure_warning" in rule_names
        assert "rate_limit_exceeded" in rule_names
        assert "circuit_breaker_open" in rule_names
        assert "api_latency_spike" in rule_names

    def test_agent_cascade_failure_rule(self):
        """Agent cascade failure rule fires correctly."""
        rules = create_critical_alert_rules()
        rule = next(r for r in rules if r.name == "agent_cascade_failure")

        # Should fire with 2+ providers failing
        metrics_firing = {
            "agent_failures": {"claude": 1, "gpt-4": 1, "gemini": 0},
        }
        assert rule.condition(metrics_firing) is True

        # Should not fire with only 1 provider failing
        metrics_ok = {
            "agent_failures": {"claude": 1, "gpt-4": 0, "gemini": 0},
        }
        assert rule.condition(metrics_ok) is False

    def test_queue_saturation_rule(self):
        """Queue saturation rule fires correctly."""
        rules = create_critical_alert_rules()
        rule = next(r for r in rules if r.name == "queue_saturation_critical")

        # Should fire at >90% saturation
        assert rule.condition({"queue_saturation": 0.95}) is True
        assert rule.condition({"queue_saturation": 0.91}) is True

        # Should not fire at <=90%
        assert rule.condition({"queue_saturation": 0.90}) is False
        assert rule.condition({"queue_saturation": 0.50}) is False

    def test_consensus_failure_pattern_rule(self):
        """Consensus failure pattern rule fires correctly."""
        rules = create_critical_alert_rules()
        rule = next(r for r in rules if r.name == "consensus_failure_pattern")

        # Should fire with >10% failure rate and >10 debates
        assert (
            rule.condition(
                {
                    "total_debates_hourly": 100,
                    "consensus_failures_hourly": 15,
                }
            )
            is True
        )

        # Should not fire with <=10% failure rate
        assert (
            rule.condition(
                {
                    "total_debates_hourly": 100,
                    "consensus_failures_hourly": 10,
                }
            )
            is False
        )

        # Should not fire with <=10 total debates
        assert (
            rule.condition(
                {
                    "total_debates_hourly": 5,
                    "consensus_failures_hourly": 2,
                }
            )
            is False
        )

    def test_memory_pressure_rule(self):
        """Memory pressure rule fires correctly."""
        rules = create_critical_alert_rules()
        rule = next(r for r in rules if r.name == "memory_pressure_warning")

        # Should fire at >5% eviction rate
        assert rule.condition({"memory_eviction_rate": 0.06}) is True

        # Should not fire at <=5%
        assert rule.condition({"memory_eviction_rate": 0.05}) is False
        assert rule.condition({"memory_eviction_rate": 0.01}) is False

    def test_circuit_breaker_open_rule(self):
        """Circuit breaker open rule fires correctly."""
        rules = create_critical_alert_rules()
        rule = next(r for r in rules if r.name == "circuit_breaker_open")

        # Should fire with any open circuit
        assert (
            rule.condition(
                {
                    "circuit_breaker_states": {"claude": "open", "gpt-4": "closed"},
                }
            )
            is True
        )

        # Should not fire with all closed
        assert (
            rule.condition(
                {
                    "circuit_breaker_states": {"claude": "closed", "gpt-4": "closed"},
                }
            )
            is False
        )

        # Should not fire with empty states
        assert rule.condition({"circuit_breaker_states": {}}) is False

    def test_api_latency_spike_rule(self):
        """API latency spike rule fires correctly."""
        rules = create_critical_alert_rules()
        rule = next(r for r in rules if r.name == "api_latency_spike")

        # Should fire when p99 > 2x baseline
        assert (
            rule.condition(
                {
                    "api_latency_p99": 0.5,
                    "api_latency_baseline": 0.2,
                }
            )
            is True
        )

        # Should not fire when p99 <= 2x baseline
        assert (
            rule.condition(
                {
                    "api_latency_p99": 0.4,
                    "api_latency_baseline": 0.2,
                }
            )
            is False
        )

        # Should not fire with zero values
        assert (
            rule.condition(
                {
                    "api_latency_p99": 0,
                    "api_latency_baseline": 0,
                }
            )
            is False
        )

    def test_rules_have_severity(self):
        """All rules have appropriate severity."""
        rules = create_critical_alert_rules()

        for rule in rules:
            assert rule.severity in AlertSeverity

    def test_rules_have_notification_channels(self):
        """All rules have notification channels."""
        rules = create_critical_alert_rules()

        for rule in rules:
            assert len(rule.notification_channels) > 0

    def test_critical_rules_have_runbooks(self):
        """Critical severity rules have runbook URLs."""
        rules = create_critical_alert_rules()

        critical_rules = [r for r in rules if r.severity == AlertSeverity.CRITICAL]
        for rule in critical_rules:
            assert rule.runbook_url is not None
            assert rule.runbook_url.startswith("https://")


# =============================================================================
# Test Global Functions
# =============================================================================


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_init_alerting_creates_manager(self):
        """init_alerting creates a global manager."""
        # Clean up any existing manager
        shutdown_alerting()

        manager = init_alerting(include_critical_rules=False)

        assert manager is not None
        assert get_alert_manager() is manager

        # Cleanup
        shutdown_alerting()

    def test_init_alerting_with_slack(self):
        """init_alerting configures Slack channel."""
        shutdown_alerting()

        manager = init_alerting(
            slack_webhook_url="https://hooks.slack.com/test",
            include_critical_rules=False,
        )

        assert "slack" in manager._channels

        shutdown_alerting()

    def test_init_alerting_with_email(self):
        """init_alerting configures Email channel."""
        shutdown_alerting()

        manager = init_alerting(
            smtp_host="smtp.example.com",
            email_recipients=["test@example.com"],
            include_critical_rules=False,
        )

        assert "email" in manager._channels

        shutdown_alerting()

    def test_init_alerting_with_prometheus(self):
        """init_alerting configures Prometheus channel."""
        shutdown_alerting()

        manager = init_alerting(
            prometheus_alertmanager_url="http://alertmanager:9093",
            include_critical_rules=False,
        )

        assert "prometheus" in manager._channels

        shutdown_alerting()

    def test_init_alerting_includes_critical_rules(self):
        """init_alerting includes critical rules by default."""
        shutdown_alerting()

        manager = init_alerting(include_critical_rules=True)

        rules = manager.get_rules()
        assert len(rules) == 8

        shutdown_alerting()

    def test_init_alerting_from_env(self):
        """init_alerting reads from environment."""
        shutdown_alerting()

        with patch.dict(
            "os.environ",
            {
                "SLACK_WEBHOOK_URL": "https://hooks.slack.com/env-test",
                "ALERTING_CHECK_INTERVAL_SECONDS": "45",
            },
        ):
            manager = init_alerting(include_critical_rules=False)

            assert "slack" in manager._channels
            assert manager.check_interval == 45.0

        shutdown_alerting()

    def test_init_alerting_disabled(self):
        """init_alerting respects ALERTING_ENABLED=false."""
        shutdown_alerting()

        with patch.dict("os.environ", {"ALERTING_ENABLED": "false"}):
            manager = init_alerting(
                slack_webhook_url="https://hooks.slack.com/test",
                include_critical_rules=True,
            )

            # Manager exists but has no channels or rules
            assert len(manager._channels) == 0
            assert len(manager.get_rules()) == 0

        shutdown_alerting()

    def test_shutdown_alerting(self):
        """shutdown_alerting clears global manager."""
        init_alerting(include_critical_rules=False)
        assert get_alert_manager() is not None

        shutdown_alerting()
        assert get_alert_manager() is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the alerting system."""

    @pytest.mark.asyncio
    async def test_full_alert_lifecycle(self):
        """Test complete alert lifecycle from firing to resolution."""
        manager = AlertManager(
            check_interval_seconds=0.1,
            default_cooldown_seconds=0,
        )

        condition_active = {"active": False}

        rule = AlertRule(
            name="lifecycle_test",
            condition=lambda m: condition_active["active"],
            severity=AlertSeverity.WARNING,
            for_duration_seconds=0,
        )
        manager.add_rule(rule)

        # Start monitoring
        await manager.start_monitoring()

        try:
            # Initially no alerts
            await asyncio.sleep(0.15)
            assert len(manager.get_active_alerts()) == 0

            # Trigger condition
            condition_active["active"] = True
            await asyncio.sleep(0.15)
            assert len(manager.get_active_alerts()) == 1
            assert manager.get_active_alerts()[0].state == AlertState.FIRING

            # Clear condition
            condition_active["active"] = False
            await asyncio.sleep(0.15)
            assert len(manager.get_active_alerts()) == 0

        finally:
            await manager.stop_monitoring()

    @pytest.mark.asyncio
    async def test_multiple_rules_independent(self):
        """Multiple rules fire independently."""
        manager = AlertManager(default_cooldown_seconds=0)

        rule1 = AlertRule(
            name="rule1",
            condition=lambda m: m.get("value1", 0) > 10,
            severity=AlertSeverity.WARNING,
            for_duration_seconds=0,
        )
        rule2 = AlertRule(
            name="rule2",
            condition=lambda m: m.get("value2", 0) > 20,
            severity=AlertSeverity.CRITICAL,
            for_duration_seconds=0,
        )
        manager.add_rule(rule1)
        manager.add_rule(rule2)

        # Mock collector to return specific values
        async def mock_collect():
            snapshot = MetricsSnapshot()
            snapshot.custom = {"value1": 15, "value2": 25}
            return snapshot

        manager._collector.collect = mock_collect

        alerts = await manager.evaluate_rules()

        assert len(alerts) == 2
        rule_names = {a.rule_name for a in alerts}
        assert "rule1" in rule_names
        assert "rule2" in rule_names

    @pytest.mark.asyncio
    async def test_notification_sent_to_channels(self):
        """Notifications are sent to configured channels."""
        manager = AlertManager(default_cooldown_seconds=0)

        mock_channel = MagicMock()
        mock_channel.get_name.return_value = "mock"
        mock_channel.send = AsyncMock(return_value=True)
        manager.add_channel(mock_channel)

        rule = AlertRule(
            name="notify_test",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
            notification_channels=["mock"],
            for_duration_seconds=0,
        )
        manager.add_rule(rule)

        await manager.evaluate_rules()

        mock_channel.send.assert_called_once()
        call_args = mock_channel.send.call_args
        assert call_args[0][0].rule_name == "notify_test"
        assert isinstance(call_args[0][1], AlertRule)
