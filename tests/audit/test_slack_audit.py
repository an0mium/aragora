"""
Tests for Slack Integration Audit Logger.

Tests the SlackAuditLogger class which provides audit logging
for Slack commands, events, OAuth operations, rate limits,
and signature verification failures.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from aragora.audit.log import AuditCategory, AuditEvent, AuditLog, AuditOutcome
from aragora.audit.slack_audit import (
    SlackAuditLogger,
    get_slack_audit_logger,
    reset_slack_audit_logger,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_audit_log(tmp_path):
    """Create a real AuditLog backed by a temporary SQLite database."""
    db_path = tmp_path / "test_audit.db"
    audit = AuditLog(db_path=db_path)
    return audit


@pytest.fixture
def slack_logger(mock_audit_log):
    """Create a SlackAuditLogger with a real AuditLog backend."""
    return SlackAuditLogger(audit_log=mock_audit_log)


# ===========================================================================
# Tests: SlackAuditLogger Initialization
# ===========================================================================


class TestSlackAuditLoggerInit:
    """Tests for SlackAuditLogger initialization."""

    def test_init_with_audit_log(self, mock_audit_log):
        """Test initialization with explicit AuditLog."""
        logger = SlackAuditLogger(audit_log=mock_audit_log)
        assert logger._audit is mock_audit_log

    def test_init_without_audit_log(self):
        """Test initialization without AuditLog uses None initially."""
        logger = SlackAuditLogger()
        assert logger._audit is None

    def test_audit_property_lazy_init(self):
        """Test audit property lazily initializes from singleton."""
        logger = SlackAuditLogger()
        with patch("aragora.audit.slack_audit.get_audit_log") as mock_get:
            mock_log = MagicMock()
            mock_get.return_value = mock_log
            result = logger.audit
            assert result is mock_log
            mock_get.assert_called_once()

    def test_resource_type_constants(self):
        """Test resource type constants are defined."""
        assert SlackAuditLogger.RESOURCE_SLACK_WORKSPACE == "slack_workspace"
        assert SlackAuditLogger.RESOURCE_SLACK_CHANNEL == "slack_channel"
        assert SlackAuditLogger.RESOURCE_SLACK_USER == "slack_user"
        assert SlackAuditLogger.RESOURCE_SLACK_COMMAND == "slack_command"

    def test_action_constants(self):
        """Test action constants are defined."""
        assert SlackAuditLogger.ACTION_COMMAND_EXECUTE == "slack_command_execute"
        assert SlackAuditLogger.ACTION_EVENT_RECEIVE == "slack_event_receive"
        assert SlackAuditLogger.ACTION_OAUTH_INSTALL == "slack_oauth_install"
        assert SlackAuditLogger.ACTION_OAUTH_UNINSTALL == "slack_oauth_uninstall"
        assert SlackAuditLogger.ACTION_OAUTH_TOKEN_REFRESH == "slack_oauth_token_refresh"
        assert SlackAuditLogger.ACTION_RATE_LIMIT == "slack_rate_limit"
        assert SlackAuditLogger.ACTION_SIGNATURE_VERIFY == "slack_signature_verify"


# ===========================================================================
# Tests: log_command
# ===========================================================================


class TestLogCommand:
    """Tests for log_command method."""

    def test_log_command_success(self, slack_logger, mock_audit_log):
        """Test logging a successful command."""
        event_id = slack_logger.log_command(
            workspace_id="T12345",
            user_id="U67890",
            command="/ask",
            args="What is AI?",
            result="success",
        )

        assert event_id is not None
        assert isinstance(event_id, str)

        # Verify event was stored
        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_command_execute"))
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.SUCCESS
        assert events[0].workspace_id == "T12345"
        assert events[0].actor_id == "U67890"

    def test_log_command_error(self, slack_logger, mock_audit_log):
        """Test logging a failed command."""
        event_id = slack_logger.log_command(
            workspace_id="T12345",
            user_id="U67890",
            command="/ask",
            result="error",
            error="API timeout",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_command_execute"))
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.ERROR
        assert events[0].reason == "API timeout"

    def test_log_command_rate_limited(self, slack_logger, mock_audit_log):
        """Test logging a rate-limited command."""
        event_id = slack_logger.log_command(
            workspace_id="T12345",
            user_id="U67890",
            command="/ask",
            result="rate_limited",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_command_execute"))
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.DENIED
        assert events[0].reason == "Rate limit exceeded"

    def test_log_command_denied(self, slack_logger, mock_audit_log):
        """Test logging a denied command."""
        event_id = slack_logger.log_command(
            workspace_id="T12345",
            user_id="U67890",
            command="/admin",
            result="denied",
            error="Insufficient permissions",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_command_execute"))
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.DENIED
        assert events[0].reason == "Insufficient permissions"

    def test_log_command_with_response_time(self, slack_logger, mock_audit_log):
        """Test logging a command with response time."""
        slack_logger.log_command(
            workspace_id="T12345",
            user_id="U67890",
            command="/ask",
            response_time_ms=150.5,
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_command_execute"))
        assert len(events) == 1
        assert events[0].details.get("response_time_ms") == 150.5

    def test_log_command_with_channel(self, slack_logger, mock_audit_log):
        """Test logging a command with channel ID."""
        slack_logger.log_command(
            workspace_id="T12345",
            user_id="U67890",
            command="/ask",
            channel_id="C99999",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_command_execute"))
        assert len(events) == 1
        assert events[0].details.get("channel_id") == "C99999"

    def test_log_command_args_length_recorded(self, slack_logger, mock_audit_log):
        """Test that args length is recorded but not the args themselves."""
        test_args = "This is a long question about something"
        slack_logger.log_command(
            workspace_id="T12345",
            user_id="U67890",
            command="/ask",
            args=test_args,
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_command_execute"))
        assert len(events) == 1
        assert events[0].details.get("args_length") == len(test_args)

    def test_log_command_empty_args(self, slack_logger, mock_audit_log):
        """Test that empty args results in args_length of 0."""
        slack_logger.log_command(
            workspace_id="T12345",
            user_id="U67890",
            command="/help",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_command_execute"))
        assert len(events) == 1
        assert events[0].details.get("args_length") == 0


# ===========================================================================
# Tests: log_event
# ===========================================================================


class TestLogEvent:
    """Tests for log_event method."""

    def test_log_event_success(self, slack_logger, mock_audit_log):
        """Test logging a successful event."""
        event_id = slack_logger.log_event(
            workspace_id="T12345",
            event_type="message",
            payload_summary={"channel": "C12345", "type": "message"},
        )

        assert event_id is not None

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_event_receive"))
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.SUCCESS

    def test_log_event_failure(self, slack_logger, mock_audit_log):
        """Test logging a failed event."""
        slack_logger.log_event(
            workspace_id="T12345",
            event_type="message",
            payload_summary={},
            success=False,
            error="Processing failed",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_event_receive"))
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.ERROR
        assert events[0].reason == "Processing failed"

    def test_log_event_with_user(self, slack_logger, mock_audit_log):
        """Test logging an event with user ID."""
        slack_logger.log_event(
            workspace_id="T12345",
            event_type="app_mention",
            payload_summary={"channel": "C12345"},
            user_id="U67890",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_event_receive"))
        assert len(events) == 1
        assert events[0].actor_id == "U67890"

    def test_log_event_without_user(self, slack_logger, mock_audit_log):
        """Test logging an event without user ID defaults to slack_event."""
        slack_logger.log_event(
            workspace_id="T12345",
            event_type="message",
            payload_summary={},
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_event_receive"))
        assert len(events) == 1
        assert events[0].actor_id == "slack_event"

    def test_log_event_payload_merged(self, slack_logger, mock_audit_log):
        """Test that payload_summary is merged into details."""
        slack_logger.log_event(
            workspace_id="T12345",
            event_type="reaction_added",
            payload_summary={"emoji": "thumbsup", "item_type": "message"},
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_event_receive"))
        assert len(events) == 1
        assert events[0].details.get("emoji") == "thumbsup"
        assert events[0].details.get("event_type") == "reaction_added"


# ===========================================================================
# Tests: log_oauth
# ===========================================================================


class TestLogOAuth:
    """Tests for log_oauth method."""

    def test_log_oauth_install_success(self, slack_logger, mock_audit_log):
        """Test logging a successful OAuth install."""
        event_id = slack_logger.log_oauth(
            workspace_id="T12345",
            action="install",
            success=True,
            scopes=["chat:write", "commands"],
        )

        assert event_id is not None

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_oauth_install"))
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.SUCCESS
        assert events[0].category == AuditCategory.SECURITY
        assert events[0].details.get("scopes") == ["chat:write", "commands"]

    def test_log_oauth_install_failure(self, slack_logger, mock_audit_log):
        """Test logging a failed OAuth install."""
        slack_logger.log_oauth(
            workspace_id="",
            action="install",
            success=False,
            error="Invalid code",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_oauth_install"))
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.FAILURE
        assert events[0].resource_id == "unknown"

    def test_log_oauth_uninstall(self, slack_logger, mock_audit_log):
        """Test logging an OAuth uninstall."""
        slack_logger.log_oauth(
            workspace_id="T12345",
            action="uninstall",
            success=True,
            user_id="U67890",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_oauth_uninstall"))
        assert len(events) == 1
        assert events[0].category == AuditCategory.SECURITY

    def test_log_oauth_token_refresh(self, slack_logger, mock_audit_log):
        """Test logging an OAuth token refresh."""
        slack_logger.log_oauth(
            workspace_id="T12345",
            action="token_refresh",
            success=True,
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_oauth_token_refresh"))
        assert len(events) == 1
        # Token refresh is AUTH category, not SECURITY
        assert events[0].category == AuditCategory.AUTH

    def test_log_oauth_unknown_action(self, slack_logger, mock_audit_log):
        """Test logging an unknown OAuth action uses dynamic action name."""
        slack_logger.log_oauth(
            workspace_id="T12345",
            action="reauthorize",
            success=True,
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_oauth_reauthorize"))
        assert len(events) == 1

    def test_log_oauth_actor_default(self, slack_logger, mock_audit_log):
        """Test OAuth without user_id defaults to oauth_flow."""
        slack_logger.log_oauth(
            workspace_id="T12345",
            action="install",
            success=True,
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_oauth_install"))
        assert len(events) == 1
        assert events[0].actor_id == "oauth_flow"


# ===========================================================================
# Tests: log_rate_limit
# ===========================================================================


class TestLogRateLimit:
    """Tests for log_rate_limit method."""

    def test_log_rate_limit(self, slack_logger, mock_audit_log):
        """Test logging a rate limit event."""
        event_id = slack_logger.log_rate_limit(
            workspace_id="T12345",
            user_id="U67890",
            command="/ask",
        )

        assert event_id is not None

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_rate_limit"))
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.DENIED
        assert events[0].category == AuditCategory.SECURITY
        assert events[0].details.get("limit_type") == "user"

    def test_log_rate_limit_workspace_type(self, slack_logger, mock_audit_log):
        """Test logging a workspace-level rate limit."""
        slack_logger.log_rate_limit(
            workspace_id="T12345",
            user_id="U67890",
            command="/debate",
            limit_type="workspace",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_rate_limit"))
        assert len(events) == 1
        assert events[0].details.get("limit_type") == "workspace"
        assert "workspace" in events[0].reason

    def test_log_rate_limit_global_type(self, slack_logger, mock_audit_log):
        """Test logging a global rate limit."""
        slack_logger.log_rate_limit(
            workspace_id="T12345",
            user_id="U67890",
            command="/ask",
            limit_type="global",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_rate_limit"))
        assert len(events) == 1
        assert events[0].details.get("limit_type") == "global"


# ===========================================================================
# Tests: log_signature_failure
# ===========================================================================


class TestLogSignatureFailure:
    """Tests for log_signature_failure method."""

    def test_log_signature_failure(self, slack_logger, mock_audit_log):
        """Test logging a signature verification failure."""
        event_id = slack_logger.log_signature_failure(
            workspace_id="T12345",
            ip_address="192.168.1.100",
            user_agent="curl/7.68",
        )

        assert event_id is not None

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_signature_verify"))
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.DENIED
        assert events[0].category == AuditCategory.SECURITY
        assert events[0].ip_address == "192.168.1.100"
        assert events[0].user_agent == "curl/7.68"
        assert events[0].details.get("suspected_attack") is True

    def test_log_signature_failure_unknown_workspace(self, slack_logger, mock_audit_log):
        """Test logging failure without workspace defaults resource_id to unknown."""
        slack_logger.log_signature_failure(
            workspace_id="",
        )

        from aragora.audit.log import AuditQuery

        events = mock_audit_log.query(AuditQuery(action="slack_signature_verify"))
        assert len(events) == 1
        assert events[0].resource_id == "unknown"
        assert events[0].actor_id == "unknown"


# ===========================================================================
# Tests: Singleton Management
# ===========================================================================


class TestSingletonManagement:
    """Tests for module-level singleton functions."""

    def test_get_slack_audit_logger(self):
        """Test getting the singleton logger."""
        reset_slack_audit_logger()

        with patch("aragora.audit.slack_audit.get_audit_log") as mock_get:
            mock_get.return_value = MagicMock(spec=AuditLog)
            logger1 = get_slack_audit_logger()
            logger2 = get_slack_audit_logger()
            assert logger1 is logger2

        reset_slack_audit_logger()

    def test_reset_slack_audit_logger(self):
        """Test resetting the singleton."""
        reset_slack_audit_logger()

        with patch("aragora.audit.slack_audit.get_audit_log") as mock_get:
            mock_get.return_value = MagicMock(spec=AuditLog)
            logger1 = get_slack_audit_logger()
            reset_slack_audit_logger()
            logger2 = get_slack_audit_logger()
            assert logger1 is not logger2

        reset_slack_audit_logger()


# ===========================================================================
# Tests: Hash Chain Integrity
# ===========================================================================


class TestAuditIntegrity:
    """Tests for audit log integrity with Slack events."""

    def test_multiple_commands_maintain_hash_chain(self, slack_logger, mock_audit_log):
        """Test that multiple logged events maintain hash chain integrity."""
        slack_logger.log_command(
            workspace_id="T12345",
            user_id="U67890",
            command="/ask",
        )
        slack_logger.log_command(
            workspace_id="T12345",
            user_id="U67890",
            command="/debate",
        )
        slack_logger.log_rate_limit(
            workspace_id="T12345",
            user_id="U67890",
            command="/ask",
        )

        is_valid, errors = mock_audit_log.verify_integrity()
        assert is_valid is True
        assert len(errors) == 0
