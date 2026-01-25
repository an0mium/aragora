"""
Slack Integration Audit Logger.

Provides audit logging for Slack commands, events, and OAuth operations
using the enterprise audit log infrastructure.

Usage:
    from aragora.audit.slack_audit import SlackAuditLogger

    audit = SlackAuditLogger()

    # Log a slash command
    audit.log_command(
        workspace_id="T12345",
        user_id="U67890",
        command="/ask",
        args="What is the meaning of life?",
        result="success",
    )

    # Log an event
    audit.log_event(
        workspace_id="T12345",
        event_type="message",
        payload_summary={"channel": "C12345", "user": "U67890"},
    )

    # Log OAuth attempt
    audit.log_oauth(
        workspace_id="T12345",
        action="install",
        success=True,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .log import (
    AuditCategory,
    AuditEvent,
    AuditLog,
    AuditOutcome,
    get_audit_log,
)

logger = logging.getLogger(__name__)


class SlackAuditLogger:
    """
    Audit logger for Slack integration operations.

    Provides specialized methods for logging:
    - Slash commands
    - Event callbacks
    - OAuth operations
    - Rate limit events
    - Error conditions
    """

    # Resource types for Slack operations
    RESOURCE_SLACK_WORKSPACE = "slack_workspace"
    RESOURCE_SLACK_CHANNEL = "slack_channel"
    RESOURCE_SLACK_USER = "slack_user"
    RESOURCE_SLACK_COMMAND = "slack_command"

    # Action names
    ACTION_COMMAND_EXECUTE = "slack_command_execute"
    ACTION_EVENT_RECEIVE = "slack_event_receive"
    ACTION_OAUTH_INSTALL = "slack_oauth_install"
    ACTION_OAUTH_UNINSTALL = "slack_oauth_uninstall"
    ACTION_OAUTH_TOKEN_REFRESH = "slack_oauth_token_refresh"
    ACTION_RATE_LIMIT = "slack_rate_limit"
    ACTION_SIGNATURE_VERIFY = "slack_signature_verify"

    def __init__(self, audit_log: Optional[AuditLog] = None) -> None:
        """
        Initialize Slack audit logger.

        Args:
            audit_log: Optional AuditLog instance (uses singleton if not provided)
        """
        self._audit = audit_log

    @property
    def audit(self) -> AuditLog:
        """Get the audit log instance (lazy initialization)."""
        if self._audit is None:
            self._audit = get_audit_log()
        return self._audit

    def log_command(
        self,
        workspace_id: str,
        user_id: str,
        command: str,
        args: str = "",
        result: str = "success",
        channel_id: str = "",
        response_time_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> str:
        """
        Log a Slack slash command execution.

        Args:
            workspace_id: Slack workspace ID (team_id)
            user_id: Slack user ID executing the command
            command: The slash command (e.g., "/ask", "/debate")
            args: Command arguments
            result: Execution result ("success", "error", "rate_limited")
            channel_id: Channel where command was executed
            response_time_ms: Response time in milliseconds
            error: Error message if result is not success

        Returns:
            Audit event ID
        """
        outcome = AuditOutcome.SUCCESS
        reason = ""

        if result == "error":
            outcome = AuditOutcome.ERROR
            reason = error or "Command execution failed"
        elif result == "rate_limited":
            outcome = AuditOutcome.DENIED
            reason = "Rate limit exceeded"
        elif result == "denied":
            outcome = AuditOutcome.DENIED
            reason = error or "Command not permitted"

        details: dict[str, Any] = {
            "command": command,
            "args_length": len(args) if args else 0,
            "channel_id": channel_id,
        }
        if response_time_ms is not None:
            details["response_time_ms"] = response_time_ms

        event = AuditEvent(
            category=AuditCategory.API,
            action=self.ACTION_COMMAND_EXECUTE,
            actor_id=user_id,
            resource_type=self.RESOURCE_SLACK_COMMAND,
            resource_id=command,
            outcome=outcome,
            workspace_id=workspace_id,
            details=details,
            reason=reason,
        )

        event_id = self.audit.log(event)
        logger.debug(
            f"slack_command_logged workspace={workspace_id} user={user_id} "
            f"command={command} result={result}"
        )
        return event_id

    def log_event(
        self,
        workspace_id: str,
        event_type: str,
        payload_summary: dict[str, Any],
        user_id: str = "",
        channel_id: str = "",
        success: bool = True,
        error: Optional[str] = None,
    ) -> str:
        """
        Log a Slack event callback.

        Args:
            workspace_id: Slack workspace ID
            event_type: Slack event type (e.g., "message", "app_mention")
            payload_summary: Summary of event payload (sanitized, no PII)
            user_id: User who triggered the event (if applicable)
            channel_id: Channel where event occurred
            success: Whether event was processed successfully
            error: Error message if processing failed

        Returns:
            Audit event ID
        """
        outcome = AuditOutcome.SUCCESS if success else AuditOutcome.ERROR

        details: dict[str, Any] = {
            "event_type": event_type,
            "channel_id": channel_id,
            **payload_summary,
        }

        event = AuditEvent(
            category=AuditCategory.API,
            action=self.ACTION_EVENT_RECEIVE,
            actor_id=user_id or "slack_event",
            resource_type=self.RESOURCE_SLACK_WORKSPACE,
            resource_id=workspace_id,
            outcome=outcome,
            workspace_id=workspace_id,
            details=details,
            reason=error or "",
        )

        event_id = self.audit.log(event)
        logger.debug(
            f"slack_event_logged workspace={workspace_id} type={event_type} success={success}"
        )
        return event_id

    def log_oauth(
        self,
        workspace_id: str,
        action: str,
        success: bool,
        error: Optional[str] = None,
        user_id: str = "",
        scopes: Optional[list[str]] = None,
    ) -> str:
        """
        Log a Slack OAuth operation.

        Args:
            workspace_id: Slack workspace ID (empty for failed installs)
            action: OAuth action ("install", "uninstall", "token_refresh")
            success: Whether the operation succeeded
            error: Error message if operation failed
            user_id: User who initiated the operation
            scopes: OAuth scopes granted (for successful installs)

        Returns:
            Audit event ID
        """
        action_map = {
            "install": self.ACTION_OAUTH_INSTALL,
            "uninstall": self.ACTION_OAUTH_UNINSTALL,
            "token_refresh": self.ACTION_OAUTH_TOKEN_REFRESH,
        }
        audit_action = action_map.get(action, f"slack_oauth_{action}")

        outcome = AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE

        details: dict[str, Any] = {"oauth_action": action}
        if scopes:
            details["scopes"] = scopes

        # For security logging, OAuth operations are critical
        category = (
            AuditCategory.SECURITY if action in ("install", "uninstall") else AuditCategory.AUTH
        )

        event = AuditEvent(
            category=category,
            action=audit_action,
            actor_id=user_id or "oauth_flow",
            resource_type=self.RESOURCE_SLACK_WORKSPACE,
            resource_id=workspace_id or "unknown",
            outcome=outcome,
            workspace_id=workspace_id,
            details=details,
            reason=error or "",
        )

        event_id = self.audit.log(event)
        logger.info(
            f"slack_oauth_logged workspace={workspace_id or 'unknown'} "
            f"action={action} success={success}"
        )
        return event_id

    def log_rate_limit(
        self,
        workspace_id: str,
        user_id: str,
        command: str,
        limit_type: str = "user",
    ) -> str:
        """
        Log a rate limit event.

        Args:
            workspace_id: Slack workspace ID
            user_id: User who hit the rate limit
            command: Command that was rate limited
            limit_type: Type of rate limit ("user", "workspace", "global")

        Returns:
            Audit event ID
        """
        event = AuditEvent(
            category=AuditCategory.SECURITY,
            action=self.ACTION_RATE_LIMIT,
            actor_id=user_id,
            resource_type=self.RESOURCE_SLACK_COMMAND,
            resource_id=command,
            outcome=AuditOutcome.DENIED,
            workspace_id=workspace_id,
            details={"limit_type": limit_type, "command": command},
            reason=f"Rate limit exceeded ({limit_type})",
        )

        event_id = self.audit.log(event)
        logger.warning(
            f"slack_rate_limit workspace={workspace_id} user={user_id} "
            f"command={command} type={limit_type}"
        )
        return event_id

    def log_signature_failure(
        self,
        workspace_id: str,
        ip_address: str = "",
        user_agent: str = "",
    ) -> str:
        """
        Log a Slack signature verification failure (potential attack).

        Args:
            workspace_id: Claimed workspace ID (may be spoofed)
            ip_address: Source IP address
            user_agent: Request user agent

        Returns:
            Audit event ID
        """
        event = AuditEvent(
            category=AuditCategory.SECURITY,
            action=self.ACTION_SIGNATURE_VERIFY,
            actor_id="unknown",
            resource_type=self.RESOURCE_SLACK_WORKSPACE,
            resource_id=workspace_id or "unknown",
            outcome=AuditOutcome.DENIED,
            ip_address=ip_address,
            user_agent=user_agent,
            workspace_id=workspace_id,
            details={"suspected_attack": True},
            reason="Slack request signature verification failed",
        )

        event_id = self.audit.log(event)
        logger.error(
            f"slack_signature_failure workspace={workspace_id or 'unknown'} "
            f"ip={ip_address} - POTENTIAL ATTACK"
        )
        return event_id


# Module-level singleton
_slack_audit_logger: Optional[SlackAuditLogger] = None


def get_slack_audit_logger() -> SlackAuditLogger:
    """Get the singleton Slack audit logger."""
    global _slack_audit_logger
    if _slack_audit_logger is None:
        _slack_audit_logger = SlackAuditLogger()
    return _slack_audit_logger


def reset_slack_audit_logger() -> None:
    """Reset the singleton (for testing)."""
    global _slack_audit_logger
    _slack_audit_logger = None


__all__ = [
    "SlackAuditLogger",
    "get_slack_audit_logger",
    "reset_slack_audit_logger",
]
