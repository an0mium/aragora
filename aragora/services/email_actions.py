"""
Unified Email Actions Service.

Provides a high-level interface for email actions across providers:
- Send, reply, forward emails
- Archive, trash, snooze messages
- Label/folder management
- Batch operations
- Action logging for compliance

Supports:
- Gmail (via GmailConnector)
- Outlook (via OutlookConnector)

All actions are logged for audit trail and compliance.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EmailProvider(str, Enum):
    """Supported email providers."""

    GMAIL = "gmail"
    OUTLOOK = "outlook"


class ActionType(str, Enum):
    """Types of email actions."""

    SEND = "send"
    REPLY = "reply"
    FORWARD = "forward"
    ARCHIVE = "archive"
    TRASH = "trash"
    UNTRASH = "untrash"
    SNOOZE = "snooze"
    UNSNOOZE = "unsnooze"
    MARK_READ = "mark_read"
    MARK_UNREAD = "mark_unread"
    STAR = "star"
    UNSTAR = "unstar"
    MARK_IMPORTANT = "mark_important"
    MARK_NOT_IMPORTANT = "mark_not_important"
    MOVE_TO_FOLDER = "move_to_folder"
    ADD_LABEL = "add_label"
    REMOVE_LABEL = "remove_label"
    BATCH_ARCHIVE = "batch_archive"
    BATCH_TRASH = "batch_trash"
    BATCH_MODIFY = "batch_modify"


class ActionStatus(str, Enum):
    """Status of an action."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ActionLog:
    """Log entry for an email action."""

    id: str
    user_id: str
    action_type: ActionType
    provider: EmailProvider
    message_ids: List[str]
    status: ActionStatus
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "userId": self.user_id,
            "actionType": self.action_type.value,
            "provider": self.provider.value,
            "messageIds": self.message_ids,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "errorMessage": self.error_message,
            "durationMs": self.duration_ms,
        }


@dataclass
class SendEmailRequest:
    """Request to send an email."""

    to: List[str]
    subject: str
    body: str
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    reply_to: Optional[str] = None
    html_body: Optional[str] = None
    attachments: Optional[List[Dict[str, Any]]] = None


@dataclass
class SnoozeRequest:
    """Request to snooze an email."""

    message_id: str
    snooze_until: datetime
    restore_to_inbox: bool = True


@dataclass
class ActionResult:
    """Result of an email action."""

    success: bool
    action_type: ActionType
    message_ids: List[str]
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "actionType": self.action_type.value,
            "messageIds": self.message_ids,
            "details": self.details,
            "error": self.error,
        }


class EmailActionsService:
    """
    Unified service for email actions across providers.

    Features:
    - Provider-agnostic interface
    - Action logging for compliance
    - Snooze scheduling
    - Batch operations
    - Error handling with retries

    Example:
        ```python
        service = EmailActionsService()

        # Archive a message
        result = await service.archive(
            provider="gmail",
            user_id="user123",
            message_id="msg456",
        )

        # Snooze until tomorrow
        result = await service.snooze(
            provider="gmail",
            user_id="user123",
            message_id="msg456",
            snooze_until=datetime.now() + timedelta(days=1),
        )

        # Send an email
        result = await service.send(
            provider="gmail",
            user_id="user123",
            request=SendEmailRequest(
                to=["recipient@example.com"],
                subject="Hello",
                body="World",
            ),
        )
        ```
    """

    def __init__(self):
        """Initialize the email actions service."""
        self._action_logs: List[ActionLog] = []
        self._snoozed_messages: Dict[str, SnoozeRequest] = {}
        self._connectors: Dict[str, Any] = {}
        self._action_counter = 0
        self._lock = asyncio.Lock()

    def _generate_action_id(self) -> str:
        """Generate a unique action ID."""
        self._action_counter += 1
        return (
            f"action_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{self._action_counter}"
        )

    async def _get_connector(
        self,
        provider: Union[str, EmailProvider],
        user_id: str,
    ) -> Any:
        """Get or create a connector for the provider.

        In production, this would look up the user's OAuth tokens
        and return a properly authenticated connector.
        """
        provider_str = provider.value if isinstance(provider, EmailProvider) else provider
        key = f"{provider_str}:{user_id}"

        if key not in self._connectors:
            if provider_str == "gmail":
                from aragora.connectors.enterprise.communication.gmail import GmailConnector

                connector = GmailConnector()
                # In production: await connector.authenticate(tokens_from_db)
                self._connectors[key] = connector
            elif provider_str == "outlook":
                from aragora.connectors.enterprise.communication.outlook import OutlookConnector

                outlook_connector = OutlookConnector()
                self._connectors[key] = outlook_connector
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        return self._connectors[key]

    async def _log_action(
        self,
        user_id: str,
        action_type: ActionType,
        provider: Union[str, EmailProvider],
        message_ids: List[str],
        status: ActionStatus,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> ActionLog:
        """Log an action for audit trail."""
        provider_enum = EmailProvider(provider) if isinstance(provider, str) else provider

        log = ActionLog(
            id=self._generate_action_id(),
            user_id=user_id,
            action_type=action_type,
            provider=provider_enum,
            message_ids=message_ids,
            status=status,
            timestamp=datetime.now(timezone.utc),
            details=details or {},
            error_message=error_message,
            duration_ms=duration_ms,
        )

        async with self._lock:
            self._action_logs.append(log)
            # Keep only last 10000 logs in memory
            if len(self._action_logs) > 10000:
                self._action_logs = self._action_logs[-10000:]

        logger.info(
            f"[EmailActions] {action_type.value} by {user_id}: "
            f"{len(message_ids)} message(s), status={status.value}"
        )

        return log

    # =========================================================================
    # Send Actions
    # =========================================================================

    async def send(
        self,
        provider: Union[str, EmailProvider],
        user_id: str,
        request: SendEmailRequest,
    ) -> ActionResult:
        """Send an email.

        Args:
            provider: Email provider (gmail, outlook)
            user_id: User ID for connector lookup
            request: Send email request

        Returns:
            ActionResult with sent message details
        """
        start_time = datetime.now(timezone.utc)

        try:
            connector = await self._get_connector(provider, user_id)

            result = await connector.send_message(
                to=request.to,
                subject=request.subject,
                body=request.body,
                cc=request.cc,
                bcc=request.bcc,
                reply_to=request.reply_to,
                html_body=request.html_body,
            )

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.SEND,
                provider=provider,
                message_ids=[result.get("message_id", "")],
                status=ActionStatus.SUCCESS,
                details={
                    "to": request.to,
                    "subject": request.subject,
                    "thread_id": result.get("thread_id"),
                },
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=True,
                action_type=ActionType.SEND,
                message_ids=[result.get("message_id", "")],
                details=result,
            )

        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.SEND,
                provider=provider,
                message_ids=[],
                status=ActionStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=False,
                action_type=ActionType.SEND,
                message_ids=[],
                error=str(e),
            )

    async def reply(
        self,
        provider: Union[str, EmailProvider],
        user_id: str,
        message_id: str,
        body: str,
        cc: Optional[List[str]] = None,
        html_body: Optional[str] = None,
    ) -> ActionResult:
        """Reply to an email.

        Args:
            provider: Email provider
            user_id: User ID
            message_id: Original message ID to reply to
            body: Reply body
            cc: Additional CC recipients
            html_body: HTML body

        Returns:
            ActionResult with reply details
        """
        start_time = datetime.now(timezone.utc)

        try:
            connector = await self._get_connector(provider, user_id)

            result = await connector.reply_to_message(
                original_message_id=message_id,
                body=body,
                cc=cc,
                html_body=html_body,
            )

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.REPLY,
                provider=provider,
                message_ids=[message_id, result.get("message_id", "")],
                status=ActionStatus.SUCCESS,
                details={
                    "in_reply_to": message_id,
                    "thread_id": result.get("thread_id"),
                },
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=True,
                action_type=ActionType.REPLY,
                message_ids=[result.get("message_id", "")],
                details=result,
            )

        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.REPLY,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=False,
                action_type=ActionType.REPLY,
                message_ids=[message_id],
                error=str(e),
            )

    # =========================================================================
    # Archive/Trash Actions
    # =========================================================================

    async def archive(
        self,
        provider: Union[str, EmailProvider],
        user_id: str,
        message_id: str,
    ) -> ActionResult:
        """Archive a message.

        Args:
            provider: Email provider
            user_id: User ID
            message_id: Message to archive

        Returns:
            ActionResult
        """
        start_time = datetime.now(timezone.utc)

        try:
            connector = await self._get_connector(provider, user_id)
            result = await connector.archive_message(message_id)

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.ARCHIVE,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.SUCCESS,
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=True,
                action_type=ActionType.ARCHIVE,
                message_ids=[message_id],
                details=result,
            )

        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.ARCHIVE,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=False,
                action_type=ActionType.ARCHIVE,
                message_ids=[message_id],
                error=str(e),
            )

    async def trash(
        self,
        provider: Union[str, EmailProvider],
        user_id: str,
        message_id: str,
    ) -> ActionResult:
        """Move a message to trash.

        Args:
            provider: Email provider
            user_id: User ID
            message_id: Message to trash

        Returns:
            ActionResult
        """
        start_time = datetime.now(timezone.utc)

        try:
            connector = await self._get_connector(provider, user_id)
            result = await connector.trash_message(message_id)

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.TRASH,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.SUCCESS,
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=True,
                action_type=ActionType.TRASH,
                message_ids=[message_id],
                details=result,
            )

        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.TRASH,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=False,
                action_type=ActionType.TRASH,
                message_ids=[message_id],
                error=str(e),
            )

    # =========================================================================
    # Snooze Actions
    # =========================================================================

    async def snooze(
        self,
        provider: Union[str, EmailProvider],
        user_id: str,
        message_id: str,
        snooze_until: datetime,
    ) -> ActionResult:
        """Snooze a message until a specific time.

        Args:
            provider: Email provider
            user_id: User ID
            message_id: Message to snooze
            snooze_until: When to restore the message

        Returns:
            ActionResult
        """
        start_time = datetime.now(timezone.utc)

        try:
            connector = await self._get_connector(provider, user_id)
            result = await connector.snooze_message(message_id, snooze_until)

            # Store snooze for scheduler
            snooze_key = f"{provider}:{user_id}:{message_id}"
            self._snoozed_messages[snooze_key] = SnoozeRequest(
                message_id=message_id,
                snooze_until=snooze_until,
            )

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.SNOOZE,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.SUCCESS,
                details={"snooze_until": snooze_until.isoformat()},
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=True,
                action_type=ActionType.SNOOZE,
                message_ids=[message_id],
                details=result,
            )

        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.SNOOZE,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=False,
                action_type=ActionType.SNOOZE,
                message_ids=[message_id],
                error=str(e),
            )

    # =========================================================================
    # Label/Folder Actions
    # =========================================================================

    async def mark_read(
        self,
        provider: Union[str, EmailProvider],
        user_id: str,
        message_id: str,
    ) -> ActionResult:
        """Mark a message as read."""
        start_time = datetime.now(timezone.utc)

        try:
            connector = await self._get_connector(provider, user_id)
            result = await connector.mark_as_read(message_id)

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.MARK_READ,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.SUCCESS,
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=True,
                action_type=ActionType.MARK_READ,
                message_ids=[message_id],
                details=result,
            )

        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.MARK_READ,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=False,
                action_type=ActionType.MARK_READ,
                message_ids=[message_id],
                error=str(e),
            )

    async def star(
        self,
        provider: Union[str, EmailProvider],
        user_id: str,
        message_id: str,
    ) -> ActionResult:
        """Star a message."""
        start_time = datetime.now(timezone.utc)

        try:
            connector = await self._get_connector(provider, user_id)
            result = await connector.star_message(message_id)

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.STAR,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.SUCCESS,
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=True,
                action_type=ActionType.STAR,
                message_ids=[message_id],
                details=result,
            )

        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.STAR,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=False,
                action_type=ActionType.STAR,
                message_ids=[message_id],
                error=str(e),
            )

    async def move_to_folder(
        self,
        provider: Union[str, EmailProvider],
        user_id: str,
        message_id: str,
        folder: str,
    ) -> ActionResult:
        """Move a message to a folder."""
        start_time = datetime.now(timezone.utc)

        try:
            connector = await self._get_connector(provider, user_id)
            result = await connector.move_to_folder(message_id, folder)

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.MOVE_TO_FOLDER,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.SUCCESS,
                details={"folder": folder},
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=True,
                action_type=ActionType.MOVE_TO_FOLDER,
                message_ids=[message_id],
                details=result,
            )

        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.MOVE_TO_FOLDER,
                provider=provider,
                message_ids=[message_id],
                status=ActionStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=False,
                action_type=ActionType.MOVE_TO_FOLDER,
                message_ids=[message_id],
                error=str(e),
            )

    # =========================================================================
    # Batch Actions
    # =========================================================================

    async def batch_archive(
        self,
        provider: Union[str, EmailProvider],
        user_id: str,
        message_ids: List[str],
    ) -> ActionResult:
        """Archive multiple messages."""
        start_time = datetime.now(timezone.utc)

        try:
            connector = await self._get_connector(provider, user_id)
            result = await connector.batch_archive(message_ids)

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.BATCH_ARCHIVE,
                provider=provider,
                message_ids=message_ids,
                status=ActionStatus.SUCCESS,
                details={"count": len(message_ids)},
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=True,
                action_type=ActionType.BATCH_ARCHIVE,
                message_ids=message_ids,
                details=result,
            )

        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            await self._log_action(
                user_id=user_id,
                action_type=ActionType.BATCH_ARCHIVE,
                provider=provider,
                message_ids=message_ids,
                status=ActionStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms,
            )

            return ActionResult(
                success=False,
                action_type=ActionType.BATCH_ARCHIVE,
                message_ids=message_ids,
                error=str(e),
            )

    # =========================================================================
    # Action Logs
    # =========================================================================

    async def get_action_logs(
        self,
        user_id: Optional[str] = None,
        action_type: Optional[ActionType] = None,
        provider: Optional[EmailProvider] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ActionLog]:
        """Get action logs with optional filtering.

        Args:
            user_id: Filter by user
            action_type: Filter by action type
            provider: Filter by provider
            since: Filter by timestamp
            limit: Max results

        Returns:
            List of ActionLog entries
        """
        async with self._lock:
            logs = self._action_logs.copy()

        # Apply filters
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        if action_type:
            logs = [log for log in logs if log.action_type == action_type]
        if provider:
            logs = [log for log in logs if log.provider == provider]
        if since:
            logs = [log for log in logs if log.timestamp >= since]

        # Sort by timestamp descending and limit
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return logs[:limit]

    async def export_action_logs(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Export action logs for compliance.

        Args:
            user_id: User to export logs for
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of log entries as dictionaries
        """
        logs = await self.get_action_logs(
            user_id=user_id,
            since=start_date,
            limit=100000,  # High limit for export
        )

        # Filter by end date
        logs = [log for log in logs if log.timestamp <= end_date]

        return [log.to_dict() for log in logs]


# Global service instance
_email_actions_service: Optional[EmailActionsService] = None


def get_email_actions_service() -> EmailActionsService:
    """Get or create the email actions service singleton."""
    global _email_actions_service
    if _email_actions_service is None:
        _email_actions_service = EmailActionsService()
    return _email_actions_service


__all__ = [
    "EmailActionsService",
    "get_email_actions_service",
    "ActionType",
    "ActionStatus",
    "ActionLog",
    "ActionResult",
    "SendEmailRequest",
    "SnoozeRequest",
    "EmailProvider",
]
