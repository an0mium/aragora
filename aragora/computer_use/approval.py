"""
Human-in-the-Loop Approval Workflow for Computer Use.

Provides approval mechanisms for sensitive computer-use actions:
- Async approval requests with configurable timeout
- Multi-approver support
- Approval audit trails
- Integration with notification systems

Safety: Sensitive actions require explicit human approval before execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalPriority(str, Enum):
    """Priority level of approval request."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalCategory(str, Enum):
    """Category of action requiring approval."""

    CREDENTIAL_ACCESS = "credential_access"
    SENSITIVE_DATA = "sensitive_data"
    DESTRUCTIVE_ACTION = "destructive_action"
    EXTERNAL_SYSTEM = "external_system"
    FINANCIAL = "financial"
    PII_ACCESS = "pii_access"
    SYSTEM_MODIFICATION = "system_modification"
    UNKNOWN = "unknown"


@dataclass
class ApprovalContext:
    """Context for an approval request."""

    task_id: str
    action_type: str
    action_details: dict[str, Any]
    category: ApprovalCategory
    reason: str
    risk_level: str = "medium"
    screenshot_b64: str | None = None
    current_url: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalRequest:
    """A request for human approval."""

    id: str
    context: ApprovalContext
    status: ApprovalStatus
    priority: ApprovalPriority
    created_at: float
    expires_at: float
    approved_by: str | None = None
    denied_by: str | None = None
    decision_at: float | None = None
    decision_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if request has expired."""
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task_id": self.context.task_id,
            "action_type": self.context.action_type,
            "category": self.context.category.value,
            "reason": self.context.reason,
            "risk_level": self.context.risk_level,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "approved_by": self.approved_by,
            "denied_by": self.denied_by,
            "decision_at": self.decision_at,
            "decision_reason": self.decision_reason,
            "has_screenshot": self.context.screenshot_b64 is not None,
            "current_url": self.context.current_url,
        }


@dataclass
class ApprovalConfig:
    """Configuration for approval workflow."""

    default_timeout_seconds: float = 300.0
    max_timeout_seconds: float = 3600.0
    min_timeout_seconds: float = 30.0
    require_reason: bool = False
    require_multiple_approvers: bool = False
    min_approvers: int = 1
    auto_approve_low_risk: bool = False
    auto_approve_same_user: bool = False
    notify_on_request: bool = True
    notify_on_decision: bool = True
    notify_on_expiry: bool = True
    log_all_requests: bool = True
    log_screenshots: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ApprovalNotifier(ABC):
    """Abstract base class for approval notifiers."""

    @abstractmethod
    async def notify_request(self, request: ApprovalRequest) -> None:
        """Notify about a new approval request."""
        ...

    @abstractmethod
    async def notify_decision(self, request: ApprovalRequest) -> None:
        """Notify about an approval decision."""
        ...

    @abstractmethod
    async def notify_expiry(self, request: ApprovalRequest) -> None:
        """Notify about an expired request."""
        ...


class LoggingNotifier(ApprovalNotifier):
    """Simple logging-based notifier."""

    async def notify_request(self, request: ApprovalRequest) -> None:
        """Log new approval request."""
        logger.info(
            f"[Approval Request] {request.id}: {request.context.action_type} "
            f"- {request.context.reason} (priority: {request.priority.value})"
        )

    async def notify_decision(self, request: ApprovalRequest) -> None:
        """Log approval decision."""
        decision = request.status.value.upper()
        by = request.approved_by or request.denied_by or "unknown"
        logger.info(f"[Approval {decision}] {request.id} by {by}")

    async def notify_expiry(self, request: ApprovalRequest) -> None:
        """Log expired request."""
        logger.warning(f"[Approval Expired] {request.id}: {request.context.action_type}")


class WebhookNotifier(ApprovalNotifier):
    """Webhook-based notifier for external systems."""

    def __init__(self, webhook_url: str, auth_token: str | None = None) -> None:
        self._webhook_url = webhook_url
        self._auth_token = auth_token

    async def _send_webhook(self, payload: dict[str, Any]) -> None:
        """Send webhook notification."""
        try:
            import aiohttp

            headers = {"Content-Type": "application/json"}
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=10.0,
                ) as response:
                    if not response.ok:
                        logger.warning(f"Webhook failed: {response.status}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")

    async def notify_request(self, request: ApprovalRequest) -> None:
        """Send webhook for new request."""
        await self._send_webhook({"event": "approval_request", "request": request.to_dict()})

    async def notify_decision(self, request: ApprovalRequest) -> None:
        """Send webhook for decision."""
        await self._send_webhook({"event": "approval_decision", "request": request.to_dict()})

    async def notify_expiry(self, request: ApprovalRequest) -> None:
        """Send webhook for expiry."""
        await self._send_webhook({"event": "approval_expired", "request": request.to_dict()})


class ApprovalWorkflow:
    """Manages human-in-the-loop approval workflow for computer-use."""

    def __init__(
        self,
        config: ApprovalConfig | None = None,
        notifiers: list[ApprovalNotifier] | None = None,
    ) -> None:
        self._config = config or ApprovalConfig()
        self._notifiers = notifiers or [LoggingNotifier()]
        self._requests: dict[str, ApprovalRequest] = {}
        self._pending_events: dict[str, asyncio.Event] = {}
        self._callbacks: dict[str, list[Callable]] = {}
        self._lock = asyncio.Lock()
        self._expiry_tasks: dict[str, asyncio.Task] = {}

    async def request_approval(
        self,
        context: ApprovalContext,
        priority: ApprovalPriority = ApprovalPriority.MEDIUM,
        timeout_seconds: float | None = None,
    ) -> ApprovalRequest:
        """Create a new approval request."""
        timeout = timeout_seconds or self._config.default_timeout_seconds
        timeout = max(
            self._config.min_timeout_seconds,
            min(timeout, self._config.max_timeout_seconds),
        )

        request_id = str(uuid.uuid4())
        now = time.time()

        request = ApprovalRequest(
            id=request_id,
            context=context,
            status=ApprovalStatus.PENDING,
            priority=priority,
            created_at=now,
            expires_at=now + timeout,
        )

        async with self._lock:
            self._requests[request_id] = request
            self._pending_events[request_id] = asyncio.Event()

        if self._config.notify_on_request:
            await self._notify_all("request", request)

        task = asyncio.create_task(self._handle_expiry(request_id, timeout))
        self._expiry_tasks[request_id] = task

        if self._config.log_all_requests:
            logger.info(
                f"Approval request created: {request_id} "
                f"for {context.action_type} ({context.category.value})"
            )

        return request

    async def wait_for_decision(
        self,
        request_id: str,
        timeout: float | None = None,
    ) -> ApprovalStatus:
        """Wait for an approval decision."""
        event = self._pending_events.get(request_id)
        if not event:
            request = self._requests.get(request_id)
            if request:
                return request.status
            raise ValueError(f"Request not found: {request_id}")

        try:
            request = self._requests.get(request_id)
            if not request:
                raise ValueError(f"Request not found: {request_id}")

            if timeout is None:
                timeout = request.expires_at - time.time()

            timeout = max(0.1, timeout)
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

        request = self._requests.get(request_id)
        return request.status if request else ApprovalStatus.EXPIRED

    async def approve(
        self,
        request_id: str,
        approver_id: str,
        reason: str | None = None,
    ) -> bool:
        """Approve a request."""
        async with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            if request.status != ApprovalStatus.PENDING:
                return False

            if request.is_expired():
                request.status = ApprovalStatus.EXPIRED
                return False

            request.status = ApprovalStatus.APPROVED
            request.approved_by = approver_id
            request.decision_at = time.time()
            request.decision_reason = reason

        event = self._pending_events.get(request_id)
        if event:
            event.set()

        expiry_task = self._expiry_tasks.pop(request_id, None)
        if expiry_task:
            expiry_task.cancel()

        if self._config.notify_on_decision:
            await self._notify_all("decision", request)

        logger.info(f"Request {request_id} approved by {approver_id}")
        return True

    async def deny(
        self,
        request_id: str,
        denier_id: str,
        reason: str | None = None,
    ) -> bool:
        """Deny a request."""
        async with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            if request.status != ApprovalStatus.PENDING:
                return False

            request.status = ApprovalStatus.DENIED
            request.denied_by = denier_id
            request.decision_at = time.time()
            request.decision_reason = reason

        event = self._pending_events.get(request_id)
        if event:
            event.set()

        expiry_task = self._expiry_tasks.pop(request_id, None)
        if expiry_task:
            expiry_task.cancel()

        if self._config.notify_on_decision:
            await self._notify_all("decision", request)

        logger.info(f"Request {request_id} denied by {denier_id}")
        return True

    async def cancel(self, request_id: str) -> bool:
        """Cancel a pending request."""
        async with self._lock:
            request = self._requests.get(request_id)
            if not request or request.status != ApprovalStatus.PENDING:
                return False

            request.status = ApprovalStatus.CANCELLED
            request.decision_at = time.time()

        event = self._pending_events.get(request_id)
        if event:
            event.set()

        expiry_task = self._expiry_tasks.pop(request_id, None)
        if expiry_task:
            expiry_task.cancel()

        logger.info(f"Request {request_id} cancelled")
        return True

    async def _handle_expiry(self, request_id: str, timeout: float) -> None:
        """Handle request expiry after timeout."""
        try:
            await asyncio.sleep(timeout)

            async with self._lock:
                request = self._requests.get(request_id)
                if request and request.status == ApprovalStatus.PENDING:
                    request.status = ApprovalStatus.EXPIRED
                    request.decision_at = time.time()

            event = self._pending_events.get(request_id)
            if event:
                event.set()

            if self._config.notify_on_expiry:
                request = self._requests.get(request_id)
                if request:
                    await self._notify_all("expiry", request)

            logger.warning(f"Request {request_id} expired")

        except asyncio.CancelledError:
            pass
        finally:
            # Ensure expired/cancelled tasks are removed from tracking
            self._expiry_tasks.pop(request_id, None)

    async def _notify_all(self, event_type: str, request: ApprovalRequest) -> None:
        """Send notifications to all notifiers."""
        for notifier in self._notifiers:
            try:
                if event_type == "request":
                    await notifier.notify_request(request)
                elif event_type == "decision":
                    await notifier.notify_decision(request)
                elif event_type == "expiry":
                    await notifier.notify_expiry(request)
            except Exception as e:
                logger.error(f"Notifier error: {e}")

    async def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get a request by ID."""
        return self._requests.get(request_id)

    async def list_pending(self) -> list[ApprovalRequest]:
        """List all pending approval requests."""
        return [
            r
            for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING and not r.is_expired()
        ]

    async def list_all(
        self,
        limit: int = 100,
        status: ApprovalStatus | None = None,
    ) -> list[ApprovalRequest]:
        """List approval requests with optional filtering."""
        requests = list(self._requests.values())

        if status:
            requests = [r for r in requests if r.status == status]

        requests.sort(key=lambda r: r.created_at, reverse=True)
        return requests[:limit]

    async def cleanup_expired(self, max_age_seconds: float = 3600) -> int:
        """Remove old expired/completed requests."""
        cutoff = time.time() - max_age_seconds
        to_remove = []

        for request_id, request in self._requests.items():
            if request.status != ApprovalStatus.PENDING and request.created_at < cutoff:
                to_remove.append(request_id)

        for request_id in to_remove:
            self._requests.pop(request_id, None)
            self._pending_events.pop(request_id, None)
            expiry_task = self._expiry_tasks.pop(request_id, None)
            if expiry_task:
                expiry_task.cancel()

        return len(to_remove)

    async def get_stats(self) -> dict[str, Any]:
        """Get approval workflow statistics."""
        by_status: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_priority: dict[str, int] = {}

        for request in self._requests.values():
            status = request.status.value
            by_status[status] = by_status.get(status, 0) + 1

            category = request.context.category.value
            by_category[category] = by_category.get(category, 0) + 1

            priority = request.priority.value
            by_priority[priority] = by_priority.get(priority, 0) + 1

        pending = [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]
        avg_wait_time = 0.0
        if pending:
            total_wait = sum(time.time() - r.created_at for r in pending)
            avg_wait_time = total_wait / len(pending)

        return {
            "total_requests": len(self._requests),
            "pending_count": len(pending),
            "by_status": by_status,
            "by_category": by_category,
            "by_priority": by_priority,
            "avg_wait_time_seconds": avg_wait_time,
        }


def create_approval_workflow(
    webhook_url: str | None = None,
    require_reason: bool = False,
    timeout_seconds: float = 300.0,
) -> ApprovalWorkflow:
    """Create an approval workflow with common settings."""
    config = ApprovalConfig(
        default_timeout_seconds=timeout_seconds,
        require_reason=require_reason,
    )

    notifiers: list[ApprovalNotifier] = [LoggingNotifier()]
    if webhook_url:
        notifiers.append(WebhookNotifier(webhook_url))

    return ApprovalWorkflow(config=config, notifiers=notifiers)
