"""
RBAC Approval Workflows.

Implements access request and approval workflows for enterprise RBAC:
- Request elevated permissions
- Multi-approver workflows
- Time-limited access grants
- Audit trail for all decisions

Usage:
    from aragora.rbac.approvals import ApprovalWorkflow, ApprovalRequest

    workflow = ApprovalWorkflow()

    # Request access
    request = await workflow.request_access(
        requester_id="user-123",
        permission="debates:delete",
        resource_type="debates",
        resource_id="debate-456",
        justification="Need to clean up test debates",
    )

    # Approve request (by an approver)
    await workflow.approve(
        ctx=approver_context,
        request_id=request.id,
        comment="Approved for testing purposes",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class ApprovalDecision:
    """Individual approver's decision on a request."""

    approver_id: str
    decision: str  # "approved" or "rejected"
    comment: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "approver_id": self.approver_id,
            "decision": self.decision,
            "comment": self.comment,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ApprovalRequest:
    """Request for elevated permissions or resource access."""

    id: str
    requester_id: str
    permission: str
    resource_type: str
    resource_id: Optional[str]
    justification: str
    status: ApprovalStatus
    approvers: list[str]  # User IDs who can approve
    required_approvals: int  # Number of approvals needed
    decisions: list[ApprovalDecision] = field(default_factory=list)
    org_id: Optional[str] = None
    workspace_id: Optional[str] = None
    duration_hours: int = 24  # How long access lasts once approved
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=7)
    )
    resolved_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set default expiration if not provided."""
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(days=7)

    @property
    def approval_count(self) -> int:
        """Count of approvals received."""
        return sum(1 for d in self.decisions if d.decision == "approved")

    @property
    def rejection_count(self) -> int:
        """Count of rejections received."""
        return sum(1 for d in self.decisions if d.decision == "rejected")

    @property
    def is_approved(self) -> bool:
        """Check if request has enough approvals."""
        return self.approval_count >= self.required_approvals

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "requester_id": self.requester_id,
            "permission": self.permission,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "justification": self.justification,
            "status": self.status.value,
            "approvers": self.approvers,
            "required_approvals": self.required_approvals,
            "decisions": [d.to_dict() for d in self.decisions],
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "duration_hours": self.duration_hours,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "approval_count": self.approval_count,
            "rejection_count": self.rejection_count,
            "metadata": self.metadata,
        }


class ApprovalWorkflow:
    """
    Manages approval workflows for permission elevation.

    Features:
    - Multi-approver support (require N of M approvals)
    - Time-limited access grants
    - Automatic expiration of pending requests
    - Full audit trail
    """

    # Default expiration for pending requests (days)
    DEFAULT_REQUEST_EXPIRY_DAYS = 7

    # Maximum access duration (hours)
    MAX_ACCESS_DURATION_HOURS = 24 * 30  # 30 days

    # Default required approvals
    DEFAULT_REQUIRED_APPROVALS = 1

    def __init__(self):
        """Initialize the approval workflow engine."""
        self._requests: dict[str, ApprovalRequest] = {}
        self._by_requester: dict[str, list[str]] = {}  # requester_id -> request_ids
        self._by_approver: dict[str, list[str]] = {}  # approver_id -> request_ids

    async def request_access(
        self,
        requester_id: str,
        permission: str,
        resource_type: str,
        justification: str,
        resource_id: Optional[str] = None,
        approvers: Optional[list[str]] = None,
        required_approvals: int = 1,
        duration_hours: int = 24,
        org_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """
        Create a new access request.

        Args:
            requester_id: User requesting access
            permission: Permission being requested
            resource_type: Type of resource
            justification: Reason for the request
            resource_id: Specific resource ID (optional)
            approvers: List of user IDs who can approve
            required_approvals: Number of approvals needed
            duration_hours: How long access should last
            org_id: Organization context
            workspace_id: Workspace context
            metadata: Additional metadata

        Returns:
            Created ApprovalRequest
        """
        # Validate duration
        if duration_hours > self.MAX_ACCESS_DURATION_HOURS:
            raise ValueError(f"Duration cannot exceed {self.MAX_ACCESS_DURATION_HOURS} hours")

        # Default approvers if not specified
        if not approvers:
            approvers = await self._get_default_approvers(
                permission, resource_type, org_id, workspace_id
            )

        # Validate we have approvers
        if not approvers:
            raise ValueError("No approvers available for this request")

        # Create request
        request = ApprovalRequest(
            id=f"req-{uuid4().hex[:12]}",
            requester_id=requester_id,
            permission=permission,
            resource_type=resource_type,
            resource_id=resource_id,
            justification=justification,
            status=ApprovalStatus.PENDING,
            approvers=approvers,
            required_approvals=min(required_approvals, len(approvers)),
            duration_hours=duration_hours,
            org_id=org_id,
            workspace_id=workspace_id,
            metadata=metadata or {},
        )

        # Store request
        self._requests[request.id] = request

        # Index by requester
        if requester_id not in self._by_requester:
            self._by_requester[requester_id] = []
        self._by_requester[requester_id].append(request.id)

        # Index by approvers
        for approver_id in approvers:
            if approver_id not in self._by_approver:
                self._by_approver[approver_id] = []
            self._by_approver[approver_id].append(request.id)

        logger.info(
            f"Access request created: id={request.id}, "
            f"requester={requester_id}, permission={permission}"
        )

        # Log audit event
        await self._audit_log(
            "access_request_created",
            request_id=request.id,
            requester_id=requester_id,
            permission=permission,
            resource_type=resource_type,
            resource_id=resource_id,
        )

        return request

    async def approve(
        self,
        approver_id: str,
        request_id: str,
        comment: Optional[str] = None,
    ) -> ApprovalRequest:
        """
        Approve an access request.

        Args:
            approver_id: User approving the request
            request_id: Request to approve
            comment: Optional approval comment

        Returns:
            Updated ApprovalRequest

        Raises:
            ValueError: If request not found or user cannot approve
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        # Validate approver
        if approver_id not in request.approvers:
            raise ValueError(f"User {approver_id} is not an approver for this request")

        # Check if already decided
        existing = [d for d in request.decisions if d.approver_id == approver_id]
        if existing:
            raise ValueError(f"User {approver_id} has already made a decision")

        # Check status
        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request is not pending: {request.status.value}")

        # Check expiration
        if request.is_expired:
            request.status = ApprovalStatus.EXPIRED
            raise ValueError("Request has expired")

        # Record decision
        decision = ApprovalDecision(
            approver_id=approver_id,
            decision="approved",
            comment=comment,
        )
        request.decisions.append(decision)

        # Check if fully approved
        if request.is_approved:
            request.status = ApprovalStatus.APPROVED
            request.resolved_at = datetime.now(timezone.utc)

            # Grant the permission
            await self._grant_temporary_permission(request)

            logger.info(f"Access request approved: id={request_id}")

        # Audit log
        await self._audit_log(
            "access_request_approved",
            request_id=request_id,
            approver_id=approver_id,
            comment=comment,
            fully_approved=request.is_approved,
        )

        return request

    async def reject(
        self,
        approver_id: str,
        request_id: str,
        reason: str,
    ) -> ApprovalRequest:
        """
        Reject an access request.

        Args:
            approver_id: User rejecting the request
            request_id: Request to reject
            reason: Reason for rejection

        Returns:
            Updated ApprovalRequest

        Raises:
            ValueError: If request not found or user cannot reject
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        # Validate approver
        if approver_id not in request.approvers:
            raise ValueError(f"User {approver_id} is not an approver for this request")

        # Check status
        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request is not pending: {request.status.value}")

        # Record decision
        decision = ApprovalDecision(
            approver_id=approver_id,
            decision="rejected",
            comment=reason,
        )
        request.decisions.append(decision)

        # Mark as rejected (any rejection = request rejected)
        request.status = ApprovalStatus.REJECTED
        request.resolved_at = datetime.now(timezone.utc)

        logger.info(f"Access request rejected: id={request_id}, reason={reason}")

        # Audit log
        await self._audit_log(
            "access_request_rejected",
            request_id=request_id,
            approver_id=approver_id,
            reason=reason,
        )

        return request

    async def cancel(
        self,
        requester_id: str,
        request_id: str,
        reason: Optional[str] = None,
    ) -> ApprovalRequest:
        """
        Cancel a pending access request.

        Args:
            requester_id: User cancelling (must be requester)
            request_id: Request to cancel
            reason: Optional cancellation reason

        Returns:
            Updated ApprovalRequest
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        # Validate requester
        if request.requester_id != requester_id:
            raise ValueError("Only the requester can cancel a request")

        # Check status
        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request is not pending: {request.status.value}")

        request.status = ApprovalStatus.CANCELLED
        request.resolved_at = datetime.now(timezone.utc)
        request.metadata["cancellation_reason"] = reason

        logger.info(f"Access request cancelled: id={request_id}")

        await self._audit_log(
            "access_request_cancelled",
            request_id=request_id,
            requester_id=requester_id,
            reason=reason,
        )

        return request

    async def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a request by ID."""
        return self._requests.get(request_id)

    async def get_pending_for_approver(
        self,
        approver_id: str,
        limit: int = 100,
    ) -> list[ApprovalRequest]:
        """
        Get pending requests for an approver.

        Args:
            approver_id: User who can approve
            limit: Maximum results

        Returns:
            List of pending ApprovalRequests
        """
        request_ids = self._by_approver.get(approver_id, [])
        requests = []

        for rid in request_ids:
            request = self._requests.get(rid)
            if request and request.status == ApprovalStatus.PENDING:
                # Check if approver hasn't already decided
                has_decided = any(d.approver_id == approver_id for d in request.decisions)
                if not has_decided:
                    requests.append(request)

            if len(requests) >= limit:
                break

        return requests

    async def get_requests_by_requester(
        self,
        requester_id: str,
        status: Optional[ApprovalStatus] = None,
        limit: int = 100,
    ) -> list[ApprovalRequest]:
        """
        Get requests submitted by a user.

        Args:
            requester_id: User who submitted requests
            status: Filter by status (optional)
            limit: Maximum results

        Returns:
            List of ApprovalRequests
        """
        request_ids = self._by_requester.get(requester_id, [])
        requests = []

        for rid in request_ids:
            request = self._requests.get(rid)
            if request:
                if status is None or request.status == status:
                    requests.append(request)

            if len(requests) >= limit:
                break

        return sorted(requests, key=lambda r: r.created_at, reverse=True)

    async def expire_old_requests(self) -> int:
        """
        Expire requests that have passed their expiration time.

        Returns:
            Number of requests expired
        """
        count = 0
        now = datetime.now(timezone.utc)

        for request in self._requests.values():
            if request.status == ApprovalStatus.PENDING and request.expires_at < now:
                request.status = ApprovalStatus.EXPIRED
                request.resolved_at = now
                count += 1

                await self._audit_log(
                    "access_request_expired",
                    request_id=request.id,
                )

        if count > 0:
            logger.info(f"Expired {count} old access requests")

        return count

    async def _get_default_approvers(
        self,
        permission: str,
        resource_type: str,
        org_id: Optional[str],
        workspace_id: Optional[str],
    ) -> list[str]:
        """
        Get default approvers for a permission request.

        Finds users with admin permission for the resource type using
        the RBAC PermissionChecker's reverse lookup.

        Args:
            permission: The permission being requested
            resource_type: Type of resource (e.g., "debates", "workspaces")
            org_id: Optional organization filter
            workspace_id: Optional workspace filter

        Returns:
            List of user IDs who can approve this request
        """
        try:
            from aragora.rbac.checker import get_permission_checker

            checker = get_permission_checker()

            # Look for users with admin permission on this resource type
            admin_permission = f"{resource_type}.admin"

            approvers = checker.get_users_with_permission(
                admin_permission,
                org_id=org_id,
                workspace_id=workspace_id,
                limit=10,  # Reasonable limit for approvers
            )

            if approvers:
                logger.debug(
                    f"Found {len(approvers)} approvers for {admin_permission}: {approvers}"
                )
                return approvers

            # Fallback: try wildcard admin permission
            wildcard_permission = f"{resource_type}.*"
            approvers = checker.get_users_with_permission(
                wildcard_permission,
                org_id=org_id,
                workspace_id=workspace_id,
                limit=10,
            )

            return approvers

        except ImportError as e:
            logger.warning(f"PermissionChecker not available: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error finding default approvers: {e}")
            return []

    async def _grant_temporary_permission(self, request: ApprovalRequest) -> None:
        """
        Grant temporary permission after approval.

        This should integrate with the RBAC resource permissions system.
        """
        try:
            from aragora.rbac.resource_permissions import ResourcePermissionStore

            store = ResourcePermissionStore()

            # Calculate expiration
            expires_at = datetime.now(timezone.utc) + timedelta(hours=request.duration_hours)

            # Grant the permission
            await store.grant(
                user_id=request.requester_id,
                permission=request.permission,
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                granted_by="approval_workflow",
                expires_at=expires_at,
                conditions={"approval_request_id": request.id},
            )

            logger.info(
                f"Granted temporary permission: user={request.requester_id}, "
                f"permission={request.permission}, expires={expires_at}"
            )

        except ImportError:
            logger.warning("ResourcePermissionStore not available, permission grant skipped")

    async def _audit_log(self, event_type: str, **kwargs) -> None:
        """Log an audit event for approval workflow actions."""
        try:
            from aragora.rbac.audit import AuthorizationAuditor

            auditor = AuthorizationAuditor()
            await auditor.log_event(
                event_type=event_type,
                details=kwargs,
            )
        except ImportError:
            # Fallback to standard logging
            logger.info(f"Approval audit: {event_type} - {kwargs}")


# Singleton instance
_workflow: Optional[ApprovalWorkflow] = None


def get_approval_workflow() -> ApprovalWorkflow:
    """Get the global ApprovalWorkflow instance."""
    global _workflow
    if _workflow is None:
        _workflow = ApprovalWorkflow()
    return _workflow


__all__ = [
    "ApprovalStatus",
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalWorkflow",
    "get_approval_workflow",
]
