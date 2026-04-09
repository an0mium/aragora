"""
Durable capability approvals and dispatch audit records.

This module provides a small persistent gate for consequential capabilities
such as code execution, browser automation, git push, and chain writes.
It is intentionally storage-backed so approval and dispatch state survives
process restarts in production.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from aragora.storage.backends import DatabaseBackend, get_database_backend

logger = logging.getLogger(__name__)

UTC = timezone.utc


class Capability(str, Enum):
    """Consequential capabilities that require explicit approval."""

    CODE_EXEC = "code_exec"
    BROWSER_EXEC = "browser_exec"
    GIT_WRITE = "git_write"
    GIT_PUSH = "git_push"
    WEBHOOK_EMIT = "webhook_emit"
    CHAIN_WRITE = "chain_write"


class ApprovalStatus(str, Enum):
    """Lifecycle state for a capability approval."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class ActionStatus(str, Enum):
    """Lifecycle state for a gated action dispatch."""

    PENDING = "pending"
    ALLOWED = "allowed"
    DENIED = "denied"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass(slots=True)
class CapabilityApprovalRecord:
    """Durable approval record for a consequential capability."""

    approval_id: str
    capability: Capability
    actor_id: str
    target_resource: str
    normalized_input_hash: str
    receipt_id: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    expires_at: str = field(
        default_factory=lambda: (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    )
    approved_by: str = ""
    approved_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CapabilityActionRecord:
    """Durable dispatch record for a capability invocation."""

    action_id: str
    capability: Capability
    actor_id: str
    target_resource: str
    normalized_input_hash: str
    receipt_id: str = ""
    approval_id: str = ""
    status: ActionStatus = ActionStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class CapabilityApprovalRequiredError(PermissionError):
    """Raised when a consequential capability lacks a valid approval record."""

    def __init__(self, capability: Capability, reason: str, action_id: str = "") -> None:
        self.capability = capability
        self.reason = reason
        self.action_id = action_id
        super().__init__(f"{capability.value} requires explicit approval: {reason}")


def _is_dev_mode() -> bool:
    env = (
        str(
            os.getenv("ARAGORA_ENV")
            or os.getenv("ARAGORA_ENVIRONMENT")
            or os.getenv("NODE_ENV")
            or ""
        )
        .strip()
        .lower()
    )
    if not env:
        return False
    return env in {"dev", "development", "local", "test", "testing"}


def normalize_input_hash(payload: Any) -> str:
    """Compute a stable SHA-256 hash for an approval-sensitive input payload."""

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = str(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Failed to parse capability gate JSON payload", exc_info=True)
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "Capability gate JSON payload was not an object: %s",
            type(data).__name__,
        )
        return {}
    return data


def _parse_capability(row_value: Any) -> Capability:
    return row_value if isinstance(row_value, Capability) else Capability(str(row_value))


def _parse_approval_status(row_value: Any) -> ApprovalStatus:
    return row_value if isinstance(row_value, ApprovalStatus) else ApprovalStatus(str(row_value))


def _parse_action_status(row_value: Any) -> ActionStatus:
    return row_value if isinstance(row_value, ActionStatus) else ActionStatus(str(row_value))


class CapabilityGateStore:
    """Storage-backed approvals and action audit records."""

    def __init__(self, backend: DatabaseBackend | None = None) -> None:
        self._backend = backend or get_database_backend()
        self._init_db()

    def _init_db(self) -> None:
        self._backend.execute_write(
            """
            CREATE TABLE IF NOT EXISTS capability_approvals (
                approval_id TEXT PRIMARY KEY,
                capability TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                target_resource TEXT NOT NULL,
                normalized_input_hash TEXT NOT NULL,
                receipt_id TEXT DEFAULT '',
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                approved_by TEXT DEFAULT '',
                approved_at TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}'
            )
            """
        )
        self._backend.execute_write(
            """
            CREATE INDEX IF NOT EXISTS idx_capability_approvals_status
            ON capability_approvals(status)
            """
        )
        self._backend.execute_write(
            """
            CREATE INDEX IF NOT EXISTS idx_capability_approvals_capability
            ON capability_approvals(capability, actor_id)
            """
        )
        self._backend.execute_write(
            """
            CREATE TABLE IF NOT EXISTS capability_actions (
                action_id TEXT PRIMARY KEY,
                capability TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                target_resource TEXT NOT NULL,
                normalized_input_hash TEXT NOT NULL,
                receipt_id TEXT DEFAULT '',
                approval_id TEXT DEFAULT '',
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}'
            )
            """
        )
        self._backend.execute_write(
            """
            CREATE INDEX IF NOT EXISTS idx_capability_actions_capability
            ON capability_actions(capability, actor_id, status)
            """
        )

    def create_approval(
        self,
        *,
        capability: Capability,
        actor_id: str,
        target_resource: str,
        input_payload: Any,
        receipt_id: str = "",
        expires_in_seconds: int = 3600,
        approved: bool = False,
        approved_by: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CapabilityApprovalRecord:
        now = datetime.now(UTC)
        approval = CapabilityApprovalRecord(
            approval_id=f"cap-{uuid.uuid4().hex[:12]}",
            capability=capability,
            actor_id=actor_id,
            target_resource=target_resource,
            normalized_input_hash=normalize_input_hash(input_payload),
            receipt_id=receipt_id,
            status=ApprovalStatus.APPROVED if approved else ApprovalStatus.PENDING,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=expires_in_seconds)).isoformat(),
            approved_by=approved_by if approved else "",
            approved_at=now.isoformat() if approved else "",
            metadata=metadata or {},
        )
        self._backend.execute_write(
            """
            INSERT INTO capability_approvals (
                approval_id, capability, actor_id, target_resource,
                normalized_input_hash, receipt_id, status, created_at,
                expires_at, approved_by, approved_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                approval.approval_id,
                approval.capability.value,
                approval.actor_id,
                approval.target_resource,
                approval.normalized_input_hash,
                approval.receipt_id,
                approval.status.value,
                approval.created_at,
                approval.expires_at,
                approval.approved_by,
                approval.approved_at,
                json.dumps(approval.metadata, sort_keys=True, default=str),
            ),
        )
        return approval

    def approve(self, approval_id: str, approved_by: str) -> CapabilityApprovalRecord | None:
        approval = self.get_approval(approval_id)
        if approval is None:
            return None
        now = datetime.now(UTC).isoformat()
        self._backend.execute_write(
            """
            UPDATE capability_approvals
            SET status = ?, approved_by = ?, approved_at = ?
            WHERE approval_id = ?
            """,
            (
                ApprovalStatus.APPROVED.value,
                approved_by,
                now,
                approval_id,
            ),
        )
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: str) -> CapabilityApprovalRecord | None:
        row = self._backend.fetch_one(
            """
            SELECT approval_id, capability, actor_id, target_resource,
                   normalized_input_hash, receipt_id, status, created_at,
                   expires_at, approved_by, approved_at, metadata_json
            FROM capability_approvals
            WHERE approval_id = ?
            """,
            (approval_id,),
        )
        if row is None:
            return None
        return CapabilityApprovalRecord(
            approval_id=str(row[0]),
            capability=_parse_capability(row[1]),
            actor_id=str(row[2]),
            target_resource=str(row[3]),
            normalized_input_hash=str(row[4]),
            receipt_id=str(row[5] or ""),
            status=_parse_approval_status(row[6]),
            created_at=str(row[7]),
            expires_at=str(row[8]),
            approved_by=str(row[9] or ""),
            approved_at=str(row[10] or ""),
            metadata=_parse_json(row[11]),
        )

    def create_action(
        self,
        *,
        capability: Capability,
        actor_id: str,
        target_resource: str,
        input_payload: Any,
        receipt_id: str = "",
        approval_id: str = "",
        status: ActionStatus = ActionStatus.PENDING,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CapabilityActionRecord:
        now = datetime.now(UTC).isoformat()
        action = CapabilityActionRecord(
            action_id=f"act-{uuid.uuid4().hex[:12]}",
            capability=capability,
            actor_id=actor_id,
            target_resource=target_resource,
            normalized_input_hash=normalize_input_hash(input_payload),
            receipt_id=receipt_id,
            approval_id=approval_id,
            status=status,
            created_at=now,
            updated_at=now,
            error=error,
            metadata=metadata or {},
        )
        self._backend.execute_write(
            """
            INSERT INTO capability_actions (
                action_id, capability, actor_id, target_resource,
                normalized_input_hash, receipt_id, approval_id, status,
                created_at, updated_at, error, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                action.action_id,
                action.capability.value,
                action.actor_id,
                action.target_resource,
                action.normalized_input_hash,
                action.receipt_id,
                action.approval_id,
                action.status.value,
                action.created_at,
                action.updated_at,
                action.error,
                json.dumps(action.metadata, sort_keys=True, default=str),
            ),
        )
        return action

    def update_action(
        self,
        action_id: str,
        *,
        status: ActionStatus,
        error: str = "",
        approval_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CapabilityActionRecord | None:
        current = self.get_action(action_id)
        if current is None:
            return None
        next_metadata = dict(current.metadata)
        if metadata:
            next_metadata.update(metadata)
        self._backend.execute_write(
            """
            UPDATE capability_actions
            SET status = ?, updated_at = ?, error = ?, approval_id = ?, metadata_json = ?
            WHERE action_id = ?
            """,
            (
                status.value,
                datetime.now(UTC).isoformat(),
                error,
                approval_id if approval_id is not None else current.approval_id,
                json.dumps(next_metadata, sort_keys=True, default=str),
                action_id,
            ),
        )
        return self.get_action(action_id)

    def get_action(self, action_id: str) -> CapabilityActionRecord | None:
        row = self._backend.fetch_one(
            """
            SELECT action_id, capability, actor_id, target_resource,
                   normalized_input_hash, receipt_id, approval_id, status,
                   created_at, updated_at, error, metadata_json
            FROM capability_actions
            WHERE action_id = ?
            """,
            (action_id,),
        )
        if row is None:
            return None
        return CapabilityActionRecord(
            action_id=str(row[0]),
            capability=_parse_capability(row[1]),
            actor_id=str(row[2]),
            target_resource=str(row[3]),
            normalized_input_hash=str(row[4]),
            receipt_id=str(row[5] or ""),
            approval_id=str(row[6] or ""),
            status=_parse_action_status(row[7]),
            created_at=str(row[8]),
            updated_at=str(row[9]),
            error=str(row[10] or ""),
            metadata=_parse_json(row[11]),
        )


_store: CapabilityGateStore | None = None


def get_capability_gate_store() -> CapabilityGateStore:
    global _store
    if _store is None:
        _store = CapabilityGateStore()
    return _store


def ensure_approved_capability_approval(
    *,
    capability: Capability,
    actor_id: str,
    target_resource: str,
    input_payload: Any,
    receipt_id: str = "",
    approved_by: str,
    metadata: dict[str, Any] | None = None,
    expires_in_seconds: int = 3600,
) -> CapabilityApprovalRecord:
    """Create an already-approved capability record for an admin or trusted lane."""

    return get_capability_gate_store().create_approval(
        capability=capability,
        actor_id=actor_id,
        target_resource=target_resource,
        input_payload=input_payload,
        receipt_id=receipt_id,
        expires_in_seconds=expires_in_seconds,
        approved=True,
        approved_by=approved_by,
        metadata=metadata,
    )


def ensure_capability_approval_id(
    *,
    capability: Capability,
    actor_id: str,
    target_resource: str,
    input_payload: Any,
    approval_id: str = "",
    receipt_id: str = "",
    admin_approved: bool = False,
    approved_by: str = "",
    metadata: dict[str, Any] | None = None,
    expires_in_seconds: int = 3600,
) -> str:
    """Return an approval id, creating one only for explicit admin-approved flows."""

    normalized_approval_id = str(approval_id or "").strip()
    if normalized_approval_id:
        return normalized_approval_id
    if not admin_approved:
        return ""
    approval = ensure_approved_capability_approval(
        capability=capability,
        actor_id=actor_id,
        target_resource=target_resource,
        input_payload=input_payload,
        receipt_id=receipt_id,
        approved_by=approved_by or actor_id or "system",
        metadata=metadata,
        expires_in_seconds=expires_in_seconds,
    )
    return approval.approval_id


def _approval_validation_error(
    approval: CapabilityApprovalRecord | None,
    *,
    capability: Capability,
    actor_id: str,
    target_resource: str,
    input_hash: str,
    receipt_id: str,
) -> str:
    if approval is None:
        return "missing approval record"
    if approval.capability != capability:
        return "capability mismatch"
    if approval.status != ApprovalStatus.APPROVED:
        return f"approval is {approval.status.value}"
    if approval.expires_at and datetime.fromisoformat(approval.expires_at) <= datetime.now(UTC):
        return "approval expired"
    if approval.actor_id and approval.actor_id != actor_id:
        return "actor mismatch"
    if approval.target_resource and approval.target_resource != target_resource:
        return "target resource mismatch"
    if approval.normalized_input_hash and approval.normalized_input_hash != input_hash:
        return "input hash mismatch"
    if receipt_id and approval.receipt_id and approval.receipt_id != receipt_id:
        return "receipt mismatch"
    return ""


def authorize_capability_dispatch(
    *,
    capability: Capability,
    actor_id: str,
    target_resource: str,
    input_payload: Any,
    approval_id: str = "",
    receipt_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> CapabilityActionRecord:
    """Validate approval and create a durable dispatch audit record."""

    store = get_capability_gate_store()
    input_hash = normalize_input_hash(input_payload)
    approval = store.get_approval(approval_id) if approval_id else None
    reason = _approval_validation_error(
        approval,
        capability=capability,
        actor_id=actor_id,
        target_resource=target_resource,
        input_hash=input_hash,
        receipt_id=receipt_id,
    )
    status = ActionStatus.ALLOWED if not reason else ActionStatus.DENIED
    action = store.create_action(
        capability=capability,
        actor_id=actor_id,
        target_resource=target_resource,
        input_payload=input_payload,
        receipt_id=receipt_id,
        approval_id=approval_id,
        status=status,
        error=reason,
        metadata=metadata,
    )
    if reason:
        raise CapabilityApprovalRequiredError(capability, reason, action_id=action.action_id)
    return action


class CapabilityApprovalEnforcer:
    """Minimal approval enforcer backed by the durable capability gate."""

    def __init__(
        self,
        *,
        capability: Capability,
        actor_id: str,
        target_resource: str,
        input_payload: Any,
        receipt_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._capability = capability
        self._actor_id = actor_id
        self._target_resource = target_resource
        self._input_payload = input_payload
        self._receipt_id = receipt_id
        self._metadata = metadata or {}

    async def enforce(self, request: Any) -> Any:
        try:
            from aragora.security.approval_enforcer import (
                EnforcementDecision,
                EnforcementResult,
            )
        except ImportError as exc:  # pragma: no cover - defensive fallback
            raise RuntimeError("Unified approval enforcer types unavailable") from exc

        approval_id = str(getattr(request, "approval_id", "") or "").strip()
        try:
            action = authorize_capability_dispatch(
                capability=self._capability,
                actor_id=self._actor_id,
                target_resource=self._target_resource,
                input_payload=self._input_payload,
                approval_id=approval_id,
                receipt_id=self._receipt_id,
                metadata={
                    **self._metadata,
                    "enforcement_source": getattr(request, "source", ""),
                    "enforcement_action_type": getattr(request, "action_type", ""),
                },
            )
            return EnforcementDecision(
                id=action.action_id,
                result=EnforcementResult.ALLOWED,
                reason=f"Capability gate approved via {approval_id}",
                request=request,
                approval_request_id=approval_id or None,
                metadata={"capability": self._capability.value},
            )
        except CapabilityApprovalRequiredError as exc:
            return EnforcementDecision(
                id=exc.action_id or f"cap-deny-{uuid.uuid4().hex[:12]}",
                result=EnforcementResult.DENIED,
                reason=exc.reason,
                request=request,
                approval_request_id=approval_id or None,
                metadata={"capability": self._capability.value},
            )

    async def wait_for_approval(
        self,
        approval_id: str,
        timeout: float | None = None,
    ) -> bool:
        del timeout
        approval = get_capability_gate_store().get_approval(approval_id)
        if approval is None:
            return False
        if approval.status != ApprovalStatus.APPROVED:
            return False
        if approval.expires_at and datetime.fromisoformat(approval.expires_at) <= datetime.now(UTC):
            return False
        return True


__all__ = [
    "ActionStatus",
    "Capability",
    "CapabilityActionRecord",
    "CapabilityApprovalEnforcer",
    "CapabilityApprovalRecord",
    "CapabilityApprovalRequiredError",
    "CapabilityGateStore",
    "ApprovalStatus",
    "authorize_capability_dispatch",
    "ensure_capability_approval_id",
    "ensure_approved_capability_approval",
    "get_capability_gate_store",
    "normalize_input_hash",
]
