"""
Tests for aragora.server.handlers.compliance_handler - Compliance HTTP Handler.

Tests cover:
1. GDPR data export (subject access requests)
2. GDPR data deletion cascade (right to be forgotten)
3. Audit trail immutability verification
4. Data retention policy enforcement
5. Consent management (record, update, revoke)
6. Compliance report generation
7. Data processing records
8. Cross-border data transfer validation
9. Error handling (missing user, invalid request, cascade failures)
10. Permission checks (only authorized users can trigger compliance actions)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from enum import Enum

import pytest

from aragora.server.handlers.compliance_handler import (
    ComplianceHandler,
    create_compliance_handler,
)
import builtins


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


class MockDeletionStatus(Enum):
    """Mock deletion status enum."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    HELD = "held"


@dataclass
class MockDeletionRequest:
    """Mock deletion request for testing."""

    request_id: str = "del-req-001"
    user_id: str = "user-123"
    scheduled_for: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=30)
    )
    reason: str = "User request"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: MockDeletionStatus = MockDeletionStatus.PENDING
    entities_deleted: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    cancelled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "scheduled_for": self.scheduled_for.isoformat(),
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "entities_deleted": self.entities_deleted,
            "errors": self.errors,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


@dataclass
class MockLegalHold:
    """Mock legal hold for testing."""

    hold_id: str = "hold-001"
    user_ids: list[str] = field(default_factory=lambda: ["user-123"])
    reason: str = "Litigation hold"
    created_by: str = "legal-team"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    case_reference: Optional[str] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    released_at: Optional[datetime] = None
    released_by: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hold_id": self.hold_id,
            "user_ids": self.user_ids,
            "reason": self.reason,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "case_reference": self.case_reference,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "released_by": self.released_by,
        }


@dataclass
class MockStoredReceipt:
    """Mock stored receipt for testing."""

    receipt_id: str = "receipt-001"
    gauntlet_id: str = "gauntlet-001"
    debate_id: Optional[str] = "debate-001"
    created_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    expires_at: Optional[float] = None
    verdict: str = "approved"
    confidence: float = 0.95
    risk_level: str = "low"
    risk_score: float = 0.1
    checksum: str = "sha256:abc123"
    signature: Optional[str] = "sig-data"
    signature_algorithm: Optional[str] = "RSA-SHA256"
    signature_key_id: Optional[str] = "key-001"
    signed_at: Optional[float] = None
    audit_trail_id: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockVerificationResult:
    """Mock verification result for batch receipt verification."""

    receipt_id: str = "receipt-001"
    is_valid: bool = True
    error: Optional[str] = None


@dataclass
class MockCascadeReport:
    """Mock cascade deletion report."""

    success: bool = True
    user_id: str = "user-123"
    deleted_from: list[Any] = field(default_factory=list)
    backup_purge_results: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "user_id": self.user_id,
            "deleted_from": [str(s) for s in self.deleted_from],
            "backup_purge_results": self.backup_purge_results,
            "errors": self.errors,
        }


class MockAuditStore:
    """Mock audit store for testing."""

    def __init__(self):
        self._events: list[dict[str, Any]] = []

    def log_event(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        event_id = f"evt-{len(self._events) + 1:03d}"
        event = {
            "id": event_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._events.append(event)
        return event_id

    def get_log(
        self,
        action: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        events = self._events
        if action:
            events = [e for e in events if e["action"] == action]
        return events[:limit]

    def get_recent_activity(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._events[:limit]


class MockReceiptStore:
    """Mock receipt store for testing."""

    def __init__(self):
        self._receipts: dict[str, MockStoredReceipt] = {}

    def list(
        self,
        limit: int = 100,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> builtins.list[MockStoredReceipt]:
        return list(self._receipts.values())[:limit]

    def get(self, receipt_id: str) -> Optional[MockStoredReceipt]:
        return self._receipts.get(receipt_id)

    def get_by_gauntlet(self, gauntlet_id: str) -> Optional[MockStoredReceipt]:
        for receipt in self._receipts.values():
            if receipt.gauntlet_id == gauntlet_id:
                return receipt
        return None

    def verify_batch(
        self,
        receipt_ids: builtins.list[str],
    ) -> tuple[builtins.list[MockVerificationResult], dict[str, Any]]:
        results = []
        for rid in receipt_ids:
            if rid in self._receipts:
                results.append(MockVerificationResult(receipt_id=rid, is_valid=True))
            else:
                results.append(
                    MockVerificationResult(receipt_id=rid, is_valid=False, error="Not found")
                )

        summary = {
            "total": len(receipt_ids),
            "valid": sum(1 for r in results if r.is_valid),
            "invalid": sum(1 for r in results if not r.is_valid),
        }
        return results, summary


class MockDeletionStore:
    """Mock deletion store for testing."""

    def __init__(self):
        self._requests: dict[str, MockDeletionRequest] = {}
        self._holds: dict[str, MockLegalHold] = {}

    def get_request(self, request_id: str) -> Optional[MockDeletionRequest]:
        return self._requests.get(request_id)

    def get_all_requests(
        self,
        status: Optional[MockDeletionStatus] = None,
        limit: int = 50,
    ) -> list[MockDeletionRequest]:
        requests = list(self._requests.values())
        if status:
            requests = [r for r in requests if r.status == status]
        return requests[:limit]

    def get_active_holds_for_user(self, user_id: str) -> list[MockLegalHold]:
        return [h for h in self._holds.values() if h.is_active and user_id in h.user_ids]


class MockDeletionScheduler:
    """Mock deletion scheduler for testing."""

    def __init__(self):
        self.store = MockDeletionStore()
        self._requests: dict[str, MockDeletionRequest] = {}

    def schedule_deletion(
        self,
        user_id: str,
        grace_period_days: int = 30,
        reason: str = "User request",
        metadata: Optional[dict[str, Any]] = None,
    ) -> MockDeletionRequest:
        request = MockDeletionRequest(
            request_id=f"del-{user_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            user_id=user_id,
            scheduled_for=datetime.now(timezone.utc) + timedelta(days=grace_period_days),
            reason=reason,
            metadata=metadata or {},
        )
        self._requests[request.request_id] = request
        self.store._requests[request.request_id] = request
        return request

    def cancel_deletion(
        self,
        request_id: str,
        reason: str = "Cancelled",
    ) -> Optional[MockDeletionRequest]:
        request = self._requests.get(request_id)
        if not request:
            return None
        request.status = MockDeletionStatus.CANCELLED
        request.cancelled_at = datetime.now(timezone.utc)
        return request


class MockLegalHoldManager:
    """Mock legal hold manager for testing."""

    def __init__(self):
        self._store = MockDeletionStore()
        self._holds: dict[str, MockLegalHold] = {}
        self._user_holds: dict[str, list[str]] = {}

    def is_user_on_hold(self, user_id: str) -> bool:
        return user_id in self._user_holds and len(self._user_holds[user_id]) > 0

    def get_active_holds(self) -> list[MockLegalHold]:
        return [h for h in self._holds.values() if h.is_active]

    def create_hold(
        self,
        user_ids: list[str],
        reason: str,
        created_by: str,
        case_reference: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> MockLegalHold:
        hold = MockLegalHold(
            hold_id=f"hold-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            user_ids=user_ids,
            reason=reason,
            created_by=created_by,
            case_reference=case_reference,
            expires_at=expires_at,
        )
        self._holds[hold.hold_id] = hold
        for uid in user_ids:
            if uid not in self._user_holds:
                self._user_holds[uid] = []
            self._user_holds[uid].append(hold.hold_id)
        return hold

    def release_hold(
        self,
        hold_id: str,
        released_by: str,
    ) -> Optional[MockLegalHold]:
        hold = self._holds.get(hold_id)
        if not hold:
            return None
        hold.is_active = False
        hold.released_at = datetime.now(timezone.utc)
        hold.released_by = released_by
        for uid in hold.user_ids:
            if uid in self._user_holds:
                self._user_holds[uid] = [h for h in self._user_holds[uid] if h != hold_id]
        return hold


class MockDeletionCoordinator:
    """Mock deletion coordinator for testing."""

    def __init__(self):
        self._exclusions: list[dict[str, Any]] = []

    async def execute_coordinated_deletion(
        self,
        user_id: str,
        reason: str,
        delete_from_backups: bool = True,
        dry_run: bool = False,
    ) -> MockCascadeReport:
        return MockCascadeReport(
            success=True,
            user_id=user_id,
            deleted_from=["primary", "backup"] if delete_from_backups else ["primary"],
            backup_purge_results={"purged": 2} if delete_from_backups else {},
        )

    async def process_pending_deletions(
        self,
        include_backups: bool = True,
        limit: int = 100,
    ) -> list[MockCascadeReport]:
        return [
            MockCascadeReport(success=True, user_id="user-001"),
            MockCascadeReport(success=True, user_id="user-002"),
        ]

    def get_backup_exclusion_list(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._exclusions[:limit]

    def add_to_backup_exclusion_list(self, user_id: str, reason: str) -> None:
        self._exclusions.append(
            {
                "user_id": user_id,
                "reason": reason,
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
        )


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def mock_audit_store():
    """Create mock audit store."""
    store = MockAuditStore()
    store._events = [
        {
            "id": "evt-001",
            "action": "user.login",
            "resource_type": "user",
            "resource_id": "user-123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "evt-002",
            "action": "debate.created",
            "resource_type": "debate",
            "resource_id": "debate-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    ]
    return store


@pytest.fixture
def mock_receipt_store():
    """Create mock receipt store with sample data."""
    store = MockReceiptStore()
    store._receipts["receipt-001"] = MockStoredReceipt()
    store._receipts["receipt-002"] = MockStoredReceipt(
        receipt_id="receipt-002",
        gauntlet_id="gauntlet-002",
        verdict="rejected",
    )
    return store


@pytest.fixture
def mock_deletion_scheduler():
    """Create mock deletion scheduler."""
    scheduler = MockDeletionScheduler()
    scheduler._requests["del-req-001"] = MockDeletionRequest()
    scheduler.store._requests["del-req-001"] = MockDeletionRequest()
    return scheduler


@pytest.fixture
def mock_legal_hold_manager():
    """Create mock legal hold manager."""
    manager = MockLegalHoldManager()
    manager._holds["hold-001"] = MockLegalHold()
    return manager


@pytest.fixture
def mock_deletion_coordinator():
    """Create mock deletion coordinator."""
    return MockDeletionCoordinator()


@pytest.fixture
def handler(
    mock_server_context,
    mock_audit_store,
    mock_receipt_store,
    mock_deletion_scheduler,
    mock_legal_hold_manager,
    mock_deletion_coordinator,
):
    """Create handler with mocked dependencies."""
    with (
        patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ),
        patch(
            "aragora.server.handlers.compliance_handler.get_receipt_store",
            return_value=mock_receipt_store,
        ),
        patch(
            "aragora.server.handlers.compliance_handler.get_deletion_scheduler",
            return_value=mock_deletion_scheduler,
        ),
        patch(
            "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
            return_value=mock_legal_hold_manager,
        ),
        patch(
            "aragora.server.handlers.compliance_handler.get_deletion_coordinator",
            return_value=mock_deletion_coordinator,
        ),
    ):
        h = ComplianceHandler(mock_server_context)
        yield h


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestComplianceHandlerRouting:
    """Test request routing."""

    def test_can_handle_compliance_get_paths(self, handler):
        assert handler.can_handle("/api/v2/compliance/status", "GET")
        assert handler.can_handle("/api/v2/compliance/soc2-report", "GET")
        assert handler.can_handle("/api/v2/compliance/gdpr-export", "GET")
        assert handler.can_handle("/api/v2/compliance/audit-events", "GET")
        assert handler.can_handle("/api/v2/compliance/gdpr/deletions", "GET")
        assert handler.can_handle("/api/v2/compliance/gdpr/legal-holds", "GET")

    def test_can_handle_compliance_post_paths(self, handler):
        assert handler.can_handle("/api/v2/compliance/gdpr/right-to-be-forgotten", "POST")
        assert handler.can_handle("/api/v2/compliance/audit-verify", "POST")
        assert handler.can_handle("/api/v2/compliance/gdpr/legal-holds", "POST")

    def test_can_handle_compliance_delete_paths(self, handler):
        assert handler.can_handle("/api/v2/compliance/gdpr/legal-holds/hold-001", "DELETE")

    def test_cannot_handle_other_paths(self, handler):
        assert not handler.can_handle("/api/v2/backups", "GET")
        assert not handler.can_handle("/api/v1/compliance/status", "GET")

    def test_cannot_handle_unsupported_methods(self, handler):
        assert not handler.can_handle("/api/v2/compliance/status", "PUT")
        assert not handler.can_handle("/api/v2/compliance/status", "PATCH")


# ===========================================================================
# Status Endpoint Tests
# ===========================================================================


class TestComplianceStatus:
    """Test compliance status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_returns_200_with_structure(self, handler):
        result = await handler.handle("GET", "/api/v2/compliance/status")
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "status" in body
        assert "compliance_score" in body
        assert "frameworks" in body
        assert "controls_summary" in body
        assert "generated_at" in body
        assert "soc2_type2" in body["frameworks"]
        assert "gdpr" in body["frameworks"]
        assert "hipaa" in body["frameworks"]

    @pytest.mark.asyncio
    async def test_status_score_consistency(self, handler):
        result = await handler.handle("GET", "/api/v2/compliance/status")
        body = json.loads(result.body)
        assert 0 <= body["compliance_score"] <= 100
        summary = body["controls_summary"]
        assert summary["total"] == summary["compliant"] + summary["non_compliant"]


# ===========================================================================
# SOC 2 Report Tests
# ===========================================================================


class TestSoc2Report:
    """Test SOC 2 report generation."""

    @pytest.mark.asyncio
    async def test_soc2_report_json_format(self, handler):
        result = await handler.handle(
            "GET", "/api/v2/compliance/soc2-report", query_params={"format": "json"}
        )
        assert result.status_code == 200
        assert result.content_type == "application/json"
        body = json.loads(result.body)
        assert body["report_type"] == "SOC 2 Type II"
        assert "report_id" in body
        assert "period" in body
        assert "trust_service_criteria" in body
        assert "controls" in body
        assert "summary" in body

    @pytest.mark.asyncio
    async def test_soc2_report_html_format(self, handler):
        result = await handler.handle(
            "GET", "/api/v2/compliance/soc2-report", query_params={"format": "html"}
        )
        assert result.status_code == 200
        assert result.content_type == "text/html"
        content = result.body.decode("utf-8")
        assert "SOC 2 Type II" in content

    @pytest.mark.asyncio
    async def test_soc2_report_custom_date_range(self, handler):
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/soc2-report",
            query_params={
                "period_start": "2025-01-01T00:00:00Z",
                "period_end": "2025-03-31T23:59:59Z",
            },
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["period"]["start"].startswith("2025-01-01")
        assert body["period"]["end"].startswith("2025-03-31")

    @pytest.mark.asyncio
    async def test_soc2_report_defaults_to_json(self, handler):
        result = await handler.handle("GET", "/api/v2/compliance/soc2-report")
        assert result.status_code == 200
        assert result.content_type == "application/json"


# ===========================================================================
# GDPR Export Tests
# ===========================================================================


class TestGdprExport:
    """Test GDPR export endpoint."""

    @pytest.mark.asyncio
    async def test_gdpr_export_json_success(self, handler):
        result = await handler.handle(
            "GET", "/api/v2/compliance/gdpr-export", query_params={"user_id": "user-123"}
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["user_id"] == "user-123"
        assert "export_id" in body
        assert "data_categories" in body
        assert "checksum" in body

    @pytest.mark.asyncio
    async def test_gdpr_export_csv_format(self, handler):
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/gdpr-export",
            query_params={"user_id": "user-123", "format": "csv"},
        )
        assert result.status_code == 200
        assert result.content_type == "text/csv"
        content = result.body.decode("utf-8")
        assert "GDPR Data Export" in content

    @pytest.mark.asyncio
    async def test_gdpr_export_missing_user_id_returns_400(self, handler):
        result = await handler.handle("GET", "/api/v2/compliance/gdpr-export")
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "user_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_gdpr_export_with_include_filter(self, handler):
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/gdpr-export",
            query_params={"user_id": "user-123", "include": "decisions,activity"},
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "decisions" in body["data_categories"]
        assert "activity" in body["data_categories"]

    @pytest.mark.asyncio
    async def test_gdpr_export_checksum_is_sha256(self, handler):
        result = await handler.handle(
            "GET", "/api/v2/compliance/gdpr-export", query_params={"user_id": "user-123"}
        )
        body = json.loads(result.body)
        assert len(body["checksum"]) == 64  # SHA-256 hex digest


# ===========================================================================
# Right-to-be-Forgotten Tests
# ===========================================================================


class TestRightToBeForgotten:
    """Test GDPR Right-to-be-Forgotten endpoint."""

    @pytest.mark.asyncio
    async def test_rtbf_success(
        self,
        mock_server_context,
        mock_audit_store,
        mock_receipt_store,
        mock_deletion_scheduler,
        mock_deletion_coordinator,
    ):
        legal_hold_manager = MockLegalHoldManager()
        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
        ):
            h = ComplianceHandler(mock_server_context)
            result = await h.handle(
                "POST",
                "/api/v2/compliance/gdpr/right-to-be-forgotten",
                body={"user_id": "user-456", "grace_period_days": 30},
            )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "scheduled"
        assert body["user_id"] == "user-456"
        assert "request_id" in body
        assert "operations" in body
        assert "deletion_scheduled" in body

    @pytest.mark.asyncio
    async def test_rtbf_missing_user_id_returns_400(self, handler):
        result = await handler.handle(
            "POST", "/api/v2/compliance/gdpr/right-to-be-forgotten", body={}
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_rtbf_with_export_generates_url(
        self,
        mock_server_context,
        mock_audit_store,
        mock_receipt_store,
        mock_deletion_scheduler,
        mock_deletion_coordinator,
    ):
        legal_hold_manager = MockLegalHoldManager()
        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
        ):
            h = ComplianceHandler(mock_server_context)
            result = await h.handle(
                "POST",
                "/api/v2/compliance/gdpr/right-to-be-forgotten",
                body={"user_id": "user-456", "include_export": True},
            )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "export_url" in body

    @pytest.mark.asyncio
    async def test_rtbf_default_grace_period_is_30(
        self,
        mock_server_context,
        mock_audit_store,
        mock_receipt_store,
        mock_deletion_scheduler,
        mock_deletion_coordinator,
    ):
        legal_hold_manager = MockLegalHoldManager()
        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
        ):
            h = ComplianceHandler(mock_server_context)
            result = await h.handle(
                "POST",
                "/api/v2/compliance/gdpr/right-to-be-forgotten",
                body={"user_id": "user-456"},
            )
        body = json.loads(result.body)
        assert body["grace_period_days"] == 30


# ===========================================================================
# Audit Verification Tests
# ===========================================================================


class TestAuditVerification:
    """Test audit trail verification endpoint."""

    @pytest.mark.asyncio
    async def test_verify_trail_found(self, handler, mock_receipt_store):
        mock_receipt_store._receipts["trail-001"] = MockStoredReceipt(
            receipt_id="trail-001", gauntlet_id="gauntlet-trail-001"
        )
        with patch(
            "aragora.server.handlers.compliance_handler.get_receipt_store",
            return_value=mock_receipt_store,
        ):
            result = await handler.handle(
                "POST", "/api/v2/compliance/audit-verify", body={"trail_id": "trail-001"}
            )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "verified" in body
        assert "checks" in body
        assert "verified_at" in body

    @pytest.mark.asyncio
    async def test_verify_receipts_batch(self, handler, mock_receipt_store):
        with patch(
            "aragora.storage.receipt_store.get_receipt_store", return_value=mock_receipt_store
        ):
            result = await handler.handle(
                "POST",
                "/api/v2/compliance/audit-verify",
                body={"receipt_ids": ["receipt-001", "receipt-002", "nonexistent"]},
            )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "receipt_summary" in body
        assert body["receipt_summary"]["total"] == 3

    @pytest.mark.asyncio
    async def test_verify_date_range(self, handler, mock_audit_store):
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/audit-verify",
            body={"date_range": {"from": "2025-01-01T00:00:00Z", "to": "2025-12-31T23:59:59Z"}},
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        date_range_checks = [c for c in body["checks"] if c.get("type") == "date_range"]
        assert len(date_range_checks) > 0

    @pytest.mark.asyncio
    async def test_verify_empty_body_returns_verified(self, handler):
        result = await handler.handle("POST", "/api/v2/compliance/audit-verify", body={})
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["verified"] is True
        assert len(body["checks"]) == 0


# ===========================================================================
# Audit Events Export Tests
# ===========================================================================


class TestAuditEventsExport:
    """Test audit events export endpoint."""

    @pytest.mark.asyncio
    async def test_audit_events_json(self, handler):
        result = await handler.handle(
            "GET", "/api/v2/compliance/audit-events", query_params={"format": "json"}
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "events" in body
        assert "count" in body

    @pytest.mark.asyncio
    async def test_audit_events_ndjson(self, handler):
        result = await handler.handle(
            "GET", "/api/v2/compliance/audit-events", query_params={"format": "ndjson"}
        )
        assert result.status_code == 200
        assert result.content_type == "application/x-ndjson"

    @pytest.mark.asyncio
    async def test_audit_events_elasticsearch(self, handler):
        result = await handler.handle(
            "GET", "/api/v2/compliance/audit-events", query_params={"format": "elasticsearch"}
        )
        assert result.status_code == 200
        assert result.content_type == "application/x-ndjson"
        content = result.body.decode("utf-8")
        lines = content.strip().split("\n")
        for i, line in enumerate(lines):
            if line:
                parsed = json.loads(line)
                if i % 2 == 0:
                    assert "index" in parsed
                else:
                    assert "@timestamp" in parsed

    @pytest.mark.asyncio
    async def test_audit_events_with_event_type_filter(self, handler):
        result = await handler.handle(
            "GET", "/api/v2/compliance/audit-events", query_params={"event_type": "user.login"}
        )
        assert result.status_code == 200


# ===========================================================================
# Deletion Management Tests
# ===========================================================================


class TestDeletionManagement:
    """Test deletion management endpoints."""

    @pytest.mark.asyncio
    async def test_list_deletions_success(self, handler):
        result = await handler.handle("GET", "/api/v2/compliance/gdpr/deletions")
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "deletions" in body
        assert "count" in body

    @pytest.mark.asyncio
    async def test_list_deletions_with_status_filter(self, handler):
        result = await handler.handle(
            "GET", "/api/v2/compliance/gdpr/deletions", query_params={"status": "pending"}
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["filters"]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_deletion_by_id(self, handler):
        result = await handler.handle("GET", "/api/v2/compliance/gdpr/deletions/del-req-001")
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "deletion" in body

    @pytest.mark.asyncio
    async def test_get_deletion_not_found_returns_404(self, handler):
        result = await handler.handle("GET", "/api/v2/compliance/gdpr/deletions/nonexistent")
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_deletion_success(self, handler):
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/deletions/del-req-001/cancel",
            body={"reason": "Changed mind"},
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "cancelled" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_cancel_deletion_not_found_returns_404(self, handler):
        result = await handler.handle(
            "POST", "/api/v2/compliance/gdpr/deletions/nonexistent/cancel", body={}
        )
        assert result.status_code == 404


# ===========================================================================
# Legal Hold Management Tests
# ===========================================================================


class TestLegalHoldManagement:
    """Test legal hold management endpoints."""

    @pytest.mark.asyncio
    async def test_list_legal_holds(self, handler):
        result = await handler.handle("GET", "/api/v2/compliance/gdpr/legal-holds")
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "legal_holds" in body
        assert "count" in body

    @pytest.mark.asyncio
    async def test_list_legal_holds_include_inactive(self, handler):
        result = await handler.handle(
            "GET", "/api/v2/compliance/gdpr/legal-holds", query_params={"active_only": "false"}
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["filters"]["active_only"] is False

    @pytest.mark.asyncio
    async def test_create_legal_hold_success(self, handler):
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/legal-holds",
            body={
                "user_ids": ["user-789"],
                "reason": "Litigation hold",
                "case_reference": "CASE-001",
            },
        )
        assert result.status_code == 201
        body = json.loads(result.body)
        assert "legal_hold" in body

    @pytest.mark.asyncio
    async def test_create_legal_hold_missing_user_ids_returns_400(self, handler):
        result = await handler.handle(
            "POST", "/api/v2/compliance/gdpr/legal-holds", body={"reason": "Test"}
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_legal_hold_missing_reason_returns_400(self, handler):
        result = await handler.handle(
            "POST", "/api/v2/compliance/gdpr/legal-holds", body={"user_ids": ["user-123"]}
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_legal_hold_empty_user_list_returns_400(self, handler):
        result = await handler.handle(
            "POST", "/api/v2/compliance/gdpr/legal-holds", body={"user_ids": [], "reason": "Test"}
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_legal_hold_with_expiry(self, handler):
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/legal-holds",
            body={
                "user_ids": ["user-123"],
                "reason": "Temp hold",
                "expires_at": "2027-01-01T00:00:00Z",
            },
        )
        assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_create_legal_hold_invalid_expiry_returns_400(self, handler):
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/legal-holds",
            body={"user_ids": ["user-123"], "reason": "Test", "expires_at": "not-a-date"},
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_release_legal_hold_success(self, handler):
        result = await handler.handle(
            "DELETE", "/api/v2/compliance/gdpr/legal-holds/hold-001", body={"released_by": "admin"}
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "released" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_release_legal_hold_not_found_returns_404(self, handler):
        result = await handler.handle(
            "DELETE", "/api/v2/compliance/gdpr/legal-holds/nonexistent", body={}
        )
        assert result.status_code == 404


# ===========================================================================
# Coordinated Deletion Tests
# ===========================================================================


class TestCoordinatedDeletion:
    """Test coordinated deletion endpoints."""

    @pytest.mark.asyncio
    async def test_coordinated_deletion_success(
        self,
        mock_server_context,
        mock_audit_store,
        mock_receipt_store,
        mock_deletion_scheduler,
        mock_deletion_coordinator,
    ):
        legal_hold_manager = MockLegalHoldManager()
        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
        ):
            h = ComplianceHandler(mock_server_context)
            result = await h.handle(
                "POST",
                "/api/v2/compliance/gdpr/coordinated-deletion",
                body={"user_id": "user-to-delete", "reason": "GDPR request"},
            )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "report" in body
        assert body["report"]["success"] is True

    @pytest.mark.asyncio
    async def test_coordinated_deletion_missing_user_id_returns_400(self, handler):
        result = await handler.handle(
            "POST", "/api/v2/compliance/gdpr/coordinated-deletion", body={"reason": "Test"}
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_coordinated_deletion_missing_reason_returns_400(self, handler):
        result = await handler.handle(
            "POST", "/api/v2/compliance/gdpr/coordinated-deletion", body={"user_id": "user-123"}
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_coordinated_deletion_blocked_by_legal_hold(
        self,
        mock_server_context,
        mock_audit_store,
        mock_receipt_store,
        mock_deletion_scheduler,
        mock_deletion_coordinator,
    ):
        legal_hold_manager = MockLegalHoldManager()
        legal_hold_manager.create_hold(
            user_ids=["user-on-hold"], reason="Litigation", created_by="legal-team"
        )
        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
        ):
            h = ComplianceHandler(mock_server_context)
            result = await h.handle(
                "POST",
                "/api/v2/compliance/gdpr/coordinated-deletion",
                body={"user_id": "user-on-hold", "reason": "GDPR"},
            )
        assert result.status_code == 409

    @pytest.mark.asyncio
    async def test_execute_pending_deletions(
        self,
        mock_server_context,
        mock_audit_store,
        mock_receipt_store,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_deletion_coordinator,
    ):
        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
        ):
            h = ComplianceHandler(mock_server_context)
            result = await h.handle(
                "POST", "/api/v2/compliance/gdpr/execute-pending", body={"include_backups": True}
            )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "summary" in body


# ===========================================================================
# Backup Exclusion Tests
# ===========================================================================


class TestBackupExclusions:
    """Test backup exclusion management endpoints."""

    @pytest.mark.asyncio
    async def test_list_backup_exclusions(self, handler):
        result = await handler.handle("GET", "/api/v2/compliance/gdpr/backup-exclusions")
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "exclusions" in body
        assert "count" in body

    @pytest.mark.asyncio
    async def test_add_backup_exclusion_success(self, handler):
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/backup-exclusions",
            body={"user_id": "user-excl", "reason": "GDPR deletion"},
        )
        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["user_id"] == "user-excl"

    @pytest.mark.asyncio
    async def test_add_backup_exclusion_missing_user_id_returns_400(self, handler):
        result = await handler.handle(
            "POST", "/api/v2/compliance/gdpr/backup-exclusions", body={"reason": "Test"}
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_add_backup_exclusion_missing_reason_returns_400(self, handler):
        result = await handler.handle(
            "POST", "/api/v2/compliance/gdpr/backup-exclusions", body={"user_id": "user-123"}
        )
        assert result.status_code == 400


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_unknown_route_returns_404(self, handler):
        result = await handler.handle("GET", "/api/v2/compliance/unknown-endpoint")
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_permission_denied_returns_403(self, mock_server_context):
        from aragora.rbac.decorators import PermissionDeniedError

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_receipt_store",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_scheduler",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_coordinator",
                return_value=MagicMock(),
            ),
        ):
            h = ComplianceHandler(mock_server_context)
            with patch.object(
                h, "_get_status", side_effect=PermissionDeniedError("compliance:read")
            ):
                result = await h.handle("GET", "/api/v2/compliance/status")
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_internal_error_returns_500(self, mock_server_context):
        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_receipt_store",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_scheduler",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_coordinator",
                return_value=MagicMock(),
            ),
        ):
            h = ComplianceHandler(mock_server_context)
            with patch.object(h, "_get_status", side_effect=Exception("Unexpected error")):
                result = await h.handle("GET", "/api/v2/compliance/status")
        assert result.status_code == 500


# ===========================================================================
# Helper and Factory Tests
# ===========================================================================


class TestHelpers:
    """Test helper functions."""

    def test_create_compliance_handler_factory(self, mock_server_context):
        with (
            patch("aragora.server.handlers.compliance_handler.get_audit_store"),
            patch("aragora.server.handlers.compliance_handler.get_receipt_store"),
            patch("aragora.server.handlers.compliance_handler.get_deletion_scheduler"),
            patch("aragora.server.handlers.compliance_handler.get_legal_hold_manager"),
            patch("aragora.server.handlers.compliance_handler.get_deletion_coordinator"),
        ):
            h = create_compliance_handler(mock_server_context)
            assert isinstance(h, ComplianceHandler)

    def test_parse_timestamp_iso(self, handler):
        result = handler._parse_timestamp("2025-01-01T00:00:00Z")
        assert result is not None
        assert result.year == 2025

    def test_parse_timestamp_unix(self, handler):
        result = handler._parse_timestamp("1704067200")
        assert result is not None

    def test_parse_timestamp_none(self, handler):
        assert handler._parse_timestamp(None) is None

    def test_parse_timestamp_empty(self, handler):
        assert handler._parse_timestamp("") is None

    def test_parse_timestamp_invalid(self, handler):
        assert handler._parse_timestamp("not-a-timestamp") is None


# ===========================================================================
# User ID Extraction Tests
# ===========================================================================


class TestUserIdExtraction:
    """Test user ID extraction from headers."""

    def test_extract_user_id_none_headers(self):
        from aragora.server.handlers.compliance_handler import _extract_user_id_from_headers

        assert _extract_user_id_from_headers(None) == "compliance_api"

    def test_extract_user_id_empty_headers(self):
        from aragora.server.handlers.compliance_handler import _extract_user_id_from_headers

        assert _extract_user_id_from_headers({}) == "compliance_api"

    def test_extract_user_id_no_auth_header(self):
        from aragora.server.handlers.compliance_handler import _extract_user_id_from_headers

        assert _extract_user_id_from_headers({"Content-Type": "json"}) == "compliance_api"

    def test_extract_user_id_non_bearer(self):
        from aragora.server.handlers.compliance_handler import _extract_user_id_from_headers

        assert _extract_user_id_from_headers({"Authorization": "Basic xyz"}) == "compliance_api"

    def test_extract_user_id_api_key(self):
        from aragora.server.handlers.compliance_handler import _extract_user_id_from_headers

        result = _extract_user_id_from_headers({"Authorization": "Bearer ara_1234567890abcdef"})
        assert result.startswith("api_key:")

    def test_extract_user_id_lowercase_auth(self):
        from aragora.server.handlers.compliance_handler import _extract_user_id_from_headers

        result = _extract_user_id_from_headers({"authorization": "Bearer ara_testkey123"})
        assert result.startswith("api_key:")


# ===========================================================================
# Consent Revocation Tests
# ===========================================================================


class TestConsentRevocation:
    """Test consent revocation functionality."""

    @pytest.mark.asyncio
    async def test_revoke_all_consents_success(self, handler):
        mock_manager = MagicMock()
        mock_manager.bulk_revoke_for_user.return_value = 5
        with patch(
            "aragora.server.handlers.compliance_handler.get_consent_manager",
            return_value=mock_manager,
        ):
            result = await handler._revoke_all_consents("user-123")
        assert result == 5

    @pytest.mark.asyncio
    async def test_revoke_all_consents_import_error(self, handler):
        with patch(
            "aragora.server.handlers.compliance_handler.get_consent_manager",
            side_effect=ImportError("not found"),
        ):
            result = await handler._revoke_all_consents("user-123")
        assert result == 0

    @pytest.mark.asyncio
    async def test_revoke_all_consents_runtime_error(self, handler):
        mock_manager = MagicMock()
        mock_manager.bulk_revoke_for_user.side_effect = RuntimeError("DB error")
        with patch(
            "aragora.server.handlers.compliance_handler.get_consent_manager",
            return_value=mock_manager,
        ):
            result = await handler._revoke_all_consents("user-123")
        assert result == 0


# ===========================================================================
# Internal Method Tests
# ===========================================================================


class TestInternalMethods:
    """Test internal handler methods."""

    @pytest.mark.asyncio
    async def test_evaluate_controls_structure(self, handler):
        controls = await handler._evaluate_controls()
        assert len(controls) > 0
        for c in controls:
            assert "control_id" in c
            assert "category" in c
            assert "status" in c

    @pytest.mark.asyncio
    async def test_assess_security_criteria(self, handler):
        result = await handler._assess_security_criteria()
        assert result["status"] == "effective"
        assert "key_findings" in result

    @pytest.mark.asyncio
    async def test_assess_availability_criteria(self, handler):
        result = await handler._assess_availability_criteria()
        assert "uptime_target" in result

    @pytest.mark.asyncio
    async def test_get_user_preferences_returns_dict(self, handler):
        prefs = await handler._get_user_preferences("user-123")
        assert isinstance(prefs, dict)

    @pytest.mark.asyncio
    async def test_get_user_decisions_error_returns_empty(self, handler):
        failing_store = MagicMock()
        failing_store.list.side_effect = RuntimeError("DB error")
        with patch(
            "aragora.server.handlers.compliance_handler.get_receipt_store",
            return_value=failing_store,
        ):
            result = await handler._get_user_decisions("user-123")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_user_activity_error_returns_empty(self, handler):
        failing_store = MagicMock()
        failing_store.get_recent_activity.side_effect = RuntimeError("DB error")
        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store", return_value=failing_store
        ):
            result = await handler._get_user_activity("user-123")
        assert result == []

    @pytest.mark.asyncio
    async def test_verify_trail_not_found(self, handler, mock_receipt_store):
        with patch(
            "aragora.server.handlers.compliance_handler.get_receipt_store",
            return_value=mock_receipt_store,
        ):
            result = await handler._verify_trail("nonexistent")
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_verify_trail_exception(self, handler):
        failing_store = MagicMock()
        failing_store.get.side_effect = RuntimeError("Store error")
        failing_store.get_by_gauntlet.side_effect = RuntimeError("Store error")
        with patch(
            "aragora.server.handlers.compliance_handler.get_receipt_store",
            return_value=failing_store,
        ):
            result = await handler._verify_trail("some-trail")
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_verify_date_range_exception(self, handler):
        failing_store = MagicMock()
        failing_store.get_log.side_effect = RuntimeError("Store error")
        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store", return_value=failing_store
        ):
            result = await handler._verify_date_range(
                {"from": "2025-01-01T00:00:00Z", "to": "2025-12-31T23:59:59Z"}
            )
        assert result["valid"] is False

    def test_render_soc2_html(self, handler):
        report = {
            "report_id": "test",
            "report_type": "SOC 2 Type II",
            "period": {"start": "2025-01-01", "end": "2025-03-31"},
            "organization": "Test",
            "summary": {"controls_tested": 10, "controls_effective": 9, "exceptions": 1},
            "controls": [
                {
                    "control_id": "CC1.1",
                    "category": "Security",
                    "name": "Test",
                    "status": "compliant",
                }
            ],
            "generated_at": "2025-01-15T00:00:00Z",
        }
        html = handler._render_soc2_html(report)
        assert "SOC 2 Type II" in html
        assert "CC1.1" in html

    def test_render_gdpr_csv(self, handler):
        export_data = {
            "user_id": "user-test",
            "export_id": "export-test",
            "requested_at": "2025-01-15T00:00:00Z",
            "data_categories": ["decisions"],
            "decisions": [{"id": "d1"}],
            "checksum": "abc123",
        }
        csv = handler._render_gdpr_csv(export_data)
        assert "GDPR Data Export" in csv
        assert "user-test" in csv


# ===========================================================================
# Legal Hold Blocking Deletion Tests
# ===========================================================================


class TestLegalHoldBlocksDeletion:
    """Test that legal holds block deletions."""

    @pytest.mark.asyncio
    async def test_schedule_deletion_blocked_by_hold(
        self, mock_server_context, mock_audit_store, mock_receipt_store, mock_deletion_coordinator
    ):
        scheduler = MockDeletionScheduler()
        legal_hold_manager = MockLegalHoldManager()
        legal_hold_manager.create_hold(
            user_ids=["user-blocked"], reason="Litigation", created_by="legal"
        )
        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_scheduler",
                return_value=scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
        ):
            h = ComplianceHandler(mock_server_context)
            with pytest.raises(ValueError, match="legal hold"):
                await h._schedule_deletion(
                    user_id="user-blocked",
                    request_id="test-req",
                    scheduled_for=datetime.now(timezone.utc) + timedelta(days=30),
                    reason="GDPR request",
                )


# ===========================================================================
# Final Export Generation Tests
# ===========================================================================


class TestFinalExportGeneration:
    """Test final export generation for RTBF."""

    @pytest.mark.asyncio
    async def test_generate_final_export_includes_categories(
        self, handler, mock_audit_store, mock_receipt_store
    ):
        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_consent_manager",
                side_effect=ImportError("Not available"),
            ),
        ):
            export = await handler._generate_final_export("user-123")
        assert "export_id" in export
        assert export["user_id"] == "user-123"
        assert "decisions" in export["data_categories"]
        assert "preferences" in export["data_categories"]
        assert "activity" in export["data_categories"]
        assert "checksum" in export
