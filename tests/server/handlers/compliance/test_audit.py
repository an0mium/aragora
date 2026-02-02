"""
Tests for Audit Verification and Event Export Handler.

Tests cover:
- Audit trail verification
- Receipt verification
- Date range verification
- SIEM event export (JSON, NDJSON, Elasticsearch)
- Timestamp parsing
- RBAC permission enforcement
- Error handling
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.compliance.handler import ComplianceHandler
from aragora.server.handlers.base import HandlerResult


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def compliance_handler():
    """Create a compliance handler instance."""
    return ComplianceHandler(server_context={})


@pytest.fixture
def mock_audit_store():
    """Create a mock audit store."""
    store = MagicMock()
    store.log_event = MagicMock()
    store.get_log = MagicMock(return_value=[])
    return store


@pytest.fixture
def mock_receipt_store():
    """Create a mock receipt store."""
    store = MagicMock()
    store.get = MagicMock(return_value=None)
    store.get_by_gauntlet = MagicMock(return_value=None)
    store.verify_batch = MagicMock(return_value=([], {}))
    return store


# ============================================================================
# Audit Verification Tests
# ============================================================================


class TestAuditVerification:
    """Tests for audit trail verification endpoint."""

    @pytest.mark.asyncio
    async def test_verify_empty_body(self, compliance_handler):
        """Verification with empty body returns verified."""
        result = await compliance_handler._verify_audit({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["verified"] is True
        assert body["checks"] == []
        assert "verified_at" in body

    @pytest.mark.asyncio
    async def test_verify_trail_success(self, compliance_handler, mock_receipt_store):
        """Trail verification returns success for valid trail."""
        mock_receipt = MagicMock()
        mock_receipt.receipt_id = "receipt-001"
        mock_receipt.signature = "valid-signature"
        mock_receipt.verdict = "approved"
        mock_receipt_store.get.return_value = mock_receipt

        with patch(
            "aragora.server.handlers.compliance_handler.get_receipt_store",
            return_value=mock_receipt_store,
        ):
            result = await compliance_handler._verify_audit({"trail_id": "receipt-001"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["verified"] is True
        assert len(body["checks"]) == 1
        assert body["checks"][0]["type"] == "audit_trail"
        assert body["checks"][0]["valid"] is True

    @pytest.mark.asyncio
    async def test_verify_trail_not_found(self, compliance_handler, mock_receipt_store):
        """Trail verification fails for unknown trail."""
        mock_receipt_store.get.return_value = None
        mock_receipt_store.get_by_gauntlet.return_value = None

        with patch(
            "aragora.server.handlers.compliance_handler.get_receipt_store",
            return_value=mock_receipt_store,
        ):
            result = await compliance_handler._verify_audit({"trail_id": "nonexistent"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["verified"] is False
        assert body["checks"][0]["valid"] is False
        assert "not found" in body["checks"][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_verify_trail_by_gauntlet_id(self, compliance_handler, mock_receipt_store):
        """Trail verification falls back to gauntlet ID."""
        mock_receipt = MagicMock()
        mock_receipt.receipt_id = "receipt-001"
        mock_receipt.signature = None
        mock_receipt.verdict = "approved"
        mock_receipt_store.get.return_value = None
        mock_receipt_store.get_by_gauntlet.return_value = mock_receipt

        with patch(
            "aragora.server.handlers.compliance_handler.get_receipt_store",
            return_value=mock_receipt_store,
        ):
            result = await compliance_handler._verify_audit({"trail_id": "gauntlet-001"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["verified"] is True
        mock_receipt_store.get_by_gauntlet.assert_called_with("gauntlet-001")

    @pytest.mark.asyncio
    async def test_verify_receipts_batch(self, compliance_handler, mock_receipt_store):
        """Receipt batch verification works correctly."""
        mock_result = MagicMock()
        mock_result.receipt_id = "receipt-001"
        mock_result.is_valid = True
        mock_result.error = None

        mock_receipt_store.verify_batch.return_value = (
            [mock_result],
            {"total": 1, "valid": 1},
        )

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_receipt_store,
        ):
            result = await compliance_handler._verify_audit({"receipt_ids": ["receipt-001"]})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["verified"] is True
        assert "receipt_summary" in body
        assert body["receipt_summary"]["total"] == 1

    @pytest.mark.asyncio
    async def test_verify_receipts_with_failures(self, compliance_handler, mock_receipt_store):
        """Receipt verification reports failures."""
        valid_result = MagicMock()
        valid_result.receipt_id = "receipt-001"
        valid_result.is_valid = True
        valid_result.error = None

        invalid_result = MagicMock()
        invalid_result.receipt_id = "receipt-002"
        invalid_result.is_valid = False
        invalid_result.error = "Checksum mismatch"

        mock_receipt_store.verify_batch.return_value = (
            [valid_result, invalid_result],
            {"total": 2, "valid": 1},
        )

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_receipt_store,
        ):
            result = await compliance_handler._verify_audit(
                {"receipt_ids": ["receipt-001", "receipt-002"]}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["verified"] is False
        assert len(body["errors"]) == 1
        assert "receipt-002" in body["errors"][0]

    @pytest.mark.asyncio
    async def test_verify_date_range(self, compliance_handler, mock_audit_store):
        """Date range verification works correctly."""
        mock_audit_store.get_log.return_value = [
            {
                "id": "event-001",
                "action": "user_login",
                "timestamp": "2025-01-15T00:00:00Z",
            }
        ]

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._verify_audit(
                {
                    "date_range": {
                        "from": "2025-01-01T00:00:00Z",
                        "to": "2025-01-31T00:00:00Z",
                    }
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["verified"] is True
        range_check = body["checks"][0]
        assert range_check["type"] == "date_range"
        assert range_check["events_checked"] == 1

    @pytest.mark.asyncio
    async def test_verify_date_range_missing_action(self, compliance_handler, mock_audit_store):
        """Date range verification catches missing action fields."""
        mock_audit_store.get_log.return_value = [
            {
                "id": "event-001",
                "timestamp": "2025-01-15T00:00:00Z",
                # Missing "action" field
            }
        ]

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._verify_audit(
                {
                    "date_range": {
                        "from": "2025-01-01T00:00:00Z",
                        "to": "2025-01-31T00:00:00Z",
                    }
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["verified"] is False
        assert len(body["errors"]) > 0

    @pytest.mark.asyncio
    async def test_verify_all_types(self, compliance_handler, mock_receipt_store, mock_audit_store):
        """Verification can check trail, receipts, and date range together."""
        mock_receipt = MagicMock()
        mock_receipt.receipt_id = "receipt-001"
        mock_receipt.signature = "sig"
        mock_receipt.verdict = "approved"
        mock_receipt_store.get.return_value = mock_receipt

        mock_batch_result = MagicMock()
        mock_batch_result.receipt_id = "receipt-002"
        mock_batch_result.is_valid = True
        mock_batch_result.error = None
        mock_receipt_store.verify_batch.return_value = ([mock_batch_result], {})

        mock_audit_store.get_log.return_value = [
            {"id": "1", "action": "test", "timestamp": "2025-01-15T00:00:00Z"}
        ]

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await compliance_handler._verify_audit(
                {
                    "trail_id": "receipt-001",
                    "receipt_ids": ["receipt-002"],
                    "date_range": {"from": "2025-01-01", "to": "2025-01-31"},
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["checks"]) == 3  # trail, receipt, date_range


# ============================================================================
# Audit Events Export Tests
# ============================================================================


class TestAuditEventsExport:
    """Tests for SIEM-compatible audit event export."""

    @pytest.mark.asyncio
    async def test_get_events_json_format(self, compliance_handler, mock_audit_store):
        """Default export returns JSON format."""
        mock_audit_store.get_log.return_value = [
            {
                "id": "event-001",
                "action": "user_login",
                "timestamp": "2025-01-15T00:00:00Z",
            }
        ]

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._get_audit_events({})

        assert result.status_code == 200
        assert result.content_type == "application/json"

        body = json.loads(result.body)
        assert "events" in body
        assert "count" in body
        assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_get_events_ndjson_format(self, compliance_handler, mock_audit_store):
        """NDJSON export returns newline-delimited JSON."""
        mock_audit_store.get_log.return_value = [
            {"id": "1", "action": "login", "timestamp": "2025-01-15T00:00:00Z"},
            {"id": "2", "action": "logout", "timestamp": "2025-01-15T01:00:00Z"},
        ]

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._get_audit_events({"format": "ndjson"})

        assert result.status_code == 200
        assert result.content_type == "application/x-ndjson"

        content = result.body.decode("utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 2
        # Each line should be valid JSON
        for line in lines:
            json.loads(line)

    @pytest.mark.asyncio
    async def test_get_events_elasticsearch_format(self, compliance_handler, mock_audit_store):
        """Elasticsearch export returns bulk format."""
        mock_audit_store.get_log.return_value = [
            {
                "id": "event-001",
                "action": "user_login",
                "timestamp": "2025-01-15T00:00:00Z",
            }
        ]

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._get_audit_events({"format": "elasticsearch"})

        assert result.status_code == 200
        assert result.content_type == "application/x-ndjson"

        content = result.body.decode("utf-8")
        lines = content.strip().split("\n")
        # Should have 2 lines per event (index action + document)
        assert len(lines) == 2

        # First line is index action
        index_action = json.loads(lines[0])
        assert "index" in index_action
        assert index_action["index"]["_index"] == "aragora-audit"

        # Second line is document
        doc = json.loads(lines[1])
        assert "@timestamp" in doc
        assert "event.category" in doc
        assert doc["event.category"] == "audit"

    @pytest.mark.asyncio
    async def test_get_events_with_time_filter(self, compliance_handler, mock_audit_store):
        """Event export filters by time range."""
        mock_audit_store.get_log.return_value = [
            {"id": "1", "action": "a", "timestamp": "2025-01-10T00:00:00Z"},
            {"id": "2", "action": "b", "timestamp": "2025-01-15T00:00:00Z"},
            {"id": "3", "action": "c", "timestamp": "2025-01-20T00:00:00Z"},
        ]

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._get_audit_events(
                {"from": "2025-01-12T00:00:00Z", "to": "2025-01-18T00:00:00Z"}
            )

        body = json.loads(result.body)
        assert body["count"] == 1
        assert body["events"][0]["id"] == "2"

    @pytest.mark.asyncio
    async def test_get_events_with_type_filter(self, compliance_handler, mock_audit_store):
        """Event export filters by event type."""
        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            await compliance_handler._get_audit_events({"event_type": "user_login"})

        mock_audit_store.get_log.assert_called_once()
        call_kwargs = mock_audit_store.get_log.call_args.kwargs
        assert call_kwargs["action"] == "user_login"

    @pytest.mark.asyncio
    async def test_get_events_limit_default(self, compliance_handler, mock_audit_store):
        """Event export defaults to 1000 limit."""
        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            await compliance_handler._get_audit_events({})

        call_kwargs = mock_audit_store.get_log.call_args.kwargs
        assert call_kwargs["limit"] == 1000

    @pytest.mark.asyncio
    async def test_get_events_limit_max(self, compliance_handler, mock_audit_store):
        """Event export enforces max limit of 10000."""
        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            await compliance_handler._get_audit_events({"limit": "50000"})

        call_kwargs = mock_audit_store.get_log.call_args.kwargs
        assert call_kwargs["limit"] == 10000

    @pytest.mark.asyncio
    async def test_get_events_empty_result(self, compliance_handler, mock_audit_store):
        """Event export handles empty results."""
        mock_audit_store.get_log.return_value = []

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._get_audit_events({})

        body = json.loads(result.body)
        assert body["count"] == 0
        assert body["events"] == []

    @pytest.mark.asyncio
    async def test_get_events_store_error(self, compliance_handler, mock_audit_store):
        """Event export handles store errors gracefully."""
        mock_audit_store.get_log.side_effect = RuntimeError("Database error")

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._get_audit_events({})

        # Should return empty events, not error
        body = json.loads(result.body)
        assert body["events"] == []


# ============================================================================
# Timestamp Parsing Tests
# ============================================================================


class TestTimestampParsing:
    """Tests for timestamp parsing utility."""

    def test_parse_none(self, compliance_handler):
        """None value returns None."""
        result = compliance_handler._parse_timestamp(None)
        assert result is None

    def test_parse_unix_timestamp(self, compliance_handler):
        """Unix timestamp parses correctly."""
        # 2025-01-15 00:00:00 UTC
        result = compliance_handler._parse_timestamp("1736899200")
        assert result is not None
        assert result.year == 2025
        assert result.month == 1

    def test_parse_iso_timestamp(self, compliance_handler):
        """ISO timestamp parses correctly."""
        result = compliance_handler._parse_timestamp("2025-01-15T00:00:00Z")
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15

    def test_parse_iso_with_offset(self, compliance_handler):
        """ISO timestamp with offset parses correctly."""
        result = compliance_handler._parse_timestamp("2025-01-15T00:00:00+05:00")
        assert result is not None

    def test_parse_invalid_returns_none(self, compliance_handler):
        """Invalid timestamp returns None."""
        result = compliance_handler._parse_timestamp("not-a-timestamp")
        assert result is None


# ============================================================================
# RBAC Permission Tests
# ============================================================================


class TestAuditPermissions:
    """Tests for audit handler RBAC permission enforcement."""

    def test_verify_audit_has_permission_decorator(self):
        """Audit verify requires compliance:audit permission."""
        import inspect

        source = inspect.getsource(ComplianceHandler._verify_audit)
        assert "require_permission" in source
        assert "compliance:audit" in source

    def test_get_audit_events_has_permission_decorator(self):
        """Audit events requires compliance:audit permission."""
        import inspect

        source = inspect.getsource(ComplianceHandler._get_audit_events)
        assert "require_permission" in source
        assert "compliance:audit" in source


# ============================================================================
# Handler Tracking Tests
# ============================================================================


class TestAuditTracking:
    """Tests for handler metrics tracking."""

    def test_verify_audit_has_track_handler_decorator(self):
        """Audit verify has metrics tracking."""
        import inspect

        source = inspect.getsource(ComplianceHandler._verify_audit)
        assert "track_handler" in source
        assert "compliance/audit-verify" in source

    def test_get_audit_events_has_track_handler_decorator(self):
        """Audit events has metrics tracking."""
        import inspect

        source = inspect.getsource(ComplianceHandler._get_audit_events)
        assert "track_handler" in source
        assert "compliance/audit-events" in source


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestAuditEdgeCases:
    """Tests for edge cases in audit operations."""

    @pytest.mark.asyncio
    async def test_verify_trail_handles_exception(self, compliance_handler, mock_receipt_store):
        """Trail verification handles exceptions."""
        mock_receipt_store.get.side_effect = RuntimeError("Database error")

        with patch(
            "aragora.server.handlers.compliance_handler.get_receipt_store",
            return_value=mock_receipt_store,
        ):
            result = await compliance_handler._verify_audit({"trail_id": "test"})

        body = json.loads(result.body)
        assert body["verified"] is False
        assert body["checks"][0]["valid"] is False
        assert "error" in body["checks"][0]

    @pytest.mark.asyncio
    async def test_verify_date_range_handles_exception(self, compliance_handler, mock_audit_store):
        """Date range verification handles exceptions."""
        mock_audit_store.get_log.side_effect = RuntimeError("Database error")

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._verify_audit(
                {"date_range": {"from": "2025-01-01", "to": "2025-01-31"}}
            )

        body = json.loads(result.body)
        assert body["verified"] is False

    @pytest.mark.asyncio
    async def test_get_events_elasticsearch_uses_event_id(
        self, compliance_handler, mock_audit_store
    ):
        """Elasticsearch export uses event_id field if id not present."""
        mock_audit_store.get_log.return_value = [
            {
                "event_id": "evt-001",
                "action": "test",
                "timestamp": "2025-01-15T00:00:00Z",
            }
        ]

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._get_audit_events({"format": "elasticsearch"})

        content = result.body.decode("utf-8")
        lines = content.strip().split("\n")
        index_action = json.loads(lines[0])
        assert index_action["index"]["_id"] == "evt-001"

    @pytest.mark.asyncio
    async def test_get_events_handles_unparseable_timestamps(
        self, compliance_handler, mock_audit_store
    ):
        """Event export handles events with unparseable timestamps."""
        mock_audit_store.get_log.return_value = [
            {"id": "1", "action": "test", "timestamp": "invalid-date"},
            {"id": "2", "action": "test", "timestamp": "2025-01-15T00:00:00Z"},
        ]

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._get_audit_events({"from": "2025-01-01T00:00:00Z"})

        # Both events should be included (invalid timestamp event is not filtered)
        body = json.loads(result.body)
        assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_verify_date_range_filters_by_date(self, compliance_handler, mock_audit_store):
        """Date range verification filters events outside range."""
        mock_audit_store.get_log.return_value = [
            {"id": "1", "action": "a", "timestamp": "2024-12-15T00:00:00Z"},  # Before
            {"id": "2", "action": "b", "timestamp": "2025-01-15T00:00:00Z"},  # In range
            {"id": "3", "action": "c", "timestamp": "2025-02-15T00:00:00Z"},  # After
        ]

        with patch(
            "aragora.server.handlers.compliance_handler.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await compliance_handler._verify_audit(
                {
                    "date_range": {
                        "from": "2025-01-01T00:00:00Z",
                        "to": "2025-01-31T00:00:00Z",
                    }
                }
            )

        body = json.loads(result.body)
        range_check = body["checks"][0]
        assert range_check["events_checked"] == 1

    @pytest.mark.asyncio
    async def test_verify_receipt_unsigned(self, compliance_handler, mock_receipt_store):
        """Verification shows unsigned status for receipt without signature."""
        mock_receipt = MagicMock()
        mock_receipt.receipt_id = "receipt-001"
        mock_receipt.signature = None
        mock_receipt.verdict = "approved"
        mock_receipt_store.get.return_value = mock_receipt

        with patch(
            "aragora.server.handlers.compliance_handler.get_receipt_store",
            return_value=mock_receipt_store,
        ):
            result = await compliance_handler._verify_audit({"trail_id": "receipt-001"})

        body = json.loads(result.body)
        assert body["checks"][0]["signed"] is False
