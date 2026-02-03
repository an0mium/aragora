"""Tests for integrations/openclaw/audit_bridge.py — OpenClaw audit bridge."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.integrations.openclaw.audit_bridge import (
    AuditRecord,
    OpenClawAuditBridge,
)


# =============================================================================
# AuditRecord
# =============================================================================


class TestAuditRecord:
    def test_defaults(self):
        rec = AuditRecord(
            record_id="r-1",
            event_type="shell_exec",
            timestamp=1000.0,
            source="openclaw_proxy",
        )
        assert rec.record_id == "r-1"
        assert rec.success is True
        assert rec.signature is None
        assert rec.metadata == {}

    def test_full_record(self):
        rec = AuditRecord(
            record_id="r-2",
            event_type="file_write",
            timestamp=2000.0,
            source="openclaw_proxy",
            user_id="u-1",
            session_id="s-1",
            tenant_id="t-1",
            action_type="write",
            action_target="/tmp/file.txt",
            action_params={"size": 100},
            success=False,
            error="permission denied",
        )
        assert rec.user_id == "u-1"
        assert rec.success is False
        assert rec.error == "permission denied"


# =============================================================================
# OpenClawAuditBridge — init and config
# =============================================================================


class TestBridgeInit:
    def test_defaults(self):
        bridge = OpenClawAuditBridge()
        assert bridge._km is None
        assert bridge._workspace_id == "default"
        assert bridge._signing_key is None
        assert bridge._batch == []

    def test_custom_config(self):
        km = MagicMock()
        bridge = OpenClawAuditBridge(
            knowledge_mound=km,
            workspace_id="enterprise",
            signing_key="secret-key",
            batch_size=50,
            flush_interval_seconds=10.0,
        )
        assert bridge._km is km
        assert bridge._workspace_id == "enterprise"
        assert bridge._signing_key == "secret-key"
        assert bridge._batch_size == 50


# =============================================================================
# capture_event
# =============================================================================


class TestCaptureEvent:
    def test_capture_adds_to_batch(self):
        bridge = OpenClawAuditBridge(batch_size=100)
        bridge.capture_event({"event_type": "shell_exec", "command": "ls -la"})
        assert len(bridge._batch) == 1
        assert bridge._stats["events_captured"] == 1

    def test_capture_with_signing(self):
        bridge = OpenClawAuditBridge(signing_key="test-key", batch_size=100)
        bridge.capture_event({"event_type": "shell_exec"})
        record = bridge._batch[0]
        assert record.signature is not None
        assert record.previous_hash is None  # first record

    def test_capture_chain_hashes(self):
        bridge = OpenClawAuditBridge(signing_key="test-key", batch_size=100)
        bridge.capture_event({"event_type": "event1"})
        bridge.capture_event({"event_type": "event2"})
        assert bridge._batch[1].previous_hash == bridge._batch[0].signature

    def test_capture_triggers_callback(self):
        callback = MagicMock()
        bridge = OpenClawAuditBridge(event_callback=callback, batch_size=100)
        bridge.capture_event({"event_type": "test"})
        callback.assert_called_once()
        assert isinstance(callback.call_args[0][0], AuditRecord)

    def test_capture_callback_failure_does_not_raise(self):
        callback = MagicMock(side_effect=RuntimeError("callback failed"))
        bridge = OpenClawAuditBridge(event_callback=callback, batch_size=100)
        bridge.capture_event({"event_type": "test"})  # should not raise
        assert bridge._stats["events_captured"] == 1

    def test_capture_triggers_flush_on_batch_size(self):
        bridge = OpenClawAuditBridge(batch_size=2)
        bridge.capture_event({"event_type": "event1"})
        bridge.capture_event({"event_type": "event2"})
        # Batch flushed (no KM so records dropped silently)
        assert bridge._stats["batches_flushed"] == 1
        assert len(bridge._batch) == 0


# =============================================================================
# _event_to_record
# =============================================================================


class TestEventToRecord:
    def test_basic_event(self):
        bridge = OpenClawAuditBridge()
        record = bridge._event_to_record(
            {
                "event_type": "shell_exec",
                "command": "ls",
                "user_id": "u-1",
                "session_id": "s-1",
            }
        )
        assert record.event_type == "shell_exec"
        assert record.user_id == "u-1"
        assert record.action_target == "ls"

    def test_path_as_action_target(self):
        bridge = OpenClawAuditBridge()
        record = bridge._event_to_record({"event_type": "file_read", "path": "/etc/passwd"})
        assert record.action_target == "/etc/passwd"

    def test_url_as_action_target(self):
        bridge = OpenClawAuditBridge()
        record = bridge._event_to_record(
            {"event_type": "browser_navigate", "url": "https://example.com"}
        )
        assert record.action_target == "https://example.com"

    def test_error_from_reason_field(self):
        bridge = OpenClawAuditBridge()
        record = bridge._event_to_record(
            {"event_type": "shell_exec", "success": False, "reason": "timeout"}
        )
        assert record.error == "timeout"
        assert record.success is False


# =============================================================================
# _sign_record
# =============================================================================


class TestSignRecord:
    def test_signature_deterministic(self):
        bridge = OpenClawAuditBridge(signing_key="key")
        record = AuditRecord(record_id="r-1", event_type="test", timestamp=1000.0, source="proxy")
        sig1 = bridge._sign_record(record)
        sig2 = bridge._sign_record(record)
        assert sig1 == sig2

    def test_different_records_different_sigs(self):
        bridge = OpenClawAuditBridge(signing_key="key")
        r1 = AuditRecord(record_id="r-1", event_type="a", timestamp=1000.0, source="proxy")
        r2 = AuditRecord(record_id="r-2", event_type="b", timestamp=2000.0, source="proxy")
        assert bridge._sign_record(r1) != bridge._sign_record(r2)


# =============================================================================
# flush (async)
# =============================================================================


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_empty(self):
        bridge = OpenClawAuditBridge()
        count = await bridge.flush()
        assert count == 0

    @pytest.mark.asyncio
    async def test_flush_with_km(self):
        km = AsyncMock()
        km.ingest = AsyncMock(return_value="item-id")
        bridge = OpenClawAuditBridge(knowledge_mound=km, batch_size=100)
        bridge.capture_event({"event_type": "test1"})
        bridge.capture_event({"event_type": "test2"})
        count = await bridge.flush()
        assert count == 2
        assert km.ingest.call_count == 2
        assert bridge._stats["events_stored"] == 2

    @pytest.mark.asyncio
    async def test_flush_handles_km_error(self):
        km = AsyncMock()
        km.ingest = AsyncMock(side_effect=RuntimeError("KM down"))
        bridge = OpenClawAuditBridge(knowledge_mound=km, batch_size=100)
        bridge.capture_event({"event_type": "test"})
        count = await bridge.flush()
        assert count == 0
        assert bridge._stats["events_failed"] == 1

    @pytest.mark.asyncio
    async def test_flush_no_km(self):
        bridge = OpenClawAuditBridge(batch_size=100)
        bridge.capture_event({"event_type": "test"})
        count = await bridge.flush()
        assert count == 1  # stored but no-op without KM


# =============================================================================
# _store_record_sync
# =============================================================================


class TestStoreRecordSync:
    def test_no_km_returns_none(self):
        bridge = OpenClawAuditBridge()
        record = AuditRecord(record_id="r-1", event_type="test", timestamp=0, source="x")
        assert bridge._store_record_sync(record) is None

    def test_with_km_calls_ingest_sync(self):
        km = MagicMock()
        km.ingest_sync = MagicMock(return_value="item-1")
        bridge = OpenClawAuditBridge(knowledge_mound=km)
        record = AuditRecord(record_id="r-1", event_type="test", timestamp=0, source="x")
        result = bridge._store_record_sync(record)
        assert result == "item-1"
        km.ingest_sync.assert_called_once()


# =============================================================================
# query_audit_trail
# =============================================================================


class TestQueryAuditTrail:
    @pytest.mark.asyncio
    async def test_no_km_returns_empty(self):
        bridge = OpenClawAuditBridge()
        results = await bridge.query_audit_trail(user_id="u-1")
        assert results == []

    @pytest.mark.asyncio
    async def test_with_results(self):
        km = AsyncMock()
        km.query = AsyncMock(
            return_value=[
                {
                    "content": {
                        "record_id": "r-1",
                        "event_type": "shell_exec",
                        "timestamp": 1000.0,
                        "source": "proxy",
                        "user_id": "u-1",
                    },
                    "metadata": {"signature": "sig-1"},
                }
            ]
        )
        bridge = OpenClawAuditBridge(knowledge_mound=km)
        results = await bridge.query_audit_trail(user_id="u-1")
        assert len(results) == 1
        assert results[0].record_id == "r-1"
        assert results[0].signature == "sig-1"

    @pytest.mark.asyncio
    async def test_query_error_returns_empty(self):
        km = AsyncMock()
        km.query = AsyncMock(side_effect=RuntimeError("query failed"))
        bridge = OpenClawAuditBridge(knowledge_mound=km)
        results = await bridge.query_audit_trail()
        assert results == []


# =============================================================================
# verify_integrity
# =============================================================================


class TestVerifyIntegrity:
    @pytest.mark.asyncio
    async def test_no_records(self):
        bridge = OpenClawAuditBridge()
        result = await bridge.verify_integrity()
        assert result["valid"] is True
        assert result["records_checked"] == 0

    @pytest.mark.asyncio
    async def test_valid_chain(self):
        km = AsyncMock()
        km.query = AsyncMock(
            return_value=[
                {
                    "content": {
                        "record_id": "r-1",
                        "event_type": "test",
                        "timestamp": 1000.0,
                        "source": "proxy",
                    },
                    "metadata": {"signature": "sig-1", "previous_hash": None},
                },
                {
                    "content": {
                        "record_id": "r-2",
                        "event_type": "test",
                        "timestamp": 2000.0,
                        "source": "proxy",
                    },
                    "metadata": {"signature": "sig-2", "previous_hash": "sig-1"},
                },
            ]
        )
        bridge = OpenClawAuditBridge(knowledge_mound=km)
        result = await bridge.verify_integrity()
        assert result["valid"] is True
        assert result["records_checked"] == 2

    @pytest.mark.asyncio
    async def test_broken_chain(self):
        km = AsyncMock()
        km.query = AsyncMock(
            return_value=[
                {
                    "content": {
                        "record_id": "r-1",
                        "event_type": "test",
                        "timestamp": 1000.0,
                        "source": "proxy",
                    },
                    "metadata": {"signature": "sig-1", "previous_hash": None},
                },
                {
                    "content": {
                        "record_id": "r-2",
                        "event_type": "test",
                        "timestamp": 2000.0,
                        "source": "proxy",
                    },
                    "metadata": {"signature": "sig-2", "previous_hash": "WRONG"},
                },
            ]
        )
        bridge = OpenClawAuditBridge(knowledge_mound=km)
        result = await bridge.verify_integrity()
        assert result["valid"] is False
        assert len(result["issues"]) == 1
        assert result["issues"][0]["type"] == "chain_break"


# =============================================================================
# get_stats
# =============================================================================


class TestGetStats:
    def test_initial_stats(self):
        bridge = OpenClawAuditBridge()
        stats = bridge.get_stats()
        assert stats["events_captured"] == 0
        assert stats["has_knowledge_mound"] is False
        assert stats["signing_enabled"] is False

    def test_stats_after_events(self):
        bridge = OpenClawAuditBridge(signing_key="key", batch_size=100)
        bridge.capture_event({"event_type": "test"})
        stats = bridge.get_stats()
        assert stats["events_captured"] == 1
        assert stats["pending_batch_size"] == 1
        assert stats["signing_enabled"] is True


# =============================================================================
# close
# =============================================================================


class TestClose:
    @pytest.mark.asyncio
    async def test_close_flushes(self):
        km = AsyncMock()
        km.ingest = AsyncMock(return_value="item-id")
        bridge = OpenClawAuditBridge(knowledge_mound=km, batch_size=100)
        bridge.capture_event({"event_type": "pending"})
        await bridge.close()
        assert len(bridge._batch) == 0
        assert bridge._stats["events_stored"] == 1
