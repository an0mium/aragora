"""
Edge case tests for observability modules.

Comprehensive tests covering edge cases, error handling, and boundary conditions for:
1. N+1 Query Detection (n1_detector.py)
2. SIEM Integration (siem.py)
3. Memory Profiler (memory_profiler.py)
4. Immutable Audit Log (immutable_log.py)

Target: 25-30 tests covering edge cases not in primary test files.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# =============================================================================
# N+1 Query Detector Edge Cases
# =============================================================================

from aragora.observability.n1_detector import (
    N1Detection,
    N1QueryDetector,
    N1QueryError,
    QueryRecord,
    _parse_detection_mode,
    _parse_threshold,
    detect_n1,
    get_current_detector,
    n1_detection_scope,
    record_query,
)


class TestN1DetectorEdgeCases:
    """Edge case tests for N+1 query detector."""

    def test_parse_detection_mode_invalid_values(self):
        """Invalid mode should fall back to 'off' with warning."""
        with patch("aragora.observability.n1_detector.logger") as mock_logger:
            result = _parse_detection_mode("invalid")
            assert result == "off"
            mock_logger.warning.assert_called()

    def test_parse_detection_mode_case_insensitive(self):
        """Mode parsing should be case insensitive."""
        assert _parse_detection_mode("WARN") == "warn"
        assert _parse_detection_mode("Error") == "error"
        assert _parse_detection_mode("OFF") == "off"

    def test_parse_threshold_negative_value(self):
        """Negative threshold should be normalized to 1."""
        with patch("aragora.observability.n1_detector.logger") as mock_logger:
            result = _parse_threshold("-5")
            assert result == 1
            mock_logger.warning.assert_called()

    def test_parse_threshold_zero_value(self):
        """Zero threshold should be normalized to 1."""
        with patch("aragora.observability.n1_detector.logger") as mock_logger:
            result = _parse_threshold("0")
            assert result == 1
            mock_logger.warning.assert_called()

    def test_parse_threshold_non_numeric(self):
        """Non-numeric threshold should use default."""
        with patch("aragora.observability.n1_detector.logger") as mock_logger:
            result = _parse_threshold("abc")
            assert result == 5
            mock_logger.warning.assert_called()

    def test_normalize_query_with_in_clause(self):
        """Should normalize IN clauses with varying items."""
        detector = N1QueryDetector()
        query1 = "SELECT * FROM users WHERE id IN (1, 2, 3)"
        query2 = "SELECT * FROM users WHERE id IN (4, 5, 6, 7, 8)"

        # Both should normalize to the same pattern
        assert detector._normalize_query(query1) == detector._normalize_query(query2)

    def test_normalize_query_whitespace_handling(self):
        """Should handle irregular whitespace in queries."""
        detector = N1QueryDetector()
        query1 = "SELECT   *   FROM    users WHERE   id = 1"
        query2 = "SELECT * FROM users WHERE id = 1"

        assert detector._normalize_query(query1) == detector._normalize_query(query2)

    def test_analyze_empty_queries(self):
        """Should return empty dict when no queries recorded."""
        detector = N1QueryDetector(mode="warn")
        result = detector.analyze()
        assert result == {}

    def test_context_manager_exception_propagation(self):
        """Context manager should propagate exceptions from inner code."""
        detector = N1QueryDetector(mode="warn", threshold=10)

        with pytest.raises(ValueError):
            with detector:
                raise ValueError("Test error")

    def test_context_manager_error_mode_raises_on_violation(self):
        """Error mode should raise N1QueryError when threshold exceeded."""
        detector = N1QueryDetector(mode="error", threshold=2)

        with pytest.raises(N1QueryError) as exc_info:
            with detector:
                for i in range(5):
                    detector.record_query("users", f"SELECT * FROM users WHERE id = {i}")

        assert exc_info.value.table == "users"
        assert exc_info.value.count >= 2

    def test_record_query_with_duration(self):
        """Should track query duration in total calculation."""
        detector = N1QueryDetector(mode="warn", threshold=10)

        detector.record_query("users", "SELECT * FROM users WHERE id = 1", duration_ms=10.5)
        detector.record_query("users", "SELECT * FROM users WHERE id = 2", duration_ms=20.5)

        results = detector.analyze()
        assert results["users"].total_duration_ms == 31.0

    def test_get_violations_filters_below_threshold(self):
        """Should only return tables exceeding threshold."""
        detector = N1QueryDetector(mode="warn", threshold=3)

        # Add 5 queries to 'users' (above threshold)
        for i in range(5):
            detector.record_query("users", f"SELECT * FROM users WHERE id = {i}")

        # Add 2 queries to 'posts' (below threshold)
        for i in range(2):
            detector.record_query("posts", f"SELECT * FROM posts WHERE id = {i}")

        violations = detector.get_violations()
        assert len(violations) == 1
        assert violations[0].table == "users"

    def test_include_patterns_option(self):
        """Should include query patterns when option is set."""
        detector = N1QueryDetector(mode="warn", include_patterns=True)

        detector.record_query("users", "SELECT * FROM users WHERE id = 1")

        results = detector.analyze()
        assert len(results["users"].queries) == 1
        assert results["users"].queries[0].table == "users"

    def test_global_record_query_no_detector(self):
        """Global record_query should be no-op without active detector."""
        # Should not raise
        record_query("table", "SELECT * FROM table", 1.0)
        assert get_current_detector() is None

    def test_detect_n1_decorator_async_function(self):
        """Decorator should work with async functions."""

        @detect_n1(threshold=2, mode="warn")
        async def async_handler():
            detector = get_current_detector()
            assert detector is not None
            return "result"

        result = asyncio.run(async_handler())
        assert result == "result"

    def test_detect_n1_decorator_sync_function(self):
        """Decorator should work with sync functions."""

        @detect_n1(threshold=2, mode="warn")
        def sync_handler():
            detector = get_current_detector()
            assert detector is not None
            return "result"

        result = sync_handler()
        assert result == "result"


# =============================================================================
# SIEM Integration Edge Cases
# =============================================================================

from aragora.observability.siem import (
    SecurityEvent,
    SecurityEventType,
    SIEMBackend,
    SIEMClient,
    SIEMConfig,
    emit_security_event,
)


class TestSIEMEdgeCases:
    """Edge case tests for SIEM integration."""

    def test_security_event_minimal_fields(self):
        """Event should work with only required fields."""
        event = SecurityEvent(event_type=SecurityEventType.AUTH_LOGIN_SUCCESS)

        d = event.to_dict()
        assert d["event_type"] == "auth.login.success"
        assert d["user_id"] is None
        assert d["source"] == "aragora"

    def test_security_event_all_severity_levels(self):
        """Should support all severity levels."""
        for severity in ["info", "warning", "error", "critical"]:
            event = SecurityEvent(
                event_type=SecurityEventType.SECURITY_RATE_LIMIT, severity=severity
            )
            assert event.severity == severity

    def test_security_event_complex_metadata(self):
        """Should handle complex nested metadata."""
        metadata = {
            "nested": {"level1": {"level2": [1, 2, 3]}},
            "list": [{"a": 1}, {"b": 2}],
            "unicode": "test",
        }
        event = SecurityEvent(event_type=SecurityEventType.DATA_READ, metadata=metadata)

        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["metadata"]["nested"]["level1"]["level2"] == [1, 2, 3]

    def test_siem_config_disabled(self):
        """Config with disabled flag should prevent emission."""
        with patch.dict(os.environ, {"SIEM_ENABLED": "false"}):
            config = SIEMConfig.from_env()
            assert config.enabled is False

    def test_siem_client_emit_when_disabled(self):
        """Emit should be no-op when disabled."""
        config = SIEMConfig(enabled=False, backend=SIEMBackend.SPLUNK)
        client = SIEMClient(config)

        event = SecurityEvent(event_type=SecurityEventType.AUTH_LOGIN_SUCCESS)
        # Should not raise
        client.emit(event)

        # Queue should be empty since disabled
        assert client._queue.empty()

    def test_siem_client_shutdown_timeout(self):
        """Shutdown should respect timeout."""
        config = SIEMConfig(backend=SIEMBackend.NONE, enabled=True)
        client = SIEMClient(config)

        start = time.time()
        client.shutdown(timeout=0.1)
        elapsed = time.time() - start

        # Should complete quickly since no worker
        assert elapsed < 1.0

    def test_siem_client_splunk_backend_batch(self):
        """Splunk backend should format events correctly."""
        config = SIEMConfig(
            backend=SIEMBackend.SPLUNK,
            endpoint="https://test.splunk.com/hec",
            token="test-token",
            index="test-index",
            enabled=True,
        )

        with patch("httpx.post") as mock_post:
            client = SIEMClient(config)

            events = [
                SecurityEvent(event_type=SecurityEventType.AUTH_LOGIN_SUCCESS, user_id="user-1")
            ]
            client._send_batch(events)

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "Splunk test-token" in str(call_args)

    def test_siem_client_cloudwatch_without_boto3(self):
        """CloudWatch backend should handle missing boto3 gracefully."""
        config = SIEMConfig(backend=SIEMBackend.CLOUDWATCH, enabled=True)
        client = SIEMClient(config)

        events = [SecurityEvent(event_type=SecurityEventType.AUTH_LOGIN_SUCCESS)]

        # Mock the import to fail
        with patch.dict("sys.modules", {"boto3": None}):
            with patch("aragora.observability.siem.logger") as mock_logger:
                try:
                    client._send_to_cloudwatch(events)
                except ImportError:
                    pass  # Expected in some environments

    def test_siem_client_empty_batch(self):
        """Send batch should handle empty list."""
        config = SIEMConfig(backend=SIEMBackend.SPLUNK, enabled=True)
        client = SIEMClient(config)

        # Should not raise
        client._send_batch([])


# =============================================================================
# Memory Profiler Edge Cases
# =============================================================================

from aragora.observability.memory_profiler import (
    AllocationRecord,
    MemoryCategory,
    MemoryProfiler,
    MemoryProfileResult,
    MemorySnapshot,
    MEMORY_CRITICAL_MB,
    MEMORY_WARNING_MB,
)


class TestMemoryProfilerEdgeCases:
    """Edge case tests for memory profiler."""

    def test_memory_snapshot_zero_values(self):
        """Snapshot should handle zero values correctly."""
        snapshot = MemorySnapshot(
            timestamp=0.0,
            current_bytes=0,
            peak_bytes=0,
            traced_bytes=0,
            traced_blocks=0,
            gc_objects=0,
        )

        assert snapshot.current_mb == 0.0
        assert snapshot.peak_mb == 0.0
        assert snapshot.traced_mb == 0.0

    def test_memory_snapshot_large_values(self):
        """Snapshot should handle large values (GB scale)."""
        gb = 1024 * 1024 * 1024  # 1 GB in bytes
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            current_bytes=5 * gb,
            peak_bytes=10 * gb,
            traced_bytes=5 * gb,
            traced_blocks=1000000,
            gc_objects=5000000,
        )

        assert snapshot.current_mb == 5 * 1024  # 5 GB = 5120 MB
        assert snapshot.peak_mb == 10 * 1024  # 10 GB = 10240 MB

    def test_allocation_record_formatting(self):
        """AllocationRecord str format should be readable."""
        record = AllocationRecord(
            file="/very/long/path/to/some/deeply/nested/file.py",
            line=999,
            size_bytes=50 * 1024 * 1024,  # 50 MB
            count=5000,
        )

        str_repr = str(record)
        assert "file.py:999" in str_repr
        assert "50.00MB" in str_repr
        assert "5000 blocks" in str_repr

    def test_memory_profile_result_negative_delta(self):
        """Result should handle negative memory delta (freed memory)."""
        start = MemorySnapshot(
            timestamp=1.0,
            current_bytes=100 * 1024 * 1024,  # 100 MB
            peak_bytes=100 * 1024 * 1024,
            traced_bytes=100 * 1024 * 1024,
            traced_blocks=1000,
            gc_objects=10000,
        )
        end = MemorySnapshot(
            timestamp=2.0,
            current_bytes=50 * 1024 * 1024,  # 50 MB (freed 50 MB)
            peak_bytes=100 * 1024 * 1024,
            traced_bytes=50 * 1024 * 1024,
            traced_blocks=500,
            gc_objects=5000,
        )

        result = MemoryProfileResult(
            category=MemoryCategory.GENERAL,
            operation="test",
            start_snapshot=start,
            end_snapshot=end,
            duration_ms=100.0,
        )

        assert result.delta_mb == -50.0  # Negative indicates freed memory
        assert result.delta_bytes == -50 * 1024 * 1024

    def test_memory_profiler_exception_in_context(self):
        """Profiler should still capture data even if exception raised."""
        profiler = MemoryProfiler(category=MemoryCategory.GENERAL)

        with pytest.raises(RuntimeError):
            with profiler.profile("exception_test"):
                raise RuntimeError("Test error")

        # Result should still be captured
        assert profiler.result is not None
        assert profiler.result.operation == "exception_test"

    def test_memory_profile_result_report_with_warnings(self):
        """Report should include warnings when present."""
        start = MemorySnapshot(
            timestamp=1.0,
            current_bytes=0,
            peak_bytes=0,
            traced_bytes=0,
            traced_blocks=0,
            gc_objects=0,
        )
        end = MemorySnapshot(
            timestamp=2.0,
            current_bytes=0,
            peak_bytes=0,
            traced_bytes=0,
            traced_blocks=0,
            gc_objects=0,
        )

        result = MemoryProfileResult(
            category=MemoryCategory.KM_STORE,
            operation="test",
            start_snapshot=start,
            end_snapshot=end,
            duration_ms=50.0,
            warnings=["Test warning 1", "Test warning 2"],
        )

        report = result.report()
        assert "WARNINGS:" in report
        assert "Test warning 1" in report
        assert "Test warning 2" in report

    def test_memory_profile_result_report_verbose(self):
        """Verbose report should show more allocations."""
        start = MemorySnapshot(
            timestamp=1.0,
            current_bytes=0,
            peak_bytes=0,
            traced_bytes=0,
            traced_blocks=0,
            gc_objects=0,
        )
        end = MemorySnapshot(
            timestamp=2.0,
            current_bytes=0,
            peak_bytes=0,
            traced_bytes=0,
            traced_blocks=0,
            gc_objects=0,
        )

        # Create many allocation records
        allocations = [
            AllocationRecord(file=f"file{i}.py", line=i, size_bytes=1024 * i, count=i)
            for i in range(1, 15)
        ]

        result = MemoryProfileResult(
            category=MemoryCategory.GENERAL,
            operation="test",
            start_snapshot=start,
            end_snapshot=end,
            duration_ms=50.0,
            top_allocations=allocations,
        )

        # Non-verbose should show fewer
        report_normal = result.report(verbose=False)
        report_verbose = result.report(verbose=True)

        # Verbose should have more content
        assert len(report_verbose) > len(report_normal)

    @pytest.mark.slow
    @pytest.mark.slow
    def test_all_memory_categories(self):
        """All memory categories should be valid."""
        for category in MemoryCategory:
            profiler = MemoryProfiler(category=category)
            with profiler.profile("test"):
                pass
            assert profiler.result.category == category


# =============================================================================
# Immutable Audit Log Edge Cases
# =============================================================================

from aragora.observability.log_types import (
    AuditBackend,
    AuditEntry,
    DailyAnchor,
    VerificationResult,
)
from aragora.observability.log_backends import (
    AuditLogBackend,
    LocalFileBackend,
)
from aragora.observability.immutable_log import ImmutableAuditLog


class TestImmutableLogEdgeCases:
    """Edge case tests for immutable audit log."""

    def test_audit_entry_from_dict_missing_optional_fields(self):
        """Should handle missing optional fields gracefully."""
        data = {
            "id": "entry-1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence_number": 1,
            "previous_hash": "0" * 64,
            "entry_hash": "a" * 64,
            "event_type": "test",
            "actor": "user@test.com",
            "resource_type": "doc",
            "resource_id": "doc-1",
            "action": "create",
            # Missing: actor_type, details, correlation_id, workspace_id, etc.
        }

        entry = AuditEntry.from_dict(data)
        assert entry.actor_type == "user"  # Default
        assert entry.details == {}
        assert entry.correlation_id is None

    def test_audit_entry_compute_hash_different_previous_hash(self):
        """Hash should change when previous_hash changes."""
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

        entry1 = AuditEntry(
            id="entry-1",
            sequence_number=2,
            timestamp=timestamp,
            previous_hash="a" * 64,
            entry_hash="",
            event_type="test",
            actor="user",
            actor_type="user",
            resource_type="doc",
            resource_id="doc-1",
            action="create",
        )

        entry2 = AuditEntry(
            id="entry-1",
            sequence_number=2,
            timestamp=timestamp,
            previous_hash="b" * 64,  # Different previous hash
            entry_hash="",
            event_type="test",
            actor="user",
            actor_type="user",
            resource_type="doc",
            resource_id="doc-1",
            action="create",
        )

        assert entry1.compute_hash() != entry2.compute_hash()

    def test_daily_anchor_to_dict(self):
        """DailyAnchor should serialize correctly."""
        anchor = DailyAnchor(
            date="2024-01-15",
            first_sequence=1,
            last_sequence=100,
            entry_count=100,
            merkle_root="m" * 64,
            chain_hash="c" * 64,
            created_at=datetime(2024, 1, 15, 23, 59, 59, tzinfo=timezone.utc),
        )

        d = anchor.to_dict()
        assert d["date"] == "2024-01-15"
        assert d["entry_count"] == 100
        assert "created_at" in d

    def test_verification_result_with_errors(self):
        """VerificationResult should track errors properly."""
        result = VerificationResult(
            is_valid=False,
            entries_checked=50,
            errors=["Hash mismatch at seq 25", "Missing entry at seq 30"],
            warnings=["Gap in sequence"],
            first_error_sequence=25,
            verification_time_ms=150.5,
        )

        assert not result.is_valid
        assert len(result.errors) == 2
        assert result.first_error_sequence == 25

    @pytest.mark.asyncio
    async def test_local_file_backend_empty_directory(self):
        """Backend should handle empty directory gracefully."""
        with TemporaryDirectory() as tmp_dir:
            backend = LocalFileBackend(tmp_dir)

            # Should return None for missing entry
            entry = await backend.get_last_entry()
            assert entry is None

            # Should return empty list for range query
            entries = await backend.get_entries_range(1, 10)
            assert entries == []

    @pytest.mark.asyncio
    async def test_local_file_backend_append_and_retrieve(self):
        """Should append and retrieve entries correctly."""
        with TemporaryDirectory() as tmp_dir:
            backend = LocalFileBackend(tmp_dir)

            entry = AuditEntry(
                id="test-entry-1",
                sequence_number=1,
                timestamp=datetime.now(timezone.utc),
                previous_hash="0" * 64,
                entry_hash="a" * 64,
                event_type="test",
                actor="user@test.com",
                actor_type="user",
                resource_type="document",
                resource_id="doc-123",
                action="create",
            )

            await backend.append(entry)

            # Retrieve by ID
            retrieved = await backend.get_entry("test-entry-1")
            assert retrieved is not None
            assert retrieved.id == "test-entry-1"

            # Retrieve by sequence
            by_seq = await backend.get_by_sequence(1)
            assert by_seq is not None
            assert by_seq.sequence_number == 1

    @pytest.mark.asyncio
    async def test_immutable_log_merkle_root_empty(self):
        """Merkle root of empty list should be genesis hash."""
        with TemporaryDirectory() as tmp_dir:
            backend = LocalFileBackend(tmp_dir)
            log = ImmutableAuditLog(backend=backend)

            root = log._compute_merkle_root([])
            assert root == ImmutableAuditLog.GENESIS_HASH

    @pytest.mark.asyncio
    async def test_immutable_log_merkle_root_single_hash(self):
        """Merkle root of single hash should be that hash after padding."""
        with TemporaryDirectory() as tmp_dir:
            backend = LocalFileBackend(tmp_dir)
            log = ImmutableAuditLog(backend=backend)

            test_hash = "abc123" + "0" * 58
            root = log._compute_merkle_root([test_hash])

            # Single hash gets padded and combined with itself
            assert len(root) == 64  # SHA-256 hex length

    @pytest.mark.asyncio
    async def test_immutable_log_sign_entry_without_key(self):
        """Signing without key should return None."""
        with TemporaryDirectory() as tmp_dir:
            backend = LocalFileBackend(tmp_dir)
            log = ImmutableAuditLog(backend=backend, signing_key=None)

            entry = AuditEntry(
                id="test",
                sequence_number=1,
                timestamp=datetime.now(timezone.utc),
                previous_hash="0" * 64,
                entry_hash="a" * 64,
                event_type="test",
                actor="user",
                actor_type="user",
                resource_type="doc",
                resource_id="1",
                action="create",
            )

            signature = log._sign_entry(entry)
            assert signature is None

    @pytest.mark.asyncio
    async def test_immutable_log_sign_entry_with_key(self):
        """Signing with key should produce HMAC signature."""
        with TemporaryDirectory() as tmp_dir:
            backend = LocalFileBackend(tmp_dir)
            signing_key = b"test-secret-key-12345"
            log = ImmutableAuditLog(backend=backend, signing_key=signing_key)

            entry = AuditEntry(
                id="test",
                sequence_number=1,
                timestamp=datetime.now(timezone.utc),
                previous_hash="0" * 64,
                entry_hash="a" * 64,
                event_type="test",
                actor="user",
                actor_type="user",
                resource_type="doc",
                resource_id="1",
                action="create",
            )

            signature = log._sign_entry(entry)
            assert signature is not None
            assert len(signature) == 64  # SHA-256 HMAC hex length

    @pytest.mark.asyncio
    async def test_immutable_log_append_first_entry(self):
        """First entry should use genesis hash as previous."""
        with TemporaryDirectory() as tmp_dir:
            backend = LocalFileBackend(tmp_dir)
            log = ImmutableAuditLog(backend=backend)

            entry = await log.append(
                event_type="first_event",
                actor="user@test.com",
                resource_type="doc",
                resource_id="doc-1",
                action="create",
            )

            assert entry.sequence_number == 1
            assert entry.previous_hash == ImmutableAuditLog.GENESIS_HASH

    @pytest.mark.asyncio
    async def test_immutable_log_append_chained_entries(self):
        """Subsequent entries should chain to previous hash."""
        with TemporaryDirectory() as tmp_dir:
            backend = LocalFileBackend(tmp_dir)
            log = ImmutableAuditLog(backend=backend)

            entry1 = await log.append(
                event_type="event1",
                actor="user@test.com",
                resource_type="doc",
                resource_id="doc-1",
                action="create",
            )

            entry2 = await log.append(
                event_type="event2",
                actor="user@test.com",
                resource_type="doc",
                resource_id="doc-2",
                action="create",
            )

            assert entry2.sequence_number == 2
            assert entry2.previous_hash == entry1.entry_hash

    @pytest.mark.asyncio
    async def test_immutable_log_verify_empty_log(self):
        """Verification of empty log should return valid with warning."""
        with TemporaryDirectory() as tmp_dir:
            backend = LocalFileBackend(tmp_dir)
            log = ImmutableAuditLog(backend=backend)

            result = await log.verify_integrity()

            assert result.is_valid
            assert result.entries_checked == 0
            assert any("No entries" in w for w in result.warnings)
