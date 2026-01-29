"""
Tests for SLO History Persistence.

Validates that SLO violations are correctly stored, queried, and cleaned up.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aragora.observability.slo_history import (
    SLOHistoryStore,
    SLOViolationRecord,
    reset_slo_history_store,
    slo_history_callback,
)


@pytest.fixture
def store(tmp_path: Path) -> SLOHistoryStore:
    """Create a fresh SLO history store with a temporary database."""
    return SLOHistoryStore(db_path=str(tmp_path / "test_slo.db"))


@pytest.fixture(autouse=True)
def reset_global():
    """Reset global store between tests."""
    reset_slo_history_store()
    yield
    reset_slo_history_store()


class TestSLOHistoryStore:
    """Test SLOHistoryStore CRUD operations."""

    def test_record_and_query(self, store: SLOHistoryStore):
        """Should record a violation and retrieve it."""
        row_id = store.record_violation(
            slo_name="availability",
            severity="critical",
            current_value=0.995,
            target_value=0.999,
            error_budget_remaining=0.1,
            burn_rate=2.5,
            message="Availability dropped below SLO target",
        )

        assert row_id > 0

        records = store.query(slo_name="availability")
        assert len(records) == 1
        assert records[0].slo_name == "availability"
        assert records[0].severity == "critical"
        assert records[0].current_value == 0.995
        assert records[0].target_value == 0.999

    def test_query_by_severity(self, store: SLOHistoryStore):
        """Should filter by severity."""
        store.record_violation(
            slo_name="latency",
            severity="warning",
            current_value=450,
            target_value=500,
            error_budget_remaining=0.5,
            burn_rate=1.2,
            message="Latency approaching SLO",
        )
        store.record_violation(
            slo_name="latency",
            severity="critical",
            current_value=600,
            target_value=500,
            error_budget_remaining=0.0,
            burn_rate=3.0,
            message="Latency exceeded SLO",
        )

        warnings = store.query(severity="warning")
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"

        critical = store.query(severity="critical")
        assert len(critical) == 1
        assert critical[0].current_value == 600

    def test_query_by_time_range(self, store: SLOHistoryStore):
        """Should filter by time range."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=48)

        store.record_violation(
            slo_name="availability",
            severity="critical",
            current_value=0.99,
            target_value=0.999,
            error_budget_remaining=0.0,
            burn_rate=5.0,
            message="Old violation",
            timestamp=old,
        )
        store.record_violation(
            slo_name="availability",
            severity="warning",
            current_value=0.998,
            target_value=0.999,
            error_budget_remaining=0.3,
            burn_rate=1.5,
            message="Recent violation",
            timestamp=now,
        )

        recent = store.query(hours=24)
        assert len(recent) == 1
        assert recent[0].message == "Recent violation"

        all_records = store.query()
        assert len(all_records) == 2

    def test_query_limit(self, store: SLOHistoryStore):
        """Should respect limit parameter."""
        for i in range(10):
            store.record_violation(
                slo_name="debate_success",
                severity="warning",
                current_value=0.9 + i * 0.001,
                target_value=0.95,
                error_budget_remaining=0.5,
                burn_rate=1.0,
                message=f"Violation #{i}",
            )

        limited = store.query(limit=3)
        assert len(limited) == 3

    def test_query_returns_newest_first(self, store: SLOHistoryStore):
        """Results should be ordered by timestamp descending."""
        now = datetime.now(timezone.utc)
        for i in range(3):
            store.record_violation(
                slo_name="latency",
                severity="warning",
                current_value=float(i),
                target_value=500,
                error_budget_remaining=0.5,
                burn_rate=1.0,
                message=f"Violation {i}",
                timestamp=now - timedelta(hours=i),
            )

        records = store.query()
        assert records[0].current_value == 0.0  # Most recent
        assert records[2].current_value == 2.0  # Oldest

    def test_count(self, store: SLOHistoryStore):
        """Should count violations correctly."""
        for severity in ["warning", "warning", "critical"]:
            store.record_violation(
                slo_name="availability",
                severity=severity,
                current_value=0.99,
                target_value=0.999,
                error_budget_remaining=0.1,
                burn_rate=2.0,
                message="test",
            )

        assert store.count() == 3
        assert store.count(severity="warning") == 2
        assert store.count(severity="critical") == 1
        assert store.count(slo_name="availability") == 3
        assert store.count(slo_name="latency") == 0

    def test_cleanup(self, store: SLOHistoryStore):
        """Should remove old records based on retention period."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=100)

        store.record_violation(
            slo_name="availability",
            severity="critical",
            current_value=0.99,
            target_value=0.999,
            error_budget_remaining=0.0,
            burn_rate=5.0,
            message="Old violation",
            timestamp=old,
        )
        store.record_violation(
            slo_name="availability",
            severity="warning",
            current_value=0.998,
            target_value=0.999,
            error_budget_remaining=0.3,
            burn_rate=1.5,
            message="Recent violation",
            timestamp=now,
        )

        deleted = store.cleanup(retention_days=90)
        assert deleted == 1

        remaining = store.query()
        assert len(remaining) == 1
        assert remaining[0].message == "Recent violation"

    def test_metadata_storage(self, store: SLOHistoryStore):
        """Should store and retrieve metadata as JSON."""
        store.record_violation(
            slo_name="availability",
            severity="critical",
            current_value=0.99,
            target_value=0.999,
            error_budget_remaining=0.0,
            burn_rate=5.0,
            message="test",
            metadata={"source": "prometheus", "cluster": "prod-us-east"},
        )

        records = store.query()
        assert len(records) == 1
        d = records[0].to_dict()
        assert d["metadata"]["source"] == "prometheus"
        assert d["metadata"]["cluster"] == "prod-us-east"

    def test_get_summary(self, store: SLOHistoryStore):
        """Should return a summary of violations."""
        for name, sev in [
            ("availability", "critical"),
            ("availability", "warning"),
            ("latency", "warning"),
        ]:
            store.record_violation(
                slo_name=name,
                severity=sev,
                current_value=0.0,
                target_value=1.0,
                error_budget_remaining=0.0,
                burn_rate=1.0,
                message="test",
            )

        summary = store.get_summary(hours=24)
        assert summary["total"] == 3
        assert summary["by_slo"]["availability"] == 2
        assert summary["by_slo"]["latency"] == 1
        assert summary["by_severity"]["warning"] == 2
        assert summary["by_severity"]["critical"] == 1


class TestSLOHistoryCallback:
    """Test the SLOAlertMonitor callback integration."""

    def test_callback_persists_breach(self, tmp_path: Path):
        """slo_history_callback should persist a breach to the store."""
        from aragora.observability.slo_history import get_slo_history_store

        store = get_slo_history_store(db_path=str(tmp_path / "cb_test.db"))

        # Create a mock breach (matches SLOBreach interface)
        class MockBreach:
            slo_name = "availability"
            severity = "critical"
            current_value = 0.995
            target_value = 0.999
            error_budget_remaining = 0.1
            burn_rate = 2.5
            message = "Availability dropped"
            timestamp = datetime.now(timezone.utc)

        slo_history_callback(MockBreach())

        records = store.query()
        assert len(records) == 1
        assert records[0].slo_name == "availability"
        assert records[0].severity == "critical"

    def test_callback_handles_errors_gracefully(self, tmp_path: Path):
        """Callback should not raise even if persistence fails."""

        # Pass an object missing required fields
        class BadBreach:
            slo_name = "test"
            # Missing other fields

        # Should not raise
        slo_history_callback(BadBreach())
