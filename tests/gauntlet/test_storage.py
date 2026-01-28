"""
Tests for Gauntlet Storage.

Tests the storage module that provides:
- Persistent storage for Gauntlet results
- SQLite and PostgreSQL backends
- List, filter, and comparison operations
- Inflight run management for durability
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield str(db_path)
    db_path.unlink(missing_ok=True)


@pytest.fixture
def storage(temp_db_path):
    """Create a GauntletStorage instance for testing."""
    from aragora.gauntlet.storage import GauntletStorage

    store = GauntletStorage(db_path=temp_db_path, backend="sqlite")
    yield store
    store.close()


@pytest.fixture
def sample_result():
    """Create a sample gauntlet result for testing."""

    @dataclass
    class MockResult:
        gauntlet_id: str = "gauntlet-test-123"
        input_hash: str = "abc123"
        input_summary: str = "Test input summary"
        verdict: str = "pass"
        confidence: float = 0.95
        robustness_score: float = 0.8
        risk_summary: dict = None
        agents_used: list = None
        template_used: str = "security-review"
        duration_seconds: float = 45.5

        def __post_init__(self):
            if self.risk_summary is None:
                self.risk_summary = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            if self.agents_used is None:
                self.agents_used = ["claude", "gpt-4"]

        def to_dict(self):
            return {
                "gauntlet_id": self.gauntlet_id,
                "verdict": self.verdict,
                "confidence": self.confidence,
                "robustness_score": self.robustness_score,
                "risk_summary": self.risk_summary,
            }

    return MockResult()


# ===========================================================================
# Tests: GauntletMetadata Dataclass
# ===========================================================================


class TestGauntletMetadata:
    """Tests for GauntletMetadata dataclass."""

    def test_creation(self):
        """Test GauntletMetadata creation."""
        from aragora.gauntlet.storage import GauntletMetadata

        meta = GauntletMetadata(
            gauntlet_id="gauntlet-123",
            input_hash="hash123",
            input_summary="Test summary",
            verdict="pass",
            confidence=0.95,
            robustness_score=0.8,
            critical_count=0,
            high_count=1,
            total_findings=5,
            agents_used=["claude"],
            template_used="security",
            created_at=datetime.now(),
            duration_seconds=30.0,
        )

        assert meta.gauntlet_id == "gauntlet-123"
        assert meta.verdict == "pass"
        assert meta.critical_count == 0


# ===========================================================================
# Tests: GauntletInflightRun Dataclass
# ===========================================================================


class TestGauntletInflightRun:
    """Tests for GauntletInflightRun dataclass."""

    def test_creation(self):
        """Test GauntletInflightRun creation."""
        from aragora.gauntlet.storage import GauntletInflightRun

        now = datetime.now()
        run = GauntletInflightRun(
            gauntlet_id="gauntlet-123",
            status="running",
            input_type="spec",
            input_summary="Test spec",
            input_hash="hash123",
            persona="security",
            profile="default",
            agents=["claude", "gpt-4"],
            created_at=now,
            updated_at=now,
        )

        assert run.gauntlet_id == "gauntlet-123"
        assert run.status == "running"
        assert run.progress_percent == 0.0

    def test_to_dict(self):
        """Test GauntletInflightRun.to_dict."""
        from aragora.gauntlet.storage import GauntletInflightRun

        now = datetime.now()
        run = GauntletInflightRun(
            gauntlet_id="gauntlet-123",
            status="pending",
            input_type="spec",
            input_summary="Test",
            input_hash="hash",
            persona=None,
            profile="default",
            agents=["claude"],
            created_at=now,
            updated_at=now,
            current_phase="analysis",
            progress_percent=50.0,
        )

        result = run.to_dict()

        assert result["gauntlet_id"] == "gauntlet-123"
        assert result["status"] == "pending"
        assert result["progress_percent"] == 50.0
        assert "created_at" in result


# ===========================================================================
# Tests: GauntletStorage Initialization
# ===========================================================================


class TestGauntletStorageInit:
    """Tests for GauntletStorage initialization."""

    def test_init_sqlite_default(self, temp_db_path):
        """Test initialization with SQLite (default)."""
        from aragora.gauntlet.storage import GauntletStorage

        storage = GauntletStorage(db_path=temp_db_path)

        assert storage.backend_type == "sqlite"
        storage.close()

    def test_init_sqlite_explicit(self, temp_db_path):
        """Test initialization with explicit SQLite backend."""
        from aragora.gauntlet.storage import GauntletStorage

        storage = GauntletStorage(db_path=temp_db_path, backend="sqlite")

        assert storage.backend_type == "sqlite"
        storage.close()

    def test_init_postgresql_requires_url(self):
        """Test PostgreSQL requires DATABASE_URL."""
        from aragora.gauntlet.storage import GauntletStorage

        with pytest.raises(ValueError, match="DATABASE_URL"):
            GauntletStorage(backend="postgresql")


# ===========================================================================
# Tests: Save and Get Methods
# ===========================================================================


class TestSaveAndGet:
    """Tests for save and get methods."""

    def test_save_result(self, storage, sample_result):
        """Test saving a gauntlet result."""
        gauntlet_id = storage.save(sample_result)

        assert gauntlet_id == "gauntlet-test-123"

    def test_get_result(self, storage, sample_result):
        """Test getting a saved result."""
        storage.save(sample_result)

        result = storage.get("gauntlet-test-123")

        assert result is not None
        assert result["gauntlet_id"] == "gauntlet-test-123"

    def test_get_nonexistent(self, storage):
        """Test getting a non-existent result."""
        result = storage.get("nonexistent")

        assert result is None

    def test_get_with_org_id(self, storage, sample_result):
        """Test getting a result with org_id filter."""
        storage.save(sample_result, org_id="org-123")

        # Should find with matching org_id
        result = storage.get("gauntlet-test-123", org_id="org-123")
        assert result is not None

        # Should not find with different org_id
        result = storage.get("gauntlet-test-123", org_id="org-456")
        assert result is None

    def test_save_updates_existing(self, storage, sample_result):
        """Test save updates existing result."""
        storage.save(sample_result)

        # Modify and re-save
        sample_result.verdict = "fail"
        storage.save(sample_result)

        result = storage.get("gauntlet-test-123")

        assert result["verdict"] == "fail"

    def test_save_with_findings_lists(self, storage):
        """Test save with findings as lists."""

        @dataclass
        class ResultWithLists:
            gauntlet_id: str = "gauntlet-lists-123"
            input_hash: str = "hash"
            verdict: str = "conditional"
            confidence: float = 0.7
            robustness_score: float = 0.5
            critical_findings: list = None
            high_findings: list = None
            agents_used: list = None
            duration_seconds: float = 30.0

            def __post_init__(self):
                self.critical_findings = [{"id": "c1"}]
                self.high_findings = [{"id": "h1"}, {"id": "h2"}]
                self.agents_used = ["claude"]

            def to_dict(self):
                return {"gauntlet_id": self.gauntlet_id, "verdict": self.verdict}

        result = ResultWithLists()
        storage.save(result)

        # Verify via list_recent
        recent = storage.list_recent(limit=1)
        assert len(recent) == 1
        assert recent[0].critical_count == 1
        assert recent[0].high_count == 2


# ===========================================================================
# Tests: List and Filter Methods
# ===========================================================================


class TestListAndFilter:
    """Tests for list and filter methods."""

    def test_list_recent_empty(self, storage):
        """Test list_recent with no results."""
        recent = storage.list_recent()

        assert recent == []

    def test_list_recent_with_results(self, storage, sample_result):
        """Test list_recent with results."""
        storage.save(sample_result)

        recent = storage.list_recent()

        assert len(recent) == 1
        assert recent[0].gauntlet_id == "gauntlet-test-123"

    def test_list_recent_pagination(self, storage):
        """Test list_recent pagination."""

        @dataclass
        class SimpleResult:
            gauntlet_id: str
            input_hash: str = "hash"
            verdict: str = "pass"
            confidence: float = 0.9
            robustness_score: float = 0.8
            agents_used: list = None
            duration_seconds: float = 10.0

            def __post_init__(self):
                self.agents_used = []

            def to_dict(self):
                return {"gauntlet_id": self.gauntlet_id}

        # Save multiple results
        for i in range(5):
            storage.save(SimpleResult(gauntlet_id=f"gauntlet-{i}"))

        # Test pagination
        page1 = storage.list_recent(limit=2, offset=0)
        page2 = storage.list_recent(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2

    def test_list_recent_filter_verdict(self, storage):
        """Test list_recent with verdict filter."""

        @dataclass
        class SimpleResult:
            gauntlet_id: str
            verdict: str
            input_hash: str = "hash"
            confidence: float = 0.9
            robustness_score: float = 0.8
            agents_used: list = None
            duration_seconds: float = 10.0

            def __post_init__(self):
                self.agents_used = []

            def to_dict(self):
                return {"gauntlet_id": self.gauntlet_id, "verdict": self.verdict}

        storage.save(SimpleResult(gauntlet_id="pass-1", verdict="pass"))
        storage.save(SimpleResult(gauntlet_id="fail-1", verdict="fail"))
        storage.save(SimpleResult(gauntlet_id="pass-2", verdict="pass"))

        passing = storage.list_recent(verdict="pass")

        assert len(passing) == 2
        assert all(r.verdict == "pass" for r in passing)

    def test_list_recent_filter_org_id(self, storage):
        """Test list_recent with org_id filter."""

        @dataclass
        class SimpleResult:
            gauntlet_id: str
            input_hash: str = "hash"
            verdict: str = "pass"
            confidence: float = 0.9
            robustness_score: float = 0.8
            agents_used: list = None
            duration_seconds: float = 10.0

            def __post_init__(self):
                self.agents_used = []

            def to_dict(self):
                return {"gauntlet_id": self.gauntlet_id}

        storage.save(SimpleResult(gauntlet_id="org1-1"), org_id="org-1")
        storage.save(SimpleResult(gauntlet_id="org2-1"), org_id="org-2")

        org1_results = storage.list_recent(org_id="org-1")

        assert len(org1_results) == 1
        assert org1_results[0].gauntlet_id == "org1-1"


# ===========================================================================
# Tests: History Methods
# ===========================================================================


class TestHistory:
    """Tests for history methods."""

    def test_get_history(self, storage):
        """Test get_history for same input."""

        @dataclass
        class SimpleResult:
            gauntlet_id: str
            input_hash: str
            verdict: str = "pass"
            confidence: float = 0.9
            robustness_score: float = 0.8
            agents_used: list = None
            duration_seconds: float = 10.0

            def __post_init__(self):
                self.agents_used = []

            def to_dict(self):
                return {"gauntlet_id": self.gauntlet_id}

        # Save multiple results with same input hash
        storage.save(SimpleResult(gauntlet_id="run-1", input_hash="same-hash"))
        storage.save(SimpleResult(gauntlet_id="run-2", input_hash="same-hash"))
        storage.save(SimpleResult(gauntlet_id="run-3", input_hash="different-hash"))

        history = storage.get_history("same-hash")

        assert len(history) == 2

    def test_get_history_empty(self, storage):
        """Test get_history with no matching results."""
        history = storage.get_history("nonexistent-hash")

        assert history == []


# ===========================================================================
# Tests: Compare Method
# ===========================================================================


class TestCompare:
    """Tests for compare method."""

    def test_compare_two_results(self, storage):
        """Test comparing two results."""

        @dataclass
        class ComparableResult:
            gauntlet_id: str
            verdict: str
            confidence: float
            robustness_score: float
            input_hash: str = "hash"
            agents_used: list = None
            duration_seconds: float = 10.0
            risk_summary: dict = None

            def __post_init__(self):
                self.agents_used = []
                if self.risk_summary is None:
                    self.risk_summary = {
                        "critical": 0,
                        "high": 0,
                        "medium": 0,
                        "low": 0,
                        "total": 0,
                    }

            def to_dict(self):
                return {
                    "gauntlet_id": self.gauntlet_id,
                    "verdict": self.verdict,
                    "confidence": self.confidence,
                    "robustness_score": self.robustness_score,
                    "risk_summary": self.risk_summary,
                }

        storage.save(
            ComparableResult(
                gauntlet_id="old-run",
                verdict="fail",
                confidence=0.7,
                robustness_score=0.5,
                risk_summary={"critical": 2, "high": 3, "medium": 1, "low": 0, "total": 6},
            )
        )
        storage.save(
            ComparableResult(
                gauntlet_id="new-run",
                verdict="pass",
                confidence=0.9,
                robustness_score=0.8,
                risk_summary={"critical": 0, "high": 1, "medium": 1, "low": 0, "total": 2},
            )
        )

        comparison = storage.compare("new-run", "old-run")

        assert comparison is not None
        assert comparison["verdict_changed"] is True
        assert comparison["verdict_improved"] is True
        assert comparison["improved"] is True
        assert comparison["deltas"]["critical"] == 2  # Reduction of 2

    def test_compare_nonexistent(self, storage, sample_result):
        """Test comparing with non-existent result."""
        storage.save(sample_result)

        comparison = storage.compare("gauntlet-test-123", "nonexistent")

        assert comparison is None


# ===========================================================================
# Tests: Delete and Count Methods
# ===========================================================================


class TestDeleteAndCount:
    """Tests for delete and count methods."""

    def test_delete_result(self, storage, sample_result):
        """Test deleting a result."""
        storage.save(sample_result)

        deleted = storage.delete("gauntlet-test-123")

        assert deleted is True
        assert storage.get("gauntlet-test-123") is None

    def test_delete_nonexistent(self, storage):
        """Test deleting non-existent result."""
        deleted = storage.delete("nonexistent")

        assert deleted is False

    def test_count_all(self, storage, sample_result):
        """Test counting all results."""
        assert storage.count() == 0

        storage.save(sample_result)

        assert storage.count() == 1

    def test_count_with_filters(self, storage):
        """Test counting with filters."""

        @dataclass
        class SimpleResult:
            gauntlet_id: str
            verdict: str
            input_hash: str = "hash"
            confidence: float = 0.9
            robustness_score: float = 0.8
            agents_used: list = None
            duration_seconds: float = 10.0

            def __post_init__(self):
                self.agents_used = []

            def to_dict(self):
                return {"gauntlet_id": self.gauntlet_id, "verdict": self.verdict}

        storage.save(SimpleResult(gauntlet_id="pass-1", verdict="pass"))
        storage.save(SimpleResult(gauntlet_id="fail-1", verdict="fail"))

        assert storage.count() == 2
        assert storage.count(verdict="pass") == 1
        assert storage.count(verdict="fail") == 1


# ===========================================================================
# Tests: Inflight Run Management
# ===========================================================================


class TestInflightRuns:
    """Tests for inflight run management."""

    def test_save_inflight(self, storage):
        """Test saving an inflight run."""
        gauntlet_id = storage.save_inflight(
            gauntlet_id="gauntlet-inflight-123",
            status="pending",
            input_type="spec",
            input_summary="Test spec",
            input_hash="hash123",
            persona="security",
            profile="default",
            agents=["claude", "gpt-4"],
        )

        assert gauntlet_id == "gauntlet-inflight-123"

    def test_get_inflight(self, storage):
        """Test getting an inflight run."""
        storage.save_inflight(
            gauntlet_id="gauntlet-inflight-123",
            status="running",
            input_type="spec",
            input_summary="Test",
            input_hash="hash",
            persona=None,
            profile="default",
            agents=["claude"],
        )

        run = storage.get_inflight("gauntlet-inflight-123")

        assert run is not None
        assert run.status == "running"
        assert run.agents == ["claude"]

    def test_get_inflight_nonexistent(self, storage):
        """Test getting non-existent inflight run."""
        run = storage.get_inflight("nonexistent")

        assert run is None

    def test_update_inflight_status(self, storage):
        """Test updating inflight run status."""
        storage.save_inflight(
            gauntlet_id="gauntlet-123",
            status="pending",
            input_type="spec",
            input_summary="Test",
            input_hash="hash",
            persona=None,
            profile="default",
            agents=[],
        )

        storage.update_inflight_status(
            gauntlet_id="gauntlet-123",
            status="running",
            current_phase="analysis",
            progress_percent=50.0,
        )

        run = storage.get_inflight("gauntlet-123")

        assert run.status == "running"
        assert run.current_phase == "analysis"
        assert run.progress_percent == 50.0

    def test_update_inflight_with_error(self, storage):
        """Test updating inflight run with error."""
        storage.save_inflight(
            gauntlet_id="gauntlet-123",
            status="running",
            input_type="spec",
            input_summary="Test",
            input_hash="hash",
            persona=None,
            profile="default",
            agents=[],
        )

        storage.update_inflight_status(
            gauntlet_id="gauntlet-123",
            status="failed",
            error="Connection timeout",
        )

        run = storage.get_inflight("gauntlet-123")

        assert run.status == "failed"
        assert run.error == "Connection timeout"

    def test_list_inflight(self, storage):
        """Test listing inflight runs."""
        storage.save_inflight("g1", "pending", "spec", "Test 1", "h1", None, "default", [])
        storage.save_inflight("g2", "running", "spec", "Test 2", "h2", None, "default", [])
        storage.save_inflight("g3", "completed", "spec", "Test 3", "h3", None, "default", [])

        all_runs = storage.list_inflight()
        pending = storage.list_inflight(status="pending")
        running = storage.list_inflight(status="running")

        assert len(all_runs) == 3
        assert len(pending) == 1
        assert len(running) == 1

    def test_delete_inflight(self, storage):
        """Test deleting an inflight run."""
        storage.save_inflight("gauntlet-123", "completed", "spec", "Test", "h", None, "default", [])

        storage.delete_inflight("gauntlet-123")

        run = storage.get_inflight("gauntlet-123")
        assert run is None

    def test_cleanup_completed_inflight(self, storage):
        """Test cleaning up completed inflight runs."""
        storage.save_inflight("g1", "completed", "spec", "Test 1", "h1", None, "default", [])
        storage.save_inflight("g2", "failed", "spec", "Test 2", "h2", None, "default", [])
        storage.save_inflight("g3", "running", "spec", "Test 3", "h3", None, "default", [])

        count = storage.cleanup_completed_inflight()

        assert count == 2
        assert storage.get_inflight("g1") is None
        assert storage.get_inflight("g2") is None
        assert storage.get_inflight("g3") is not None


# ===========================================================================
# Tests: Module Helper Functions
# ===========================================================================


class TestModuleHelpers:
    """Tests for module-level helper functions."""

    def test_get_storage_creates_singleton(self, temp_db_path):
        """Test get_storage creates singleton."""
        from aragora.gauntlet import storage as storage_module

        # Reset singleton
        storage_module._default_storage = None

        store1 = storage_module.get_storage(db_path=temp_db_path)
        store2 = storage_module.get_storage()

        assert store1 is store2

        store1.close()
        storage_module._default_storage = None

    def test_reset_storage(self, temp_db_path):
        """Test reset_storage clears singleton."""
        from aragora.gauntlet import storage as storage_module

        storage_module._default_storage = None
        store1 = storage_module.get_storage(db_path=temp_db_path)

        storage_module.reset_storage()

        assert storage_module._default_storage is None
