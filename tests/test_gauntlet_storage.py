"""
Tests for GauntletStorage - SQLite persistence for gauntlet results.

Tests cover:
- Basic CRUD operations (save, get, delete)
- Listing and pagination
- History tracking by input hash
- Result comparison
- Multi-tenancy (org_id filtering)
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

from aragora.gauntlet.storage import GauntletStorage, GauntletMetadata


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def storage(tmp_path):
    """Create storage with temp database."""
    db_path = tmp_path / "test_gauntlet.db"
    return GauntletStorage(str(db_path))


@pytest.fixture
def mock_result():
    """Create mock GauntletResult for testing."""
    result = Mock()
    result.gauntlet_id = "gauntlet-test-123"
    result.input_hash = "abc123hash"
    result.input_summary = "Test input content for gauntlet validation"
    result.verdict = Mock(value="pass")
    result.confidence = 0.85
    result.robustness_score = 0.9
    result.risk_summary = Mock(critical=0, high=1, medium=2, low=3)
    result.vulnerabilities = [Mock() for _ in range(6)]
    result.agents_used = ["claude", "gpt-4"]
    result.template_used = "security"
    result.duration_seconds = 45.5
    result.to_dict = Mock(return_value={
        "gauntlet_id": "gauntlet-test-123",
        "verdict": "pass",
        "confidence": 0.85,
        "robustness_score": 0.9,
        "risk_summary": {"critical": 0, "high": 1, "medium": 2, "low": 3},
    })
    return result


@pytest.fixture
def mock_result_fail():
    """Create mock failed result."""
    result = Mock()
    result.gauntlet_id = "gauntlet-test-456"
    result.input_hash = "def456hash"
    result.input_summary = "Another test input"
    result.verdict = Mock(value="fail")
    result.confidence = 0.95
    result.robustness_score = 0.3
    result.risk_summary = Mock(critical=2, high=3, medium=1, low=0)
    result.vulnerabilities = [Mock() for _ in range(6)]
    result.agents_used = ["claude"]
    result.template_used = None
    result.duration_seconds = 30.0
    result.to_dict = Mock(return_value={
        "gauntlet_id": "gauntlet-test-456",
        "verdict": "fail",
        "confidence": 0.95,
        "risk_summary": {"critical": 2, "high": 3, "medium": 1, "low": 0},
    })
    return result


# ============================================================================
# Basic CRUD Tests
# ============================================================================


class TestBasicOperations:
    """Tests for basic save/get/delete operations."""

    def test_save_and_get(self, storage, mock_result):
        """Test saving and retrieving a result."""
        gauntlet_id = storage.save(mock_result)

        assert gauntlet_id == "gauntlet-test-123"

        retrieved = storage.get(gauntlet_id)
        assert retrieved is not None
        assert retrieved["gauntlet_id"] == "gauntlet-test-123"
        assert retrieved["verdict"] == "pass"

    def test_get_nonexistent(self, storage):
        """Test getting a non-existent result."""
        result = storage.get("gauntlet-nonexistent")
        assert result is None

    def test_delete(self, storage, mock_result):
        """Test deleting a result."""
        storage.save(mock_result)

        deleted = storage.delete("gauntlet-test-123")
        assert deleted is True

        result = storage.get("gauntlet-test-123")
        assert result is None

    def test_delete_nonexistent(self, storage):
        """Test deleting a non-existent result."""
        deleted = storage.delete("gauntlet-nonexistent")
        assert deleted is False

    def test_save_updates_existing(self, storage, mock_result):
        """Test that save updates existing results."""
        storage.save(mock_result)

        # Modify and save again
        mock_result.confidence = 0.99
        mock_result.to_dict = Mock(return_value={
            "gauntlet_id": "gauntlet-test-123",
            "verdict": "pass",
            "confidence": 0.99,
        })
        storage.save(mock_result)

        retrieved = storage.get("gauntlet-test-123")
        assert retrieved["confidence"] == 0.99


# ============================================================================
# Listing and Pagination Tests
# ============================================================================


class TestListing:
    """Tests for listing and pagination."""

    def test_list_recent_empty(self, storage):
        """Test listing when no results exist."""
        results = storage.list_recent()
        assert results == []

    def test_list_recent_with_results(self, storage, mock_result, mock_result_fail):
        """Test listing recent results."""
        storage.save(mock_result)
        storage.save(mock_result_fail)

        results = storage.list_recent(limit=10)
        assert len(results) == 2
        assert all(isinstance(r, GauntletMetadata) for r in results)

    def test_list_recent_with_limit(self, storage, mock_result, mock_result_fail):
        """Test listing with limit."""
        storage.save(mock_result)
        storage.save(mock_result_fail)

        results = storage.list_recent(limit=1)
        assert len(results) == 1

    def test_list_recent_with_offset(self, storage, mock_result, mock_result_fail):
        """Test listing with offset (pagination)."""
        storage.save(mock_result)
        storage.save(mock_result_fail)

        results_page1 = storage.list_recent(limit=1, offset=0)
        results_page2 = storage.list_recent(limit=1, offset=1)

        assert len(results_page1) == 1
        assert len(results_page2) == 1
        assert results_page1[0].gauntlet_id != results_page2[0].gauntlet_id

    def test_list_recent_with_verdict_filter(self, storage, mock_result, mock_result_fail):
        """Test filtering by verdict."""
        storage.save(mock_result)
        storage.save(mock_result_fail)

        pass_results = storage.list_recent(verdict="pass")
        fail_results = storage.list_recent(verdict="fail")

        assert len(pass_results) == 1
        assert pass_results[0].verdict == "pass"

        assert len(fail_results) == 1
        assert fail_results[0].verdict == "fail"

    def test_list_recent_with_min_severity(self, storage, mock_result, mock_result_fail):
        """Test filtering by minimum severity."""
        storage.save(mock_result)
        storage.save(mock_result_fail)

        # mock_result_fail has critical=2
        critical_results = storage.list_recent(min_severity="critical")
        assert len(critical_results) == 1
        assert critical_results[0].gauntlet_id == "gauntlet-test-456"


# ============================================================================
# History Tests
# ============================================================================


class TestHistory:
    """Tests for input hash history tracking."""

    def test_get_history_empty(self, storage):
        """Test getting history for unknown input hash."""
        history = storage.get_history("unknown-hash")
        assert history == []

    def test_get_history(self, storage, mock_result):
        """Test getting history for an input hash."""
        # Save multiple results with same input hash
        storage.save(mock_result)

        mock_result.gauntlet_id = "gauntlet-test-124"
        mock_result.to_dict = Mock(return_value={
            "gauntlet_id": "gauntlet-test-124",
            "verdict": "pass",
        })
        storage.save(mock_result)

        history = storage.get_history("abc123hash")
        assert len(history) == 2

    def test_get_history_with_limit(self, storage, mock_result):
        """Test history with limit."""
        storage.save(mock_result)

        mock_result.gauntlet_id = "gauntlet-test-124"
        storage.save(mock_result)

        history = storage.get_history("abc123hash", limit=1)
        assert len(history) == 1


# ============================================================================
# Comparison Tests
# ============================================================================


class TestComparison:
    """Tests for comparing two results."""

    def test_compare_results(self, storage, mock_result, mock_result_fail):
        """Test comparing two results."""
        storage.save(mock_result)
        storage.save(mock_result_fail)

        comparison = storage.compare(
            "gauntlet-test-123",
            "gauntlet-test-456"
        )

        assert comparison is not None
        assert comparison["result1_id"] == "gauntlet-test-123"
        assert comparison["result2_id"] == "gauntlet-test-456"
        assert comparison["verdict_changed"] is True
        assert "deltas" in comparison

    def test_compare_nonexistent(self, storage, mock_result):
        """Test comparing with non-existent result."""
        storage.save(mock_result)

        comparison = storage.compare(
            "gauntlet-test-123",
            "gauntlet-nonexistent"
        )
        assert comparison is None

    def test_compare_improvement(self, storage, mock_result, mock_result_fail):
        """Test comparison shows improvement correctly."""
        storage.save(mock_result)  # Pass result (better)
        storage.save(mock_result_fail)  # Fail result (worse)

        # Compare: mock_result (pass) vs mock_result_fail (fail)
        comparison = storage.compare(
            "gauntlet-test-123",
            "gauntlet-test-456"
        )

        assert comparison is not None
        # mock_result has fewer critical issues
        assert comparison["deltas"]["critical"] > 0  # 2 - 0 = 2 (reduction is positive)


# ============================================================================
# Multi-tenancy Tests
# ============================================================================


class TestMultiTenancy:
    """Tests for org_id filtering."""

    def test_save_with_org_id(self, storage, mock_result):
        """Test saving with org_id."""
        storage.save(mock_result, org_id="org-123")

        # Can retrieve with matching org_id
        result = storage.get("gauntlet-test-123", org_id="org-123")
        assert result is not None

        # Cannot retrieve with different org_id
        result = storage.get("gauntlet-test-123", org_id="org-456")
        assert result is None

    def test_list_with_org_id(self, storage, mock_result, mock_result_fail):
        """Test listing filtered by org_id."""
        storage.save(mock_result, org_id="org-123")
        storage.save(mock_result_fail, org_id="org-456")

        org123_results = storage.list_recent(org_id="org-123")
        org456_results = storage.list_recent(org_id="org-456")

        assert len(org123_results) == 1
        assert len(org456_results) == 1
        assert org123_results[0].gauntlet_id == "gauntlet-test-123"
        assert org456_results[0].gauntlet_id == "gauntlet-test-456"

    def test_delete_with_org_id(self, storage, mock_result):
        """Test delete respects org_id."""
        storage.save(mock_result, org_id="org-123")

        # Cannot delete with wrong org_id
        deleted = storage.delete("gauntlet-test-123", org_id="org-456")
        assert deleted is False

        # Can delete with correct org_id
        deleted = storage.delete("gauntlet-test-123", org_id="org-123")
        assert deleted is True

    def test_count_with_org_id(self, storage, mock_result, mock_result_fail):
        """Test counting with org_id filter."""
        storage.save(mock_result, org_id="org-123")
        storage.save(mock_result_fail, org_id="org-456")

        total = storage.count()
        org123_count = storage.count(org_id="org-123")

        assert total == 2
        assert org123_count == 1


# ============================================================================
# Utility Tests
# ============================================================================


class TestUtilities:
    """Tests for utility methods."""

    def test_count(self, storage, mock_result, mock_result_fail):
        """Test counting results."""
        assert storage.count() == 0

        storage.save(mock_result)
        assert storage.count() == 1

        storage.save(mock_result_fail)
        assert storage.count() == 2

    def test_count_with_verdict_filter(self, storage, mock_result, mock_result_fail):
        """Test counting with verdict filter."""
        storage.save(mock_result)
        storage.save(mock_result_fail)

        pass_count = storage.count(verdict="pass")
        fail_count = storage.count(verdict="fail")

        assert pass_count == 1
        assert fail_count == 1


# ============================================================================
# Database Schema Tests
# ============================================================================


class TestSchema:
    """Tests for database schema initialization."""

    def test_creates_database_file(self, tmp_path):
        """Test that database file is created."""
        db_path = tmp_path / "new_gauntlet.db"
        storage = GauntletStorage(str(db_path))

        assert db_path.exists()

    def test_multiple_storage_instances_same_db(self, tmp_path):
        """Test multiple instances can share the same database."""
        db_path = tmp_path / "shared_gauntlet.db"

        storage1 = GauntletStorage(str(db_path))
        storage2 = GauntletStorage(str(db_path))

        mock_result = Mock()
        mock_result.gauntlet_id = "test-shared"
        mock_result.input_hash = "hash123"
        mock_result.input_summary = "Test input summary"
        mock_result.verdict = Mock(value="pass")
        mock_result.confidence = 0.9
        mock_result.robustness_score = 0.8
        mock_result.risk_summary = Mock(critical=0, high=0, medium=0, low=0)
        mock_result.vulnerabilities = []
        mock_result.agents_used = []
        mock_result.template_used = None
        mock_result.duration_seconds = 10.0
        mock_result.to_dict = Mock(return_value={"gauntlet_id": "test-shared"})

        storage1.save(mock_result)

        # Second instance should see the result
        result = storage2.get("test-shared")
        assert result is not None
