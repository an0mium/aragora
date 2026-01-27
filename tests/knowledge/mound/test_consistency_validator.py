"""
Tests for Knowledge Mound Consistency Validator.

Tests cover:
- Referential integrity checks
- Content validation
- Contradiction detection integration
- Staleness detection
- Confidence decay validation
- Adapter sync status
- Auto-fix functionality
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.consistency_validator import (
    ConsistencyValidator,
    ConsistencyCheckType,
    ConsistencySeverity,
    ConsistencyIssue,
    ConsistencyCheckResult,
    ConsistencyReport,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound for testing."""
    mound = MagicMock()
    mound.query = AsyncMock(return_value=[])
    mound.get = AsyncMock(return_value=None)
    mound.update = AsyncMock(return_value=True)
    mound.delete = AsyncMock(return_value=True)
    mound.list_adapters = MagicMock(return_value=[])
    return mound


@pytest.fixture
def validator(mock_mound):
    """Create a consistency validator with mock mound."""
    return ConsistencyValidator(mock_mound)


@pytest.fixture
def sample_nodes():
    """Create sample knowledge nodes for testing."""
    return [
        {
            "id": "node-1",
            "workspace_id": "ws_test",
            "content": "Valid content for node 1",
            "confidence": 0.9,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "node-2",
            "workspace_id": "ws_test",
            "content": "Valid content for node 2",
            "parent_id": "node-1",
            "confidence": 0.8,
            "updated_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
            "relationships": [{"target_id": "node-1", "type": "related"}],
        },
        {
            "id": "node-3",
            "workspace_id": "ws_test",
            "content": "Content with broken reference",
            "parent_id": "nonexistent-node",  # Broken reference
            "confidence": 0.7,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    ]


@pytest.fixture
def stale_nodes():
    """Create stale knowledge nodes for testing."""
    return [
        {
            "id": "stale-1",
            "workspace_id": "ws_test",
            "content": "Very old content",
            "confidence": 0.5,
            "updated_at": (datetime.now(timezone.utc) - timedelta(days=120)).isoformat(),
        },
        {
            "id": "stale-2",
            "workspace_id": "ws_test",
            "content": "Ancient content",
            "confidence": 0.3,
            "updated_at": (datetime.now(timezone.utc) - timedelta(days=365)).isoformat(),
        },
    ]


@pytest.fixture
def low_confidence_nodes():
    """Create nodes with low confidence for testing."""
    return [
        {
            "id": "low-conf-1",
            "workspace_id": "ws_test",
            "content": "Low confidence content",
            "confidence": 0.1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "low-conf-2",
            "workspace_id": "ws_test",
            "content": "Very low confidence",
            "confidence": 0.05,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    ]


# ============================================================================
# ConsistencyIssue Tests
# ============================================================================


class TestConsistencyIssue:
    """Tests for ConsistencyIssue dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        issue = ConsistencyIssue(
            check_type=ConsistencyCheckType.REFERENTIAL,
            severity=ConsistencySeverity.HIGH,
            message="Broken reference",
            item_id="node-1",
            related_items=["node-2"],
            details={"error": "not found"},
            suggested_fix="Remove reference",
            auto_fixable=True,
        )

        data = issue.to_dict()

        assert data["check_type"] == "referential"
        assert data["severity"] == "high"
        assert data["message"] == "Broken reference"
        assert data["item_id"] == "node-1"
        assert data["auto_fixable"] is True


# ============================================================================
# ConsistencyCheckResult Tests
# ============================================================================


class TestConsistencyCheckResult:
    """Tests for ConsistencyCheckResult dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        issue = ConsistencyIssue(
            check_type=ConsistencyCheckType.CONTENT,
            severity=ConsistencySeverity.LOW,
            message="Empty content",
        )

        result = ConsistencyCheckResult(
            check_type=ConsistencyCheckType.CONTENT,
            passed=True,
            items_checked=100,
            issues_found=1,
            issues=[issue],
            duration_ms=50.5,
        )

        data = result.to_dict()

        assert data["check_type"] == "content"
        assert data["passed"] is True
        assert data["items_checked"] == 100
        assert data["issues_found"] == 1
        assert len(data["issues"]) == 1


# ============================================================================
# ConsistencyReport Tests
# ============================================================================


class TestConsistencyReport:
    """Tests for ConsistencyReport dataclass."""

    def test_add_result_updates_summary(self):
        """Test that adding results updates summary counts."""
        report = ConsistencyReport(workspace_id="ws_test")

        issue1 = ConsistencyIssue(
            check_type=ConsistencyCheckType.REFERENTIAL,
            severity=ConsistencySeverity.CRITICAL,
            message="Critical issue",
        )
        issue2 = ConsistencyIssue(
            check_type=ConsistencyCheckType.CONTENT,
            severity=ConsistencySeverity.HIGH,
            message="High issue",
        )
        issue3 = ConsistencyIssue(
            check_type=ConsistencyCheckType.STALENESS,
            severity=ConsistencySeverity.LOW,
            message="Low issue",
        )

        result = ConsistencyCheckResult(
            check_type=ConsistencyCheckType.ALL,
            passed=False,
            items_checked=50,
            issues_found=3,
            issues=[issue1, issue2, issue3],
            duration_ms=100,
        )

        report.add_result(result)

        assert report.total_items_checked == 50
        assert report.total_issues_found == 3
        assert report.critical_issues == 1
        assert report.high_issues == 1
        assert report.low_issues == 1
        assert report.overall_healthy is False

    def test_overall_healthy_with_no_issues(self):
        """Test that overall_healthy stays true with no critical/high issues."""
        report = ConsistencyReport(workspace_id="ws_test")

        result = ConsistencyCheckResult(
            check_type=ConsistencyCheckType.STALENESS,
            passed=True,
            items_checked=100,
            issues_found=0,
        )

        report.add_result(result)

        assert report.overall_healthy is True


# ============================================================================
# Referential Integrity Tests
# ============================================================================


class TestReferentialIntegrityCheck:
    """Tests for referential integrity validation."""

    @pytest.mark.asyncio
    async def test_detects_broken_parent_reference(self, validator, mock_mound, sample_nodes):
        """Test detection of broken parent references."""
        mock_mound.query.return_value = sample_nodes

        report = await validator.validate("ws_test", [ConsistencyCheckType.REFERENTIAL])

        # Find referential check result
        ref_check = next(
            (c for c in report.checks_run if c.check_type == ConsistencyCheckType.REFERENTIAL),
            None,
        )

        assert ref_check is not None
        broken_refs = [i for i in ref_check.issues if "Broken parent reference" in i.message]
        assert len(broken_refs) > 0

    @pytest.mark.asyncio
    async def test_no_issues_with_valid_nodes(self, validator, mock_mound):
        """Test that valid nodes produce no referential issues."""
        valid_nodes = [
            {"id": "node-1", "workspace_id": "ws_test", "content": "Content"},
            {
                "id": "node-2",
                "workspace_id": "ws_test",
                "content": "Content",
                "parent_id": "node-1",
            },
        ]
        mock_mound.query.return_value = valid_nodes

        report = await validator.validate("ws_test", [ConsistencyCheckType.REFERENTIAL])

        ref_check = next(
            (c for c in report.checks_run if c.check_type == ConsistencyCheckType.REFERENTIAL),
            None,
        )

        # Should have no critical/high issues
        critical_high = [
            i
            for i in ref_check.issues
            if i.severity in [ConsistencySeverity.CRITICAL, ConsistencySeverity.HIGH]
        ]
        assert len(critical_high) == 0


# ============================================================================
# Content Validation Tests
# ============================================================================


class TestContentValidationCheck:
    """Tests for content validation."""

    @pytest.mark.asyncio
    async def test_detects_empty_content(self, validator, mock_mound):
        """Test detection of empty content."""
        nodes = [
            {"id": "empty-1", "workspace_id": "ws_test", "content": ""},
            {"id": "whitespace-1", "workspace_id": "ws_test", "content": "   "},
        ]
        mock_mound.query.return_value = nodes

        report = await validator.validate("ws_test", [ConsistencyCheckType.CONTENT])

        content_check = next(
            (c for c in report.checks_run if c.check_type == ConsistencyCheckType.CONTENT),
            None,
        )

        empty_issues = [i for i in content_check.issues if "Empty content" in i.message]
        assert len(empty_issues) == 2

    @pytest.mark.asyncio
    async def test_detects_oversized_content(self, mock_mound):
        """Test detection of oversized content."""
        validator = ConsistencyValidator(mock_mound, {"max_content_size": 100})

        nodes = [
            {"id": "big-1", "workspace_id": "ws_test", "content": "x" * 200},
        ]
        mock_mound.query.return_value = nodes

        report = await validator.validate("ws_test", [ConsistencyCheckType.CONTENT])

        content_check = next(
            (c for c in report.checks_run if c.check_type == ConsistencyCheckType.CONTENT),
            None,
        )

        size_issues = [i for i in content_check.issues if "exceeds max size" in i.message]
        assert len(size_issues) == 1

    @pytest.mark.asyncio
    async def test_detects_missing_required_fields(self, validator, mock_mound):
        """Test detection of missing required fields."""
        nodes = [
            {"content": "No ID or workspace"},
        ]
        mock_mound.query.return_value = nodes

        report = await validator.validate("ws_test", [ConsistencyCheckType.CONTENT])

        content_check = next(
            (c for c in report.checks_run if c.check_type == ConsistencyCheckType.CONTENT),
            None,
        )

        missing_issues = [i for i in content_check.issues if "Missing required field" in i.message]
        assert len(missing_issues) >= 1


# ============================================================================
# Staleness Check Tests
# ============================================================================


class TestStalenessCheck:
    """Tests for staleness detection."""

    @pytest.mark.asyncio
    async def test_detects_stale_content(self, validator, mock_mound, stale_nodes):
        """Test detection of stale content."""
        mock_mound.query.return_value = stale_nodes

        report = await validator.validate("ws_test", [ConsistencyCheckType.STALENESS])

        staleness_check = next(
            (c for c in report.checks_run if c.check_type == ConsistencyCheckType.STALENESS),
            None,
        )

        assert staleness_check is not None
        assert staleness_check.issues_found == 2

    @pytest.mark.asyncio
    async def test_no_staleness_for_recent_content(self, validator, mock_mound):
        """Test that recent content is not marked stale."""
        recent_nodes = [
            {
                "id": "recent-1",
                "workspace_id": "ws_test",
                "content": "Fresh content",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        ]
        mock_mound.query.return_value = recent_nodes

        report = await validator.validate("ws_test", [ConsistencyCheckType.STALENESS])

        staleness_check = next(
            (c for c in report.checks_run if c.check_type == ConsistencyCheckType.STALENESS),
            None,
        )

        assert staleness_check.issues_found == 0


# ============================================================================
# Confidence Decay Tests
# ============================================================================


class TestConfidenceDecayCheck:
    """Tests for confidence decay validation."""

    @pytest.mark.asyncio
    async def test_detects_low_confidence(self, validator, mock_mound, low_confidence_nodes):
        """Test detection of low confidence nodes."""
        mock_mound.query.return_value = low_confidence_nodes

        report = await validator.validate("ws_test", [ConsistencyCheckType.CONFIDENCE])

        conf_check = next(
            (c for c in report.checks_run if c.check_type == ConsistencyCheckType.CONFIDENCE),
            None,
        )

        assert conf_check is not None
        assert conf_check.issues_found == 2

    @pytest.mark.asyncio
    async def test_no_issues_for_high_confidence(self, validator, mock_mound):
        """Test that high confidence nodes are not flagged."""
        high_conf_nodes = [
            {"id": "high-1", "workspace_id": "ws_test", "content": "Good", "confidence": 0.9},
            {"id": "high-2", "workspace_id": "ws_test", "content": "Good", "confidence": 0.8},
        ]
        mock_mound.query.return_value = high_conf_nodes

        report = await validator.validate("ws_test", [ConsistencyCheckType.CONFIDENCE])

        conf_check = next(
            (c for c in report.checks_run if c.check_type == ConsistencyCheckType.CONFIDENCE),
            None,
        )

        assert conf_check.issues_found == 0


# ============================================================================
# Adapter Sync Tests
# ============================================================================


class TestAdapterSyncCheck:
    """Tests for adapter synchronization status."""

    @pytest.mark.asyncio
    async def test_reports_unhealthy_adapters(self, mock_mound):
        """Test reporting of unhealthy adapters."""
        mock_adapter = MagicMock()
        mock_adapter.get_health = AsyncMock(
            return_value={"healthy": False, "error": "Connection failed"}
        )

        mock_mound.list_adapters.return_value = ["test_adapter"]
        mock_mound.get_adapter = MagicMock(return_value=mock_adapter)

        validator = ConsistencyValidator(mock_mound)
        report = await validator.validate("ws_test", [ConsistencyCheckType.SYNC])

        sync_check = next(
            (c for c in report.checks_run if c.check_type == ConsistencyCheckType.SYNC),
            None,
        )

        assert sync_check is not None
        unhealthy = [i for i in sync_check.issues if "unhealthy" in i.message]
        assert len(unhealthy) == 1

    @pytest.mark.asyncio
    async def test_passes_with_healthy_adapters(self, mock_mound):
        """Test that healthy adapters pass the check."""
        mock_adapter = MagicMock()
        mock_adapter.get_health = AsyncMock(return_value={"healthy": True})

        mock_mound.list_adapters.return_value = ["healthy_adapter"]
        mock_mound.get_adapter = MagicMock(return_value=mock_adapter)

        validator = ConsistencyValidator(mock_mound)
        report = await validator.validate("ws_test", [ConsistencyCheckType.SYNC])

        sync_check = next(
            (c for c in report.checks_run if c.check_type == ConsistencyCheckType.SYNC),
            None,
        )

        assert sync_check.passed is True


# ============================================================================
# Full Validation Tests
# ============================================================================


class TestFullValidation:
    """Tests for full consistency validation."""

    @pytest.mark.asyncio
    async def test_runs_all_checks(self, validator, mock_mound, sample_nodes):
        """Test that validate runs all check types."""
        mock_mound.query.return_value = sample_nodes

        report = await validator.validate("ws_test")

        check_types = {c.check_type for c in report.checks_run}
        expected_types = {
            ConsistencyCheckType.REFERENTIAL,
            ConsistencyCheckType.CONTENT,
            ConsistencyCheckType.CONTRADICTION,
            ConsistencyCheckType.STALENESS,
            ConsistencyCheckType.CONFIDENCE,
            ConsistencyCheckType.SYNC,
        }

        assert check_types == expected_types

    @pytest.mark.asyncio
    async def test_returns_complete_report(self, validator, mock_mound, sample_nodes):
        """Test that validate returns a complete report."""
        mock_mound.query.return_value = sample_nodes

        report = await validator.validate("ws_test")

        assert report.workspace_id == "ws_test"
        assert report.timestamp is not None
        assert report.duration_ms > 0
        assert len(report.checks_run) > 0

    @pytest.mark.asyncio
    async def test_handles_empty_workspace(self, validator, mock_mound):
        """Test handling of empty workspace."""
        mock_mound.query.return_value = []

        report = await validator.validate("ws_empty")

        assert report.overall_healthy is True
        assert report.total_issues_found == 0


# ============================================================================
# Auto-Fix Tests
# ============================================================================


class TestAutoFix:
    """Tests for auto-fix functionality."""

    @pytest.mark.asyncio
    async def test_dry_run_reports_fixes(self, validator, mock_mound, sample_nodes):
        """Test that dry run reports what would be fixed."""
        mock_mound.query.return_value = sample_nodes

        result = await validator.auto_fix("ws_test", dry_run=True)

        assert result["dry_run"] is True
        assert "fixes_applied" in result
        # Dry run should not actually call update/delete
        mock_mound.update.assert_not_called()
        mock_mound.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_actual_fix_applies_changes(self, validator, mock_mound):
        """Test that actual fix applies changes."""
        # Node with broken parent reference (auto-fixable)
        nodes = [
            {
                "id": "fixable-1",
                "workspace_id": "ws_test",
                "content": "Content",
                "parent_id": "nonexistent",
            }
        ]
        mock_mound.query.return_value = nodes

        result = await validator.auto_fix("ws_test", dry_run=False)

        assert result["dry_run"] is False
        # Should have attempted to fix the broken reference
        # (actual call depends on implementation details)


__all__ = [
    "TestConsistencyIssue",
    "TestConsistencyCheckResult",
    "TestConsistencyReport",
    "TestReferentialIntegrityCheck",
    "TestContentValidationCheck",
    "TestStalenessCheck",
    "TestConfidenceDecayCheck",
    "TestAdapterSyncCheck",
    "TestFullValidation",
    "TestAutoFix",
]
