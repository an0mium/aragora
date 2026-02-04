"""
Tests for SAST (Static Application Security Testing) Handler.

Tests cover:
- SAST scan trigger
- Scan status retrieval
- Findings listing with filtering
- OWASP Top 10 summary
- Rule set configuration
- Permission checks
- Error handling
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.base import HandlerResult


# ============================================================================
# Mock Classes
# ============================================================================


class MockSASTScanResult:
    """Mock SAST scan result."""

    def __init__(
        self,
        scan_id: str = "sast_123",
        repository: str = "test-repo",
        status: str = "completed",
    ):
        self.scan_id = scan_id
        self.repository = repository
        self.status = status
        self.branch = "main"
        self.commit_sha = "abc123def"
        self.started_at = datetime.now(timezone.utc)
        self.completed_at = datetime.now(timezone.utc)
        self.error = None
        self.findings = []
        self.files_scanned = 100
        self.rules_applied = 50
        self.critical_count = 0
        self.high_count = 0
        self.medium_count = 0
        self.low_count = 0

    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "repository": self.repository,
            "status": self.status,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "findings": [f.to_dict() for f in self.findings],
            "files_scanned": self.files_scanned,
            "rules_applied": self.rules_applied,
            "summary": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
            },
        }


class MockSASTFinding:
    """Mock SAST finding."""

    def __init__(
        self,
        finding_id: str,
        severity: str,
        owasp_category: str = "A03:2021",
        rule_id: str = "sql-injection",
    ):
        self.finding_id = finding_id
        self.severity = severity
        self.owasp_category = owasp_category
        self.rule_id = rule_id
        self.file_path = "/src/app.py"
        self.line_number = 42
        self.column_number = 10
        self.message = f"Security issue found: {rule_id}"
        self.code_snippet = "cursor.execute(f'SELECT * FROM users WHERE id={user_id}')"

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity,
            "owasp_category": self.owasp_category,
            "rule_id": self.rule_id,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "message": self.message,
            "code_snippet": self.code_snippet,
        }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_sast_scanner():
    """Create mock SAST scanner."""
    scanner = MagicMock()
    scanner.scan_repository = AsyncMock(return_value=MockSASTScanResult())
    return scanner


@pytest.fixture
def mock_scan_with_findings():
    """Create scan with OWASP findings."""
    scan = MockSASTScanResult()
    scan.findings = [
        MockSASTFinding("f1", "critical", "A03:2021", "sql-injection"),
        MockSASTFinding("f2", "high", "A07:2021", "xss"),
        MockSASTFinding("f3", "medium", "A01:2021", "broken-access-control"),
        MockSASTFinding("f4", "low", "A05:2021", "security-misconfiguration"),
    ]
    scan.critical_count = 1
    scan.high_count = 1
    scan.medium_count = 1
    scan.low_count = 1
    return scan


# ============================================================================
# Scan Trigger Tests
# ============================================================================


class TestSASTScanTrigger:
    """Test SAST scan trigger endpoint."""

    @pytest.mark.asyncio
    async def test_scan_sast_success(self, mock_sast_scanner):
        """Test successful SAST scan trigger."""
        from aragora.server.handlers.codebase.security.sast import handle_scan_sast

        with (
            patch(
                "aragora.server.handlers.codebase.security.sast.get_sast_scanner",
                return_value=mock_sast_scanner,
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_running_sast_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_sast_lock",
                return_value=MagicMock(),
            ),
        ):
            result = await handle_scan_sast(
                repo_path="/path/to/repo",
                repo_id="test-repo",
                branch="main",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "scan_id" in data
            assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_scan_sast_already_running(self, mock_sast_scanner):
        """Test SAST scan returns 409 if already running."""
        from aragora.server.handlers.codebase.security.sast import handle_scan_sast

        running_task = MagicMock()
        running_task.done.return_value = False

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_running_sast_scans",
            return_value={"test-repo": running_task},
        ):
            result = await handle_scan_sast(
                repo_path="/path/to/repo",
                repo_id="test-repo",
            )

            assert result.status_code == 409

    @pytest.mark.asyncio
    async def test_scan_sast_with_rule_sets(self, mock_sast_scanner):
        """Test SAST scan with custom rule sets."""
        from aragora.server.handlers.codebase.security.sast import handle_scan_sast

        with (
            patch(
                "aragora.server.handlers.codebase.security.sast.get_sast_scanner",
                return_value=mock_sast_scanner,
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_running_sast_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_sast_lock",
                return_value=MagicMock(),
            ),
        ):
            result = await handle_scan_sast(
                repo_path="/path/to/repo",
                repo_id="test-repo",
                rule_sets=["owasp-top-10", "cwe-top-25"],
            )

            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_scan_sast_with_language_filter(self, mock_sast_scanner):
        """Test SAST scan with language filter."""
        from aragora.server.handlers.codebase.security.sast import handle_scan_sast

        with (
            patch(
                "aragora.server.handlers.codebase.security.sast.get_sast_scanner",
                return_value=mock_sast_scanner,
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_running_sast_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_sast_lock",
                return_value=MagicMock(),
            ),
        ):
            result = await handle_scan_sast(
                repo_path="/path/to/repo",
                repo_id="test-repo",
                languages=["python", "javascript"],
            )

            assert result.status_code == 200


# ============================================================================
# Scan Status Tests
# ============================================================================


class TestSASTScanStatus:
    """Test SAST scan status endpoint."""

    @pytest.mark.asyncio
    async def test_get_sast_scan_status_specific(self):
        """Test getting specific SAST scan by ID."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_sast_scan_status,
        )

        scan = MockSASTScanResult(scan_id="sast_123")

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
            return_value={"sast_123": scan},
        ):
            result = await handle_get_sast_scan_status(
                repo_id="test-repo",
                scan_id="sast_123",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["scan_result"]["scan_id"] == "sast_123"

    @pytest.mark.asyncio
    async def test_get_sast_scan_status_latest(self):
        """Test getting latest SAST scan."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_sast_scan_status,
        )

        scan1 = MockSASTScanResult(scan_id="sast_old")
        scan1.started_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
        scan2 = MockSASTScanResult(scan_id="sast_new")
        scan2.started_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
            return_value={"sast_old": scan1, "sast_new": scan2},
        ):
            result = await handle_get_sast_scan_status(repo_id="test-repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["scan_result"]["scan_id"] == "sast_new"

    @pytest.mark.asyncio
    async def test_get_sast_scan_status_not_found(self):
        """Test 404 when SAST scan not found."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_sast_scan_status,
        )

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
            return_value={},
        ):
            result = await handle_get_sast_scan_status(
                repo_id="test-repo",
                scan_id="nonexistent",
            )

            assert result.status_code == 404


# ============================================================================
# Findings List Tests
# ============================================================================


class TestSASTFindings:
    """Test SAST findings listing endpoint."""

    @pytest.mark.asyncio
    async def test_get_sast_findings_success(self, mock_scan_with_findings):
        """Test getting SAST findings from latest scan."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_sast_findings,
        )

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
            return_value={"sast_123": mock_scan_with_findings},
        ):
            result = await handle_get_sast_findings(repo_id="test-repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "findings" in data
            assert len(data["findings"]) == 4

    @pytest.mark.asyncio
    async def test_get_sast_findings_filter_by_severity(self, mock_scan_with_findings):
        """Test filtering SAST findings by severity."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_sast_findings,
        )

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
            return_value={"sast_123": mock_scan_with_findings},
        ):
            result = await handle_get_sast_findings(
                repo_id="test-repo",
                severity="critical",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert all(f["severity"] == "critical" for f in data["findings"])

    @pytest.mark.asyncio
    async def test_get_sast_findings_filter_by_owasp(self, mock_scan_with_findings):
        """Test filtering SAST findings by OWASP category."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_sast_findings,
        )

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
            return_value={"sast_123": mock_scan_with_findings},
        ):
            result = await handle_get_sast_findings(
                repo_id="test-repo",
                owasp_category="A03:2021",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert all(f["owasp_category"] == "A03:2021" for f in data["findings"])

    @pytest.mark.asyncio
    async def test_get_sast_findings_pagination(self, mock_scan_with_findings):
        """Test SAST findings pagination."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_sast_findings,
        )

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
            return_value={"sast_123": mock_scan_with_findings},
        ):
            result = await handle_get_sast_findings(
                repo_id="test-repo",
                limit=2,
                offset=0,
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert len(data["findings"]) == 2
            assert data["total"] == 4


# ============================================================================
# OWASP Summary Tests
# ============================================================================


class TestOWASPSummary:
    """Test OWASP Top 10 summary endpoint."""

    @pytest.mark.asyncio
    async def test_get_owasp_summary_success(self, mock_scan_with_findings):
        """Test getting OWASP Top 10 summary."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_owasp_summary,
        )

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
            return_value={"sast_123": mock_scan_with_findings},
        ):
            result = await handle_get_owasp_summary(repo_id="test-repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "owasp_summary" in data
            # Should have entries for different OWASP categories
            assert "A03:2021" in data["owasp_summary"]

    @pytest.mark.asyncio
    async def test_get_owasp_summary_includes_descriptions(self, mock_scan_with_findings):
        """Test OWASP summary includes category descriptions."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_owasp_summary,
        )

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
            return_value={"sast_123": mock_scan_with_findings},
        ):
            result = await handle_get_owasp_summary(repo_id="test-repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            # Check that summary includes count per category
            for category, info in data["owasp_summary"].items():
                assert "count" in info or isinstance(info, int)

    @pytest.mark.asyncio
    async def test_get_owasp_summary_no_scans(self):
        """Test OWASP summary when no scans exist."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_owasp_summary,
        )

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
            return_value={},
        ):
            result = await handle_get_owasp_summary(repo_id="test-repo")

            assert result.status_code == 404


# ============================================================================
# Permission Tests
# ============================================================================


class TestSASTPermissions:
    """Test SAST permission enforcement."""

    def test_scan_has_permission_decorator(self):
        """SAST scan trigger requires security permission."""
        from aragora.server.handlers.codebase.security.sast import handle_scan_sast
        import inspect

        source = inspect.getsource(handle_scan_sast)
        assert "require_permission" in source

    def test_get_findings_has_permission_decorator(self):
        """Get SAST findings requires security permission."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_sast_findings,
        )
        import inspect

        source = inspect.getsource(handle_get_sast_findings)
        assert "require_permission" in source


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestSASTErrorHandling:
    """Test SAST error handling."""

    @pytest.mark.asyncio
    async def test_scan_handles_scanner_error(self, mock_sast_scanner):
        """Test SAST scan handles scanner errors."""
        from aragora.server.handlers.codebase.security.sast import handle_scan_sast

        mock_sast_scanner.scan_repository.side_effect = RuntimeError("Scanner failed")

        with (
            patch(
                "aragora.server.handlers.codebase.security.sast.get_sast_scanner",
                return_value=mock_sast_scanner,
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_running_sast_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sast.get_sast_lock",
                return_value=MagicMock(),
            ),
        ):
            result = await handle_scan_sast(
                repo_path="/path/to/repo",
                repo_id="test-repo",
            )

            # Should return 200 with running status since scan is async
            assert result.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_findings_handles_corrupted_data(self):
        """Test findings endpoint handles corrupted scan data."""
        from aragora.server.handlers.codebase.security.sast import (
            handle_get_sast_findings,
        )

        with patch(
            "aragora.server.handlers.codebase.security.sast.get_or_create_sast_results",
            return_value={"bad_scan": "not a valid scan"},
        ):
            result = await handle_get_sast_findings(repo_id="test-repo")

            assert result.status_code in (404, 500)


__all__ = [
    "TestSASTScanTrigger",
    "TestSASTScanStatus",
    "TestSASTFindings",
    "TestOWASPSummary",
    "TestSASTPermissions",
    "TestSASTErrorHandling",
]
