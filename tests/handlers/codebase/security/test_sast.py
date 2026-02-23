"""
Tests for SAST (Static Application Security Testing) handler functions
(aragora/server/handlers/codebase/security/sast.py).

Covers all four handler functions:
- handle_scan_sast: POST trigger a SAST scan
- handle_get_sast_scan_status: GET scan status and results
- handle_get_sast_findings: GET SAST findings with filters
- handle_get_owasp_summary: GET OWASP Top 10 summary

Tests include: happy path, error paths, edge cases, filtering,
pagination, security input validation, concurrent scan protection,
and background task behavior.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.codebase.sast.models import (
    OWASPCategory,
    SASTFinding,
    SASTScanResult,
    SASTSeverity,
)
from aragora.server.handlers.codebase.security.sast import (
    handle_get_owasp_summary,
    handle_get_sast_findings,
    handle_get_sast_scan_status,
    handle_scan_sast,
)


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict:
    """Extract the data payload from a HandlerResult.

    success_response wraps in {"success": true, "data": {...}}.
    error_response wraps in {"error": "..."}.
    """
    import json

    if isinstance(result, dict):
        raw = result
    else:
        raw = json.loads(result.body)
    if isinstance(raw, dict) and raw.get("success") and "data" in raw:
        return raw["data"]
    return raw


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_finding(
    rule_id: str = "python.lang.security.injection.sql-injection",
    file_path: str = "app/db.py",
    line_start: int = 42,
    line_end: int = 42,
    column_start: int = 0,
    column_end: int = 50,
    message: str = "Possible SQL injection via string formatting",
    severity: SASTSeverity = SASTSeverity.ERROR,
    confidence: float = 0.9,
    language: str = "python",
    cwe_ids: list[str] | None = None,
    owasp_category: OWASPCategory = OWASPCategory.A03_INJECTION,
    remediation: str = "Use parameterized queries",
) -> SASTFinding:
    """Create a SASTFinding for testing."""
    return SASTFinding(
        rule_id=rule_id,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        column_start=column_start,
        column_end=column_end,
        message=message,
        severity=severity,
        confidence=confidence,
        language=language,
        cwe_ids=cwe_ids or ["CWE-89"],
        owasp_category=owasp_category,
        remediation=remediation,
    )


def _make_scan_result(
    scan_id: str = "sast_abc123",
    repo_path: str = "/tmp/test-repo",
    findings: list[SASTFinding] | None = None,
    scanned_files: int = 100,
    skipped_files: int = 5,
    scan_duration_ms: float = 1234.5,
    languages_detected: list[str] | None = None,
    rules_used: list[str] | None = None,
    scanned_at: datetime | None = None,
) -> SASTScanResult:
    """Create a SASTScanResult for testing."""
    return SASTScanResult(
        repository_path=repo_path,
        scan_id=scan_id,
        findings=findings or [],
        scanned_files=scanned_files,
        skipped_files=skipped_files,
        scan_duration_ms=scan_duration_ms,
        languages_detected=languages_detected or ["python"],
        rules_used=rules_used or ["p/owasp-top-ten"],
        scanned_at=scanned_at or datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_sast_storage():
    """Clear in-memory SAST scan stores between tests."""
    from aragora.server.handlers.codebase.security.storage import (
        _sast_scan_results,
        _running_sast_scans,
    )

    _sast_scan_results.clear()
    _running_sast_scans.clear()
    yield
    _sast_scan_results.clear()
    _running_sast_scans.clear()


@pytest.fixture(autouse=True)
def mock_sast_scanner_and_events(monkeypatch):
    """Mock SASTScanner and emit_sast_events for all tests.

    This prevents background tasks from doing real file I/O.
    """
    import aragora.server.handlers.codebase.security.sast as sast_mod

    mock_scanner = AsyncMock()
    mock_scanner.initialize = AsyncMock()
    mock_scanner.scan_repository = AsyncMock(return_value=_make_scan_result())
    mock_scanner.get_owasp_summary = AsyncMock(return_value={
        "owasp_top_10": {},
        "total_findings": 0,
        "most_common": [],
    })

    monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
    monkeypatch.setattr(sast_mod, "emit_sast_events", AsyncMock())
    return mock_scanner


@pytest.fixture
def sast_scan_results():
    """Get the SAST scan results storage."""
    from aragora.server.handlers.codebase.security.storage import (
        get_sast_scan_results,
    )

    return get_sast_scan_results()


@pytest.fixture
def running_scans():
    """Get the running SAST scans dictionary."""
    from aragora.server.handlers.codebase.security.storage import (
        get_running_sast_scans,
    )

    return get_running_sast_scans()


@pytest.fixture
def sast_scan_lock():
    """Get the SAST scan lock."""
    from aragora.server.handlers.codebase.security.storage import (
        get_sast_scan_lock,
    )

    return get_sast_scan_lock()


# ============================================================================
# handle_scan_sast tests
# ============================================================================


class TestHandleScanSast:
    """Tests for handle_scan_sast."""

    @pytest.mark.asyncio
    async def test_start_scan_success(self):
        """Starting a new SAST scan returns 200 with scan_id."""
        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "SAST scan started"
        assert "scan_id" in body
        assert body["scan_id"].startswith("sast_")
        assert body["repo_id"] == "test-repo"

    @pytest.mark.asyncio
    async def test_start_scan_generates_repo_id_when_none(self):
        """When repo_id is None, a UUID-based one is generated."""
        result = await handle_scan_sast(repo_path="/tmp/repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["repo_id"].startswith("repo_")

    @pytest.mark.asyncio
    async def test_scan_id_format(self):
        """Scan ID has the expected format (sast_ prefix + hex)."""
        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        body = _body(result)
        scan_id = body["scan_id"]
        assert scan_id.startswith("sast_")
        hex_part = scan_id[len("sast_"):]
        assert len(hex_part) == 12
        assert all(c in "0123456789abcdef" for c in hex_part)

    @pytest.mark.asyncio
    async def test_duplicate_scan_returns_409(self, running_scans):
        """If a SAST scan is already running for the repo, 409 is returned."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        task = asyncio.ensure_future(future)
        running_scans["test-repo"] = task

        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 409
        body = _body(result)
        assert "already in progress" in body.get("error", "").lower()

        # Clean up
        future.set_result(None)
        await task

    @pytest.mark.asyncio
    async def test_completed_scan_allows_rescan(self, running_scans):
        """If a previous scan is done, a new scan can be started."""
        completed_task = asyncio.ensure_future(asyncio.sleep(0))
        await completed_task
        running_scans["test-repo"] = completed_task

        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_scan_creates_async_task(self, running_scans):
        """A background task is created and stored in running_scans."""
        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 200
        assert "test-repo" in running_scans
        task = running_scans["test-repo"]
        assert isinstance(task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_scan_with_rule_sets(self):
        """Custom rule sets are accepted."""
        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
            rule_sets=["p/owasp-top-ten", "p/security-audit"],
        )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_scan_with_workspace_id(self):
        """Workspace ID is accepted."""
        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
            workspace_id="ws_001",
        )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_scan_creates_storage_entry(self, sast_scan_results):
        """Scan creates a storage entry for the repo."""
        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 200
        assert "test-repo" in sast_scan_results

    @pytest.mark.asyncio
    async def test_scan_internal_error_returns_500(self, monkeypatch):
        """Internal errors during scan setup return 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_running_sast_scans",
            MagicMock(side_effect=RuntimeError("storage failure")),
        )

        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 500
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_scan_returns_consistent_repo_id(self):
        """The returned repo_id matches what was provided."""
        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="my-custom-repo",
        )

        body = _body(result)
        assert body["repo_id"] == "my-custom-repo"

    @pytest.mark.asyncio
    async def test_concurrent_scans_different_repos(self, running_scans):
        """Different repos can have concurrent scans."""
        result_a = await handle_scan_sast(
            repo_path="/tmp/repo-a",
            repo_id="repo-a",
        )
        result_b = await handle_scan_sast(
            repo_path="/tmp/repo-b",
            repo_id="repo-b",
        )

        assert _status(result_a) == 200
        assert _status(result_b) == 200
        assert "repo-a" in running_scans
        assert "repo-b" in running_scans

    @pytest.mark.asyncio
    async def test_scan_lock_error_returns_500(self, monkeypatch):
        """Error from sast_scan_lock returns 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_sast_scan_lock",
            MagicMock(side_effect=ValueError("lock error")),
        )

        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_scan_results_error_returns_500(self, monkeypatch):
        """Error from get_sast_scan_results returns 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_sast_scan_results",
            MagicMock(side_effect=TypeError("results error")),
        )

        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 500


# ============================================================================
# handle_get_sast_scan_status tests
# ============================================================================


class TestHandleGetSastScanStatus:
    """Tests for handle_get_sast_scan_status."""

    @pytest.mark.asyncio
    async def test_get_completed_scan(self, sast_scan_results):
        """Get a completed scan status returns 200 with results."""
        scan = _make_scan_result(scan_id="sast_abc123")
        sast_scan_results["test-repo"] = {"sast_abc123": scan}

        result = await handle_get_sast_scan_status(
            repo_id="test-repo",
            scan_id="sast_abc123",
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["scan_id"] == "sast_abc123"
        assert body["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_scan_includes_to_dict_data(self, sast_scan_results):
        """The completed scan result includes to_dict() data."""
        finding = _make_finding()
        scan = _make_scan_result(scan_id="sast_full", findings=[finding])
        sast_scan_results["test-repo"] = {"sast_full": scan}

        result = await handle_get_sast_scan_status(
            repo_id="test-repo",
            scan_id="sast_full",
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "completed"
        assert "findings" in body
        assert "findings_count" in body
        assert body["findings_count"] == 1
        assert "scanned_files" in body
        assert "scan_duration_ms" in body

    @pytest.mark.asyncio
    async def test_repo_not_found(self):
        """Returns 404 when repository is not found."""
        result = await handle_get_sast_scan_status(
            repo_id="nonexistent-repo",
            scan_id="sast_abc123",
        )

        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_scan_not_found(self, sast_scan_results):
        """Returns 404 when scan is not found and not running."""
        sast_scan_results["test-repo"] = {}

        result = await handle_get_sast_scan_status(
            repo_id="test-repo",
            scan_id="nonexistent",
        )

        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_scan_still_running(self, sast_scan_results, running_scans):
        """Returns running status when scan is still in progress."""
        sast_scan_results["test-repo"] = {}
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        task = asyncio.ensure_future(future)
        running_scans["test-repo"] = task

        result = await handle_get_sast_scan_status(
            repo_id="test-repo",
            scan_id="sast_running",
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "running"
        assert body["scan_id"] == "sast_running"
        assert body["findings_count"] == 0

        # Clean up
        future.set_result(None)
        await task

    @pytest.mark.asyncio
    async def test_get_status_internal_error(self, monkeypatch):
        """Internal errors return 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_sast_scan_results",
            MagicMock(side_effect=ValueError("unexpected")),
        )

        result = await handle_get_sast_scan_status(
            repo_id="test-repo",
            scan_id="sast_abc123",
        )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_status_key_error(self, monkeypatch):
        """KeyError during status retrieval returns 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_sast_scan_lock",
            MagicMock(side_effect=KeyError("missing key")),
        )

        result = await handle_get_sast_scan_status(
            repo_id="test-repo",
            scan_id="sast_abc123",
        )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_status_type_error(self, monkeypatch):
        """TypeError during status retrieval returns 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_sast_scan_results",
            MagicMock(side_effect=TypeError("type error")),
        )

        result = await handle_get_sast_scan_status(
            repo_id="test-repo",
            scan_id="sast_abc123",
        )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_completed_scan_with_no_findings(self, sast_scan_results):
        """Completed scan with zero findings returns empty list."""
        scan = _make_scan_result(scan_id="sast_empty", findings=[])
        sast_scan_results["test-repo"] = {"sast_empty": scan}

        result = await handle_get_sast_scan_status(
            repo_id="test-repo",
            scan_id="sast_empty",
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["findings_count"] == 0
        assert body["findings"] == []

    @pytest.mark.asyncio
    async def test_completed_scan_with_multiple_findings(self, sast_scan_results):
        """Completed scan with multiple findings includes all."""
        findings = [
            _make_finding(rule_id=f"rule_{i}", file_path=f"file_{i}.py")
            for i in range(5)
        ]
        scan = _make_scan_result(scan_id="sast_multi", findings=findings)
        sast_scan_results["test-repo"] = {"sast_multi": scan}

        result = await handle_get_sast_scan_status(
            repo_id="test-repo",
            scan_id="sast_multi",
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["findings_count"] == 5


# ============================================================================
# handle_get_sast_findings tests
# ============================================================================


class TestHandleGetSastFindings:
    """Tests for handle_get_sast_findings."""

    @pytest.mark.asyncio
    async def test_get_findings_success(self, sast_scan_results):
        """Returns findings from the latest scan."""
        finding = _make_finding()
        scan = _make_scan_result(scan_id="sast_find1", findings=[finding])
        sast_scan_results["test-repo"] = {"sast_find1": scan}

        result = await handle_get_sast_findings(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert len(body["findings"]) == 1
        assert body["scan_id"] == "sast_find1"

    @pytest.mark.asyncio
    async def test_repo_not_found(self):
        """Returns 404 when repository is not found."""
        result = await handle_get_sast_findings(repo_id="nonexistent-repo")

        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_no_scans_returns_empty(self, sast_scan_results):
        """When repo has no scans, returns empty findings list."""
        sast_scan_results["test-repo"] = {}

        result = await handle_get_sast_findings(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["findings"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_uses_latest_scan(self, sast_scan_results):
        """Gets findings from the scan with the most recent scanned_at."""
        old_finding = _make_finding(rule_id="old_rule")
        new_finding = _make_finding(rule_id="new_rule")

        old_scan = _make_scan_result(
            scan_id="sast_old",
            findings=[old_finding],
            scanned_at=datetime(2024, 1, 10, 10, 0, tzinfo=timezone.utc),
        )
        new_scan = _make_scan_result(
            scan_id="sast_new",
            findings=[new_finding],
            scanned_at=datetime(2024, 1, 20, 10, 0, tzinfo=timezone.utc),
        )
        sast_scan_results["test-repo"] = {
            "sast_old": old_scan,
            "sast_new": new_scan,
        }

        result = await handle_get_sast_findings(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scan_id"] == "sast_new"
        assert body["findings"][0]["rule_id"] == "new_rule"

    @pytest.mark.asyncio
    async def test_filter_by_severity(self, sast_scan_results):
        """Filtering by severity returns only matching findings."""
        findings = [
            _make_finding(rule_id="r1", severity=SASTSeverity.CRITICAL),
            _make_finding(rule_id="r2", severity=SASTSeverity.ERROR),
            _make_finding(rule_id="r3", severity=SASTSeverity.WARNING),
            _make_finding(rule_id="r4", severity=SASTSeverity.INFO),
        ]
        scan = _make_scan_result(scan_id="sast_sev", findings=findings)
        sast_scan_results["test-repo"] = {"sast_sev": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", severity="critical"
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["findings"][0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_filter_by_severity_error(self, sast_scan_results):
        """Filtering by error severity works correctly."""
        findings = [
            _make_finding(rule_id="r1", severity=SASTSeverity.CRITICAL),
            _make_finding(rule_id="r2", severity=SASTSeverity.ERROR),
            _make_finding(rule_id="r3", severity=SASTSeverity.ERROR),
        ]
        scan = _make_scan_result(scan_id="sast_err", findings=findings)
        sast_scan_results["test-repo"] = {"sast_err": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", severity="error"
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert all(f["severity"] == "error" for f in body["findings"])

    @pytest.mark.asyncio
    async def test_filter_by_severity_warning(self, sast_scan_results):
        """Filtering by warning severity returns only warnings."""
        findings = [
            _make_finding(rule_id="r1", severity=SASTSeverity.CRITICAL),
            _make_finding(rule_id="r2", severity=SASTSeverity.WARNING),
        ]
        scan = _make_scan_result(scan_id="sast_warn", findings=findings)
        sast_scan_results["test-repo"] = {"sast_warn": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", severity="warning"
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["findings"][0]["severity"] == "warning"

    @pytest.mark.asyncio
    async def test_filter_by_severity_info(self, sast_scan_results):
        """Filtering by info severity returns only info findings."""
        findings = [
            _make_finding(rule_id="r1", severity=SASTSeverity.INFO),
            _make_finding(rule_id="r2", severity=SASTSeverity.WARNING),
        ]
        scan = _make_scan_result(scan_id="sast_info", findings=findings)
        sast_scan_results["test-repo"] = {"sast_info": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", severity="info"
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["findings"][0]["severity"] == "info"

    @pytest.mark.asyncio
    async def test_filter_by_owasp_category(self, sast_scan_results):
        """Filtering by OWASP category works correctly."""
        findings = [
            _make_finding(rule_id="r1", owasp_category=OWASPCategory.A03_INJECTION),
            _make_finding(
                rule_id="r2",
                owasp_category=OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
            ),
            _make_finding(rule_id="r3", owasp_category=OWASPCategory.A03_INJECTION),
        ]
        scan = _make_scan_result(scan_id="sast_owasp", findings=findings)
        sast_scan_results["test-repo"] = {"sast_owasp": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", owasp_category="Injection"
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_filter_by_owasp_broken_access_control(self, sast_scan_results):
        """Filtering by Broken Access Control OWASP category works."""
        findings = [
            _make_finding(
                rule_id="r1",
                owasp_category=OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
            ),
            _make_finding(rule_id="r2", owasp_category=OWASPCategory.A03_INJECTION),
        ]
        scan = _make_scan_result(scan_id="sast_ac", findings=findings)
        sast_scan_results["test-repo"] = {"sast_ac": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", owasp_category="Broken Access Control"
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_combined_severity_and_owasp_filter(self, sast_scan_results):
        """Both severity and OWASP filters can be applied simultaneously."""
        findings = [
            _make_finding(
                rule_id="r1",
                severity=SASTSeverity.CRITICAL,
                owasp_category=OWASPCategory.A03_INJECTION,
            ),
            _make_finding(
                rule_id="r2",
                severity=SASTSeverity.ERROR,
                owasp_category=OWASPCategory.A03_INJECTION,
            ),
            _make_finding(
                rule_id="r3",
                severity=SASTSeverity.CRITICAL,
                owasp_category=OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
            ),
        ]
        scan = _make_scan_result(scan_id="sast_combo", findings=findings)
        sast_scan_results["test-repo"] = {"sast_combo": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo",
            severity="critical",
            owasp_category="Injection",
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["findings"][0]["rule_id"] == "r1"

    @pytest.mark.asyncio
    async def test_pagination_limit(self, sast_scan_results):
        """Pagination limit is respected."""
        findings = [
            _make_finding(rule_id=f"r_{i}") for i in range(10)
        ]
        scan = _make_scan_result(scan_id="sast_page", findings=findings)
        sast_scan_results["test-repo"] = {"sast_page": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", limit=3
        )

        assert _status(result) == 200
        body = _body(result)
        assert len(body["findings"]) == 3
        assert body["total"] == 10
        assert body["limit"] == 3

    @pytest.mark.asyncio
    async def test_pagination_offset(self, sast_scan_results):
        """Pagination offset is respected."""
        findings = [
            _make_finding(rule_id=f"r_{i}") for i in range(10)
        ]
        scan = _make_scan_result(scan_id="sast_off", findings=findings)
        sast_scan_results["test-repo"] = {"sast_off": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", limit=3, offset=8
        )

        assert _status(result) == 200
        body = _body(result)
        assert len(body["findings"]) == 2  # Only 2 left after offset 8
        assert body["total"] == 10
        assert body["offset"] == 8

    @pytest.mark.asyncio
    async def test_pagination_beyond_total(self, sast_scan_results):
        """Offset beyond total returns empty list."""
        findings = [_make_finding()]
        scan = _make_scan_result(scan_id="sast_beyond", findings=findings)
        sast_scan_results["test-repo"] = {"sast_beyond": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", offset=100
        )

        assert _status(result) == 200
        body = _body(result)
        assert len(body["findings"]) == 0
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_default_pagination(self, sast_scan_results):
        """Default pagination is limit=100, offset=0."""
        findings = [_make_finding(rule_id=f"r_{i}") for i in range(5)]
        scan = _make_scan_result(scan_id="sast_default", findings=findings)
        sast_scan_results["test-repo"] = {"sast_default": scan}

        result = await handle_get_sast_findings(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["limit"] == 100
        assert body["offset"] == 0

    @pytest.mark.asyncio
    async def test_large_limit_returns_all(self, sast_scan_results):
        """A limit larger than total returns all findings."""
        findings = [_make_finding(rule_id=f"r_{i}") for i in range(3)]
        scan = _make_scan_result(scan_id="sast_large", findings=findings)
        sast_scan_results["test-repo"] = {"sast_large": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", limit=10000
        )

        assert _status(result) == 200
        body = _body(result)
        assert len(body["findings"]) == 3
        assert body["total"] == 3

    @pytest.mark.asyncio
    async def test_empty_findings(self, sast_scan_results):
        """Scan with no findings returns empty list."""
        scan = _make_scan_result(scan_id="sast_empty", findings=[])
        sast_scan_results["test-repo"] = {"sast_empty": scan}

        result = await handle_get_sast_findings(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["findings"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_filter_no_matching_severity(self, sast_scan_results):
        """Filtering that produces no matches returns empty list."""
        findings = [_make_finding(severity=SASTSeverity.WARNING)]
        scan = _make_scan_result(scan_id="sast_nomatch", findings=findings)
        sast_scan_results["test-repo"] = {"sast_nomatch": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", severity="critical"
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["findings"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_filter_no_matching_owasp(self, sast_scan_results):
        """Filtering by non-matching OWASP category returns empty list."""
        findings = [_make_finding(owasp_category=OWASPCategory.A03_INJECTION)]
        scan = _make_scan_result(scan_id="sast_nowasp", findings=findings)
        sast_scan_results["test-repo"] = {"sast_nowasp": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", owasp_category="SSRF"
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["findings"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_response_includes_pagination_metadata(self, sast_scan_results):
        """Response includes total, limit, offset, and scan_id."""
        findings = [_make_finding()]
        scan = _make_scan_result(scan_id="sast_meta", findings=findings)
        sast_scan_results["test-repo"] = {"sast_meta": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", limit=50, offset=0
        )

        assert _status(result) == 200
        body = _body(result)
        assert "total" in body
        assert "limit" in body
        assert "offset" in body
        assert "scan_id" in body
        assert body["limit"] == 50
        assert body["offset"] == 0

    @pytest.mark.asyncio
    async def test_findings_to_dict_format(self, sast_scan_results):
        """Each finding in the response matches to_dict() output format."""
        finding = _make_finding(
            rule_id="test_rule",
            file_path="src/app.py",
            severity=SASTSeverity.CRITICAL,
            owasp_category=OWASPCategory.A03_INJECTION,
            remediation="Use parameterized queries",
        )
        scan = _make_scan_result(scan_id="sast_dict", findings=[finding])
        sast_scan_results["test-repo"] = {"sast_dict": scan}

        result = await handle_get_sast_findings(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        f = body["findings"][0]
        assert f["rule_id"] == "test_rule"
        assert f["file_path"] == "src/app.py"
        assert f["severity"] == "critical"
        assert f["owasp_category"] == OWASPCategory.A03_INJECTION.value
        assert f["remediation"] == "Use parameterized queries"
        assert "line_start" in f
        assert "line_end" in f
        assert "message" in f
        assert "confidence" in f
        assert "language" in f

    @pytest.mark.asyncio
    async def test_get_findings_internal_error(self, monkeypatch):
        """Internal errors return 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_sast_scan_results",
            MagicMock(side_effect=TypeError("unexpected")),
        )

        result = await handle_get_sast_findings(repo_id="test-repo")

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_findings_key_error(self, monkeypatch):
        """KeyError returns 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_sast_scan_lock",
            MagicMock(side_effect=KeyError("missing")),
        )

        result = await handle_get_sast_findings(repo_id="test-repo")

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_findings_attribute_error(self, monkeypatch):
        """AttributeError returns 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_sast_scan_results",
            MagicMock(side_effect=AttributeError("no attr")),
        )

        result = await handle_get_sast_findings(repo_id="test-repo")

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_many_findings_paginated(self, sast_scan_results):
        """Large number of findings are properly paginated."""
        findings = [
            _make_finding(rule_id=f"r_{i}", file_path=f"file_{i}.py")
            for i in range(200)
        ]
        scan = _make_scan_result(scan_id="sast_many", findings=findings)
        sast_scan_results["test-repo"] = {"sast_many": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", limit=10
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 200
        assert len(body["findings"]) == 10

    @pytest.mark.asyncio
    async def test_latest_scan_among_many(self, sast_scan_results):
        """Latest scan is correctly identified among many scans."""
        scans = {}
        for i in range(5):
            scan = _make_scan_result(
                scan_id=f"sast_{i:03d}",
                findings=[_make_finding(rule_id=f"rule_from_scan_{i}")],
                scanned_at=datetime(2024, 1, 10 + i, 10, 0, tzinfo=timezone.utc),
            )
            scans[f"sast_{i:03d}"] = scan
        sast_scan_results["test-repo"] = scans

        result = await handle_get_sast_findings(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scan_id"] == "sast_004"
        assert body["findings"][0]["rule_id"] == "rule_from_scan_4"

    @pytest.mark.asyncio
    async def test_filter_severity_after_owasp(self, sast_scan_results):
        """Filtering by severity is applied after OWASP filter."""
        findings = [
            _make_finding(
                rule_id="r1",
                severity=SASTSeverity.CRITICAL,
                owasp_category=OWASPCategory.A03_INJECTION,
            ),
            _make_finding(
                rule_id="r2",
                severity=SASTSeverity.WARNING,
                owasp_category=OWASPCategory.A03_INJECTION,
            ),
            _make_finding(
                rule_id="r3",
                severity=SASTSeverity.CRITICAL,
                owasp_category=OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
            ),
        ]
        scan = _make_scan_result(scan_id="sast_both", findings=findings)
        sast_scan_results["test-repo"] = {"sast_both": scan}

        # Filter: severity=critical + owasp=Injection => only r1
        result = await handle_get_sast_findings(
            repo_id="test-repo",
            severity="critical",
            owasp_category="Injection",
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["findings"][0]["rule_id"] == "r1"

    @pytest.mark.asyncio
    async def test_zero_limit(self, sast_scan_results):
        """Zero limit returns no findings but correct total."""
        findings = [_make_finding()]
        scan = _make_scan_result(scan_id="sast_zero", findings=findings)
        sast_scan_results["test-repo"] = {"sast_zero": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", limit=0
        )

        assert _status(result) == 200
        body = _body(result)
        assert len(body["findings"]) == 0
        assert body["total"] == 1


# ============================================================================
# handle_get_owasp_summary tests
# ============================================================================


class TestHandleGetOwaspSummary:
    """Tests for handle_get_owasp_summary."""

    @pytest.mark.asyncio
    async def test_get_owasp_summary_success(
        self, sast_scan_results, mock_sast_scanner_and_events
    ):
        """Returns OWASP summary from the latest scan."""
        finding = _make_finding(owasp_category=OWASPCategory.A03_INJECTION)
        scan = _make_scan_result(scan_id="sast_owasp1", findings=[finding])
        sast_scan_results["test-repo"] = {"sast_owasp1": scan}

        expected_summary = {
            "owasp_top_10": {
                OWASPCategory.A03_INJECTION.value: {
                    "count": 1,
                    "critical": 0,
                    "error": 1,
                    "warning": 0,
                    "findings": [{"file": "app/db.py", "line": 42, "message": "Possible SQL injection via string formatting"}],
                },
            },
            "total_findings": 1,
            "most_common": [OWASPCategory.A03_INJECTION.value],
        }
        mock_sast_scanner_and_events.get_owasp_summary.return_value = expected_summary

        result = await handle_get_owasp_summary(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scan_id"] == "sast_owasp1"
        assert body["total_findings"] == 1

    @pytest.mark.asyncio
    async def test_repo_not_found(self):
        """Returns 404 when repository is not found."""
        result = await handle_get_owasp_summary(repo_id="nonexistent-repo")

        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_no_scans_returns_empty(self, sast_scan_results):
        """When repo has no scans, returns empty summary."""
        sast_scan_results["test-repo"] = {}

        result = await handle_get_owasp_summary(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["owasp_summary"] == {}
        assert body["total_findings"] == 0

    @pytest.mark.asyncio
    async def test_uses_latest_scan(
        self, sast_scan_results, mock_sast_scanner_and_events
    ):
        """Gets OWASP summary from the scan with the most recent scanned_at."""
        old_scan = _make_scan_result(
            scan_id="sast_old",
            findings=[_make_finding(rule_id="old")],
            scanned_at=datetime(2024, 1, 10, 10, 0, tzinfo=timezone.utc),
        )
        new_scan = _make_scan_result(
            scan_id="sast_new",
            findings=[_make_finding(rule_id="new")],
            scanned_at=datetime(2024, 1, 20, 10, 0, tzinfo=timezone.utc),
        )
        sast_scan_results["test-repo"] = {
            "sast_old": old_scan,
            "sast_new": new_scan,
        }

        mock_sast_scanner_and_events.get_owasp_summary.return_value = {
            "owasp_top_10": {},
            "total_findings": 1,
            "most_common": [],
        }

        result = await handle_get_owasp_summary(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scan_id"] == "sast_new"

    @pytest.mark.asyncio
    async def test_calls_scanner_get_owasp_summary(
        self, sast_scan_results, mock_sast_scanner_and_events
    ):
        """The handler calls scanner.get_owasp_summary with the findings."""
        findings = [
            _make_finding(rule_id="r1", owasp_category=OWASPCategory.A03_INJECTION),
            _make_finding(
                rule_id="r2",
                owasp_category=OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
            ),
        ]
        scan = _make_scan_result(scan_id="sast_call", findings=findings)
        sast_scan_results["test-repo"] = {"sast_call": scan}

        mock_sast_scanner_and_events.get_owasp_summary.return_value = {
            "owasp_top_10": {},
            "total_findings": 2,
            "most_common": [],
        }

        await handle_get_owasp_summary(repo_id="test-repo")

        mock_sast_scanner_and_events.get_owasp_summary.assert_called_once()
        call_args = mock_sast_scanner_and_events.get_owasp_summary.call_args
        assert len(call_args[0][0]) == 2  # Passed 2 findings

    @pytest.mark.asyncio
    async def test_owasp_summary_internal_error(self, monkeypatch):
        """Internal errors return 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_sast_scan_results",
            MagicMock(side_effect=ValueError("unexpected")),
        )

        result = await handle_get_owasp_summary(repo_id="test-repo")

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_owasp_summary_key_error(self, monkeypatch):
        """KeyError returns 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_sast_scan_lock",
            MagicMock(side_effect=KeyError("missing")),
        )

        result = await handle_get_owasp_summary(repo_id="test-repo")

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_owasp_summary_type_error(self, monkeypatch):
        """TypeError returns 500."""
        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(
            sast_mod,
            "get_sast_scan_results",
            MagicMock(side_effect=TypeError("type error")),
        )

        result = await handle_get_owasp_summary(repo_id="test-repo")

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_owasp_summary_response_includes_scan_id(
        self, sast_scan_results, mock_sast_scanner_and_events
    ):
        """The response includes the scan_id from the latest scan."""
        scan = _make_scan_result(scan_id="sast_scanid", findings=[_make_finding()])
        sast_scan_results["test-repo"] = {"sast_scanid": scan}

        mock_sast_scanner_and_events.get_owasp_summary.return_value = {
            "owasp_top_10": {},
            "total_findings": 1,
            "most_common": [],
        }

        result = await handle_get_owasp_summary(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scan_id"] == "sast_scanid"

    @pytest.mark.asyncio
    async def test_owasp_summary_with_no_findings(
        self, sast_scan_results, mock_sast_scanner_and_events
    ):
        """OWASP summary with no findings returns empty summary."""
        scan = _make_scan_result(scan_id="sast_nofind", findings=[])
        sast_scan_results["test-repo"] = {"sast_nofind": scan}

        mock_sast_scanner_and_events.get_owasp_summary.return_value = {
            "owasp_top_10": {},
            "total_findings": 0,
            "most_common": [],
        }

        result = await handle_get_owasp_summary(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["total_findings"] == 0


# ============================================================================
# Background scan task behavior tests
# ============================================================================


class TestScanBackgroundTask:
    """Tests for the background SAST scan task behavior."""

    @pytest.mark.asyncio
    async def test_background_scan_stores_result(
        self, sast_scan_results, monkeypatch
    ):
        """The background task stores the scan result on success."""
        scan_result = _make_scan_result(scan_id="bg_sast")

        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(return_value=scan_result)

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
        monkeypatch.setattr(sast_mod, "emit_sast_events", AsyncMock())

        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        scan_id = _body(result)["scan_id"]

        # Let the background task complete
        await asyncio.sleep(0.1)

        # The scan result should be stored
        assert "test-repo" in sast_scan_results
        assert scan_id in sast_scan_results["test-repo"]

    @pytest.mark.asyncio
    async def test_background_scan_initializes_scanner(self, monkeypatch):
        """The background task initializes the scanner before scanning."""
        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(return_value=_make_scan_result())

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
        monkeypatch.setattr(sast_mod, "emit_sast_events", AsyncMock())

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        # Let the background task complete
        await asyncio.sleep(0.1)

        mock_scanner.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_background_scan_passes_rule_sets(self, monkeypatch):
        """The background task passes rule_sets to the scanner."""
        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(return_value=_make_scan_result())

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
        monkeypatch.setattr(sast_mod, "emit_sast_events", AsyncMock())

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
            rule_sets=["p/owasp-top-ten", "p/security-audit"],
        )

        # Let the background task complete
        await asyncio.sleep(0.1)

        mock_scanner.scan_repository.assert_called_once()
        call_kwargs = mock_scanner.scan_repository.call_args[1]
        assert call_kwargs["repo_path"] == "/tmp/test-repo"
        assert call_kwargs["rule_sets"] == ["p/owasp-top-ten", "p/security-audit"]

    @pytest.mark.asyncio
    async def test_background_scan_passes_scan_id(self, monkeypatch):
        """The background task passes the scan_id to the scanner."""
        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(return_value=_make_scan_result())

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
        monkeypatch.setattr(sast_mod, "emit_sast_events", AsyncMock())

        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        scan_id = _body(result)["scan_id"]

        # Let the background task complete
        await asyncio.sleep(0.1)

        call_kwargs = mock_scanner.scan_repository.call_args[1]
        assert call_kwargs["scan_id"] == scan_id

    @pytest.mark.asyncio
    async def test_background_scan_failure_cleans_up(
        self, running_scans, monkeypatch
    ):
        """On failure, the task is removed from running_scans."""
        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(
            side_effect=RuntimeError("disk error")
        )

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        # Let the background task fail
        await asyncio.sleep(0.1)

        assert "test-repo" not in running_scans

    @pytest.mark.asyncio
    async def test_background_scan_success_cleans_up(
        self, running_scans, monkeypatch
    ):
        """On success, the task is removed from running_scans."""
        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(return_value=_make_scan_result())

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
        monkeypatch.setattr(sast_mod, "emit_sast_events", AsyncMock())

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        # Initially in running_scans
        assert "test-repo" in running_scans

        # Let the background task complete
        await asyncio.sleep(0.1)

        assert "test-repo" not in running_scans

    @pytest.mark.asyncio
    async def test_background_scan_os_error_cleans_up(
        self, running_scans, monkeypatch
    ):
        """OSError during scan still cleans up running_scans."""
        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(
            side_effect=OSError("no access")
        )

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        # Let the background task fail
        await asyncio.sleep(0.1)

        assert "test-repo" not in running_scans

    @pytest.mark.asyncio
    async def test_background_scan_value_error_cleans_up(
        self, running_scans, monkeypatch
    ):
        """ValueError during scan still cleans up running_scans."""
        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(
            side_effect=ValueError("bad path")
        )

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        await asyncio.sleep(0.1)
        assert "test-repo" not in running_scans

    @pytest.mark.asyncio
    async def test_background_scan_type_error_cleans_up(
        self, running_scans, monkeypatch
    ):
        """TypeError during scan still cleans up running_scans."""
        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(
            side_effect=TypeError("wrong type")
        )

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        await asyncio.sleep(0.1)
        assert "test-repo" not in running_scans

    @pytest.mark.asyncio
    async def test_background_scan_emits_events_for_critical(
        self, sast_scan_results, monkeypatch
    ):
        """The background task emits events when critical findings exist."""
        critical_finding = _make_finding(
            severity=SASTSeverity.CRITICAL,
        )
        scan_result = _make_scan_result(
            scan_id="bg_events",
            findings=[critical_finding],
        )

        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(return_value=scan_result)
        mock_emit = AsyncMock()

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
        monkeypatch.setattr(sast_mod, "emit_sast_events", mock_emit)

        result = await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
            workspace_id="ws_test",
        )

        scan_id = _body(result)["scan_id"]

        # Let background task complete
        await asyncio.sleep(0.1)

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][1] == "test-repo"  # repo_id
        assert call_args[0][2] == scan_id  # scan_id
        assert call_args[0][3] == "ws_test"  # workspace_id

    @pytest.mark.asyncio
    async def test_background_scan_emits_events_for_error_severity(
        self, sast_scan_results, monkeypatch
    ):
        """The background task emits events for error (high) severity findings."""
        error_finding = _make_finding(
            severity=SASTSeverity.ERROR,
        )
        scan_result = _make_scan_result(
            scan_id="bg_error",
            findings=[error_finding],
        )

        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(return_value=scan_result)
        mock_emit = AsyncMock()

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
        monkeypatch.setattr(sast_mod, "emit_sast_events", mock_emit)

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        await asyncio.sleep(0.1)

        mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_background_scan_no_events_for_warning_only(
        self, sast_scan_results, monkeypatch
    ):
        """The background task does not emit events for warning-only findings."""
        warning_finding = _make_finding(
            severity=SASTSeverity.WARNING,
        )
        scan_result = _make_scan_result(
            scan_id="bg_warn",
            findings=[warning_finding],
        )

        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(return_value=scan_result)
        mock_emit = AsyncMock()

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
        monkeypatch.setattr(sast_mod, "emit_sast_events", mock_emit)

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        await asyncio.sleep(0.1)

        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_background_scan_no_events_for_info_only(
        self, sast_scan_results, monkeypatch
    ):
        """The background task does not emit events for info-only findings."""
        info_finding = _make_finding(
            severity=SASTSeverity.INFO,
        )
        scan_result = _make_scan_result(
            scan_id="bg_info",
            findings=[info_finding],
        )

        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(return_value=scan_result)
        mock_emit = AsyncMock()

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
        monkeypatch.setattr(sast_mod, "emit_sast_events", mock_emit)

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        await asyncio.sleep(0.1)

        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_background_scan_no_events_for_no_findings(
        self, sast_scan_results, monkeypatch
    ):
        """The background task does not emit events when there are no findings."""
        scan_result = _make_scan_result(
            scan_id="bg_none",
            findings=[],
        )

        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(return_value=scan_result)
        mock_emit = AsyncMock()

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
        monkeypatch.setattr(sast_mod, "emit_sast_events", mock_emit)

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        await asyncio.sleep(0.1)

        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_background_scan_initializer_error_cleans_up(
        self, running_scans, monkeypatch
    ):
        """If scanner.initialize() fails, running_scans is cleaned up."""
        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock(
            side_effect=RuntimeError("init failed")
        )

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        await asyncio.sleep(0.1)
        assert "test-repo" not in running_scans


# ============================================================================
# Security tests
# ============================================================================


class TestSecurityInputValidation:
    """Security-related tests for input validation."""

    @pytest.mark.asyncio
    async def test_repo_id_with_alphanumeric(self):
        """Alphanumeric repo IDs are accepted."""
        result = await handle_scan_sast(
            repo_path="/tmp/test",
            repo_id="my-repo_123",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_empty_repo_path(self):
        """Empty repo_path is accepted (scanner validates it)."""
        result = await handle_scan_sast(
            repo_path="",
            repo_id="test-repo",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_very_long_repo_path(self):
        """Very long repo_path does not cause handler crash."""
        result = await handle_scan_sast(
            repo_path="/" + "a" * 10000,
            repo_id="test-repo",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_none_rule_sets(self):
        """None rule_sets is accepted (uses defaults)."""
        result = await handle_scan_sast(
            repo_path="/tmp/test",
            repo_id="test-repo",
            rule_sets=None,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_empty_rule_sets(self):
        """Empty rule_sets list is accepted."""
        result = await handle_scan_sast(
            repo_path="/tmp/test",
            repo_id="test-repo",
            rule_sets=[],
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_none_workspace_id(self):
        """None workspace_id is accepted."""
        result = await handle_scan_sast(
            repo_path="/tmp/test",
            repo_id="test-repo",
            workspace_id=None,
        )
        assert _status(result) == 200


# ============================================================================
# Edge case tests
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_multiple_repos_isolated(self, sast_scan_results):
        """Scans from different repos are isolated."""
        scan_a = _make_scan_result(scan_id="sast_a")
        scan_b = _make_scan_result(scan_id="sast_b")
        sast_scan_results["repo-a"] = {"sast_a": scan_a}
        sast_scan_results["repo-b"] = {"sast_b": scan_b}

        result_a = await handle_get_sast_scan_status(
            repo_id="repo-a", scan_id="sast_a"
        )
        result_b = await handle_get_sast_scan_status(
            repo_id="repo-b", scan_id="sast_b"
        )

        assert _status(result_a) == 200
        assert _status(result_b) == 200

    @pytest.mark.asyncio
    async def test_cross_repo_scan_not_visible(self, sast_scan_results):
        """Scan from repo-a is not visible when querying repo-b."""
        scan = _make_scan_result(scan_id="sast_a")
        sast_scan_results["repo-a"] = {"sast_a": scan}

        result = await handle_get_sast_scan_status(
            repo_id="repo-b", scan_id="sast_a"
        )

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_scan_with_many_findings(self, sast_scan_results):
        """Handler handles a large number of findings efficiently."""
        findings = [
            _make_finding(rule_id=f"r_{i}", file_path=f"file_{i}.py")
            for i in range(500)
        ]
        scan = _make_scan_result(scan_id="sast_many", findings=findings)
        sast_scan_results["test-repo"] = {"sast_many": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", limit=10
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 500
        assert len(body["findings"]) == 10

    @pytest.mark.asyncio
    async def test_findings_data_integrity(self, sast_scan_results):
        """Finding data is preserved through storage and retrieval."""
        finding = SASTFinding(
            rule_id="integrity.test.rule",
            file_path="src/critical_module.py",
            line_start=99,
            line_end=105,
            column_start=4,
            column_end=80,
            message="Critical SQL injection in production code",
            severity=SASTSeverity.CRITICAL,
            confidence=0.98,
            language="python",
            snippet="cursor.execute(f'SELECT * FROM {table}')",
            cwe_ids=["CWE-89", "CWE-94"],
            owasp_category=OWASPCategory.A03_INJECTION,
            vulnerability_class="injection",
            remediation="Use parameterized queries",
            source="semgrep",
            rule_name="python-sql-injection",
        )
        scan = _make_scan_result(scan_id="sast_integrity", findings=[finding])
        sast_scan_results["test-repo"] = {"sast_integrity": scan}

        result = await handle_get_sast_findings(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        f = body["findings"][0]
        assert f["rule_id"] == "integrity.test.rule"
        assert f["file_path"] == "src/critical_module.py"
        assert f["line_start"] == 99
        assert f["line_end"] == 105
        assert f["severity"] == "critical"
        assert f["confidence"] == 0.98
        assert f["language"] == "python"
        assert f["cwe_ids"] == ["CWE-89", "CWE-94"]
        assert f["owasp_category"] == OWASPCategory.A03_INJECTION.value
        assert f["remediation"] == "Use parameterized queries"

    @pytest.mark.asyncio
    async def test_owasp_categories_all_filterable(self, sast_scan_results):
        """All OWASP categories can be filtered."""
        categories = [
            OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
            OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
            OWASPCategory.A03_INJECTION,
            OWASPCategory.A04_INSECURE_DESIGN,
            OWASPCategory.A05_SECURITY_MISCONFIGURATION,
        ]
        findings = [
            _make_finding(
                rule_id=f"r_{i}",
                owasp_category=cat,
            )
            for i, cat in enumerate(categories)
        ]
        scan = _make_scan_result(scan_id="sast_all_cats", findings=findings)
        sast_scan_results["test-repo"] = {"sast_all_cats": scan}

        for cat in categories:
            # Use a substring from the category value for filtering
            # The handler checks: owasp_category in f.owasp_category.value
            filter_str = cat.value.split(" - ")[1]  # e.g., "Injection"
            result = await handle_get_sast_findings(
                repo_id="test-repo", owasp_category=filter_str
            )

            assert _status(result) == 200
            body = _body(result)
            assert body["total"] >= 1, f"Expected at least 1 finding for {cat.value}"

    @pytest.mark.asyncio
    async def test_severity_levels_all_filterable(self, sast_scan_results):
        """All severity levels can be filtered."""
        severities = [
            SASTSeverity.CRITICAL,
            SASTSeverity.ERROR,
            SASTSeverity.WARNING,
            SASTSeverity.INFO,
        ]
        findings = [
            _make_finding(rule_id=f"r_{sev.value}", severity=sev)
            for sev in severities
        ]
        scan = _make_scan_result(scan_id="sast_all_sev", findings=findings)
        sast_scan_results["test-repo"] = {"sast_all_sev": scan}

        for sev in severities:
            result = await handle_get_sast_findings(
                repo_id="test-repo", severity=sev.value
            )

            assert _status(result) == 200
            body = _body(result)
            assert body["total"] == 1, f"Expected 1 finding for {sev.value}"
            assert body["findings"][0]["severity"] == sev.value

    @pytest.mark.asyncio
    async def test_concurrent_scans_different_repos(self, running_scans):
        """Different repos can have concurrent scans."""
        result_a = await handle_scan_sast(
            repo_path="/tmp/repo-a", repo_id="repo-a"
        )
        result_b = await handle_scan_sast(
            repo_path="/tmp/repo-b", repo_id="repo-b"
        )

        assert _status(result_a) == 200
        assert _status(result_b) == 200
        assert "repo-a" in running_scans
        assert "repo-b" in running_scans

    @pytest.mark.asyncio
    async def test_pagination_limit_and_offset_combined(self, sast_scan_results):
        """Pagination with both limit and offset works together."""
        findings = [
            _make_finding(rule_id=f"r_{i}") for i in range(20)
        ]
        scan = _make_scan_result(scan_id="sast_pag", findings=findings)
        sast_scan_results["test-repo"] = {"sast_pag": scan}

        # Get page 2 (offset=5, limit=5)
        result = await handle_get_sast_findings(
            repo_id="test-repo", limit=5, offset=5
        )

        assert _status(result) == 200
        body = _body(result)
        assert len(body["findings"]) == 5
        assert body["total"] == 20
        assert body["limit"] == 5
        assert body["offset"] == 5

    @pytest.mark.asyncio
    async def test_negative_offset(self, sast_scan_results):
        """Negative offset is handled by Python slicing."""
        findings = [_make_finding(rule_id=f"r_{i}") for i in range(5)]
        scan = _make_scan_result(scan_id="sast_neg", findings=findings)
        sast_scan_results["test-repo"] = {"sast_neg": scan}

        result = await handle_get_sast_findings(
            repo_id="test-repo", offset=-1
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 5

    @pytest.mark.asyncio
    async def test_scan_result_to_dict_in_status(self, sast_scan_results):
        """The scan status response includes fields from to_dict()."""
        finding = _make_finding()
        scan = _make_scan_result(
            scan_id="sast_todict",
            findings=[finding],
            scanned_files=150,
            skipped_files=10,
            scan_duration_ms=5678.9,
            languages_detected=["python", "javascript"],
            rules_used=["p/owasp-top-ten", "p/security-audit"],
        )
        sast_scan_results["test-repo"] = {"sast_todict": scan}

        result = await handle_get_sast_scan_status(
            repo_id="test-repo", scan_id="sast_todict"
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["scanned_files"] == 150
        assert body["skipped_files"] == 10
        assert body["scan_duration_ms"] == 5678.9
        assert body["languages_detected"] == ["python", "javascript"]
        assert body["rules_used"] == ["p/owasp-top-ten", "p/security-audit"]

    @pytest.mark.asyncio
    async def test_scan_result_summary_in_status(self, sast_scan_results):
        """The scan status includes summary with severity and OWASP breakdown."""
        findings = [
            _make_finding(
                rule_id="r1",
                severity=SASTSeverity.CRITICAL,
                owasp_category=OWASPCategory.A03_INJECTION,
            ),
            _make_finding(
                rule_id="r2",
                severity=SASTSeverity.ERROR,
                owasp_category=OWASPCategory.A03_INJECTION,
            ),
            _make_finding(
                rule_id="r3",
                severity=SASTSeverity.WARNING,
                owasp_category=OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
            ),
        ]
        scan = _make_scan_result(scan_id="sast_summ", findings=findings)
        sast_scan_results["test-repo"] = {"sast_summ": scan}

        result = await handle_get_sast_scan_status(
            repo_id="test-repo", scan_id="sast_summ"
        )

        assert _status(result) == 200
        body = _body(result)
        assert "summary" in body
        summary = body["summary"]
        assert "by_severity" in summary
        assert "by_owasp" in summary
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_severity"]["error"] == 1
        assert summary["by_severity"]["warning"] == 1

    @pytest.mark.asyncio
    async def test_background_scan_mixed_severity_events(
        self, sast_scan_results, monkeypatch
    ):
        """Events are emitted when mixed severity includes critical/error."""
        findings = [
            _make_finding(rule_id="r1", severity=SASTSeverity.CRITICAL),
            _make_finding(rule_id="r2", severity=SASTSeverity.WARNING),
            _make_finding(rule_id="r3", severity=SASTSeverity.INFO),
        ]
        scan_result = _make_scan_result(
            scan_id="bg_mixed",
            findings=findings,
        )

        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository = AsyncMock(return_value=scan_result)
        mock_emit = AsyncMock()

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)
        monkeypatch.setattr(sast_mod, "emit_sast_events", mock_emit)

        await handle_scan_sast(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        await asyncio.sleep(0.1)

        # Should emit because there is a critical finding
        mock_emit.assert_called_once()
