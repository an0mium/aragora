"""Tests for codebase metrics handler (aragora/server/handlers/codebase/metrics.py).

Covers all routes and behavior of the MetricsHandler class
and standalone helper functions:

- MetricsHandler: can_handle(), handle(), RBAC enforcement
- handle_post_analyze: validation, success, in-progress guard, error cases
- handle_get_metrics: latest, by analysis_id, not found
- handle_get_hotspots: filtering, sorting, limit, no data
- handle_get_duplicates: filtering, sorting, limit, no data
- handle_get_file_metrics: found, not found, suffix matching
- handle_list_analyses: pagination, sorting, empty
- _get_metrics_analyzer: factory with thresholds
- _get_or_create_repo_metrics: storage creation
- _get_user_id: context extraction
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.codebase import (
    DuplicateBlock,
    FileMetrics,
    FunctionMetrics,
    HotspotFinding,
    MetricsReport,
)
from aragora.server.handlers.codebase.metrics import (
    METRICS_ANALYZE_PERMISSION,
    METRICS_READ_PERMISSION,
    MetricsHandler,
    _get_metrics_analyzer,
    _get_or_create_repo_metrics,
    _metrics_lock,
    _metrics_reports,
    _running_analyses,
    handle_analyze_metrics,
    handle_get_duplicates,
    handle_get_file_metrics,
    handle_get_hotspots,
    handle_get_metrics,
    handle_list_analyses,
)


# ============================================================================
# Helpers
# ============================================================================


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for MetricsHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
        roles: str = "",
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        if roles:
            self.headers["X-User-Roles"] = roles
        self.rfile = MagicMock()

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ============================================================================
# Test data builders
# ============================================================================


def _make_function_metrics(
    name: str = "my_func",
    file_path: str = "src/main.py",
    start_line: int = 1,
    end_line: int = 10,
    lines_of_code: int = 10,
    cyclomatic_complexity: int = 3,
    cognitive_complexity: int = 2,
    parameter_count: int = 2,
    nested_depth: int = 1,
) -> FunctionMetrics:
    return FunctionMetrics(
        name=name,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        lines_of_code=lines_of_code,
        cyclomatic_complexity=cyclomatic_complexity,
        cognitive_complexity=cognitive_complexity,
        parameter_count=parameter_count,
        nested_depth=nested_depth,
    )


def _make_file_metrics(
    file_path: str = "src/main.py",
    language: str = "python",
    lines_of_code: int = 100,
    lines_of_comments: int = 20,
    blank_lines: int = 10,
    classes: int = 1,
    imports: int = 5,
    avg_complexity: float = 3.5,
    max_complexity: int = 8,
    maintainability_index: float = 75.0,
    functions: list[FunctionMetrics] | None = None,
) -> FileMetrics:
    return FileMetrics(
        file_path=file_path,
        language=language,
        lines_of_code=lines_of_code,
        lines_of_comments=lines_of_comments,
        blank_lines=blank_lines,
        classes=classes,
        imports=imports,
        avg_complexity=avg_complexity,
        max_complexity=max_complexity,
        maintainability_index=maintainability_index,
        functions=functions or [_make_function_metrics(file_path=file_path)],
    )


def _make_hotspot(
    file_path: str = "src/complex.py",
    function_name: str = "complex_func",
    complexity: float = 15.0,
    change_frequency: int = 10,
) -> HotspotFinding:
    return HotspotFinding(
        file_path=file_path,
        function_name=function_name,
        complexity=complexity,
        change_frequency=change_frequency,
    )


def _make_duplicate(
    hash_val: str = "abcdef1234567890",
    lines: int = 10,
    occurrences: list[tuple[str, int, int]] | None = None,
) -> DuplicateBlock:
    return DuplicateBlock(
        hash=hash_val,
        lines=lines,
        occurrences=occurrences or [("src/a.py", 1, 10), ("src/b.py", 20, 30)],
    )


def _make_report(
    repository: str = "test-repo",
    scan_id: str = "metrics_abc123",
    scanned_at: datetime | None = None,
    total_files: int = 10,
    total_lines: int = 1000,
    avg_complexity: float = 5.0,
    max_complexity: int = 15,
    maintainability_index: float = 70.0,
    files: list[FileMetrics] | None = None,
    hotspots: list[HotspotFinding] | None = None,
    duplicates: list[DuplicateBlock] | None = None,
) -> MetricsReport:
    return MetricsReport(
        repository=repository,
        scan_id=scan_id,
        scanned_at=scanned_at or datetime.now(timezone.utc),
        total_files=total_files,
        total_lines=total_lines,
        avg_complexity=avg_complexity,
        max_complexity=max_complexity,
        maintainability_index=maintainability_index,
        files=files or [],
        hotspots=hotspots or [],
        duplicates=duplicates or [],
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_metrics_state():
    """Clear global metrics state before each test."""
    _metrics_reports.clear()
    _running_analyses.clear()
    yield
    _metrics_reports.clear()
    _running_analyses.clear()


@pytest.fixture
def handler():
    """Create a MetricsHandler with a minimal mock server context."""
    ctx: dict[str, Any] = {}
    return MetricsHandler(ctx)


@pytest.fixture
def mock_http():
    """Create a basic mock HTTP handler."""
    return MockHTTPHandler()


# ============================================================================
# Tests: _get_metrics_analyzer
# ============================================================================


class TestGetMetricsAnalyzer:
    """Tests for _get_metrics_analyzer factory."""

    def test_default_thresholds(self):
        analyzer = _get_metrics_analyzer()
        assert analyzer.complexity_warning == 10
        assert analyzer.complexity_error == 20
        # duplication_threshold is forwarded to DuplicateDetector as min_lines
        assert analyzer.duplicate_detector.min_lines == 6

    def test_custom_thresholds(self):
        analyzer = _get_metrics_analyzer(
            complexity_warning=5,
            complexity_error=15,
            duplication_threshold=3,
        )
        assert analyzer.complexity_warning == 5
        assert analyzer.complexity_error == 15
        assert analyzer.duplicate_detector.min_lines == 3

    def test_returns_new_instance_each_call(self):
        a1 = _get_metrics_analyzer()
        a2 = _get_metrics_analyzer()
        assert a1 is not a2


# ============================================================================
# Tests: _get_or_create_repo_metrics
# ============================================================================


class TestGetOrCreateRepoMetrics:
    """Tests for _get_or_create_repo_metrics."""

    def test_creates_new_repo(self):
        result = _get_or_create_repo_metrics("new-repo")
        assert isinstance(result, dict)
        assert len(result) == 0
        assert "new-repo" in _metrics_reports

    def test_returns_existing_repo(self):
        _metrics_reports["existing"] = {"id1": _make_report(scan_id="id1")}
        result = _get_or_create_repo_metrics("existing")
        assert "id1" in result

    def test_thread_safe(self):
        """Concurrent calls for same repo don't create duplicates."""
        results = []

        def create():
            results.append(_get_or_create_repo_metrics("concurrent-repo"))

        threads = [threading.Thread(target=create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert "concurrent-repo" in _metrics_reports


# ============================================================================
# Tests: handle_analyze_metrics
# ============================================================================


class TestHandleAnalyzeMetrics:
    """Tests for handle_analyze_metrics standalone function."""

    @pytest.mark.asyncio
    async def test_success(self):
        result = await handle_analyze_metrics(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )
        assert result["success"] is True
        assert result["status"] == "running"
        assert result["repository"] == "test-repo"
        assert result["analysis_id"].startswith("metrics_")

    @pytest.mark.asyncio
    async def test_generates_repo_id_when_none(self):
        result = await handle_analyze_metrics(repo_path="/tmp/test-repo")
        assert result["success"] is True
        assert result["repository"].startswith("repo_")

    @pytest.mark.asyncio
    async def test_analysis_already_running(self):
        """When an analysis is already running for the repo, returns error."""
        # Create a fake running task
        future = asyncio.get_event_loop().create_future()
        task = asyncio.ensure_future(asyncio.sleep(100))
        _running_analyses["running-repo"] = task
        try:
            result = await handle_analyze_metrics(
                repo_path="/tmp/test",
                repo_id="running-repo",
            )
            assert result["success"] is False
            assert "already in progress" in result["error"]
            assert result["analysis_id"] is None
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_analysis_replaces_completed_task(self):
        """When a previous analysis is done, allow a new one."""
        done_task = asyncio.ensure_future(asyncio.sleep(0))
        await done_task  # Let it complete
        _running_analyses["done-repo"] = done_task

        result = await handle_analyze_metrics(
            repo_path="/tmp/test",
            repo_id="done-repo",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_custom_thresholds_forwarded(self):
        result = await handle_analyze_metrics(
            repo_path="/tmp/test",
            repo_id="repo-custom",
            complexity_warning=5,
            complexity_error=15,
            duplication_threshold=3,
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_include_exclude_patterns(self):
        result = await handle_analyze_metrics(
            repo_path="/tmp/test",
            repo_id="repo-patterns",
            include_patterns=["src/**/*.py"],
            exclude_patterns=["**/tests/**"],
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_workspace_and_user_forwarded(self):
        result = await handle_analyze_metrics(
            repo_path="/tmp/test",
            repo_id="repo-ws",
            workspace_id="ws-1",
            user_id="user-1",
        )
        assert result["success"] is True


# ============================================================================
# Tests: handle_get_metrics
# ============================================================================


class TestHandleGetMetrics:
    """Tests for handle_get_metrics standalone function."""

    @pytest.mark.asyncio
    async def test_no_analyses(self):
        result = await handle_get_metrics(repo_id="empty-repo")
        assert result["success"] is False
        assert "No analyses found" in result["error"]

    @pytest.mark.asyncio
    async def test_get_latest(self):
        older = _make_report(
            scan_id="older",
            scanned_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            total_files=5,
        )
        newer = _make_report(
            scan_id="newer",
            scanned_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            total_files=10,
        )
        _metrics_reports["my-repo"] = {"older": older, "newer": newer}

        result = await handle_get_metrics(repo_id="my-repo")
        assert result["success"] is True
        assert result["report"]["scan_id"] == "newer"

    @pytest.mark.asyncio
    async def test_get_by_analysis_id(self):
        report = _make_report(scan_id="specific-id")
        _metrics_reports["my-repo"] = {"specific-id": report}

        result = await handle_get_metrics(repo_id="my-repo", analysis_id="specific-id")
        assert result["success"] is True
        assert result["report"]["scan_id"] == "specific-id"

    @pytest.mark.asyncio
    async def test_analysis_not_found(self):
        _metrics_reports["my-repo"] = {}

        result = await handle_get_metrics(repo_id="my-repo", analysis_id="nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_report_to_dict_fields(self):
        report = _make_report(
            scan_id="fields-test",
            total_files=42,
            total_lines=5000,
        )
        _metrics_reports["my-repo"] = {"fields-test": report}

        result = await handle_get_metrics(repo_id="my-repo", analysis_id="fields-test")
        assert result["success"] is True
        r = result["report"]
        assert r["repository"] == "test-repo"
        assert "summary" in r
        assert r["summary"]["total_files"] == 42
        assert r["summary"]["total_lines"] == 5000


# ============================================================================
# Tests: handle_get_hotspots
# ============================================================================


class TestHandleGetHotspots:
    """Tests for handle_get_hotspots standalone function."""

    @pytest.mark.asyncio
    async def test_no_analyses(self):
        result = await handle_get_hotspots(repo_id="empty")
        assert result["success"] is False
        assert "No analyses found" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_hotspots_above_min_complexity(self):
        h1 = _make_hotspot(complexity=3.0, function_name="low")
        h2 = _make_hotspot(complexity=8.0, function_name="medium")
        h3 = _make_hotspot(complexity=15.0, function_name="high")
        report = _make_report(scan_id="hs-1", hotspots=[h1, h2, h3])
        _metrics_reports["repo"] = {"hs-1": report}

        result = await handle_get_hotspots(repo_id="repo", min_complexity=5)
        assert result["success"] is True
        names = [h["function_name"] for h in result["hotspots"]]
        assert "low" not in names
        assert "medium" in names
        assert "high" in names

    @pytest.mark.asyncio
    async def test_limit(self):
        hotspots = [
            _make_hotspot(complexity=float(i + 10), function_name=f"f{i}") for i in range(10)
        ]
        report = _make_report(scan_id="hs-lim", hotspots=hotspots)
        _metrics_reports["repo"] = {"hs-lim": report}

        result = await handle_get_hotspots(repo_id="repo", limit=3)
        assert result["success"] is True
        assert len(result["hotspots"]) == 3

    @pytest.mark.asyncio
    async def test_sorted_by_risk_score_descending(self):
        h_low = _make_hotspot(complexity=5.0, change_frequency=1)
        h_high = _make_hotspot(complexity=20.0, change_frequency=50)
        report = _make_report(scan_id="hs-sort", hotspots=[h_low, h_high])
        _metrics_reports["repo"] = {"hs-sort": report}

        result = await handle_get_hotspots(repo_id="repo", min_complexity=1)
        assert result["success"] is True
        scores = [h["risk_score"] for h in result["hotspots"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_total_count(self):
        hotspots = [_make_hotspot(complexity=float(i + 10)) for i in range(5)]
        report = _make_report(scan_id="hs-ct", hotspots=hotspots)
        _metrics_reports["repo"] = {"hs-ct": report}

        result = await handle_get_hotspots(repo_id="repo")
        assert result["total"] == 5

    @pytest.mark.asyncio
    async def test_analysis_id_returned(self):
        report = _make_report(scan_id="hs-aid", hotspots=[_make_hotspot()])
        _metrics_reports["repo"] = {"hs-aid": report}

        result = await handle_get_hotspots(repo_id="repo")
        assert result["analysis_id"] == "hs-aid"

    @pytest.mark.asyncio
    async def test_empty_hotspots(self):
        report = _make_report(scan_id="hs-empty", hotspots=[])
        _metrics_reports["repo"] = {"hs-empty": report}

        result = await handle_get_hotspots(repo_id="repo")
        assert result["success"] is True
        assert result["hotspots"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_min_complexity_filters_all(self):
        hotspots = [_make_hotspot(complexity=3.0)]
        report = _make_report(scan_id="hs-all", hotspots=hotspots)
        _metrics_reports["repo"] = {"hs-all": report}

        result = await handle_get_hotspots(repo_id="repo", min_complexity=50)
        assert result["success"] is True
        assert result["hotspots"] == []


# ============================================================================
# Tests: handle_get_duplicates
# ============================================================================


class TestHandleGetDuplicates:
    """Tests for handle_get_duplicates standalone function."""

    @pytest.mark.asyncio
    async def test_no_analyses(self):
        result = await handle_get_duplicates(repo_id="empty")
        assert result["success"] is False
        assert "No analyses found" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_duplicates_above_min_lines(self):
        d1 = _make_duplicate(hash_val="short", lines=3)
        d2 = _make_duplicate(hash_val="long_enough", lines=10)
        report = _make_report(scan_id="dup-1", duplicates=[d1, d2])
        _metrics_reports["repo"] = {"dup-1": report}

        result = await handle_get_duplicates(repo_id="repo", min_lines=6)
        assert result["success"] is True
        assert len(result["duplicates"]) == 1
        assert result["duplicates"][0]["hash"] == "long_eno"  # hash[:8]

    @pytest.mark.asyncio
    async def test_limit(self):
        dups = [_make_duplicate(hash_val=f"hash{i:04d}", lines=10 + i) for i in range(10)]
        report = _make_report(scan_id="dup-lim", duplicates=dups)
        _metrics_reports["repo"] = {"dup-lim": report}

        result = await handle_get_duplicates(repo_id="repo", limit=3)
        assert result["success"] is True
        assert len(result["duplicates"]) == 3

    @pytest.mark.asyncio
    async def test_sorted_by_lines_times_occurrences(self):
        d_small = _make_duplicate(
            hash_val="small00000000000", lines=5, occurrences=[("a.py", 1, 5)]
        )
        d_large = _make_duplicate(
            hash_val="large00000000000",
            lines=20,
            occurrences=[("a.py", 1, 20), ("b.py", 1, 20), ("c.py", 1, 20)],
        )
        report = _make_report(scan_id="dup-sort", duplicates=[d_small, d_large])
        _metrics_reports["repo"] = {"dup-sort": report}

        result = await handle_get_duplicates(repo_id="repo", min_lines=1)
        assert result["success"] is True
        # d_large: 20*3=60 > d_small: 5*1=5
        assert result["duplicates"][0]["lines"] == 20

    @pytest.mark.asyncio
    async def test_occurrences_format(self):
        dup = _make_duplicate(
            hash_val="occ_test12345678",
            lines=8,
            occurrences=[("src/a.py", 10, 18), ("src/b.py", 30, 38)],
        )
        report = _make_report(scan_id="dup-occ", duplicates=[dup])
        _metrics_reports["repo"] = {"dup-occ": report}

        result = await handle_get_duplicates(repo_id="repo")
        d = result["duplicates"][0]
        assert d["hash"] == "occ_test"
        assert len(d["occurrences"]) == 2
        assert d["occurrences"][0] == {"file": "src/a.py", "start": 10, "end": 18}

    @pytest.mark.asyncio
    async def test_total_count(self):
        dups = [_make_duplicate(hash_val=f"h{i:015d}", lines=10) for i in range(7)]
        report = _make_report(scan_id="dup-ct", duplicates=dups)
        _metrics_reports["repo"] = {"dup-ct": report}

        result = await handle_get_duplicates(repo_id="repo")
        assert result["total"] == 7

    @pytest.mark.asyncio
    async def test_empty_duplicates(self):
        report = _make_report(scan_id="dup-empty", duplicates=[])
        _metrics_reports["repo"] = {"dup-empty": report}

        result = await handle_get_duplicates(repo_id="repo")
        assert result["success"] is True
        assert result["duplicates"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_min_lines_filters_all(self):
        dups = [_make_duplicate(lines=5)]
        report = _make_report(scan_id="dup-fa", duplicates=dups)
        _metrics_reports["repo"] = {"dup-fa": report}

        result = await handle_get_duplicates(repo_id="repo", min_lines=100)
        assert result["success"] is True
        assert result["duplicates"] == []


# ============================================================================
# Tests: handle_get_file_metrics
# ============================================================================


class TestHandleGetFileMetrics:
    """Tests for handle_get_file_metrics standalone function."""

    @pytest.mark.asyncio
    async def test_no_analyses(self):
        result = await handle_get_file_metrics(repo_id="empty", file_path="any.py")
        assert result["success"] is False
        assert "No analyses found" in result["error"]

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        report = _make_report(
            scan_id="fm-1",
            files=[_make_file_metrics(file_path="src/other.py")],
        )
        _metrics_reports["repo"] = {"fm-1": report}

        result = await handle_get_file_metrics(repo_id="repo", file_path="src/missing.py")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_exact_match(self):
        fm = _make_file_metrics(file_path="src/main.py", lines_of_code=200)
        report = _make_report(scan_id="fm-exact", files=[fm])
        _metrics_reports["repo"] = {"fm-exact": report}

        result = await handle_get_file_metrics(repo_id="repo", file_path="src/main.py")
        assert result["success"] is True
        assert result["file"]["file_path"] == "src/main.py"
        assert result["file"]["lines_of_code"] == 200

    @pytest.mark.asyncio
    async def test_suffix_match(self):
        fm = _make_file_metrics(file_path="/full/path/to/main.py")
        report = _make_report(scan_id="fm-suffix", files=[fm])
        _metrics_reports["repo"] = {"fm-suffix": report}

        result = await handle_get_file_metrics(repo_id="repo", file_path="main.py")
        assert result["success"] is True
        assert result["file"]["file_path"] == "/full/path/to/main.py"

    @pytest.mark.asyncio
    async def test_file_response_fields(self):
        func = _make_function_metrics(
            name="compute",
            start_line=5,
            end_line=25,
            lines_of_code=20,
            cyclomatic_complexity=7,
            cognitive_complexity=4,
            parameter_count=3,
            nested_depth=2,
        )
        fm = _make_file_metrics(
            file_path="src/app.py",
            language="python",
            lines_of_code=150,
            lines_of_comments=30,
            blank_lines=15,
            classes=2,
            imports=8,
            avg_complexity=4.5,
            max_complexity=10,
            maintainability_index=68.0,
            functions=[func],
        )
        report = _make_report(scan_id="fm-fields", files=[fm])
        _metrics_reports["repo"] = {"fm-fields": report}

        result = await handle_get_file_metrics(repo_id="repo", file_path="src/app.py")
        assert result["success"] is True
        f = result["file"]
        assert f["language"] == "python"
        assert f["lines_of_code"] == 150
        assert f["lines_of_comments"] == 30
        assert f["blank_lines"] == 15
        assert f["classes"] == 2
        assert f["imports"] == 8
        assert f["avg_complexity"] == 4.5
        assert f["max_complexity"] == 10
        assert f["maintainability_index"] == 68.0

        # Check function fields
        assert len(f["functions"]) == 1
        fn = f["functions"][0]
        assert fn["name"] == "compute"
        assert fn["start_line"] == 5
        assert fn["end_line"] == 25
        assert fn["lines_of_code"] == 20
        assert fn["cyclomatic_complexity"] == 7
        assert fn["cognitive_complexity"] == 4
        assert fn["parameter_count"] == 3
        assert fn["nested_depth"] == 2

    @pytest.mark.asyncio
    async def test_analysis_id_returned(self):
        fm = _make_file_metrics(file_path="src/main.py")
        report = _make_report(scan_id="fm-aid", files=[fm])
        _metrics_reports["repo"] = {"fm-aid": report}

        result = await handle_get_file_metrics(repo_id="repo", file_path="src/main.py")
        assert result["analysis_id"] == "fm-aid"

    @pytest.mark.asyncio
    async def test_multiple_files_first_match(self):
        f1 = _make_file_metrics(file_path="src/main.py")
        f2 = _make_file_metrics(file_path="tests/main.py")
        report = _make_report(scan_id="fm-multi", files=[f1, f2])
        _metrics_reports["repo"] = {"fm-multi": report}

        result = await handle_get_file_metrics(repo_id="repo", file_path="src/main.py")
        assert result["success"] is True
        assert result["file"]["file_path"] == "src/main.py"


# ============================================================================
# Tests: handle_list_analyses
# ============================================================================


class TestHandleListAnalyses:
    """Tests for handle_list_analyses standalone function."""

    @pytest.mark.asyncio
    async def test_empty_repo(self):
        result = await handle_list_analyses(repo_id="empty-repo")
        assert result["success"] is True
        assert result["analyses"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_analyses_sorted_by_time_desc(self):
        r1 = _make_report(
            scan_id="r1",
            scanned_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            total_files=5,
        )
        r2 = _make_report(
            scan_id="r2",
            scanned_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            total_files=10,
        )
        _metrics_reports["repo"] = {"r1": r1, "r2": r2}

        result = await handle_list_analyses(repo_id="repo")
        assert result["success"] is True
        assert len(result["analyses"]) == 2
        assert result["analyses"][0]["analysis_id"] == "r2"
        assert result["analyses"][1]["analysis_id"] == "r1"

    @pytest.mark.asyncio
    async def test_pagination_limit(self):
        for i in range(5):
            r = _make_report(
                scan_id=f"r{i}",
                scanned_at=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
            )
            _metrics_reports.setdefault("repo", {})[f"r{i}"] = r

        result = await handle_list_analyses(repo_id="repo", limit=2)
        assert result["success"] is True
        assert len(result["analyses"]) == 2
        assert result["total"] == 5
        assert result["limit"] == 2

    @pytest.mark.asyncio
    async def test_pagination_offset(self):
        for i in range(5):
            r = _make_report(
                scan_id=f"r{i}",
                scanned_at=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
            )
            _metrics_reports.setdefault("repo", {})[f"r{i}"] = r

        result = await handle_list_analyses(repo_id="repo", limit=2, offset=3)
        assert result["success"] is True
        assert len(result["analyses"]) == 2
        assert result["offset"] == 3

    @pytest.mark.asyncio
    async def test_pagination_offset_past_end(self):
        r = _make_report(scan_id="r0")
        _metrics_reports["repo"] = {"r0": r}

        result = await handle_list_analyses(repo_id="repo", offset=10)
        assert result["success"] is True
        assert result["analyses"] == []
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_summary_fields(self):
        hotspots = [_make_hotspot() for _ in range(3)]
        dups = [_make_duplicate() for _ in range(2)]
        report = _make_report(
            scan_id="s1",
            total_files=42,
            total_lines=5000,
            avg_complexity=7.123,
            max_complexity=25,
            maintainability_index=65.456,
            hotspots=hotspots,
            duplicates=dups,
        )
        _metrics_reports["repo"] = {"s1": report}

        result = await handle_list_analyses(repo_id="repo")
        s = result["analyses"][0]["summary"]
        assert s["total_files"] == 42
        assert s["total_lines"] == 5000
        assert s["avg_complexity"] == 7.12  # rounded to 2 decimal places
        assert s["max_complexity"] == 25
        assert s["maintainability_index"] == 65.46
        assert s["hotspot_count"] == 3
        assert s["duplicate_count"] == 2


# ============================================================================
# Tests: MetricsHandler.can_handle
# ============================================================================


class TestCanHandle:
    """Tests for MetricsHandler.can_handle."""

    def test_metrics_path(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/metrics") is True

    def test_metrics_analyze(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/metrics/analyze") is True

    def test_metrics_file_path(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/metrics/file/src/main.py") is True

    def test_metrics_history(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/metrics/history") is True

    def test_hotspots(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/hotspots") is True

    def test_duplicates(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/duplicates") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates/abc") is False

    def test_codebase_without_metrics(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/scans") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/codebase") is False


# ============================================================================
# Tests: MetricsHandler.handle (RBAC)
# ============================================================================


class TestHandleRBAC:
    """Tests for MetricsHandler.handle RBAC enforcement."""

    @pytest.mark.asyncio
    async def test_authenticated_returns_none(self, handler, mock_http):
        """Authenticated user passes through (returns None for further routing)."""
        result = await handler.handle("/api/v1/codebase/repo/metrics", {}, mock_http)
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_unauthenticated_returns_401(self, handler, mock_http):
        """Unauthenticated request returns 401."""
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        original = SecureHandler.get_auth_context

        async def raise_unauth(self, req, require_auth=True):
            raise UnauthorizedError("No token")

        try:
            SecureHandler.get_auth_context = raise_unauth
            result = await handler.handle("/api/v1/codebase/repo/metrics", {}, mock_http)
            assert _status(result) == 401
            assert "Authentication required" in _body(result).get("error", "")
        finally:
            SecureHandler.get_auth_context = original

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_forbidden_returns_403(self, handler, mock_http):
        """Authenticated but lacking permission returns 403."""
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError

        original_get = SecureHandler.get_auth_context
        original_check = SecureHandler.check_permission

        mock_ctx = MagicMock()

        async def mock_get(self, req, require_auth=True):
            return mock_ctx

        def mock_check(self, ctx, perm):
            raise ForbiddenError(f"Missing {perm}")

        try:
            SecureHandler.get_auth_context = mock_get
            SecureHandler.check_permission = mock_check
            result = await handler.handle("/api/v1/codebase/repo/metrics", {}, mock_http)
            assert _status(result) == 403
            assert "Permission denied" in _body(result).get("error", "")
        finally:
            SecureHandler.get_auth_context = original_get
            SecureHandler.check_permission = original_check


# ============================================================================
# Tests: MetricsHandler.handle_post_analyze
# ============================================================================


class TestHandlePostAnalyze:
    """Tests for MetricsHandler.handle_post_analyze."""

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, handler):
        result = await handler.handle_post_analyze(data={}, repo_id="repo")
        assert _status(result) == 400
        assert "repo_path required" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_success(self, handler):
        data = {"repo_path": "/tmp/test-repo"}
        result = await handler.handle_post_analyze(data=data, repo_id="repo")
        assert _status(result) == 200
        body = _body(result)
        assert body.get("success") is True or body.get("data", {}).get("success") is True

    @pytest.mark.asyncio
    async def test_with_all_options(self, handler):
        data = {
            "repo_path": "/tmp/test-repo",
            "include_patterns": ["src/**/*.py"],
            "exclude_patterns": ["**/tests/**"],
            "complexity_warning": 5,
            "complexity_error": 15,
            "duplication_threshold": 3,
            "workspace_id": "ws-1",
        }
        result = await handler.handle_post_analyze(data=data, repo_id="repo")
        assert _status(result) == 200

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_rbac_unauthenticated(self, handler, mock_http):
        """POST with unauthenticated handler returns 401."""
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        original = SecureHandler.get_auth_context

        async def raise_unauth(self, req, require_auth=True):
            raise UnauthorizedError("No token")

        try:
            SecureHandler.get_auth_context = raise_unauth
            data = {"repo_path": "/tmp/test"}
            result = await handler.handle_post_analyze(data=data, repo_id="repo", handler=mock_http)
            assert _status(result) == 401
        finally:
            SecureHandler.get_auth_context = original

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_rbac_forbidden(self, handler, mock_http):
        """POST without permission returns 403."""
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError

        original_get = SecureHandler.get_auth_context
        original_check = SecureHandler.check_permission

        async def mock_get(self, req, require_auth=True):
            return MagicMock()

        def mock_check(self, ctx, perm):
            raise ForbiddenError("Nope")

        try:
            SecureHandler.get_auth_context = mock_get
            SecureHandler.check_permission = mock_check
            data = {"repo_path": "/tmp/test"}
            result = await handler.handle_post_analyze(data=data, repo_id="repo", handler=mock_http)
            assert _status(result) == 403
        finally:
            SecureHandler.get_auth_context = original_get
            SecureHandler.check_permission = original_check

    @pytest.mark.asyncio
    async def test_no_handler_skips_rbac(self, handler):
        """When handler kwarg is None, RBAC check is skipped."""
        data = {"repo_path": "/tmp/test-repo"}
        result = await handler.handle_post_analyze(data=data, repo_id="repo", handler=None)
        assert _status(result) == 200


# ============================================================================
# Tests: MetricsHandler.handle_get_metrics
# ============================================================================


class TestHandlerGetMetrics:
    """Tests for MetricsHandler.handle_get_metrics."""

    @pytest.mark.asyncio
    async def test_no_analyses(self, handler):
        result = await handler.handle_get_metrics(params={}, repo_id="empty")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_with_data(self, handler):
        report = _make_report(scan_id="hgm-1")
        _metrics_reports["repo"] = {"hgm-1": report}

        result = await handler.handle_get_metrics(params={}, repo_id="repo")
        assert _status(result) == 200


# ============================================================================
# Tests: MetricsHandler.handle_get_metrics_by_id
# ============================================================================


class TestHandlerGetMetricsById:
    """Tests for MetricsHandler.handle_get_metrics_by_id."""

    @pytest.mark.asyncio
    async def test_found(self, handler):
        report = _make_report(scan_id="byid-1")
        _metrics_reports["repo"] = {"byid-1": report}

        result = await handler.handle_get_metrics_by_id(
            params={}, repo_id="repo", analysis_id="byid-1"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_not_found(self, handler):
        _metrics_reports["repo"] = {}
        result = await handler.handle_get_metrics_by_id(
            params={}, repo_id="repo", analysis_id="nonexistent"
        )
        assert _status(result) == 404


# ============================================================================
# Tests: MetricsHandler.handle_get_hotspots
# ============================================================================


class TestHandlerGetHotspots:
    """Tests for MetricsHandler.handle_get_hotspots."""

    @pytest.mark.asyncio
    async def test_no_analyses(self, handler):
        result = await handler.handle_get_hotspots(params={}, repo_id="empty")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_with_data(self, handler):
        report = _make_report(scan_id="hh-1", hotspots=[_make_hotspot()])
        _metrics_reports["repo"] = {"hh-1": report}

        result = await handler.handle_get_hotspots(params={}, repo_id="repo")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_query_params(self, handler):
        hotspots = [_make_hotspot(complexity=float(i + 5)) for i in range(10)]
        report = _make_report(scan_id="hh-qp", hotspots=hotspots)
        _metrics_reports["repo"] = {"hh-qp": report}

        result = await handler.handle_get_hotspots(
            params={"min_complexity": "8", "limit": "3"}, repo_id="repo"
        )
        assert _status(result) == 200


# ============================================================================
# Tests: MetricsHandler.handle_get_duplicates
# ============================================================================


class TestHandlerGetDuplicates:
    """Tests for MetricsHandler.handle_get_duplicates."""

    @pytest.mark.asyncio
    async def test_no_analyses(self, handler):
        result = await handler.handle_get_duplicates(params={}, repo_id="empty")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_with_data(self, handler):
        report = _make_report(scan_id="hd-1", duplicates=[_make_duplicate()])
        _metrics_reports["repo"] = {"hd-1": report}

        result = await handler.handle_get_duplicates(params={}, repo_id="repo")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_query_params(self, handler):
        dups = [_make_duplicate(hash_val=f"h{i:015d}", lines=10 + i) for i in range(10)]
        report = _make_report(scan_id="hd-qp", duplicates=dups)
        _metrics_reports["repo"] = {"hd-qp": report}

        result = await handler.handle_get_duplicates(
            params={"min_lines": "12", "limit": "3"}, repo_id="repo"
        )
        assert _status(result) == 200


# ============================================================================
# Tests: MetricsHandler.handle_get_file_metrics
# ============================================================================


class TestHandlerGetFileMetrics:
    """Tests for MetricsHandler.handle_get_file_metrics."""

    @pytest.mark.asyncio
    async def test_no_analyses(self, handler):
        result = await handler.handle_get_file_metrics(
            params={}, repo_id="empty", file_path="main.py"
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_found(self, handler):
        fm = _make_file_metrics(file_path="src/main.py")
        report = _make_report(scan_id="hfm-1", files=[fm])
        _metrics_reports["repo"] = {"hfm-1": report}

        result = await handler.handle_get_file_metrics(
            params={}, repo_id="repo", file_path="src/main.py"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_not_found(self, handler):
        fm = _make_file_metrics(file_path="src/main.py")
        report = _make_report(scan_id="hfm-nf", files=[fm])
        _metrics_reports["repo"] = {"hfm-nf": report}

        result = await handler.handle_get_file_metrics(
            params={}, repo_id="repo", file_path="missing.py"
        )
        assert _status(result) == 404


# ============================================================================
# Tests: MetricsHandler.handle_list_analyses
# ============================================================================


class TestHandlerListAnalyses:
    """Tests for MetricsHandler.handle_list_analyses."""

    @pytest.mark.asyncio
    async def test_empty(self, handler):
        result = await handler.handle_list_analyses(params={}, repo_id="empty")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_with_data(self, handler):
        for i in range(3):
            r = _make_report(
                scan_id=f"la{i}",
                scanned_at=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
            )
            _metrics_reports.setdefault("repo", {})[f"la{i}"] = r

        result = await handler.handle_list_analyses(params={}, repo_id="repo")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_query_params(self, handler):
        for i in range(5):
            r = _make_report(
                scan_id=f"la-qp{i}",
                scanned_at=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
            )
            _metrics_reports.setdefault("repo", {})[f"la-qp{i}"] = r

        result = await handler.handle_list_analyses(
            params={"limit": "2", "offset": "1"}, repo_id="repo"
        )
        assert _status(result) == 200


# ============================================================================
# Tests: MetricsHandler._get_user_id
# ============================================================================


class TestGetUserId:
    """Tests for MetricsHandler._get_user_id."""

    def test_returns_user_id_from_context(self):
        ctx_mock = MagicMock()
        ctx_mock.user_id = "user-abc"
        handler = MetricsHandler({"auth_context": ctx_mock})
        assert handler._get_user_id() == "user-abc"

    def test_returns_default_when_no_context(self):
        handler = MetricsHandler({})
        assert handler._get_user_id() == "default"

    def test_returns_default_when_no_user_id_attr(self):
        ctx_mock = MagicMock(spec=[])  # no attributes
        handler = MetricsHandler({"auth_context": ctx_mock})
        assert handler._get_user_id() == "default"


# ============================================================================
# Tests: Permission constants
# ============================================================================


class TestPermissionConstants:
    """Tests for permission constants."""

    def test_read_permission(self):
        assert METRICS_READ_PERMISSION == "codebase:metrics:read"

    def test_analyze_permission(self):
        assert METRICS_ANALYZE_PERMISSION == "codebase:metrics:analyze"


# ============================================================================
# Tests: Edge cases and error handling
# ============================================================================


class TestEdgeCases:
    """Additional edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_handle_get_metrics_internal_error(self):
        """Simulate an internal error in handle_get_metrics."""
        # Create a report-like object whose to_dict raises
        bad_report = MagicMock()
        bad_report.scanned_at = datetime.now(timezone.utc)
        bad_report.to_dict = MagicMock(side_effect=TypeError("boom"))
        _metrics_reports["repo"] = {"bad": bad_report}

        result = await handle_get_metrics(repo_id="repo")
        assert result["success"] is False
        assert "Internal server error" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_handle_get_hotspots_internal_error(self):
        """Simulate internal error in handle_get_hotspots."""
        bad_report = MagicMock()
        bad_report.scanned_at = datetime.now(timezone.utc)
        bad_report.hotspots = MagicMock()
        bad_report.hotspots.__iter__ = MagicMock(side_effect=AttributeError("oops"))
        _metrics_reports["repo"] = {"bad": bad_report}

        result = await handle_get_hotspots(repo_id="repo")
        assert result["success"] is False
        assert "Internal server error" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_handle_get_duplicates_internal_error(self):
        """Simulate internal error in handle_get_duplicates."""
        bad_report = MagicMock()
        bad_report.scanned_at = datetime.now(timezone.utc)
        bad_report.duplicates = MagicMock()
        bad_report.duplicates.__iter__ = MagicMock(side_effect=ValueError("crash"))
        _metrics_reports["repo"] = {"bad": bad_report}

        result = await handle_get_duplicates(repo_id="repo")
        assert result["success"] is False
        assert "Internal server error" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_handle_get_file_metrics_internal_error(self):
        """Simulate internal error in handle_get_file_metrics."""
        bad_report = MagicMock()
        bad_report.scanned_at = datetime.now(timezone.utc)
        bad_report.files = MagicMock()
        bad_report.files.__iter__ = MagicMock(side_effect=KeyError("kaboom"))
        _metrics_reports["repo"] = {"bad": bad_report}

        result = await handle_get_file_metrics(repo_id="repo", file_path="x.py")
        assert result["success"] is False
        assert "Internal server error" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_handle_list_analyses_internal_error(self):
        """Simulate internal error in handle_list_analyses."""
        bad_report = MagicMock()
        bad_report.scanned_at = MagicMock()
        bad_report.scanned_at.__lt__ = MagicMock(side_effect=TypeError("compare fail"))
        _metrics_reports["repo"] = {"bad": bad_report, "bad2": bad_report}

        result = await handle_list_analyses(repo_id="repo")
        assert result["success"] is False
        assert "Internal server error" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_handle_analyze_metrics_outer_exception(self):
        """Test that outer exception in handle_analyze_metrics is caught."""
        # Patch _get_or_create_repo_metrics to raise
        with patch(
            "aragora.server.handlers.codebase.metrics._get_or_create_repo_metrics",
            side_effect=ValueError("storage error"),
        ):
            result = await handle_analyze_metrics(
                repo_path="/tmp/test",
                repo_id="repo",
            )
            assert result["success"] is False
            assert "Internal server error" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_handle_get_metrics_single_report(self):
        """Single report is returned as latest."""
        report = _make_report(scan_id="only-one", total_files=99)
        _metrics_reports["repo"] = {"only-one": report}

        result = await handle_get_metrics(repo_id="repo")
        assert result["success"] is True
        assert result["report"]["scan_id"] == "only-one"

    @pytest.mark.asyncio
    async def test_handler_route_prefixes(self, handler):
        """Verify ROUTE_PREFIXES contains expected value."""
        assert "/api/v1/codebase/" in handler.ROUTE_PREFIXES

    def test_can_handle_metrics_in_path(self, handler):
        """Paths with /metrics segment under codebase prefix are handled."""
        assert handler.can_handle("/api/v1/codebase/r/metrics/analyze") is True

    def test_can_handle_hotspots_in_path(self, handler):
        """Paths with /hotspots segment under codebase prefix are handled."""
        assert handler.can_handle("/api/v1/codebase/r/hotspots") is True

    def test_can_handle_duplicates_in_path(self, handler):
        """Paths with /duplicates segment under codebase prefix are handled."""
        assert handler.can_handle("/api/v1/codebase/r/duplicates") is True

    def test_can_handle_no_match_non_metrics(self, handler):
        """Non-metrics codebase paths are not handled."""
        assert handler.can_handle("/api/v1/codebase/r/reviews") is False

    @pytest.mark.asyncio
    async def test_get_latest_picks_most_recent(self):
        """Multiple reports return the one with latest scanned_at."""
        r_old = _make_report(
            scan_id="old",
            scanned_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        r_mid = _make_report(
            scan_id="mid",
            scanned_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        r_new = _make_report(
            scan_id="new",
            scanned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        _metrics_reports["repo"] = {"old": r_old, "mid": r_mid, "new": r_new}

        result = await handle_get_metrics(repo_id="repo")
        assert result["report"]["scan_id"] == "new"

    @pytest.mark.asyncio
    async def test_duplicates_hash_truncated(self):
        """Duplicate hashes are truncated to 8 chars."""
        dup = _make_duplicate(hash_val="0123456789abcdef", lines=10)
        report = _make_report(scan_id="trunc", duplicates=[dup])
        _metrics_reports["repo"] = {"trunc": report}

        result = await handle_get_duplicates(repo_id="repo")
        assert result["duplicates"][0]["hash"] == "01234567"

    @pytest.mark.asyncio
    async def test_list_analyses_scanned_at_isoformat(self):
        """Listed analyses have scanned_at in ISO format."""
        dt = datetime(2025, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        report = _make_report(scan_id="iso-test", scanned_at=dt)
        _metrics_reports["repo"] = {"iso-test": report}

        result = await handle_list_analyses(repo_id="repo")
        assert result["analyses"][0]["scanned_at"] == dt.isoformat()
