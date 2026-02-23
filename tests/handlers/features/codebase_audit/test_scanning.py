"""Comprehensive tests for the Codebase Audit scanning module.

Tests the scanning logic in aragora/server/handlers/features/codebase_audit/scanning.py:
- Path validation (_validate_scan_path)
- Language validation (_validate_languages)
- In-memory storage helpers (_get_tenant_scans, _get_tenant_findings)
- Scanner functions (run_sast_scan, run_bug_scan, run_secrets_scan,
  run_dependency_scan, run_metrics_analysis)
- Severity mapping (_map_severity)
- Mock data generators (demo mode fallback data)
- ScanPathError exception
- Constants (_ALLOWED_LANGUAGES, _MAX_PATH_LENGTH)
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.codebase_audit.scanning import (
    ScanPathError,
    _ALLOWED_LANGUAGES,
    _MAX_PATH_LENGTH,
    _finding_store,
    _get_mock_bug_findings,
    _get_mock_dependency_findings,
    _get_mock_metrics,
    _get_mock_sast_findings,
    _get_mock_secrets_findings,
    _get_tenant_findings,
    _get_tenant_scans,
    _map_severity,
    _scan_store,
    _validate_languages,
    _validate_scan_path,
    run_bug_scan,
    run_dependency_scan,
    run_metrics_analysis,
    run_sast_scan,
    run_secrets_scan,
)
from aragora.server.handlers.features.codebase_audit.rules import (
    Finding,
    FindingSeverity,
    ScanType,
)

# ---------------------------------------------------------------------------
# Module-path constants for patching scanner imports
# ---------------------------------------------------------------------------

_SAST_MODULE = "aragora.analysis.codebase.sast_scanner"
_BUG_MODULE = "aragora.analysis.codebase.bug_detector"
_SECRETS_MODULE = "aragora.analysis.codebase.secrets_scanner"
_DEP_MODULE = "aragora.analysis.codebase.scanner"
_METRICS_MODULE = "aragora.analysis.codebase.metrics"


def _block_import(module_name: str):
    """Return a patch.dict that makes `import module_name` raise ImportError.

    This forces the scanner function to take its fallback path (mock data).
    We temporarily remove the real module from sys.modules and insert a
    sentinel that triggers ImportError.
    """
    sentinel = MagicMock()
    sentinel.__bool__ = lambda self: True  # noqa: E731

    # Set the module entry to None so `from X import Y` raises ImportError
    return patch.dict("sys.modules", {module_name: None})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_stores():
    """Clear in-memory stores between tests."""
    _scan_store.clear()
    _finding_store.clear()
    yield
    _scan_store.clear()
    _finding_store.clear()


# ===========================================================================
# 1. ScanPathError
# ===========================================================================


class TestScanPathError:
    """Tests for the ScanPathError exception."""

    def test_is_value_error_subclass(self):
        assert issubclass(ScanPathError, ValueError)

    def test_message_preserved(self):
        err = ScanPathError("bad path")
        assert str(err) == "bad path"

    def test_can_be_caught_as_value_error(self):
        with pytest.raises(ValueError, match="test"):
            raise ScanPathError("test")


# ===========================================================================
# 2. _validate_scan_path -- valid paths
# ===========================================================================


class TestValidateScanPathValid:
    """Tests for _validate_scan_path with valid inputs."""

    def test_current_dir(self):
        assert _validate_scan_path(".") == "."

    def test_simple_relative_path(self):
        assert _validate_scan_path("src") == "src"

    def test_nested_relative_path(self):
        assert _validate_scan_path("src/utils/helpers") == "src/utils/helpers"

    def test_path_with_underscores(self):
        assert _validate_scan_path("my_project/sub_dir") == "my_project/sub_dir"

    def test_path_with_hyphens(self):
        assert _validate_scan_path("my-project/sub-dir") == "my-project/sub-dir"

    def test_path_with_dots_in_filename(self):
        assert _validate_scan_path("src/file.py") == "src/file.py"

    def test_whitespace_stripped(self):
        assert _validate_scan_path("  src  ") == "src"

    def test_path_with_numbers(self):
        assert _validate_scan_path("v2/module3") == "v2/module3"


# ===========================================================================
# 3. _validate_scan_path -- invalid paths
# ===========================================================================


class TestValidateScanPathInvalid:
    """Tests for _validate_scan_path rejecting invalid inputs."""

    def test_empty_string(self):
        with pytest.raises(ScanPathError, match="cannot be empty"):
            _validate_scan_path("")

    def test_whitespace_only(self):
        with pytest.raises(ScanPathError, match="cannot be empty"):
            _validate_scan_path("   ")

    def test_non_string_type(self):
        with pytest.raises(ScanPathError, match="must be a string"):
            _validate_scan_path(123)  # type: ignore[arg-type]

    def test_none_type(self):
        with pytest.raises(ScanPathError, match="must be a string"):
            _validate_scan_path(None)  # type: ignore[arg-type]

    def test_exceeds_max_length(self):
        long_path = "a" * (_MAX_PATH_LENGTH + 1)
        with pytest.raises(ScanPathError, match="exceeds maximum length"):
            _validate_scan_path(long_path)

    def test_exactly_max_length_does_not_raise_length_error(self):
        path = "a" * _MAX_PATH_LENGTH
        # Should not raise the length error (may raise a different one)
        try:
            _validate_scan_path(path)
        except ScanPathError as e:
            assert "exceeds maximum length" not in str(e)

    def test_null_byte(self):
        with pytest.raises(ScanPathError, match="null byte"):
            _validate_scan_path("src/\x00evil")

    def test_semicolon_injection(self):
        with pytest.raises(ScanPathError, match="disallowed character"):
            _validate_scan_path("src; rm -rf /")

    def test_pipe_injection(self):
        with pytest.raises(ScanPathError, match="disallowed character"):
            _validate_scan_path("src | cat /etc/passwd")

    def test_ampersand_injection(self):
        with pytest.raises(ScanPathError, match="disallowed character"):
            _validate_scan_path("src & whoami")

    def test_dollar_sign_injection(self):
        with pytest.raises(ScanPathError, match="disallowed character"):
            _validate_scan_path("src/$HOME")

    def test_backtick_injection(self):
        with pytest.raises(ScanPathError, match="disallowed character"):
            _validate_scan_path("src/`whoami`")

    def test_parentheses_injection(self):
        with pytest.raises(ScanPathError, match="disallowed character"):
            _validate_scan_path("src/(evil)")

    def test_curly_braces_injection(self):
        with pytest.raises(ScanPathError, match="disallowed character"):
            _validate_scan_path("src/{evil}")

    def test_angle_brackets_injection(self):
        with pytest.raises(ScanPathError, match="disallowed character"):
            _validate_scan_path("src/<evil>")

    def test_newline_injection(self):
        with pytest.raises(ScanPathError, match="disallowed character"):
            _validate_scan_path("src\nrm -rf /")

    def test_carriage_return_injection(self):
        with pytest.raises(ScanPathError, match="disallowed character"):
            _validate_scan_path("src\r\nevil")

    def test_home_expansion(self):
        with pytest.raises(ScanPathError, match="Home directory expansion"):
            _validate_scan_path("~/secrets")

    def test_home_expansion_with_user(self):
        with pytest.raises(ScanPathError, match="Home directory expansion"):
            _validate_scan_path("~root/.ssh")

    def test_absolute_path_unix(self):
        with pytest.raises(ScanPathError, match="Absolute paths are not allowed"):
            _validate_scan_path("/etc/passwd")

    def test_absolute_path_root(self):
        with pytest.raises(ScanPathError, match="Absolute paths are not allowed"):
            _validate_scan_path("/")

    def test_traversal_outside_workspace(self):
        with pytest.raises(ScanPathError, match="resolves outside"):
            _validate_scan_path("../../../etc/passwd")


# ===========================================================================
# 4. _validate_languages
# ===========================================================================


class TestValidateLanguages:
    """Tests for _validate_languages."""

    def test_none_returns_none(self):
        assert _validate_languages(None) is None

    def test_empty_list_returns_none(self):
        assert _validate_languages([]) is None

    def test_single_valid_language(self):
        result = _validate_languages(["python"])
        assert result == ["python"]

    def test_multiple_valid_languages(self):
        result = _validate_languages(["python", "javascript", "go"])
        assert result == ["python", "javascript", "go"]

    def test_language_normalized_to_lowercase(self):
        result = _validate_languages(["Python", "JAVASCRIPT"])
        assert result == ["python", "javascript"]

    def test_language_whitespace_stripped(self):
        result = _validate_languages(["  python  ", " go "])
        assert result == ["python", "go"]

    def test_invalid_language_raises(self):
        with pytest.raises(ScanPathError, match="Unsupported language"):
            _validate_languages(["brainfuck"])

    def test_non_string_element_raises(self):
        with pytest.raises(ScanPathError, match="Invalid language identifier"):
            _validate_languages([123])  # type: ignore[list-item]

    def test_non_list_raises(self):
        with pytest.raises(ScanPathError, match="must be a list"):
            _validate_languages("python")  # type: ignore[arg-type]

    def test_mixed_valid_invalid_raises(self):
        with pytest.raises(ScanPathError, match="Unsupported language"):
            _validate_languages(["python", "nonexistent"])

    def test_all_allowed_languages_accepted(self):
        """Every language in _ALLOWED_LANGUAGES should pass validation."""
        result = _validate_languages(list(_ALLOWED_LANGUAGES))
        assert set(result) == _ALLOWED_LANGUAGES


# ===========================================================================
# 5. Allowed languages constant
# ===========================================================================


class TestAllowedLanguages:
    """Tests for the _ALLOWED_LANGUAGES constant."""

    def test_is_frozenset(self):
        assert isinstance(_ALLOWED_LANGUAGES, frozenset)

    def test_contains_python(self):
        assert "python" in _ALLOWED_LANGUAGES

    def test_contains_javascript(self):
        assert "javascript" in _ALLOWED_LANGUAGES

    def test_contains_typescript(self):
        assert "typescript" in _ALLOWED_LANGUAGES

    def test_contains_go(self):
        assert "go" in _ALLOWED_LANGUAGES

    def test_contains_rust(self):
        assert "rust" in _ALLOWED_LANGUAGES

    def test_contains_java(self):
        assert "java" in _ALLOWED_LANGUAGES

    def test_all_lowercase(self):
        for lang in _ALLOWED_LANGUAGES:
            assert lang == lang.lower(), f"Language {lang!r} is not lowercase"


# ===========================================================================
# 6. In-memory storage helpers
# ===========================================================================


class TestTenantStorage:
    """Tests for _get_tenant_scans and _get_tenant_findings."""

    def test_get_tenant_scans_creates_empty_dict(self):
        scans = _get_tenant_scans("new-tenant")
        assert scans == {}
        assert "new-tenant" in _scan_store

    def test_get_tenant_scans_returns_same_dict(self):
        s1 = _get_tenant_scans("t1")
        s2 = _get_tenant_scans("t1")
        assert s1 is s2

    def test_get_tenant_scans_isolated(self):
        s1 = _get_tenant_scans("t1")
        s2 = _get_tenant_scans("t2")
        s1["scan1"] = {"data": "a"}
        assert "scan1" not in s2

    def test_get_tenant_findings_creates_empty_dict(self):
        findings = _get_tenant_findings("new-tenant")
        assert findings == {}
        assert "new-tenant" in _finding_store

    def test_get_tenant_findings_returns_same_dict(self):
        f1 = _get_tenant_findings("t1")
        f2 = _get_tenant_findings("t1")
        assert f1 is f2

    def test_get_tenant_findings_isolated(self):
        f1 = _get_tenant_findings("t1")
        f2 = _get_tenant_findings("t2")
        f1["find1"] = MagicMock()
        assert "find1" not in f2


# ===========================================================================
# 7. _map_severity
# ===========================================================================


class TestMapSeverity:
    """Tests for the _map_severity helper."""

    def test_critical_string(self):
        assert _map_severity("critical") == FindingSeverity.CRITICAL

    def test_high_string(self):
        assert _map_severity("high") == FindingSeverity.HIGH

    def test_medium_string(self):
        assert _map_severity("medium") == FindingSeverity.MEDIUM

    def test_low_string(self):
        assert _map_severity("low") == FindingSeverity.LOW

    def test_info_string(self):
        assert _map_severity("info") == FindingSeverity.INFO

    def test_moderate_maps_to_medium(self):
        assert _map_severity("moderate") == FindingSeverity.MEDIUM

    def test_informational_maps_to_info(self):
        assert _map_severity("informational") == FindingSeverity.INFO

    def test_case_insensitive_critical(self):
        assert _map_severity("CRITICAL") == FindingSeverity.CRITICAL

    def test_case_insensitive_high(self):
        assert _map_severity("High") == FindingSeverity.HIGH

    def test_unknown_defaults_to_medium(self):
        assert _map_severity("unknown") == FindingSeverity.MEDIUM

    def test_empty_string_defaults_to_medium(self):
        assert _map_severity("") == FindingSeverity.MEDIUM

    def test_enum_with_value_attribute(self):
        """Enum-like objects with .value should be unwrapped."""

        class MockSev(Enum):
            CRITICAL = "critical"

        assert _map_severity(MockSev.CRITICAL) == FindingSeverity.CRITICAL

    def test_integer_severity_defaults_to_medium(self):
        assert _map_severity(42) == FindingSeverity.MEDIUM

    def test_none_defaults_to_medium(self):
        assert _map_severity(None) == FindingSeverity.MEDIUM

    def test_object_with_value_property(self):
        """Objects that have a .value attribute (like enums) get unwrapped."""

        class FakeSev:
            value = "high"

        assert _map_severity(FakeSev()) == FindingSeverity.HIGH


# ===========================================================================
# 8. Mock data generators
# ===========================================================================


class TestMockSASTFindings:
    """Tests for _get_mock_sast_findings."""

    def test_returns_list(self):
        result = _get_mock_sast_findings("scan1")
        assert isinstance(result, list)

    def test_returns_multiple_findings(self):
        result = _get_mock_sast_findings("scan1")
        assert len(result) == 3

    def test_findings_have_correct_scan_id(self):
        result = _get_mock_sast_findings("my_scan")
        for finding in result:
            assert finding.scan_id == "my_scan"

    def test_findings_are_sast_type(self):
        result = _get_mock_sast_findings("scan1")
        for finding in result:
            assert finding.scan_type == ScanType.SAST

    def test_findings_have_unique_ids(self):
        result = _get_mock_sast_findings("scan1")
        ids = [f.id for f in result]
        assert len(set(ids)) == len(ids)

    def test_findings_have_ids_with_sast_prefix(self):
        result = _get_mock_sast_findings("scan1")
        for finding in result:
            assert finding.id.startswith("sast_")

    def test_findings_have_severity(self):
        result = _get_mock_sast_findings("scan1")
        severities = {f.severity for f in result}
        assert FindingSeverity.CRITICAL in severities
        assert FindingSeverity.HIGH in severities

    def test_findings_have_cwe_ids(self):
        result = _get_mock_sast_findings("scan1")
        cwe_ids = [f.cwe_id for f in result if f.cwe_id]
        assert len(cwe_ids) > 0
        for cwe in cwe_ids:
            assert cwe.startswith("CWE-")

    def test_findings_have_owasp_category(self):
        result = _get_mock_sast_findings("scan1")
        owasp = [f.owasp_category for f in result if f.owasp_category]
        assert len(owasp) > 0

    def test_findings_have_remediation(self):
        result = _get_mock_sast_findings("scan1")
        for finding in result:
            assert finding.remediation is not None


class TestMockBugFindings:
    """Tests for _get_mock_bug_findings."""

    def test_returns_list(self):
        result = _get_mock_bug_findings("scan1")
        assert isinstance(result, list)

    def test_returns_findings(self):
        result = _get_mock_bug_findings("scan1")
        assert len(result) == 2

    def test_findings_are_bugs_type(self):
        result = _get_mock_bug_findings("scan1")
        for finding in result:
            assert finding.scan_type == ScanType.BUGS

    def test_findings_have_correct_scan_id(self):
        result = _get_mock_bug_findings("my_scan")
        for finding in result:
            assert finding.scan_id == "my_scan"

    def test_findings_have_bug_prefix(self):
        result = _get_mock_bug_findings("scan1")
        for finding in result:
            assert finding.id.startswith("bug_")


class TestMockSecretsFindings:
    """Tests for _get_mock_secrets_findings."""

    def test_returns_list(self):
        result = _get_mock_secrets_findings("scan1")
        assert isinstance(result, list)

    def test_returns_findings(self):
        result = _get_mock_secrets_findings("scan1")
        assert len(result) == 1

    def test_findings_are_secrets_type(self):
        result = _get_mock_secrets_findings("scan1")
        for finding in result:
            assert finding.scan_type == ScanType.SECRETS

    def test_findings_are_critical(self):
        result = _get_mock_secrets_findings("scan1")
        for finding in result:
            assert finding.severity == FindingSeverity.CRITICAL

    def test_findings_have_secret_prefix(self):
        result = _get_mock_secrets_findings("scan1")
        for finding in result:
            assert finding.id.startswith("secret_")

    def test_findings_have_remediation(self):
        result = _get_mock_secrets_findings("scan1")
        for finding in result:
            assert finding.remediation is not None


class TestMockDependencyFindings:
    """Tests for _get_mock_dependency_findings."""

    def test_returns_list(self):
        result = _get_mock_dependency_findings("scan1")
        assert isinstance(result, list)

    def test_returns_findings(self):
        result = _get_mock_dependency_findings("scan1")
        assert len(result) == 2

    def test_findings_are_dependencies_type(self):
        result = _get_mock_dependency_findings("scan1")
        for finding in result:
            assert finding.scan_type == ScanType.DEPENDENCIES

    def test_findings_have_dep_prefix(self):
        result = _get_mock_dependency_findings("scan1")
        for finding in result:
            assert finding.id.startswith("dep_")

    def test_findings_have_cwe_ids(self):
        result = _get_mock_dependency_findings("scan1")
        for finding in result:
            assert finding.cwe_id is not None


class TestMockMetrics:
    """Tests for _get_mock_metrics."""

    def test_returns_dict(self):
        result = _get_mock_metrics()
        assert isinstance(result, dict)

    def test_has_total_lines(self):
        result = _get_mock_metrics()
        assert "total_lines" in result
        assert isinstance(result["total_lines"], int)
        assert result["total_lines"] > 0

    def test_has_code_lines(self):
        result = _get_mock_metrics()
        assert "code_lines" in result
        assert result["code_lines"] > 0

    def test_has_comment_lines(self):
        result = _get_mock_metrics()
        assert "comment_lines" in result

    def test_has_blank_lines(self):
        result = _get_mock_metrics()
        assert "blank_lines" in result

    def test_has_files_analyzed(self):
        result = _get_mock_metrics()
        assert "files_analyzed" in result
        assert result["files_analyzed"] > 0

    def test_has_average_complexity(self):
        result = _get_mock_metrics()
        assert "average_complexity" in result
        assert isinstance(result["average_complexity"], (int, float))

    def test_has_max_complexity(self):
        result = _get_mock_metrics()
        assert "max_complexity" in result

    def test_has_maintainability_index(self):
        result = _get_mock_metrics()
        assert "maintainability_index" in result

    def test_has_duplicate_blocks(self):
        result = _get_mock_metrics()
        assert "duplicate_blocks" in result

    def test_has_hotspots(self):
        result = _get_mock_metrics()
        assert "hotspots" in result
        assert isinstance(result["hotspots"], list)
        assert len(result["hotspots"]) > 0

    def test_hotspot_structure(self):
        result = _get_mock_metrics()
        hotspot = result["hotspots"][0]
        assert "file_path" in hotspot
        assert "complexity" in hotspot
        assert "risk_score" in hotspot
        assert "reason" in hotspot

    def test_line_counts_are_consistent(self):
        result = _get_mock_metrics()
        total = result["code_lines"] + result["comment_lines"] + result["blank_lines"]
        assert total == result["total_lines"]


# ===========================================================================
# 9. run_sast_scan
# ===========================================================================


class TestRunSASTScan:
    """Tests for run_sast_scan."""

    @pytest.mark.asyncio
    async def test_falls_back_to_mock_on_import_error(self):
        """When SASTScanner is unavailable, returns mock findings."""
        with _block_import(_SAST_MODULE):
            result = await run_sast_scan(".", "scan1", "tenant1")
        assert isinstance(result, list)
        assert len(result) > 0
        for finding in result:
            assert isinstance(finding, Finding)
            assert finding.scan_type == ScanType.SAST

    @pytest.mark.asyncio
    async def test_findings_have_correct_scan_id(self):
        with _block_import(_SAST_MODULE):
            result = await run_sast_scan(".", "my_scan_id", "tenant1")
        for finding in result:
            assert finding.scan_id == "my_scan_id"

    @pytest.mark.asyncio
    async def test_validates_path_absolute(self):
        with pytest.raises(ScanPathError, match="Absolute paths"):
            await run_sast_scan("/etc/passwd", "scan1", "tenant1")

    @pytest.mark.asyncio
    async def test_validates_path_traversal(self):
        with pytest.raises(ScanPathError):
            await run_sast_scan("../../../etc", "scan1", "tenant1")

    @pytest.mark.asyncio
    async def test_validates_languages_invalid(self):
        with pytest.raises(ScanPathError, match="Unsupported language"):
            await run_sast_scan(".", "scan1", "tenant1", languages=["nonexistent"])

    @pytest.mark.asyncio
    async def test_accepts_valid_languages(self):
        with _block_import(_SAST_MODULE):
            result = await run_sast_scan(".", "scan1", "tenant1", languages=["python"])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_none_languages_accepted(self):
        with _block_import(_SAST_MODULE):
            result = await run_sast_scan(".", "scan1", "tenant1", languages=None)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_with_mocked_scanner_success(self):
        """When the real scanner is available, its results are used."""
        mock_vuln = MagicMock()
        mock_vuln.severity = "high"
        mock_vuln.message = "SQL injection"
        mock_vuln.file_path = "db.py"
        mock_vuln.line_start = 42
        mock_vuln.snippet = "query = ..."
        mock_vuln.rule_id = "sql-injection"
        mock_vuln.cwe_ids = ["CWE-89"]
        mock_vuln.owasp_category = MagicMock(value="A03:2021")
        mock_vuln.remediation = "Use parameterized queries"
        mock_vuln.confidence = 0.95

        mock_results = MagicMock()
        mock_results.findings = [mock_vuln]

        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(return_value=mock_results)

        mock_module = MagicMock()
        mock_module.SASTScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_SAST_MODULE: mock_module}):
            result = await run_sast_scan(".", "scan1", "tenant1")

        assert len(result) == 1
        assert result[0].title == "SQL injection"
        assert result[0].cwe_id == "CWE-89"

    @pytest.mark.asyncio
    async def test_falls_back_on_runtime_error(self):
        """If the real scanner raises RuntimeError, falls back to mock."""
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(side_effect=RuntimeError("scanner crash"))
        mock_module = MagicMock()
        mock_module.SASTScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_SAST_MODULE: mock_module}):
            result = await run_sast_scan(".", "scan1", "tenant1")

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_empty_path_rejected(self):
        with pytest.raises(ScanPathError, match="cannot be empty"):
            await run_sast_scan("", "scan1", "tenant1")

    @pytest.mark.asyncio
    async def test_scanner_with_empty_results(self):
        mock_results = MagicMock()
        mock_results.findings = []

        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(return_value=mock_results)

        mock_module = MagicMock()
        mock_module.SASTScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_SAST_MODULE: mock_module}):
            result = await run_sast_scan(".", "scan1", "tenant1")

        assert result == []

    @pytest.mark.asyncio
    async def test_scanner_vuln_without_cwe(self):
        mock_vuln = MagicMock()
        mock_vuln.severity = "medium"
        mock_vuln.message = "Some issue"
        mock_vuln.file_path = "x.py"
        mock_vuln.line_start = 1
        mock_vuln.snippet = "..."
        mock_vuln.rule_id = "rule-1"
        mock_vuln.cwe_ids = []  # empty CWE list
        mock_vuln.owasp_category = "A03:2021"
        mock_vuln.remediation = "Fix it"
        mock_vuln.confidence = 0.5

        mock_results = MagicMock()
        mock_results.findings = [mock_vuln]

        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(return_value=mock_results)

        mock_module = MagicMock()
        mock_module.SASTScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_SAST_MODULE: mock_module}):
            result = await run_sast_scan(".", "scan1", "tenant1")

        assert len(result) == 1
        assert result[0].cwe_id is None

    @pytest.mark.asyncio
    async def test_falls_back_on_os_error(self):
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(side_effect=OSError("perm denied"))
        mock_module = MagicMock()
        mock_module.SASTScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_SAST_MODULE: mock_module}):
            result = await run_sast_scan(".", "scan1", "tenant1")

        assert isinstance(result, list)
        assert len(result) > 0  # mock fallback


# ===========================================================================
# 10. run_bug_scan
# ===========================================================================


class TestRunBugScan:
    """Tests for run_bug_scan."""

    @pytest.mark.asyncio
    async def test_falls_back_to_mock_on_import_error(self):
        with _block_import(_BUG_MODULE):
            result = await run_bug_scan(".", "scan1", "tenant1")
        assert isinstance(result, list)
        assert len(result) > 0
        for finding in result:
            assert finding.scan_type == ScanType.BUGS

    @pytest.mark.asyncio
    async def test_findings_have_correct_scan_id(self):
        with _block_import(_BUG_MODULE):
            result = await run_bug_scan(".", "my_scan", "tenant1")
        for finding in result:
            assert finding.scan_id == "my_scan"

    @pytest.mark.asyncio
    async def test_validates_path(self):
        with pytest.raises(ScanPathError):
            await run_bug_scan("/etc/passwd", "scan1", "tenant1")

    @pytest.mark.asyncio
    async def test_shell_injection_blocked(self):
        with pytest.raises(ScanPathError, match="disallowed character"):
            await run_bug_scan("src; cat /etc/shadow", "scan1", "tenant1")

    @pytest.mark.asyncio
    async def test_with_mocked_scanner_success(self):
        mock_bug = MagicMock()
        mock_bug.severity = "medium"
        mock_bug.message = "Null dereference"
        mock_bug.description = "Variable may be None"
        mock_bug.file_path = "service.py"
        mock_bug.line_number = 100
        mock_bug.snippet = "x.y"
        mock_bug.bug_type = MagicMock(value="null-deref")
        mock_bug.suggested_fix = "Add null check"
        mock_bug.confidence = 0.8

        mock_results = MagicMock()
        mock_results.bugs = [mock_bug]

        mock_detector = MagicMock()
        mock_detector.scan_repository = AsyncMock(return_value=mock_results)

        mock_module = MagicMock()
        mock_module.BugDetector = lambda: mock_detector

        with patch.dict("sys.modules", {_BUG_MODULE: mock_module}):
            result = await run_bug_scan(".", "scan1", "tenant1")

        assert len(result) == 1
        assert result[0].title == "Null dereference"

    @pytest.mark.asyncio
    async def test_falls_back_on_os_error(self):
        mock_detector = MagicMock()
        mock_detector.scan_repository = AsyncMock(side_effect=OSError("disk error"))
        mock_module = MagicMock()
        mock_module.BugDetector = lambda: mock_detector

        with patch.dict("sys.modules", {_BUG_MODULE: mock_module}):
            result = await run_bug_scan(".", "scan1", "tenant1")

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_bug_with_string_bug_type(self):
        mock_bug = MagicMock()
        mock_bug.severity = "low"
        mock_bug.message = "Unreachable code"
        mock_bug.description = "Code after return"
        mock_bug.file_path = "util.py"
        mock_bug.line_number = 5
        mock_bug.snippet = "return x\nprint(y)"
        mock_bug.bug_type = "dead-code"  # string, no .value
        mock_bug.suggested_fix = "Remove dead code"
        mock_bug.confidence = 0.6

        mock_results = MagicMock()
        mock_results.bugs = [mock_bug]

        mock_detector = MagicMock()
        mock_detector.scan_repository = AsyncMock(return_value=mock_results)
        mock_module = MagicMock()
        mock_module.BugDetector = lambda: mock_detector

        with patch.dict("sys.modules", {_BUG_MODULE: mock_module}):
            result = await run_bug_scan(".", "scan1", "tenant1")

        assert len(result) == 1
        assert result[0].rule_id == "dead-code"


# ===========================================================================
# 11. run_secrets_scan
# ===========================================================================


class TestRunSecretsScan:
    """Tests for run_secrets_scan."""

    @pytest.mark.asyncio
    async def test_falls_back_to_mock_on_import_error(self):
        with _block_import(_SECRETS_MODULE):
            result = await run_secrets_scan(".", "scan1", "tenant1")
        assert isinstance(result, list)
        assert len(result) > 0
        for finding in result:
            assert finding.scan_type == ScanType.SECRETS

    @pytest.mark.asyncio
    async def test_findings_have_correct_scan_id(self):
        with _block_import(_SECRETS_MODULE):
            result = await run_secrets_scan(".", "my_scan", "tenant1")
        for finding in result:
            assert finding.scan_id == "my_scan"

    @pytest.mark.asyncio
    async def test_mock_findings_are_critical(self):
        with _block_import(_SECRETS_MODULE):
            result = await run_secrets_scan(".", "scan1", "tenant1")
        for finding in result:
            assert finding.severity == FindingSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_validates_path(self):
        with pytest.raises(ScanPathError):
            await run_secrets_scan("/root", "scan1", "tenant1")

    @pytest.mark.asyncio
    async def test_with_mocked_scanner_success(self):
        mock_secret = MagicMock()
        mock_secret.secret_type = "api_key"
        mock_secret.file_path = "config.py"
        mock_secret.line_number = 5
        mock_secret.context_line = 'KEY = "sk-abc..."'
        mock_secret.confidence = 0.99

        mock_results = MagicMock()
        mock_results.secrets = [mock_secret]

        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(return_value=mock_results)

        mock_module = MagicMock()
        mock_module.SecretsScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_SECRETS_MODULE: mock_module}):
            result = await run_secrets_scan(".", "scan1", "tenant1")

        assert len(result) == 1
        assert result[0].severity == FindingSeverity.CRITICAL
        assert "api_key" in result[0].title

    @pytest.mark.asyncio
    async def test_falls_back_on_value_error(self):
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(side_effect=ValueError("bad data"))
        mock_module = MagicMock()
        mock_module.SecretsScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_SECRETS_MODULE: mock_module}):
            result = await run_secrets_scan(".", "scan1", "tenant1")

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_falls_back_on_type_error(self):
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(side_effect=TypeError("bad"))
        mock_module = MagicMock()
        mock_module.SecretsScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_SECRETS_MODULE: mock_module}):
            result = await run_secrets_scan(".", "scan1", "tenant1")

        assert isinstance(result, list)
        assert len(result) > 0


# ===========================================================================
# 12. run_dependency_scan
# ===========================================================================


class TestRunDependencyScan:
    """Tests for run_dependency_scan."""

    @pytest.mark.asyncio
    async def test_falls_back_to_mock_on_import_error(self):
        with _block_import(_DEP_MODULE):
            result = await run_dependency_scan(".", "scan1", "tenant1")
        assert isinstance(result, list)
        assert len(result) > 0
        for finding in result:
            assert finding.scan_type == ScanType.DEPENDENCIES

    @pytest.mark.asyncio
    async def test_findings_have_correct_scan_id(self):
        with _block_import(_DEP_MODULE):
            result = await run_dependency_scan(".", "my_scan", "tenant1")
        for finding in result:
            assert finding.scan_id == "my_scan"

    @pytest.mark.asyncio
    async def test_validates_path(self):
        with pytest.raises(ScanPathError):
            await run_dependency_scan("~/secrets", "scan1", "tenant1")

    @pytest.mark.asyncio
    async def test_with_mocked_scanner_success(self):
        mock_vuln = MagicMock()
        mock_vuln.severity = "high"
        mock_vuln.description = "Prototype pollution"
        mock_vuln.cwe_ids = ["CWE-1321"]
        mock_vuln.recommended_version = "4.17.21"

        mock_dep = MagicMock()
        mock_dep.has_vulnerabilities = True
        mock_dep.name = "lodash"
        mock_dep.version = "4.17.15"
        mock_dep.file_path = "package.json"
        mock_dep.vulnerabilities = [mock_vuln]

        mock_results = MagicMock()
        mock_results.dependencies = [mock_dep]

        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(return_value=mock_results)

        mock_module = MagicMock()
        mock_module.DependencyScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_DEP_MODULE: mock_module}):
            result = await run_dependency_scan(".", "scan1", "tenant1")

        assert len(result) == 1
        assert "lodash" in result[0].title
        assert result[0].cwe_id == "CWE-1321"

    @pytest.mark.asyncio
    async def test_skips_deps_without_vulnerabilities(self):
        mock_dep = MagicMock()
        mock_dep.has_vulnerabilities = False
        mock_dep.name = "safe-lib"

        mock_results = MagicMock()
        mock_results.dependencies = [mock_dep]

        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(return_value=mock_results)
        mock_module = MagicMock()
        mock_module.DependencyScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_DEP_MODULE: mock_module}):
            result = await run_dependency_scan(".", "scan1", "tenant1")

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_no_recommended_version(self):
        mock_vuln = MagicMock()
        mock_vuln.severity = "medium"
        mock_vuln.description = "Issue"
        mock_vuln.cwe_ids = []
        mock_vuln.recommended_version = None

        mock_dep = MagicMock()
        mock_dep.has_vulnerabilities = True
        mock_dep.name = "old-lib"
        mock_dep.version = "1.0.0"
        mock_dep.file_path = "requirements.txt"
        mock_dep.vulnerabilities = [mock_vuln]

        mock_results = MagicMock()
        mock_results.dependencies = [mock_dep]

        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(return_value=mock_results)
        mock_module = MagicMock()
        mock_module.DependencyScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_DEP_MODULE: mock_module}):
            result = await run_dependency_scan(".", "scan1", "tenant1")

        assert len(result) == 1
        assert "No fix available" in result[0].remediation

    @pytest.mark.asyncio
    async def test_dep_without_file_path_uses_default(self):
        mock_vuln = MagicMock()
        mock_vuln.severity = "low"
        mock_vuln.description = "Minor issue"
        mock_vuln.cwe_ids = []
        mock_vuln.recommended_version = "2.0.0"

        mock_dep = MagicMock()
        mock_dep.has_vulnerabilities = True
        mock_dep.name = "lib"
        mock_dep.version = "1.0.0"
        mock_dep.file_path = None
        mock_dep.vulnerabilities = [mock_vuln]

        mock_results = MagicMock()
        mock_results.dependencies = [mock_dep]

        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(return_value=mock_results)
        mock_module = MagicMock()
        mock_module.DependencyScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_DEP_MODULE: mock_module}):
            result = await run_dependency_scan(".", "scan1", "tenant1")

        assert result[0].file_path == "package.json"

    @pytest.mark.asyncio
    async def test_falls_back_on_key_error(self):
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(side_effect=KeyError("missing key"))
        mock_module = MagicMock()
        mock_module.DependencyScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_DEP_MODULE: mock_module}):
            result = await run_dependency_scan(".", "scan1", "tenant1")

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_multiple_vulns_in_single_dep(self):
        mock_vuln1 = MagicMock()
        mock_vuln1.severity = "high"
        mock_vuln1.description = "Issue 1"
        mock_vuln1.cwe_ids = ["CWE-100"]
        mock_vuln1.recommended_version = "2.0.0"

        mock_vuln2 = MagicMock()
        mock_vuln2.severity = "critical"
        mock_vuln2.description = "Issue 2"
        mock_vuln2.cwe_ids = ["CWE-200"]
        mock_vuln2.recommended_version = "2.0.0"

        mock_dep = MagicMock()
        mock_dep.has_vulnerabilities = True
        mock_dep.name = "badlib"
        mock_dep.version = "1.0.0"
        mock_dep.file_path = "requirements.txt"
        mock_dep.vulnerabilities = [mock_vuln1, mock_vuln2]

        mock_results = MagicMock()
        mock_results.dependencies = [mock_dep]

        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(return_value=mock_results)
        mock_module = MagicMock()
        mock_module.DependencyScanner = lambda: mock_scanner

        with patch.dict("sys.modules", {_DEP_MODULE: mock_module}):
            result = await run_dependency_scan(".", "scan1", "tenant1")

        assert len(result) == 2


# ===========================================================================
# 13. run_metrics_analysis
# ===========================================================================


class TestRunMetricsAnalysis:
    """Tests for run_metrics_analysis."""

    @pytest.mark.asyncio
    async def test_falls_back_to_mock_on_import_error(self):
        with _block_import(_METRICS_MODULE):
            result = await run_metrics_analysis(".", "scan1", "tenant1")
        assert isinstance(result, dict)
        assert "total_lines" in result

    @pytest.mark.asyncio
    async def test_returns_expected_keys(self):
        with _block_import(_METRICS_MODULE):
            result = await run_metrics_analysis(".", "scan1", "tenant1")
        expected_keys = {
            "total_lines",
            "code_lines",
            "comment_lines",
            "blank_lines",
            "files_analyzed",
            "average_complexity",
            "max_complexity",
            "maintainability_index",
            "duplicate_blocks",
            "hotspots",
        }
        assert expected_keys.issubset(set(result.keys()))

    @pytest.mark.asyncio
    async def test_validates_path(self):
        with pytest.raises(ScanPathError):
            await run_metrics_analysis("/etc", "scan1", "tenant1")

    @pytest.mark.asyncio
    async def test_with_mocked_analyzer_success(self):
        mock_hotspot = MagicMock()
        mock_hotspot.to_dict.return_value = {
            "file_path": "complex.py",
            "complexity": 30,
        }

        mock_results = MagicMock()
        mock_results.total_lines = 5000
        mock_results.total_code_lines = 3500
        mock_results.total_comment_lines = 1000
        mock_results.total_blank_lines = 500
        mock_results.total_files = 50
        mock_results.avg_complexity = 5.0
        mock_results.max_complexity = 30
        mock_results.maintainability_index = 75.0
        mock_results.duplicates = [MagicMock(), MagicMock()]
        mock_results.hotspots = [mock_hotspot]

        mock_module = MagicMock()
        mock_module.CodeMetricsAnalyzer = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {_METRICS_MODULE: mock_module}):
            with patch(
                "asyncio.to_thread",
                new=AsyncMock(return_value=mock_results),
            ):
                result = await run_metrics_analysis(".", "scan1", "tenant1")

        assert result["total_lines"] == 5000
        assert result["code_lines"] == 3500
        assert result["duplicate_blocks"] == 2

    @pytest.mark.asyncio
    async def test_falls_back_on_type_error(self):
        mock_module = MagicMock()
        mock_module.CodeMetricsAnalyzer = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {_METRICS_MODULE: mock_module}):
            with patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=TypeError("bad type")),
            ):
                result = await run_metrics_analysis(".", "scan1", "tenant1")

        assert isinstance(result, dict)
        assert "total_lines" in result

    @pytest.mark.asyncio
    async def test_falls_back_on_runtime_error(self):
        mock_module = MagicMock()
        mock_module.CodeMetricsAnalyzer = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {_METRICS_MODULE: mock_module}):
            with patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=RuntimeError("crash")),
            ):
                result = await run_metrics_analysis(".", "scan1", "tenant1")

        assert isinstance(result, dict)
        assert "total_lines" in result

    @pytest.mark.asyncio
    async def test_empty_path_rejected(self):
        with pytest.raises(ScanPathError, match="cannot be empty"):
            await run_metrics_analysis("", "scan1", "tenant1")

    @pytest.mark.asyncio
    async def test_hotspots_limited_to_ten(self):
        hotspots = [MagicMock() for _ in range(15)]
        for i, h in enumerate(hotspots):
            h.to_dict.return_value = {"file_path": f"f{i}.py", "complexity": i}

        mock_results = MagicMock()
        mock_results.total_lines = 100
        mock_results.total_code_lines = 80
        mock_results.total_comment_lines = 10
        mock_results.total_blank_lines = 10
        mock_results.total_files = 15
        mock_results.avg_complexity = 3.0
        mock_results.max_complexity = 20
        mock_results.maintainability_index = 70.0
        mock_results.duplicates = []
        mock_results.hotspots = hotspots

        mock_module = MagicMock()
        mock_module.CodeMetricsAnalyzer = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {_METRICS_MODULE: mock_module}):
            with patch(
                "asyncio.to_thread",
                new=AsyncMock(return_value=mock_results),
            ):
                result = await run_metrics_analysis(".", "scan1", "tenant1")

        assert len(result["hotspots"]) == 10


# ===========================================================================
# 14. Cross-cutting: scanner path validation is defense-in-depth
# ===========================================================================


class TestDefenseInDepthPathValidation:
    """Verify that all scanners validate paths independently."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scanner_name,scanner_fn",
        [
            ("sast", run_sast_scan),
            ("bug", run_bug_scan),
            ("secrets", run_secrets_scan),
            ("dependency", run_dependency_scan),
            ("metrics", run_metrics_analysis),
        ],
    )
    async def test_all_scanners_reject_absolute_paths(self, scanner_name, scanner_fn):
        with pytest.raises(ScanPathError, match="Absolute paths"):
            await scanner_fn("/etc/passwd", "scan1", "tenant1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scanner_name,scanner_fn",
        [
            ("sast", run_sast_scan),
            ("bug", run_bug_scan),
            ("secrets", run_secrets_scan),
            ("dependency", run_dependency_scan),
            ("metrics", run_metrics_analysis),
        ],
    )
    async def test_all_scanners_reject_shell_injection(self, scanner_name, scanner_fn):
        with pytest.raises(ScanPathError, match="disallowed character"):
            await scanner_fn("src; rm -rf /", "scan1", "tenant1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scanner_name,scanner_fn",
        [
            ("sast", run_sast_scan),
            ("bug", run_bug_scan),
            ("secrets", run_secrets_scan),
            ("dependency", run_dependency_scan),
            ("metrics", run_metrics_analysis),
        ],
    )
    async def test_all_scanners_reject_null_bytes(self, scanner_name, scanner_fn):
        with pytest.raises(ScanPathError, match="null byte"):
            await scanner_fn("src/\x00evil", "scan1", "tenant1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scanner_name,scanner_fn",
        [
            ("sast", run_sast_scan),
            ("bug", run_bug_scan),
            ("secrets", run_secrets_scan),
            ("dependency", run_dependency_scan),
            ("metrics", run_metrics_analysis),
        ],
    )
    async def test_all_scanners_reject_home_expansion(self, scanner_name, scanner_fn):
        with pytest.raises(ScanPathError, match="Home directory expansion"):
            await scanner_fn("~/evil", "scan1", "tenant1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scanner_name,scanner_fn",
        [
            ("sast", run_sast_scan),
            ("bug", run_bug_scan),
            ("secrets", run_secrets_scan),
            ("dependency", run_dependency_scan),
            ("metrics", run_metrics_analysis),
        ],
    )
    async def test_all_scanners_reject_empty_path(self, scanner_name, scanner_fn):
        with pytest.raises(ScanPathError, match="cannot be empty"):
            await scanner_fn("", "scan1", "tenant1")


# ===========================================================================
# 15. MAX_PATH_LENGTH constant
# ===========================================================================


class TestMaxPathLength:
    """Tests for the _MAX_PATH_LENGTH constant."""

    def test_is_4096(self):
        assert _MAX_PATH_LENGTH == 4096

    def test_is_integer(self):
        assert isinstance(_MAX_PATH_LENGTH, int)

    def test_positive(self):
        assert _MAX_PATH_LENGTH > 0
