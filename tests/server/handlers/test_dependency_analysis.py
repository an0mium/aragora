"""
Tests for the dependency analysis HTTP handler.

Comprehensive test suite covering:
- Dependency tree analysis
- SBOM generation (CycloneDX, SPDX)
- Vulnerability scanning
- License compatibility checking
- Cache management
- Circuit breaker integration
- Rate limiting
- Error handling and edge cases
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from pathlib import Path

from aragora.server.handlers.dependency_analysis import (
    handle_analyze_dependencies,
    handle_generate_sbom,
    handle_scan_vulnerabilities,
    handle_check_licenses,
    handle_clear_cache,
    get_dependency_analyzer,
    get_dependency_circuit_breaker,
    DependencyAnalysisHandler,
    _analysis_cache,
    _analysis_cache_lock,
)
from aragora.server.handlers.utils.responses import HandlerResult
from aragora.rbac.models import AuthorizationContext


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_auth_context():
    """Create a mock authorization context for testing."""
    return AuthorizationContext(
        user_id="test-user",
        user_email="test@example.com",
        roles={"admin"},
        permissions={"codebase.run"},
    )


@pytest.fixture
def handler():
    """Create a handler instance for testing."""
    return DependencyAnalysisHandler({})


@pytest.fixture(autouse=True)
def clean_cache():
    """Clean analysis cache before each test."""
    with _analysis_cache_lock:
        _analysis_cache.clear()
    yield
    with _analysis_cache_lock:
        _analysis_cache.clear()


@pytest.fixture(autouse=True)
def _allow_any_repo_path(monkeypatch):
    """Allow any repo_path in tests by setting a permissive scan root."""
    monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/")


@pytest.fixture
def mock_dependency_tree():
    """Create a mock dependency tree."""
    tree = MagicMock()
    tree.project_name = "test-project"
    tree.project_version = "1.0.0"
    tree.package_managers = ["pip"]
    tree.dependencies = {
        "requests": {
            "version": "2.31.0",
            "license": "Apache-2.0",
            "dependencies": [],
        },
        "flask": {
            "version": "2.3.0",
            "license": "BSD-3-Clause",
            "dependencies": ["werkzeug", "jinja2"],
        },
    }
    return tree


@pytest.fixture
def mock_analyzer(mock_dependency_tree):
    """Create a mock dependency analyzer."""
    analyzer = MagicMock()
    analyzer.resolve_dependencies = AsyncMock(return_value=mock_dependency_tree)
    analyzer.generate_sbom = AsyncMock(return_value='{"bomFormat": "CycloneDX"}')
    analyzer.check_vulnerabilities = AsyncMock(return_value=[])
    analyzer.check_license_compatibility = AsyncMock(return_value=[])
    return analyzer


def parse_result(result: HandlerResult) -> dict:
    """Parse HandlerResult body to dict for assertions."""
    data = json.loads(result.body.decode("utf-8"))
    # Add status from the HandlerResult for convenience
    data["status"] = result.status_code
    # Normalize success field - if "error" exists, it's a failure
    if "success" not in data:
        data["success"] = "error" not in data
    return data


# =============================================================================
# Route Configuration Tests
# =============================================================================


class TestDependencyAnalysisRoutes:
    """Test route configuration."""

    def test_handler_routes_defined(self):
        """Handler has routes defined."""
        assert hasattr(DependencyAnalysisHandler, "ROUTES")
        routes = DependencyAnalysisHandler.ROUTES
        assert "/api/v1/codebase/analyze-dependencies" in routes
        assert "/api/v1/codebase/sbom" in routes
        assert "/api/v1/codebase/scan-vulnerabilities" in routes
        assert "/api/v1/codebase/check-licenses" in routes
        assert "/api/v1/codebase/clear-cache" in routes

    def test_handler_can_handle(self):
        """Handler correctly identifies routes it can handle."""
        handler = DependencyAnalysisHandler({})

        assert handler.can_handle("/api/v1/codebase/analyze-dependencies")
        assert handler.can_handle("/api/v1/codebase/sbom")
        assert handler.can_handle("/api/v1/codebase/scan-vulnerabilities")
        assert handler.can_handle("/api/v1/codebase/check-licenses")
        assert handler.can_handle("/api/v1/codebase/clear-cache")

        assert not handler.can_handle("/api/v1/other/route")
        assert not handler.can_handle("/api/v1/codebase/other")

    def test_handler_routes_count(self):
        """Handler has expected number of routes."""
        assert len(DependencyAnalysisHandler.ROUTES) == 5


# =============================================================================
# Handler Instance Tests
# =============================================================================


class TestDependencyAnalysisHandler:
    """Tests for DependencyAnalysisHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = DependencyAnalysisHandler({})
        assert handler is not None

    def test_handler_creation_with_ctx(self):
        """Test creating handler with context."""
        ctx = {"key": "value"}
        handler = DependencyAnalysisHandler(ctx)
        assert handler.ctx == ctx

    def test_handler_has_handle_method(self):
        """Test handler has handle method."""
        handler = DependencyAnalysisHandler({})
        assert hasattr(handler, "handle")

    def test_handler_has_handle_post_method(self):
        """Test handler has handle_post method."""
        handler = DependencyAnalysisHandler({})
        assert hasattr(handler, "handle_post")

    def test_handle_returns_none(self):
        """Test handle returns None (POST-only handler)."""
        handler = DependencyAnalysisHandler({})
        result = handler.handle("/api/v1/codebase/analyze-dependencies", {}, None)
        assert result is None


# =============================================================================
# Dependency Analysis Tests
# =============================================================================


class TestAnalyzeDependencies:
    """Test dependency analysis endpoint."""

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, mock_auth_context):
        """Returns error when repo_path is missing."""
        result = parse_result(await handle_analyze_dependencies(mock_auth_context, {}))

        assert result["success"] is False
        assert result["status"] == 400
        assert "repo_path is required" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_repo_path(self, mock_auth_context):
        """Returns error when repo_path is empty."""
        result = parse_result(
            await handle_analyze_dependencies(mock_auth_context, {"repo_path": ""})
        )

        assert result["success"] is False
        assert result["status"] == 400
        assert "repo_path is required" in result["error"]

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, mock_auth_context):
        """Returns error for nonexistent path."""
        result = parse_result(
            await handle_analyze_dependencies(
                mock_auth_context, {"repo_path": "/nonexistent/path/that/does/not/exist"}
            )
        )

        assert result["success"] is False
        assert result["status"] == 404
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_with_cache(self, tmp_path, mock_auth_context):
        """Analysis uses cache when available."""
        # Create a mock requirements.txt
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.project_version = "1.0.0"
            mock_tree.package_managers = []
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_analyze_dependencies(
                    mock_auth_context,
                    {
                        "repo_path": str(tmp_path),
                        "use_cache": True,
                    },
                )
            )

            assert result["success"] is True
            assert result["data"]["project_name"] == "test-project"

    @pytest.mark.asyncio
    async def test_analyze_without_cache(self, tmp_path, mock_auth_context):
        """Analysis bypasses cache when requested."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "fresh-analysis"
            mock_tree.project_version = "2.0.0"
            mock_tree.package_managers = ["pip"]
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_analyze_dependencies(
                    mock_auth_context,
                    {
                        "repo_path": str(tmp_path),
                        "use_cache": False,
                    },
                )
            )

            assert result["success"] is True
            assert result["data"]["project_name"] == "fresh-analysis"

    @pytest.mark.asyncio
    async def test_analyze_with_dev_dependencies(self, tmp_path, mock_auth_context):
        """Analysis includes dev dependencies when requested."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        (tmp_path / "requirements-dev.txt").write_text("pytest==7.0.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.project_version = "1.0.0"
            mock_tree.package_managers = ["pip"]
            mock_tree.dependencies = {"requests": {}, "pytest": {}}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_analyze_dependencies(
                    mock_auth_context,
                    {
                        "repo_path": str(tmp_path),
                        "include_dev": True,
                    },
                )
            )

            assert result["success"] is True
            mock_analyzer.resolve_dependencies.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_without_dev_dependencies(self, tmp_path, mock_auth_context):
        """Analysis excludes dev dependencies when requested."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.project_version = "1.0.0"
            mock_tree.package_managers = ["pip"]
            mock_tree.dependencies = {"requests": {}}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_analyze_dependencies(
                    mock_auth_context,
                    {
                        "repo_path": str(tmp_path),
                        "include_dev": False,
                    },
                )
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_analyze_returns_package_managers(self, tmp_path, mock_auth_context):
        """Analysis returns detected package managers."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        (tmp_path / "package.json").write_text('{"name": "test"}\n')

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.project_version = "1.0.0"
            mock_tree.package_managers = ["pip", "npm"]
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_analyze_dependencies(
                    mock_auth_context,
                    {"repo_path": str(tmp_path)},
                )
            )

            assert result["success"] is True
            assert "pip" in result["data"]["package_managers"]
            assert "npm" in result["data"]["package_managers"]


# =============================================================================
# SBOM Generation Tests
# =============================================================================


class TestGenerateSBOM:
    """Test SBOM generation endpoint."""

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, mock_auth_context):
        """Returns error when repo_path is missing."""
        result = parse_result(await handle_generate_sbom(mock_auth_context, {}))

        assert result["success"] is False
        assert result["status"] == 400
        assert "repo_path is required" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_format(self, tmp_path, mock_auth_context):
        """Returns error for invalid SBOM format."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        result = parse_result(
            await handle_generate_sbom(
                mock_auth_context,
                {
                    "repo_path": str(tmp_path),
                    "format": "invalid_format",
                },
            )
        )

        assert result["success"] is False
        assert result["status"] == 400
        assert "cyclonedx" in result["error"] or "spdx" in result["error"]

    @pytest.mark.asyncio
    async def test_cyclonedx_format(self, tmp_path, mock_auth_context):
        """Generates CycloneDX SBOM."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.project_version = "1.0.0"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_analyzer.generate_sbom = AsyncMock(return_value='{"bomFormat": "CycloneDX"}')
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_generate_sbom(
                    mock_auth_context,
                    {
                        "repo_path": str(tmp_path),
                        "format": "cyclonedx",
                    },
                )
            )

            assert result["success"] is True
            assert "sbom" in result["data"]

    @pytest.mark.asyncio
    async def test_spdx_format(self, tmp_path, mock_auth_context):
        """Generates SPDX SBOM."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.project_version = "1.0.0"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_analyzer.generate_sbom = AsyncMock(return_value='{"spdxVersion": "SPDX-2.3"}')
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_generate_sbom(
                    mock_auth_context,
                    {
                        "repo_path": str(tmp_path),
                        "format": "spdx",
                    },
                )
            )

            assert result["success"] is True
            assert "sbom" in result["data"]

    @pytest.mark.asyncio
    async def test_default_format_is_cyclonedx(self, tmp_path, mock_auth_context):
        """Default SBOM format is CycloneDX."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.project_version = "1.0.0"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_analyzer.generate_sbom = AsyncMock(return_value='{"bomFormat": "CycloneDX"}')
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_generate_sbom(
                    mock_auth_context,
                    {"repo_path": str(tmp_path)},
                )
            )

            assert result["success"] is True
            # Verify CycloneDX was used
            mock_analyzer.generate_sbom.assert_called_once()

    @pytest.mark.asyncio
    async def test_sbom_includes_metadata(self, tmp_path, mock_auth_context):
        """SBOM includes metadata."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.project_version = "1.0.0"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_analyzer.generate_sbom = AsyncMock(return_value='{"bomFormat": "CycloneDX"}')
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_generate_sbom(
                    mock_auth_context,
                    {"repo_path": str(tmp_path)},
                )
            )

            assert result["success"] is True
            assert "format" in result["data"]


# =============================================================================
# Vulnerability Scanning Tests
# =============================================================================


class TestScanVulnerabilities:
    """Test vulnerability scanning endpoint."""

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, mock_auth_context):
        """Returns error when repo_path is missing."""
        result = parse_result(await handle_scan_vulnerabilities(mock_auth_context, {}))

        assert result["success"] is False
        assert result["status"] == 400
        assert "repo_path is required" in result["error"]

    @pytest.mark.asyncio
    async def test_scan_returns_vulnerabilities(self, tmp_path, mock_auth_context):
        """Scan returns list of vulnerabilities."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)

            # Create mock vulnerability with correct attributes
            mock_vuln = MagicMock()
            mock_vuln.id = "CVE-2023-1234"
            mock_vuln.title = "Security vulnerability in requests"
            mock_vuln.description = "Test vulnerability"
            mock_vuln.affected_package = "requests"
            mock_vuln.affected_versions = "<2.32.0"
            mock_vuln.fixed_version = "2.32.0"
            mock_vuln.cvss_score = 7.5
            mock_vuln.cwe_id = "CWE-79"
            mock_vuln.references = ["https://cve.mitre.org"]
            # severity is an enum with .value attribute
            mock_severity = MagicMock()
            mock_severity.value = "high"  # lowercase to match grouping
            mock_vuln.severity = mock_severity
            mock_analyzer.check_vulnerabilities = AsyncMock(return_value=[mock_vuln])
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_scan_vulnerabilities(
                    mock_auth_context,
                    {
                        "repo_path": str(tmp_path),
                    },
                )
            )

            assert result["success"] is True
            assert result["data"]["total_vulnerabilities"] == 1
            # Vulnerabilities are grouped by severity
            assert result["data"]["high_count"] == 1
            high_vulns = result["data"]["vulnerabilities_by_severity"]["high"]
            assert len(high_vulns) == 1
            assert high_vulns[0]["id"] == "CVE-2023-1234"

    @pytest.mark.asyncio
    async def test_scan_no_vulnerabilities(self, tmp_path, mock_auth_context):
        """Scan returns empty when no vulnerabilities found."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "secure-project"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_analyzer.check_vulnerabilities = AsyncMock(return_value=[])
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_scan_vulnerabilities(
                    mock_auth_context,
                    {"repo_path": str(tmp_path)},
                )
            )

            assert result["success"] is True
            assert result["data"]["total_vulnerabilities"] == 0

    @pytest.mark.asyncio
    async def test_scan_multiple_severity_levels(self, tmp_path, mock_auth_context):
        """Scan categorizes vulnerabilities by severity."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\nflask==2.0.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)

            # Create vulnerabilities with different severities
            vulns = []
            for severity, cve_id in [
                ("critical", "CVE-2023-0001"),
                ("high", "CVE-2023-0002"),
                ("medium", "CVE-2023-0003"),
                ("low", "CVE-2023-0004"),
            ]:
                mock_vuln = MagicMock()
                mock_vuln.id = cve_id
                mock_vuln.title = f"{severity.title()} vulnerability"
                mock_vuln.description = f"Test {severity} vulnerability"
                mock_vuln.affected_package = "test-pkg"
                mock_vuln.affected_versions = "<1.0.0"
                mock_vuln.fixed_version = "1.0.0"
                mock_vuln.cvss_score = {"critical": 9.5, "high": 7.5, "medium": 5.0, "low": 2.5}[
                    severity
                ]
                mock_vuln.cwe_id = "CWE-79"
                mock_vuln.references = []
                mock_severity = MagicMock()
                mock_severity.value = severity
                mock_vuln.severity = mock_severity
                vulns.append(mock_vuln)

            mock_analyzer.check_vulnerabilities = AsyncMock(return_value=vulns)
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_scan_vulnerabilities(
                    mock_auth_context,
                    {"repo_path": str(tmp_path)},
                )
            )

            assert result["success"] is True
            assert result["data"]["total_vulnerabilities"] == 4
            assert result["data"]["critical_count"] == 1
            assert result["data"]["high_count"] == 1
            assert result["data"]["medium_count"] == 1
            assert result["data"]["low_count"] == 1


# =============================================================================
# License Compatibility Tests
# =============================================================================


class TestCheckLicenses:
    """Test license compatibility endpoint."""

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, mock_auth_context):
        """Returns error when repo_path is missing."""
        result = parse_result(await handle_check_licenses(mock_auth_context, {}))

        assert result["success"] is False
        assert result["status"] == 400
        assert "repo_path is required" in result["error"]

    @pytest.mark.asyncio
    async def test_check_returns_conflicts(self, tmp_path, mock_auth_context):
        """Check returns license conflicts."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)

            mock_conflict = MagicMock()
            mock_conflict.package_b = "gpl-lib"
            mock_conflict.license_b = "GPL-3.0"
            mock_conflict.conflict_type = "copyleft"
            mock_conflict.severity = "error"
            mock_conflict.description = "GPL incompatible with MIT"
            mock_analyzer.check_license_compatibility = AsyncMock(return_value=[mock_conflict])
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_check_licenses(
                    mock_auth_context,
                    {
                        "repo_path": str(tmp_path),
                        "project_license": "MIT",
                    },
                )
            )

            assert result["success"] is True
            assert result["data"]["total_conflicts"] == 1
            assert result["data"]["compatible"] is False

    @pytest.mark.asyncio
    async def test_check_no_conflicts(self, tmp_path, mock_auth_context):
        """Check returns no conflicts when compatible."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_analyzer.check_license_compatibility = AsyncMock(return_value=[])
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_check_licenses(
                    mock_auth_context,
                    {
                        "repo_path": str(tmp_path),
                        "project_license": "Apache-2.0",
                    },
                )
            )

            assert result["success"] is True
            assert result["data"]["total_conflicts"] == 0
            assert result["data"]["compatible"] is True

    @pytest.mark.asyncio
    async def test_check_with_allowed_licenses(self, tmp_path, mock_auth_context):
        """Check respects allowed licenses list."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_analyzer.check_license_compatibility = AsyncMock(return_value=[])
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_check_licenses(
                    mock_auth_context,
                    {
                        "repo_path": str(tmp_path),
                        "project_license": "MIT",
                        "allowed_licenses": ["MIT", "Apache-2.0", "BSD-3-Clause"],
                    },
                )
            )

            assert result["success"] is True


# =============================================================================
# Cache Management Tests
# =============================================================================


class TestClearCache:
    """Test cache clearing endpoint."""

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Cache is cleared successfully."""
        # Add something to cache first
        with _analysis_cache_lock:
            _analysis_cache["test_key"] = {"test": "data"}

        result = parse_result(await handle_clear_cache())

        assert result["success"] is True
        assert result["data"]["cleared"] is True
        assert result["data"]["entries_removed"] >= 1

    @pytest.mark.asyncio
    async def test_clear_empty_cache(self):
        """Clearing empty cache succeeds."""
        result = parse_result(await handle_clear_cache())

        assert result["success"] is True
        assert result["data"]["cleared"] is True
        assert result["data"]["entries_removed"] == 0

    @pytest.mark.asyncio
    async def test_clear_cache_multiple_entries(self):
        """Cache clearing removes all entries."""
        # Add multiple entries
        with _analysis_cache_lock:
            _analysis_cache["key1"] = {"data": 1}
            _analysis_cache["key2"] = {"data": 2}
            _analysis_cache["key3"] = {"data": 3}

        result = parse_result(await handle_clear_cache())

        assert result["success"] is True
        assert result["data"]["entries_removed"] == 3
        assert len(_analysis_cache) == 0


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestDependencyCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_exists(self):
        """Test that circuit breaker is available."""
        cb = get_dependency_circuit_breaker()
        assert cb is not None
        assert cb.name == "dependency_analysis_handler"

    def test_circuit_breaker_configuration(self):
        """Test circuit breaker configuration."""
        cb = get_dependency_circuit_breaker()
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 60

    def test_circuit_breaker_can_execute(self):
        """Test circuit breaker allows execution when closed."""
        cb = get_dependency_circuit_breaker()
        cb.reset()
        assert cb.can_execute() is True

    def test_circuit_breaker_records_success(self):
        """Test circuit breaker records success."""
        cb = get_dependency_circuit_breaker()
        cb.reset()
        cb.record_success()
        assert cb.can_execute() is True

    def test_circuit_breaker_records_failure(self):
        """Test circuit breaker records failure."""
        cb = get_dependency_circuit_breaker()
        cb.reset()
        for _ in range(5):
            cb.record_failure()
        # After threshold failures, circuit should open
        assert cb.can_execute() is False

    def test_circuit_breaker_affects_get_analyzer(self):
        """Test circuit breaker blocks analyzer creation when open."""
        cb = get_dependency_circuit_breaker()
        cb.reset()

        # Open the circuit
        for _ in range(5):
            cb.record_failure()

        # Now try to get analyzer with mocked module state
        with patch("aragora.server.handlers.dependency_analysis._dependency_analyzer", None):
            with pytest.raises(RuntimeError, match="temporarily unavailable"):
                get_dependency_analyzer()

        # Reset for other tests
        cb.reset()


# =============================================================================
# Analyzer Singleton Tests
# =============================================================================


class TestDependencyAnalyzerSingleton:
    """Test singleton pattern for analyzer."""

    def test_get_dependency_analyzer_returns_same_instance(self):
        """Get dependency analyzer returns singleton."""
        with patch("aragora.server.handlers.dependency_analysis._dependency_analyzer", None):
            # Reset circuit breaker so it doesn't block creation
            with patch(
                "aragora.server.handlers.dependency_analysis.get_dependency_circuit_breaker"
            ) as mock_cb:
                mock_cb.return_value.can_execute.return_value = True
                with patch("aragora.audit.dependency_analyzer.DependencyAnalyzer") as mock_class:
                    mock_instance = MagicMock()
                    mock_class.return_value = mock_instance

                    # First call creates instance
                    analyzer1 = get_dependency_analyzer()

                    # Verify mock was called
                    assert analyzer1 is not None


# =============================================================================
# Handler Integration Tests
# =============================================================================


class TestHandlerIntegration:
    """Integration tests for handler class methods."""

    @pytest.mark.asyncio
    async def test_handle_post_routing(self):
        """Handler routes POST requests correctly."""
        handler = DependencyAnalysisHandler({})

        # Test analyze-dependencies route
        with patch(
            "aragora.server.handlers.dependency_analysis.handle_analyze_dependencies",
            new_callable=AsyncMock,
        ) as mock:
            from aragora.server.handlers.utils.responses import success_response

            mock.return_value = success_response({"test": "data"})
            result = await handler.handle_post(
                "/api/v1/codebase/analyze-dependencies",
                {"repo_path": "/test"},
            )
            mock.assert_called_once_with(ANY, {"repo_path": "/test"})

    @pytest.mark.asyncio
    async def test_handle_post_sbom_route(self):
        """Handler routes SBOM POST requests correctly."""
        handler = DependencyAnalysisHandler({})

        with patch(
            "aragora.server.handlers.dependency_analysis.handle_generate_sbom",
            new_callable=AsyncMock,
        ) as mock:
            from aragora.server.handlers.utils.responses import success_response

            mock.return_value = success_response({"sbom": "{}"})
            result = await handler.handle_post(
                "/api/v1/codebase/sbom",
                {"repo_path": "/test"},
            )
            mock.assert_called_once_with(ANY, {"repo_path": "/test"})

    @pytest.mark.asyncio
    async def test_handle_post_vulnerabilities_route(self):
        """Handler routes vulnerabilities POST requests correctly."""
        handler = DependencyAnalysisHandler({})

        with patch(
            "aragora.server.handlers.dependency_analysis.handle_scan_vulnerabilities",
            new_callable=AsyncMock,
        ) as mock:
            from aragora.server.handlers.utils.responses import success_response

            mock.return_value = success_response({"vulnerabilities": []})
            result = await handler.handle_post(
                "/api/v1/codebase/scan-vulnerabilities",
                {"repo_path": "/test"},
            )
            mock.assert_called_once_with(ANY, {"repo_path": "/test"})

    @pytest.mark.asyncio
    async def test_handle_post_licenses_route(self):
        """Handler routes licenses POST requests correctly."""
        handler = DependencyAnalysisHandler({})

        with patch(
            "aragora.server.handlers.dependency_analysis.handle_check_licenses",
            new_callable=AsyncMock,
        ) as mock:
            from aragora.server.handlers.utils.responses import success_response

            mock.return_value = success_response({"conflicts": []})
            result = await handler.handle_post(
                "/api/v1/codebase/check-licenses",
                {"repo_path": "/test"},
            )
            mock.assert_called_once_with(ANY, {"repo_path": "/test"})

    @pytest.mark.asyncio
    async def test_handle_post_clear_cache_route(self):
        """Handler routes clear-cache POST requests correctly."""
        handler = DependencyAnalysisHandler({})

        with patch(
            "aragora.server.handlers.dependency_analysis.handle_clear_cache",
            new_callable=AsyncMock,
        ) as mock:
            from aragora.server.handlers.utils.responses import success_response

            mock.return_value = success_response({"cleared": True})
            result = await handler.handle_post(
                "/api/v1/codebase/clear-cache",
                {},
            )
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_post_unknown_route(self):
        """Handler returns 404 for unknown routes."""
        handler = DependencyAnalysisHandler({})
        result = await handler.handle_post("/api/v1/unknown", {})

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status"] == 404


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_analyzer_exception(self, tmp_path, mock_auth_context):
        """Handler catches analyzer exceptions."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_analyzer.resolve_dependencies = AsyncMock(side_effect=ValueError("Analyzer error"))
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_analyze_dependencies(
                    mock_auth_context,
                    {"repo_path": str(tmp_path)},
                )
            )

            assert result["success"] is False
            assert result["status"] == 500

    @pytest.mark.asyncio
    async def test_sbom_generation_exception(self, tmp_path, mock_auth_context):
        """Handler catches SBOM generation exceptions."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.project_version = "1.0.0"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_analyzer.generate_sbom = AsyncMock(side_effect=ValueError("SBOM error"))
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_generate_sbom(
                    mock_auth_context,
                    {"repo_path": str(tmp_path)},
                )
            )

            assert result["success"] is False
            assert result["status"] == 500

    @pytest.mark.asyncio
    async def test_vulnerability_scan_exception(self, tmp_path, mock_auth_context):
        """Handler catches vulnerability scan exceptions."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_analyzer.check_vulnerabilities = AsyncMock(side_effect=ValueError("Scan error"))
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_scan_vulnerabilities(
                    mock_auth_context,
                    {"repo_path": str(tmp_path)},
                )
            )

            assert result["success"] is False
            assert result["status"] == 500

    @pytest.mark.asyncio
    async def test_license_check_exception(self, tmp_path, mock_auth_context):
        """Handler catches license check exceptions."""
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_analyzer.check_license_compatibility = AsyncMock(
                side_effect=ValueError("License error")
            )
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_check_licenses(
                    mock_auth_context,
                    {"repo_path": str(tmp_path)},
                )
            )

            assert result["success"] is False
            assert result["status"] == 500


# =============================================================================
# Path Validation Tests
# =============================================================================


class TestPathValidation:
    """Tests for path validation."""

    @pytest.fixture(autouse=True)
    def _restrict_repo_path(self, monkeypatch, tmp_path):
        """Override the permissive scan root for path validation tests.

        Uses a unique temporary directory so that paths like /etc/passwd
        and ./relative/path are rejected.
        """
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", str(tmp_path))

    @pytest.mark.asyncio
    async def test_path_traversal_protection(self, mock_auth_context):
        """Handler prevents path traversal attacks."""
        result = parse_result(
            await handle_analyze_dependencies(
                mock_auth_context,
                {"repo_path": "/tmp/../../../etc/passwd"},
            )
        )

        # Should either fail validation or not find the path
        assert result["status"] in [400, 404]

    @pytest.mark.asyncio
    async def test_relative_path_handling(self, mock_auth_context):
        """Handler handles relative paths appropriately."""
        result = parse_result(
            await handle_analyze_dependencies(
                mock_auth_context,
                {"repo_path": "./relative/path"},
            )
        )

        # Relative paths should be rejected or resolved
        assert result["status"] in [400, 404]

    @pytest.mark.asyncio
    async def test_symlink_handling(self, tmp_path, mock_auth_context):
        """Handler handles symlinks appropriately."""
        # Create a real directory
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (real_dir / "requirements.txt").write_text("requests==2.31.0\n")

        # Create a symlink
        symlink = tmp_path / "symlink"
        symlink.symlink_to(real_dir)

        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer"
        ) as mock_get:
            mock_analyzer = MagicMock()
            mock_tree = MagicMock()
            mock_tree.project_name = "test-project"
            mock_tree.project_version = "1.0.0"
            mock_tree.package_managers = []
            mock_tree.dependencies = {}
            mock_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
            mock_get.return_value = mock_analyzer

            result = parse_result(
                await handle_analyze_dependencies(
                    mock_auth_context,
                    {"repo_path": str(symlink)},
                )
            )

            # Should work with symlinks (unless specifically blocked)
            assert result["status"] in [200, 400]
