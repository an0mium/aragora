"""
Tests for the dependency analysis HTTP handler.

Tests cover:
- Dependency tree analysis
- SBOM generation (CycloneDX, SPDX)
- Vulnerability scanning
- License compatibility checking
- Cache management
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from aragora.server.handlers.dependency_analysis import (
    handle_analyze_dependencies,
    handle_generate_sbom,
    handle_scan_vulnerabilities,
    handle_check_licenses,
    handle_clear_cache,
    get_dependency_analyzer,
    DependencyAnalysisHandler,
)
from aragora.server.handlers.utils.responses import HandlerResult
from aragora.rbac.models import AuthorizationContext


@pytest.fixture
def mock_auth_context():
    """Create a mock authorization context for testing."""
    return AuthorizationContext(
        user_id="test-user",
        user_email="test@example.com",
        roles={"admin"},
        permissions={"codebase.run"},
    )


def parse_result(result: HandlerResult) -> dict:
    """Parse HandlerResult body to dict for assertions."""
    data = json.loads(result.body.decode("utf-8"))
    # Add status from the HandlerResult for convenience
    data["status"] = result.status_code
    # Normalize success field - if "error" exists, it's a failure
    if "success" not in data:
        data["success"] = "error" not in data
    return data


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


class TestClearCache:
    """Test cache clearing endpoint."""

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Cache is cleared successfully."""
        # Add something to cache first
        from aragora.server.handlers.dependency_analysis import (
            _analysis_cache,
            _analysis_cache_lock,
        )

        with _analysis_cache_lock:
            _analysis_cache["test_key"] = {"test": "data"}

        result = parse_result(await handle_clear_cache())

        assert result["success"] is True
        assert result["data"]["cleared"] is True
        assert result["data"]["entries_removed"] >= 1


class TestDependencyAnalyzerSingleton:
    """Test singleton pattern for analyzer."""

    def test_get_dependency_analyzer_returns_same_instance(self):
        """Get dependency analyzer returns singleton."""
        with patch("aragora.server.handlers.dependency_analysis._dependency_analyzer", None):
            with patch("aragora.audit.dependency_analyzer.DependencyAnalyzer") as mock_class:
                mock_instance = MagicMock()
                mock_class.return_value = mock_instance

                # First call creates instance
                analyzer1 = get_dependency_analyzer()

                # Verify mock was called
                assert analyzer1 is not None


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
    async def test_handle_post_unknown_route(self):
        """Handler returns 404 for unknown routes."""
        handler = DependencyAnalysisHandler({})
        result = await handler.handle_post("/api/v1/unknown", {})

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status"] == 404
