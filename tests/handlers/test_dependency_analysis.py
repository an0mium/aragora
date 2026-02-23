"""Tests for aragora.server.handlers.dependency_analysis module.

Covers all endpoints:
- POST /api/v1/codebase/analyze-dependencies
- POST /api/v1/codebase/sbom
- POST /api/v1/codebase/scan-vulnerabilities
- POST /api/v1/codebase/check-licenses
- POST /api/v1/codebase/clear-cache

Plus utility functions, caching, circuit breaker, path validation,
the DependencyAnalysisHandler class, and route registration.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.dependency_analysis import (
    DependencyAnalysisHandler,
    _analysis_cache,
    _analysis_cache_lock,
    _coerce_enum_value,
    _safe_int,
    _serialize_dependency,
    _validate_repo_path,
    get_dependency_analysis_routes,
    get_dependency_analyzer,
    get_dependency_circuit_breaker,
    handle_analyze_dependencies,
    handle_check_licenses,
    handle_clear_cache,
    handle_generate_sbom,
    handle_scan_vulnerabilities,
)


# =============================================================================
# Helper mock classes
# =============================================================================


class MockPackageManager(Enum):
    PIP = "pip"
    NPM = "npm"
    CARGO = "cargo"


class MockDepType(Enum):
    RUNTIME = "runtime"
    DEV = "dev"


class MockSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class MockDependency:
    name: str
    version: str | None = None
    dependency_type: MockDepType | None = None
    package_manager: MockPackageManager | None = None
    license: str | None = None
    purl: str | None = None


@dataclass
class MockDependencyTree:
    project_name: str = "test-project"
    project_version: str = "1.0.0"
    package_managers: list = field(default_factory=lambda: [MockPackageManager.PIP])
    dependencies: dict = field(default_factory=dict)
    total_direct: int = 3
    total_transitive: int = 5
    total_dev: int = 2
    analyzed_at: datetime = field(default_factory=lambda: datetime(2026, 2, 23, tzinfo=timezone.utc))


@dataclass
class MockVulnerability:
    id: str
    title: str
    description: str
    affected_package: str
    affected_versions: str
    fixed_version: str | None
    cvss_score: float
    cwe_id: str | None
    references: list[str]
    severity: MockSeverity


@dataclass
class MockLicenseConflict:
    package_b: str
    license_b: str
    conflict_type: str
    severity: str  # "error" or "warning"
    description: str


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level state between tests."""
    import aragora.server.handlers.dependency_analysis as mod

    original_analyzer = mod._dependency_analyzer
    # Clear cache
    with _analysis_cache_lock:
        _analysis_cache.clear()
    # Reset the dependency analyzer singleton
    mod._dependency_analyzer = None
    # Reset circuit breaker
    cb = get_dependency_circuit_breaker()
    cb.reset()
    yield
    # Restore
    mod._dependency_analyzer = original_analyzer
    with _analysis_cache_lock:
        _analysis_cache.clear()


@pytest.fixture
def auth_context():
    return AuthorizationContext(
        user_id="test-user",
        roles={"admin"},
        permissions={"*"},
    )


@pytest.fixture
def mock_analyzer():
    analyzer = AsyncMock()
    return analyzer


@pytest.fixture
def mock_tree():
    deps = {
        "requests": MockDependency(
            name="requests",
            version="2.31.0",
            dependency_type=MockDepType.RUNTIME,
            package_manager=MockPackageManager.PIP,
            license="Apache-2.0",
            purl="pkg:pypi/requests@2.31.0",
        ),
        "pytest": MockDependency(
            name="pytest",
            version="7.4.0",
            dependency_type=MockDepType.DEV,
            package_manager=MockPackageManager.PIP,
            license="MIT",
            purl="pkg:pypi/pytest@7.4.0",
        ),
    }
    return MockDependencyTree(dependencies=deps)


@pytest.fixture
def patch_analyzer(mock_analyzer):
    """Patch get_dependency_analyzer to return mock."""
    with patch(
        "aragora.server.handlers.dependency_analysis.get_dependency_analyzer",
        return_value=mock_analyzer,
    ):
        yield mock_analyzer


@pytest.fixture
def real_path(tmp_path):
    """Create a real temp directory for path validation tests."""
    return tmp_path


# =============================================================================
# Tests for _coerce_enum_value
# =============================================================================


class TestCoerceEnumValue:
    def test_none_returns_none(self):
        assert _coerce_enum_value(None) is None

    def test_enum_returns_value(self):
        assert _coerce_enum_value(MockPackageManager.PIP) == "pip"

    def test_string_returns_string(self):
        assert _coerce_enum_value("pip") == "pip"

    def test_int_returns_int(self):
        assert _coerce_enum_value(42) == 42

    def test_enum_with_int_value(self):
        class NumEnum(Enum):
            ONE = 1

        assert _coerce_enum_value(NumEnum.ONE) == 1


# =============================================================================
# Tests for _safe_int
# =============================================================================


class TestSafeInt:
    def test_int_returns_int(self):
        assert _safe_int(5) == 5

    def test_string_int_returns_int(self):
        assert _safe_int("10") == 10

    def test_none_returns_default(self):
        assert _safe_int(None) == 0

    def test_none_returns_custom_default(self):
        assert _safe_int(None, default=42) == 42

    def test_invalid_string_returns_default(self):
        assert _safe_int("not_a_number") == 0

    def test_float_string_returns_default(self):
        assert _safe_int("3.14") == 0

    def test_zero(self):
        assert _safe_int(0) == 0

    def test_negative_int(self):
        assert _safe_int(-5) == -5

    def test_bool_true(self):
        # bool is subclass of int in Python
        assert _safe_int(True) == 1

    def test_list_returns_default(self):
        assert _safe_int([1, 2, 3]) == 0


# =============================================================================
# Tests for _serialize_dependency
# =============================================================================


class TestSerializeDependency:
    def test_object_with_name_attr(self):
        dep = MockDependency(
            name="requests",
            version="2.31.0",
            dependency_type=MockDepType.RUNTIME,
            package_manager=MockPackageManager.PIP,
            license="Apache-2.0",
            purl="pkg:pypi/requests@2.31.0",
        )
        result = _serialize_dependency("requests", dep)
        assert result["name"] == "requests"
        assert result["version"] == "2.31.0"
        assert result["type"] == "runtime"
        assert result["package_manager"] == "pip"
        assert result["license"] == "Apache-2.0"
        assert result["purl"] == "pkg:pypi/requests@2.31.0"

    def test_dict_dependency(self):
        dep = {
            "version": "1.0.0",
            "type": "dev",
            "package_manager": "npm",
            "license": "MIT",
            "purl": "pkg:npm/lodash@1.0.0",
        }
        result = _serialize_dependency("lodash", dep)
        assert result["name"] == "lodash"
        assert result["version"] == "1.0.0"
        assert result["type"] == "dev"
        assert result["package_manager"] == "npm"
        assert result["license"] == "MIT"
        assert result["purl"] == "pkg:npm/lodash@1.0.0"

    def test_dict_with_dependency_type_key(self):
        dep = {"version": "2.0.0", "dependency_type": "runtime"}
        result = _serialize_dependency("express", dep)
        assert result["type"] == "runtime"

    def test_dict_with_no_optional_fields(self):
        dep = {}
        result = _serialize_dependency("some-pkg", dep)
        assert result["name"] == "some-pkg"
        assert result["version"] is None
        assert result["type"] is None

    def test_generic_object_fallback(self):
        """Object without .name attribute and not a dict."""

        class SimpleDep:
            version = "3.0.0"

        dep = SimpleDep()
        result = _serialize_dependency("generic-pkg", dep)
        assert result["name"] == "generic-pkg"
        assert result["version"] == "3.0.0"
        assert result["type"] is None
        assert result["package_manager"] is None
        assert result["license"] is None
        assert result["purl"] is None

    def test_object_with_enum_type_attrs(self):
        dep = MockDependency(
            name="flask",
            version="3.0.0",
            dependency_type=MockDepType.DEV,
            package_manager=MockPackageManager.NPM,
        )
        result = _serialize_dependency("flask", dep)
        assert result["type"] == "dev"
        assert result["package_manager"] == "npm"


# =============================================================================
# Tests for _validate_repo_path
# =============================================================================


class TestValidateRepoPath:
    def test_empty_path_returns_error(self):
        path, err = _validate_repo_path("")
        assert path is None
        assert err is not None
        assert err.status_code == 400

    def test_null_byte_returns_error(self):
        path, err = _validate_repo_path("/some/path\x00exploit")
        assert path is None
        assert err is not None
        assert err.status_code == 400

    def test_valid_path_returns_path(self, real_path):
        path, err = _validate_repo_path(str(real_path))
        assert err is None
        assert path == real_path

    def test_nonexistent_path_returns_404(self):
        path, err = _validate_repo_path("/nonexistent/path/abc123xyz")
        assert path is None
        assert err is not None
        assert err.status_code == 404

    def test_scan_root_enforced(self, real_path, monkeypatch):
        """When ARAGORA_SCAN_ROOT is set, paths must be within it."""
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", str(real_path))
        # Valid path within scan root
        sub = real_path / "sub"
        sub.mkdir()
        path, err = _validate_repo_path(str(sub))
        assert err is None
        assert path is not None

    def test_scan_root_blocks_outside_path(self, real_path, monkeypatch, tmp_path_factory):
        """Paths outside ARAGORA_SCAN_ROOT are rejected."""
        scan_root = real_path / "allowed"
        scan_root.mkdir()
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", str(scan_root))
        outside = tmp_path_factory.mktemp("outside")
        path, err = _validate_repo_path(str(outside))
        assert path is None
        assert err is not None
        assert err.status_code == 400

    def test_scan_root_as_filesystem_root(self, real_path, monkeypatch):
        """When ARAGORA_SCAN_ROOT is /, all paths are allowed."""
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/")
        path, err = _validate_repo_path(str(real_path))
        assert err is None
        assert path is not None

    def test_scan_root_exact_match(self, real_path, monkeypatch):
        """The scan root itself is a valid path."""
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", str(real_path))
        path, err = _validate_repo_path(str(real_path))
        assert err is None
        assert path is not None


# =============================================================================
# Tests for get_dependency_circuit_breaker
# =============================================================================


class TestGetDependencyCircuitBreaker:
    def test_returns_circuit_breaker(self):
        cb = get_dependency_circuit_breaker()
        assert cb is not None
        assert cb.name == "dependency_analysis_handler"

    def test_returns_same_instance(self):
        cb1 = get_dependency_circuit_breaker()
        cb2 = get_dependency_circuit_breaker()
        assert cb1 is cb2


# =============================================================================
# Tests for get_dependency_analyzer
# =============================================================================


class TestGetDependencyAnalyzer:
    def test_creates_analyzer_on_first_call(self):
        with patch(
            "aragora.server.handlers.dependency_analysis.DependencyAnalyzer",
            create=True,
        ) as mock_cls:
            # Patch the import inside get_dependency_analyzer
            mock_instance = MagicMock()
            with patch.dict(
                "sys.modules",
                {"aragora.audit.dependency_analyzer": MagicMock(DependencyAnalyzer=lambda: mock_instance)},
            ):
                result = get_dependency_analyzer()
                assert result is mock_instance

    def test_returns_cached_analyzer(self):
        import aragora.server.handlers.dependency_analysis as mod

        mock_analyzer = MagicMock()
        mod._dependency_analyzer = mock_analyzer
        result = get_dependency_analyzer()
        assert result is mock_analyzer

    def test_circuit_breaker_open_raises(self):
        cb = get_dependency_circuit_breaker()
        # Force circuit breaker open by recording many failures
        for _ in range(10):
            cb.record_failure()
        with pytest.raises(RuntimeError, match="temporarily unavailable"):
            get_dependency_analyzer()

    def test_import_error_records_failure(self):
        cb = get_dependency_circuit_breaker()
        with patch.dict("sys.modules", {"aragora.audit.dependency_analyzer": None}):
            with pytest.raises(ImportError):
                get_dependency_analyzer()


# =============================================================================
# Tests for handle_analyze_dependencies
# =============================================================================


class TestHandleAnalyzeDependencies:
    @pytest.mark.asyncio
    async def test_success(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        data = {"repo_path": str(real_path), "include_dev": True, "use_cache": False}
        result = await handle_analyze_dependencies(auth_context, data)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["data"]["project_name"] == "test-project"
        assert body["data"]["total_dependencies"] == 2
        assert body["data"]["from_cache"] is False

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, auth_context, patch_analyzer):
        data = {}
        result = await handle_analyze_dependencies(auth_context, data)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_nonexistent_repo_path(self, auth_context, patch_analyzer):
        data = {"repo_path": "/nonexistent/path/abc123"}
        result = await handle_analyze_dependencies(auth_context, data)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_cache_hit(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        data = {"repo_path": str(real_path), "include_dev": True, "use_cache": True}

        # First call populates cache
        result1 = await handle_analyze_dependencies(auth_context, data)
        assert result1.status_code == 200
        body1 = json.loads(result1.body)
        assert body1["data"]["from_cache"] is False

        # Second call should be from cache
        result2 = await handle_analyze_dependencies(auth_context, data)
        assert result2.status_code == 200
        body2 = json.loads(result2.body)
        assert body2["data"]["from_cache"] is True

    @pytest.mark.asyncio
    async def test_cache_bypass(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        data = {"repo_path": str(real_path), "include_dev": True, "use_cache": True}

        # First call
        await handle_analyze_dependencies(auth_context, data)

        # Second call with use_cache=False
        data["use_cache"] = False
        result = await handle_analyze_dependencies(auth_context, data)
        body = json.loads(result.body)
        assert body["data"]["from_cache"] is False

    @pytest.mark.asyncio
    async def test_cache_key_varies_by_include_dev(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)

        data_dev = {"repo_path": str(real_path), "include_dev": True, "use_cache": True}
        data_no_dev = {"repo_path": str(real_path), "include_dev": False, "use_cache": True}

        await handle_analyze_dependencies(auth_context, data_dev)
        # Different include_dev means different cache key
        result = await handle_analyze_dependencies(auth_context, data_no_dev)
        body = json.loads(result.body)
        assert body["data"]["from_cache"] is False

    @pytest.mark.asyncio
    async def test_dependencies_limited_to_100(self, auth_context, patch_analyzer, real_path):
        # Create 120 dependencies
        deps = {}
        for i in range(120):
            name = f"dep-{i}"
            deps[name] = MockDependency(name=name, version="1.0.0")
        tree = MockDependencyTree(dependencies=deps)
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=tree)

        data = {"repo_path": str(real_path), "use_cache": False}
        result = await handle_analyze_dependencies(auth_context, data)
        body = json.loads(result.body)
        assert body["data"]["total_dependencies"] == 120
        assert len(body["data"]["dependencies"]) == 100

    @pytest.mark.asyncio
    async def test_analyzer_runtime_error(self, auth_context, real_path):
        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer",
            side_effect=RuntimeError("service down"),
        ):
            data = {"repo_path": str(real_path)}
            result = await handle_analyze_dependencies(auth_context, data)
            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_default_include_dev_true(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        data = {"repo_path": str(real_path), "use_cache": False}
        await handle_analyze_dependencies(auth_context, data)
        patch_analyzer.resolve_dependencies.assert_called_once_with(
            repo_path=real_path, include_dev=True,
        )

    @pytest.mark.asyncio
    async def test_include_dev_false(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        data = {"repo_path": str(real_path), "include_dev": False, "use_cache": False}
        await handle_analyze_dependencies(auth_context, data)
        patch_analyzer.resolve_dependencies.assert_called_once_with(
            repo_path=real_path, include_dev=False,
        )

    @pytest.mark.asyncio
    async def test_result_includes_correct_fields(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        data = {"repo_path": str(real_path), "use_cache": False}
        result = await handle_analyze_dependencies(auth_context, data)
        body = json.loads(result.body)
        d = body["data"]
        assert "project_name" in d
        assert "project_version" in d
        assert "package_managers" in d
        assert "total_dependencies" in d
        assert "direct_dependencies" in d
        assert "transitive_dependencies" in d
        assert "dev_dependencies" in d
        assert "dependencies" in d
        assert "analyzed_at" in d
        assert "from_cache" in d

    @pytest.mark.asyncio
    async def test_package_managers_coerced(self, auth_context, patch_analyzer, real_path):
        tree = MockDependencyTree(
            package_managers=[MockPackageManager.PIP, MockPackageManager.NPM],
        )
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=tree)
        data = {"repo_path": str(real_path), "use_cache": False}
        result = await handle_analyze_dependencies(auth_context, data)
        body = json.loads(result.body)
        assert body["data"]["package_managers"] == ["pip", "npm"]

    @pytest.mark.asyncio
    async def test_null_byte_in_path(self, auth_context, patch_analyzer):
        data = {"repo_path": "/tmp/test\x00evil"}
        result = await handle_analyze_dependencies(auth_context, data)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_value_error_returns_500(self, auth_context, real_path):
        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer",
        ) as mock_get:
            mock_a = AsyncMock()
            mock_a.resolve_dependencies = AsyncMock(side_effect=ValueError("bad value"))
            mock_get.return_value = mock_a
            data = {"repo_path": str(real_path), "use_cache": False}
            result = await handle_analyze_dependencies(auth_context, data)
            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_os_error_returns_500(self, auth_context, real_path):
        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer",
        ) as mock_get:
            mock_a = AsyncMock()
            mock_a.resolve_dependencies = AsyncMock(side_effect=OSError("disk error"))
            mock_get.return_value = mock_a
            data = {"repo_path": str(real_path), "use_cache": False}
            result = await handle_analyze_dependencies(auth_context, data)
            assert result.status_code == 500


# =============================================================================
# Tests for handle_generate_sbom
# =============================================================================


class TestHandleGenerateSbom:
    @pytest.mark.asyncio
    async def test_success_cyclonedx(self, auth_context, patch_analyzer, mock_tree, real_path):
        sbom_data = {"bomFormat": "CycloneDX", "components": []}
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.generate_sbom = AsyncMock(return_value=json.dumps(sbom_data))

        data = {"repo_path": str(real_path), "format": "cyclonedx"}
        result = await handle_generate_sbom(auth_context, data)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["data"]["format"] == "cyclonedx"
        assert body["data"]["project_name"] == "test-project"
        assert body["data"]["component_count"] == 2
        assert body["data"]["sbom"] == sbom_data
        assert body["data"]["sbom_json"] == json.dumps(sbom_data)

    @pytest.mark.asyncio
    async def test_success_spdx(self, auth_context, patch_analyzer, mock_tree, real_path):
        sbom_data = {"spdxVersion": "SPDX-2.3"}
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.generate_sbom = AsyncMock(return_value=json.dumps(sbom_data))

        data = {"repo_path": str(real_path), "format": "spdx"}
        result = await handle_generate_sbom(auth_context, data)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["data"]["format"] == "spdx"

    @pytest.mark.asyncio
    async def test_default_format_cyclonedx(self, auth_context, patch_analyzer, mock_tree, real_path):
        sbom_data = {"bomFormat": "CycloneDX"}
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.generate_sbom = AsyncMock(return_value=json.dumps(sbom_data))

        data = {"repo_path": str(real_path)}
        result = await handle_generate_sbom(auth_context, data)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["data"]["format"] == "cyclonedx"

    @pytest.mark.asyncio
    async def test_invalid_format(self, auth_context, patch_analyzer, real_path):
        data = {"repo_path": str(real_path), "format": "invalid_format"}
        result = await handle_generate_sbom(auth_context, data)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, auth_context, patch_analyzer):
        data = {}
        result = await handle_generate_sbom(auth_context, data)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_include_vulnerabilities_true(self, auth_context, patch_analyzer, mock_tree, real_path):
        sbom_data = {"bomFormat": "CycloneDX"}
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.generate_sbom = AsyncMock(return_value=json.dumps(sbom_data))

        data = {"repo_path": str(real_path), "include_vulnerabilities": True}
        await handle_generate_sbom(auth_context, data)
        patch_analyzer.generate_sbom.assert_called_once_with(
            tree=mock_tree, format="cyclonedx", include_vulnerabilities=True,
        )

    @pytest.mark.asyncio
    async def test_include_vulnerabilities_false(self, auth_context, patch_analyzer, mock_tree, real_path):
        sbom_data = {"bomFormat": "CycloneDX"}
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.generate_sbom = AsyncMock(return_value=json.dumps(sbom_data))

        data = {"repo_path": str(real_path), "include_vulnerabilities": False}
        await handle_generate_sbom(auth_context, data)
        patch_analyzer.generate_sbom.assert_called_once_with(
            tree=mock_tree, format="cyclonedx", include_vulnerabilities=False,
        )

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, auth_context, real_path):
        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer",
            side_effect=RuntimeError("unavailable"),
        ):
            data = {"repo_path": str(real_path)}
            result = await handle_generate_sbom(auth_context, data)
            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_json_decode_error_returns_500(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.generate_sbom = AsyncMock(return_value="not valid json {{{")

        data = {"repo_path": str(real_path)}
        result = await handle_generate_sbom(auth_context, data)
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, auth_context, patch_analyzer):
        data = {"repo_path": "/nonexistent/abc123xyz"}
        result = await handle_generate_sbom(auth_context, data)
        assert result.status_code == 404


# =============================================================================
# Tests for handle_scan_vulnerabilities
# =============================================================================


class TestHandleScanVulnerabilities:
    @pytest.mark.asyncio
    async def test_success_no_vulns(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_vulnerabilities = AsyncMock(return_value=[])

        data = {"repo_path": str(real_path)}
        result = await handle_scan_vulnerabilities(auth_context, data)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["data"]["total_vulnerabilities"] == 0
        assert body["data"]["critical_count"] == 0
        assert body["data"]["high_count"] == 0
        assert body["data"]["medium_count"] == 0
        assert body["data"]["low_count"] == 0

    @pytest.mark.asyncio
    async def test_success_with_vulns(self, auth_context, patch_analyzer, mock_tree, real_path):
        vulns = [
            MockVulnerability(
                id="CVE-2024-0001",
                title="Critical RCE",
                description="Remote code execution",
                affected_package="requests",
                affected_versions="<2.32.0",
                fixed_version="2.32.0",
                cvss_score=9.8,
                cwe_id="CWE-94",
                references=["https://nvd.nist.gov/CVE-2024-0001"],
                severity=MockSeverity.CRITICAL,
            ),
            MockVulnerability(
                id="CVE-2024-0002",
                title="High severity XSS",
                description="Cross-site scripting",
                affected_package="flask",
                affected_versions="<3.1.0",
                fixed_version="3.1.0",
                cvss_score=7.5,
                cwe_id="CWE-79",
                references=[],
                severity=MockSeverity.HIGH,
            ),
            MockVulnerability(
                id="CVE-2024-0003",
                title="Medium info leak",
                description="Information disclosure",
                affected_package="requests",
                affected_versions="<2.31.1",
                fixed_version="2.31.1",
                cvss_score=5.3,
                cwe_id=None,
                references=[],
                severity=MockSeverity.MEDIUM,
            ),
        ]
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_vulnerabilities = AsyncMock(return_value=vulns)

        data = {"repo_path": str(real_path)}
        result = await handle_scan_vulnerabilities(auth_context, data)
        assert result.status_code == 200
        body = json.loads(result.body)
        d = body["data"]
        assert d["total_vulnerabilities"] == 3
        assert d["critical_count"] == 1
        assert d["high_count"] == 1
        assert d["medium_count"] == 1
        assert d["low_count"] == 0
        assert len(d["vulnerabilities_by_severity"]["critical"]) == 1
        assert d["vulnerabilities_by_severity"]["critical"][0]["id"] == "CVE-2024-0001"
        assert d["scan_summary"]["packages_scanned"] == 2
        assert d["scan_summary"]["packages_with_vulnerabilities"] == 2

    @pytest.mark.asyncio
    async def test_unknown_severity_grouped(self, auth_context, patch_analyzer, mock_tree, real_path):
        vulns = [
            MockVulnerability(
                id="CVE-2024-9999",
                title="Unknown",
                description="Unknown severity vuln",
                affected_package="pkg",
                affected_versions="*",
                fixed_version=None,
                cvss_score=0.0,
                cwe_id=None,
                references=[],
                severity=MockSeverity.UNKNOWN,
            ),
        ]
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_vulnerabilities = AsyncMock(return_value=vulns)

        data = {"repo_path": str(real_path)}
        result = await handle_scan_vulnerabilities(auth_context, data)
        body = json.loads(result.body)
        assert len(body["data"]["vulnerabilities_by_severity"]["unknown"]) == 1

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, auth_context, patch_analyzer):
        data = {}
        result = await handle_scan_vulnerabilities(auth_context, data)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, auth_context, patch_analyzer):
        data = {"repo_path": "/nonexistent/abc123xyz"}
        result = await handle_scan_vulnerabilities(auth_context, data)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, auth_context, real_path):
        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer",
            side_effect=RuntimeError("unavailable"),
        ):
            data = {"repo_path": str(real_path)}
            result = await handle_scan_vulnerabilities(auth_context, data)
            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_vuln_fields_serialized_correctly(self, auth_context, patch_analyzer, mock_tree, real_path):
        vulns = [
            MockVulnerability(
                id="CVE-2024-1234",
                title="Test vuln",
                description="A test vulnerability",
                affected_package="test-pkg",
                affected_versions=">=1.0, <2.0",
                fixed_version="2.0.0",
                cvss_score=6.5,
                cwe_id="CWE-200",
                references=["https://example.com/advisory"],
                severity=MockSeverity.MEDIUM,
            ),
        ]
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_vulnerabilities = AsyncMock(return_value=vulns)

        data = {"repo_path": str(real_path)}
        result = await handle_scan_vulnerabilities(auth_context, data)
        body = json.loads(result.body)
        vuln_data = body["data"]["vulnerabilities_by_severity"]["medium"][0]
        assert vuln_data["id"] == "CVE-2024-1234"
        assert vuln_data["title"] == "Test vuln"
        assert vuln_data["description"] == "A test vulnerability"
        assert vuln_data["affected_package"] == "test-pkg"
        assert vuln_data["affected_versions"] == ">=1.0, <2.0"
        assert vuln_data["fixed_version"] == "2.0.0"
        assert vuln_data["cvss_score"] == 6.5
        assert vuln_data["cwe_id"] == "CWE-200"
        assert vuln_data["references"] == ["https://example.com/advisory"]

    @pytest.mark.asyncio
    async def test_low_severity_grouped(self, auth_context, patch_analyzer, mock_tree, real_path):
        vulns = [
            MockVulnerability(
                id="CVE-2024-0010",
                title="Low severity",
                description="Minor issue",
                affected_package="pkg-low",
                affected_versions="*",
                fixed_version=None,
                cvss_score=2.0,
                cwe_id=None,
                references=[],
                severity=MockSeverity.LOW,
            ),
        ]
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_vulnerabilities = AsyncMock(return_value=vulns)

        data = {"repo_path": str(real_path)}
        result = await handle_scan_vulnerabilities(auth_context, data)
        body = json.loads(result.body)
        assert body["data"]["low_count"] == 1
        assert len(body["data"]["vulnerabilities_by_severity"]["low"]) == 1

    @pytest.mark.asyncio
    async def test_multiple_vulns_same_package(self, auth_context, patch_analyzer, mock_tree, real_path):
        vulns = [
            MockVulnerability(
                id="CVE-A", title="A", description="A", affected_package="requests",
                affected_versions="*", fixed_version=None, cvss_score=9.0,
                cwe_id=None, references=[], severity=MockSeverity.CRITICAL,
            ),
            MockVulnerability(
                id="CVE-B", title="B", description="B", affected_package="requests",
                affected_versions="*", fixed_version=None, cvss_score=7.0,
                cwe_id=None, references=[], severity=MockSeverity.HIGH,
            ),
        ]
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_vulnerabilities = AsyncMock(return_value=vulns)

        data = {"repo_path": str(real_path)}
        result = await handle_scan_vulnerabilities(auth_context, data)
        body = json.loads(result.body)
        assert body["data"]["total_vulnerabilities"] == 2
        # Only 1 unique package has vulnerabilities
        assert body["data"]["scan_summary"]["packages_with_vulnerabilities"] == 1


# =============================================================================
# Tests for handle_check_licenses
# =============================================================================


class TestHandleCheckLicenses:
    @pytest.mark.asyncio
    async def test_success_no_conflicts(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_license_compatibility = AsyncMock(return_value=[])

        data = {"repo_path": str(real_path)}
        result = await handle_check_licenses(auth_context, data)
        assert result.status_code == 200
        body = json.loads(result.body)
        d = body["data"]
        assert d["compatible"] is True
        assert d["total_conflicts"] == 0
        assert d["error_count"] == 0
        assert d["warning_count"] == 0

    @pytest.mark.asyncio
    async def test_success_with_conflicts(self, auth_context, patch_analyzer, mock_tree, real_path):
        conflicts = [
            MockLicenseConflict(
                package_b="gpl-pkg",
                license_b="GPL-3.0",
                conflict_type="copyleft_incompatible",
                severity="error",
                description="GPL-3.0 is incompatible with MIT",
            ),
            MockLicenseConflict(
                package_b="lgpl-pkg",
                license_b="LGPL-2.1",
                conflict_type="weak_copyleft",
                severity="warning",
                description="LGPL requires linking compliance",
            ),
        ]
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_license_compatibility = AsyncMock(return_value=conflicts)

        data = {"repo_path": str(real_path)}
        result = await handle_check_licenses(auth_context, data)
        assert result.status_code == 200
        body = json.loads(result.body)
        d = body["data"]
        assert d["compatible"] is False
        assert d["total_conflicts"] == 2
        assert d["error_count"] == 1
        assert d["warning_count"] == 1
        assert d["conflicts"][0]["package"] == "gpl-pkg"
        assert d["conflicts"][0]["license"] == "GPL-3.0"
        assert d["conflicts"][0]["severity"] == "error"

    @pytest.mark.asyncio
    async def test_default_project_license_mit(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_license_compatibility = AsyncMock(return_value=[])

        data = {"repo_path": str(real_path)}
        result = await handle_check_licenses(auth_context, data)
        body = json.loads(result.body)
        assert body["data"]["project_license"] == "MIT"
        patch_analyzer.check_license_compatibility.assert_called_once_with(
            tree=mock_tree, project_license="MIT",
        )

    @pytest.mark.asyncio
    async def test_custom_project_license(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_license_compatibility = AsyncMock(return_value=[])

        data = {"repo_path": str(real_path), "project_license": "Apache-2.0"}
        result = await handle_check_licenses(auth_context, data)
        body = json.loads(result.body)
        assert body["data"]["project_license"] == "Apache-2.0"

    @pytest.mark.asyncio
    async def test_license_distribution(self, auth_context, patch_analyzer, mock_tree, real_path):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_license_compatibility = AsyncMock(return_value=[])

        data = {"repo_path": str(real_path)}
        result = await handle_check_licenses(auth_context, data)
        body = json.loads(result.body)
        dist = body["data"]["license_distribution"]
        assert dist["Apache-2.0"] == 1
        assert dist["MIT"] == 1

    @pytest.mark.asyncio
    async def test_unknown_license_counted(self, auth_context, patch_analyzer, real_path):
        deps = {
            "no-license-pkg": MockDependency(name="no-license-pkg", version="1.0", license=None),
        }
        tree = MockDependencyTree(dependencies=deps)
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=tree)
        patch_analyzer.check_license_compatibility = AsyncMock(return_value=[])

        data = {"repo_path": str(real_path)}
        result = await handle_check_licenses(auth_context, data)
        body = json.loads(result.body)
        assert body["data"]["license_distribution"]["Unknown"] == 1

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, auth_context, patch_analyzer):
        data = {}
        result = await handle_check_licenses(auth_context, data)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, auth_context, patch_analyzer):
        data = {"repo_path": "/nonexistent/abc123xyz"}
        result = await handle_check_licenses(auth_context, data)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, auth_context, real_path):
        with patch(
            "aragora.server.handlers.dependency_analysis.get_dependency_analyzer",
            side_effect=RuntimeError("unavailable"),
        ):
            data = {"repo_path": str(real_path)}
            result = await handle_check_licenses(auth_context, data)
            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_warnings_only_still_compatible(self, auth_context, patch_analyzer, mock_tree, real_path):
        conflicts = [
            MockLicenseConflict(
                package_b="warn-pkg",
                license_b="LGPL-2.1",
                conflict_type="weak_copyleft",
                severity="warning",
                description="Minor concern",
            ),
        ]
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_license_compatibility = AsyncMock(return_value=conflicts)

        data = {"repo_path": str(real_path)}
        result = await handle_check_licenses(auth_context, data)
        body = json.loads(result.body)
        # No errors means compatible is True
        assert body["data"]["compatible"] is True
        assert body["data"]["warning_count"] == 1


# =============================================================================
# Tests for handle_clear_cache
# =============================================================================


class TestHandleClearCache:
    @pytest.mark.asyncio
    async def test_clear_empty_cache(self):
        result = await handle_clear_cache()
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["data"]["cleared"] is True
        assert body["data"]["entries_removed"] == 0

    @pytest.mark.asyncio
    async def test_clear_populated_cache(self):
        with _analysis_cache_lock:
            _analysis_cache["key1"] = {"test": "data1"}
            _analysis_cache["key2"] = {"test": "data2"}

        result = await handle_clear_cache()
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["data"]["cleared"] is True
        assert body["data"]["entries_removed"] == 2

        with _analysis_cache_lock:
            assert len(_analysis_cache) == 0

    @pytest.mark.asyncio
    async def test_clear_cache_returns_correct_count(self):
        with _analysis_cache_lock:
            for i in range(5):
                _analysis_cache[f"key{i}"] = {"data": i}

        result = await handle_clear_cache()
        body = json.loads(result.body)
        assert body["data"]["entries_removed"] == 5


# =============================================================================
# Tests for get_dependency_analysis_routes
# =============================================================================


class TestGetDependencyAnalysisRoutes:
    def test_returns_five_routes(self):
        routes = get_dependency_analysis_routes()
        assert len(routes) == 5

    def test_all_routes_are_post(self):
        routes = get_dependency_analysis_routes()
        for method, path, handler in routes:
            assert method == "POST"

    def test_routes_have_correct_paths(self):
        routes = get_dependency_analysis_routes()
        paths = [path for _, path, _ in routes]
        assert "/api/v1/codebase/analyze-dependencies" in paths
        assert "/api/v1/codebase/sbom" in paths
        assert "/api/v1/codebase/scan-vulnerabilities" in paths
        assert "/api/v1/codebase/check-licenses" in paths
        assert "/api/v1/codebase/clear-cache" in paths

    def test_routes_have_callable_handlers(self):
        routes = get_dependency_analysis_routes()
        for _, _, handler in routes:
            assert callable(handler)

    def test_routes_tuple_structure(self):
        routes = get_dependency_analysis_routes()
        for route in routes:
            assert len(route) == 3
            assert isinstance(route[0], str)
            assert isinstance(route[1], str)


# =============================================================================
# Tests for DependencyAnalysisHandler class
# =============================================================================


class TestDependencyAnalysisHandler:
    @pytest.fixture
    def handler(self):
        ctx = {"config": {}}
        return DependencyAnalysisHandler(ctx)

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) == 5

    def test_can_handle_valid_paths(self, handler):
        assert handler.can_handle("/api/v1/codebase/analyze-dependencies") is True
        assert handler.can_handle("/api/v1/codebase/sbom") is True
        assert handler.can_handle("/api/v1/codebase/scan-vulnerabilities") is True
        assert handler.can_handle("/api/v1/codebase/check-licenses") is True
        assert handler.can_handle("/api/v1/codebase/clear-cache") is True

    def test_can_handle_invalid_path(self, handler):
        assert handler.can_handle("/api/v1/other") is False
        assert handler.can_handle("/api/v1/codebase/unknown") is False
        assert handler.can_handle("") is False

    def test_handle_returns_none(self, handler):
        """The sync handle method returns None (all routing is in handle_post)."""
        result = handler.handle("/api/v1/codebase/sbom", {}, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_analyze_dependencies(self, handler, real_path, patch_analyzer, mock_tree):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        data = {"repo_path": str(real_path), "use_cache": False}
        result = await handler.handle_post("/api/v1/codebase/analyze-dependencies", data)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_sbom(self, handler, real_path, patch_analyzer, mock_tree):
        sbom_data = {"bomFormat": "CycloneDX"}
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.generate_sbom = AsyncMock(return_value=json.dumps(sbom_data))
        data = {"repo_path": str(real_path)}
        result = await handler.handle_post("/api/v1/codebase/sbom", data)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_scan_vulnerabilities(self, handler, real_path, patch_analyzer, mock_tree):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_vulnerabilities = AsyncMock(return_value=[])
        data = {"repo_path": str(real_path)}
        result = await handler.handle_post("/api/v1/codebase/scan-vulnerabilities", data)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_check_licenses(self, handler, real_path, patch_analyzer, mock_tree):
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        patch_analyzer.check_license_compatibility = AsyncMock(return_value=[])
        data = {"repo_path": str(real_path)}
        result = await handler.handle_post("/api/v1/codebase/check-licenses", data)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_clear_cache(self, handler):
        result = await handler.handle_post("/api/v1/codebase/clear-cache", {})
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_unknown_path(self, handler):
        result = await handler.handle_post("/api/v1/codebase/unknown", {})
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_handle_post_with_handler_object(self, handler, real_path, patch_analyzer, mock_tree):
        """When handler is provided, read_json_body is called."""
        patch_analyzer.resolve_dependencies = AsyncMock(return_value=mock_tree)
        mock_http_handler = MagicMock()
        handler.read_json_body = MagicMock(return_value={
            "repo_path": str(real_path), "use_cache": False,
        })
        result = await handler.handle_post(
            "/api/v1/codebase/analyze-dependencies", {}, mock_http_handler,
        )
        handler.read_json_body.assert_called_once_with(mock_http_handler)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_with_none_handler_uses_query_params(self, handler, patch_analyzer):
        """When handler is None, query_params are used as data."""
        result = await handler.handle_post("/api/v1/codebase/clear-cache", None)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_read_json_body_returns_none(self, handler, patch_analyzer):
        """When read_json_body returns None, empty dict is used."""
        mock_http_handler = MagicMock()
        handler.read_json_body = MagicMock(return_value=None)
        result = await handler.handle_post(
            "/api/v1/codebase/clear-cache", {}, mock_http_handler,
        )
        assert result.status_code == 200
