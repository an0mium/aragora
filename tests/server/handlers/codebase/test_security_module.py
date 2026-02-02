"""
Comprehensive tests for aragora.server.handlers.codebase.security module.

Tests cover:
- Path traversal prevention (safe_repo_id validation)
- Repository scanning endpoints
- CVE query endpoints
- Secrets scanning endpoints
- SAST scanning endpoints
- SBOM generation endpoints
- Input validation
- Error handling
- SecurityHandler class methods
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Path Traversal Protection Tests (safe_repo_id)
# =============================================================================


class TestSafeRepoId:
    """Test safe_repo_id validation function to prevent path traversal attacks."""

    def test_valid_repo_id_simple(self):
        """Test valid simple repo ID."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("my-repo")
        assert is_valid is True
        assert err is None

    def test_valid_repo_id_with_numbers(self):
        """Test valid repo ID with numbers."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("repo123")
        assert is_valid is True
        assert err is None

    def test_valid_repo_id_with_underscore(self):
        """Test valid repo ID with underscores."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("my_repo_123")
        assert is_valid is True
        assert err is None

    def test_valid_repo_id_with_hyphen(self):
        """Test valid repo ID with hyphens."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("my-repo-123")
        assert is_valid is True
        assert err is None

    def test_valid_repo_id_mixed(self):
        """Test valid repo ID with mixed characters."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("My_Repo-123_test")
        assert is_valid is True
        assert err is None

    def test_invalid_repo_id_empty(self):
        """Test empty repo ID is rejected."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("")
        assert is_valid is False
        assert "cannot be empty" in err.lower()

    def test_invalid_repo_id_path_traversal_basic(self):
        """Test basic path traversal attempt is blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("../etc/passwd")
        assert is_valid is False
        assert "path traversal" in err.lower()

    def test_invalid_repo_id_path_traversal_double_dot(self):
        """Test double dot path traversal is blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("..")
        assert is_valid is False
        assert "path traversal" in err.lower()

    def test_invalid_repo_id_path_traversal_embedded(self):
        """Test embedded path traversal is blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("repo/../secrets")
        assert is_valid is False
        assert "path traversal" in err.lower()

    def test_invalid_repo_id_path_traversal_windows(self):
        """Test Windows-style path traversal is blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("..\\windows\\system32")
        assert is_valid is False
        # Can match either path traversal or path separator error

    def test_invalid_repo_id_forward_slash(self):
        """Test forward slash in repo ID is blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("repo/subdir")
        assert is_valid is False
        assert "path separator" in err.lower()

    def test_invalid_repo_id_backslash(self):
        """Test backslash in repo ID is blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("repo\\subdir")
        assert is_valid is False
        assert "path separator" in err.lower()

    def test_invalid_repo_id_absolute_path(self):
        """Test absolute path is blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("/etc/passwd")
        assert is_valid is False
        assert "path separator" in err.lower()

    def test_invalid_repo_id_special_chars(self):
        """Test special characters are blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("repo@#$%")
        assert is_valid is False
        assert "alphanumeric" in err.lower()

    def test_invalid_repo_id_spaces(self):
        """Test spaces in repo ID are blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("my repo")
        assert is_valid is False
        assert "alphanumeric" in err.lower()

    def test_invalid_repo_id_null_byte(self):
        """Test null byte injection is blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("repo\x00.txt")
        assert is_valid is False

    def test_invalid_repo_id_unicode_escape(self):
        """Test unicode path separator is blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        # Unicode forward slash
        is_valid, err = safe_repo_id("repo\u2215subdir")
        assert is_valid is False

    def test_invalid_repo_id_too_long(self):
        """Test repo ID exceeding max length is blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        # SAFE_ID_PATTERN allows 1-64 characters
        is_valid, err = safe_repo_id("a" * 65)
        assert is_valid is False
        assert "alphanumeric" in err.lower()

    def test_valid_repo_id_max_length(self):
        """Test repo ID at max length is accepted."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        is_valid, err = safe_repo_id("a" * 64)
        assert is_valid is True
        assert err is None

    def test_invalid_repo_id_url_encoded(self):
        """Test URL-encoded path traversal is blocked."""
        from aragora.server.handlers.codebase.security.storage import safe_repo_id

        # %2e%2e = ..
        is_valid, err = safe_repo_id("%2e%2e/etc")
        assert is_valid is False


# =============================================================================
# SecurityHandler Path Traversal Tests
# =============================================================================


class TestSecurityHandlerPathTraversal:
    """Test that SecurityHandler validates repo_id in all endpoints."""

    @pytest.fixture
    def handler(self):
        """Create SecurityHandler instance."""
        from aragora.server.handlers.codebase.security.handler import SecurityHandler

        return SecurityHandler(ctx={})

    @pytest.mark.asyncio
    async def test_handle_post_scan_rejects_path_traversal(self, handler):
        """Test POST scan rejects path traversal in repo_id."""
        result = await handler.handle_post_scan(
            data={"repo_path": "/path/to/repo"},
            repo_id="../etc/passwd",
        )
        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "path traversal" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_handle_get_scan_latest_rejects_path_traversal(self, handler):
        """Test GET scan latest rejects path traversal."""
        result = await handler.handle_get_scan_latest(
            params={},
            repo_id="../../../etc/passwd",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_get_scan_rejects_path_traversal(self, handler):
        """Test GET specific scan rejects path traversal."""
        result = await handler.handle_get_scan(
            params={},
            repo_id="repo/../secret",
            scan_id="scan_123",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_get_vulnerabilities_rejects_path_traversal(self, handler):
        """Test GET vulnerabilities rejects path traversal."""
        result = await handler.handle_get_vulnerabilities(
            params={},
            repo_id="/etc/passwd",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_list_scans_rejects_path_traversal(self, handler):
        """Test list scans rejects path traversal."""
        result = await handler.handle_list_scans(
            params={},
            repo_id="..\\windows\\system32",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_post_secrets_scan_rejects_path_traversal(self, handler):
        """Test POST secrets scan rejects path traversal."""
        result = await handler.handle_post_secrets_scan(
            data={"repo_path": "/path/to/repo"},
            repo_id="../../../",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_get_secrets_rejects_path_traversal(self, handler):
        """Test GET secrets rejects path traversal."""
        result = await handler.handle_get_secrets(
            params={},
            repo_id="repo/../../",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_scan_sast_rejects_path_traversal(self, handler):
        """Test SAST scan rejects path traversal."""
        result = await handler.handle_scan_sast(
            params={"repo_path": "/path/to/repo"},
            repo_id="..%2f..%2f",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_get_sast_findings_rejects_path_traversal(self, handler):
        """Test GET SAST findings rejects path traversal."""
        result = await handler.handle_get_sast_findings(
            params={},
            repo_id="../",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_post_sbom_rejects_path_traversal(self, handler):
        """Test POST SBOM rejects path traversal."""
        result = await handler.handle_post_sbom(
            data={"repo_path": "/path/to/repo"},
            repo_id="../../etc/shadow",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_get_sbom_latest_rejects_path_traversal(self, handler):
        """Test GET SBOM latest rejects path traversal."""
        result = await handler.handle_get_sbom_latest(
            params={},
            repo_id="/root/.ssh",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_compare_sbom_rejects_path_traversal(self, handler):
        """Test SBOM compare rejects path traversal."""
        result = await handler.handle_compare_sbom(
            data={"sbom_id_a": "sbom_1", "sbom_id_b": "sbom_2"},
            repo_id="..\\..\\windows",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_valid_repo_id_passes_validation(self, handler):
        """Test that valid repo_id passes validation (and proceeds to next check)."""
        # With valid repo_id but missing repo_path, should get 400 for missing path
        result = await handler.handle_post_scan(
            data={},  # Missing repo_path
            repo_id="valid-repo-123",
        )
        assert result.status_code == 400
        body = json.loads(result.body.decode())
        # Error should be about repo_path, not repo_id
        assert "repo_path required" in body.get("error", "").lower()


# =============================================================================
# Mock Classes
# =============================================================================


class MockScanResult:
    """Mock scan result for testing."""

    def __init__(
        self,
        scan_id: str = "scan_123",
        repository: str = "test-repo",
        status: str = "completed",
        dependencies: list | None = None,
    ):
        self.scan_id = scan_id
        self.repository = repository
        self.status = status
        self.branch = "main"
        self.commit_sha = "abc123"
        self.started_at = datetime.now(timezone.utc)
        self.completed_at = datetime.now(timezone.utc)
        self.error = None
        self.dependencies = dependencies or []
        self.total_dependencies = len(self.dependencies)
        self.vulnerable_dependencies = sum(1 for d in self.dependencies if d.vulnerabilities)
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
            "dependencies": [
                d.to_dict() if hasattr(d, "to_dict") else d for d in self.dependencies
            ],
        }


class MockDependency:
    """Mock dependency with vulnerabilities."""

    def __init__(self, name: str, version: str, ecosystem: str = "npm"):
        self.name = name
        self.version = version
        self.ecosystem = ecosystem
        self.vulnerabilities = []

    def add_vulnerability(self, cve_id: str, severity: str):
        vuln = MagicMock()
        vuln.id = cve_id
        vuln.cve_id = cve_id
        vuln.severity = severity
        vuln.title = f"Test vulnerability {cve_id}"
        vuln.description = f"Description for {cve_id}"
        vuln.remediation_guidance = "Update to latest version"
        vuln.to_dict = lambda: {"cve_id": cve_id, "severity": severity}
        self.vulnerabilities.append(vuln)
        return self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "ecosystem": self.ecosystem,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
        }


class MockSecretsScanResult:
    """Mock secrets scan result."""

    def __init__(self, scan_id: str = "secrets_123", secrets: list | None = None):
        self.scan_id = scan_id
        self.status = "completed"
        self.started_at = datetime.now(timezone.utc)
        self.completed_at = datetime.now(timezone.utc)
        self.secrets = secrets or []
        self.files_scanned = 100
        self.scanned_history = False
        self.history_depth = 0
        self.critical_count = 0
        self.high_count = 0
        self.medium_count = 0
        self.low_count = 0

    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "secrets": [s.to_dict() if hasattr(s, "to_dict") else s for s in self.secrets],
        }


class MockSASTScanResult:
    """Mock SAST scan result."""

    def __init__(self, scan_id: str = "sast_123", findings: list | None = None):
        self.scan_id = scan_id
        self.status = "completed"
        self.scanned_at = datetime.now(timezone.utc)
        self.findings = findings or []

    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "status": self.status,
            "findings": [f.to_dict() if hasattr(f, "to_dict") else f for f in self.findings],
        }


class MockSBOMResult:
    """Mock SBOM result."""

    def __init__(self, sbom_id: str = "sbom_123"):
        self.sbom_id = sbom_id
        self.format = MagicMock(value="cyclonedx-json")
        self.filename = "sbom.json"
        self.component_count = 50
        self.vulnerability_count = 5
        self.license_count = 10
        self.generated_at = datetime.now(timezone.utc)
        self.content = '{"bomFormat": "CycloneDX"}'
        self.errors = []

    def to_dict(self) -> dict:
        return {
            "sbom_id": self.sbom_id,
            "format": self.format.value,
            "filename": self.filename,
            "component_count": self.component_count,
        }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_scanner():
    """Create mock dependency scanner."""
    scanner = MagicMock()
    scanner.scan_repository = AsyncMock(return_value=MockScanResult())
    return scanner


@pytest.fixture
def mock_secrets_scanner():
    """Create mock secrets scanner."""
    scanner = MagicMock()
    scanner.scan_repository = AsyncMock(return_value=MockSecretsScanResult())
    scanner.scan_git_history = AsyncMock(return_value=MockSecretsScanResult())
    return scanner


@pytest.fixture
def mock_sast_scanner():
    """Create mock SAST scanner."""
    scanner = MagicMock()
    scanner.initialize = AsyncMock()
    scanner.scan_repository = AsyncMock(return_value=MockSASTScanResult())
    scanner.get_owasp_summary = AsyncMock(return_value={"A01": 0})
    return scanner


@pytest.fixture
def mock_cve_client():
    """Create mock CVE client."""
    client = MagicMock()
    mock_vuln = MagicMock()
    mock_vuln.to_dict.return_value = {
        "cve_id": "CVE-2023-1234",
        "severity": "high",
        "description": "Test vulnerability",
        "references": [],
    }
    client.get_cve = AsyncMock(return_value=mock_vuln)
    client.query_package = AsyncMock(return_value=[mock_vuln])
    return client


@pytest.fixture
def mock_service_registry(mock_scanner, mock_secrets_scanner, mock_sast_scanner, mock_cve_client):
    """Create mock service registry with all scanners."""
    registry = MagicMock()
    registry.has.return_value = True

    def resolve(cls):
        from aragora.analysis.codebase import (
            CVEClient,
            DependencyScanner,
            SASTScanner,
            SecretsScanner,
        )

        if cls == DependencyScanner:
            return mock_scanner
        elif cls == SecretsScanner:
            return mock_secrets_scanner
        elif cls == SASTScanner:
            return mock_sast_scanner
        elif cls == CVEClient:
            return mock_cve_client
        return MagicMock()

    registry.resolve.side_effect = resolve
    return registry


@pytest.fixture(autouse=True)
def clear_scan_storage():
    """Clear scan storage between tests."""
    from aragora.server.handlers.codebase import security

    security._scan_results.clear()
    security._running_scans.clear()
    security._secrets_scan_results.clear()
    security._running_secrets_scans.clear()
    security._sast_scan_results.clear()
    security._running_sast_scans.clear()
    yield


# =============================================================================
# Repository Scanning Tests
# =============================================================================


class TestRepositoryScanning:
    """Test repository scanning endpoints."""

    @pytest.mark.asyncio
    async def test_scan_repository_success(self, mock_service_registry):
        """Test successful repository scan trigger."""
        from aragora.server.handlers.codebase.security import handle_scan_repository

        with patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry.get",
            return_value=mock_service_registry,
        ):
            with patch(
                "aragora.server.handlers.codebase.security.events.emit_scan_events",
                new_callable=AsyncMock,
            ):
                result = await handle_scan_repository(
                    repo_path="/path/to/repo",
                    repo_id="test-repo",
                    branch="main",
                )

                assert result.status_code == 200
                response = json.loads(result.body.decode())
                data = response.get("data", response)
                assert "scan_id" in data
                assert data["status"] == "running"
                assert data["repository"] == "test-repo"

    @pytest.mark.asyncio
    async def test_scan_repository_already_running(self, mock_service_registry):
        """Test scan returns 409 when already running."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_scan_repository

        # Simulate running scan
        running_task = MagicMock()
        running_task.done.return_value = False
        security._running_scans["test-repo"] = running_task

        result = await handle_scan_repository(
            repo_path="/path/to/repo",
            repo_id="test-repo",
        )

        assert result.status_code == 409

    @pytest.mark.asyncio
    async def test_get_scan_status_success(self):
        """Test getting scan status."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_scan_status

        # Add a scan result
        scan = MockScanResult(scan_id="scan_123")
        security._scan_results["test-repo"] = {"scan_123": scan}

        result = await handle_get_scan_status(
            repo_id="test-repo",
            scan_id="scan_123",
        )

        assert result.status_code == 200
        response = json.loads(result.body.decode())
        data = response.get("data", response)
        assert data["scan_result"]["scan_id"] == "scan_123"

    @pytest.mark.asyncio
    async def test_get_scan_status_not_found(self):
        """Test 404 when scan not found."""
        from aragora.server.handlers.codebase.security import handle_get_scan_status

        result = await handle_get_scan_status(
            repo_id="nonexistent",
            scan_id="scan_123",
        )

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_list_scans_success(self):
        """Test listing scans for a repository."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_list_scans

        # Add multiple scans
        scan1 = MockScanResult(scan_id="scan_1")
        scan2 = MockScanResult(scan_id="scan_2")
        security._scan_results["test-repo"] = {"scan_1": scan1, "scan_2": scan2}

        result = await handle_list_scans(repo_id="test-repo")

        assert result.status_code == 200
        response = json.loads(result.body.decode())
        data = response.get("data", response)
        assert len(data["scans"]) == 2
        assert data["total"] == 2


# =============================================================================
# Vulnerability Listing Tests
# =============================================================================


class TestVulnerabilityListing:
    """Test vulnerability listing and filtering."""

    @pytest.mark.asyncio
    async def test_get_vulnerabilities_success(self):
        """Test getting vulnerabilities from latest scan."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_vulnerabilities

        dep1 = MockDependency("lodash", "4.17.0")
        dep1.add_vulnerability("CVE-2021-23337", "high")
        dep2 = MockDependency("express", "4.17.1")
        dep2.add_vulnerability("CVE-2022-24999", "medium")

        scan = MockScanResult(dependencies=[dep1, dep2])
        security._scan_results["test-repo"] = {"scan_123": scan}

        result = await handle_get_vulnerabilities(repo_id="test-repo")

        assert result.status_code == 200
        response = json.loads(result.body.decode())
        data = response.get("data", response)
        assert "vulnerabilities" in data
        assert len(data["vulnerabilities"]) == 2

    @pytest.mark.asyncio
    async def test_get_vulnerabilities_filter_by_severity(self):
        """Test filtering vulnerabilities by severity."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_vulnerabilities

        dep1 = MockDependency("lodash", "4.17.0")
        dep1.add_vulnerability("CVE-2021-23337", "high")
        dep2 = MockDependency("express", "4.17.1")
        dep2.add_vulnerability("CVE-2022-24999", "medium")

        scan = MockScanResult(dependencies=[dep1, dep2])
        security._scan_results["test-repo"] = {"scan_123": scan}

        result = await handle_get_vulnerabilities(
            repo_id="test-repo",
            severity="high",
        )

        assert result.status_code == 200
        response = json.loads(result.body.decode())
        data = response.get("data", response)
        assert len(data["vulnerabilities"]) == 1
        assert data["vulnerabilities"][0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_get_vulnerabilities_pagination(self):
        """Test vulnerability pagination."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_vulnerabilities

        dep = MockDependency("vulnerable-pkg", "1.0.0")
        for i in range(10):
            dep.add_vulnerability(f"CVE-2023-{i:04d}", "high")

        scan = MockScanResult(dependencies=[dep])
        security._scan_results["test-repo"] = {"scan_123": scan}

        result = await handle_get_vulnerabilities(
            repo_id="test-repo",
            limit=5,
            offset=0,
        )

        assert result.status_code == 200
        response = json.loads(result.body.decode())
        data = response.get("data", response)
        assert len(data["vulnerabilities"]) == 5
        assert data["total"] == 10


# =============================================================================
# CVE Query Tests
# =============================================================================


class TestCVEQueries:
    """Test CVE query endpoints."""

    @pytest.mark.asyncio
    async def test_get_cve_details_success(self, mock_cve_client):
        """Test successful CVE lookup."""
        from aragora.server.handlers.codebase.security import handle_get_cve_details

        with patch(
            "aragora.server.handlers.codebase.security.vulnerability.CVEClient",
            return_value=mock_cve_client,
        ):
            result = await handle_get_cve_details("CVE-2023-1234")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "vulnerability" in data
            assert data["vulnerability"]["cve_id"] == "CVE-2023-1234"

    @pytest.mark.asyncio
    async def test_get_cve_details_not_found(self):
        """Test 404 when CVE not found."""
        from aragora.server.handlers.codebase.security.vulnerability import (
            handle_get_cve_details,
        )

        # Create a fresh mock with get_cve returning None
        mock_client = MagicMock()
        mock_client.get_cve = AsyncMock(return_value=None)

        # The handler imports from aragora.server.handlers.codebase.security (security_module)
        # and uses security_module.CVEClient(), so we need to patch it there
        with patch(
            "aragora.server.handlers.codebase.security.CVEClient",
            return_value=mock_client,
        ):
            result = await handle_get_cve_details("CVE-9999-9999")

            assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_query_package_vulnerabilities(self, mock_cve_client):
        """Test querying vulnerabilities for a package."""
        from aragora.server.handlers.codebase.security import handle_query_package_vulnerabilities

        with patch(
            "aragora.server.handlers.codebase.security.vulnerability.CVEClient",
            return_value=mock_cve_client,
        ):
            result = await handle_query_package_vulnerabilities(
                package_name="lodash",
                ecosystem="npm",
                version="4.17.0",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["package"] == "lodash"
            assert data["ecosystem"] == "npm"


# =============================================================================
# Secrets Scanning Tests
# =============================================================================


class TestSecretsScanning:
    """Test secrets scanning endpoints."""

    @pytest.mark.asyncio
    async def test_scan_secrets_success(self, mock_secrets_scanner):
        """Test successful secrets scan."""
        from aragora.server.handlers.codebase.security import handle_scan_secrets

        with patch(
            "aragora.server.handlers.codebase.security.secrets.SecretsScanner",
            return_value=mock_secrets_scanner,
        ):
            with patch(
                "aragora.server.handlers.codebase.security.events.emit_secrets_events",
                new_callable=AsyncMock,
            ):
                result = await handle_scan_secrets(
                    repo_path="/path/to/repo",
                    repo_id="test-repo",
                )

                assert result.status_code == 200
                response = json.loads(result.body.decode())
                data = response.get("data", response)
                assert "scan_id" in data
                assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_secrets_scan_status(self):
        """Test getting secrets scan status."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_secrets_scan_status

        scan = MockSecretsScanResult(scan_id="secrets_123")
        security._secrets_scan_results["test-repo"] = {"secrets_123": scan}

        result = await handle_get_secrets_scan_status(
            repo_id="test-repo",
            scan_id="secrets_123",
        )

        assert result.status_code == 200


# =============================================================================
# SAST Scanning Tests
# =============================================================================


class TestSASTScanning:
    """Test SAST scanning endpoints."""

    @pytest.mark.asyncio
    async def test_scan_sast_success(self, mock_service_registry):
        """Test successful SAST scan."""
        from aragora.server.handlers.codebase.security import handle_scan_sast

        with patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry.get",
            return_value=mock_service_registry,
        ):
            result = await handle_scan_sast(
                repo_path="/path/to/repo",
                repo_id="test-repo",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "scan_id" in data

    @pytest.mark.asyncio
    async def test_get_sast_scan_status(self):
        """Test getting SAST scan status."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_sast_scan_status

        scan = MockSASTScanResult(scan_id="sast_123")
        security._sast_scan_results["test-repo"] = {"sast_123": scan}

        result = await handle_get_sast_scan_status(
            repo_id="test-repo",
            scan_id="sast_123",
        )

        assert result.status_code == 200


# =============================================================================
# SBOM Generation Tests
# =============================================================================


class TestSBOMGeneration:
    """Test SBOM generation endpoints."""

    @pytest.mark.asyncio
    async def test_generate_sbom_handler_exists(self):
        """Test SBOM generation handler is importable."""
        from aragora.server.handlers.codebase.security import handle_generate_sbom

        assert callable(handle_generate_sbom)

    @pytest.mark.asyncio
    async def test_get_sbom_handler_exists(self):
        """Test get SBOM handler is importable."""
        from aragora.server.handlers.codebase.security import handle_get_sbom

        assert callable(handle_get_sbom)

    @pytest.mark.asyncio
    async def test_list_sboms_handler_exists(self):
        """Test list SBOMs handler is importable."""
        from aragora.server.handlers.codebase.security import handle_list_sboms

        assert callable(handle_list_sboms)


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Test input validation in handlers."""

    @pytest.fixture
    def handler(self):
        """Create SecurityHandler instance."""
        from aragora.server.handlers.codebase.security.handler import SecurityHandler

        return SecurityHandler(ctx={})

    @pytest.mark.asyncio
    async def test_post_scan_requires_repo_path(self, handler):
        """Test POST scan requires repo_path."""
        result = await handler.handle_post_scan(
            data={},  # Missing repo_path
            repo_id="valid-repo",
        )
        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "repo_path required" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_post_secrets_scan_requires_repo_path(self, handler):
        """Test POST secrets scan requires repo_path."""
        result = await handler.handle_post_secrets_scan(
            data={},  # Missing repo_path
            repo_id="valid-repo",
        )
        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "repo_path required" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_post_sbom_requires_repo_path(self, handler):
        """Test POST SBOM requires repo_path."""
        result = await handler.handle_post_sbom(
            data={},  # Missing repo_path
            repo_id="valid-repo",
        )
        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "repo_path required" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_compare_sbom_requires_both_ids(self, handler):
        """Test SBOM compare requires both sbom_id_a and sbom_id_b."""
        result = await handler.handle_compare_sbom(
            data={"sbom_id_a": "sbom_1"},  # Missing sbom_id_b
            repo_id="valid-repo",
        )
        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "sbom_id_a and sbom_id_b required" in body.get("error", "").lower()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in security endpoints."""

    @pytest.mark.asyncio
    async def test_get_vulnerabilities_handles_invalid_data(self):
        """Test handling of corrupted scan data."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_vulnerabilities

        # Add corrupted scan data
        security._scan_results["test-repo"] = {"bad_scan": "not a scan object"}

        result = await handle_get_vulnerabilities(repo_id="test-repo")

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_get_scan_status_handles_missing_repo(self):
        """Test graceful handling of missing repository."""
        from aragora.server.handlers.codebase.security import handle_get_scan_status

        result = await handle_get_scan_status(repo_id="nonexistent-repo")

        assert result.status_code == 404


# =============================================================================
# SecurityHandler Class Tests
# =============================================================================


class TestSecurityHandler:
    """Test SecurityHandler class methods."""

    @pytest.fixture
    def handler(self):
        """Create SecurityHandler instance."""
        from aragora.server.handlers.codebase.security.handler import SecurityHandler

        return SecurityHandler(ctx={})

    def test_can_handle_scan_endpoint(self, handler):
        """Test can_handle returns True for scan endpoints."""
        assert handler.can_handle("/api/v1/codebase/my-repo/scan") is True
        assert handler.can_handle("/api/v1/codebase/my-repo/scan/latest") is True

    def test_can_handle_vulnerabilities_endpoint(self, handler):
        """Test can_handle returns True for vulnerabilities endpoint."""
        assert handler.can_handle("/api/v1/codebase/my-repo/vulnerabilities") is True

    def test_can_handle_cve_endpoint(self, handler):
        """Test can_handle returns True for CVE endpoint."""
        assert handler.can_handle("/api/v1/cve/CVE-2023-1234") is True

    def test_can_handle_secrets_endpoint(self, handler):
        """Test can_handle returns True for secrets endpoint."""
        assert handler.can_handle("/api/v1/codebase/my-repo/secrets") is True
        assert handler.can_handle("/api/v1/codebase/my-repo/scan/secrets") is True

    def test_can_handle_sbom_endpoint(self, handler):
        """Test can_handle returns True for SBOM endpoint."""
        assert handler.can_handle("/api/v1/codebase/my-repo/sbom") is True
        assert handler.can_handle("/api/v1/codebase/my-repo/sbom/latest") is True

    def test_can_handle_unrelated_endpoint(self, handler):
        """Test can_handle returns False for unrelated endpoints."""
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/users") is False
        assert handler.can_handle("/api/v1/codebase/my-repo/analysis") is False

    def test_get_user_id_with_auth_context(self, handler):
        """Test _get_user_id with auth context."""
        auth_ctx = MagicMock()
        auth_ctx.user_id = "user_123"
        handler.ctx["auth_context"] = auth_ctx

        assert handler._get_user_id() == "user_123"

    def test_get_user_id_without_auth_context(self, handler):
        """Test _get_user_id without auth context returns default."""
        assert handler._get_user_id() == "default"


# =============================================================================
# Storage Tests
# =============================================================================


class TestStorage:
    """Test storage helper functions."""

    def test_get_or_create_repo_scans_creates_new(self):
        """Test creating new storage for repo."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security.storage import get_or_create_repo_scans

        result = get_or_create_repo_scans("new-repo")

        assert "new-repo" in security._scan_results
        assert result == security._scan_results["new-repo"]

    def test_get_or_create_repo_scans_returns_existing(self):
        """Test returning existing storage."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security.storage import get_or_create_repo_scans

        # Pre-populate
        security._scan_results["existing-repo"] = {"scan_1": "data"}

        result = get_or_create_repo_scans("existing-repo")

        assert result == {"scan_1": "data"}

    def test_get_or_create_secrets_scans(self):
        """Test secrets scan storage helper."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security.storage import get_or_create_secrets_scans

        result = get_or_create_secrets_scans("test-repo")

        assert "test-repo" in security._secrets_scan_results
        assert result == security._secrets_scan_results["test-repo"]


__all__ = [
    "TestSafeRepoId",
    "TestSecurityHandlerPathTraversal",
    "TestRepositoryScanning",
    "TestVulnerabilityListing",
    "TestCVEQueries",
    "TestSecretsScanning",
    "TestSASTScanning",
    "TestSBOMGeneration",
    "TestInputValidation",
    "TestErrorHandling",
    "TestSecurityHandler",
    "TestStorage",
]
