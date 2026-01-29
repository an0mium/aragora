"""
Tests for aragora.server.handlers.codebase.security - Security Scan Handler.

Tests cover:
- Scan trigger and status endpoints
- Vulnerability listing and filtering
- CVE lookup
- Secrets scanning
- SAST scanning
- SBOM generation
- Permission checks
- Error handling
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Classes
# =============================================================================


class MockScanResult:
    """Mock scan result."""

    def __init__(
        self,
        scan_id: str = "scan_123",
        repository: str = "test-repo",
        status: str = "completed",
        dependencies: list = None,
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
        self.summary = MagicMock()
        self.summary.vulnerable_dependencies = 2

    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "repository": self.repository,
            "status": self.status,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "started_at": self.started_at.isoformat(),
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
        vuln.cve_id = cve_id
        vuln.severity = severity
        vuln.to_dict = lambda: {"cve_id": cve_id, "severity": severity}
        self.vulnerabilities.append(vuln)
        return self


class MockSecretsScanResult:
    """Mock secrets scan result."""

    def __init__(self, scan_id: str = "secrets_123", findings: list = None):
        self.scan_id = scan_id
        self.status = "completed"
        self.started_at = datetime.now(timezone.utc)
        self.completed_at = datetime.now(timezone.utc)
        self.findings = findings or []

    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "status": self.status,
            "findings": self.findings,
        }


class MockSASTScanResult:
    """Mock SAST scan result."""

    def __init__(self, scan_id: str = "sast_123", findings: list = None):
        self.scan_id = scan_id
        self.status = "completed"
        self.started_at = datetime.now(timezone.utc)
        self.completed_at = datetime.now(timezone.utc)
        self.findings = findings or []

    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "status": self.status,
            "findings": self.findings,
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
    return scanner


@pytest.fixture
def mock_sast_scanner():
    """Create mock SAST scanner."""
    scanner = MagicMock()
    scanner.scan_repository = AsyncMock(return_value=MockSASTScanResult())
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
    return client


@pytest.fixture
def mock_service_registry(mock_scanner, mock_secrets_scanner, mock_sast_scanner, mock_cve_client):
    """Create mock service registry with all scanners."""
    registry = MagicMock()
    registry.has.return_value = True

    def resolve(cls):
        from aragora.analysis.codebase import (
            DependencyScanner,
            SecretsScanner,
            SASTScanner,
            CVEClient,
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
# Scan Trigger Tests
# =============================================================================


class TestScanTrigger:
    """Test scan trigger endpoint."""

    @pytest.mark.asyncio
    async def test_scan_repository_success(self, mock_service_registry):
        """Test successful scan trigger returns scan ID."""
        from aragora.server.handlers.codebase.security import handle_scan_repository

        with patch(
            "aragora.server.handlers.codebase.security.ServiceRegistry.get",
            return_value=mock_service_registry,
        ):
            with patch(
                "aragora.server.handlers.codebase.security._emit_scan_events",
                new_callable=AsyncMock,
            ):
                result = await handle_scan_repository(
                    repo_path="/path/to/repo",
                    repo_id="test-repo",
                    branch="main",
                )

                assert result.status_code == 200
                import json

                response = json.loads(result.body.decode())
                # Handle nested data structure
                data = response.get("data", response)
                assert "scan_id" in data
                assert data["status"] == "running"
                assert data["repository"] == "test-repo"

    @pytest.mark.asyncio
    async def test_scan_repository_already_running(self, mock_service_registry):
        """Test scan returns 409 if already running."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_scan_repository

        # Simulate running scan
        never_done_task = MagicMock()
        never_done_task.done.return_value = False
        security._running_scans["test-repo"] = never_done_task

        result = await handle_scan_repository(
            repo_path="/path/to/repo",
            repo_id="test-repo",
        )

        assert result.status_code == 409

    @pytest.mark.asyncio
    async def test_scan_repository_generates_repo_id(self, mock_service_registry):
        """Test scan generates repo_id if not provided."""
        from aragora.server.handlers.codebase.security import handle_scan_repository

        with patch(
            "aragora.server.handlers.codebase.security.ServiceRegistry.get",
            return_value=mock_service_registry,
        ):
            with patch(
                "aragora.server.handlers.codebase.security._emit_scan_events",
                new_callable=AsyncMock,
            ):
                result = await handle_scan_repository(
                    repo_path="/path/to/repo",
                )

                assert result.status_code == 200
                import json

                response = json.loads(result.body.decode())
                data = response.get("data", response)
                assert data["repository"].startswith("repo_")


# =============================================================================
# Scan Status Tests
# =============================================================================


class TestScanStatus:
    """Test scan status endpoint."""

    @pytest.mark.asyncio
    async def test_get_scan_status_specific(self):
        """Test getting specific scan by ID."""
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
        import json

        response = json.loads(result.body.decode())
        data = response.get("data", response)
        assert "scan_result" in data
        assert data["scan_result"]["scan_id"] == "scan_123"

    @pytest.mark.asyncio
    async def test_get_scan_status_latest(self):
        """Test getting latest scan."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_scan_status

        # Add multiple scans
        scan1 = MockScanResult(scan_id="scan_old")
        scan1.started_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
        scan2 = MockScanResult(scan_id="scan_new")
        scan2.started_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        security._scan_results["test-repo"] = {"scan_old": scan1, "scan_new": scan2}

        result = await handle_get_scan_status(repo_id="test-repo")

        assert result.status_code == 200
        import json

        response = json.loads(result.body.decode())
        data = response.get("data", response)
        assert data["scan_result"]["scan_id"] == "scan_new"

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
    async def test_get_scan_status_no_scans(self):
        """Test 404 when no scans exist for repo."""
        from aragora.server.handlers.codebase.security import handle_get_scan_status

        result = await handle_get_scan_status(repo_id="empty-repo")

        assert result.status_code == 404


# =============================================================================
# Vulnerability List Tests
# =============================================================================


class TestVulnerabilityList:
    """Test vulnerability listing endpoint."""

    @pytest.mark.asyncio
    async def test_get_vulnerabilities_success(self):
        """Test getting vulnerabilities from latest scan."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_vulnerabilities

        # Create scan with vulnerabilities
        dep1 = MockDependency("lodash", "4.17.0")
        dep1.add_vulnerability("CVE-2021-23337", "high")
        dep2 = MockDependency("express", "4.17.1")
        dep2.add_vulnerability("CVE-2022-24999", "medium")

        scan = MockScanResult(dependencies=[dep1, dep2])
        security._scan_results["test-repo"] = {"scan_123": scan}

        result = await handle_get_vulnerabilities(repo_id="test-repo")

        assert result.status_code == 200
        import json

        response = json.loads(result.body.decode())
        data = response.get("data", response)
        assert "vulnerabilities" in data
        assert len(data["vulnerabilities"]) == 2
        assert data["total"] == 2

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
        import json

        response = json.loads(result.body.decode())
        data = response.get("data", response)
        assert len(data["vulnerabilities"]) == 1
        assert data["vulnerabilities"][0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_get_vulnerabilities_filter_by_package(self):
        """Test filtering vulnerabilities by package name."""
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
            package="lodash",
        )

        assert result.status_code == 200
        import json

        response = json.loads(result.body.decode())
        data = response.get("data", response)
        assert len(data["vulnerabilities"]) == 1
        assert data["vulnerabilities"][0]["package_name"] == "lodash"

    @pytest.mark.asyncio
    async def test_get_vulnerabilities_pagination(self):
        """Test vulnerability pagination."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_vulnerabilities

        # Create many vulnerabilities
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
        import json

        response = json.loads(result.body.decode())
        data = response.get("data", response)
        assert len(data["vulnerabilities"]) == 5
        assert data["total"] == 10
        assert data["limit"] == 5
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_get_vulnerabilities_no_completed_scans(self):
        """Test 404 when no completed scans."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_vulnerabilities

        scan = MockScanResult(status="running")
        security._scan_results["test-repo"] = {"scan_123": scan}

        result = await handle_get_vulnerabilities(repo_id="test-repo")

        assert result.status_code == 404


# =============================================================================
# CVE Lookup Tests
# =============================================================================


class TestCVELookup:
    """Test CVE lookup endpoint."""

    @pytest.mark.asyncio
    async def test_get_cve_details_success(self, mock_cve_client):
        """Test successful CVE lookup."""
        from aragora.server.handlers.codebase.security import handle_get_cve_details

        with patch(
            "aragora.server.handlers.codebase.security.CVEClient",
            return_value=mock_cve_client,
        ):
            result = await handle_get_cve_details("CVE-2023-1234")

            assert result.status_code == 200
            import json

            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "vulnerability" in data
            assert data["vulnerability"]["cve_id"] == "CVE-2023-1234"

    @pytest.mark.asyncio
    async def test_get_cve_details_not_found(self, mock_cve_client):
        """Test 404 when CVE not found."""
        from aragora.server.handlers.codebase.security import handle_get_cve_details

        mock_cve_client.get_cve = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.codebase.security.CVEClient",
            return_value=mock_cve_client,
        ):
            result = await handle_get_cve_details("CVE-9999-9999")

            assert result.status_code == 404


# =============================================================================
# Secrets Scanning Tests
# =============================================================================


class TestSecretsScanning:
    """Test secrets scanning endpoints."""

    @pytest.mark.asyncio
    async def test_scan_secrets_success(self, mock_service_registry):
        """Test successful secrets scan."""
        from aragora.server.handlers.codebase.security import handle_scan_secrets

        with patch(
            "aragora.server.handlers.codebase.security.ServiceRegistry.get",
            return_value=mock_service_registry,
        ):
            result = await handle_scan_secrets(
                repo_path="/path/to/repo",
                repo_id="test-repo",
            )

            assert result.status_code == 200
            import json

            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "scan_id" in data
            assert data["status"] == "running"


class TestSASTScanning:
    """Test SAST scanning endpoints."""

    @pytest.mark.asyncio
    async def test_scan_sast_success(self, mock_service_registry):
        """Test successful SAST scan."""
        from aragora.server.handlers.codebase.security import handle_scan_sast

        with patch(
            "aragora.server.handlers.codebase.security.ServiceRegistry.get",
            return_value=mock_service_registry,
        ):
            result = await handle_scan_sast(
                repo_path="/path/to/repo",
                repo_id="test-repo",
            )

            assert result.status_code == 200
            import json

            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "scan_id" in data


# =============================================================================
# SBOM Generation Tests
# =============================================================================


class TestSBOMGeneration:
    """Test SBOM generation endpoints."""

    @pytest.mark.asyncio
    async def test_generate_sbom_endpoint_exists(self):
        """Test SBOM generation endpoint is importable."""
        from aragora.server.handlers.codebase.security import handle_generate_sbom

        assert callable(handle_generate_sbom)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in security endpoints."""

    @pytest.mark.asyncio
    async def test_get_vulnerabilities_internal_error(self):
        """Test 500 on vulnerability lookup error."""
        from aragora.server.handlers.codebase import security
        from aragora.server.handlers.codebase.security import handle_get_vulnerabilities

        # Add a corrupted scan that will cause an error
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
# Service Registry Integration Tests
# =============================================================================


class TestServiceRegistry:
    """Test service registry integration."""

    def test_get_scanner_creates_new(self):
        """Test scanner is created if not in registry."""
        from aragora.server.handlers.codebase.security import _get_scanner

        mock_registry = MagicMock()
        mock_registry.has.return_value = False

        with patch(
            "aragora.server.handlers.codebase.security.ServiceRegistry.get",
            return_value=mock_registry,
        ):
            _get_scanner()

            mock_registry.register.assert_called_once()

    def test_get_scanner_returns_existing(self):
        """Test existing scanner is returned from registry."""
        from aragora.server.handlers.codebase.security import _get_scanner

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        expected_scanner = MagicMock()
        mock_registry.resolve.return_value = expected_scanner

        with patch(
            "aragora.server.handlers.codebase.security.ServiceRegistry.get",
            return_value=mock_registry,
        ):
            result = _get_scanner()

            assert result == expected_scanner
            mock_registry.register.assert_not_called()


# =============================================================================
# Storage Tests
# =============================================================================


class TestScanStorage:
    """Test scan result storage."""

    def test_get_or_create_repo_scans_creates_new(self):
        """Test new storage is created for new repo."""
        from aragora.server.handlers.codebase.security import _get_or_create_repo_scans
        from aragora.server.handlers.codebase import security

        result = _get_or_create_repo_scans("new-repo")

        assert "new-repo" in security._scan_results
        assert result == security._scan_results["new-repo"]

    def test_get_or_create_repo_scans_returns_existing(self):
        """Test existing storage is returned."""
        from aragora.server.handlers.codebase.security import _get_or_create_repo_scans
        from aragora.server.handlers.codebase import security

        # Pre-populate
        security._scan_results["existing-repo"] = {"scan_1": "data"}

        result = _get_or_create_repo_scans("existing-repo")

        assert result == {"scan_1": "data"}


__all__ = [
    "TestScanTrigger",
    "TestScanStatus",
    "TestVulnerabilityList",
    "TestCVELookup",
    "TestSecretsScanning",
    "TestSASTScanning",
    "TestSBOMGeneration",
    "TestErrorHandling",
    "TestServiceRegistry",
    "TestScanStorage",
]
