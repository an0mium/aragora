"""Tests for Dependency Analysis namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestDependencyAnalysis:
    """Tests for dependency analysis methods."""

    def test_analyze_with_repository_url(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_dependencies": 50,
                "direct_dependencies": 10,
                "transitive_dependencies": 40,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.dependency_analysis.analyze(
                repository_url="https://github.com/org/repo",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/analyze-dependencies",
                json={
                    "include_dev_dependencies": False,
                    "include_transitive": True,
                    "repository_url": "https://github.com/org/repo",
                },
            )
            assert result["total_dependencies"] == 50
            assert result["direct_dependencies"] == 10
            client.close()

    def test_analyze_with_all_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_dependencies": 25}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.dependency_analysis.analyze(
                repository_url="https://github.com/org/repo",
                local_path="/path/to/project",
                package_managers=["npm", "pip"],
                include_dev=True,
                include_transitive=False,
                max_depth=3,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/analyze-dependencies",
                json={
                    "include_dev_dependencies": True,
                    "include_transitive": False,
                    "repository_url": "https://github.com/org/repo",
                    "local_path": "/path/to/project",
                    "package_managers": ["npm", "pip"],
                    "max_depth": 3,
                },
            )
            client.close()


class TestSBOMGeneration:
    """Tests for SBOM generation methods."""

    def test_generate_sbom_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "format": "cyclonedx",
                "components": [{"name": "lodash", "version": "4.17.21"}],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.dependency_analysis.generate_sbom(
                repository_url="https://github.com/org/repo",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/sbom",
                json={
                    "format": "cyclonedx",
                    "include_checksums": False,
                    "include_licenses": True,
                    "repository_url": "https://github.com/org/repo",
                },
            )
            assert result["format"] == "cyclonedx"
            assert len(result["components"]) == 1
            client.close()

    def test_generate_sbom_spdx_with_checksums(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"format": "spdx"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.dependency_analysis.generate_sbom(
                repository_url="https://github.com/org/repo",
                format="spdx",
                include_checksums=True,
                include_licenses=False,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/sbom",
                json={
                    "format": "spdx",
                    "include_checksums": True,
                    "include_licenses": False,
                    "repository_url": "https://github.com/org/repo",
                },
            )
            client.close()


class TestVulnerabilityScanning:
    """Tests for vulnerability scanning methods."""

    def test_scan_vulnerabilities_basic(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_vulnerabilities": 5,
                "critical": 1,
                "high": 2,
                "medium": 2,
                "low": 0,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.dependency_analysis.scan_vulnerabilities(
                repository_url="https://github.com/org/repo",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/scan-vulnerabilities",
                json={
                    "severity_threshold": "low",
                    "repository_url": "https://github.com/org/repo",
                },
            )
            assert result["total_vulnerabilities"] == 5
            assert result["critical"] == 1
            client.close()

    def test_scan_vulnerabilities_with_ignore_cves(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_vulnerabilities": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.dependency_analysis.scan_vulnerabilities(
                repository_url="https://github.com/org/repo",
                severity_threshold="high",
                ignore_cves=["CVE-2023-1234", "CVE-2023-5678"],
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/scan-vulnerabilities",
                json={
                    "severity_threshold": "high",
                    "repository_url": "https://github.com/org/repo",
                    "ignore_cves": ["CVE-2023-1234", "CVE-2023-5678"],
                },
            )
            client.close()


class TestLicenseCompliance:
    """Tests for license compliance methods."""

    def test_check_licenses_default_policy(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "compliant": True,
                "total_packages": 25,
                "compatible": 24,
                "incompatible": 1,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.dependency_analysis.check_licenses(
                repository_url="https://github.com/org/repo",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/check-licenses",
                json={
                    "policy": "permissive",
                    "repository_url": "https://github.com/org/repo",
                },
            )
            assert result["compliant"] is True
            assert result["total_packages"] == 25
            client.close()

    def test_check_licenses_commercial_with_allow_deny(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"compliant": False, "incompatible": 3}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.dependency_analysis.check_licenses(
                repository_url="https://github.com/org/repo",
                allowed_licenses=["MIT", "Apache-2.0"],
                denied_licenses=["GPL-3.0"],
                policy="commercial",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/check-licenses",
                json={
                    "policy": "commercial",
                    "repository_url": "https://github.com/org/repo",
                    "allowed_licenses": ["MIT", "Apache-2.0"],
                    "denied_licenses": ["GPL-3.0"],
                },
            )
            client.close()


class TestCacheAndConvenience:
    """Tests for cache management and convenience methods."""

    def test_clear_cache_all(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"cleared": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.dependency_analysis.clear_cache()
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/clear-cache",
                json={},
            )
            assert result["cleared"] is True
            client.close()

    def test_clear_cache_for_repository(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"cleared": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.dependency_analysis.clear_cache(
                repository_url="https://github.com/org/repo",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/clear-cache",
                json={"repository_url": "https://github.com/org/repo"},
            )
            client.close()

    def test_full_audit_high_risk(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.side_effect = [
                {"total_dependencies": 50, "direct_dependencies": 10},
                {"format": "cyclonedx", "components": []},
                {"total_vulnerabilities": 2, "critical": 0, "high": 1, "medium": 1},
                {"compliant": True},
            ]
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.dependency_analysis.full_audit(
                {"repository_url": "https://github.com/org/repo"}
            )
            assert mock_request.call_count == 4
            assert result["summary"]["total_dependencies"] == 50
            assert result["summary"]["total_vulnerabilities"] == 2
            assert result["summary"]["license_compliant"] is True
            assert result["summary"]["risk_level"] == "high"
            client.close()

    def test_full_audit_critical_risk(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.side_effect = [
                {"total_dependencies": 10},
                {"format": "cyclonedx"},
                {"total_vulnerabilities": 1, "critical": 1, "high": 0, "medium": 0},
                {"compliant": True},
            ]
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.dependency_analysis.full_audit(
                {"repository_url": "https://github.com/org/repo"}
            )
            assert result["summary"]["risk_level"] == "critical"
            client.close()


class TestAsyncDependencyAnalysis:
    """Tests for async dependency analysis methods."""

    @pytest.mark.asyncio
    async def test_analyze(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"total_dependencies": 50}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.dependency_analysis.analyze(
                repository_url="https://github.com/org/repo",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/analyze-dependencies",
                json={
                    "include_dev_dependencies": False,
                    "include_transitive": True,
                    "repository_url": "https://github.com/org/repo",
                },
            )
            assert result["total_dependencies"] == 50
            await client.close()

    @pytest.mark.asyncio
    async def test_generate_sbom(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"format": "spdx", "components": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.dependency_analysis.generate_sbom(
                repository_url="https://github.com/org/repo",
                format="spdx",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/sbom",
                json={
                    "format": "spdx",
                    "include_checksums": False,
                    "include_licenses": True,
                    "repository_url": "https://github.com/org/repo",
                },
            )
            assert result["format"] == "spdx"
            await client.close()

    @pytest.mark.asyncio
    async def test_scan_vulnerabilities(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "total_vulnerabilities": 3,
                "critical": 0,
                "high": 1,
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.dependency_analysis.scan_vulnerabilities(
                repository_url="https://github.com/org/repo",
                severity_threshold="medium",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/scan-vulnerabilities",
                json={
                    "severity_threshold": "medium",
                    "repository_url": "https://github.com/org/repo",
                },
            )
            assert result["total_vulnerabilities"] == 3
            await client.close()

    @pytest.mark.asyncio
    async def test_full_audit(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.side_effect = [
                {"total_dependencies": 30},
                {"format": "cyclonedx"},
                {"total_vulnerabilities": 0, "critical": 0, "high": 0, "medium": 0},
                {"compliant": True},
            ]
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.dependency_analysis.full_audit(
                {"repository_url": "https://github.com/org/repo"}
            )
            assert mock_request.call_count == 4
            assert result["summary"]["risk_level"] == "low"
            assert result["summary"]["license_compliant"] is True
            await client.close()
