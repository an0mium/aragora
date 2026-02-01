"""Tests for Dependency Analysis SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock client."""
    return MagicMock()


class TestDependencyAnalysisAPI:
    """Test synchronous DependencyAnalysisAPI."""

    def test_init(self, mock_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        api = DependencyAnalysisAPI(mock_client)
        assert api._client is mock_client

    def test_analyze(self, mock_client: MagicMock) -> None:
        """Test analyze calls correct endpoint."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        mock_client.request.return_value = {
            "total_dependencies": 50,
            "direct_dependencies": 10,
            "transitive_dependencies": 40,
        }

        api = DependencyAnalysisAPI(mock_client)
        result = api.analyze(repository_url="https://github.com/org/repo")

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/codebase/analyze-dependencies")
        assert call_args[1]["json"]["repository_url"] == "https://github.com/org/repo"
        assert call_args[1]["json"]["include_transitive"] is True
        assert result["total_dependencies"] == 50

    def test_analyze_with_options(self, mock_client: MagicMock) -> None:
        """Test analyze with all options."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        mock_client.request.return_value = {"total_dependencies": 25}

        api = DependencyAnalysisAPI(mock_client)
        api.analyze(
            repository_url="https://github.com/org/repo",
            local_path="/path/to/project",
            package_managers=["npm", "pip"],
            include_dev=True,
            include_transitive=False,
            max_depth=3,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["repository_url"] == "https://github.com/org/repo"
        assert json_body["local_path"] == "/path/to/project"
        assert json_body["package_managers"] == ["npm", "pip"]
        assert json_body["include_dev_dependencies"] is True
        assert json_body["include_transitive"] is False
        assert json_body["max_depth"] == 3

    def test_generate_sbom(self, mock_client: MagicMock) -> None:
        """Test generate_sbom calls correct endpoint."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        mock_client.request.return_value = {
            "format": "cyclonedx",
            "components": [],
        }

        api = DependencyAnalysisAPI(mock_client)
        result = api.generate_sbom(repository_url="https://github.com/org/repo")

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/codebase/sbom")
        assert call_args[1]["json"]["format"] == "cyclonedx"
        assert result["format"] == "cyclonedx"

    def test_generate_sbom_with_options(self, mock_client: MagicMock) -> None:
        """Test generate_sbom with all options."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        mock_client.request.return_value = {"format": "spdx"}

        api = DependencyAnalysisAPI(mock_client)
        api.generate_sbom(
            repository_url="https://github.com/org/repo",
            local_path="/path/to/project",
            format="spdx",
            include_checksums=True,
            include_licenses=False,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["format"] == "spdx"
        assert json_body["include_checksums"] is True
        assert json_body["include_licenses"] is False

    def test_scan_vulnerabilities(self, mock_client: MagicMock) -> None:
        """Test scan_vulnerabilities calls correct endpoint."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        mock_client.request.return_value = {
            "total_vulnerabilities": 5,
            "critical": 1,
            "high": 2,
            "medium": 2,
            "low": 0,
        }

        api = DependencyAnalysisAPI(mock_client)
        result = api.scan_vulnerabilities(repository_url="https://github.com/org/repo")

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/codebase/scan-vulnerabilities")
        assert result["total_vulnerabilities"] == 5

    def test_scan_vulnerabilities_with_options(self, mock_client: MagicMock) -> None:
        """Test scan_vulnerabilities with all options."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        mock_client.request.return_value = {"total_vulnerabilities": 0}

        api = DependencyAnalysisAPI(mock_client)
        api.scan_vulnerabilities(
            repository_url="https://github.com/org/repo",
            local_path="/path/to/project",
            sbom={"format": "cyclonedx", "components": []},
            severity_threshold="high",
            ignore_cves=["CVE-2023-1234", "CVE-2023-5678"],
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["severity_threshold"] == "high"
        assert json_body["ignore_cves"] == ["CVE-2023-1234", "CVE-2023-5678"]
        assert json_body["sbom"]["format"] == "cyclonedx"

    def test_check_licenses(self, mock_client: MagicMock) -> None:
        """Test check_licenses calls correct endpoint."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        mock_client.request.return_value = {
            "compliant": True,
            "total_packages": 25,
            "compatible": 24,
            "incompatible": 1,
        }

        api = DependencyAnalysisAPI(mock_client)
        result = api.check_licenses(repository_url="https://github.com/org/repo")

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/codebase/check-licenses")
        assert result["compliant"] is True

    def test_check_licenses_with_options(self, mock_client: MagicMock) -> None:
        """Test check_licenses with all options."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        mock_client.request.return_value = {"compliant": False}

        api = DependencyAnalysisAPI(mock_client)
        api.check_licenses(
            repository_url="https://github.com/org/repo",
            local_path="/path/to/project",
            allowed_licenses=["MIT", "Apache-2.0"],
            denied_licenses=["GPL-3.0"],
            policy="commercial",
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["allowed_licenses"] == ["MIT", "Apache-2.0"]
        assert json_body["denied_licenses"] == ["GPL-3.0"]
        assert json_body["policy"] == "commercial"

    def test_clear_cache(self, mock_client: MagicMock) -> None:
        """Test clear_cache calls correct endpoint."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        mock_client.request.return_value = {"cleared": True}

        api = DependencyAnalysisAPI(mock_client)
        result = api.clear_cache()

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/codebase/clear-cache")
        assert result["cleared"] is True

    def test_clear_cache_with_repository(self, mock_client: MagicMock) -> None:
        """Test clear_cache with repository URL."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        mock_client.request.return_value = {"cleared": True}

        api = DependencyAnalysisAPI(mock_client)
        api.clear_cache(repository_url="https://github.com/org/repo")

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["repository_url"] == "https://github.com/org/repo"

    def test_full_audit(self, mock_client: MagicMock) -> None:
        """Test full_audit convenience method."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        # Mock responses for each call
        mock_client.request.side_effect = [
            {"total_dependencies": 50, "direct_dependencies": 10},  # analyze
            {"format": "cyclonedx", "components": []},  # generate_sbom
            {"total_vulnerabilities": 2, "critical": 0, "high": 1, "medium": 1},  # scan
            {"compliant": True},  # check_licenses
        ]

        api = DependencyAnalysisAPI(mock_client)
        result = api.full_audit({"repository_url": "https://github.com/org/repo"})

        assert mock_client.request.call_count == 4
        assert result["summary"]["total_dependencies"] == 50
        assert result["summary"]["total_vulnerabilities"] == 2
        assert result["summary"]["license_compliant"] is True
        assert result["summary"]["risk_level"] == "high"  # Due to 1 high vulnerability

    def test_full_audit_critical_risk(self, mock_client: MagicMock) -> None:
        """Test full_audit with critical vulnerabilities."""
        from aragora.namespaces.dependency_analysis import DependencyAnalysisAPI

        mock_client.request.side_effect = [
            {"total_dependencies": 10},
            {"format": "cyclonedx"},
            {"total_vulnerabilities": 1, "critical": 1, "high": 0, "medium": 0},
            {"compliant": True},
        ]

        api = DependencyAnalysisAPI(mock_client)
        result = api.full_audit({"repository_url": "https://github.com/org/repo"})

        assert result["summary"]["risk_level"] == "critical"


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock async client."""
    from unittest.mock import AsyncMock

    client = MagicMock()
    client.request = AsyncMock()
    return client


class TestAsyncDependencyAnalysisAPI:
    """Test asynchronous AsyncDependencyAnalysisAPI."""

    @pytest.mark.asyncio
    async def test_analyze(self, mock_async_client: MagicMock) -> None:
        """Test analyze calls correct endpoint."""
        from aragora.namespaces.dependency_analysis import AsyncDependencyAnalysisAPI

        mock_async_client.request.return_value = {"total_dependencies": 50}

        api = AsyncDependencyAnalysisAPI(mock_async_client)
        result = await api.analyze(repository_url="https://github.com/org/repo")

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/codebase/analyze-dependencies")
        assert result["total_dependencies"] == 50

    @pytest.mark.asyncio
    async def test_analyze_with_options(self, mock_async_client: MagicMock) -> None:
        """Test analyze with all options."""
        from aragora.namespaces.dependency_analysis import AsyncDependencyAnalysisAPI

        mock_async_client.request.return_value = {"total_dependencies": 25}

        api = AsyncDependencyAnalysisAPI(mock_async_client)
        await api.analyze(
            repository_url="https://github.com/org/repo",
            package_managers=["cargo", "go"],
            include_dev=True,
            max_depth=5,
        )

        call_args = mock_async_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["package_managers"] == ["cargo", "go"]
        assert json_body["include_dev_dependencies"] is True
        assert json_body["max_depth"] == 5

    @pytest.mark.asyncio
    async def test_generate_sbom(self, mock_async_client: MagicMock) -> None:
        """Test generate_sbom calls correct endpoint."""
        from aragora.namespaces.dependency_analysis import AsyncDependencyAnalysisAPI

        mock_async_client.request.return_value = {"format": "cyclonedx"}

        api = AsyncDependencyAnalysisAPI(mock_async_client)
        await api.generate_sbom(
            repository_url="https://github.com/org/repo",
            format="spdx",
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/codebase/sbom")
        assert call_args[1]["json"]["format"] == "spdx"

    @pytest.mark.asyncio
    async def test_scan_vulnerabilities(self, mock_async_client: MagicMock) -> None:
        """Test scan_vulnerabilities calls correct endpoint."""
        from aragora.namespaces.dependency_analysis import AsyncDependencyAnalysisAPI

        mock_async_client.request.return_value = {
            "total_vulnerabilities": 3,
            "critical": 0,
            "high": 1,
        }

        api = AsyncDependencyAnalysisAPI(mock_async_client)
        result = await api.scan_vulnerabilities(
            repository_url="https://github.com/org/repo",
            severity_threshold="medium",
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/codebase/scan-vulnerabilities")
        assert result["total_vulnerabilities"] == 3

    @pytest.mark.asyncio
    async def test_check_licenses(self, mock_async_client: MagicMock) -> None:
        """Test check_licenses calls correct endpoint."""
        from aragora.namespaces.dependency_analysis import AsyncDependencyAnalysisAPI

        mock_async_client.request.return_value = {"compliant": True, "unknown": 2}

        api = AsyncDependencyAnalysisAPI(mock_async_client)
        result = await api.check_licenses(
            repository_url="https://github.com/org/repo",
            policy="copyleft",
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/codebase/check-licenses")
        assert result["compliant"] is True

    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_async_client: MagicMock) -> None:
        """Test clear_cache calls correct endpoint."""
        from aragora.namespaces.dependency_analysis import AsyncDependencyAnalysisAPI

        mock_async_client.request.return_value = {"cleared": True}

        api = AsyncDependencyAnalysisAPI(mock_async_client)
        result = await api.clear_cache()

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/codebase/clear-cache")
        assert result["cleared"] is True

    @pytest.mark.asyncio
    async def test_full_audit(self, mock_async_client: MagicMock) -> None:
        """Test full_audit convenience method."""
        from aragora.namespaces.dependency_analysis import AsyncDependencyAnalysisAPI

        mock_async_client.request.side_effect = [
            {"total_dependencies": 30},
            {"format": "cyclonedx"},
            {"total_vulnerabilities": 0, "critical": 0, "high": 0, "medium": 0},
            {"compliant": True},
        ]

        api = AsyncDependencyAnalysisAPI(mock_async_client)
        result = await api.full_audit({"repository_url": "https://github.com/org/repo"})

        assert mock_async_client.request.call_count == 4
        assert result["summary"]["risk_level"] == "low"
        assert result["summary"]["license_compliant"] is True
