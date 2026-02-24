"""Tests for Codebase namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestCodebaseDependencies:
    """Tests for dependency and SBOM operations."""

    def test_analyze_dependencies(self) -> None:
        """Analyze dependencies from manifest."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"dependencies": [], "vulnerabilities": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            manifest = '{"dependencies": {"react": "^18.0.0"}}'
            client.codebase.analyze_dependencies(manifest, manifest_type="npm")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/analyze-dependencies",
                json={"manifest": manifest, "type": "npm"},
            )
            client.close()

    def test_generate_sbom(self) -> None:
        """Generate Software Bill of Materials."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"sbom": {}, "format": "spdx"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.generate_sbom("myorg/myrepo", format="cyclonedx")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/sbom",
                json={"repo": "myorg/myrepo", "format": "cyclonedx"},
            )
            client.close()

    def test_check_licenses(self) -> None:
        """Check dependency licenses."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"compliant": True, "licenses": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.check_licenses("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/check-licenses",
                json={"repo": "myorg/myrepo"},
            )
            client.close()

    def test_scan_vulnerabilities(self) -> None:
        """Scan for known vulnerabilities."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"vulnerabilities": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.scan_vulnerabilities("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/scan-vulnerabilities",
                json={"repo": "myorg/myrepo"},
            )
            client.close()


class TestCodebaseAudit:
    """Tests for audit operations."""

    def test_clear_cache(self) -> None:
        """Clear analysis cache."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"cleared": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.clear_cache("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/clear-cache",
                json={"repo": "myorg/myrepo"},
            )
            client.close()


class TestAsyncCodebase:
    """Tests for async codebase API."""

    @pytest.mark.asyncio
    async def test_async_scan_vulnerabilities(self) -> None:
        """Scan for vulnerabilities asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"vulnerabilities": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.codebase.scan_vulnerabilities("myorg/myrepo")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/codebase/scan-vulnerabilities",
                    json={"repo": "myorg/myrepo"},
                )
