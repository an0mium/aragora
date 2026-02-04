"""
Tests for SBOM (Software Bill of Materials) Handler.

Tests cover:
- SBOM generation (CycloneDX, SPDX formats)
- SBOM retrieval (specific and latest)
- SBOM listing
- SBOM download
- SBOM comparison
- Permission checks
- Error handling
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.base import HandlerResult


# ============================================================================
# Mock Classes
# ============================================================================


class MockSBOMResult:
    """Mock SBOM generation result."""

    def __init__(
        self,
        sbom_id: str = "sbom_123",
        format_value: str = "cyclonedx-json",
    ):
        self.sbom_id = sbom_id
        self.format = MagicMock()
        self.format.value = format_value
        self.filename = f"sbom.{format_value.split('-')[-1]}"
        self.component_count = 150
        self.vulnerability_count = 5
        self.license_count = 25
        self.generated_at = datetime.now(timezone.utc)
        self.content = self._generate_content(format_value)
        self.errors = []

    def _generate_content(self, format_value: str) -> str:
        if "json" in format_value:
            return json.dumps(
                {
                    "bomFormat": "CycloneDX" if "cyclone" in format_value else "SPDX",
                    "specVersion": "1.4",
                    "components": [
                        {"name": "lodash", "version": "4.17.21", "type": "library"},
                        {"name": "react", "version": "18.2.0", "type": "framework"},
                    ],
                }
            )
        return "SPDXVersion: SPDX-2.3\nDataLicense: CC0-1.0"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_sbom_generator():
    """Create mock SBOM generator."""
    generator = MagicMock()
    generator.include_dev_dependencies = True
    generator.include_vulnerabilities = True
    generator.generate_from_repo = AsyncMock(return_value=MockSBOMResult())
    return generator


@pytest.fixture
def mock_sbom_results():
    """Create mock SBOM results storage."""
    return {}


# ============================================================================
# SBOM Generation Tests
# ============================================================================


class TestSBOMGeneration:
    """Test SBOM generation endpoint."""

    @pytest.mark.asyncio
    async def test_generate_sbom_cyclonedx_json(self, mock_sbom_generator):
        """Test generating CycloneDX JSON SBOM."""
        from aragora.server.handlers.codebase.security.sbom import handle_generate_sbom

        with (
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_generator",
                return_value=mock_sbom_generator,
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_lock",
                return_value=MagicMock(),
            ),
        ):
            result = await handle_generate_sbom(
                repo_path="/path/to/repo",
                repo_id="test-repo",
                format="cyclonedx-json",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "sbom_id" in data
            assert data["format"] == "cyclonedx-json"
            assert data["component_count"] == 150

    @pytest.mark.asyncio
    async def test_generate_sbom_spdx_json(self, mock_sbom_generator):
        """Test generating SPDX JSON SBOM."""
        from aragora.server.handlers.codebase.security.sbom import handle_generate_sbom

        mock_sbom_generator.generate_from_repo.return_value = MockSBOMResult(
            format_value="spdx-json"
        )

        with (
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_generator",
                return_value=mock_sbom_generator,
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_lock",
                return_value=MagicMock(),
            ),
        ):
            result = await handle_generate_sbom(
                repo_path="/path/to/repo",
                repo_id="test-repo",
                format="spdx-json",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["format"] == "spdx-json"

    @pytest.mark.asyncio
    async def test_generate_sbom_invalid_format(self, mock_sbom_generator):
        """Test generating SBOM with invalid format returns 400."""
        from aragora.server.handlers.codebase.security.sbom import handle_generate_sbom

        with (
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_generator",
                return_value=mock_sbom_generator,
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_lock",
                return_value=MagicMock(),
            ),
        ):
            result = await handle_generate_sbom(
                repo_path="/path/to/repo",
                repo_id="test-repo",
                format="invalid-format",
            )

            assert result.status_code == 400
            response = json.loads(result.body.decode())
            assert "invalid" in response.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_generate_sbom_with_project_info(self, mock_sbom_generator):
        """Test generating SBOM with project name and version."""
        from aragora.server.handlers.codebase.security.sbom import handle_generate_sbom

        with (
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_generator",
                return_value=mock_sbom_generator,
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_lock",
                return_value=MagicMock(),
            ),
        ):
            result = await handle_generate_sbom(
                repo_path="/path/to/repo",
                repo_id="test-repo",
                project_name="MyProject",
                project_version="1.0.0",
            )

            assert result.status_code == 200
            mock_sbom_generator.generate_from_repo.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_sbom_exclude_dev_dependencies(self, mock_sbom_generator):
        """Test generating SBOM excluding dev dependencies."""
        from aragora.server.handlers.codebase.security.sbom import handle_generate_sbom

        with (
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_generator",
                return_value=mock_sbom_generator,
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_lock",
                return_value=MagicMock(),
            ),
        ):
            result = await handle_generate_sbom(
                repo_path="/path/to/repo",
                repo_id="test-repo",
                include_dev=False,
            )

            assert result.status_code == 200
            assert mock_sbom_generator.include_dev_dependencies is False


# ============================================================================
# SBOM Retrieval Tests
# ============================================================================


class TestSBOMRetrieval:
    """Test SBOM retrieval endpoint."""

    @pytest.mark.asyncio
    async def test_get_sbom_specific(self):
        """Test getting specific SBOM by ID."""
        from aragora.server.handlers.codebase.security.sbom import handle_get_sbom

        sbom = MockSBOMResult(sbom_id="sbom_123")

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={"sbom_123": sbom},
        ):
            result = await handle_get_sbom(
                repo_id="test-repo",
                sbom_id="sbom_123",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["sbom_id"] == "sbom_123"

    @pytest.mark.asyncio
    async def test_get_sbom_latest(self):
        """Test getting latest SBOM."""
        from aragora.server.handlers.codebase.security.sbom import handle_get_sbom

        sbom1 = MockSBOMResult(sbom_id="sbom_old")
        sbom1.generated_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
        sbom2 = MockSBOMResult(sbom_id="sbom_new")
        sbom2.generated_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={"sbom_old": sbom1, "sbom_new": sbom2},
        ):
            result = await handle_get_sbom(repo_id="test-repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            # Should return latest by generated_at
            assert data["sbom_id"] in ("sbom_new", "sbom_latest")

    @pytest.mark.asyncio
    async def test_get_sbom_not_found(self):
        """Test 404 when SBOM not found."""
        from aragora.server.handlers.codebase.security.sbom import handle_get_sbom

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={},
        ):
            result = await handle_get_sbom(
                repo_id="test-repo",
                sbom_id="nonexistent",
            )

            assert result.status_code == 404


# ============================================================================
# SBOM Listing Tests
# ============================================================================


class TestSBOMListing:
    """Test SBOM listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_sboms_success(self):
        """Test listing SBOMs."""
        from aragora.server.handlers.codebase.security.sbom import handle_list_sboms

        sbom1 = MockSBOMResult(sbom_id="sbom_1")
        sbom2 = MockSBOMResult(sbom_id="sbom_2")

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={"sbom_1": sbom1, "sbom_2": sbom2},
        ):
            result = await handle_list_sboms(repo_id="test-repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["count"] == 2
            assert len(data["sboms"]) == 2

    @pytest.mark.asyncio
    async def test_list_sboms_with_limit(self):
        """Test listing SBOMs with limit."""
        from aragora.server.handlers.codebase.security.sbom import handle_list_sboms

        sboms = {f"sbom_{i}": MockSBOMResult(sbom_id=f"sbom_{i}") for i in range(10)}

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value=sboms,
        ):
            result = await handle_list_sboms(repo_id="test-repo", limit=5)

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert len(data["sboms"]) == 5

    @pytest.mark.asyncio
    async def test_list_sboms_empty(self):
        """Test listing SBOMs when none exist."""
        from aragora.server.handlers.codebase.security.sbom import handle_list_sboms

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={},
        ):
            result = await handle_list_sboms(repo_id="test-repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["count"] == 0
            assert data["sboms"] == []


# ============================================================================
# SBOM Download Tests
# ============================================================================


class TestSBOMDownload:
    """Test SBOM download endpoint."""

    @pytest.mark.asyncio
    async def test_download_sbom_json(self):
        """Test downloading JSON SBOM."""
        from aragora.server.handlers.codebase.security.sbom import handle_download_sbom

        sbom = MockSBOMResult(sbom_id="sbom_123", format_value="cyclonedx-json")

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={"sbom_123": sbom},
        ):
            result = await handle_download_sbom(
                repo_id="test-repo",
                sbom_id="sbom_123",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["content_type"] == "application/json"
            assert "content" in data

    @pytest.mark.asyncio
    async def test_download_sbom_xml(self):
        """Test downloading XML SBOM."""
        from aragora.server.handlers.codebase.security.sbom import handle_download_sbom

        sbom = MockSBOMResult(sbom_id="sbom_123", format_value="cyclonedx-xml")
        sbom.format.value = "cyclonedx-xml"

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={"sbom_123": sbom},
        ):
            result = await handle_download_sbom(
                repo_id="test-repo",
                sbom_id="sbom_123",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["content_type"] == "application/xml"

    @pytest.mark.asyncio
    async def test_download_sbom_not_found(self):
        """Test 404 when downloading nonexistent SBOM."""
        from aragora.server.handlers.codebase.security.sbom import handle_download_sbom

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={},
        ):
            result = await handle_download_sbom(
                repo_id="test-repo",
                sbom_id="nonexistent",
            )

            assert result.status_code == 404


# ============================================================================
# SBOM Comparison Tests
# ============================================================================


class TestSBOMComparison:
    """Test SBOM comparison endpoint."""

    @pytest.mark.asyncio
    async def test_compare_sboms_success(self):
        """Test comparing two SBOMs."""
        from aragora.server.handlers.codebase.security.sbom import handle_compare_sboms

        sbom_a = MockSBOMResult(sbom_id="sbom_a")
        sbom_b = MockSBOMResult(sbom_id="sbom_b")
        # Make content different
        sbom_b.content = json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.4",
                "components": [
                    {"name": "lodash", "version": "4.17.21", "type": "library"},
                    {"name": "react", "version": "19.0.0", "type": "framework"},  # Updated
                    {"name": "axios", "version": "1.0.0", "type": "library"},  # Added
                ],
            }
        )

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={"sbom_a": sbom_a, "sbom_b": sbom_b},
        ):
            result = await handle_compare_sboms(
                repo_id="test-repo",
                sbom_id_a="sbom_a",
                sbom_id_b="sbom_b",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "diff" in data
            assert "summary" in data

    @pytest.mark.asyncio
    async def test_compare_sboms_identifies_added(self):
        """Test comparison identifies added components."""
        from aragora.server.handlers.codebase.security.sbom import handle_compare_sboms

        sbom_a = MockSBOMResult(sbom_id="sbom_a")
        sbom_a.content = json.dumps(
            {
                "bomFormat": "CycloneDX",
                "components": [{"name": "lodash", "version": "4.17.21"}],
            }
        )
        sbom_b = MockSBOMResult(sbom_id="sbom_b")
        sbom_b.content = json.dumps(
            {
                "bomFormat": "CycloneDX",
                "components": [
                    {"name": "lodash", "version": "4.17.21"},
                    {"name": "new-package", "version": "1.0.0"},
                ],
            }
        )

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={"sbom_a": sbom_a, "sbom_b": sbom_b},
        ):
            result = await handle_compare_sboms(
                repo_id="test-repo",
                sbom_id_a="sbom_a",
                sbom_id_b="sbom_b",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["summary"]["total_added"] >= 1

    @pytest.mark.asyncio
    async def test_compare_sboms_not_found_a(self):
        """Test 404 when first SBOM not found."""
        from aragora.server.handlers.codebase.security.sbom import handle_compare_sboms

        sbom_b = MockSBOMResult(sbom_id="sbom_b")

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={"sbom_b": sbom_b},
        ):
            result = await handle_compare_sboms(
                repo_id="test-repo",
                sbom_id_a="nonexistent",
                sbom_id_b="sbom_b",
            )

            assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_compare_sboms_not_found_b(self):
        """Test 404 when second SBOM not found."""
        from aragora.server.handlers.codebase.security.sbom import handle_compare_sboms

        sbom_a = MockSBOMResult(sbom_id="sbom_a")

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={"sbom_a": sbom_a},
        ):
            result = await handle_compare_sboms(
                repo_id="test-repo",
                sbom_id_a="sbom_a",
                sbom_id_b="nonexistent",
            )

            assert result.status_code == 404


# ============================================================================
# Permission Tests
# ============================================================================


class TestSBOMPermissions:
    """Test SBOM permission enforcement."""

    def test_generate_has_permission_decorator(self):
        """SBOM generation requires security permission."""
        from aragora.server.handlers.codebase.security.sbom import handle_generate_sbom
        import inspect

        source = inspect.getsource(handle_generate_sbom)
        assert "require_permission" in source

    def test_get_has_permission_decorator(self):
        """Get SBOM requires security permission."""
        from aragora.server.handlers.codebase.security.sbom import handle_get_sbom
        import inspect

        source = inspect.getsource(handle_get_sbom)
        assert "require_permission" in source


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestSBOMErrorHandling:
    """Test SBOM error handling."""

    @pytest.mark.asyncio
    async def test_generate_handles_generator_error(self, mock_sbom_generator):
        """Test SBOM generation handles generator errors."""
        from aragora.server.handlers.codebase.security.sbom import handle_generate_sbom

        mock_sbom_generator.generate_from_repo.side_effect = RuntimeError("Generator failed")

        with (
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_generator",
                return_value=mock_sbom_generator,
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.sbom.get_sbom_lock",
                return_value=MagicMock(),
            ),
        ):
            result = await handle_generate_sbom(
                repo_path="/path/to/repo",
                repo_id="test-repo",
            )

            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_compare_handles_parse_error(self):
        """Test SBOM comparison handles JSON parse errors gracefully."""
        from aragora.server.handlers.codebase.security.sbom import handle_compare_sboms

        sbom_a = MockSBOMResult(sbom_id="sbom_a")
        sbom_a.content = "invalid json {"
        sbom_b = MockSBOMResult(sbom_id="sbom_b")

        with patch(
            "aragora.server.handlers.codebase.security.sbom.get_or_create_sbom_results",
            return_value={"sbom_a": sbom_a, "sbom_b": sbom_b},
        ):
            result = await handle_compare_sboms(
                repo_id="test-repo",
                sbom_id_a="sbom_a",
                sbom_id_b="sbom_b",
            )

            # Should handle gracefully - either 200 with empty diff or 500
            assert result.status_code in (200, 500)


__all__ = [
    "TestSBOMGeneration",
    "TestSBOMRetrieval",
    "TestSBOMListing",
    "TestSBOMDownload",
    "TestSBOMComparison",
    "TestSBOMPermissions",
    "TestSBOMErrorHandling",
]
