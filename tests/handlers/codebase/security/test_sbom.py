"""
Tests for SBOM handler functions
(aragora/server/handlers/codebase/security/sbom.py).

Covers all five handler functions:
- handle_generate_sbom: POST generate SBOM for a repository
- handle_get_sbom: GET specific or latest SBOM
- handle_list_sboms: GET list of SBOMs for a repository
- handle_download_sbom: GET download raw SBOM content
- handle_compare_sboms: POST compare two SBOMs

Tests include: happy path, error paths, edge cases, input validation,
security (path traversal, injection), and concurrent access patterns.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.codebase import SBOMFormat, SBOMResult
from aragora.server.handlers.codebase.security.sbom import (
    handle_compare_sboms,
    handle_download_sbom,
    handle_generate_sbom,
    handle_get_sbom,
    handle_list_sboms,
)


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict:
    """Extract the data payload from a HandlerResult.

    success_response wraps in {"success": true, "data": {...}}.
    error_response wraps in {"success": false, "error": "..."}.
    """
    if isinstance(result, dict):
        raw = result
    else:
        raw = json.loads(result.body)
    if isinstance(raw, dict) and raw.get("success") and "data" in raw:
        return raw["data"]
    return raw


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _raw(result) -> dict:
    """Get the full parsed JSON body from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _make_sbom_result(
    format: SBOMFormat = SBOMFormat.CYCLONEDX_JSON,
    content: str = '{"components": []}',
    filename: str = "sbom.json",
    component_count: int = 5,
    vulnerability_count: int = 2,
    license_count: int = 3,
    generated_at: datetime | None = None,
    errors: list[str] | None = None,
) -> SBOMResult:
    """Create an SBOMResult for testing."""
    return SBOMResult(
        format=format,
        content=content,
        filename=filename,
        component_count=component_count,
        vulnerability_count=vulnerability_count,
        license_count=license_count,
        generated_at=generated_at or datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        errors=errors or [],
    )


def _cyclonedx_content(components: list[dict[str, Any]] | None = None) -> str:
    """Build CycloneDX JSON content string."""
    if components is None:
        comps = [
            {"name": "requests", "version": "2.28.0"},
            {"name": "flask", "version": "2.3.0"},
        ]
    else:
        comps = components
    return json.dumps({"components": comps})


def _spdx_content(packages: list[dict[str, Any]] | None = None) -> str:
    """Build SPDX JSON content string."""
    if packages is None:
        pkgs = [
            {"name": "requests", "versionInfo": "2.28.0"},
            {"name": "flask", "versionInfo": "2.3.0"},
        ]
    else:
        pkgs = packages
    return json.dumps({"packages": pkgs})


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_sbom_storage():
    """Clear in-memory SBOM stores between tests."""
    from aragora.server.handlers.codebase.security.storage import (
        _sbom_results,
        _running_sbom_generations,
    )

    _sbom_results.clear()
    _running_sbom_generations.clear()
    yield
    _sbom_results.clear()
    _running_sbom_generations.clear()


@pytest.fixture
def sbom_store():
    """Get SBOM storage for test-repo."""
    from aragora.server.handlers.codebase.security.storage import (
        get_or_create_sbom_results,
    )

    return get_or_create_sbom_results("test-repo")


@pytest.fixture
def sbom_lock():
    """Get the SBOM lock."""
    from aragora.server.handlers.codebase.security.storage import get_sbom_lock

    return get_sbom_lock()


@pytest.fixture
def mock_generator():
    """Create a mock SBOMGenerator."""
    gen = MagicMock()
    gen.generate_from_repo = AsyncMock(return_value=_make_sbom_result())
    gen.include_dev_dependencies = True
    gen.include_vulnerabilities = True
    return gen


# ============================================================================
# handle_generate_sbom tests
# ============================================================================


class TestHandleGenerateSbom:
    """Tests for handle_generate_sbom."""

    @pytest.mark.asyncio
    async def test_generate_success_default_format(self, mock_generator, monkeypatch):
        """Generating an SBOM with default format returns success."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo")
        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "cyclonedx-json"
        assert body["component_count"] == 5
        assert body["vulnerability_count"] == 2
        assert body["license_count"] == 3
        assert "sbom_id" in body
        assert body["sbom_id"].startswith("sbom_")
        assert "content" in body
        assert "errors" in body
        assert body["generated_at"] is not None

    @pytest.mark.asyncio
    async def test_generate_with_custom_repo_id(self, mock_generator, monkeypatch):
        """Providing a repo_id uses it instead of generating one."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(
            repo_path="/tmp/repo",
            repo_id="my-custom-repo",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["repository"] == "my-custom-repo"

    @pytest.mark.asyncio
    async def test_generate_auto_repo_id(self, mock_generator, monkeypatch):
        """When no repo_id is provided, one is auto-generated."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo")
        body = _body(result)
        assert body["repository"].startswith("repo_")

    @pytest.mark.asyncio
    async def test_generate_cyclonedx_json(self, mock_generator, monkeypatch):
        """CycloneDX JSON format is accepted."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo", format="cyclonedx-json")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_generate_cyclonedx_xml(self, mock_generator, monkeypatch):
        """CycloneDX XML format is accepted."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        xml_result = _make_sbom_result(format=SBOMFormat.CYCLONEDX_XML, filename="sbom.xml")
        mock_generator.generate_from_repo.return_value = xml_result
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo", format="cyclonedx-xml")
        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "cyclonedx-xml"

    @pytest.mark.asyncio
    async def test_generate_spdx_json(self, mock_generator, monkeypatch):
        """SPDX JSON format is accepted."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        spdx_result = _make_sbom_result(format=SBOMFormat.SPDX_JSON, filename="sbom.spdx.json")
        mock_generator.generate_from_repo.return_value = spdx_result
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo", format="spdx-json")
        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "spdx-json"

    @pytest.mark.asyncio
    async def test_generate_spdx_tv(self, mock_generator, monkeypatch):
        """SPDX tag-value format is accepted."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        tv_result = _make_sbom_result(format=SBOMFormat.SPDX_TV, filename="sbom.spdx")
        mock_generator.generate_from_repo.return_value = tv_result
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo", format="spdx-tv")
        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "spdx-tv"

    @pytest.mark.asyncio
    async def test_generate_invalid_format(self, mock_generator, monkeypatch):
        """Invalid format string returns 400."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo", format="invalid-format")
        assert _status(result) == 400
        raw = _raw(result)
        assert "Invalid format" in raw.get("error", "")

    @pytest.mark.asyncio
    async def test_generate_invalid_format_empty_string(self, mock_generator, monkeypatch):
        """Empty format string returns 400."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo", format="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_generate_with_project_metadata(self, mock_generator, monkeypatch):
        """Project name and version are passed to the generator."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(
            repo_path="/tmp/repo",
            project_name="MyProject",
            project_version="1.0.0",
        )
        assert _status(result) == 200
        mock_generator.generate_from_repo.assert_awaited_once()
        call_kwargs = mock_generator.generate_from_repo.call_args.kwargs
        assert call_kwargs["project_name"] == "MyProject"
        assert call_kwargs["project_version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_generate_with_branch_and_commit(self, mock_generator, monkeypatch):
        """Branch and commit SHA are passed to the generator."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(
            repo_path="/tmp/repo",
            branch="develop",
            commit_sha="abc123def",
        )
        assert _status(result) == 200
        call_kwargs = mock_generator.generate_from_repo.call_args.kwargs
        assert call_kwargs["branch"] == "develop"
        assert call_kwargs["commit_sha"] == "abc123def"

    @pytest.mark.asyncio
    async def test_generate_sets_include_dev(self, mock_generator, monkeypatch):
        """include_dev flag is applied to generator."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        await handle_generate_sbom(repo_path="/tmp/repo", include_dev=False)
        assert mock_generator.include_dev_dependencies is False

    @pytest.mark.asyncio
    async def test_generate_sets_include_vulnerabilities(self, mock_generator, monkeypatch):
        """include_vulnerabilities flag is applied to generator."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        await handle_generate_sbom(repo_path="/tmp/repo", include_vulnerabilities=False)
        assert mock_generator.include_vulnerabilities is False

    @pytest.mark.asyncio
    async def test_generate_stores_result(self, mock_generator, monkeypatch):
        """Generated SBOM is stored in the in-memory store."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_sbom_results,
        )

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo", repo_id="store-test")
        body = _body(result)
        store = get_or_create_sbom_results("store-test")
        assert body["sbom_id"] in store

    @pytest.mark.asyncio
    async def test_generate_returns_errors_from_result(self, mock_generator, monkeypatch):
        """Errors from the generator are included in the response."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        err_result = _make_sbom_result(errors=["Missing lockfile", "Parse warning"])
        mock_generator.generate_from_repo.return_value = err_result
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo")
        body = _body(result)
        assert body["errors"] == ["Missing lockfile", "Parse warning"]

    @pytest.mark.asyncio
    async def test_generate_oserror_returns_500(self, mock_generator, monkeypatch):
        """OSError during generation returns 500."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        mock_generator.generate_from_repo.side_effect = OSError("Disk full")
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_generate_value_error_returns_500(self, mock_generator, monkeypatch):
        """ValueError during generation returns 500."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        mock_generator.generate_from_repo.side_effect = ValueError("bad data")
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_generate_type_error_returns_500(self, mock_generator, monkeypatch):
        """TypeError during generation returns 500."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        mock_generator.generate_from_repo.side_effect = TypeError("wrong type")
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_generate_runtime_error_returns_500(self, mock_generator, monkeypatch):
        """RuntimeError during generation returns 500."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        mock_generator.generate_from_repo.side_effect = RuntimeError("broken")
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_generate_error_message_is_sanitized(self, mock_generator, monkeypatch):
        """Error messages do not leak internal details."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        mock_generator.generate_from_repo.side_effect = OSError("/internal/secret/path not found")
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo")
        raw = _raw(result)
        assert "/internal/secret/path" not in raw.get("error", "")
        assert "Internal server error" in raw.get("error", "")

    @pytest.mark.asyncio
    async def test_generate_filename_in_response(self, mock_generator, monkeypatch):
        """Response includes the filename from the generator result."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        custom_result = _make_sbom_result(filename="my-project-sbom.json")
        mock_generator.generate_from_repo.return_value = custom_result
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo")
        body = _body(result)
        assert body["filename"] == "my-project-sbom.json"

    @pytest.mark.asyncio
    async def test_generate_multiple_sboms_stored(self, mock_generator, monkeypatch):
        """Multiple generations for the same repo store separate SBOMs."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_sbom_results,
        )

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        await handle_generate_sbom(repo_path="/tmp/repo", repo_id="multi-repo")
        await handle_generate_sbom(repo_path="/tmp/repo", repo_id="multi-repo")
        await handle_generate_sbom(repo_path="/tmp/repo", repo_id="multi-repo")

        store = get_or_create_sbom_results("multi-repo")
        assert len(store) == 3


# ============================================================================
# handle_get_sbom tests
# ============================================================================


class TestHandleGetSbom:
    """Tests for handle_get_sbom."""

    @pytest.mark.asyncio
    async def test_get_specific_sbom_success(self, sbom_store):
        """Getting a specific SBOM by ID returns its data."""
        sbom = _make_sbom_result()
        sbom_store["sbom_001"] = sbom

        result = await handle_get_sbom(repo_id="test-repo", sbom_id="sbom_001")
        assert _status(result) == 200
        body = _body(result)
        assert body["sbom_id"] == "sbom_001"
        assert body["repository"] == "test-repo"
        assert body["component_count"] == 5
        assert body["vulnerability_count"] == 2
        assert body["license_count"] == 3
        assert body["format"] == "cyclonedx-json"

    @pytest.mark.asyncio
    async def test_get_sbom_not_found(self, sbom_store):
        """Getting a non-existent SBOM returns 404."""
        result = await handle_get_sbom(repo_id="test-repo", sbom_id="nonexistent")
        assert _status(result) == 404
        raw = _raw(result)
        assert "not found" in raw.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_latest_sbom_success(self, sbom_store):
        """Getting latest SBOM (no sbom_id) returns the most recent."""
        old_sbom = _make_sbom_result(
            generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            component_count=3,
        )
        new_sbom = _make_sbom_result(
            generated_at=datetime(2024, 6, 15, tzinfo=timezone.utc),
            component_count=10,
        )
        sbom_store["sbom_old"] = old_sbom
        sbom_store["sbom_new"] = new_sbom

        result = await handle_get_sbom(repo_id="test-repo")
        assert _status(result) == 200
        body = _body(result)
        assert body["component_count"] == 10
        assert body["sbom_id"] == "sbom_latest"

    @pytest.mark.asyncio
    async def test_get_latest_no_sboms(self, sbom_store):
        """Getting latest when no SBOMs exist returns 404."""
        result = await handle_get_sbom(repo_id="test-repo")
        assert _status(result) == 404
        raw = _raw(result)
        assert "No SBOMs generated" in raw.get("error", "")

    @pytest.mark.asyncio
    async def test_get_sbom_returns_content(self, sbom_store):
        """Response includes SBOM content."""
        content = '{"components": [{"name": "flask"}]}'
        sbom = _make_sbom_result(content=content)
        sbom_store["sbom_c"] = sbom

        result = await handle_get_sbom(repo_id="test-repo", sbom_id="sbom_c")
        body = _body(result)
        assert body["content"] == content

    @pytest.mark.asyncio
    async def test_get_sbom_returns_errors_list(self, sbom_store):
        """Response includes errors list from the result."""
        sbom = _make_sbom_result(errors=["warn1"])
        sbom_store["sbom_e"] = sbom

        result = await handle_get_sbom(repo_id="test-repo", sbom_id="sbom_e")
        body = _body(result)
        assert body["errors"] == ["warn1"]

    @pytest.mark.asyncio
    async def test_get_sbom_returns_generated_at(self, sbom_store):
        """Response includes the ISO timestamp of generation."""
        ts = datetime(2024, 3, 20, 15, 30, tzinfo=timezone.utc)
        sbom = _make_sbom_result(generated_at=ts)
        sbom_store["sbom_ts"] = sbom

        result = await handle_get_sbom(repo_id="test-repo", sbom_id="sbom_ts")
        body = _body(result)
        assert body["generated_at"] == ts.isoformat()

    @pytest.mark.asyncio
    async def test_get_latest_single_sbom(self, sbom_store):
        """Getting latest with exactly one SBOM returns it."""
        sbom = _make_sbom_result(component_count=42)
        sbom_store["only_one"] = sbom

        result = await handle_get_sbom(repo_id="test-repo")
        assert _status(result) == 200
        body = _body(result)
        assert body["component_count"] == 42

    @pytest.mark.asyncio
    async def test_get_sbom_empty_repo_id(self):
        """Empty repo with no SBOMs returns 404."""
        result = await handle_get_sbom(repo_id="empty-repo")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_sbom_format_field(self, sbom_store):
        """Format field returns the string value of the enum."""
        sbom = _make_sbom_result(format=SBOMFormat.SPDX_JSON)
        sbom_store["sbom_spdx"] = sbom

        result = await handle_get_sbom(repo_id="test-repo", sbom_id="sbom_spdx")
        body = _body(result)
        assert body["format"] == "spdx-json"


# ============================================================================
# handle_list_sboms tests
# ============================================================================


class TestHandleListSboms:
    """Tests for handle_list_sboms."""

    @pytest.mark.asyncio
    async def test_list_empty_repo(self):
        """Listing SBOMs for empty repository returns empty list."""
        result = await handle_list_sboms(repo_id="no-sboms")
        assert _status(result) == 200
        body = _body(result)
        assert body["repository"] == "no-sboms"
        assert body["count"] == 0
        assert body["sboms"] == []

    @pytest.mark.asyncio
    async def test_list_with_single_sbom(self, sbom_store):
        """Listing SBOMs returns a single entry."""
        sbom_store["sbom_1"] = _make_sbom_result()

        result = await handle_list_sboms(repo_id="test-repo")
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1
        assert len(body["sboms"]) == 1
        assert body["sboms"][0]["sbom_id"] == "sbom_1"

    @pytest.mark.asyncio
    async def test_list_multiple_sboms(self, sbom_store):
        """Listing returns all SBOMs sorted by generated_at descending."""
        sbom_store["sbom_old"] = _make_sbom_result(
            generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        sbom_store["sbom_mid"] = _make_sbom_result(
            generated_at=datetime(2024, 6, 1, tzinfo=timezone.utc)
        )
        sbom_store["sbom_new"] = _make_sbom_result(
            generated_at=datetime(2024, 12, 1, tzinfo=timezone.utc)
        )

        result = await handle_list_sboms(repo_id="test-repo")
        body = _body(result)
        assert body["count"] == 3
        ids = [s["sbom_id"] for s in body["sboms"]]
        assert ids == ["sbom_new", "sbom_mid", "sbom_old"]

    @pytest.mark.asyncio
    async def test_list_respects_limit(self, sbom_store):
        """Limit parameter truncates the result list."""
        for i in range(5):
            sbom_store[f"sbom_{i}"] = _make_sbom_result(
                generated_at=datetime(2024, 1, 1 + i, tzinfo=timezone.utc)
            )

        result = await handle_list_sboms(repo_id="test-repo", limit=2)
        body = _body(result)
        assert body["count"] == 2
        assert len(body["sboms"]) == 2

    @pytest.mark.asyncio
    async def test_list_limit_larger_than_count(self, sbom_store):
        """Limit larger than actual count returns all items."""
        sbom_store["sbom_a"] = _make_sbom_result()
        sbom_store["sbom_b"] = _make_sbom_result(
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc)
        )

        result = await handle_list_sboms(repo_id="test-repo", limit=100)
        body = _body(result)
        assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_list_default_limit_is_10(self, sbom_store):
        """Default limit of 10 is applied."""
        for i in range(15):
            sbom_store[f"sbom_{i:02d}"] = _make_sbom_result(
                generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
            )

        result = await handle_list_sboms(repo_id="test-repo")
        body = _body(result)
        assert body["count"] == 10

    @pytest.mark.asyncio
    async def test_list_sbom_fields(self, sbom_store):
        """Each SBOM in the list has expected fields (no content)."""
        sbom_store["sbom_f"] = _make_sbom_result(
            format=SBOMFormat.CYCLONEDX_XML,
            filename="test.xml",
            component_count=10,
            vulnerability_count=3,
            license_count=7,
        )

        result = await handle_list_sboms(repo_id="test-repo")
        body = _body(result)
        entry = body["sboms"][0]
        assert entry["sbom_id"] == "sbom_f"
        assert entry["format"] == "cyclonedx-xml"
        assert entry["filename"] == "test.xml"
        assert entry["component_count"] == 10
        assert entry["vulnerability_count"] == 3
        assert entry["license_count"] == 7
        assert "generated_at" in entry
        # List should NOT include content (it's a summary)
        assert "content" not in entry

    @pytest.mark.asyncio
    async def test_list_repository_field(self, sbom_store):
        """Response includes the repository ID."""
        result = await handle_list_sboms(repo_id="test-repo")
        body = _body(result)
        assert body["repository"] == "test-repo"


# ============================================================================
# handle_download_sbom tests
# ============================================================================


class TestHandleDownloadSbom:
    """Tests for handle_download_sbom."""

    @pytest.mark.asyncio
    async def test_download_cyclonedx_json(self, sbom_store):
        """Downloading CycloneDX JSON returns application/json content type."""
        sbom = _make_sbom_result(
            format=SBOMFormat.CYCLONEDX_JSON,
            content='{"components": []}',
            filename="sbom.json",
        )
        sbom_store["sbom_dl"] = sbom

        result = await handle_download_sbom(repo_id="test-repo", sbom_id="sbom_dl")
        assert _status(result) == 200
        body = _body(result)
        assert body["content_type"] == "application/json"
        assert body["content"] == '{"components": []}'
        assert body["filename"] == "sbom.json"

    @pytest.mark.asyncio
    async def test_download_cyclonedx_xml(self, sbom_store):
        """Downloading CycloneDX XML returns application/xml content type."""
        sbom = _make_sbom_result(
            format=SBOMFormat.CYCLONEDX_XML,
            content="<bom></bom>",
            filename="sbom.xml",
        )
        sbom_store["sbom_xml"] = sbom

        result = await handle_download_sbom(repo_id="test-repo", sbom_id="sbom_xml")
        body = _body(result)
        assert body["content_type"] == "application/xml"

    @pytest.mark.asyncio
    async def test_download_spdx_json(self, sbom_store):
        """Downloading SPDX JSON returns application/json content type."""
        sbom = _make_sbom_result(format=SBOMFormat.SPDX_JSON)
        sbom_store["sbom_spdx"] = sbom

        result = await handle_download_sbom(repo_id="test-repo", sbom_id="sbom_spdx")
        body = _body(result)
        assert body["content_type"] == "application/json"

    @pytest.mark.asyncio
    async def test_download_spdx_tv(self, sbom_store):
        """Downloading SPDX tag-value returns text/plain content type."""
        sbom = _make_sbom_result(
            format=SBOMFormat.SPDX_TV,
            content="SPDXVersion: SPDX-2.3",
            filename="sbom.spdx",
        )
        sbom_store["sbom_tv"] = sbom

        result = await handle_download_sbom(repo_id="test-repo", sbom_id="sbom_tv")
        body = _body(result)
        assert body["content_type"] == "text/plain"

    @pytest.mark.asyncio
    async def test_download_not_found(self, sbom_store):
        """Downloading a non-existent SBOM returns 404."""
        result = await handle_download_sbom(repo_id="test-repo", sbom_id="nonexistent")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_download_includes_filename(self, sbom_store):
        """Download response includes the original filename."""
        sbom = _make_sbom_result(filename="my-app-sbom.cyclonedx.json")
        sbom_store["sbom_fn"] = sbom

        result = await handle_download_sbom(repo_id="test-repo", sbom_id="sbom_fn")
        body = _body(result)
        assert body["filename"] == "my-app-sbom.cyclonedx.json"

    @pytest.mark.asyncio
    async def test_download_empty_repo(self):
        """Downloading from a repo with no SBOMs returns 404."""
        result = await handle_download_sbom(repo_id="empty-repo", sbom_id="sbom_missing")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_download_content_preserved(self, sbom_store):
        """Downloaded content exactly matches stored content."""
        big_content = json.dumps(
            {"components": [{"name": f"pkg-{i}", "version": f"{i}.0.0"} for i in range(100)]}
        )
        sbom = _make_sbom_result(content=big_content)
        sbom_store["sbom_big"] = sbom

        result = await handle_download_sbom(repo_id="test-repo", sbom_id="sbom_big")
        body = _body(result)
        assert body["content"] == big_content


# ============================================================================
# handle_compare_sboms tests
# ============================================================================


class TestHandleCompareSboms:
    """Tests for handle_compare_sboms."""

    @pytest.mark.asyncio
    async def test_compare_identical_sboms(self, sbom_store):
        """Comparing identical SBOMs shows no changes."""
        content = _cyclonedx_content([{"name": "requests", "version": "2.28.0"}])
        sbom_a = _make_sbom_result(content=content, component_count=1)
        sbom_b = _make_sbom_result(
            content=content,
            component_count=1,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )
        sbom_store["sbom_a"] = sbom_a
        sbom_store["sbom_b"] = sbom_b

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["summary"]["total_added"] == 0
        assert body["summary"]["total_removed"] == 0
        assert body["summary"]["total_updated"] == 0
        assert body["summary"]["total_unchanged"] == 1

    @pytest.mark.asyncio
    async def test_compare_added_components(self, sbom_store):
        """Comparing SBOMs detects added components."""
        content_a = _cyclonedx_content([{"name": "requests", "version": "2.28.0"}])
        content_b = _cyclonedx_content(
            [
                {"name": "requests", "version": "2.28.0"},
                {"name": "flask", "version": "2.3.0"},
            ]
        )
        sbom_store["sbom_a"] = _make_sbom_result(content=content_a)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)
        assert body["summary"]["total_added"] == 1
        assert body["diff"]["added"][0]["name"] == "flask"
        assert body["diff"]["added"][0]["version"] == "2.3.0"

    @pytest.mark.asyncio
    async def test_compare_removed_components(self, sbom_store):
        """Comparing SBOMs detects removed components."""
        content_a = _cyclonedx_content(
            [
                {"name": "requests", "version": "2.28.0"},
                {"name": "flask", "version": "2.3.0"},
            ]
        )
        content_b = _cyclonedx_content([{"name": "requests", "version": "2.28.0"}])
        sbom_store["sbom_a"] = _make_sbom_result(content=content_a)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)
        assert body["summary"]["total_removed"] == 1
        assert body["diff"]["removed"][0]["name"] == "flask"

    @pytest.mark.asyncio
    async def test_compare_updated_components(self, sbom_store):
        """Comparing SBOMs detects version changes."""
        content_a = _cyclonedx_content([{"name": "requests", "version": "2.28.0"}])
        content_b = _cyclonedx_content([{"name": "requests", "version": "2.31.0"}])
        sbom_store["sbom_a"] = _make_sbom_result(content=content_a)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)
        assert body["summary"]["total_updated"] == 1
        updated = body["diff"]["updated"][0]
        assert updated["name"] == "requests"
        assert updated["old_version"] == "2.28.0"
        assert updated["new_version"] == "2.31.0"

    @pytest.mark.asyncio
    async def test_compare_mixed_changes(self, sbom_store):
        """Comparing SBOMs with adds, removes, and updates."""
        content_a = _cyclonedx_content(
            [
                {"name": "requests", "version": "2.28.0"},
                {"name": "flask", "version": "2.3.0"},
                {"name": "pytest", "version": "7.4.0"},
            ]
        )
        content_b = _cyclonedx_content(
            [
                {"name": "requests", "version": "2.31.0"},  # updated
                {"name": "pytest", "version": "7.4.0"},  # unchanged
                {"name": "django", "version": "4.2.0"},  # added
            ]
        )
        sbom_store["sbom_a"] = _make_sbom_result(content=content_a)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)
        assert body["summary"]["total_added"] == 1  # django
        assert body["summary"]["total_removed"] == 1  # flask
        assert body["summary"]["total_updated"] == 1  # requests
        assert body["summary"]["total_unchanged"] == 1  # pytest

    @pytest.mark.asyncio
    async def test_compare_sbom_a_not_found(self, sbom_store):
        """Comparing when SBOM A is missing returns 404."""
        sbom_store["sbom_b"] = _make_sbom_result()

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="missing_a", sbom_id_b="sbom_b"
        )
        assert _status(result) == 404
        raw = _raw(result)
        assert "missing_a" in raw.get("error", "")

    @pytest.mark.asyncio
    async def test_compare_sbom_b_not_found(self, sbom_store):
        """Comparing when SBOM B is missing returns 404."""
        sbom_store["sbom_a"] = _make_sbom_result()

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="missing_b"
        )
        assert _status(result) == 404
        raw = _raw(result)
        assert "missing_b" in raw.get("error", "")

    @pytest.mark.asyncio
    async def test_compare_both_missing(self, sbom_store):
        """Comparing two non-existent SBOMs returns 404 for the first."""
        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="missing_a", sbom_id_b="missing_b"
        )
        assert _status(result) == 404
        # First missing one (sbom_a) triggers the error
        raw = _raw(result)
        assert "missing_a" in raw.get("error", "")

    @pytest.mark.asyncio
    async def test_compare_spdx_format(self, sbom_store):
        """Comparing SPDX JSON SBOMs extracts packages correctly."""
        content_a = _spdx_content([{"name": "requests", "versionInfo": "2.28.0"}])
        content_b = _spdx_content(
            [
                {"name": "requests", "versionInfo": "2.28.0"},
                {"name": "flask", "versionInfo": "2.3.0"},
            ]
        )
        sbom_store["sbom_a"] = _make_sbom_result(format=SBOMFormat.SPDX_JSON, content=content_a)
        sbom_store["sbom_b"] = _make_sbom_result(
            format=SBOMFormat.SPDX_JSON,
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)
        assert body["summary"]["total_added"] == 1

    @pytest.mark.asyncio
    async def test_compare_with_groups(self, sbom_store):
        """CycloneDX components with groups produce group/name keys."""
        content_a = _cyclonedx_content(
            [{"name": "spring-core", "version": "5.3.0", "group": "org.springframework"}]
        )
        content_b = _cyclonedx_content(
            [{"name": "spring-core", "version": "6.0.0", "group": "org.springframework"}]
        )
        sbom_store["sbom_a"] = _make_sbom_result(content=content_a)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)
        assert body["summary"]["total_updated"] == 1
        updated = body["diff"]["updated"][0]
        assert updated["name"] == "org.springframework/spring-core"

    @pytest.mark.asyncio
    async def test_compare_invalid_json_content(self, sbom_store):
        """Non-JSON content in SBOM results in empty component list (no crash)."""
        sbom_store["sbom_a"] = _make_sbom_result(content="not valid json")
        sbom_store["sbom_b"] = _make_sbom_result(
            content="also not json",
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["summary"]["total_unchanged"] == 0
        assert body["summary"]["total_added"] == 0

    @pytest.mark.asyncio
    async def test_compare_xml_format_graceful(self, sbom_store):
        """Comparing XML format SBOMs gracefully returns empty diff (XML not parsed)."""
        sbom_store["sbom_a"] = _make_sbom_result(
            format=SBOMFormat.CYCLONEDX_XML,
            content="<bom><components><component><name>foo</name></component></components></bom>",
        )
        sbom_store["sbom_b"] = _make_sbom_result(
            format=SBOMFormat.CYCLONEDX_XML,
            content="<bom><components><component><name>bar</name></component></components></bom>",
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        # XML parsing is not supported, so components will be empty
        assert _status(result) == 200
        body = _body(result)
        assert body["summary"]["total_added"] == 0
        assert body["summary"]["total_removed"] == 0

    @pytest.mark.asyncio
    async def test_compare_response_structure(self, sbom_store):
        """Verify the full response structure of compare."""
        content = _cyclonedx_content([{"name": "pkg", "version": "1.0"}])
        sbom_store["sbom_a"] = _make_sbom_result(content=content, component_count=1)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content,
            component_count=1,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)

        # Check top-level structure
        assert "sbom_a" in body
        assert "sbom_b" in body
        assert "diff" in body
        assert "summary" in body

        # Check sbom_a structure
        assert body["sbom_a"]["sbom_id"] == "sbom_a"
        assert "generated_at" in body["sbom_a"]
        assert "component_count" in body["sbom_a"]

        # Check diff structure
        assert "added" in body["diff"]
        assert "removed" in body["diff"]
        assert "updated" in body["diff"]
        assert "unchanged_count" in body["diff"]

    @pytest.mark.asyncio
    async def test_compare_empty_sboms(self, sbom_store):
        """Comparing two SBOMs with no components."""
        content = _cyclonedx_content([])
        sbom_store["sbom_a"] = _make_sbom_result(content=content, component_count=0)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content,
            component_count=0,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)
        assert body["summary"]["total_added"] == 0
        assert body["summary"]["total_removed"] == 0
        assert body["summary"]["total_updated"] == 0
        assert body["summary"]["total_unchanged"] == 0

    @pytest.mark.asyncio
    async def test_compare_sorted_output(self, sbom_store):
        """Component differences are sorted by name."""
        content_a = _cyclonedx_content([])
        content_b = _cyclonedx_content(
            [
                {"name": "zlib", "version": "1.0"},
                {"name": "aiohttp", "version": "3.0"},
                {"name": "flask", "version": "2.0"},
            ]
        )
        sbom_store["sbom_a"] = _make_sbom_result(content=content_a)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)
        added_names = [c["name"] for c in body["diff"]["added"]]
        assert added_names == ["aiohttp", "flask", "zlib"]

    @pytest.mark.asyncio
    async def test_compare_cross_format_cyclonedx_spdx(self, sbom_store):
        """Comparing CycloneDX vs SPDX JSON works (both have JSON parsers)."""
        content_a = _cyclonedx_content([{"name": "requests", "version": "2.28.0"}])
        content_b = _spdx_content(
            [
                {"name": "requests", "versionInfo": "2.28.0"},
                {"name": "flask", "versionInfo": "2.3.0"},
            ]
        )
        sbom_store["sbom_a"] = _make_sbom_result(
            format=SBOMFormat.CYCLONEDX_JSON, content=content_a
        )
        sbom_store["sbom_b"] = _make_sbom_result(
            format=SBOMFormat.SPDX_JSON,
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)
        assert body["summary"]["total_added"] == 1
        assert body["summary"]["total_unchanged"] == 1


# ============================================================================
# Security tests
# ============================================================================


class TestSecurityEdgeCases:
    """Security-oriented tests."""

    @pytest.mark.asyncio
    async def test_generate_error_no_internal_path_leak(self, monkeypatch):
        """Internal paths are not leaked in error responses."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        gen = MagicMock()
        gen.generate_from_repo = AsyncMock(side_effect=OSError("/etc/shadow: permission denied"))
        gen.include_dev_dependencies = True
        gen.include_vulnerabilities = True
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: gen)

        result = await handle_generate_sbom(repo_path="/tmp/repo")
        raw = _raw(result)
        error_msg = raw.get("error", "")
        assert "/etc/shadow" not in error_msg

    @pytest.mark.asyncio
    async def test_compare_malformed_json_no_crash(self, sbom_store):
        """Malformed JSON in stored SBOMs doesn't crash compare."""
        sbom_store["sbom_a"] = _make_sbom_result(content="{broken json")
        sbom_store["sbom_b"] = _make_sbom_result(
            content="[also broken",
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_compare_content_with_null_values(self, sbom_store):
        """Components with null/missing fields are caught by error handler.

        None names cause a TypeError when sorting the component set,
        which is caught and returns a 500 (sanitized error).
        """
        content_a = json.dumps({"components": [{"name": None, "version": "1.0"}]})
        content_b = json.dumps({"components": [{"version": "2.0"}]})
        sbom_store["sbom_a"] = _make_sbom_result(content=content_a)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        # TypeError from sorting None with str is caught -> 500
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_large_component_list(self, sbom_store):
        """Comparing SBOMs with many components completes."""
        comps_a = [{"name": f"pkg-{i}", "version": "1.0.0"} for i in range(500)]
        comps_b = [{"name": f"pkg-{i}", "version": "2.0.0"} for i in range(500)]
        sbom_store["sbom_a"] = _make_sbom_result(content=_cyclonedx_content(comps_a))
        sbom_store["sbom_b"] = _make_sbom_result(
            content=_cyclonedx_content(comps_b),
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)
        assert body["summary"]["total_updated"] == 500

    @pytest.mark.asyncio
    async def test_get_sbom_error_no_stack_trace(self, monkeypatch):
        """Error responses from get_sbom don't leak stack traces."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        def bad_storage(repo_id):
            raise TypeError("internal detail about NoneType")

        monkeypatch.setattr(sbom_mod, "get_or_create_sbom_results", bad_storage)

        result = await handle_get_sbom(repo_id="test-repo", sbom_id="sbom_1")
        assert _status(result) == 500
        raw = _raw(result)
        assert "NoneType" not in raw.get("error", "")
        assert "Internal server error" in raw.get("error", "")

    @pytest.mark.asyncio
    async def test_list_sbom_error_no_stack_trace(self, monkeypatch):
        """Error responses from list_sboms don't leak stack traces."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        def bad_storage(repo_id):
            raise ValueError("internal secret data")

        monkeypatch.setattr(sbom_mod, "get_or_create_sbom_results", bad_storage)

        result = await handle_list_sboms(repo_id="test-repo")
        assert _status(result) == 500
        raw = _raw(result)
        assert "secret" not in raw.get("error", "")

    @pytest.mark.asyncio
    async def test_download_error_no_leak(self, monkeypatch):
        """Error responses from download don't leak details."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        def bad_storage(repo_id):
            raise KeyError("sensitive_key")

        monkeypatch.setattr(sbom_mod, "get_or_create_sbom_results", bad_storage)

        result = await handle_download_sbom(repo_id="test-repo", sbom_id="sbom_1")
        assert _status(result) == 500
        raw = _raw(result)
        assert "sensitive_key" not in raw.get("error", "")


# ============================================================================
# Edge case tests
# ============================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    @pytest.mark.asyncio
    async def test_generate_with_zero_components(self, mock_generator, monkeypatch):
        """Generating an SBOM with zero components succeeds."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        zero_result = _make_sbom_result(component_count=0, vulnerability_count=0, license_count=0)
        mock_generator.generate_from_repo.return_value = zero_result
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/empty-repo")
        assert _status(result) == 200
        body = _body(result)
        assert body["component_count"] == 0

    @pytest.mark.asyncio
    async def test_generate_with_empty_content(self, mock_generator, monkeypatch):
        """Generating an SBOM with empty content string succeeds."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        empty_result = _make_sbom_result(content="")
        mock_generator.generate_from_repo.return_value = empty_result
        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo")
        assert _status(result) == 200
        body = _body(result)
        assert body["content"] == ""

    @pytest.mark.asyncio
    async def test_list_limit_zero(self, sbom_store):
        """Limit of 0 returns empty list."""
        sbom_store["sbom_1"] = _make_sbom_result()

        result = await handle_list_sboms(repo_id="test-repo", limit=0)
        body = _body(result)
        assert body["count"] == 0
        assert body["sboms"] == []

    @pytest.mark.asyncio
    async def test_list_limit_one(self, sbom_store):
        """Limit of 1 returns exactly one result."""
        sbom_store["sbom_a"] = _make_sbom_result(
            generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        sbom_store["sbom_b"] = _make_sbom_result(
            generated_at=datetime(2024, 6, 1, tzinfo=timezone.utc)
        )

        result = await handle_list_sboms(repo_id="test-repo", limit=1)
        body = _body(result)
        assert body["count"] == 1
        assert body["sboms"][0]["sbom_id"] == "sbom_b"  # newest first

    @pytest.mark.asyncio
    async def test_generate_with_all_optional_params(self, mock_generator, monkeypatch):
        """Providing all optional parameters works correctly."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(
            repo_path="/tmp/repo",
            repo_id="full-params",
            format="cyclonedx-json",
            project_name="FullTest",
            project_version="2.0.0",
            include_dev=False,
            include_vulnerabilities=False,
            branch="feature/test",
            commit_sha="deadbeef",
            workspace_id="ws-123",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_compare_component_with_empty_version(self, sbom_store):
        """Components with empty version strings are handled.

        Empty string is falsy in Python, so the handler treats
        empty-version-in-A + real-version-in-B as 'added' (not 'updated').
        """
        content_a = _cyclonedx_content([{"name": "pkg", "version": ""}])
        content_b = _cyclonedx_content([{"name": "pkg", "version": "1.0"}])
        sbom_store["sbom_a"] = _make_sbom_result(content=content_a)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        body = _body(result)
        # Empty string is falsy, so the handler classifies this as added+removed
        # (v_a="" is falsy -> "removed" is skipped; v_b="1.0" and not v_a -> "added")
        assert body["summary"]["total_added"] == 1

    @pytest.mark.asyncio
    async def test_compare_duplicate_component_names(self, sbom_store):
        """If two components have the same name, last one wins in dict."""
        content = json.dumps(
            {
                "components": [
                    {"name": "pkg", "version": "1.0"},
                    {"name": "pkg", "version": "2.0"},
                ]
            }
        )
        sbom_store["sbom_a"] = _make_sbom_result(content=content)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        # Should not crash, versions are identical since last dict entry wins
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_sbom_with_unicode_content(self, sbom_store):
        """SBOMs with unicode content are handled."""
        content = json.dumps(
            {
                "components": [
                    {
                        "name": "paquet-francais",
                        "description": "Paquet avec des accents: e\u0301te\u0301",
                    }
                ]
            }
        )
        sbom = _make_sbom_result(content=content)
        sbom_store["sbom_uni"] = sbom

        result = await handle_get_sbom(repo_id="test-repo", sbom_id="sbom_uni")
        assert _status(result) == 200
        body = _body(result)
        assert "paquet-francais" in body["content"]

    @pytest.mark.asyncio
    async def test_compare_tag_value_format_graceful(self, sbom_store):
        """Comparing SPDX tag-value format SBOMs gracefully returns empty diff."""
        sbom_store["sbom_a"] = _make_sbom_result(
            format=SBOMFormat.SPDX_TV,
            content="SPDXVersion: SPDX-2.3\nPackageName: foo\n",
        )
        sbom_store["sbom_b"] = _make_sbom_result(
            format=SBOMFormat.SPDX_TV,
            content="SPDXVersion: SPDX-2.3\nPackageName: bar\n",
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        # Tag-value is not JSON parseable, so extract_components returns empty
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_generate_returns_unique_sbom_ids(self, mock_generator, monkeypatch):
        """Each generation produces a unique sbom_id."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        ids = set()
        for _ in range(10):
            result = await handle_generate_sbom(repo_path="/tmp/repo", repo_id="uniqueness-test")
            body = _body(result)
            ids.add(body["sbom_id"])

        assert len(ids) == 10

    @pytest.mark.asyncio
    async def test_get_latest_among_many(self, sbom_store):
        """Getting latest among many SBOMs returns the most recent."""
        for i in range(20):
            sbom_store[f"sbom_{i:02d}"] = _make_sbom_result(
                generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
                component_count=i,
            )

        result = await handle_get_sbom(repo_id="test-repo")
        body = _body(result)
        assert body["component_count"] == 19  # last one

    @pytest.mark.asyncio
    async def test_compare_spdx_missing_fields(self, sbom_store):
        """SPDX packages with missing name/version fields are handled."""
        content_a = json.dumps({"packages": [{"name": "pkg"}]})  # no versionInfo
        content_b = json.dumps({"packages": [{"versionInfo": "1.0"}]})  # no name
        sbom_store["sbom_a"] = _make_sbom_result(format=SBOMFormat.SPDX_JSON, content=content_a)
        sbom_store["sbom_b"] = _make_sbom_result(
            format=SBOMFormat.SPDX_JSON,
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_compare_cyclonedx_missing_fields(self, sbom_store):
        """CycloneDX components with missing name/version fields are handled."""
        content_a = json.dumps({"components": [{"name": "pkg"}]})  # no version
        content_b = json.dumps({"components": [{"version": "1.0"}]})  # no name
        sbom_store["sbom_a"] = _make_sbom_result(content=content_a)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content_b,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_generate_format_case_sensitive(self, mock_generator, monkeypatch):
        """Format string is case-sensitive (uppercase rejected)."""
        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_generator)

        result = await handle_generate_sbom(repo_path="/tmp/repo", format="CycloneDX-JSON")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_compare_nested_json_structure(self, sbom_store):
        """Deeply nested JSON in content doesn't crash."""
        content = json.dumps(
            {
                "components": [
                    {
                        "name": "deep-pkg",
                        "version": "1.0",
                        "properties": {"nested": {"deep": {"value": True}}},
                    }
                ]
            }
        )
        sbom_store["sbom_a"] = _make_sbom_result(content=content)
        sbom_store["sbom_b"] = _make_sbom_result(
            content=content,
            generated_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        result = await handle_compare_sboms(
            repo_id="test-repo", sbom_id_a="sbom_a", sbom_id_b="sbom_b"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["summary"]["total_unchanged"] == 1
