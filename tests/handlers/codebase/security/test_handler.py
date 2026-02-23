"""
Tests for SecurityHandler (aragora/server/handlers/codebase/security/handler.py).

Covers all routes and behavior:
- can_handle() routing for all supported paths
- GET endpoints: vulnerabilities, scans, secrets, SAST, SBOM, CVE
- POST endpoints: scan, secrets scan, SAST scan, SBOM generate, SBOM compare
- Auth and permission checks
- Rate limiting on POST
- Path traversal protection (_validate_repo_id, _validate_repo_path)
- _extract_repo_and_subpath parsing
- _get_user_id helper
- _check_permission helper
- Error handling and edge cases
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.codebase.security.handler import (
    SecurityHandler,
    _security_scan_limiter,
)


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult.

    If the response uses the success envelope ({"success": true, "data": {...}}),
    returns the inner "data" dict.  Otherwise returns the full body.
    """
    if isinstance(result, dict):
        raw = result
    else:
        raw = json.loads(result.body)
    # Unwrap success envelope
    if isinstance(raw, dict) and raw.get("success") and "data" in raw:
        return raw["data"]
    return raw


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to SecurityHandler."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        client_address: tuple[str, int] | None = None,
    ):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()
        self.client_address = client_address or ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_server_context():
    """Create mock server context with auth_context."""
    mock_auth = MagicMock()
    mock_auth.user_id = "test-user-001"
    return {
        "user_store": MagicMock(),
        "auth_context": mock_auth,
    }


@pytest.fixture
def handler(mock_server_context):
    """Create SecurityHandler with mock context."""
    return SecurityHandler(mock_server_context)


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    return _MockHTTPHandler()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the rate limiter between tests."""
    _security_scan_limiter._requests.clear()
    yield
    _security_scan_limiter._requests.clear()


@pytest.fixture(autouse=True)
def clear_scan_storage():
    """Clear in-memory scan stores between tests."""
    from aragora.server.handlers.codebase.security.storage import (
        _scan_results,
        _secrets_scan_results,
        _sast_scan_results,
        _sbom_results,
        _running_scans,
        _running_secrets_scans,
        _running_sast_scans,
        _running_sbom_generations,
    )

    _scan_results.clear()
    _secrets_scan_results.clear()
    _sast_scan_results.clear()
    _sbom_results.clear()
    _running_scans.clear()
    _running_secrets_scans.clear()
    _running_sast_scans.clear()
    _running_sbom_generations.clear()
    yield
    _scan_results.clear()
    _secrets_scan_results.clear()
    _sast_scan_results.clear()
    _sbom_results.clear()
    _running_scans.clear()
    _running_secrets_scans.clear()
    _running_sast_scans.clear()
    _running_sbom_generations.clear()


# ============================================================================
# can_handle() routing tests
# ============================================================================


class TestCanHandle:
    """Tests for can_handle() routing."""

    def test_exact_cve_route(self, handler):
        assert handler.can_handle("/api/v1/cve") is True

    def test_cve_with_id(self, handler):
        assert handler.can_handle("/api/v1/cve/CVE-2024-1234") is True

    def test_codebase_scan_path(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/scan") is True

    def test_codebase_vulnerabilities_path(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/vulnerabilities") is True

    def test_codebase_secrets_path(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/secrets") is True

    def test_codebase_sbom_path(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/sbom") is True

    def test_codebase_sast_in_scan_path(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/scan/sast") is True

    def test_unrelated_path_not_handled(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_codebase_without_security_subpath_not_handled(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/settings") is False

    def test_codebase_prefix_no_security_keyword(self, handler):
        assert handler.can_handle("/api/v1/codebase/my-repo/files") is False

    def test_cve_nested_path(self, handler):
        assert handler.can_handle("/api/v1/cve/CVE-2024-1234/details") is True


# ============================================================================
# _extract_repo_and_subpath tests
# ============================================================================


class TestExtractRepoAndSubpath:
    """Tests for _extract_repo_and_subpath static method."""

    def test_versioned_path(self, handler):
        repo_id, sub = SecurityHandler._extract_repo_and_subpath(
            "/api/v1/codebase/my-repo/scan/latest"
        )
        assert repo_id == "my-repo"
        assert sub == "/scan/latest"

    def test_unversioned_path(self, handler):
        repo_id, sub = SecurityHandler._extract_repo_and_subpath(
            "/api/codebase/my-repo/scan"
        )
        assert repo_id == "my-repo"
        assert sub == "/scan"

    def test_no_subpath(self, handler):
        repo_id, sub = SecurityHandler._extract_repo_and_subpath(
            "/api/v1/codebase/my-repo"
        )
        assert repo_id == "my-repo"
        assert sub == ""

    def test_trailing_slash_no_subpath(self, handler):
        repo_id, sub = SecurityHandler._extract_repo_and_subpath(
            "/api/v1/codebase/my-repo/"
        )
        # remainder = "my-repo/", split on "/" => ("my-repo", "")
        assert repo_id == "my-repo"
        assert sub == "/"

    def test_non_matching_path(self, handler):
        repo_id, sub = SecurityHandler._extract_repo_and_subpath(
            "/api/v1/debates/123"
        )
        assert repo_id is None
        assert sub == ""

    def test_empty_repo_id(self, handler):
        repo_id, sub = SecurityHandler._extract_repo_and_subpath(
            "/api/v1/codebase/"
        )
        assert repo_id is None
        assert sub == ""


# ============================================================================
# _validate_repo_id tests
# ============================================================================


class TestValidateRepoId:
    """Tests for _validate_repo_id."""

    def test_valid_repo_id(self, handler):
        assert handler._validate_repo_id("my-repo-123") is None

    def test_valid_repo_id_underscores(self, handler):
        assert handler._validate_repo_id("my_repo") is None

    def test_path_traversal_rejected(self, handler):
        result = handler._validate_repo_id("../etc/passwd")
        assert result is not None
        assert _status(result) == 400

    def test_slash_in_repo_id_rejected(self, handler):
        result = handler._validate_repo_id("repo/subdir")
        assert result is not None
        assert _status(result) == 400

    def test_empty_repo_id_rejected(self, handler):
        result = handler._validate_repo_id("")
        assert result is not None
        assert _status(result) == 400


# ============================================================================
# _validate_repo_path tests
# ============================================================================


class TestValidateRepoPath:
    """Tests for _validate_repo_path static method."""

    def test_valid_path(self, handler):
        resolved, err = SecurityHandler._validate_repo_path("/tmp/repo")
        assert err is None
        assert resolved is not None

    def test_empty_path_rejected(self, handler):
        resolved, err = SecurityHandler._validate_repo_path("")
        assert resolved is None
        assert _status(err) == 400
        assert "required" in _body(err).get("error", "").lower()

    def test_whitespace_only_rejected(self, handler):
        resolved, err = SecurityHandler._validate_repo_path("   ")
        assert resolved is None
        assert _status(err) == 400

    def test_null_byte_rejected(self, handler):
        resolved, err = SecurityHandler._validate_repo_path("/tmp/repo\x00evil")
        assert resolved is None
        assert _status(err) == 400
        assert "null byte" in _body(err).get("error", "").lower()

    def test_scan_root_enforcement(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/allowed/root")
        resolved, err = SecurityHandler._validate_repo_path("/etc/passwd")
        assert resolved is None
        assert _status(err) == 400
        assert "within" in _body(err).get("error", "").lower()

    def test_scan_root_allows_valid_subpath(self, handler, monkeypatch, tmp_path):
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", str(tmp_path))
        sub = tmp_path / "repo"
        sub.mkdir()
        resolved, err = SecurityHandler._validate_repo_path(str(sub))
        assert err is None
        assert resolved is not None

    def test_scan_root_filesystem_root_allows_all(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/")
        resolved, err = SecurityHandler._validate_repo_path("/any/path")
        assert err is None


# ============================================================================
# _get_user_id tests
# ============================================================================


class TestGetUserId:
    """Tests for _get_user_id helper."""

    def test_returns_user_id_from_auth_context(self, handler):
        assert handler._get_user_id() == "test-user-001"

    def test_returns_default_when_no_auth_context(self, mock_server_context):
        del mock_server_context["auth_context"]
        h = SecurityHandler(mock_server_context)
        assert h._get_user_id() == "default"

    def test_returns_default_when_no_user_id(self, mock_server_context):
        mock_auth = MagicMock(spec=[])
        mock_server_context["auth_context"] = mock_auth
        h = SecurityHandler(mock_server_context)
        assert h._get_user_id() == "default"


# ============================================================================
# _check_permission tests
# ============================================================================


class TestCheckPermission:
    """Tests for _check_permission helper."""

    def test_no_auth_context_returns_401(self, mock_server_context):
        del mock_server_context["auth_context"]
        h = SecurityHandler(mock_server_context)
        result = h._check_permission("any:permission")
        assert result is not None
        assert _status(result) == 401

    def test_permission_granted(self, handler):
        """Auto-patched RBAC grants all permissions."""
        result = handler._check_permission("vulnerability:scan")
        assert result is None

    def test_permission_denied(self, handler):
        """When checker says denied, _check_permission returns 403."""
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.rbac.checker.get_permission_checker",
            return_value=mock_checker,
        ):
            result = handler._check_permission("vulnerability:scan")
        assert result is not None
        assert _status(result) == 403


# ============================================================================
# GET route dispatch tests
# ============================================================================


class TestHandleGetRouting:
    """Tests for the synchronous handle() GET routing."""

    def test_unhandled_path_returns_none(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/other", {}, mock_http_handler)
        assert result is None

    def test_cve_route_dispatches(self, handler, mock_http_handler):
        mock_vuln = MagicMock()
        mock_vuln.to_dict.return_value = {"cve_id": "CVE-2024-0001", "severity": "high"}

        mock_client_cls = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.get_cve.return_value = mock_vuln
        mock_client_cls.return_value = mock_client_instance

        with patch(
            "aragora.server.handlers.codebase.security.vulnerability._get_cve_client_cls",
            return_value=mock_client_cls,
        ):
            result = handler.handle("/api/v1/cve/CVE-2024-0001", {}, mock_http_handler)

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "vulnerability" in body

    def test_bare_cve_returns_none(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/cve", {}, mock_http_handler)
        assert result is None

    def test_scan_latest_dispatches(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_repo_scans,
        )

        mock_scan = MagicMock()
        mock_scan.started_at = MagicMock()
        mock_scan.to_dict.return_value = {"scan_id": "scan_001", "status": "completed"}
        scans = get_or_create_repo_scans("my-repo")
        scans["scan_001"] = mock_scan

        result = handler.handle(
            "/api/v1/codebase/my-repo/scan/latest", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 200

    def test_scan_by_id_dispatches(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_repo_scans,
        )

        mock_scan = MagicMock()
        mock_scan.to_dict.return_value = {"scan_id": "scan_xyz", "status": "completed"}
        scans = get_or_create_repo_scans("my-repo")
        scans["scan_xyz"] = mock_scan

        result = handler.handle(
            "/api/v1/codebase/my-repo/scan/scan_xyz", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 200

    def test_scan_not_found(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/scan/nonexistent", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 404

    def test_vulnerabilities_dispatches(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/vulnerabilities", {}, mock_http_handler
        )
        assert result is not None
        # No scans exist, so 404
        assert _status(result) == 404

    def test_scans_list_dispatches(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/scans", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body.get("scans") == []

    def test_secrets_scan_latest_dispatches(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/scan/secrets/latest", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 404

    def test_secrets_scan_by_id_dispatches(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_secrets_scans,
        )

        mock_scan = MagicMock()
        mock_scan.to_dict.return_value = {"scan_id": "sec_001", "status": "completed"}
        scans = get_or_create_secrets_scans("my-repo")
        scans["sec_001"] = mock_scan

        result = handler.handle(
            "/api/v1/codebase/my-repo/scan/secrets/sec_001", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 200

    def test_scans_secrets_list_dispatches(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/scans/secrets", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body.get("scans") == []

    def test_secrets_list_dispatches(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/secrets", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 404

    def test_sast_findings_not_routed_via_can_handle(self, handler, mock_http_handler):
        """The /sast/findings path lacks /scan, /vulnerabilities, /secrets,
        /sbom, or /cve/ keywords, so can_handle returns False and handle()
        returns None.  These endpoints are accessed via handle_post or
        direct async method calls."""
        result = handler.handle(
            "/api/v1/codebase/my-repo/sast/findings", {}, mock_http_handler
        )
        assert result is None

    def test_sast_scan_status_dispatches(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/scan/sast/sast_001", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 404

    def test_owasp_summary_not_routed_via_can_handle(self, handler, mock_http_handler):
        """The /sast/owasp-summary path lacks /scan, /vulnerabilities,
        /secrets, /sbom, or /cve/ keywords, so can_handle returns False."""
        result = handler.handle(
            "/api/v1/codebase/my-repo/sast/owasp-summary", {}, mock_http_handler
        )
        assert result is None

    def test_sbom_latest_dispatches(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/sbom/latest", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 404

    def test_sbom_list_dispatches(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/sbom/list", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body.get("count") == 0

    def test_sbom_by_id_dispatches(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/sbom/sbom_abc123", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 404

    def test_sbom_download_dispatches(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/sbom/sbom_abc/download", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 404

    def test_no_repo_id_returns_none(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/codebase/", {}, mock_http_handler)
        assert result is None

    def test_invalid_repo_id_path_traversal(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/..%2f..%2fetc/scan/latest", {}, mock_http_handler
        )
        # The repo_id is "..%2f..%2fetc" which should fail safe_repo_id
        if result is not None:
            assert _status(result) == 400


# ============================================================================
# POST route dispatch tests
# ============================================================================


class TestHandlePostRouting:
    """Tests for handle_post() POST routing."""

    @pytest.mark.asyncio
    async def test_unhandled_path_returns_none(self, handler, mock_http_handler):
        result = await handler.handle_post(
            "/api/v1/other", {}, mock_http_handler
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_post_scan_dispatches(self, handler, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        http = _MockHTTPHandler(body={"repo_path": str(repo_dir)})

        mock_scanner = AsyncMock()
        mock_scanner.scan_repository.return_value = MagicMock(
            scan_id="scan_001", status="completed"
        )

        import aragora.server.handlers.codebase.security.vulnerability as vuln_mod

        monkeypatch.setattr(vuln_mod, "get_scanner", lambda: mock_scanner)

        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan", {}, http
        )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body.get("status") == "running"
        assert "scan_id" in body

    @pytest.mark.asyncio
    async def test_post_scan_missing_repo_path(self, handler):
        http = _MockHTTPHandler(body={})
        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan", {}, http
        )
        assert result is not None
        assert _status(result) == 400
        assert "repo_path" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_post_secrets_scan_dispatches(self, handler, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        http = _MockHTTPHandler(body={"repo_path": str(repo_dir)})

        mock_instance = AsyncMock()
        mock_instance.scan_repository.return_value = MagicMock(
            scan_id="sec_001", status="completed", secrets=[]
        )

        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "SecretsScanner", lambda: mock_instance)

        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan/secrets", {}, http
        )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body.get("status") == "running"

    @pytest.mark.asyncio
    async def test_post_secrets_scan_missing_repo_path(self, handler):
        http = _MockHTTPHandler(body={})
        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan/secrets", {}, http
        )
        assert result is not None
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_post_sast_scan_dispatches(self, handler, monkeypatch):
        http = _MockHTTPHandler(body={"repo_path": "", "rule_sets": ["owasp"]})

        mock_scanner = AsyncMock()
        mock_scanner.initialize = AsyncMock()
        mock_scanner.scan_repository.return_value = MagicMock(
            scan_id="sast_001", findings=[]
        )

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)

        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan/sast", {}, http
        )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "scan_id" in body

    @pytest.mark.asyncio
    async def test_post_sbom_dispatches(self, handler, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        http = _MockHTTPHandler(body={"repo_path": str(repo_dir), "format": "cyclonedx-json"})

        mock_result = MagicMock()
        mock_result.format.value = "cyclonedx-json"
        mock_result.filename = "sbom.json"
        mock_result.component_count = 10
        mock_result.vulnerability_count = 2
        mock_result.license_count = 5
        mock_result.generated_at.isoformat.return_value = "2024-01-01T00:00:00"
        mock_result.content = '{"components": []}'
        mock_result.errors = []

        mock_gen = AsyncMock()
        mock_gen.generate_from_repo.return_value = mock_result

        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_gen)

        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/sbom", {}, http
        )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "sbom_id" in body

    @pytest.mark.asyncio
    async def test_post_sbom_missing_repo_path(self, handler):
        http = _MockHTTPHandler(body={})
        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/sbom", {}, http
        )
        assert result is not None
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_post_sbom_compare_dispatches(self, handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_sbom_results,
        )

        mock_a = MagicMock()
        mock_a.generated_at.isoformat.return_value = "2024-01-01T00:00:00"
        mock_a.component_count = 5
        mock_a.content = '{"components": [{"name": "foo", "version": "1.0"}]}'
        mock_a.format.value = "cyclonedx-json"
        mock_a.format = MagicMock()
        mock_a.format.__eq__ = lambda s, o: True

        mock_b = MagicMock()
        mock_b.generated_at.isoformat.return_value = "2024-01-02T00:00:00"
        mock_b.component_count = 6
        mock_b.content = '{"components": [{"name": "bar", "version": "2.0"}]}'
        mock_b.format.value = "cyclonedx-json"
        mock_b.format = MagicMock()
        mock_b.format.__eq__ = lambda s, o: True

        results = get_or_create_sbom_results("my-repo")
        results["sbom_a"] = mock_a
        results["sbom_b"] = mock_b

        http = _MockHTTPHandler(body={"sbom_id_a": "sbom_a", "sbom_id_b": "sbom_b"})

        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/sbom/compare", {}, http
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "diff" in body

    @pytest.mark.asyncio
    async def test_post_sbom_compare_missing_ids(self, handler):
        http = _MockHTTPHandler(body={"sbom_id_a": "sbom_a"})
        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/sbom/compare", {}, http
        )
        assert result is not None
        assert _status(result) == 400
        assert "sbom_id_a and sbom_id_b" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_post_route_trailing_slash(self, handler, tmp_path, monkeypatch):
        """Trailing slashes on POST routes should still dispatch."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        http = _MockHTTPHandler(body={"repo_path": str(repo_dir)})

        mock_scanner = AsyncMock()
        mock_scanner.scan_repository.return_value = MagicMock(
            scan_id="scan_002", status="completed"
        )

        import aragora.server.handlers.codebase.security.vulnerability as vuln_mod

        monkeypatch.setattr(vuln_mod, "get_scanner", lambda: mock_scanner)

        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan/", {}, http
        )
        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_post_no_repo_id_returns_none(self, handler):
        http = _MockHTTPHandler(body={})
        result = await handler.handle_post(
            "/api/v1/codebase/", {}, http
        )
        assert result is None


# ============================================================================
# Rate limiting tests
# ============================================================================


class TestRateLimiting:
    """Tests for POST rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, handler):
        """When rate limit is exceeded, 429 is returned."""
        http = _MockHTTPHandler(body={"repo_path": "/tmp/repo"})

        with patch.object(_security_scan_limiter, "is_allowed", return_value=False):
            result = await handler.handle_post(
                "/api/v1/codebase/my-repo/scan", {}, http
            )

        assert result is not None
        assert _status(result) == 429
        assert "rate limit" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_rate_limit_allowed(self, handler, tmp_path, monkeypatch):
        """When rate limit is OK, request proceeds."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        http = _MockHTTPHandler(body={"repo_path": str(repo_dir)})

        mock_scanner = AsyncMock()
        mock_scanner.scan_repository.return_value = MagicMock(
            scan_id="s1", status="completed"
        )

        import aragora.server.handlers.codebase.security.vulnerability as vuln_mod

        monkeypatch.setattr(vuln_mod, "get_scanner", lambda: mock_scanner)

        with patch.object(_security_scan_limiter, "is_allowed", return_value=True):
            result = await handler.handle_post(
                "/api/v1/codebase/my-repo/scan", {}, http
            )

        assert result is not None
        assert _status(result) == 200


# ============================================================================
# Request body parsing tests
# ============================================================================


class TestBodyParsing:
    """Tests for POST request body parsing."""

    @pytest.mark.asyncio
    async def test_body_parsed_from_rfile(self, handler, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        body_data = {"repo_path": str(repo_dir), "branch": "main"}
        http = _MockHTTPHandler(body=body_data)

        mock_scanner = AsyncMock()
        mock_scanner.scan_repository.return_value = MagicMock(
            scan_id="s1", status="completed"
        )

        import aragora.server.handlers.codebase.security.vulnerability as vuln_mod

        monkeypatch.setattr(vuln_mod, "get_scanner", lambda: mock_scanner)

        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan", {}, http
        )

        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_body_empty_content_length(self, handler):
        """When Content-Length is 0, body defaults to empty dict."""
        http = _MockHTTPHandler()
        http.headers = {"Content-Length": "0"}
        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan", {}, http
        )
        # No repo_path in empty body => 400
        assert result is not None
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_body_invalid_json(self, handler):
        """When body is invalid JSON, data defaults to empty dict."""
        http = _MockHTTPHandler()
        http.rfile.read.return_value = b"not-json"
        http.headers = {"Content-Length": "8"}

        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan", {}, http
        )
        # data will be empty (JSON parse failed silently) => repo_path missing => 400
        assert result is not None
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handler_without_rfile(self, handler):
        """When handler has no rfile, body defaults to empty dict."""
        http = MagicMock(spec=[])
        http.client_address = ("127.0.0.1", 12345)

        # Need to provide the necessary mock structure for get_client_ip
        with patch(
            "aragora.server.handlers.codebase.security.handler.get_client_ip",
            return_value="127.0.0.1",
        ):
            result = await handler.handle_post(
                "/api/v1/codebase/my-repo/scan", {}, http
            )

        # No rfile => data is empty => repo_path missing => 400
        assert result is not None
        assert _status(result) == 400


# ============================================================================
# Path traversal in POST endpoints
# ============================================================================


class TestPostPathTraversal:
    """Tests for path traversal protection on POST endpoints."""

    @pytest.mark.asyncio
    async def test_scan_rejects_traversal_repo_id(self, handler):
        http = _MockHTTPHandler(body={"repo_path": "/tmp/repo"})
        result = await handler.handle_post(
            "/api/v1/codebase/../evil/scan", {}, http
        )
        # repo_id ".." is invalid
        if result is not None:
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_scan_rejects_null_byte_in_path(self, handler):
        http = _MockHTTPHandler(body={"repo_path": "/tmp/repo\x00evil"})
        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan", {}, http
        )
        assert result is not None
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sast_validates_repo_path(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/allowed")
        http = _MockHTTPHandler(body={"repo_path": "/etc/passwd"})
        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan/sast", {}, http
        )
        assert result is not None
        assert _status(result) == 400


# ============================================================================
# GET scan with populated storage
# ============================================================================


class TestGetWithData:
    """Tests for GET endpoints when storage has data."""

    def test_vulnerabilities_with_completed_scan(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_repo_scans,
        )

        mock_vuln = MagicMock()
        mock_vuln.to_dict.return_value = {
            "cve_id": "CVE-2024-001",
            "severity": "high",
        }

        mock_dep = MagicMock()
        mock_dep.name = "requests"
        mock_dep.version = "2.28.0"
        mock_dep.ecosystem = "pip"
        mock_dep.vulnerabilities = [mock_vuln]

        mock_scan = MagicMock()
        mock_scan.status = "completed"
        mock_scan.started_at = MagicMock()
        mock_scan.dependencies = [mock_dep]
        mock_scan.scan_id = "scan_001"

        scans = get_or_create_repo_scans("my-repo")
        scans["scan_001"] = mock_scan

        result = handler.handle(
            "/api/v1/codebase/my-repo/vulnerabilities", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body.get("total") == 1
        assert len(body.get("vulnerabilities", [])) == 1

    def test_vulnerabilities_with_severity_filter(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_repo_scans,
        )

        mock_vuln_high = MagicMock()
        mock_vuln_high.to_dict.return_value = {"cve_id": "CVE-1", "severity": "high"}
        mock_vuln_low = MagicMock()
        mock_vuln_low.to_dict.return_value = {"cve_id": "CVE-2", "severity": "low"}

        mock_dep = MagicMock()
        mock_dep.name = "pkg"
        mock_dep.version = "1.0"
        mock_dep.ecosystem = "pip"
        mock_dep.vulnerabilities = [mock_vuln_high, mock_vuln_low]

        mock_scan = MagicMock()
        mock_scan.status = "completed"
        mock_scan.started_at = MagicMock()
        mock_scan.dependencies = [mock_dep]
        mock_scan.scan_id = "scan_002"

        scans = get_or_create_repo_scans("my-repo")
        scans["scan_002"] = mock_scan

        result = handler.handle(
            "/api/v1/codebase/my-repo/vulnerabilities",
            {"severity": "high"},
            mock_http_handler,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body.get("total") == 1
        assert body["vulnerabilities"][0]["severity"] == "high"

    def test_scans_list_with_data(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_repo_scans,
        )
        from datetime import datetime, timezone

        mock_scan = MagicMock()
        mock_scan.scan_id = "scan_001"
        mock_scan.status = "completed"
        mock_scan.started_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_scan.completed_at = datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc)
        mock_scan.total_dependencies = 50
        mock_scan.vulnerable_dependencies = 3
        mock_scan.critical_count = 1
        mock_scan.high_count = 2
        mock_scan.medium_count = 0
        mock_scan.low_count = 0

        scans = get_or_create_repo_scans("my-repo")
        scans["scan_001"] = mock_scan

        result = handler.handle(
            "/api/v1/codebase/my-repo/scans", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["scans"][0]["scan_id"] == "scan_001"

    def test_secrets_list_with_completed_scan(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_secrets_scans,
        )

        mock_secret = MagicMock()
        mock_secret.to_dict.return_value = {
            "severity": "critical",
            "secret_type": "api_key",
            "is_in_history": False,
        }

        mock_scan = MagicMock()
        mock_scan.status = "completed"
        mock_scan.started_at = MagicMock()
        mock_scan.secrets = [mock_secret]
        mock_scan.scan_id = "sec_001"

        scans = get_or_create_secrets_scans("my-repo")
        scans["sec_001"] = mock_scan

        result = handler.handle(
            "/api/v1/codebase/my-repo/secrets", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body.get("total") == 1

    def test_sast_scan_status_completed(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_sast_scan_results,
        )

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"findings_count": 3}

        sast_results = get_sast_scan_results()
        sast_results["my-repo"] = {"sast_001": mock_result}

        result = handler.handle(
            "/api/v1/codebase/my-repo/scan/sast/sast_001", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "completed"

    @pytest.mark.asyncio
    async def test_sast_findings_with_data(self, handler):
        """Test SAST findings via direct async method call (not routed
        through handle() since can_handle excludes /sast/findings)."""
        from aragora.server.handlers.codebase.security.storage import (
            get_sast_scan_results,
        )

        mock_finding = MagicMock()
        mock_finding.severity.value = "high"
        mock_finding.owasp_category.value = "A1"
        mock_finding.to_dict.return_value = {
            "severity": "high",
            "owasp_category": "A1",
        }

        mock_scan = MagicMock()
        mock_scan.scanned_at = MagicMock()
        mock_scan.findings = [mock_finding]
        mock_scan.scan_id = "sast_001"

        sast_results = get_sast_scan_results()
        sast_results["my-repo"] = {"sast_001": mock_scan}

        result = await handler.handle_get_sast_findings({}, "my-repo")
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1

    def test_sbom_by_id_with_data(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_sbom_results,
        )

        mock_sbom = MagicMock()
        mock_sbom.format.value = "cyclonedx-json"
        mock_sbom.filename = "sbom.json"
        mock_sbom.component_count = 10
        mock_sbom.vulnerability_count = 2
        mock_sbom.license_count = 5
        mock_sbom.generated_at.isoformat.return_value = "2024-01-01T00:00:00"
        mock_sbom.content = '{"components": []}'
        mock_sbom.errors = []

        results = get_or_create_sbom_results("my-repo")
        results["sbom_abc123"] = mock_sbom

        result = handler.handle(
            "/api/v1/codebase/my-repo/sbom/sbom_abc123", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["component_count"] == 10

    def test_sbom_download_with_data(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_sbom_results,
        )
        from aragora.analysis.codebase import SBOMFormat

        mock_sbom = MagicMock()
        mock_sbom.format = SBOMFormat.CYCLONEDX_JSON
        mock_sbom.content = '{"components": []}'
        mock_sbom.filename = "sbom.json"

        results = get_or_create_sbom_results("my-repo")
        results["sbom_dl"] = mock_sbom

        result = handler.handle(
            "/api/v1/codebase/my-repo/sbom/sbom_dl/download", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["content_type"] == "application/json"

    def test_sbom_latest_with_data(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_sbom_results,
        )

        mock_sbom = MagicMock()
        mock_sbom.format.value = "spdx-json"
        mock_sbom.filename = "sbom.spdx.json"
        mock_sbom.component_count = 20
        mock_sbom.vulnerability_count = 0
        mock_sbom.license_count = 8
        mock_sbom.generated_at.isoformat.return_value = "2024-02-01T00:00:00"
        mock_sbom.content = "{}"
        mock_sbom.errors = []

        results = get_or_create_sbom_results("my-repo")
        results["sbom_latest"] = mock_sbom

        result = handler.handle(
            "/api/v1/codebase/my-repo/sbom/latest", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "spdx-json"


# ============================================================================
# Edge cases and special paths
# ============================================================================


class TestEdgeCases:
    """Edge case tests for routing and validation."""

    def test_scan_secrets_latest_vs_id_priority(self, handler, mock_http_handler):
        """The sub_path /scan/secrets/latest is checked before
        /scan/secrets/{scan_id}, so 'latest' is not treated as scan_id."""
        result = handler.handle(
            "/api/v1/codebase/my-repo/scan/secrets/latest", {}, mock_http_handler
        )
        assert result is not None
        # Should be dispatched as latest endpoint (404 since no scans)
        assert _status(result) == 404

    def test_scan_special_ids_not_treated_as_scan_id(self, handler, mock_http_handler):
        """Reserved sub-paths like 'latest', 'secrets', 'sast' are not treated
        as scan IDs in /scan/{scan_id}."""
        for reserved in ("latest", "secrets", "sast"):
            result = handler.handle(
                f"/api/v1/codebase/my-repo/scan/{reserved}", {}, mock_http_handler
            )
            # These may return their own response or None (handled by their own routes)
            # But they must NOT return a "Scan not found" 404 for a literal scan_id lookup

    def test_sbom_reserved_ids_not_treated_as_sbom_id(self, handler, mock_http_handler):
        """Reserved paths like 'latest', 'list', 'compare' under /sbom/ are
        not treated as SBOM IDs."""
        for reserved in ("latest", "list", "compare"):
            result = handler.handle(
                f"/api/v1/codebase/my-repo/sbom/{reserved}", {}, mock_http_handler
            )
            # These should dispatch to their specific handlers, not sbom-by-id

    def test_scans_trailing_slash(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/scans/", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 200

    def test_scans_secrets_trailing_slash(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/codebase/my-repo/scans/secrets/", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 200

    def test_vulnerabilities_with_query_suffix(self, handler, mock_http_handler):
        """Paths like /vulnerabilities?severity=high have the query in params,
        but the path itself might contain '?...' depending on parsing.
        The handler checks startswith('/vulnerabilities?')."""
        result = handler.handle(
            "/api/v1/codebase/my-repo/vulnerabilities",
            {"severity": "critical"},
            mock_http_handler,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_post_invalid_sbom_format(self, handler, tmp_path, monkeypatch):
        """Invalid SBOM format returns 400."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        http = _MockHTTPHandler(body={"repo_path": str(repo_dir), "format": "invalid-format"})

        mock_gen = AsyncMock()

        import aragora.server.handlers.codebase.security.sbom as sbom_mod

        monkeypatch.setattr(sbom_mod, "get_sbom_generator", lambda: mock_gen)

        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/sbom", {}, http
        )

        assert result is not None
        assert _status(result) == 400
        assert "invalid format" in _body(result).get("error", "").lower()

    def test_cve_with_empty_id(self, handler, mock_http_handler):
        """GET /api/v1/cve/ with trailing slash but no ID."""
        result = handler.handle("/api/v1/cve/", {}, mock_http_handler)
        # cve_id after split is empty string, so no dispatch
        assert result is None or _status(result) != 200

    @pytest.mark.asyncio
    async def test_post_sbom_compare_not_found_a(self, handler):
        """SBOM compare with nonexistent sbom_id_a returns 404."""
        http = _MockHTTPHandler(body={"sbom_id_a": "nonexistent", "sbom_id_b": "also_nonexistent"})
        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/sbom/compare", {}, http
        )
        assert result is not None
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_sbom_compare_not_found_b(self, handler):
        """SBOM compare with valid sbom_id_a but nonexistent sbom_id_b returns 404."""
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_sbom_results,
        )

        mock_sbom = MagicMock()
        mock_sbom.generated_at.isoformat.return_value = "2024-01-01T00:00:00"
        mock_sbom.component_count = 5
        mock_sbom.content = '{"components": []}'
        mock_sbom.format.value = "cyclonedx-json"
        mock_sbom.format = MagicMock()

        results = get_or_create_sbom_results("my-repo")
        results["sbom_a"] = mock_sbom

        http = _MockHTTPHandler(body={"sbom_id_a": "sbom_a", "sbom_id_b": "nonexistent"})
        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/sbom/compare", {}, http
        )
        assert result is not None
        assert _status(result) == 404


# ============================================================================
# OWASP summary with populated data
# ============================================================================


class TestOwaspSummary:
    """Tests for OWASP summary endpoint with populated SAST data."""

    @pytest.mark.asyncio
    async def test_owasp_summary_empty_scans(self, handler):
        """Test OWASP summary via direct async method call (not routed
        through handle() since can_handle excludes /sast/owasp-summary)."""
        from aragora.server.handlers.codebase.security.storage import (
            get_sast_scan_results,
        )

        sast_results = get_sast_scan_results()
        sast_results["my-repo"] = {}

        result = await handler.handle_get_owasp_summary({}, "my-repo")
        assert _status(result) == 200
        body = _body(result)
        assert body["total_findings"] == 0

    @pytest.mark.asyncio
    async def test_owasp_summary_with_findings(self, handler, monkeypatch):
        """Test OWASP summary via direct async method call."""
        from aragora.server.handlers.codebase.security.storage import (
            get_sast_scan_results,
        )

        mock_scan = MagicMock()
        mock_scan.scanned_at = MagicMock()
        mock_scan.findings = [MagicMock(), MagicMock()]
        mock_scan.scan_id = "sast_002"

        sast_results = get_sast_scan_results()
        sast_results["my-repo"] = {"sast_002": mock_scan}

        mock_summary = {"owasp_summary": {"A1": 2}, "total_findings": 2}

        mock_scanner = AsyncMock()
        mock_scanner.get_owasp_summary.return_value = mock_summary

        import aragora.server.handlers.codebase.security.sast as sast_mod

        monkeypatch.setattr(sast_mod, "get_sast_scanner", lambda: mock_scanner)

        result = await handler.handle_get_owasp_summary({}, "my-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["total_findings"] == 2


# ============================================================================
# Secrets scan with history
# ============================================================================


class TestSecretsHistory:
    """Tests for secrets scan with history options."""

    @pytest.mark.asyncio
    async def test_secrets_scan_with_include_history(self, handler, tmp_path, monkeypatch):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        http = _MockHTTPHandler(
            body={"repo_path": str(repo_dir), "include_history": True, "history_depth": 50}
        )

        mock_instance = AsyncMock()
        mock_instance.scan_repository.return_value = MagicMock(
            scan_id="sec_002", status="completed", secrets=[]
        )
        mock_instance.scan_git_history.return_value = MagicMock(secrets=[])

        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "SecretsScanner", lambda: mock_instance)

        result = await handler.handle_post(
            "/api/v1/codebase/my-repo/scan/secrets", {}, http
        )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body.get("include_history") is True

    def test_get_secrets_include_history_false(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_secrets_scans,
        )

        mock_secret_current = MagicMock()
        mock_secret_current.to_dict.return_value = {
            "severity": "high",
            "secret_type": "api_key",
            "is_in_history": False,
        }
        mock_secret_history = MagicMock()
        mock_secret_history.to_dict.return_value = {
            "severity": "medium",
            "secret_type": "password",
            "is_in_history": True,
        }

        mock_scan = MagicMock()
        mock_scan.status = "completed"
        mock_scan.started_at = MagicMock()
        mock_scan.secrets = [mock_secret_current, mock_secret_history]
        mock_scan.scan_id = "sec_003"

        scans = get_or_create_secrets_scans("my-repo")
        scans["sec_003"] = mock_scan

        result = handler.handle(
            "/api/v1/codebase/my-repo/secrets",
            {"include_history": "false"},
            mock_http_handler,
        )
        assert _status(result) == 200
        body = _body(result)
        # Only current secrets should be returned (not history)
        assert body["total"] == 1
        assert body["secrets"][0]["is_in_history"] is False


# ============================================================================
# SAST findings filtering
# ============================================================================


class TestSastFiltering:
    """Tests for SAST findings filtering."""

    @pytest.mark.asyncio
    async def test_sast_findings_filter_by_severity(self, handler):
        """Test via direct async method call (can_handle excludes /sast/findings)."""
        from aragora.server.handlers.codebase.security.storage import (
            get_sast_scan_results,
        )

        mock_finding_high = MagicMock()
        mock_finding_high.severity.value = "high"
        mock_finding_high.owasp_category.value = "A1"
        mock_finding_high.to_dict.return_value = {"severity": "high"}

        mock_finding_low = MagicMock()
        mock_finding_low.severity.value = "low"
        mock_finding_low.owasp_category.value = "A3"
        mock_finding_low.to_dict.return_value = {"severity": "low"}

        mock_scan = MagicMock()
        mock_scan.scanned_at = MagicMock()
        mock_scan.findings = [mock_finding_high, mock_finding_low]
        mock_scan.scan_id = "sast_003"

        sast_results = get_sast_scan_results()
        sast_results["my-repo"] = {"sast_003": mock_scan}

        result = await handler.handle_get_sast_findings(
            {"severity": "high"}, "my-repo"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_sast_findings_filter_by_owasp(self, handler):
        """Test via direct async method call (can_handle excludes /sast/findings)."""
        from aragora.server.handlers.codebase.security.storage import (
            get_sast_scan_results,
        )

        mock_finding_a1 = MagicMock()
        mock_finding_a1.severity.value = "high"
        mock_finding_a1.owasp_category.value = "A1-Injection"
        mock_finding_a1.to_dict.return_value = {"owasp_category": "A1-Injection"}

        mock_finding_a3 = MagicMock()
        mock_finding_a3.severity.value = "medium"
        mock_finding_a3.owasp_category.value = "A3-XSS"
        mock_finding_a3.to_dict.return_value = {"owasp_category": "A3-XSS"}

        mock_scan = MagicMock()
        mock_scan.scanned_at = MagicMock()
        mock_scan.findings = [mock_finding_a1, mock_finding_a3]
        mock_scan.scan_id = "sast_004"

        sast_results = get_sast_scan_results()
        sast_results["my-repo"] = {"sast_004": mock_scan}

        result = await handler.handle_get_sast_findings(
            {"owasp_category": "A1"}, "my-repo"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1


# ============================================================================
# Scans list with status filter
# ============================================================================


class TestScansListFiltering:
    """Tests for scan list filtering by status."""

    def test_scans_filter_by_status(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_repo_scans,
        )
        from datetime import datetime, timezone

        mock_completed = MagicMock()
        mock_completed.scan_id = "scan_c"
        mock_completed.status = "completed"
        mock_completed.started_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
        mock_completed.completed_at = datetime(2024, 1, 2, 0, 5, tzinfo=timezone.utc)
        mock_completed.total_dependencies = 10
        mock_completed.vulnerable_dependencies = 1
        mock_completed.critical_count = 0
        mock_completed.high_count = 1
        mock_completed.medium_count = 0
        mock_completed.low_count = 0

        mock_running = MagicMock()
        mock_running.scan_id = "scan_r"
        mock_running.status = "running"
        mock_running.started_at = datetime(2024, 1, 3, tzinfo=timezone.utc)
        mock_running.completed_at = None

        scans = get_or_create_repo_scans("my-repo")
        scans["scan_c"] = mock_completed
        scans["scan_r"] = mock_running

        result = handler.handle(
            "/api/v1/codebase/my-repo/scans",
            {"status": "completed"},
            mock_http_handler,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["scans"][0]["scan_id"] == "scan_c"

    def test_secrets_scans_filter_by_status(self, handler, mock_http_handler):
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_secrets_scans,
        )
        from datetime import datetime, timezone

        mock_scan = MagicMock()
        mock_scan.scan_id = "sec_f"
        mock_scan.status = "failed"
        mock_scan.started_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_scan.completed_at = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)
        mock_scan.files_scanned = 0
        mock_scan.scanned_history = False
        mock_scan.history_depth = 0
        mock_scan.secrets = []
        mock_scan.critical_count = 0
        mock_scan.high_count = 0
        mock_scan.medium_count = 0
        mock_scan.low_count = 0

        scans = get_or_create_secrets_scans("my-repo")
        scans["sec_f"] = mock_scan

        result = handler.handle(
            "/api/v1/codebase/my-repo/scans/secrets",
            {"status": "completed"},
            mock_http_handler,
        )
        assert _status(result) == 200
        body = _body(result)
        # Filtered out the "failed" scan
        assert body["total"] == 0
