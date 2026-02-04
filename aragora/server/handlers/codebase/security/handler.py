"""
Main SecurityHandler class for HTTP routing.

This module provides the SecurityHandler class that orchestrates
all security-related endpoints by delegating to submodule handlers.
"""

from __future__ import annotations

from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
)
from aragora.server.handlers.utils.decorators import require_permission
from aragora.server.validation.query_params import safe_query_int

from .sast import (
    handle_get_owasp_summary,
    handle_get_sast_findings,
    handle_get_sast_scan_status,
    handle_scan_sast,
)
from .sbom import (
    handle_compare_sboms,
    handle_download_sbom,
    handle_generate_sbom,
    handle_get_sbom,
    handle_list_sboms,
)
from .secrets import (
    handle_get_secrets,
    handle_get_secrets_scan_status,
    handle_list_secrets_scans,
    handle_scan_secrets,
)
from .storage import safe_repo_id
from .vulnerability import (
    handle_get_cve_details,
    handle_get_scan_status,
    handle_get_vulnerabilities,
    handle_list_scans,
    handle_scan_repository,
)


class SecurityHandler(BaseHandler):
    """
    HTTP handler for codebase security endpoints.

    Integrates with the Aragora server routing system.
    """

    ROUTES = [
        "/api/v1/cve",
    ]

    ROUTE_PREFIXES = [
        "/api/v1/codebase/",
        "/api/v1/cve/",
    ]

    def __init__(self, ctx: dict[str, Any]) -> None:
        """Initialize with server context."""
        super().__init__(ctx)

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        if path in self.ROUTES:
            return True
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                # Check for security-related paths
                if (
                    "/scan" in path
                    or "/vulnerabilities" in path
                    or "/cve/" in path
                    or "/secrets" in path
                    or "/sbom" in path
                ):
                    return True
        return False

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route security endpoint requests."""
        return None

    # =========================================================================
    # Path Traversal Validation Helper
    # =========================================================================

    def _validate_repo_id(self, repo_id: str) -> HandlerResult | None:
        """
        Validate repo_id to prevent path traversal attacks.

        Returns None if valid, or an error response if invalid.
        """
        is_valid, err_msg = safe_repo_id(repo_id)
        if not is_valid:
            return error_response(err_msg or "Invalid repo ID", 400)
        return None

    # =========================================================================
    # Vulnerability Scan Endpoints
    # =========================================================================

    @require_permission("debates:write")
    async def handle_post_scan(self, data: dict[str, Any], repo_id: str) -> HandlerResult:
        """POST /api/v1/codebase/{repo}/scan"""
        if err := self._validate_repo_id(repo_id):
            return err

        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path required", 400)

        return await handle_scan_repository(
            repo_path=repo_path,
            repo_id=repo_id,
            branch=data.get("branch"),
            commit_sha=data.get("commit_sha"),
            workspace_id=data.get("workspace_id"),
            user_id=self._get_user_id(),
        )

    @require_permission("debates:read")
    async def handle_get_scan_latest(self, params: dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/latest"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_get_scan_status(repo_id=repo_id)

    @require_permission("debates:read")
    async def handle_get_scan(
        self, params: dict[str, Any], repo_id: str, scan_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/{scan_id}"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_get_scan_status(repo_id=repo_id, scan_id=scan_id)

    @require_permission("debates:read")
    async def handle_get_vulnerabilities(
        self, params: dict[str, Any], repo_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/vulnerabilities"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_get_vulnerabilities(
            repo_id=repo_id,
            severity=params.get("severity"),
            package=params.get("package"),
            ecosystem=params.get("ecosystem"),
            limit=safe_query_int(params, "limit", default=100, min_val=1, max_val=1000),
            offset=safe_query_int(params, "offset", default=0, min_val=0, max_val=100000),
        )

    @require_permission("debates:read")
    async def handle_get_cve(self, params: dict[str, Any], cve_id: str) -> HandlerResult:
        """GET /api/v1/cve/{cve_id}"""
        return await handle_get_cve_details(cve_id=cve_id)

    @require_permission("debates:read")
    async def handle_list_scans(self, params: dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scans"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_list_scans(
            repo_id=repo_id,
            status=params.get("status"),
            limit=safe_query_int(params, "limit", default=20, min_val=1, max_val=1000),
            offset=safe_query_int(params, "offset", default=0, min_val=0, max_val=100000),
        )

    # =========================================================================
    # Secrets Scan Endpoints
    # =========================================================================

    @require_permission("debates:write")
    async def handle_post_secrets_scan(self, data: dict[str, Any], repo_id: str) -> HandlerResult:
        """POST /api/v1/codebase/{repo}/scan/secrets"""
        if err := self._validate_repo_id(repo_id):
            return err

        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path required", 400)

        return await handle_scan_secrets(
            repo_path=repo_path,
            repo_id=repo_id,
            branch=data.get("branch"),
            include_history=data.get("include_history", False),
            history_depth=safe_query_int(
                data, "history_depth", default=100, min_val=1, max_val=10000
            ),
            workspace_id=data.get("workspace_id"),
            user_id=self._get_user_id(),
        )

    @require_permission("debates:read")
    async def handle_get_secrets_scan_latest(
        self, params: dict[str, Any], repo_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/secrets/latest"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_get_secrets_scan_status(repo_id=repo_id)

    @require_permission("debates:read")
    async def handle_get_secrets_scan(
        self, params: dict[str, Any], repo_id: str, scan_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/secrets/{scan_id}"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_get_secrets_scan_status(repo_id=repo_id, scan_id=scan_id)

    @require_permission("debates:read")
    async def handle_get_secrets(self, params: dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/secrets"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_get_secrets(
            repo_id=repo_id,
            severity=params.get("severity"),
            secret_type=params.get("secret_type"),
            include_history=params.get("include_history", "true").lower() == "true",
            limit=safe_query_int(params, "limit", default=100, min_val=1, max_val=1000),
            offset=safe_query_int(params, "offset", default=0, min_val=0, max_val=100000),
        )

    @require_permission("debates:read")
    async def handle_list_secrets_scans(
        self, params: dict[str, Any], repo_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scans/secrets"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_list_secrets_scans(
            repo_id=repo_id,
            status=params.get("status"),
            limit=safe_query_int(params, "limit", default=20, min_val=1, max_val=1000),
            offset=safe_query_int(params, "offset", default=0, min_val=0, max_val=100000),
        )

    # =========================================================================
    # SAST Scan Endpoints
    # =========================================================================

    @require_permission("debates:read")
    async def handle_scan_sast(self, params: dict[str, Any], repo_id: str) -> HandlerResult:
        """POST /api/v1/codebase/{repo}/scan/sast"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_scan_sast(
            repo_path=params.get("repo_path", ""),
            repo_id=repo_id,
            rule_sets=params.get("rule_sets"),
            workspace_id=params.get("workspace_id"),
        )

    @require_permission("debates:read")
    async def handle_get_sast_scan_status(
        self, params: dict[str, Any], repo_id: str, scan_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/sast/{scan_id}"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_get_sast_scan_status(repo_id=repo_id, scan_id=scan_id)

    @require_permission("debates:read")
    async def handle_get_sast_findings(self, params: dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sast/findings"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_get_sast_findings(
            repo_id=repo_id,
            severity=params.get("severity"),
            owasp_category=params.get("owasp_category"),
            limit=safe_query_int(params, "limit", default=100, min_val=1, max_val=1000),
            offset=safe_query_int(params, "offset", default=0, min_val=0, max_val=100000),
        )

    @require_permission("debates:read")
    async def handle_get_owasp_summary(self, params: dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sast/owasp-summary"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_get_owasp_summary(repo_id=repo_id)

    # =========================================================================
    # SBOM Endpoints
    # =========================================================================

    @require_permission("debates:write")
    async def handle_post_sbom(self, data: dict[str, Any], repo_id: str) -> HandlerResult:
        """POST /api/v1/codebase/{repo}/sbom - Generate SBOM"""
        if err := self._validate_repo_id(repo_id):
            return err

        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path required", 400)

        return await handle_generate_sbom(
            repo_path=repo_path,
            repo_id=repo_id,
            format=data.get("format", "cyclonedx-json"),
            workspace_id=data.get("workspace_id"),
        )

    @require_permission("debates:read")
    async def handle_get_sbom_latest(self, params: dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sbom/latest"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_get_sbom(repo_id=repo_id)

    @require_permission("debates:read")
    async def handle_get_sbom_by_id(
        self, params: dict[str, Any], repo_id: str, sbom_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sbom/{sbom_id}"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_get_sbom(repo_id=repo_id, sbom_id=sbom_id)

    @require_permission("debates:read")
    async def handle_list_sbom(self, params: dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sbom/list"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_list_sboms(repo_id=repo_id)

    @require_permission("debates:export")
    async def handle_download_sbom_content(
        self, params: dict[str, Any], repo_id: str, sbom_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sbom/{sbom_id}/download"""
        if err := self._validate_repo_id(repo_id):
            return err

        return await handle_download_sbom(repo_id=repo_id, sbom_id=sbom_id)

    @require_permission("debates:read")
    async def handle_compare_sbom(self, data: dict[str, Any], repo_id: str) -> HandlerResult:
        """POST /api/v1/codebase/{repo}/sbom/compare"""
        if err := self._validate_repo_id(repo_id):
            return err

        sbom_id_a = data.get("sbom_id_a")
        sbom_id_b = data.get("sbom_id_b")

        if not sbom_id_a or not sbom_id_b:
            return error_response("sbom_id_a and sbom_id_b required", 400)

        return await handle_compare_sboms(
            repo_id=repo_id,
            sbom_id_a=sbom_id_a,
            sbom_id_b=sbom_id_b,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"
