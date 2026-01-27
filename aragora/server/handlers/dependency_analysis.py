"""
HTTP API Handlers for Dependency Analysis.

Provides REST APIs for codebase dependency analysis:
- Dependency tree resolution
- Vulnerability (CVE) scanning
- SBOM generation (CycloneDX, SPDX)
- License compatibility checking

Endpoints:
- POST /api/v1/codebase/analyze-dependencies - Analyze project dependencies
- GET /api/v1/codebase/sbom - Generate SBOM
- POST /api/v1/codebase/scan-vulnerabilities - Scan for CVEs
- POST /api/v1/codebase/check-licenses - Check license compatibility
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from aragora.rbac.decorators import require_permission
from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

# Thread-safe service instance
_dependency_analyzer: Optional[Any] = None
_dependency_analyzer_lock = threading.Lock()

# Cache for analysis results
_analysis_cache: dict[str, dict[str, Any]] = {}
_analysis_cache_lock = threading.Lock()


def get_dependency_analyzer():
    """Get or create dependency analyzer (thread-safe)."""
    global _dependency_analyzer
    if _dependency_analyzer is not None:
        return _dependency_analyzer

    with _dependency_analyzer_lock:
        if _dependency_analyzer is None:
            from aragora.audit.dependency_analyzer import DependencyAnalyzer

            _dependency_analyzer = DependencyAnalyzer()
        return _dependency_analyzer


@require_permission("codebase:analyze")
async def handle_analyze_dependencies(
    context: AuthorizationContext,
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Analyze project dependencies.

    POST /api/v1/codebase/analyze-dependencies
    Body: {
        repo_path: str,
        include_dev: bool (default true),
        use_cache: bool (default true)
    }
    """
    try:
        analyzer = get_dependency_analyzer()

        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path is required", status=400)

        repo_path = Path(repo_path)
        if not repo_path.exists():
            return error_response(f"Path does not exist: {repo_path}", status=404)

        include_dev = data.get("include_dev", True)
        use_cache = data.get("use_cache", True)

        # Check cache
        cache_key = f"{repo_path}:{include_dev}"
        if use_cache:
            with _analysis_cache_lock:
                if cache_key in _analysis_cache:
                    cached = _analysis_cache[cache_key]
                    return success_response({**cached, "from_cache": True})

        # Analyze dependencies
        tree = await analyzer.resolve_dependencies(
            repo_path=repo_path,
            include_dev=include_dev,
        )

        result = {
            "project_name": tree.project_name,
            "project_version": tree.project_version,
            "package_managers": [pm.value for pm in tree.package_managers],
            "total_dependencies": len(tree.dependencies),
            "direct_dependencies": tree.total_direct,
            "transitive_dependencies": tree.total_transitive,
            "dev_dependencies": tree.total_dev,
            "dependencies": [
                {
                    "name": dep.name,
                    "version": dep.version,
                    "type": dep.dependency_type.value,
                    "package_manager": dep.package_manager.value,
                    "license": dep.license,
                    "purl": dep.purl,
                }
                for dep in list(tree.dependencies.values())[:100]  # Limit for response size
            ],
            "analyzed_at": tree.analyzed_at.isoformat(),
        }

        # Cache result
        with _analysis_cache_lock:
            _analysis_cache[cache_key] = result

        return success_response({**result, "from_cache": False})

    except Exception as e:
        logger.exception("Error analyzing dependencies")
        return error_response(f"Failed to analyze dependencies: {e}", status=500)


@require_permission("codebase:analyze")
async def handle_generate_sbom(
    context: AuthorizationContext,
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Generate Software Bill of Materials (SBOM).

    POST /api/v1/codebase/sbom
    Body: {
        repo_path: str,
        format: str (cyclonedx, spdx) - default cyclonedx,
        include_vulnerabilities: bool (default true)
    }
    """
    try:
        analyzer = get_dependency_analyzer()

        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path is required", status=400)

        repo_path = Path(repo_path)
        if not repo_path.exists():
            return error_response(f"Path does not exist: {repo_path}", status=404)

        sbom_format = data.get("format", "cyclonedx")
        if sbom_format not in ("cyclonedx", "spdx"):
            return error_response("format must be 'cyclonedx' or 'spdx'", status=400)

        include_vulns = data.get("include_vulnerabilities", True)

        # Resolve dependencies first
        tree = await analyzer.resolve_dependencies(repo_path)

        # Generate SBOM
        sbom_json = await analyzer.generate_sbom(
            tree=tree,
            format=sbom_format,
            include_vulnerabilities=include_vulns,
        )

        # Parse to return as structured data
        sbom_data = json.loads(sbom_json)

        return success_response(
            {
                "format": sbom_format,
                "project_name": tree.project_name,
                "component_count": len(tree.dependencies),
                "sbom": sbom_data,
                "sbom_json": sbom_json,  # Raw JSON string for download
            }
        )

    except Exception as e:
        logger.exception("Error generating SBOM")
        return error_response(f"Failed to generate SBOM: {e}", status=500)


@require_permission("codebase:analyze")
async def handle_scan_vulnerabilities(
    context: AuthorizationContext,
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Scan dependencies for known vulnerabilities.

    POST /api/v1/codebase/scan-vulnerabilities
    Body: {
        repo_path: str
    }
    """
    try:
        analyzer = get_dependency_analyzer()

        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path is required", status=400)

        repo_path = Path(repo_path)
        if not repo_path.exists():
            return error_response(f"Path does not exist: {repo_path}", status=404)

        # Resolve dependencies
        tree = await analyzer.resolve_dependencies(repo_path)

        # Scan for vulnerabilities
        vulnerabilities = await analyzer.check_vulnerabilities(tree)

        # Group by severity
        by_severity: dict[str, list[dict[str, Any]]] = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "unknown": [],
        }

        for vuln in vulnerabilities:
            severity = vuln.severity.value
            by_severity.get(severity, by_severity["unknown"]).append(
                {
                    "id": vuln.id,
                    "title": vuln.title,
                    "description": vuln.description,
                    "affected_package": vuln.affected_package,
                    "affected_versions": vuln.affected_versions,
                    "fixed_version": vuln.fixed_version,
                    "cvss_score": vuln.cvss_score,
                    "cwe_id": vuln.cwe_id,
                    "references": vuln.references,
                }
            )

        return success_response(
            {
                "project_name": tree.project_name,
                "total_vulnerabilities": len(vulnerabilities),
                "critical_count": len(by_severity["critical"]),
                "high_count": len(by_severity["high"]),
                "medium_count": len(by_severity["medium"]),
                "low_count": len(by_severity["low"]),
                "vulnerabilities_by_severity": by_severity,
                "scan_summary": {
                    "packages_scanned": len(tree.dependencies),
                    "packages_with_vulnerabilities": len(
                        set(v.affected_package for v in vulnerabilities)
                    ),
                },
            }
        )

    except Exception as e:
        logger.exception("Error scanning vulnerabilities")
        return error_response(f"Failed to scan vulnerabilities: {e}", status=500)


@require_permission("codebase:analyze")
async def handle_check_licenses(
    context: AuthorizationContext,
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Check license compatibility of dependencies.

    POST /api/v1/codebase/check-licenses
    Body: {
        repo_path: str,
        project_license: str (default MIT)
    }
    """
    try:
        analyzer = get_dependency_analyzer()

        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path is required", status=400)

        repo_path = Path(repo_path)
        if not repo_path.exists():
            return error_response(f"Path does not exist: {repo_path}", status=404)

        project_license = data.get("project_license", "MIT")

        # Resolve dependencies
        tree = await analyzer.resolve_dependencies(repo_path)

        # Check license compatibility
        conflicts = await analyzer.check_license_compatibility(
            tree=tree,
            project_license=project_license,
        )

        # Group by severity
        errors = [c for c in conflicts if c.severity == "error"]
        warnings = [c for c in conflicts if c.severity == "warning"]
        _info = [c for c in conflicts if c.severity == "info"]  # noqa: F841

        # Get license distribution
        license_counts: dict[str, int] = {}
        for dep in tree.dependencies.values():
            license_name = dep.license or "Unknown"
            license_counts[license_name] = license_counts.get(license_name, 0) + 1

        return success_response(
            {
                "project_name": tree.project_name,
                "project_license": project_license,
                "total_conflicts": len(conflicts),
                "error_count": len(errors),
                "warning_count": len(warnings),
                "conflicts": [
                    {
                        "package": c.package_b,
                        "license": c.license_b,
                        "conflict_type": c.conflict_type,
                        "severity": c.severity,
                        "description": c.description,
                    }
                    for c in conflicts
                ],
                "license_distribution": license_counts,
                "compatible": len(errors) == 0,
            }
        )

    except Exception as e:
        logger.exception("Error checking licenses")
        return error_response(f"Failed to check licenses: {e}", status=500)


async def handle_clear_cache(
    user_id: str = "default",
) -> HandlerResult:
    """
    Clear the dependency analysis cache.

    POST /api/v1/codebase/clear-cache
    """
    try:
        with _analysis_cache_lock:
            count = len(_analysis_cache)
            _analysis_cache.clear()

        return success_response(
            {
                "cleared": True,
                "entries_removed": count,
            }
        )

    except Exception as e:
        logger.exception("Error clearing cache")
        return error_response(f"Failed to clear cache: {e}", status=500)


# =============================================================================
# Handler Registration
# =============================================================================


def get_dependency_analysis_routes() -> list[tuple[str, str, Any]]:
    """
    Get route definitions for dependency analysis handlers.

    Returns list of (method, path, handler) tuples.
    """
    return [
        ("POST", "/api/v1/codebase/analyze-dependencies", handle_analyze_dependencies),
        ("POST", "/api/v1/codebase/sbom", handle_generate_sbom),
        ("POST", "/api/v1/codebase/scan-vulnerabilities", handle_scan_vulnerabilities),
        ("POST", "/api/v1/codebase/check-licenses", handle_check_licenses),
        ("POST", "/api/v1/codebase/clear-cache", handle_clear_cache),
    ]


class DependencyAnalysisHandler(BaseHandler):
    """
    HTTP handler for codebase dependency analysis endpoints.

    Provides dependency tree resolution, SBOM generation, CVE scanning,
    and license compatibility checking.
    Integrates with the Aragora server routing system.
    """

    ROUTES = [
        "/api/v1/codebase/analyze-dependencies",
        "/api/v1/codebase/sbom",
        "/api/v1/codebase/scan-vulnerabilities",
        "/api/v1/codebase/check-licenses",
        "/api/v1/codebase/clear-cache",
    ]

    def __init__(self, ctx: dict[str, Any]):
        """Initialize with server context."""
        super().__init__(ctx)  # type: ignore[arg-type]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route dependency analysis endpoint requests."""
        return None

    async def handle_post(self, path: str, data: dict[str, Any]) -> HandlerResult:  # type: ignore[override]
        """Handle POST requests."""
        if path == "/api/v1/codebase/analyze-dependencies":
            return await handle_analyze_dependencies(data)
        elif path == "/api/v1/codebase/sbom":
            return await handle_generate_sbom(data)
        elif path == "/api/v1/codebase/scan-vulnerabilities":
            return await handle_scan_vulnerabilities(data)
        elif path == "/api/v1/codebase/check-licenses":
            return await handle_check_licenses(data)
        elif path == "/api/v1/codebase/clear-cache":
            return await handle_clear_cache()
        return error_response("Not found", status=404)
