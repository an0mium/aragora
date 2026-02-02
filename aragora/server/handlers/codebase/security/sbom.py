"""
HTTP API handlers for SBOM (Software Bill of Materials) generation.

Provides handlers for:
- Generate SBOMs in multiple formats (CycloneDX, SPDX)
- Get and list SBOMs
- Download SBOM content
- Compare SBOMs to find differences
"""

from __future__ import annotations

import json
import logging
import uuid

from aragora.analysis.codebase import SBOMFormat
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    require_permission,
    success_response,
)

from .storage import (
    get_or_create_sbom_results,
    get_sbom_generator,
    get_sbom_lock,
)

logger = logging.getLogger(__name__)


@require_permission("security:sbom:generate")
async def handle_generate_sbom(
    repo_path: str,
    repo_id: str | None = None,
    format: str = "cyclonedx-json",
    project_name: str | None = None,
    project_version: str | None = None,
    include_dev: bool = True,
    include_vulnerabilities: bool = True,
    branch: str | None = None,
    commit_sha: str | None = None,
    workspace_id: str | None = None,
) -> HandlerResult:
    """
    Generate SBOM for a repository.

    POST /api/v1/codebase/{repo}/sbom
    {
        "repo_path": "/path/to/repo",
        "format": "cyclonedx-json",  // cyclonedx-json, cyclonedx-xml, spdx-json, spdx-tv
        "project_name": "MyProject",
        "project_version": "1.0.0",
        "include_dev": true,
        "include_vulnerabilities": true
    }

    Returns:
        SBOM generation result with content and metadata
    """
    try:
        repo_id = repo_id or f"repo_{uuid.uuid4().hex[:12]}"
        sbom_id = f"sbom_{uuid.uuid4().hex[:12]}"

        sbom_lock = get_sbom_lock()

        # Parse format
        try:
            sbom_format = SBOMFormat(format)
        except ValueError:
            return error_response(
                f"Invalid format: {format}. Valid formats: "
                "cyclonedx-json, cyclonedx-xml, spdx-json, spdx-tv",
                400,
            )

        # Run generation
        generator = get_sbom_generator()
        generator.include_dev_dependencies = include_dev
        generator.include_vulnerabilities = include_vulnerabilities

        result = await generator.generate_from_repo(
            repo_path=repo_path,
            format=sbom_format,
            project_name=project_name,
            project_version=project_version,
            branch=branch,
            commit_sha=commit_sha,
        )

        # Store result
        repo_results = get_or_create_sbom_results(repo_id)
        with sbom_lock:
            repo_results[sbom_id] = result

        logger.info(
            f"[SBOM] Generated {format} for {repo_id}: "
            f"{result.component_count} components, {result.vulnerability_count} vulnerabilities"
        )

        return success_response(
            {
                "sbom_id": sbom_id,
                "repository": repo_id,
                "format": result.format.value,
                "filename": result.filename,
                "component_count": result.component_count,
                "vulnerability_count": result.vulnerability_count,
                "license_count": result.license_count,
                "generated_at": result.generated_at.isoformat(),
                "content": result.content,
                "errors": result.errors,
            }
        )

    except (OSError, ValueError, TypeError, RuntimeError) as e:
        logger.exception(f"Failed to generate SBOM: {e}")
        return error_response(str(e), 500)


@require_permission("security:sbom:read")
async def handle_get_sbom(
    repo_id: str,
    sbom_id: str | None = None,
) -> HandlerResult:
    """
    Get SBOM content.

    GET /api/v1/codebase/{repo}/sbom/latest
    GET /api/v1/codebase/{repo}/sbom/{sbom_id}
    """
    try:
        repo_results = get_or_create_sbom_results(repo_id)

        if sbom_id:
            # Get specific SBOM
            result = repo_results.get(sbom_id)
            if not result:
                return error_response(f"SBOM not found: {sbom_id}", 404)
        else:
            # Get latest SBOM
            if not repo_results:
                return error_response("No SBOMs generated for this repository", 404)
            result = max(repo_results.values(), key=lambda r: r.generated_at)

        return success_response(
            {
                "sbom_id": sbom_id or "sbom_latest",
                "repository": repo_id,
                "format": result.format.value,
                "filename": result.filename,
                "component_count": result.component_count,
                "vulnerability_count": result.vulnerability_count,
                "license_count": result.license_count,
                "generated_at": result.generated_at.isoformat(),
                "content": result.content,
                "errors": result.errors,
            }
        )

    except (KeyError, ValueError, TypeError) as e:
        logger.exception(f"Failed to get SBOM: {e}")
        return error_response(str(e), 500)


@require_permission("security:sbom:read")
async def handle_list_sboms(
    repo_id: str,
    limit: int = 10,
) -> HandlerResult:
    """
    List SBOMs for a repository.

    GET /api/v1/codebase/{repo}/sbom/list
    """
    try:
        repo_results = get_or_create_sbom_results(repo_id)

        # Sort by generated_at descending
        sorted_results = sorted(
            repo_results.items(),
            key=lambda x: x[1].generated_at,
            reverse=True,
        )[:limit]

        sboms = [
            {
                "sbom_id": sbom_id,
                "format": result.format.value,
                "filename": result.filename,
                "component_count": result.component_count,
                "vulnerability_count": result.vulnerability_count,
                "license_count": result.license_count,
                "generated_at": result.generated_at.isoformat(),
            }
            for sbom_id, result in sorted_results
        ]

        return success_response(
            {
                "repository": repo_id,
                "count": len(sboms),
                "sboms": sboms,
            }
        )

    except (KeyError, ValueError, TypeError) as e:
        logger.exception(f"Failed to list SBOMs: {e}")
        return error_response(str(e), 500)


@require_permission("security:sbom:read")
async def handle_download_sbom(
    repo_id: str,
    sbom_id: str,
) -> HandlerResult:
    """
    Download SBOM content (raw).

    GET /api/v1/codebase/{repo}/sbom/{sbom_id}/download

    Returns the raw SBOM content with appropriate content-type header info.
    """
    try:
        repo_results = get_or_create_sbom_results(repo_id)
        result = repo_results.get(sbom_id)

        if not result:
            return error_response(f"SBOM not found: {sbom_id}", 404)

        # Determine content type
        content_types = {
            SBOMFormat.CYCLONEDX_JSON: "application/json",
            SBOMFormat.CYCLONEDX_XML: "application/xml",
            SBOMFormat.SPDX_JSON: "application/json",
            SBOMFormat.SPDX_TV: "text/plain",
        }
        content_type = content_types.get(result.format, "application/octet-stream")

        return success_response(
            {
                "content": result.content,
                "filename": result.filename,
                "content_type": content_type,
            }
        )

    except (KeyError, ValueError, TypeError) as e:
        logger.exception(f"Failed to download SBOM: {e}")
        return error_response(str(e), 500)


@require_permission("security:sbom:read")
async def handle_compare_sboms(
    repo_id: str,
    sbom_id_a: str,
    sbom_id_b: str,
) -> HandlerResult:
    """
    Compare two SBOMs to find differences.

    POST /api/v1/codebase/{repo}/sbom/compare
    {
        "sbom_id_a": "sbom_abc123",
        "sbom_id_b": "sbom_def456"
    }
    """
    try:
        repo_results = get_or_create_sbom_results(repo_id)

        result_a = repo_results.get(sbom_id_a)
        result_b = repo_results.get(sbom_id_b)

        if not result_a:
            return error_response(f"SBOM not found: {sbom_id_a}", 404)
        if not result_b:
            return error_response(f"SBOM not found: {sbom_id_b}", 404)

        # Parse components from both (simplified - works for JSON formats)
        def extract_components(content: str, format: SBOMFormat) -> dict[str, str]:
            """Extract component name -> version mapping."""
            components = {}
            try:
                if format in (SBOMFormat.CYCLONEDX_JSON, SBOMFormat.SPDX_JSON):
                    data = json.loads(content)
                    if format == SBOMFormat.CYCLONEDX_JSON:
                        for comp in data.get("components", []):
                            name = comp.get("name", "")
                            if comp.get("group"):
                                name = f"{comp['group']}/{name}"
                            components[name] = comp.get("version", "")
                    else:  # SPDX
                        for pkg in data.get("packages", []):
                            components[pkg.get("name", "")] = pkg.get("versionInfo", "")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug("Failed to parse SBOM components: %s", e)
            return components

        components_a = extract_components(result_a.content, result_a.format)
        components_b = extract_components(result_b.content, result_b.format)

        all_names = set(components_a.keys()) | set(components_b.keys())

        added = []
        removed = []
        updated = []
        unchanged = []

        for name in sorted(all_names):
            v_a = components_a.get(name)
            v_b = components_b.get(name)

            if v_a and not v_b:
                removed.append({"name": name, "version": v_a})
            elif v_b and not v_a:
                added.append({"name": name, "version": v_b})
            elif v_a != v_b:
                updated.append({"name": name, "old_version": v_a, "new_version": v_b})
            else:
                unchanged.append({"name": name, "version": v_a})

        return success_response(
            {
                "sbom_a": {
                    "sbom_id": sbom_id_a,
                    "generated_at": result_a.generated_at.isoformat(),
                    "component_count": result_a.component_count,
                },
                "sbom_b": {
                    "sbom_id": sbom_id_b,
                    "generated_at": result_b.generated_at.isoformat(),
                    "component_count": result_b.component_count,
                },
                "diff": {
                    "added": added,
                    "removed": removed,
                    "updated": updated,
                    "unchanged_count": len(unchanged),
                },
                "summary": {
                    "total_added": len(added),
                    "total_removed": len(removed),
                    "total_updated": len(updated),
                    "total_unchanged": len(unchanged),
                },
            }
        )

    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        logger.exception(f"Failed to compare SBOMs: {e}")
        return error_response(str(e), 500)
