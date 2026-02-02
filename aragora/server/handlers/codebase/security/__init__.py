"""
HTTP API Handlers for Codebase Security Analysis.

Provides REST APIs for security vulnerability scanning:
- Trigger dependency vulnerability scans
- Query CVE databases
- Get scan results and history
- View vulnerability details
- Secrets detection
- SAST (Static Application Security Testing)
- SBOM (Software Bill of Materials) generation

Endpoints:
- POST /api/v1/codebase/{repo}/scan - Trigger security scan
- GET /api/v1/codebase/{repo}/scan/latest - Get latest scan result
- GET /api/v1/codebase/{repo}/scan/{scan_id} - Get specific scan result
- GET /api/v1/codebase/{repo}/vulnerabilities - List all vulnerabilities
- GET /api/v1/cve/{cve_id} - Get CVE details
- POST /api/v1/codebase/{repo}/scan/secrets - Trigger secrets scan
- GET /api/v1/codebase/{repo}/secrets - List detected secrets
- POST /api/v1/codebase/{repo}/scan/sast - Trigger SAST scan
- GET /api/v1/codebase/{repo}/sast/findings - List SAST findings
- POST /api/v1/codebase/{repo}/sbom - Generate SBOM
- GET /api/v1/codebase/{repo}/sbom/latest - Get latest SBOM
"""

from __future__ import annotations

# Main handler class
from .handler import SecurityHandler

# Vulnerability scanning handlers
from .vulnerability import (
    handle_get_cve_details,
    handle_get_scan_status,
    handle_get_vulnerabilities,
    handle_list_scans,
    handle_query_package_vulnerabilities,
    handle_scan_repository,
)

# Secrets scanning handlers
from .secrets import (
    handle_get_secrets,
    handle_get_secrets_scan_status,
    handle_list_secrets_scans,
    handle_scan_secrets,
)

# SAST scanning handlers
from .sast import (
    handle_get_owasp_summary,
    handle_get_sast_findings,
    handle_get_sast_scan_status,
    handle_scan_sast,
)

# SBOM handlers
from .sbom import (
    handle_compare_sboms,
    handle_download_sbom,
    handle_generate_sbom,
    handle_get_sbom,
    handle_list_sboms,
)

# Event emission utilities (internal use)
from .events import (
    emit_sast_events,
    emit_scan_events,
    emit_secrets_events,
)

# Storage utilities (internal use)
from .storage import (
    get_cve_client,
    get_or_create_repo_scans,
    get_or_create_sbom_results,
    get_or_create_secrets_scans,
    get_running_sast_scans,
    get_running_sbom_generations,
    get_running_scans,
    get_running_secrets_scans,
    get_sast_scan_lock,
    get_sast_scan_results,
    get_sast_scanner,
    get_sbom_generator,
    get_sbom_lock,
    get_scan_lock,
    get_scanner,
    get_secrets_scan_lock,
    get_secrets_scanner,
)

# Backward compatibility aliases (for existing tests)
# These match the original private function names
_get_scanner = get_scanner
_get_cve_client = get_cve_client
_get_secrets_scanner = get_secrets_scanner
_get_sast_scanner = get_sast_scanner
_get_or_create_repo_scans = get_or_create_repo_scans

__all__ = [
    # Main handler
    "SecurityHandler",
    # Vulnerability scanning
    "handle_scan_repository",
    "handle_get_scan_status",
    "handle_get_vulnerabilities",
    "handle_get_cve_details",
    "handle_query_package_vulnerabilities",
    "handle_list_scans",
    # Secrets scanning
    "handle_scan_secrets",
    "handle_get_secrets_scan_status",
    "handle_get_secrets",
    "handle_list_secrets_scans",
    # SAST scanning
    "handle_scan_sast",
    "handle_get_sast_scan_status",
    "handle_get_sast_findings",
    "handle_get_owasp_summary",
    # SBOM
    "handle_generate_sbom",
    "handle_get_sbom",
    "handle_list_sboms",
    "handle_download_sbom",
    "handle_compare_sboms",
    # Events (internal)
    "emit_scan_events",
    "emit_secrets_events",
    "emit_sast_events",
    # Storage (internal)
    "get_scanner",
    "get_cve_client",
    "get_secrets_scanner",
    "get_sast_scanner",
    "get_sbom_generator",
    "get_or_create_repo_scans",
    "get_scan_lock",
    "get_running_scans",
    "get_or_create_secrets_scans",
    "get_secrets_scan_lock",
    "get_running_secrets_scans",
    "get_sast_scan_results",
    "get_sast_scan_lock",
    "get_running_sast_scans",
    "get_or_create_sbom_results",
    "get_sbom_lock",
    "get_running_sbom_generations",
    # Backward compatibility aliases
    "_get_scanner",
    "_get_cve_client",
    "_get_secrets_scanner",
    "_get_sast_scanner",
    "_get_or_create_repo_scans",
]
