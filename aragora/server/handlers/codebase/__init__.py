"""
Codebase Analysis Handlers.

Provides HTTP API handlers for:
- Security vulnerability scanning
- Code quality metrics analysis
- Dependency auditing
- Code intelligence (AST parsing, call graphs, dead code)
- SBOM generation (CycloneDX 1.5, SPDX 2.3)
"""

from .security import (
    SecurityHandler,
    handle_scan_repository,
    handle_get_vulnerabilities,
    handle_get_scan_status,
    # SBOM handlers
    handle_generate_sbom,
    handle_get_sbom,
    handle_list_sboms,
    handle_download_sbom,
    handle_compare_sboms,
)
from .metrics import (
    MetricsHandler,
    handle_analyze_metrics,
    handle_get_hotspots,
)
from .intelligence import (
    IntelligenceHandler,
    handle_analyze_codebase,
    handle_get_symbols,
    handle_get_callgraph,
    handle_find_deadcode,
    handle_analyze_impact,
    handle_understand,
    handle_audit,
    handle_get_audit_status,
)

__all__ = [
    # Security
    "SecurityHandler",
    "handle_scan_repository",
    "handle_get_vulnerabilities",
    "handle_get_scan_status",
    # SBOM
    "handle_generate_sbom",
    "handle_get_sbom",
    "handle_list_sboms",
    "handle_download_sbom",
    "handle_compare_sboms",
    # Metrics
    "MetricsHandler",
    "handle_analyze_metrics",
    "handle_get_hotspots",
    # Intelligence
    "IntelligenceHandler",
    "handle_analyze_codebase",
    "handle_get_symbols",
    "handle_get_callgraph",
    "handle_find_deadcode",
    "handle_analyze_impact",
    "handle_understand",
    "handle_audit",
    "handle_get_audit_status",
]
