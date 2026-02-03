"""
Codebase Audit Package.

Stability: STABLE
Graduated from EXPERIMENTAL on 2026-02-02.

This package provides unified REST APIs for codebase security and quality
analysis. It is decomposed into the following modules:

- ``rules`` - Enums, data classes, constants, and validation functions
- ``scanning`` - Scanner integration and mock data generators
- ``reporting`` - Dashboard data generation and risk scoring
- ``handler`` - Main CodebaseAuditHandler class and request routing

All public symbols are re-exported here for backward compatibility.
"""

# Re-export rules: enums, data classes, constants, validation
from .rules import (
    CODEBASE_AUDIT_READ_PERMISSION,
    CODEBASE_AUDIT_WRITE_PERMISSION,
    VALID_SCAN_TYPES,
    VALID_SEVERITIES,
    VALID_STATUSES,
    VALID_SCAN_STATUSES,
    MAX_PATH_LENGTH,
    MAX_LIST_LIMIT,
    ScanType,
    ScanStatus,
    FindingSeverity,
    FindingStatus,
    Finding,
    ScanResult,
    ValidationError,
    validate_repository_path,
    validate_scan_types,
    validate_severity_filter,
    validate_status_filter,
    validate_scan_status_filter,
    validate_limit,
    validate_scan_id,
    validate_finding_id,
    validate_dismiss_request,
    validate_github_repo,
    sanitize_query_params,
)

# Re-export scanning: scanners, storage, mock data
from .scanning import (
    _scan_store,
    _finding_store,
    _get_tenant_scans,
    _get_tenant_findings,
    run_sast_scan,
    run_bug_scan,
    run_secrets_scan,
    run_dependency_scan,
    run_metrics_analysis,
    _map_severity,
    _get_mock_sast_findings,
    _get_mock_bug_findings,
    _get_mock_secrets_findings,
    _get_mock_dependency_findings,
    _get_mock_metrics,
)

# Re-export reporting
from .reporting import (
    calculate_risk_score,
    build_dashboard_data,
    build_demo_data,
)

# Re-export handler
from .handler import (
    CodebaseAuditHandler,
    get_codebase_audit_handler,
    handle_codebase_audit,
    _codebase_audit_circuit_breaker,
)

__all__ = [
    # Handler
    "CodebaseAuditHandler",
    "handle_codebase_audit",
    "get_codebase_audit_handler",
    # Enums
    "ScanType",
    "ScanStatus",
    "FindingSeverity",
    "FindingStatus",
    # Data classes
    "Finding",
    "ScanResult",
    # Exceptions
    "ValidationError",
    # Constants
    "CODEBASE_AUDIT_READ_PERMISSION",
    "CODEBASE_AUDIT_WRITE_PERMISSION",
    "VALID_SCAN_TYPES",
    "VALID_SEVERITIES",
    "VALID_STATUSES",
    "VALID_SCAN_STATUSES",
    "MAX_PATH_LENGTH",
    "MAX_LIST_LIMIT",
    # Validation
    "validate_repository_path",
    "validate_scan_types",
    "validate_severity_filter",
    "validate_status_filter",
    "validate_scan_status_filter",
    "validate_limit",
    "validate_scan_id",
    "validate_finding_id",
    "validate_dismiss_request",
    "validate_github_repo",
    "sanitize_query_params",
    # Scanning
    "run_sast_scan",
    "run_bug_scan",
    "run_secrets_scan",
    "run_dependency_scan",
    "run_metrics_analysis",
    # Mock data
    "_get_mock_sast_findings",
    "_get_mock_bug_findings",
    "_get_mock_secrets_findings",
    "_get_mock_dependency_findings",
    "_get_mock_metrics",
    # Storage
    "_scan_store",
    "_finding_store",
    "_get_tenant_scans",
    "_get_tenant_findings",
    "_map_severity",
    # Circuit breaker
    "_codebase_audit_circuit_breaker",
    # Reporting
    "calculate_risk_score",
    "build_dashboard_data",
    "build_demo_data",
]
