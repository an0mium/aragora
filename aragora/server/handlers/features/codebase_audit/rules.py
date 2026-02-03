"""
Codebase Audit Rules, Enums, Data Classes, and Validation.

Contains all audit rule definitions, input validation functions,
enums, and data classes used across the codebase audit system.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# RBAC Permissions
# =============================================================================

# Read: view scan results, findings, dashboard
CODEBASE_AUDIT_READ_PERMISSION = "codebase_audit:read"
# Write: start scans, dismiss findings, create issues
CODEBASE_AUDIT_WRITE_PERMISSION = "codebase_audit:write"

# =============================================================================
# Input Validation Constants
# =============================================================================

# Valid scan types for validation
VALID_SCAN_TYPES = {"sast", "bugs", "secrets", "dependencies", "metrics", "comprehensive"}

# Valid severity values for filtering
VALID_SEVERITIES = {"critical", "high", "medium", "low", "info"}

# Valid status values for filtering
VALID_STATUSES = {"open", "dismissed", "fixed", "false_positive", "accepted_risk"}

# Valid scan status values for filtering
VALID_SCAN_STATUSES = {"pending", "running", "completed", "failed", "cancelled"}

# Maximum path length to prevent DoS
MAX_PATH_LENGTH = 4096

# Maximum limit for list queries
MAX_LIST_LIMIT = 500


# =============================================================================
# Enums
# =============================================================================


class ScanType(Enum):
    """Types of code analysis scans."""

    COMPREHENSIVE = "comprehensive"
    SAST = "sast"
    BUGS = "bugs"
    SECRETS = "secrets"
    DEPENDENCIES = "dependencies"
    METRICS = "metrics"


class ScanStatus(Enum):
    """Scan execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FindingSeverity(Enum):
    """Finding severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingStatus(Enum):
    """Finding status."""

    OPEN = "open"
    DISMISSED = "dismissed"
    FIXED = "fixed"
    FALSE_POSITIVE = "false_positive"
    ACCEPTED_RISK = "accepted_risk"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Finding:
    """A security or quality finding."""

    id: str
    scan_id: str
    scan_type: ScanType
    severity: FindingSeverity
    title: str
    description: str
    file_path: str
    line_number: int | None = None
    column: int | None = None
    code_snippet: str | None = None
    rule_id: str | None = None
    cwe_id: str | None = None
    owasp_category: str | None = None
    remediation: str | None = None
    confidence: float = 0.8
    status: FindingStatus = FindingStatus.OPEN
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dismissed_by: str | None = None
    dismissed_reason: str | None = None
    github_issue_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scan_id": self.scan_id,
            "scan_type": self.scan_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "code_snippet": self.code_snippet,
            "rule_id": self.rule_id,
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "remediation": self.remediation,
            "confidence": self.confidence,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "dismissed_by": self.dismissed_by,
            "dismissed_reason": self.dismissed_reason,
            "github_issue_url": self.github_issue_url,
        }


@dataclass
class ScanResult:
    """Result of a codebase scan."""

    id: str
    tenant_id: str
    scan_type: ScanType
    status: ScanStatus
    target_path: str
    started_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    files_scanned: int = 0
    findings_count: int = 0
    severity_counts: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    progress: float = 0.0
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "scan_type": self.scan_type.value,
            "status": self.status.value,
            "target_path": self.target_path,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "files_scanned": self.files_scanned,
            "findings_count": self.findings_count,
            "severity_counts": self.severity_counts,
            "duration_seconds": self.duration_seconds,
            "progress": self.progress,
            "metrics": self.metrics,
        }


# =============================================================================
# Exceptions
# =============================================================================


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field
        self.message = message


# =============================================================================
# Validation Functions
# =============================================================================


def validate_repository_path(path: str | None) -> tuple[bool, str | None]:
    """Validate repository path to prevent directory traversal attacks.

    Args:
        path: The repository path to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path:
        return True, None  # Default path "." is valid

    # Check path length
    if len(path) > MAX_PATH_LENGTH:
        return False, f"Path exceeds maximum length of {MAX_PATH_LENGTH} characters"

    # Prevent directory traversal
    if ".." in path:
        return False, "Path contains invalid directory traversal sequence '..'"

    # Prevent null bytes (used in path injection attacks)
    if "\x00" in path:
        return False, "Path contains invalid null byte"

    # Prevent shell metacharacters
    dangerous_chars = [";", "|", "&", "$", "`", "(", ")", "{", "}", "<", ">", "\n", "\r"]
    for char in dangerous_chars:
        if char in path:
            return False, f"Path contains invalid character: {repr(char)}"

    # Prevent absolute paths starting with / (only allow relative paths)
    if path.startswith("/"):
        return False, "Absolute paths are not allowed; use relative paths only"

    # Prevent paths that start with ~ (home directory expansion)
    if path.startswith("~"):
        return False, "Home directory expansion (~) is not allowed"

    return True, None


def validate_scan_types(scan_types: list[str] | None) -> tuple[bool, str | None]:
    """Validate scan types against allowed values.

    Args:
        scan_types: List of scan types to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not scan_types:
        return True, None

    if not isinstance(scan_types, list):
        return False, "scan_types must be a list"

    invalid_types = set(scan_types) - VALID_SCAN_TYPES
    if invalid_types:
        return False, f"Invalid scan types: {invalid_types}. Valid types are: {VALID_SCAN_TYPES}"

    return True, None


def validate_severity_filter(severity: str | None) -> tuple[bool, str | None]:
    """Validate severity filter value.

    Args:
        severity: Severity value to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not severity:
        return True, None

    if severity.lower() not in VALID_SEVERITIES:
        return False, f"Invalid severity: {severity}. Valid values are: {VALID_SEVERITIES}"

    return True, None


def validate_status_filter(status: str | None) -> tuple[bool, str | None]:
    """Validate finding status filter value.

    Args:
        status: Status value to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not status:
        return True, None

    if status.lower() not in VALID_STATUSES:
        return False, f"Invalid status: {status}. Valid values are: {VALID_STATUSES}"

    return True, None


def validate_scan_status_filter(status: str | None) -> tuple[bool, str | None]:
    """Validate scan status filter value.

    Args:
        status: Scan status value to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not status:
        return True, None

    if status.lower() not in VALID_SCAN_STATUSES:
        return False, f"Invalid scan status: {status}. Valid values are: {VALID_SCAN_STATUSES}"

    return True, None


def validate_limit(
    limit_str: str | None, max_limit: int = MAX_LIST_LIMIT
) -> tuple[int, str | None]:
    """Validate and parse limit parameter.

    Args:
        limit_str: String limit value to validate
        max_limit: Maximum allowed limit

    Returns:
        Tuple of (parsed_limit, error_message)
    """
    if not limit_str:
        return 20, None  # Default limit

    try:
        limit = int(limit_str)
    except ValueError:
        return 0, f"Invalid limit: {limit_str}. Must be a positive integer"

    if limit < 1:
        return 0, "Limit must be at least 1"

    if limit > max_limit:
        return max_limit, None  # Cap at max, don't error

    return limit, None


def validate_scan_id(scan_id: str | None) -> tuple[bool, str | None]:
    """Validate scan ID format.

    Args:
        scan_id: Scan ID to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not scan_id:
        return False, "Scan ID is required"

    # Scan IDs should be alphanumeric with underscores
    if not scan_id.replace("_", "").replace("-", "").isalnum():
        return False, "Scan ID contains invalid characters"

    if len(scan_id) > 64:
        return False, "Scan ID exceeds maximum length"

    return True, None


def validate_finding_id(finding_id: str | None) -> tuple[bool, str | None]:
    """Validate finding ID format.

    Args:
        finding_id: Finding ID to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not finding_id:
        return False, "Finding ID is required"

    # Finding IDs should be alphanumeric with underscores
    if not finding_id.replace("_", "").replace("-", "").isalnum():
        return False, "Finding ID contains invalid characters"

    if len(finding_id) > 64:
        return False, "Finding ID exceeds maximum length"

    return True, None


def validate_dismiss_request(body: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate dismiss finding request body.

    Args:
        body: Request body dict

    Returns:
        Tuple of (is_valid, error_message)
    """
    status_type = body.get("status", "dismissed")
    valid_dismiss_statuses = {"dismissed", "false_positive", "accepted_risk", "fixed"}

    if status_type not in valid_dismiss_statuses:
        return (
            False,
            f"Invalid dismiss status: {status_type}. Valid values are: {valid_dismiss_statuses}",
        )

    reason = body.get("reason", "")
    if len(reason) > 2000:
        return False, "Reason exceeds maximum length of 2000 characters"

    return True, None


def validate_github_repo(repo: str | None) -> tuple[bool, str | None]:
    """Validate GitHub repository format.

    Args:
        repo: Repository in owner/name format

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not repo:
        return False, "Repository is required"

    # Basic format: owner/name
    if "/" not in repo:
        return False, "Repository must be in 'owner/name' format"

    parts = repo.split("/")
    if len(parts) != 2:
        return False, "Repository must be in 'owner/name' format"

    owner, name = parts
    if not owner or not name:
        return False, "Repository owner and name cannot be empty"

    # Validate characters (GitHub allows alphanumeric, hyphens, underscores, dots)
    if not re.match(r"^[\w.-]+$", owner) or not re.match(r"^[\w.-]+$", name):
        return False, "Repository owner/name contains invalid characters"

    return True, None


def sanitize_query_params(params: dict[str, Any]) -> dict[str, str]:
    """Sanitize query parameters to prevent injection attacks.

    Args:
        params: Raw query parameters

    Returns:
        Sanitized parameters dict
    """
    sanitized: dict[str, str] = {}

    # Only allow known parameter keys
    allowed_keys = {"type", "status", "severity", "limit", "offset", "scan_type"}

    for key, value in params.items():
        if key not in allowed_keys:
            logger.warning(f"Unknown query parameter ignored: {key}")
            continue

        # Convert value to string and truncate
        str_value = str(value)[:256] if value is not None else ""

        # Remove any control characters
        str_value = "".join(c for c in str_value if ord(c) >= 32)

        sanitized[key] = str_value

    return sanitized
