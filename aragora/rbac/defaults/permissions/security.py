"""
RBAC Permissions for Security resources.

Contains permissions related to:
- CVE scanning and analysis
- SAST (Static Application Security Testing)
- SBOM (Software Bill of Materials)
- Secrets scanning
- Vulnerability management
"""

from __future__ import annotations

from aragora.rbac.models import Action, ResourceType

from ._helpers import _permission

# ============================================================================
# CVE PERMISSIONS
# ============================================================================

PERM_CVE_READ = _permission(
    ResourceType.SECURITY,
    Action.READ,
    "Read CVE Data",
    "View CVE (Common Vulnerabilities and Exposures) data",
)

# ============================================================================
# SAST PERMISSIONS
# ============================================================================

PERM_SAST_READ = _permission(
    ResourceType.SECURITY,
    Action.READ,
    "Read SAST Results",
    "View SAST (Static Application Security Testing) scan results",
)

PERM_SAST_SCAN = _permission(
    ResourceType.SECURITY,
    Action.EXECUTE,
    "Run SAST Scan",
    "Execute SAST security scans on code",
)

# ============================================================================
# SBOM PERMISSIONS
# ============================================================================

PERM_SBOM_READ = _permission(
    ResourceType.SECURITY,
    Action.READ,
    "Read SBOM",
    "View Software Bill of Materials",
)

PERM_SBOM_GENERATE = _permission(
    ResourceType.SECURITY,
    Action.CREATE,
    "Generate SBOM",
    "Generate new Software Bill of Materials",
)

PERM_SBOM_COMPARE = _permission(
    ResourceType.SECURITY,
    Action.COMPARE,
    "Compare SBOM",
    "Compare Software Bill of Materials versions",
)

# ============================================================================
# SECRETS PERMISSIONS
# ============================================================================

PERM_SECRETS_READ = _permission(
    ResourceType.SECURITY,
    Action.READ,
    "Read Secrets Scan Results",
    "View secrets scanning results",
)

PERM_SECRETS_SCAN = _permission(
    ResourceType.SECURITY,
    Action.EXECUTE,
    "Run Secrets Scan",
    "Execute secrets scanning on code/configs",
)

# ============================================================================
# VULNERABILITY PERMISSIONS
# ============================================================================

PERM_VULNERABILITY_READ = _permission(
    ResourceType.SECURITY,
    Action.READ,
    "Read Vulnerabilities",
    "View vulnerability scan results and analysis",
)

PERM_VULNERABILITY_SCAN = _permission(
    ResourceType.SECURITY,
    Action.EXECUTE,
    "Run Vulnerability Scan",
    "Execute vulnerability scans",
)

__all__ = [
    "PERM_CVE_READ",
    "PERM_SAST_READ",
    "PERM_SAST_SCAN",
    "PERM_SBOM_COMPARE",
    "PERM_SBOM_GENERATE",
    "PERM_SBOM_READ",
    "PERM_SECRETS_READ",
    "PERM_SECRETS_SCAN",
    "PERM_VULNERABILITY_READ",
    "PERM_VULNERABILITY_SCAN",
]
