"""
RBAC Permissions for Compliance, Audit, and Data Governance resources.

Contains permissions related to:
- Compliance checks (SOC2, GDPR, HIPAA)
- Audit logs
- Policy management
- Data classification
- Data retention
- Data lineage
- PII handling
- Findings management
- Legal holds
"""

from __future__ import annotations

from aragora.rbac.models import Action, ResourceType

from ._helpers import _permission

# ============================================================================
# COMPLIANCE PERMISSIONS
# ============================================================================

PERM_COMPLIANCE_READ = _permission(
    ResourceType.COMPLIANCE, Action.READ, "View Compliance", "View compliance status and violations"
)
PERM_COMPLIANCE_UPDATE = _permission(
    ResourceType.COMPLIANCE, Action.UPDATE, "Update Compliance", "Update violation status"
)
PERM_COMPLIANCE_CHECK = _permission(
    ResourceType.COMPLIANCE, Action.CHECK, "Run Compliance Checks", "Execute compliance validation"
)
PERM_COMPLIANCE_GDPR = _permission(
    ResourceType.COMPLIANCE,
    Action.GDPR,
    "GDPR Operations",
    "Perform GDPR compliance operations (data export, deletion)",
)
PERM_COMPLIANCE_SOC2 = _permission(
    ResourceType.COMPLIANCE,
    Action.SOC2,
    "SOC2 Operations",
    "Access SOC2 compliance reports and controls",
)
PERM_COMPLIANCE_LEGAL = _permission(
    ResourceType.COMPLIANCE,
    Action.LEGAL,
    "Legal Operations",
    "Manage legal holds and compliance requirements",
)
PERM_COMPLIANCE_AUDIT = _permission(
    ResourceType.COMPLIANCE,
    Action.AUDIT,
    "Audit Operations",
    "Perform compliance audit verification",
)

# ============================================================================
# COMPLIANCE POLICY PERMISSIONS
# ============================================================================

PERM_COMPLIANCE_POLICY_READ = _permission(
    ResourceType.COMPLIANCE_POLICY,
    Action.READ,
    "View Compliance Policies",
    "Access compliance rules (SOC2, GDPR, HIPAA)",
)
PERM_COMPLIANCE_POLICY_UPDATE = _permission(
    ResourceType.COMPLIANCE_POLICY,
    Action.UPDATE,
    "Update Compliance Policies",
    "Modify compliance rules",
)
PERM_COMPLIANCE_POLICY_ENFORCE = _permission(
    ResourceType.COMPLIANCE_POLICY,
    Action.ENFORCE,
    "Enforce Compliance",
    "Force resolution of compliance findings",
)

# ============================================================================
# POLICY PERMISSIONS
# ============================================================================

PERM_POLICY_READ = _permission(
    ResourceType.POLICY, Action.READ, "View Policies", "View governance policies"
)
PERM_POLICY_CREATE = _permission(
    ResourceType.POLICY, Action.CREATE, "Create Policies", "Create new governance policies"
)
PERM_POLICY_UPDATE = _permission(
    ResourceType.POLICY, Action.UPDATE, "Update Policies", "Modify governance policies"
)
PERM_POLICY_DELETE = _permission(
    ResourceType.POLICY, Action.DELETE, "Delete Policies", "Remove governance policies"
)

# ============================================================================
# AUDIT LOG PERMISSIONS
# ============================================================================

PERM_AUDIT_LOG_READ = _permission(
    ResourceType.AUDIT_LOG, Action.READ, "View Audit Logs", "Access audit trail"
)
PERM_AUDIT_LOG_EXPORT = _permission(
    ResourceType.AUDIT_LOG,
    Action.EXPORT_DATA,
    "Export Audit Logs",
    "Export audit logs for compliance",
)
PERM_AUDIT_LOG_SEARCH = _permission(
    ResourceType.AUDIT_LOG,
    Action.SEARCH,
    "Search Audit Logs",
    "Advanced search in audit logs",
)
PERM_AUDIT_LOG_STREAM = _permission(
    ResourceType.AUDIT_LOG,
    Action.STREAM,
    "Stream Audit Logs",
    "Stream logs to external SIEM",
)
PERM_AUDIT_LOG_CONFIGURE = _permission(
    ResourceType.AUDIT_LOG,
    Action.UPDATE,
    "Configure Audit Retention",
    "Set audit log retention policies",
)
PERM_AUDIT_LOG_DELETE = _permission(
    ResourceType.AUDIT_LOG,
    Action.DELETE,
    "Delete Audit Logs",
    "Permanently delete audit trail records (compliance-critical)",
)

# ============================================================================
# DATA GOVERNANCE PERMISSIONS
# ============================================================================

PERM_DATA_CLASSIFICATION_READ = _permission(
    ResourceType.DATA_CLASSIFICATION,
    Action.READ,
    "View Data Classifications",
    "View data sensitivity classifications",
)
PERM_DATA_CLASSIFICATION_CLASSIFY = _permission(
    ResourceType.DATA_CLASSIFICATION,
    Action.CLASSIFY,
    "Classify Data",
    "Mark data as confidential/public/internal",
)
PERM_DATA_CLASSIFICATION_UPDATE = _permission(
    ResourceType.DATA_CLASSIFICATION,
    Action.UPDATE,
    "Update Classifications",
    "Modify existing data classifications",
)
PERM_DATA_RETENTION_READ = _permission(
    ResourceType.DATA_RETENTION,
    Action.READ,
    "View Retention Policies",
    "View data retention policies",
)
PERM_DATA_RETENTION_UPDATE = _permission(
    ResourceType.DATA_RETENTION,
    Action.UPDATE,
    "Configure Retention",
    "Set and enforce retention policies",
)
PERM_DATA_LINEAGE_READ = _permission(
    ResourceType.DATA_LINEAGE,
    Action.READ,
    "View Data Lineage",
    "Track data provenance and transformations",
)

# ============================================================================
# PII PERMISSIONS
# ============================================================================

PERM_PII_READ = _permission(
    ResourceType.PII, Action.READ, "View PII", "View personally identifiable information"
)
PERM_PII_REDACT = _permission(
    ResourceType.PII, Action.REDACT, "Redact PII", "Redact personally identifiable information"
)
PERM_PII_MASK = _permission(
    ResourceType.PII, Action.MASK, "Mask PII", "Configure PII masking rules"
)

# ============================================================================
# FINDINGS PERMISSIONS
# ============================================================================

PERM_FINDINGS_READ = _permission(
    ResourceType.FINDINGS, Action.READ, "View Findings", "View audit findings and history"
)
PERM_FINDINGS_UPDATE = _permission(
    ResourceType.FINDINGS, Action.UPDATE, "Update Findings", "Modify finding status and properties"
)
PERM_FINDINGS_ASSIGN = _permission(
    ResourceType.FINDINGS, Action.ASSIGN, "Assign Findings", "Assign findings to users"
)
PERM_FINDINGS_BULK = _permission(
    ResourceType.FINDINGS,
    Action.BULK,
    "Bulk Finding Operations",
    "Perform bulk actions on findings",
)

# ============================================================================
# LEGAL PERMISSIONS
# ============================================================================

PERM_LEGAL_READ = _permission(
    ResourceType.LEGAL, Action.READ, "View Legal", "View legal documents and status"
)

# All compliance-related permission exports
__all__ = [
    # Compliance
    "PERM_COMPLIANCE_READ",
    "PERM_COMPLIANCE_UPDATE",
    "PERM_COMPLIANCE_CHECK",
    "PERM_COMPLIANCE_GDPR",
    "PERM_COMPLIANCE_SOC2",
    "PERM_COMPLIANCE_LEGAL",
    "PERM_COMPLIANCE_AUDIT",
    # Compliance Policy
    "PERM_COMPLIANCE_POLICY_READ",
    "PERM_COMPLIANCE_POLICY_UPDATE",
    "PERM_COMPLIANCE_POLICY_ENFORCE",
    # Policy
    "PERM_POLICY_READ",
    "PERM_POLICY_CREATE",
    "PERM_POLICY_UPDATE",
    "PERM_POLICY_DELETE",
    # Audit Log
    "PERM_AUDIT_LOG_READ",
    "PERM_AUDIT_LOG_EXPORT",
    "PERM_AUDIT_LOG_SEARCH",
    "PERM_AUDIT_LOG_STREAM",
    "PERM_AUDIT_LOG_CONFIGURE",
    "PERM_AUDIT_LOG_DELETE",
    # Data Classification
    "PERM_DATA_CLASSIFICATION_READ",
    "PERM_DATA_CLASSIFICATION_CLASSIFY",
    "PERM_DATA_CLASSIFICATION_UPDATE",
    # Data Retention
    "PERM_DATA_RETENTION_READ",
    "PERM_DATA_RETENTION_UPDATE",
    # Data Lineage
    "PERM_DATA_LINEAGE_READ",
    # PII
    "PERM_PII_READ",
    "PERM_PII_REDACT",
    "PERM_PII_MASK",
    # Findings
    "PERM_FINDINGS_READ",
    "PERM_FINDINGS_UPDATE",
    "PERM_FINDINGS_ASSIGN",
    "PERM_FINDINGS_BULK",
    # Legal
    "PERM_LEGAL_READ",
]
