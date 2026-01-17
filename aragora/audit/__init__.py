"""
Aragora Audit System.

Enterprise-grade audit logging for compliance (SOC 2, HIPAA, GDPR, SOX)
and document auditing for defect detection.

Usage:
    # Compliance logging
    from aragora.audit import AuditLog, AuditEvent, AuditCategory

    audit = AuditLog()
    audit.log(AuditEvent(
        category=AuditCategory.AUTH,
        action="login",
        actor_id="user_123",
    ))

    # Document auditing
    from aragora.audit import DocumentAuditor, AuditSession

    auditor = DocumentAuditor()
    session = await auditor.create_session(document_ids=["doc1"])
    await auditor.run_audit(session.id)
"""

from .log import (
    AuditCategory,
    AuditEvent,
    AuditLog,
    AuditOutcome,
    AuditQuery,
    audit_admin_action,
    audit_auth_login,
    audit_data_access,
)
from .document_auditor import (
    DocumentAuditor,
    AuditSession,
    AuditConfig,
    AuditFinding,
    AuditType,
    AuditStatus,
    FindingSeverity,
    FindingStatus,
    get_document_auditor,
)
from .hive_mind import (
    AuditHiveMind,
    HiveMindConfig,
    HiveMindResult,
    WorkerTask,
    WorkerResult,
    QueenOrchestrator,
)

__all__ = [
    # Compliance logging
    "AuditCategory",
    "AuditEvent",
    "AuditLog",
    "AuditOutcome",
    "AuditQuery",
    "audit_admin_action",
    "audit_auth_login",
    "audit_data_access",
    # Document auditing
    "DocumentAuditor",
    "AuditSession",
    "AuditConfig",
    "AuditFinding",
    "AuditType",
    "AuditStatus",
    "FindingSeverity",
    "FindingStatus",
    "get_document_auditor",
    # Hive-Mind Architecture
    "AuditHiveMind",
    "HiveMindConfig",
    "HiveMindResult",
    "WorkerTask",
    "WorkerResult",
    "QueenOrchestrator",
]
