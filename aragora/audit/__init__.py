"""
Aragora Audit System.

Enterprise-grade audit logging for compliance (SOC 2, HIPAA, GDPR, SOX).

Usage:
    from aragora.audit import AuditLog, AuditEvent, AuditCategory

    audit = AuditLog()
    audit.log(AuditEvent(
        category=AuditCategory.AUTH,
        action="login",
        actor_id="user_123",
    ))
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

__all__ = [
    "AuditCategory",
    "AuditEvent",
    "AuditLog",
    "AuditOutcome",
    "AuditQuery",
    "audit_admin_action",
    "audit_auth_login",
    "audit_data_access",
]
