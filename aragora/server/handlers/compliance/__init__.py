"""
Compliance domain handlers for Aragora.

This module consolidates compliance-related handlers:
- Audit: Audit trails, exports, logging, security auditing
- Governance: Policy management, compliance checks, GDPR

Re-exports for backward compatibility are provided so existing imports continue to work.
"""

from __future__ import annotations

# Audit handlers (audit trails, exports, security auditing)
from .audit import (
    # audit_export.py exports
    handle_audit_events,
    handle_audit_export,
    handle_audit_stats,
    handle_audit_verify,
    register_handlers as register_audit_handlers,
    get_audit_log,
    # audit_trail.py exports
    AuditTrailHandler,
    # auditing.py exports
    AuditRequestParser,
    AuditAgentFactory,
    AuditResultRecorder,
    AuditingHandler,
)

# Governance handlers (policy, compliance, GDPR)
from .governance import (
    # policy.py exports
    PolicyHandler,
    # compliance_handler.py exports
    ComplianceHandler,
    create_compliance_handler,
)

__all__ = [
    # Audit handlers
    "handle_audit_events",
    "handle_audit_export",
    "handle_audit_stats",
    "handle_audit_verify",
    "register_audit_handlers",
    "get_audit_log",
    "AuditTrailHandler",
    "AuditRequestParser",
    "AuditAgentFactory",
    "AuditResultRecorder",
    "AuditingHandler",
    # Governance handlers
    "PolicyHandler",
    "ComplianceHandler",
    "create_compliance_handler",
]
