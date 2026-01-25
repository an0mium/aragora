"""
Unified Audit Logging Facade.

Provides a single entry point for audit logging that dispatches to all
relevant audit systems while maintaining backward compatibility.

This consolidation layer:
- Provides a unified API for all audit logging needs
- Dispatches to appropriate backends (compliance, privacy, RBAC, immutable)
- Maintains consistent event format across systems
- Simplifies integration for new features
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class UnifiedAuditCategory(str, Enum):
    """Unified audit categories across all systems."""

    # Authentication and Authorization
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"
    AUTH_MFA = "auth.mfa"
    AUTH_TOKEN_ISSUED = "auth.token_issued"
    AUTH_TOKEN_REVOKED = "auth.token_revoked"

    # Access Control (RBAC)
    ACCESS_GRANTED = "access.granted"
    ACCESS_DENIED = "access.denied"
    ROLE_ASSIGNED = "access.role_assigned"
    ROLE_REVOKED = "access.role_revoked"
    PERMISSION_CHANGED = "access.permission_changed"

    # Data Operations
    DATA_READ = "data.read"
    DATA_CREATED = "data.created"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    DATA_EXPORTED = "data.exported"

    # Admin Actions
    ADMIN_CONFIG_CHANGED = "admin.config_changed"
    ADMIN_USER_CREATED = "admin.user_created"
    ADMIN_USER_DELETED = "admin.user_deleted"
    ADMIN_USER_MODIFIED = "admin.user_modified"

    # Security Events
    SECURITY_THREAT_DETECTED = "security.threat"
    SECURITY_ENCRYPTION = "security.encryption"
    SECURITY_KEY_ROTATION = "security.key_rotation"
    SECURITY_ANOMALY = "security.anomaly"

    # API Operations
    API_REQUEST = "api.request"
    API_RATE_LIMITED = "api.rate_limited"
    API_KEY_CREATED = "api.key_created"
    API_KEY_REVOKED = "api.key_revoked"

    # Debate and Workflow
    DEBATE_STARTED = "debate.started"
    DEBATE_COMPLETED = "debate.completed"
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    APPROVAL_REQUESTED = "workflow.approval_requested"
    APPROVAL_GRANTED = "workflow.approval_granted"
    APPROVAL_DENIED = "workflow.approval_denied"

    # System Events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"

    # Privacy Events
    PRIVACY_CONSENT_GIVEN = "privacy.consent_given"
    PRIVACY_CONSENT_WITHDRAWN = "privacy.consent_withdrawn"
    PRIVACY_DATA_REQUEST = "privacy.data_request"


class AuditOutcome(str, Enum):
    """Outcome of an audited action."""

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"
    PARTIAL = "partial"


class AuditSeverity(str, Enum):
    """Severity level for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class UnifiedAuditEvent:
    """
    Unified audit event that can be dispatched to any audit backend.

    This event structure is designed to be translatable to all existing
    audit systems while providing a consistent interface.
    """

    # Core fields
    category: UnifiedAuditCategory
    action: str  # Human-readable action description
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    severity: AuditSeverity = AuditSeverity.INFO

    # Actor information
    actor_id: Optional[str] = None
    actor_type: str = "user"  # user, service, system, api_key

    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None

    # Organizational context
    org_id: Optional[str] = None
    workspace_id: Optional[str] = None

    # Request context
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Event details
    details: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "action": self.action,
            "outcome": self.outcome.value,
            "severity": self.severity.value,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "details": self.details,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


class UnifiedAuditLogger:
    """
    Unified audit logger that dispatches to all configured backends.

    This provides a single point of entry for audit logging across the
    entire application, while maintaining compatibility with existing
    specialized audit systems.
    """

    def __init__(
        self,
        enable_compliance: bool = True,
        enable_privacy: bool = True,
        enable_rbac: bool = True,
        enable_immutable: bool = False,
        enable_middleware: bool = True,
    ):
        """
        Initialize the unified audit logger.

        Args:
            enable_compliance: Enable SOC2/HIPAA/GDPR compliance logging
            enable_privacy: Enable privacy-specific audit logging
            enable_rbac: Enable RBAC/authorization audit logging
            enable_immutable: Enable immutable audit log (append-only)
            enable_middleware: Enable HTTP middleware audit logging
        """
        self._enable_compliance = enable_compliance
        self._enable_privacy = enable_privacy
        self._enable_rbac = enable_rbac
        self._enable_immutable = enable_immutable
        self._enable_middleware = enable_middleware

        # Lazy-loaded backend instances
        self._compliance_logger = None
        self._privacy_logger = None
        self._rbac_auditor = None
        self._immutable_logger = None
        self._middleware_logger = None

        # Event handlers for custom integrations
        self._handlers: List[Callable[[UnifiedAuditEvent], None]] = []

    def add_handler(self, handler: Callable[[UnifiedAuditEvent], None]) -> None:
        """Add a custom event handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[UnifiedAuditEvent], None]) -> None:
        """Remove a custom event handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def _get_compliance_logger(self):
        """Lazy-load compliance audit logger."""
        if self._compliance_logger is None and self._enable_compliance:
            try:
                from aragora.audit.log import get_audit_log

                self._compliance_logger = get_audit_log()
            except ImportError:
                logger.debug("Compliance audit logger not available")
        return self._compliance_logger

    def _get_privacy_logger(self):
        """Lazy-load privacy audit logger."""
        if self._privacy_logger is None and self._enable_privacy:
            try:
                from aragora.privacy.audit_log import get_privacy_audit_log

                self._privacy_logger = get_privacy_audit_log()
            except ImportError:
                logger.debug("Privacy audit logger not available")
        return self._privacy_logger

    def _get_rbac_auditor(self):
        """Lazy-load RBAC auditor."""
        if self._rbac_auditor is None and self._enable_rbac:
            try:
                from aragora.rbac.audit import get_auditor

                self._rbac_auditor = get_auditor()
            except ImportError:
                logger.debug("RBAC auditor not available")
        return self._rbac_auditor

    def _get_immutable_logger(self):
        """Lazy-load immutable audit logger."""
        if self._immutable_logger is None and self._enable_immutable:
            try:
                from aragora.observability.immutable_log import get_audit_log

                self._immutable_logger = get_audit_log()
            except ImportError:
                logger.debug("Immutable audit logger not available")
        return self._immutable_logger

    def _get_middleware_logger(self):
        """Lazy-load middleware audit logger."""
        if self._middleware_logger is None and self._enable_middleware:
            try:
                from aragora.server.middleware.audit_logger import get_audit_logger

                self._middleware_logger = get_audit_logger()
            except ImportError:
                logger.debug("Middleware audit logger not available")
        return self._middleware_logger

    def log(self, event: UnifiedAuditEvent) -> None:
        """
        Log an audit event to all enabled backends.

        Args:
            event: The unified audit event to log
        """
        # Dispatch to compliance logger
        self._dispatch_to_compliance(event)

        # Dispatch to privacy logger for privacy-related events
        self._dispatch_to_privacy(event)

        # Dispatch to RBAC auditor for access events
        self._dispatch_to_rbac(event)

        # Dispatch to immutable logger if enabled
        self._dispatch_to_immutable(event)

        # Dispatch to middleware logger
        self._dispatch_to_middleware(event)

        # Call custom handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Audit handler error: {e}")

    def _dispatch_to_compliance(self, event: UnifiedAuditEvent) -> None:
        """Dispatch event to compliance audit logger."""
        log = self._get_compliance_logger()
        if log is None:
            return

        try:
            # Map to compliance categories
            category_map = {
                "auth.": "AUTH",
                "access.": "ACCESS",
                "data.": "DATA",
                "admin.": "ADMIN",
                "security.": "SECURITY",
                "api.": "API",
                "debate.": "DEBATE",
                "workflow.": "DEBATE",
                "system.": "SYSTEM",
            }

            category = "SYSTEM"
            for prefix, cat in category_map.items():
                if event.category.value.startswith(prefix):
                    category = cat
                    break

            log.log(
                category=category,
                action=event.action,
                actor=event.actor_id,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                outcome=event.outcome.value.upper(),
                org_id=event.org_id,
                ip_address=event.ip_address,
                details=event.details,
            )
        except Exception as e:
            logger.warning(f"Compliance audit dispatch error: {e}")

    def _dispatch_to_privacy(self, event: UnifiedAuditEvent) -> None:
        """Dispatch event to privacy audit logger."""
        log = self._get_privacy_logger()
        if log is None:
            return

        # Only dispatch privacy-relevant events
        privacy_categories = {"data.", "privacy.", "admin.user"}
        if not any(event.category.value.startswith(p) for p in privacy_categories):
            return

        try:
            # Map to privacy action
            action_map = {
                "data.read": "read",
                "data.created": "write",
                "data.updated": "write",
                "data.deleted": "delete",
                "data.exported": "export",
            }
            action = action_map.get(event.category.value, "read")

            log.log(
                actor=event.actor_id or "system",
                resource=event.resource_id or "unknown",
                action=action,
                workspace_id=event.workspace_id,
                success=event.outcome == AuditOutcome.SUCCESS,
                metadata=event.details,
            )
        except Exception as e:
            logger.warning(f"Privacy audit dispatch error: {e}")

    def _dispatch_to_rbac(self, event: UnifiedAuditEvent) -> None:
        """Dispatch event to RBAC auditor."""
        auditor = self._get_rbac_auditor()
        if auditor is None:
            return

        # Only dispatch access-related events
        if not event.category.value.startswith("access."):
            return

        try:
            if event.category == UnifiedAuditCategory.ACCESS_GRANTED:
                auditor.log_permission_granted(
                    user_id=event.actor_id,
                    permission=event.details.get("permission", "unknown"),
                    resource=event.resource_id,
                    context=event.details,
                )
            elif event.category == UnifiedAuditCategory.ACCESS_DENIED:
                auditor.log_permission_denied(
                    user_id=event.actor_id,
                    permission=event.details.get("permission", "unknown"),
                    resource=event.resource_id,
                    reason=event.reason,
                    context=event.details,
                )
        except Exception as e:
            logger.warning(f"RBAC audit dispatch error: {e}")

    def _dispatch_to_immutable(self, event: UnifiedAuditEvent) -> None:
        """Dispatch event to immutable audit logger."""
        log = self._get_immutable_logger()
        if log is None:
            return

        try:
            log.log(
                event_type=event.category.value,
                actor=event.actor_id or "system",
                action=event.action,
                resource=event.resource_id,
                details=event.to_dict(),
            )
        except Exception as e:
            logger.warning(f"Immutable audit dispatch error: {e}")

    def _dispatch_to_middleware(self, event: UnifiedAuditEvent) -> None:
        """Dispatch event to middleware audit logger."""
        log = self._get_middleware_logger()
        if log is None:
            return

        try:
            # Map severity
            severity_map = {
                AuditSeverity.DEBUG: "DEBUG",
                AuditSeverity.INFO: "INFO",
                AuditSeverity.WARNING: "WARNING",
                AuditSeverity.ERROR: "ERROR",
                AuditSeverity.CRITICAL: "CRITICAL",
            }

            log.log(
                event_type=event.category.value,
                action=event.action,
                actor=event.actor_id,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                outcome=event.outcome.value,
                severity=severity_map.get(event.severity, "INFO"),
                details=event.details,
                request_id=event.request_id,
                ip_address=event.ip_address,
            )
        except Exception as e:
            logger.warning(f"Middleware audit dispatch error: {e}")

    # Convenience methods for common audit events

    def log_auth_login(
        self,
        user_id: str,
        success: bool = True,
        ip_address: Optional[str] = None,
        method: str = "password",
        **kwargs,
    ) -> None:
        """Log a login attempt."""
        self.log(
            UnifiedAuditEvent(
                category=UnifiedAuditCategory.AUTH_LOGIN
                if success
                else UnifiedAuditCategory.AUTH_FAILED,
                action=f"User login via {method}",
                outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
                actor_id=user_id,
                ip_address=ip_address,
                details={"method": method, **kwargs},
            )
        )

    def log_auth_logout(self, user_id: str, **kwargs) -> None:
        """Log a logout."""
        self.log(
            UnifiedAuditEvent(
                category=UnifiedAuditCategory.AUTH_LOGOUT,
                action="User logout",
                actor_id=user_id,
                details=kwargs,
            )
        )

    def log_access_check(
        self,
        user_id: str,
        permission: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        granted: bool = True,
        reason: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log an access control check."""
        self.log(
            UnifiedAuditEvent(
                category=UnifiedAuditCategory.ACCESS_GRANTED
                if granted
                else UnifiedAuditCategory.ACCESS_DENIED,
                action=f"Permission check: {permission}",
                outcome=AuditOutcome.SUCCESS if granted else AuditOutcome.DENIED,
                actor_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                reason=reason,
                details={"permission": permission, **kwargs},
            )
        )

    def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str = "read",
        **kwargs,
    ) -> None:
        """Log a data access event."""
        category_map = {
            "read": UnifiedAuditCategory.DATA_READ,
            "create": UnifiedAuditCategory.DATA_CREATED,
            "update": UnifiedAuditCategory.DATA_UPDATED,
            "delete": UnifiedAuditCategory.DATA_DELETED,
            "export": UnifiedAuditCategory.DATA_EXPORTED,
        }
        self.log(
            UnifiedAuditEvent(
                category=category_map.get(action, UnifiedAuditCategory.DATA_READ),
                action=f"Data {action}: {resource_type}",
                actor_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                details=kwargs,
            )
        )

    def log_admin_action(
        self,
        admin_id: str,
        action: str,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log an administrative action."""
        self.log(
            UnifiedAuditEvent(
                category=UnifiedAuditCategory.ADMIN_CONFIG_CHANGED,
                action=f"Admin action: {action}",
                severity=AuditSeverity.WARNING,
                actor_id=admin_id,
                resource_type=target_type,
                resource_id=target_id,
                details={"action": action, **kwargs},
            )
        )

    def log_security_event(
        self,
        event_type: str,
        severity: AuditSeverity = AuditSeverity.WARNING,
        actor_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a security event."""
        category_map = {
            "threat": UnifiedAuditCategory.SECURITY_THREAT_DETECTED,
            "encryption": UnifiedAuditCategory.SECURITY_ENCRYPTION,
            "key_rotation": UnifiedAuditCategory.SECURITY_KEY_ROTATION,
            "anomaly": UnifiedAuditCategory.SECURITY_ANOMALY,
        }
        self.log(
            UnifiedAuditEvent(
                category=category_map.get(event_type, UnifiedAuditCategory.SECURITY_ANOMALY),
                action=f"Security event: {event_type}",
                severity=severity,
                actor_id=actor_id,
                details={"event_type": event_type, **kwargs},
            )
        )

    def log_debate_event(
        self,
        debate_id: str,
        action: str,  # started, completed, round_completed
        user_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a debate lifecycle event."""
        category_map = {
            "started": UnifiedAuditCategory.DEBATE_STARTED,
            "completed": UnifiedAuditCategory.DEBATE_COMPLETED,
        }
        self.log(
            UnifiedAuditEvent(
                category=category_map.get(action, UnifiedAuditCategory.DEBATE_STARTED),
                action=f"Debate {action}",
                actor_id=user_id,
                resource_type="debate",
                resource_id=debate_id,
                details={"action": action, **kwargs},
            )
        )


# Global instance
_unified_logger: Optional[UnifiedAuditLogger] = None


def get_unified_audit_logger() -> UnifiedAuditLogger:
    """Get the global unified audit logger instance."""
    global _unified_logger
    if _unified_logger is None:
        _unified_logger = UnifiedAuditLogger()
    return _unified_logger


def configure_unified_audit_logger(**kwargs) -> UnifiedAuditLogger:
    """Configure and return a new unified audit logger."""
    global _unified_logger
    _unified_logger = UnifiedAuditLogger(**kwargs)
    return _unified_logger


# Convenience functions
def audit_log(event: UnifiedAuditEvent) -> None:
    """Log an audit event using the global logger."""
    get_unified_audit_logger().log(event)


def audit_login(user_id: str, success: bool = True, **kwargs) -> None:
    """Log a login attempt."""
    get_unified_audit_logger().log_auth_login(user_id, success, **kwargs)


def audit_logout(user_id: str, **kwargs) -> None:
    """Log a logout."""
    get_unified_audit_logger().log_auth_logout(user_id, **kwargs)


def audit_access(
    user_id: str,
    permission: str,
    granted: bool = True,
    **kwargs,
) -> None:
    """Log an access control check."""
    get_unified_audit_logger().log_access_check(user_id, permission, granted=granted, **kwargs)


def audit_data(
    user_id: str,
    resource_type: str,
    resource_id: str,
    action: str = "read",
    **kwargs,
) -> None:
    """Log a data access event."""
    get_unified_audit_logger().log_data_access(
        user_id, resource_type, resource_id, action, **kwargs
    )


def audit_admin(admin_id: str, action: str, **kwargs) -> None:
    """Log an administrative action."""
    get_unified_audit_logger().log_admin_action(admin_id, action, **kwargs)


def audit_security(event_type: str, **kwargs) -> None:
    """Log a security event."""
    get_unified_audit_logger().log_security_event(event_type, **kwargs)


def audit_debate(debate_id: str, action: str, **kwargs) -> None:
    """Log a debate event."""
    get_unified_audit_logger().log_debate_event(debate_id, action, **kwargs)


__all__ = [
    "UnifiedAuditCategory",
    "AuditOutcome",
    "AuditSeverity",
    "UnifiedAuditEvent",
    "UnifiedAuditLogger",
    "get_unified_audit_logger",
    "configure_unified_audit_logger",
    "audit_log",
    "audit_login",
    "audit_logout",
    "audit_access",
    "audit_data",
    "audit_admin",
    "audit_security",
    "audit_debate",
]
