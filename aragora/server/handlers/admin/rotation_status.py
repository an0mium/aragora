"""
Key Rotation Status Admin Handler.

Endpoints:
- GET /api/v1/admin/security/rotation-status - Get rotation status for all managed secrets
- GET /api/admin/security/rotation-status    - (non-versioned backward compat)

All endpoints require admin role and admin:security:read permission.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ..base import (
    HandlerResult,
    error_response,
    json_response,
    handle_errors,
)
from ..secure import SecureHandler
from ..utils.rate_limit import RateLimiter, get_client_ip
from .admin import admin_secure_endpoint

from aragora.events.handler_events import emit_handler_event, COMPLETED
from aragora.rbac.decorators import require_permission

try:
    from aragora.rbac.checker import check_permission  # noqa: F401
    from aragora.rbac.models import AuthorizationContext  # noqa: F401

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False

from aragora.server.handlers.utils.rbac_guard import rbac_fail_closed

logger = logging.getLogger(__name__)

# Permission required for rotation status
ROTATION_STATUS_PERMISSION = "admin:security"

# Rate limiter (10 requests per minute)
_rotation_limiter = RateLimiter(requests_per_minute=10)


class RotationStatusHandler(SecureHandler):
    """Handler for key rotation status endpoints."""

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = [
        "/api/v1/admin/security/rotation-status",
        "/api/admin/security/rotation-status",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    @handle_errors("rotation status read")
    @require_permission("admin:security:read")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route rotation status GET requests."""
        # Rate limit
        client_ip = get_client_ip(handler)
        if not _rotation_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for rotation-status: %s", client_ip)
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # RBAC check
        if not RBAC_AVAILABLE:
            if rbac_fail_closed():
                return error_response(
                    "Service unavailable: access control module not loaded", 503
                )
        elif hasattr(handler, "auth_context"):
            decision = check_permission(handler.auth_context, ROTATION_STATUS_PERMISSION)
            if not decision.allowed:
                logger.warning("RBAC denied rotation status access: %s", decision.reason)
                return error_response("Permission denied", 403, code="PERMISSION_DENIED")

        if path in self.ROUTES:
            return self._get_rotation_status(handler)
        return None

    @admin_secure_endpoint(
        permission="admin.security.rotation_status",
        audit=True,
        audit_action="rotation_status_viewed",
    )
    def _get_rotation_status(self, handler: Any) -> HandlerResult:
        """Get rotation status for all managed secrets.

        Returns:
            200: JSON with rotation status per secret, scheduler info, and summary
            500: Internal error
        """
        try:
            result: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "secrets": [],
                "scheduler": {},
                "summary": {},
            }

            # 1. AWS rotation statuses (from AWSSecretRotator if available)
            aws_statuses = self._get_aws_rotation_statuses()
            result["secrets"].extend(aws_statuses)

            # 2. Local key rotation scheduler status
            scheduler_info = self._get_scheduler_status()
            result["scheduler"] = scheduler_info

            # 3. Encryption key age info
            encryption_status = self._get_encryption_key_status()
            if encryption_status:
                result["secrets"].append(encryption_status)

            # 4. Summary
            all_secrets = result["secrets"]
            due_count = sum(1 for s in all_secrets if s.get("is_due", False))
            pending_count = sum(
                1 for s in all_secrets if s.get("pending_rotation", False)
            )
            result["summary"] = {
                "total_tracked": len(all_secrets),
                "due_for_rotation": due_count,
                "pending_rotation": pending_count,
                "healthy": due_count == 0 and pending_count == 0,
            }

            emit_handler_event(
                "admin",
                COMPLETED,
                {"action": "rotation_status_viewed"},
            )
            return json_response({"data": result})

        except ImportError as e:
            logger.error("Rotation status import error: %s", e)
            return error_response("Internal server error", 500)
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Rotation status error: %s", e)
            return error_response("Internal server error", 500)

    def _get_aws_rotation_statuses(self) -> list[dict[str, Any]]:
        """Retrieve rotation statuses from the AWS rotator if configured."""
        try:
            from aragora.security.aws_key_rotation import (
                get_rotation_monitor,
            )

            monitor = get_rotation_monitor()
            if monitor is None:
                return []

            rotator = monitor.rotator
            statuses = rotator.get_all_rotation_statuses()
            return [s.to_dict() for s in statuses]
        except ImportError:
            return []
        except (RuntimeError, ValueError, AttributeError, OSError) as e:
            logger.debug("AWS rotation status unavailable: %s", e)
            return []

    def _get_scheduler_status(self) -> dict[str, Any]:
        """Get the local KeyRotationScheduler status."""
        try:
            from aragora.security.key_rotation import get_key_rotation_scheduler

            scheduler = get_key_rotation_scheduler()
            if scheduler is None:
                return {"status": "not_configured", "message": "No rotation scheduler active"}

            stats = scheduler.get_stats()
            tracked = scheduler.get_tracked_keys()

            return {
                "status": stats.status.value,
                "total_rotations": stats.total_rotations,
                "successful_rotations": stats.successful_rotations,
                "failed_rotations": stats.failed_rotations,
                "last_rotation_at": (
                    stats.last_rotation_at.isoformat()
                    if stats.last_rotation_at
                    else None
                ),
                "next_check_at": (
                    stats.next_check_at.isoformat() if stats.next_check_at else None
                ),
                "keys_tracked": stats.keys_tracked,
                "keys_expiring_soon": stats.keys_expiring_soon,
                "uptime_seconds": round(stats.uptime_seconds, 1),
                "tracked_keys": [
                    {
                        "key_id": k.key_id,
                        "provider": k.provider,
                        "version": k.version,
                        "last_rotated_at": (
                            k.last_rotated_at.isoformat()
                            if k.last_rotated_at
                            else None
                        ),
                        "next_rotation_at": (
                            k.next_rotation_at.isoformat()
                            if k.next_rotation_at
                            else None
                        ),
                        "is_active": k.is_active,
                    }
                    for k in tracked
                ],
            }
        except ImportError:
            return {"status": "unavailable", "message": "Key rotation module not available"}
        except (RuntimeError, ValueError, AttributeError, OSError) as e:
            logger.debug("Scheduler status error: %s", e)
            return {"status": "error", "message": "Failed to retrieve scheduler status"}

    def _get_encryption_key_status(self) -> dict[str, Any] | None:
        """Get current encryption key age and rotation recommendation."""
        try:
            from aragora.security.encryption import get_encryption_service, CRYPTO_AVAILABLE

            if not CRYPTO_AVAILABLE:
                return None

            service = get_encryption_service()
            active_key = service.get_active_key()
            if active_key is None:
                return {
                    "secret_id": "local/encryption_key",
                    "secret_type": "encryption_key",
                    "is_due": True,
                    "pending_rotation": False,
                    "warning": "No active encryption key",
                }

            age_days = (datetime.now(timezone.utc) - active_key.created_at).days
            is_due = age_days > 90

            return {
                "secret_id": "local/encryption_key",
                "secret_type": "encryption_key",
                "key_id": active_key.key_id,
                "version": active_key.version,
                "age_days": age_days,
                "created_at": active_key.created_at.isoformat(),
                "is_due": is_due,
                "pending_rotation": False,
                "rotation_recommended": age_days > 60,
            }
        except ImportError:
            return None
        except (RuntimeError, ValueError, OSError) as e:
            logger.debug("Encryption key status error: %s", e)
            return None


__all__ = [
    "RotationStatusHandler",
]
