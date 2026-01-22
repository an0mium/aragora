"""
Security endpoint handlers.

Endpoints:
- GET /api/admin/security/status - Get encryption and key status
- POST /api/admin/security/rotate-key - Rotate encryption key
- GET /api/admin/security/health - Check encryption health

All endpoints require admin or owner role.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..base import (
    HandlerResult,
    error_response,
    json_response,
)
from ..secure import SecureHandler
from .admin import admin_secure_endpoint

logger = logging.getLogger(__name__)


class SecurityHandler(SecureHandler):
    """Handler for security-related admin endpoints."""

    ROUTES = [
        "/api/v1/admin/security/status",
        "/api/v1/admin/security/rotate-key",
        "/api/v1/admin/security/health",
        "/api/v1/admin/security/keys",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route security endpoint requests."""
        handlers = {
            "/api/v1/admin/security/status": self._get_status,
            "/api/v1/admin/security/health": self._get_health,
            "/api/v1/admin/security/keys": self._list_keys,
        }

        # GET endpoints
        endpoint_handler = handlers.get(path)
        if endpoint_handler:
            return endpoint_handler(handler)
        return None

    def handle_post(self, path: str, data: Dict[str, Any], handler: Any) -> Optional[HandlerResult]:
        """Handle POST requests for security endpoints."""
        if path == "/api/v1/admin/security/rotate-key":
            return self._rotate_key(data, handler)
        return None

    @admin_secure_endpoint(
        permission="admin.security.status",
        audit=True,
        audit_action="security_status_viewed",
    )
    def _get_status(self, handler: Any) -> HandlerResult:
        """
        Get encryption and key status.

        Returns:
            200: Encryption status information
            500: Error getting status
        """
        try:
            from aragora.security.encryption import get_encryption_service, CRYPTO_AVAILABLE

            if not CRYPTO_AVAILABLE:
                return json_response(
                    {
                        "crypto_available": False,
                        "error": "Cryptography library not installed",
                    }
                )

            service = get_encryption_service()
            active_key = service.get_active_key()

            result: Dict[str, Any] = {
                "crypto_available": True,
                "active_key_id": service.get_active_key_id(),
            }

            if active_key:
                age_days = (datetime.now(timezone.utc) - active_key.created_at).days
                result.update(
                    {
                        "key_version": active_key.version,
                        "key_age_days": age_days,
                        "key_created_at": active_key.created_at.isoformat(),
                        "rotation_recommended": age_days > 60,
                        "rotation_required": age_days > 90,
                    }
                )
            else:
                result["warning"] = "No active encryption key found"

            # Count all keys
            all_keys = service.list_keys()
            result["total_keys"] = len(all_keys)

            return json_response(result)

        except ImportError as e:
            logger.error(f"Security status import error: {e}")
            return error_response(str(e), 500)
        except Exception as e:
            logger.error(f"Security status error: {e}")
            return error_response(str(e), 500)

    @admin_secure_endpoint(
        permission="admin.security.rotate",
        audit=True,
        audit_action="key_rotation",
    )
    def _rotate_key(self, data: Dict[str, Any], handler: Any) -> HandlerResult:
        """
        Rotate encryption key.

        Request body:
            dry_run (bool): If true, only preview what would be done
            stores (list[str]): Stores to re-encrypt (default: all)
            force (bool): Force rotation even if key is recent

        Returns:
            200: Rotation result
            400: Invalid request
            500: Rotation failed
        """
        try:
            from aragora.security.migration import rotate_encryption_key
            from aragora.security.encryption import get_encryption_service, CRYPTO_AVAILABLE

            if not CRYPTO_AVAILABLE:
                return error_response("Cryptography library not available", 400)

            dry_run = data.get("dry_run", False)
            stores = data.get("stores")
            force = data.get("force", False)

            # Check if rotation is needed (unless forced)
            if not force and not dry_run:
                service = get_encryption_service()
                active_key = service.get_active_key()
                if active_key:
                    age_days = (datetime.now(timezone.utc) - active_key.created_at).days
                    if age_days < 30:
                        return error_response(
                            f"Key is only {age_days} days old. "
                            "Use 'force: true' to rotate anyway.",
                            400,
                        )

            result = rotate_encryption_key(
                stores=stores,
                dry_run=dry_run,
            )

            return json_response(
                {
                    "success": result.success,
                    "dry_run": dry_run,
                    "old_key_version": result.old_key_version,
                    "new_key_version": result.new_key_version,
                    "stores_processed": result.stores_processed,
                    "records_reencrypted": result.records_reencrypted,
                    "failed_records": result.failed_records,
                    "duration_seconds": result.duration_seconds,
                    "errors": result.errors[:10] if result.errors else [],
                }
            )

        except ImportError as e:
            logger.error(f"Key rotation import error: {e}")
            return error_response(str(e), 500)
        except Exception as e:
            logger.error(f"Key rotation error: {e}")
            return error_response(str(e), 500)

    @admin_secure_endpoint(
        permission="admin.security.health",
        audit=False,
    )
    def _get_health(self, handler: Any) -> HandlerResult:
        """
        Check encryption health.

        Returns:
            200: Health check results
            500: Health check failed
        """
        try:
            from aragora.security.encryption import get_encryption_service, CRYPTO_AVAILABLE

            issues: List[str] = []
            warnings: List[str] = []
            checks: Dict[str, Any] = {}

            # Check 1: Crypto library
            checks["crypto_available"] = CRYPTO_AVAILABLE
            if not CRYPTO_AVAILABLE:
                issues.append("Cryptography library not installed")
                return json_response(
                    {
                        "status": "unhealthy",
                        "checks": checks,
                        "issues": issues,
                        "warnings": warnings,
                    }
                )

            # Check 2: Encryption service
            try:
                service = get_encryption_service()
                checks["service_initialized"] = True
            except Exception as e:
                checks["service_initialized"] = False
                issues.append(f"Encryption service error: {e}")
                return json_response(
                    {
                        "status": "unhealthy",
                        "checks": checks,
                        "issues": issues,
                        "warnings": warnings,
                    }
                )

            # Check 3: Active key
            active_key = service.get_active_key()
            checks["active_key"] = active_key is not None
            if active_key:
                age_days = (datetime.now(timezone.utc) - active_key.created_at).days
                checks["key_age_days"] = age_days
                checks["key_version"] = active_key.version

                if age_days > 90:
                    warnings.append(f"Key is {age_days} days old (>90 days)")
                elif age_days > 60:
                    warnings.append(f"Key is {age_days} days old, rotation recommended")
            else:
                issues.append("No active encryption key")

            # Check 4: Encrypt/decrypt round-trip
            try:
                test_data = b"health_check_test_data"
                encrypted = service.encrypt(test_data)
                decrypted = service.decrypt(encrypted)
                checks["round_trip"] = decrypted == test_data
                if decrypted != test_data:
                    issues.append("Encrypt/decrypt round-trip failed")
            except Exception as e:
                checks["round_trip"] = False
                issues.append(f"Encrypt/decrypt error: {e}")

            # Determine overall status
            if issues:
                status = "unhealthy"
            elif warnings:
                status = "degraded"
            else:
                status = "healthy"

            return json_response(
                {
                    "status": status,
                    "checks": checks,
                    "issues": issues,
                    "warnings": warnings,
                }
            )

        except Exception as e:
            logger.error(f"Security health check error: {e}")
            return error_response(str(e), 500)

    @admin_secure_endpoint(
        permission="admin.security.keys",
        audit=True,
        audit_action="keys_listed",
    )
    def _list_keys(self, handler: Any) -> HandlerResult:
        """
        List all encryption keys.

        Returns:
            200: List of keys (without sensitive data)
            500: Error listing keys
        """
        try:
            from aragora.security.encryption import get_encryption_service, CRYPTO_AVAILABLE

            if not CRYPTO_AVAILABLE:
                return error_response("Cryptography library not available", 400)

            service = get_encryption_service()
            active_key_id = service.get_active_key_id()
            all_keys = service.list_keys()

            keys_info = []
            for key in all_keys:
                age_days = (datetime.now(timezone.utc) - key.created_at).days  # type: ignore[attr-defined]
                keys_info.append(
                    {
                        "key_id": key.key_id,  # type: ignore[attr-defined]
                        "version": key.version,  # type: ignore[attr-defined]
                        "is_active": key.key_id == active_key_id,  # type: ignore[attr-defined]
                        "created_at": key.created_at.isoformat(),  # type: ignore[attr-defined]
                        "age_days": age_days,
                    }
                )

            return json_response(
                {
                    "keys": keys_info,
                    "active_key_id": active_key_id,
                    "total_keys": len(keys_info),
                }
            )

        except ImportError as e:
            logger.error(f"List keys import error: {e}")
            return error_response(str(e), 500)
        except Exception as e:
            logger.error(f"List keys error: {e}")
            return error_response(str(e), 500)
