"""
Admin API Handlers.

Provides administrative endpoints for system-wide management.
All endpoints require admin or owner role.

Endpoints:
- GET /api/admin/organizations - List all organizations
- GET /api/admin/users - List all users
- GET /api/admin/stats - Get system-wide statistics
- GET /api/admin/system/metrics - Get aggregated system metrics
- POST /api/admin/impersonate/:user_id - Create impersonation token
- POST /api/admin/users/:user_id/deactivate - Deactivate a user
- POST /api/admin/users/:user_id/activate - Activate a user
- POST /api/admin/users/:user_id/unlock - Unlock a locked user account
- GET /api/admin/nomic/status - Get detailed nomic status
- GET /api/admin/nomic/circuit-breakers - Get circuit breaker status
- POST /api/admin/nomic/reset - Reset nomic to a specific phase
- POST /api/admin/nomic/pause - Pause the nomic loop
- POST /api/admin/nomic/resume - Resume the nomic loop
- POST /api/admin/nomic/circuit-breakers/reset - Reset all circuit breakers
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from aragora.auth.lockout import get_lockout_tracker
from aragora.billing.jwt_auth import create_access_token, extract_user_from_request
from aragora.server.middleware.mfa import enforce_admin_mfa_policy

from ..base import (
    SAFE_ID_PATTERN,
    BaseHandler,
    HandlerResult,
    error_response,
    get_string_param,
    handle_errors,
    json_response,
    log_request,
    validate_path_segment,
)

logger = logging.getLogger(__name__)

# Admin roles that can access admin endpoints
ADMIN_ROLES = {"admin", "owner"}


class AdminHandler(BaseHandler):
    """Handler for admin endpoints."""

    ROUTES = [
        "/api/admin/organizations",
        "/api/admin/users",
        "/api/admin/stats",
        "/api/admin/system/metrics",
        "/api/admin/impersonate",
        "/api/admin/revenue",
        # Nomic admin endpoints
        "/api/admin/nomic/status",
        "/api/admin/nomic/circuit-breakers",
        "/api/admin/nomic/reset",
        "/api/admin/nomic/pause",
        "/api/admin/nomic/resume",
        "/api/admin/nomic/circuit-breakers/reset",
    ]

    @staticmethod
    def can_handle(path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/admin")

    def _get_user_store(self):
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _require_admin(self, handler) -> tuple[Optional[Any], Optional[HandlerResult]]:
        """
        Verify the request is from an admin user with MFA enabled.

        SOC 2 Control: CC5-01 - Administrative access requires MFA.

        Returns:
            Tuple of (auth_context, error_response).
            If error_response is not None, return it immediately.
        """
        user_store = self._get_user_store()
        if not user_store:
            return None, error_response("Service unavailable", 503)

        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return None, error_response("Not authenticated", 401)

        # Check if user has admin role
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user or user.role not in ADMIN_ROLES:
            logger.warning(f"Non-admin user {auth_ctx.user_id} attempted admin access")
            return None, error_response("Admin access required", 403)

        # Enforce MFA for admin users (SOC 2 CC5-01)
        # Returns None if compliant, or dict with enforcement details if not
        mfa_policy_result = enforce_admin_mfa_policy(user, user_store)
        if mfa_policy_result is not None:
            reason = mfa_policy_result.get("reason", "MFA required")
            action = mfa_policy_result.get("action", "enable_mfa")
            logger.warning(f"Admin user {auth_ctx.user_id} denied: {reason} (action={action})")
            return None, error_response(
                f"Administrative access requires MFA. {reason}. "
                "Please enable MFA at /api/auth/mfa/setup",
                403,
                code="ADMIN_MFA_REQUIRED",
            )

        return auth_ctx, None

    def handle(
        self, path: str, query_params: dict, handler, method: str = "GET"
    ) -> Optional[HandlerResult]:
        """Route admin requests to appropriate methods."""
        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

        # GET routes
        if method == "GET":
            if path == "/api/admin/organizations":
                return self._list_organizations(handler, query_params)

            if path == "/api/admin/users":
                return self._list_users(handler, query_params)

            if path == "/api/admin/stats":
                return self._get_stats(handler)

            if path == "/api/admin/system/metrics":
                return self._get_system_metrics(handler)

            if path == "/api/admin/revenue":
                return self._get_revenue_stats(handler)

            # Nomic admin GET routes
            if path == "/api/admin/nomic/status":
                return self._get_nomic_status(handler)

            if path == "/api/admin/nomic/circuit-breakers":
                return self._get_nomic_circuit_breakers(handler)

        # POST routes
        if method == "POST":
            # POST /api/admin/impersonate/:user_id
            if path.startswith("/api/admin/impersonate/"):
                user_id = path.split("/")[-1]
                if not validate_path_segment(user_id, "user_id", SAFE_ID_PATTERN)[0]:
                    return error_response("Invalid user ID format", 400)
                return self._impersonate_user(handler, user_id)

            # POST /api/admin/users/:user_id/deactivate
            if "/users/" in path and path.endswith("/deactivate"):
                parts = path.split("/")
                user_id = parts[-2]
                if not validate_path_segment(user_id, "user_id", SAFE_ID_PATTERN)[0]:
                    return error_response("Invalid user ID format", 400)
                return self._deactivate_user(handler, user_id)

            # POST /api/admin/users/:user_id/activate
            if "/users/" in path and path.endswith("/activate"):
                parts = path.split("/")
                user_id = parts[-2]
                if not validate_path_segment(user_id, "user_id", SAFE_ID_PATTERN)[0]:
                    return error_response("Invalid user ID format", 400)
                return self._activate_user(handler, user_id)

            # POST /api/admin/users/:user_id/unlock
            if "/users/" in path and path.endswith("/unlock"):
                parts = path.split("/")
                user_id = parts[-2]
                if not validate_path_segment(user_id, "user_id", SAFE_ID_PATTERN)[0]:
                    return error_response("Invalid user ID format", 400)
                return self._unlock_user(handler, user_id)

            # Nomic admin POST routes
            if path == "/api/admin/nomic/reset":
                return self._reset_nomic_phase(handler)

            if path == "/api/admin/nomic/pause":
                return self._pause_nomic(handler)

            if path == "/api/admin/nomic/resume":
                return self._resume_nomic(handler)

            if path == "/api/admin/nomic/circuit-breakers/reset":
                return self._reset_nomic_circuit_breakers(handler)

        return error_response("Method not allowed", 405)

    @handle_errors("list organizations")
    def _list_organizations(self, handler, query_params: dict) -> HandlerResult:
        """List all organizations with pagination."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()

        # Parse pagination params
        limit = min(int(get_string_param(query_params, "limit", "50")), 100)
        offset = int(get_string_param(query_params, "offset", "0"))
        tier_filter = get_string_param(query_params, "tier", None)

        organizations, total = user_store.list_all_organizations(
            limit=limit,
            offset=offset,
            tier_filter=tier_filter,
        )

        return json_response(
            {
                "organizations": [org.to_dict() for org in organizations],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("list users")
    def _list_users(self, handler, query_params: dict) -> HandlerResult:
        """List all users with pagination and filtering."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()

        # Parse params
        limit = min(int(get_string_param(query_params, "limit", "50")), 100)
        offset = int(get_string_param(query_params, "offset", "0"))
        org_id = get_string_param(query_params, "org_id", None)
        role = get_string_param(query_params, "role", None)
        active_only = get_string_param(query_params, "active_only", "false").lower() == "true"

        users, total = user_store.list_all_users(
            limit=limit,
            offset=offset,
            org_id_filter=org_id,
            role_filter=role,
            active_only=active_only,
        )

        # Convert users to safe dict (exclude password hashes)
        user_dicts = []
        for user in users:
            user_dict = user.to_dict()
            # Remove sensitive fields
            user_dict.pop("password_hash", None)
            user_dict.pop("password_salt", None)
            user_dict.pop("api_key", None)
            user_dict.pop("api_key_hash", None)
            user_dicts.append(user_dict)

        return json_response(
            {
                "users": user_dicts,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("get admin stats")
    def _get_stats(self, handler) -> HandlerResult:
        """Get system-wide statistics."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()
        stats = user_store.get_admin_stats()

        return json_response({"stats": stats})

    @handle_errors("get system metrics")
    def _get_system_metrics(self, handler) -> HandlerResult:
        """Get aggregated system metrics from various sources."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        metrics: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Get user store stats
        user_store = self._get_user_store()
        if user_store:
            metrics["users"] = user_store.get_admin_stats()

        # Get debate storage stats if available
        debate_storage = self.ctx.get("debate_storage")
        if debate_storage and hasattr(debate_storage, "get_statistics"):
            try:
                metrics["debates"] = debate_storage.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get debate stats: {e}")
                metrics["debates"] = {"error": "unavailable"}

        # Get circuit breaker stats if available
        try:
            from aragora.resilience import get_circuit_breaker_status

            metrics["circuit_breakers"] = get_circuit_breaker_status()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to get circuit breaker stats: {e}")

        # Get cache stats if available
        try:
            from aragora.server.handlers.admin.cache import get_cache_stats

            metrics["cache"] = get_cache_stats()
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")

        # Get rate limit stats if available
        try:
            from aragora.server.middleware.rate_limit import get_rate_limiter

            limiter = get_rate_limiter()
            if limiter and hasattr(limiter, "get_stats"):
                metrics["rate_limits"] = limiter.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get rate limit stats: {e}")

        return json_response({"metrics": metrics})

    @handle_errors("get revenue stats")
    def _get_revenue_stats(self, handler) -> HandlerResult:
        """Get revenue and billing statistics."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()

        # Get tier distribution from stats
        stats = user_store.get_admin_stats()
        tier_distribution = stats.get("tier_distribution", {})

        # Calculate monthly recurring revenue (MRR) based on tier counts
        from aragora.billing.models import TIER_LIMITS

        mrr_cents = 0
        tier_revenue = {}
        for tier_name, count in tier_distribution.items():
            tier_limits = TIER_LIMITS.get(tier_name)
            if tier_limits:
                tier_mrr = tier_limits.price_monthly_cents * count
                tier_revenue[tier_name] = {
                    "count": count,
                    "price_cents": tier_limits.price_monthly_cents,
                    "mrr_cents": tier_mrr,
                }
                mrr_cents += tier_mrr

        return json_response(
            {
                "revenue": {
                    "mrr_cents": mrr_cents,
                    "mrr_dollars": mrr_cents / 100,
                    "arr_dollars": (mrr_cents * 12) / 100,
                    "tier_breakdown": tier_revenue,
                    "total_organizations": stats.get("total_organizations", 0),
                    "paying_organizations": sum(
                        count for tier, count in tier_distribution.items() if tier != "free"
                    ),
                }
            }
        )

    @handle_errors("impersonate user")
    @log_request("admin impersonate")
    def _impersonate_user(self, handler, target_user_id: str) -> HandlerResult:
        """
        Create an impersonation token for a user.

        This allows admins to view the system as a specific user for support.
        The token is short-lived (1 hour) and logged for audit.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()

        # Verify target user exists
        target_user = user_store.get_user_by_id(target_user_id)
        if not target_user:
            return error_response("User not found", 404)

        # Create short-lived impersonation token (1 hour)
        # Note: impersonation metadata is logged below since JWT doesn't support custom claims
        impersonation_token = create_access_token(
            user_id=target_user_id,
            email=target_user.email,
            org_id=target_user.org_id,
            role=target_user.role,
            expiry_hours=1,
        )

        # Log the impersonation for audit
        logger.info(f"Admin {auth_ctx.user_id} impersonating user {target_user_id}")

        # Record in audit log if available
        try:
            user_store.record_audit_event(
                user_id=auth_ctx.user_id,
                org_id=None,
                event_type="admin_impersonate",
                action="impersonate_user",
                resource_type="user",
                resource_id=target_user_id,
                ip_address=getattr(handler, "client_address", ("unknown",))[0],
                details={"target_email": target_user.email},
            )
        except Exception as e:
            logger.warning(f"Failed to record audit event: {e}")

        return json_response(
            {
                "token": impersonation_token,
                "expires_in": 3600,
                "target_user": {
                    "id": target_user.id,
                    "email": target_user.email,
                    "name": target_user.name,
                    "role": target_user.role,
                },
                "warning": "This token grants full access as the target user. Use responsibly.",
            }
        )

    @handle_errors("deactivate user")
    @log_request("admin deactivate user")
    def _deactivate_user(self, handler, target_user_id: str) -> HandlerResult:
        """Deactivate a user account."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()

        # Verify target user exists
        target_user = user_store.get_user_by_id(target_user_id)
        if not target_user:
            return error_response("User not found", 404)

        # Prevent deactivating yourself
        if target_user_id == auth_ctx.user_id:
            return error_response("Cannot deactivate yourself", 400)

        # Deactivate the user
        user_store.update_user(target_user_id, is_active=False)

        logger.info(f"Admin {auth_ctx.user_id} deactivated user {target_user_id}")

        return json_response(
            {
                "success": True,
                "user_id": target_user_id,
                "is_active": False,
            }
        )

    @handle_errors("activate user")
    @log_request("admin activate user")
    def _activate_user(self, handler, target_user_id: str) -> HandlerResult:
        """Activate a user account."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()

        # Verify target user exists
        target_user = user_store.get_user_by_id(target_user_id)
        if not target_user:
            return error_response("User not found", 404)

        # Activate the user
        user_store.update_user(target_user_id, is_active=True)

        logger.info(f"Admin {auth_ctx.user_id} activated user {target_user_id}")

        return json_response(
            {
                "success": True,
                "user_id": target_user_id,
                "is_active": True,
            }
        )

    @handle_errors("unlock user")
    @log_request("admin unlock user")
    def _unlock_user(self, handler, target_user_id: str) -> HandlerResult:
        """
        Unlock a user account that has been locked due to failed login attempts.

        This clears both the in-memory/Redis lockout tracker and the database
        lockout state. Use this to help users who have been locked out.

        Endpoint: POST /api/admin/users/:user_id/unlock
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()

        # Verify target user exists
        target_user = user_store.get_user_by_id(target_user_id)
        if not target_user:
            return error_response("User not found", 404)

        email = target_user.email

        # Get lockout info before clearing
        lockout_tracker = get_lockout_tracker()
        lockout_info = lockout_tracker.get_info(email=email)

        # Clear lockout tracker (in-memory/Redis)
        lockout_cleared = lockout_tracker.admin_unlock(
            email=email,
            user_id=target_user_id,
        )

        # Clear database lockout state if user store supports it
        db_cleared = False
        if hasattr(user_store, "reset_failed_login_attempts"):
            db_cleared = user_store.reset_failed_login_attempts(email)

        logger.info(
            f"Admin {auth_ctx.user_id} unlocked user {target_user_id} "
            f"(email={email}, tracker_cleared={lockout_cleared}, db_cleared={db_cleared})"
        )

        # Log audit event
        try:
            if hasattr(user_store, "log_audit_event"):
                user_store.log_audit_event(
                    action="admin_unlock_user",
                    resource_type="user",
                    resource_id=target_user_id,
                    user_id=auth_ctx.user_id,
                    metadata={
                        "target_email": email,
                        "lockout_info": lockout_info,
                    },
                    ip_address=getattr(handler, "client_address", ("unknown",))[0],
                )
        except Exception as e:
            logger.warning(f"Failed to record audit event: {e}")

        return json_response(
            {
                "success": True,
                "user_id": target_user_id,
                "email": email,
                "lockout_cleared": lockout_cleared or db_cleared,
                "previous_lockout_info": lockout_info,
                "message": f"Account lockout cleared for {email}",
            }
        )

    # =========================================================================
    # Nomic Admin Endpoints
    # =========================================================================

    def _get_nomic_dir(self):
        """Get nomic directory from context or default."""
        return self.ctx.get("nomic_dir", ".nomic")

    @handle_errors("get nomic status")
    def _get_nomic_status(self, handler) -> HandlerResult:
        """
        Get detailed nomic loop status including state machine state.

        Returns comprehensive status for admin monitoring and intervention.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        import json
        from pathlib import Path

        nomic_dir = Path(self._get_nomic_dir())
        errors: list[str] = []
        status: dict[str, Any] = {
            "running": False,
            "current_phase": None,
            "cycle_id": None,
            "state_machine": None,
            "metrics": None,
            "circuit_breakers": None,
            "last_checkpoint": None,
            "errors": errors,
        }

        # Read state file
        state_file = nomic_dir / "nomic_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state_data = json.load(f)
                status["running"] = state_data.get("running", False)
                status["current_phase"] = state_data.get("phase")
                status["cycle_id"] = state_data.get("cycle_id")
                status["state_machine"] = state_data
            except Exception as e:
                errors.append(f"Failed to read state: {e}")

        # Get metrics
        try:
            from aragora.nomic.metrics import (
                check_stuck_phases,
                get_nomic_metrics_summary,
            )

            status["metrics"] = get_nomic_metrics_summary()
            stuck_info = check_stuck_phases(max_idle_seconds=1800)
            status["stuck_detection"] = stuck_info
        except ImportError:
            errors.append("Metrics module not available")
        except Exception as e:
            errors.append(f"Failed to get metrics: {e}")

        # Get circuit breaker status
        try:
            from aragora.nomic.recovery import CircuitBreakerRegistry

            registry = CircuitBreakerRegistry()
            status["circuit_breakers"] = {
                "open": registry.all_open(),
                "details": registry.to_dict(),
            }
        except Exception as e:
            errors.append(f"Failed to get circuit breakers: {e}")

        # Find latest checkpoint
        checkpoint_dir = nomic_dir / "checkpoints"
        if checkpoint_dir.exists():
            try:
                from aragora.nomic.checkpoints import list_checkpoints

                checkpoints = list_checkpoints(str(checkpoint_dir))
                if checkpoints:
                    status["last_checkpoint"] = checkpoints[0]
            except Exception as e:
                errors.append(f"Failed to list checkpoints: {e}")

        return json_response(status)

    @handle_errors("get nomic circuit breakers")
    def _get_nomic_circuit_breakers(self, handler) -> HandlerResult:
        """Get detailed circuit breaker status for the nomic loop."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        try:
            from aragora.nomic.recovery import CircuitBreakerRegistry

            registry = CircuitBreakerRegistry()
            return json_response(
                {
                    "circuit_breakers": registry.to_dict(),
                    "open_circuits": registry.all_open(),
                    "total_count": len(registry._breakers),
                }
            )
        except ImportError:
            return error_response("Nomic recovery module not available", 503)
        except Exception as e:
            logger.error(f"Failed to get circuit breakers: {e}", exc_info=True)
            return error_response(f"Failed to get circuit breakers: {e}", 500)

    @handle_errors("reset nomic phase")
    @log_request("admin reset nomic")
    def _reset_nomic_phase(self, handler) -> HandlerResult:
        """
        Reset nomic loop to a specific phase.

        This is a recovery action for stuck or failed nomic cycles.
        Supports resetting to: idle, context, debate, design, implement, verify, commit.

        Request body:
            {
                "target_phase": "context",  # Phase to reset to
                "clear_errors": true,       # Clear error history
                "reason": "Manual recovery" # Audit reason
            }
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        import json
        from pathlib import Path

        # Parse request body
        try:
            body = getattr(handler, "request_body", b"{}")
            if isinstance(body, bytes):
                body = body.decode("utf-8")
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return error_response("Invalid JSON body", 400)

        target_phase = data.get("target_phase", "idle").lower()
        clear_errors = data.get("clear_errors", False)
        reason = data.get("reason", "Admin manual reset")

        # Validate target phase
        valid_phases = {"idle", "context", "debate", "design", "implement", "verify", "commit"}
        if target_phase not in valid_phases:
            return error_response(
                f"Invalid target phase. Must be one of: {', '.join(valid_phases)}", 400
            )

        nomic_dir = Path(self._get_nomic_dir())
        state_file = nomic_dir / "nomic_state.json"

        # Read current state
        current_state = {}
        if state_file.exists():
            try:
                with open(state_file) as f:
                    current_state = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read nomic state file: {e}")
                # Continue with empty state

        # Update state
        from datetime import datetime

        new_state = {
            "phase": target_phase,
            "running": target_phase != "idle",
            "cycle_id": current_state.get(
                "cycle_id", "reset-" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
            ),
            "last_update": datetime.utcnow().isoformat() + "Z",
            "reset_by": auth_ctx.user_id,
            "reset_reason": reason,
            "previous_phase": current_state.get("phase"),
            "errors": [] if clear_errors else current_state.get("errors", []),
        }

        # Ensure directory exists
        nomic_dir.mkdir(parents=True, exist_ok=True)

        # Write new state
        try:
            with open(state_file, "w") as f:
                json.dump(new_state, f, indent=2)
        except Exception as e:
            return error_response(f"Failed to write state: {e}", 500)

        # Track metric
        try:
            from aragora.nomic.metrics import track_phase_transition

            track_phase_transition(
                from_phase=current_state.get("phase", "unknown"),
                to_phase=target_phase,
                cycle_id=new_state["cycle_id"],
            )
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"Metrics tracking skipped: {e}")

        logger.info(f"Admin {auth_ctx.user_id} reset nomic phase to {target_phase}: {reason}")

        return json_response(
            {
                "success": True,
                "previous_phase": current_state.get("phase"),
                "new_phase": target_phase,
                "cycle_id": new_state["cycle_id"],
                "message": f"Nomic reset to {target_phase}",
            }
        )

    @handle_errors("pause nomic")
    @log_request("admin pause nomic")
    def _pause_nomic(self, handler) -> HandlerResult:
        """
        Pause the nomic loop for manual intervention.

        This sets the state to 'paused' and prevents further phase transitions
        until resumed.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        import json
        from datetime import datetime
        from pathlib import Path

        # Parse reason from body
        try:
            body = getattr(handler, "request_body", b"{}")
            if isinstance(body, bytes):
                body = body.decode("utf-8")
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        reason = data.get("reason", "Admin requested pause")

        nomic_dir = Path(self._get_nomic_dir())
        state_file = nomic_dir / "nomic_state.json"

        # Read current state
        current_state = {}
        if state_file.exists():
            try:
                with open(state_file) as f:
                    current_state = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read nomic state file: {e}")
                # Continue with empty state

        # Update state
        new_state = {
            **current_state,
            "phase": "paused",
            "running": False,
            "paused_at": datetime.utcnow().isoformat() + "Z",
            "paused_by": auth_ctx.user_id,
            "pause_reason": reason,
            "previous_phase": current_state.get("phase"),
            "last_update": datetime.utcnow().isoformat() + "Z",
        }

        nomic_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(state_file, "w") as f:
                json.dump(new_state, f, indent=2)
        except Exception as e:
            return error_response(f"Failed to pause nomic: {e}", 500)

        logger.info(f"Admin {auth_ctx.user_id} paused nomic: {reason}")

        return json_response(
            {
                "success": True,
                "status": "paused",
                "previous_phase": current_state.get("phase"),
                "paused_by": auth_ctx.user_id,
                "reason": reason,
            }
        )

    @handle_errors("resume nomic")
    @log_request("admin resume nomic")
    def _resume_nomic(self, handler) -> HandlerResult:
        """
        Resume a paused nomic loop.

        Resumes from the phase that was active before the pause, or from
        a specified target phase.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        import json
        from datetime import datetime
        from pathlib import Path

        # Parse body
        try:
            body = getattr(handler, "request_body", b"{}")
            if isinstance(body, bytes):
                body = body.decode("utf-8")
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        target_phase = data.get("target_phase")  # Optional override

        nomic_dir = Path(self._get_nomic_dir())
        state_file = nomic_dir / "nomic_state.json"

        # Read current state
        current_state = {}
        if state_file.exists():
            try:
                with open(state_file) as f:
                    current_state = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read nomic state file: {e}")
                # Continue with empty state

        if current_state.get("phase") != "paused":
            return error_response("Nomic is not currently paused", 400)

        # Determine resume phase
        resume_phase = target_phase or current_state.get("previous_phase", "context")

        # Update state
        new_state = {
            **current_state,
            "phase": resume_phase,
            "running": True,
            "resumed_at": datetime.utcnow().isoformat() + "Z",
            "resumed_by": auth_ctx.user_id,
            "last_update": datetime.utcnow().isoformat() + "Z",
        }
        # Clean up pause-specific fields
        new_state.pop("paused_at", None)
        new_state.pop("paused_by", None)
        new_state.pop("pause_reason", None)

        try:
            with open(state_file, "w") as f:
                json.dump(new_state, f, indent=2)
        except Exception as e:
            return error_response(f"Failed to resume nomic: {e}", 500)

        logger.info(f"Admin {auth_ctx.user_id} resumed nomic to phase {resume_phase}")

        return json_response(
            {
                "success": True,
                "status": "resumed",
                "phase": resume_phase,
                "resumed_by": auth_ctx.user_id,
            }
        )

    @handle_errors("reset nomic circuit breakers")
    @log_request("admin reset circuit breakers")
    def _reset_nomic_circuit_breakers(self, handler) -> HandlerResult:
        """
        Reset all nomic circuit breakers.

        This clears the failure counts and closes all open circuit breakers,
        allowing the nomic loop to retry previously failing operations.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        try:
            from aragora.nomic.recovery import CircuitBreakerRegistry

            registry = CircuitBreakerRegistry()
            open_before = registry.all_open()
            registry.reset_all()

            # Also update the metrics
            try:
                from aragora.nomic.metrics import update_circuit_breaker_count

                update_circuit_breaker_count(0)
            except Exception:
                pass  # Best effort

            logger.info(
                f"Admin {auth_ctx.user_id} reset circuit breakers. Previously open: {open_before}"
            )

            return json_response(
                {
                    "success": True,
                    "previously_open": open_before,
                    "message": "All circuit breakers have been reset",
                }
            )
        except ImportError:
            return error_response("Nomic recovery module not available", 503)
        except Exception as e:
            logger.error(f"Failed to reset circuit breakers: {e}", exc_info=True)
            return error_response(f"Failed to reset circuit breakers: {e}", 500)


__all__ = ["AdminHandler"]
