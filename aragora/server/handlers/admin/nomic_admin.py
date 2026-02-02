"""
Admin Nomic Loop Control Endpoints.

Provides administrative endpoints for managing the Nomic self-improvement loop.
All endpoints require admin or owner role with MFA enabled.

Endpoints:
- GET /api/v1/admin/nomic/status - Get detailed nomic status
- GET /api/v1/admin/nomic/circuit-breakers - Get circuit breaker status
- POST /api/v1/admin/nomic/reset - Reset nomic to a specific phase
- POST /api/v1/admin/nomic/pause - Pause the nomic loop
- POST /api/v1/admin/nomic/resume - Resume the nomic loop
- POST /api/v1/admin/nomic/circuit-breakers/reset - Reset all circuit breakers
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from aragora.audit.unified import audit_admin

from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
)
from ..openapi_decorator import api_endpoint

if TYPE_CHECKING:
    from aragora.auth.context import AuthorizationContext

logger = logging.getLogger(__name__)

# RBAC Permission Constants for Nomic Operations
PERM_ADMIN_NOMIC_WRITE = "admin:nomic:write"  # Reset/pause/resume nomic loop
PERM_ADMIN_SYSTEM_WRITE = "admin:system:write"  # Reset circuit breakers


class NomicAdminMixin:
    """
    Mixin providing nomic loop control endpoints for admin.

    This mixin requires the following attributes from the base class:
    - ctx: dict[str, Any] - Server context
    - _require_admin(handler) -> tuple[AuthContext | None, HandlerResult | None]
    - _check_rbac_permission(auth_ctx, permission, resource_id=None) -> HandlerResult | None
    """

    # Type stubs for methods expected from host class (BaseHandler)
    ctx: dict[str, Any]
    _require_admin: Callable[[Any], tuple["AuthorizationContext | None", HandlerResult | None]]
    _check_rbac_permission: Callable[..., HandlerResult | None]

    def _get_nomic_dir(self) -> str:
        """Get nomic directory from context or default."""
        nomic_dir = self.ctx.get("nomic_dir", ".nomic")
        return str(nomic_dir) if nomic_dir else ".nomic"

    @api_endpoint(
        method="GET",
        path="/api/v1/admin/nomic/status",
        summary="Get detailed nomic loop status",
        tags=["Admin"],
        responses={
            "200": {
                "description": "Nomic loop status including state machine, metrics, and circuit breakers",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "running": {"type": "boolean"},
                                "current_phase": {"type": "string"},
                                "cycle_id": {"type": "string"},
                                "state_machine": {"type": "object"},
                                "metrics": {"type": "object"},
                                "circuit_breakers": {"type": "object"},
                                "last_checkpoint": {"type": "object"},
                                "errors": {"type": "array", "items": {"type": "string"}},
                            },
                            "additionalProperties": True,
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
        },
    )
    @handle_errors("get nomic status")
    def _get_nomic_status(self, handler: Any) -> HandlerResult:
        """
        Get detailed nomic loop status including state machine state.
        Requires admin:nomic:read permission.

        Returns comprehensive status for admin monitoring and intervention.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check RBAC permission
        perm_err = self._check_rbac_permission(auth_ctx, "admin.nomic.read")
        if perm_err:
            return perm_err

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

    @api_endpoint(
        method="GET",
        path="/api/v1/admin/nomic/circuit-breakers",
        summary="Get nomic circuit breaker status",
        tags=["Admin"],
        responses={
            "200": {
                "description": "Circuit breaker details and open circuit list",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "circuit_breakers": {"type": "object"},
                                "open_circuits": {"type": "array", "items": {"type": "string"}},
                                "total_count": {"type": "integer"},
                            },
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
            "503": {"description": "Nomic recovery module not available"},
        },
    )
    @handle_errors("get nomic circuit breakers")
    def _get_nomic_circuit_breakers(self, handler: Any) -> HandlerResult:
        """Get detailed circuit breaker status for the nomic loop.
        Requires admin:nomic:read permission.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check RBAC permission
        perm_err = self._check_rbac_permission(auth_ctx, "admin.nomic.read")
        if perm_err:
            return perm_err

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

    @api_endpoint(
        method="POST",
        path="/api/v1/admin/nomic/reset",
        summary="Reset nomic loop to a specific phase",
        tags=["Admin"],
        request_body={
            "description": "Reset parameters",
            "required": False,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "target_phase": {
                                "type": "string",
                                "enum": [
                                    "idle",
                                    "context",
                                    "debate",
                                    "design",
                                    "implement",
                                    "verify",
                                    "commit",
                                ],
                            },
                            "clear_errors": {"type": "boolean", "default": False},
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        },
        responses={
            "200": {
                "description": "Nomic phase reset successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "previous_phase": {"type": "string"},
                                "new_phase": {"type": "string"},
                                "cycle_id": {"type": "string"},
                                "message": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "400": {"description": "Invalid target phase or JSON body"},
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
        },
    )
    @handle_errors("reset nomic phase")
    @log_request("admin reset nomic")
    def _reset_nomic_phase(self, handler: Any) -> HandlerResult:
        """
        Reset nomic loop to a specific phase.

        This is a recovery action for stuck or failed nomic cycles.
        Supports resetting to: idle, context, debate, design, implement, verify, commit.

        Requires permission: admin:nomic:write

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

        # Check RBAC permission (CRITICAL: admin:nomic:write)
        perm_err = self._check_rbac_permission(auth_ctx, PERM_ADMIN_NOMIC_WRITE)
        if perm_err:
            return perm_err

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
        new_state = {
            "phase": target_phase,
            "running": target_phase != "idle",
            "cycle_id": current_state.get(
                "cycle_id", "reset-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            ),
            "last_update": datetime.now(timezone.utc).isoformat() + "Z",
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
        audit_admin(
            admin_id=auth_ctx.user_id,
            action="reset_nomic_phase",
            target_type="nomic",
            target_id=new_state["cycle_id"],
            previous_phase=current_state.get("phase"),
            new_phase=target_phase,
            reason=reason,
        )

        return json_response(
            {
                "success": True,
                "previous_phase": current_state.get("phase"),
                "new_phase": target_phase,
                "cycle_id": new_state["cycle_id"],
                "message": f"Nomic reset to {target_phase}",
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/admin/nomic/pause",
        summary="Pause the nomic loop",
        tags=["Admin"],
        request_body={
            "description": "Pause parameters",
            "required": False,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        },
        responses={
            "200": {
                "description": "Nomic loop paused successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "status": {"type": "string"},
                                "previous_phase": {"type": "string"},
                                "paused_by": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
        },
    )
    @handle_errors("pause nomic")
    @log_request("admin pause nomic")
    def _pause_nomic(self, handler: Any) -> HandlerResult:
        """
        Pause the nomic loop for manual intervention.

        This sets the state to 'paused' and prevents further phase transitions
        until resumed.

        Requires permission: admin:nomic:write
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check RBAC permission (CRITICAL: admin:nomic:write)
        perm_err = self._check_rbac_permission(auth_ctx, PERM_ADMIN_NOMIC_WRITE)
        if perm_err:
            return perm_err

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
            "paused_at": datetime.now(timezone.utc).isoformat() + "Z",
            "paused_by": auth_ctx.user_id,
            "pause_reason": reason,
            "previous_phase": current_state.get("phase"),
            "last_update": datetime.now(timezone.utc).isoformat() + "Z",
        }

        nomic_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(state_file, "w") as f:
                json.dump(new_state, f, indent=2)
        except Exception as e:
            return error_response(f"Failed to pause nomic: {e}", 500)

        logger.info(f"Admin {auth_ctx.user_id} paused nomic: {reason}")
        audit_admin(
            admin_id=auth_ctx.user_id,
            action="pause_nomic",
            target_type="nomic",
            target_id=current_state.get("cycle_id", "unknown"),
            previous_phase=current_state.get("phase"),
            reason=reason,
        )

        return json_response(
            {
                "success": True,
                "status": "paused",
                "previous_phase": current_state.get("phase"),
                "paused_by": auth_ctx.user_id,
                "reason": reason,
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/admin/nomic/resume",
        summary="Resume a paused nomic loop",
        tags=["Admin"],
        request_body={
            "description": "Resume parameters",
            "required": False,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "target_phase": {"type": "string"},
                        },
                    },
                },
            },
        },
        responses={
            "200": {
                "description": "Nomic loop resumed successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "status": {"type": "string"},
                                "phase": {"type": "string"},
                                "resumed_by": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "400": {"description": "Nomic is not currently paused"},
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
        },
    )
    @handle_errors("resume nomic")
    @log_request("admin resume nomic")
    def _resume_nomic(self, handler: Any) -> HandlerResult:
        """
        Resume a paused nomic loop.

        Resumes from the phase that was active before the pause, or from
        a specified target phase.

        Requires permission: admin:nomic:write
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check RBAC permission (CRITICAL: admin:nomic:write)
        perm_err = self._check_rbac_permission(auth_ctx, PERM_ADMIN_NOMIC_WRITE)
        if perm_err:
            return perm_err

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
            "resumed_at": datetime.now(timezone.utc).isoformat() + "Z",
            "resumed_by": auth_ctx.user_id,
            "last_update": datetime.now(timezone.utc).isoformat() + "Z",
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
        audit_admin(
            admin_id=auth_ctx.user_id,
            action="resume_nomic",
            target_type="nomic",
            target_id=new_state.get("cycle_id", "unknown"),
            resume_phase=resume_phase,
        )

        return json_response(
            {
                "success": True,
                "status": "resumed",
                "phase": resume_phase,
                "resumed_by": auth_ctx.user_id,
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/admin/nomic/circuit-breakers/reset",
        summary="Reset all nomic circuit breakers",
        tags=["Admin"],
        responses={
            "200": {
                "description": "All circuit breakers reset successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "previously_open": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "message": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
            "503": {"description": "Nomic recovery module not available"},
        },
    )
    @handle_errors("reset nomic circuit breakers")
    @log_request("admin reset circuit breakers")
    def _reset_nomic_circuit_breakers(self, handler: Any) -> HandlerResult:
        """
        Reset all nomic circuit breakers.

        This clears the failure counts and closes all open circuit breakers,
        allowing the nomic loop to retry previously failing operations.

        Requires permission: admin:system:write
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check RBAC permission (CRITICAL: admin:system:write)
        perm_err = self._check_rbac_permission(auth_ctx, PERM_ADMIN_SYSTEM_WRITE)
        if perm_err:
            return perm_err

        try:
            from aragora.nomic.recovery import CircuitBreakerRegistry

            registry = CircuitBreakerRegistry()
            open_before = registry.all_open()
            registry.reset_all()

            # Also update the metrics
            try:
                from aragora.nomic.metrics import update_circuit_breaker_count

                update_circuit_breaker_count(0)
            except ImportError:
                logger.debug("nomic.metrics not available for circuit breaker count update")
            except Exception as e:
                logger.debug(f"Best effort metrics update failed: {type(e).__name__}")

            logger.info(
                f"Admin {auth_ctx.user_id} reset circuit breakers. Previously open: {open_before}"
            )
            audit_admin(
                admin_id=auth_ctx.user_id,
                action="reset_circuit_breakers",
                target_type="nomic",
                target_id="circuit_breakers",
                previously_open=open_before,
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


__all__ = ["NomicAdminMixin", "PERM_ADMIN_NOMIC_WRITE", "PERM_ADMIN_SYSTEM_WRITE"]
