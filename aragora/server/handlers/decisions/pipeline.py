"""Decision Pipeline HTTP handlers.

Endpoints for the gold path: debate → plan → approve → execute → verify → learn.

All endpoints require authentication. Write operations require
'decision:manage' permission.

Endpoints:
    POST   /api/v1/decisions/plans              - Create plan from debate result
    GET    /api/v1/decisions/plans               - List plans
    GET    /api/v1/decisions/plans/{plan_id}     - Get plan details
    POST   /api/v1/decisions/plans/{plan_id}/approve - Approve plan
    POST   /api/v1/decisions/plans/{plan_id}/reject  - Reject plan
    POST   /api/v1/decisions/plans/{plan_id}/execute  - Execute approved plan
    GET    /api/v1/decisions/plans/{plan_id}/outcome  - Get execution outcome

Stability: ALPHA
"""

from __future__ import annotations

import logging
from typing import Any

from aiohttp import web

from aragora.server.handlers.utils.auth import (
    get_auth_context,
    UnauthorizedError,
    ForbiddenError,
)
from aragora.server.handlers.utils import parse_json_body
from aragora.rbac.checker import get_permission_checker
from aragora.resilience import get_circuit_breaker

logger = logging.getLogger(__name__)

# Circuit breaker for pipeline operations
_pipeline_cb = None


def _get_circuit_breaker():  # type: ignore[no-untyped-def]
    global _pipeline_cb
    if _pipeline_cb is None:
        _pipeline_cb = get_circuit_breaker(
            "decision_pipeline",
            failure_threshold=5,
            cooldown_seconds=30,
        )
    return _pipeline_cb


# RBAC permission keys
DECISION_READ_PERMISSION = "decision:read"
DECISION_MANAGE_PERMISSION = "decision:manage"


def _is_admin(auth_ctx: object) -> bool:
    roles = getattr(auth_ctx, "roles", []) or []
    return any(role in ("admin", "owner") for role in roles)


def _check_read_permission(auth_ctx: object) -> None:
    """Check read permission, allowing legacy access when permissions unset."""
    perms = getattr(auth_ctx, "permissions", set()) or set()
    if perms and not _is_admin(auth_ctx):
        checker = get_permission_checker()
        decision = checker.check_permission(auth_ctx, DECISION_READ_PERMISSION)
        if not decision.allowed:
            raise ForbiddenError(f"Permission denied: {decision.reason}")


def _check_manage_permission(auth_ctx: object) -> None:
    """Check manage permission, allowing legacy access when permissions unset."""
    perms = getattr(auth_ctx, "permissions", set()) or set()
    if perms and not _is_admin(auth_ctx):
        checker = get_permission_checker()
        decision = checker.check_permission(auth_ctx, DECISION_MANAGE_PERMISSION)
        if not decision.allowed:
            raise ForbiddenError(f"Permission denied: {decision.reason}")


class DecisionPipelineHandler:
    """HTTP handlers for the decision pipeline (gold path)."""

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    # -----------------------------------------------------------------
    # POST /api/v1/decisions/plans - Create plan from debate
    # -----------------------------------------------------------------

    @staticmethod
    async def create_plan(request: web.Request) -> web.Response:
        """Create a DecisionPlan from a completed debate.

        Body:
            debate_id: str - ID of the completed debate
            budget_limit_usd: float (optional) - Budget cap
            approval_mode: str (optional) - "always"|"risk_based"|"confidence_based"|"never"
            max_auto_risk: str (optional) - "low"|"medium"|"high"|"critical"
            metadata: dict (optional) - Extra metadata
        """
        try:
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Decision pipeline temporarily unavailable"},
                    status=503,
                )

            auth_ctx = await get_auth_context(request, require_auth=True)
            _check_manage_permission(auth_ctx)

            data, err = await parse_json_body(request, context="create_plan")
            if err:
                return err

            debate_id = data.get("debate_id")
            if not debate_id:
                return web.json_response(
                    {"success": False, "error": "debate_id is required"},
                    status=400,
                )

            # Load debate result
            debate_result = await _load_debate_result(debate_id, request)
            if debate_result is None:
                return web.json_response(
                    {"success": False, "error": f"Debate {debate_id} not found"},
                    status=404,
                )

            # Parse options
            from aragora.pipeline.decision_plan import (
                ApprovalMode,
                DecisionPlanFactory,
            )
            from aragora.pipeline.risk_register import RiskLevel

            approval_mode_str = data.get("approval_mode", "risk_based")
            try:
                approval_mode = ApprovalMode(approval_mode_str)
            except ValueError:
                approval_mode = ApprovalMode.RISK_BASED

            max_auto_risk_str = data.get("max_auto_risk", "low")
            try:
                max_auto_risk = RiskLevel(max_auto_risk_str)
            except ValueError:
                max_auto_risk = RiskLevel.LOW

            budget_limit = data.get("budget_limit_usd")
            if budget_limit is not None:
                try:
                    budget_limit = float(budget_limit)
                except (TypeError, ValueError):
                    budget_limit = None

            # Build the plan
            plan = DecisionPlanFactory.from_debate_result(
                debate_result,
                budget_limit_usd=budget_limit,
                approval_mode=approval_mode,
                max_auto_risk=max_auto_risk,
                metadata=data.get("metadata") or {},
            )

            # Store it
            from aragora.pipeline.executor import store_plan

            store_plan(plan)

            logger.info(
                "Created decision plan %s from debate %s (status=%s)",
                plan.id,
                debate_id,
                plan.status.value,
            )

            return web.json_response(
                {
                    "success": True,
                    "plan": plan.to_dict(),
                },
                status=201,
            )

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error("Error creating decision plan: %s", e)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # -----------------------------------------------------------------
    # GET /api/v1/decisions/plans - List plans
    # -----------------------------------------------------------------

    @staticmethod
    async def list_plans(request: web.Request) -> web.Response:
        """List decision plans with optional status filter.

        Query params:
            status: str (optional) - Filter by status
            limit: int (optional, default 50) - Max results
        """
        try:
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Decision pipeline temporarily unavailable"},
                    status=503,
                )

            auth_ctx = await get_auth_context(request, require_auth=True)
            _check_read_permission(auth_ctx)

            from aragora.pipeline.decision_plan import PlanStatus
            from aragora.pipeline.executor import list_plans

            status_str = request.query.get("status")
            status_filter = None
            if status_str:
                try:
                    status_filter = PlanStatus(status_str)
                except ValueError:
                    return web.json_response(
                        {"success": False, "error": f"Invalid status: {status_str}"},
                        status=400,
                    )

            limit = 50
            limit_str = request.query.get("limit")
            if limit_str:
                try:
                    limit = min(200, max(1, int(limit_str)))
                except ValueError:
                    pass

            plans = list_plans(status=status_filter, limit=limit)

            return web.json_response(
                {
                    "success": True,
                    "plans": [p.to_dict() for p in plans],
                    "count": len(plans),
                }
            )

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error("Error listing decision plans: %s", e)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # -----------------------------------------------------------------
    # GET /api/v1/decisions/plans/{plan_id} - Get plan details
    # -----------------------------------------------------------------

    @staticmethod
    async def get_plan(request: web.Request) -> web.Response:
        """Get details of a specific decision plan."""
        plan_id = request.match_info.get("plan_id")
        try:
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Decision pipeline temporarily unavailable"},
                    status=503,
                )

            auth_ctx = await get_auth_context(request, require_auth=True)
            _check_read_permission(auth_ctx)

            from aragora.pipeline.executor import get_plan, get_outcome

            plan = get_plan(plan_id)
            if not plan:
                return web.json_response(
                    {"success": False, "error": "Plan not found"},
                    status=404,
                )

            result: dict[str, Any] = {
                "success": True,
                "plan": plan.to_dict(),
            }

            # Include outcome if execution completed
            outcome = get_outcome(plan_id)
            if outcome:
                result["outcome"] = outcome.to_dict()

            return web.json_response(result)

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error("Error getting decision plan: %s", e)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # -----------------------------------------------------------------
    # POST /api/v1/decisions/plans/{plan_id}/approve - Approve plan
    # -----------------------------------------------------------------

    @staticmethod
    async def approve_plan(request: web.Request) -> web.Response:
        """Approve a decision plan for execution.

        Body:
            reason: str (optional) - Approval reason
            conditions: list[str] (optional) - Conditions for approval
        """
        plan_id = request.match_info.get("plan_id")
        try:
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Decision pipeline temporarily unavailable"},
                    status=503,
                )

            auth_ctx = await get_auth_context(request, require_auth=True)
            _check_manage_permission(auth_ctx)

            from aragora.pipeline.executor import get_plan, store_plan
            from aragora.pipeline.decision_plan import PlanStatus

            plan = get_plan(plan_id)
            if not plan:
                return web.json_response(
                    {"success": False, "error": "Plan not found"},
                    status=404,
                )

            if plan.status not in (PlanStatus.CREATED, PlanStatus.AWAITING_APPROVAL):
                return web.json_response(
                    {
                        "success": False,
                        "error": f"Plan cannot be approved in status: {plan.status.value}",
                    },
                    status=409,
                )

            data, err = await parse_json_body(request, context="approve_plan")
            if err:
                return err

            approver_id = getattr(auth_ctx, "user_id", "unknown")
            reason = data.get("reason", "")
            conditions = data.get("conditions", [])

            plan.approve(
                approver_id=approver_id,
                reason=reason,
                conditions=conditions,
            )
            store_plan(plan)

            logger.info("Plan %s approved by %s", plan_id, approver_id)

            return web.json_response(
                {
                    "success": True,
                    "plan": {
                        "id": plan.id,
                        "status": plan.status.value,
                        "approval_record": plan.approval_record.to_dict()
                        if plan.approval_record
                        else None,
                    },
                }
            )

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error("Error approving plan: %s", e)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # -----------------------------------------------------------------
    # POST /api/v1/decisions/plans/{plan_id}/reject - Reject plan
    # -----------------------------------------------------------------

    @staticmethod
    async def reject_plan(request: web.Request) -> web.Response:
        """Reject a decision plan.

        Body:
            reason: str - Reason for rejection
        """
        plan_id = request.match_info.get("plan_id")
        try:
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Decision pipeline temporarily unavailable"},
                    status=503,
                )

            auth_ctx = await get_auth_context(request, require_auth=True)
            _check_manage_permission(auth_ctx)

            from aragora.pipeline.executor import get_plan, store_plan
            from aragora.pipeline.decision_plan import PlanStatus

            plan = get_plan(plan_id)
            if not plan:
                return web.json_response(
                    {"success": False, "error": "Plan not found"},
                    status=404,
                )

            if plan.status not in (PlanStatus.CREATED, PlanStatus.AWAITING_APPROVAL):
                return web.json_response(
                    {
                        "success": False,
                        "error": f"Plan cannot be rejected in status: {plan.status.value}",
                    },
                    status=409,
                )

            data, err = await parse_json_body(request, context="reject_plan")
            if err:
                return err

            approver_id = getattr(auth_ctx, "user_id", "unknown")
            reason = data.get("reason", "No reason provided")

            plan.reject(approver_id=approver_id, reason=reason)
            store_plan(plan)

            logger.info("Plan %s rejected by %s: %s", plan_id, approver_id, reason)

            return web.json_response(
                {
                    "success": True,
                    "plan": {
                        "id": plan.id,
                        "status": plan.status.value,
                        "approval_record": plan.approval_record.to_dict()
                        if plan.approval_record
                        else None,
                    },
                }
            )

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error("Error rejecting plan: %s", e)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # -----------------------------------------------------------------
    # POST /api/v1/decisions/plans/{plan_id}/execute - Execute plan
    # -----------------------------------------------------------------

    @staticmethod
    async def execute_plan(request: web.Request) -> web.Response:
        """Execute an approved decision plan.

        Triggers asynchronous workflow execution. Returns immediately
        with the executing plan status. Poll GET /plans/{plan_id} for
        completion.
        """
        plan_id = request.match_info.get("plan_id")
        try:
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Decision pipeline temporarily unavailable"},
                    status=503,
                )

            auth_ctx = await get_auth_context(request, require_auth=True)
            _check_manage_permission(auth_ctx)

            from aragora.pipeline.executor import get_plan, PlanExecutor

            plan = get_plan(plan_id)
            if not plan:
                return web.json_response(
                    {"success": False, "error": "Plan not found"},
                    status=404,
                )

            executor = PlanExecutor()

            try:
                outcome = await executor.execute(plan)
            except ValueError as e:
                return web.json_response(
                    {"success": False, "error": str(e)},
                    status=409,
                )

            logger.info(
                "Plan %s execution completed: success=%s",
                plan_id,
                outcome.success,
            )

            return web.json_response(
                {
                    "success": True,
                    "plan": plan.to_dict(),
                    "outcome": outcome.to_dict(),
                }
            )

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error("Error executing plan: %s", e)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # -----------------------------------------------------------------
    # GET /api/v1/decisions/plans/{plan_id}/outcome - Get outcome
    # -----------------------------------------------------------------

    @staticmethod
    async def get_outcome(request: web.Request) -> web.Response:
        """Get the execution outcome for a completed plan."""
        plan_id = request.match_info.get("plan_id")
        try:
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Decision pipeline temporarily unavailable"},
                    status=503,
                )

            auth_ctx = await get_auth_context(request, require_auth=True)
            _check_read_permission(auth_ctx)

            from aragora.pipeline.executor import get_plan, get_outcome

            plan = get_plan(plan_id)
            if not plan:
                return web.json_response(
                    {"success": False, "error": "Plan not found"},
                    status=404,
                )

            outcome = get_outcome(plan_id)
            if not outcome:
                return web.json_response(
                    {"success": False, "error": "No outcome recorded yet"},
                    status=404,
                )

            return web.json_response(
                {
                    "success": True,
                    "outcome": outcome.to_dict(),
                }
            )

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error("Error getting plan outcome: %s", e)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # -----------------------------------------------------------------
    # Route registration
    # -----------------------------------------------------------------

    @staticmethod
    def register_routes(app: web.Application, prefix: str = "/api/v1/decisions") -> None:
        """Register decision pipeline routes with the application."""
        app.router.add_post(
            f"{prefix}/plans",
            DecisionPipelineHandler.create_plan,
        )
        app.router.add_get(
            f"{prefix}/plans",
            DecisionPipelineHandler.list_plans,
        )
        app.router.add_get(
            f"{prefix}/plans/{{plan_id}}",
            DecisionPipelineHandler.get_plan,
        )
        app.router.add_post(
            f"{prefix}/plans/{{plan_id}}/approve",
            DecisionPipelineHandler.approve_plan,
        )
        app.router.add_post(
            f"{prefix}/plans/{{plan_id}}/reject",
            DecisionPipelineHandler.reject_plan,
        )
        app.router.add_post(
            f"{prefix}/plans/{{plan_id}}/execute",
            DecisionPipelineHandler.execute_plan,
        )
        app.router.add_get(
            f"{prefix}/plans/{{plan_id}}/outcome",
            DecisionPipelineHandler.get_outcome,
        )


# ---------------------------------------------------------------------------
# Helper: Load a debate result by ID
# ---------------------------------------------------------------------------


async def _load_debate_result(debate_id: str, request: web.Request) -> Any | None:
    """Load a DebateResult by debate ID from available stores.

    Tries multiple sources in order:
    1. Trace files on disk
    2. Replay events
    3. Decision cache
    4. Storage backend
    """

    # Try trace files
    try:
        from aragora.debate.traces import DebateTrace
        import os
        from pathlib import Path

        nomic_dir = request.app.get("nomic_dir") or os.environ.get("ARAGORA_NOMIC_DIR")
        if nomic_dir:
            trace_path = Path(nomic_dir) / "traces" / f"{debate_id}.json"
            if trace_path.exists():
                trace = DebateTrace.load(trace_path)
                return trace.to_debate_result()
    except Exception as e:
        logger.debug("Failed to load trace for %s: %s", debate_id, e)

    # Try storage backend
    try:
        storage = request.app.get("debate_storage")
        if storage:
            result = await storage.get_result(debate_id)
            if result:
                return result
    except Exception as e:
        logger.debug("Failed to load from storage for %s: %s", debate_id, e)

    # Try decision cache
    try:
        from aragora.core.decision_cache import get_decision_cache

        cache = get_decision_cache()
        if cache:
            result = cache.get(debate_id)
            if result:
                return result
    except Exception as e:
        logger.debug("Failed to load from cache for %s: %s", debate_id, e)

    return None
