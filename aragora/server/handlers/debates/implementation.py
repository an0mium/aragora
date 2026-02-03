"""Decision integrity operations for debates."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from aragora.rbac.decorators import require_permission
from aragora.server.http_utils import run_async

from aragora.pipeline.decision_integrity import build_decision_integrity_package
from aragora.server.result_router import route_result
from aragora.implement import HybridExecutor
from aragora.autonomous.loop_enhancement import ApprovalStatus
from aragora.server.handlers.autonomous.approvals import get_approval_flow
from aragora.rbac.checker import get_permission_checker

from ..base import HandlerResult, error_response, handle_errors, json_response, require_storage
from ..openapi_decorator import api_endpoint

if TYPE_CHECKING:
    from aragora.billing.auth.context import UserAuthContext


logger = logging.getLogger(__name__)


class _DebatesHandlerProtocol(Protocol):
    ctx: dict[str, Any]

    def get_storage(self) -> Any | None: ...

    def read_json_body(
        self, handler: Any, max_size: int | None = None
    ) -> dict[str, Any] | None: ...

    def get_current_user(self, handler: Any) -> "UserAuthContext | None": ...


class ImplementationOperationsMixin:
    """Mixin providing Decision Integrity endpoints for debates."""

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{id}/decision-integrity",
        summary="Build decision integrity package",
        description="Generate a decision receipt and implementation plan from a debate.",
        tags=["Debates"],
        responses={
            "200": {"description": "Decision integrity package returned"},
            "400": {"description": "Invalid request"},
            "404": {"description": "Debate not found"},
        },
    )
    @require_permission("debates:write")
    @require_storage
    @handle_errors("build decision integrity package")
    def _create_decision_integrity(
        self: _DebatesHandlerProtocol, handler: Any, debate_id: str
    ) -> HandlerResult:
        """Generate a decision receipt and implementation plan for a debate."""
        storage = self.get_storage()
        debate = storage.get_debate(debate_id) if storage else None
        if not debate:
            return error_response("Debate not found", 404)

        payload = self.read_json_body(handler) or {}
        include_receipt = bool(payload.get("include_receipt", True))
        include_plan = bool(payload.get("include_plan", True))
        plan_strategy = str(payload.get("plan_strategy", "single_task"))
        execution_mode = str(payload.get("execution_mode", "plan_only"))
        parallel_execution = bool(payload.get("parallel_execution", False))
        notify_origin = bool(payload.get("notify_origin", False))
        risk_level = str(payload.get("risk_level", "medium"))
        approval_timeout = payload.get("approval_timeout_seconds")

        repo_root = self.ctx.get("repo_root")
        repo_path = Path(repo_root) if repo_root else None

        package = run_async(
            build_decision_integrity_package(
                debate,
                include_receipt=include_receipt,
                include_plan=include_plan,
                plan_strategy=plan_strategy,
                repo_path=repo_path,
            )
        )

        response_payload = package.to_dict()

        # Optional approval request / execution
        if execution_mode in {"request_approval", "execute"}:
            # Enforce approval-related permission if present
            user = self.get_current_user(handler)
            if user:
                try:
                    checker = get_permission_checker()
                    decision = checker.check_permission(user, "autonomous:approve")  # type: ignore[arg-type]
                    if not decision.allowed:
                        return error_response(
                            f"Permission denied: {decision.reason}",
                            403,
                        )
                except Exception:
                    # If checker unavailable, proceed (legacy compatibility)
                    pass

            changes = []
            if package.plan is not None:
                for task in package.plan.tasks:
                    changes.append(
                        {
                            "id": task.id,
                            "description": task.description,
                            "files": task.files,
                            "complexity": task.complexity,
                        }
                    )

            requested_by = getattr(user, "user_id", None) if user else "system"
            approval_flow = get_approval_flow()
            approval_request = run_async(
                approval_flow.request_approval(
                    title=f"Implement debate {debate_id}",
                    description="Execute decision implementation plan generated from debate.",
                    changes=changes,
                    risk_level=risk_level,
                    requested_by=requested_by or "system",
                    timeout_seconds=approval_timeout,
                    metadata={"debate_id": debate_id},
                )
            )
            response_payload["approval"] = {
                "id": approval_request.id,
                "title": approval_request.title,
                "description": approval_request.description,
                "changes": approval_request.changes,
                "risk_level": approval_request.risk_level,
                "requested_at": approval_request.requested_at.isoformat(),
                "requested_by": approval_request.requested_by,
                "timeout_seconds": approval_request.timeout_seconds,
                "status": approval_request.status.value,
                "approved_by": approval_request.approved_by,
                "approved_at": (
                    approval_request.approved_at.isoformat()
                    if approval_request.approved_at
                    else None
                ),
                "rejection_reason": approval_request.rejection_reason,
                "metadata": approval_request.metadata,
            }

            if execution_mode == "execute":
                if os.environ.get("ARAGORA_ENABLE_IMPLEMENTATION_EXECUTION", "0") != "1":
                    return error_response(
                        "Implementation execution disabled. Set ARAGORA_ENABLE_IMPLEMENTATION_EXECUTION=1.",
                        403,
                    )

                if approval_request.status in {
                    ApprovalStatus.APPROVED,
                    ApprovalStatus.AUTO_APPROVED,
                }:
                    if package.plan is None:
                        return error_response("No implementation plan available", 400)

                    executor = HybridExecutor(repo_path=repo_path or Path.cwd())
                    if parallel_execution:
                        results = run_async(
                            executor.execute_plan_parallel(package.plan.tasks, set())
                        )
                    else:
                        results = run_async(executor.execute_plan(package.plan.tasks, set()))
                    response_payload["execution"] = {
                        "status": "completed",
                        "results": [r.to_dict() for r in results],
                    }
                else:
                    response_payload["execution"] = {
                        "status": "pending_approval",
                        "approval_id": approval_request.id,
                    }

        if notify_origin:
            try:
                run_async(
                    route_result(
                        debate_id,
                        {
                            "debate_id": debate_id,
                            "event": "decision_integrity",
                            "package": response_payload,
                        },
                    )
                )
            except Exception as exc:
                logger.debug("Decision integrity routing failed: %s", exc)

        return json_response(response_payload)
