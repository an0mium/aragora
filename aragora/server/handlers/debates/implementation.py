"""Decision integrity operations for debates.

Provides the POST /api/v1/debates/{id}/decision-integrity endpoint which:
- Generates a decision receipt (audit trail)
- Creates an implementation plan (for multi-agent execution)
- Optionally captures a context snapshot (memory + knowledge state)
- Persists receipt and plan for later retrieval via /api/v2/receipts/
- Enforces budget limits before execution
- Supports approval flow and parallel execution
- Routes results to originating channel
"""

from __future__ import annotations

import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from aragora.rbac.decorators import require_permission
from aragora.server.http_utils import run_async

from aragora.pipeline.decision_integrity import (
    build_decision_integrity_package,
    coerce_debate_result,
)
from aragora.server.result_router import route_result
from aragora.implement import HybridExecutor
from aragora.pipeline.execution_notifier import ExecutionNotifier
from aragora.autonomous.loop_enhancement import ApprovalStatus
from aragora.server.handlers.autonomous.approvals import get_approval_flow
from aragora.rbac.checker import get_permission_checker

from ..base import HandlerResult, error_response, handle_errors, json_response, require_storage
from ..openapi_decorator import api_endpoint

if TYPE_CHECKING:
    from aragora.billing.auth.context import UserAuthContext


logger = logging.getLogger(__name__)

try:
    from aragora.storage.receipt_store import get_receipt_store as _receipt_store_get
except Exception:
    _receipt_store_get = None


def get_receipt_store() -> Any:
    """Compatibility shim for test patching."""
    if _receipt_store_get is None:
        raise RuntimeError("Receipt store unavailable")
    return _receipt_store_get()


def _persist_receipt(receipt: Any, debate_id: str) -> str | None:
    """Persist a DecisionReceipt to the receipt store for later retrieval.

    Returns the receipt_id on success, None on failure.
    """
    try:
        from aragora.storage.receipt_store import get_receipt_store

        store = get_receipt_store()
        receipt_dict = receipt.to_dict()
        receipt_dict.setdefault("debate_id", debate_id)
        return store.save(receipt_dict)
    except Exception as exc:
        logger.debug("Receipt persistence failed: %s", exc)
        return None


def _persist_plan(plan: Any, debate_id: str) -> None:
    """Store an ImplementPlan in the pipeline plan store for tracking."""
    try:
        from aragora.pipeline.executor import store_plan
        from aragora.pipeline.decision_plan import DecisionPlanFactory

        # Wrap ImplementPlan as a DecisionPlan for the store
        decision_plan = DecisionPlanFactory.from_implement_plan(plan, debate_id=debate_id)
        store_plan(decision_plan)
    except Exception as exc:
        logger.debug("Plan persistence failed: %s", exc)


def _check_execution_budget(debate_id: str, ctx: dict[str, Any]) -> tuple[bool, str]:
    """Check budget before executing an implementation plan.

    Returns (allowed, message).
    """
    try:
        cost_tracker = ctx.get("cost_tracker")
        if cost_tracker is None:
            return True, ""  # No tracker configured

        result = cost_tracker.check_debate_budget(debate_id, estimated_cost_usd=Decimal("0.10"))
        if not result.get("allowed", True):
            return False, result.get("message", "Budget exceeded")
        return True, ""
    except Exception as exc:
        logger.debug("Budget check failed (allowing): %s", exc)
        return True, ""


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
        include_context = bool(payload.get("include_context", False))
        plan_strategy = str(payload.get("plan_strategy", "single_task"))
        execution_mode = str(payload.get("execution_mode", "plan_only")).lower()
        execution_engine = str(payload.get("execution_engine", "")).lower()
        parallel_execution = bool(payload.get("parallel_execution", False))
        notify_origin = bool(payload.get("notify_origin", False))
        risk_level = str(payload.get("risk_level", "medium"))
        approval_timeout = payload.get("approval_timeout_seconds")
        approval_mode = str(payload.get("approval_mode", "risk_based"))
        max_auto_risk = str(payload.get("max_auto_risk", "low"))
        budget_limit_usd = payload.get("budget_limit_usd")
        openclaw_actions = payload.get("openclaw_actions")
        computer_use_actions = payload.get("computer_use_actions")
        openclaw_session = payload.get("openclaw_session")

        if execution_mode in {"hybrid", "computer_use"}:
            execution_engine = execution_mode
            execution_mode = "execute"

        workflow_mode = execution_mode in {
            "workflow",
            "workflow_execute",
            "execute_workflow",
        }
        execute_workflow = execution_mode in {"workflow_execute", "execute_workflow"}

        if workflow_mode and not include_plan:
            include_plan = True

        repo_root = self.ctx.get("repo_root")
        repo_path = Path(repo_root) if repo_root else None

        # Optionally pull memory systems from server context for context snapshot
        continuum_memory = self.ctx.get("continuum_memory") if include_context else None
        cross_debate_memory = self.ctx.get("cross_debate_memory") if include_context else None
        knowledge_mound = self.ctx.get("knowledge_mound") if include_context else None
        document_store = self.ctx.get("document_store") if include_context else None
        evidence_store = self.ctx.get("evidence_store") if include_context else None
        if include_context and evidence_store is None:
            try:
                from aragora.evidence.store import EvidenceStore

                evidence_store = EvidenceStore()
                self.ctx["evidence_store"] = evidence_store
            except Exception:
                evidence_store = None

        package = run_async(
            build_decision_integrity_package(
                debate,
                include_receipt=include_receipt,
                include_plan=include_plan,
                include_context=include_context,
                plan_strategy=plan_strategy,
                repo_path=repo_path,
                continuum_memory=continuum_memory,
                cross_debate_memory=cross_debate_memory,
                knowledge_mound=knowledge_mound,
                document_store=document_store,
                evidence_store=evidence_store,
            )
        )

        response_payload = package.to_dict()

        # Persist receipt and plan for later retrieval via existing endpoints
        receipt_id = None
        if package.receipt is not None:
            receipt_id = _persist_receipt(package.receipt, debate_id)
            if receipt_id:
                response_payload["receipt_id"] = receipt_id

        computer_use_plan = None
        if package.plan is not None and not workflow_mode:
            if execution_engine == "computer_use":
                try:
                    from aragora.pipeline.decision_plan import DecisionPlanFactory
                    from aragora.pipeline.executor import store_plan

                    if isinstance(debate, dict):
                        debate_task = str(debate.get("task", "") or "")
                    else:
                        debate_task = str(getattr(debate, "task", "") or "")
                    computer_use_plan = DecisionPlanFactory.from_implement_plan(
                        package.plan,
                        debate_id=debate_id,
                        task=debate_task,
                    )
                    store_plan(computer_use_plan)
                    response_payload["plan_id"] = computer_use_plan.id
                except Exception as exc:
                    logger.debug("Computer use plan persistence failed: %s", exc)
            else:
                _persist_plan(package.plan, debate_id)

        # Optional Obsidian writeback for decision integrity packages
        if os.environ.get("ARAGORA_OBSIDIAN_WRITEBACK", "0") == "1":
            try:
                from aragora.connectors.knowledge.obsidian import (
                    ObsidianConfig,
                    ObsidianConnector,
                )

                config = ObsidianConfig.from_env()
                verification_payload = None
                if receipt_id:
                    try:
                        from aragora.storage.receipt_store import get_receipt_store

                        store = get_receipt_store()
                        signature_result = store.verify_signature(receipt_id)
                        integrity_result = store.verify_integrity(receipt_id)
                        verification_payload = {
                            "signature": signature_result.to_dict()
                            if hasattr(signature_result, "to_dict")
                            else signature_result,
                            "integrity": integrity_result,
                        }
                    except Exception as exc:
                        logger.debug("Receipt verification for Obsidian writeback failed: %s", exc)

                if config is None:
                    logger.debug("Obsidian writeback enabled but vault is not configured")
                else:
                    connector = ObsidianConnector(config)
                    folder = os.environ.get(
                        "ARAGORA_OBSIDIAN_WRITEBACK_FOLDER",
                        "decisions",
                    )
                    run_async(
                        connector.write_decision_integrity_package(
                            package,
                            folder=folder,
                            verification=verification_payload,
                        )
                    )
            except Exception as exc:
                logger.debug("Obsidian writeback failed: %s", exc)

        # Workflow-based execution path (DecisionPlan + WorkflowEngine)
        if workflow_mode:
            from aragora.pipeline.decision_plan import ApprovalMode, DecisionPlanFactory
            from aragora.pipeline.executor import PlanExecutor, store_plan
            from aragora.pipeline.risk_register import RiskLevel

            debate_result = coerce_debate_result(debate)

            try:
                approval_mode_enum = ApprovalMode(approval_mode)
            except ValueError:
                approval_mode_enum = ApprovalMode.RISK_BASED

            try:
                max_auto_risk_enum = RiskLevel(max_auto_risk)
            except ValueError:
                max_auto_risk_enum = RiskLevel.LOW

            budget_limit = None
            if budget_limit_usd is not None:
                try:
                    budget_limit = float(budget_limit_usd)
                except (TypeError, ValueError):
                    budget_limit = None

            metadata: dict[str, Any] = {
                "source": "decision_integrity",
                "debate_id": debate_id,
            }
            if isinstance(openclaw_actions, list):
                metadata["openclaw_actions"] = openclaw_actions
            if isinstance(computer_use_actions, list):
                metadata["computer_use_actions"] = computer_use_actions
            if isinstance(openclaw_session, dict):
                metadata["openclaw_session"] = openclaw_session

            plan = DecisionPlanFactory.from_debate_result(
                debate_result,
                budget_limit_usd=budget_limit,
                approval_mode=approval_mode_enum,
                max_auto_risk=max_auto_risk_enum,
                repo_path=repo_path,
                metadata=metadata,
                implement_plan=package.plan,
            )
            store_plan(plan)

            response_payload["decision_plan"] = plan.to_dict()
            response_payload["plan_id"] = plan.id

            approval_request = None
            if plan.requires_human_approval:
                user = self.get_current_user(handler)
                if user:
                    try:
                        checker = get_permission_checker()
                        decision = checker.check_permission(
                            user,  # type: ignore[arg-type]
                            "autonomous:approve",
                        )
                        if not decision.allowed:
                            return error_response(
                                f"Permission denied: {decision.reason}",
                                403,
                            )
                    except Exception as e:
                        logger.warning("Permission check for autonomous:approve failed: %s", e)

                requested_by = getattr(user, "user_id", None) if user else "system"
                changes = []
                if plan.implement_plan is not None:
                    for task in plan.implement_plan.tasks:
                        changes.append(
                            {
                                "id": task.id,
                                "description": task.description,
                                "files": task.files,
                                "complexity": task.complexity,
                            }
                        )

                risk_level_for_approval = risk_level
                try:
                    risk_level_for_approval = plan.highest_risk_level.value
                except Exception as e:
                    logger.debug("Could not extract risk level from plan: %s", e)

                approval_flow = get_approval_flow()
                approval_request = run_async(
                    approval_flow.request_approval(
                        title=f"Implement debate {debate_id}",
                        description=(
                            "Execute decision plan generated from debate "
                            "(workflow-based execution)."
                        ),
                        changes=changes,
                        risk_level=risk_level_for_approval,
                        requested_by=requested_by or "system",
                        timeout_seconds=approval_timeout,
                        metadata={"debate_id": debate_id, "plan_id": plan.id},
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

                if approval_request.status in {
                    ApprovalStatus.APPROVED,
                    ApprovalStatus.AUTO_APPROVED,
                }:
                    plan.approve(
                        approver_id=approval_request.approved_by or requested_by or "system",
                        reason="Auto-approved by policy"
                        if approval_request.status == ApprovalStatus.AUTO_APPROVED
                        else "Approved",
                    )
                    store_plan(plan)

            if execute_workflow:
                if os.environ.get("ARAGORA_ENABLE_IMPLEMENTATION_EXECUTION", "0") != "1":
                    return error_response(
                        "Implementation execution disabled. Set ARAGORA_ENABLE_IMPLEMENTATION_EXECUTION=1.",
                        403,
                    )

                # Budget enforcement before execution
                budget_ok, budget_msg = _check_execution_budget(debate_id, self.ctx)
                if not budget_ok:
                    return error_response(f"Budget limit: {budget_msg}", 402)

                if plan.is_approved:
                    plan_executor = PlanExecutor(
                        continuum_memory=self.ctx.get("continuum_memory"),
                        knowledge_mound=self.ctx.get("knowledge_mound"),
                        parallel_execution=parallel_execution,
                    )
                    outcome = run_async(
                        plan_executor.execute(plan, parallel_execution=parallel_execution)
                    )
                    response_payload["workflow_execution"] = {
                        "status": "completed",
                        "outcome": outcome.to_dict(),
                    }
                else:
                    response_payload["workflow_execution"] = {
                        "status": "pending_approval",
                        "approval_id": approval_request.id if approval_request else None,
                    }

            if notify_origin:
                try:
                    run_async(
                        route_result(
                            debate_id,
                            {
                                "debate_id": debate_id,
                                "event": "decision_plan",
                                "package": response_payload,
                            },
                        )
                    )
                except Exception as exc:
                    logger.debug("Decision plan routing failed: %s", exc)

            return json_response(response_payload)

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

                # Budget enforcement before execution
                budget_ok, budget_msg = _check_execution_budget(debate_id, self.ctx)
                if not budget_ok:
                    return error_response(f"Budget limit: {budget_msg}", 402)

                engine = execution_engine or "hybrid"
                if engine not in {"hybrid", "computer_use"}:
                    engine = "hybrid"

                if approval_request.status in {
                    ApprovalStatus.APPROVED,
                    ApprovalStatus.AUTO_APPROVED,
                }:
                    if package.plan is None:
                        return error_response("No implementation plan available", 400)

                    if engine == "computer_use":
                        try:
                            from aragora.pipeline.executor import PlanExecutor, store_plan

                            if computer_use_plan is None:
                                return error_response(
                                    "No execution plan available for computer use", 400
                                )
                            computer_use_plan.approve(
                                approver_id=approval_request.approved_by
                                or requested_by
                                or "system",
                                reason="Approved",
                            )
                            store_plan(computer_use_plan)
                            plan_executor = PlanExecutor(
                                continuum_memory=self.ctx.get("continuum_memory"),
                                knowledge_mound=self.ctx.get("knowledge_mound"),
                                parallel_execution=parallel_execution,
                                execution_mode="computer_use",
                                repo_path=repo_path or Path.cwd(),
                            )
                            outcome = run_async(
                                plan_executor.execute(
                                    computer_use_plan,
                                    parallel_execution=parallel_execution,
                                    execution_mode="computer_use",
                                )
                            )
                            response_payload["execution"] = {
                                "status": "completed",
                                "mode": "computer_use",
                                "outcome": outcome.to_dict(),
                            }
                        except Exception as exc:
                            response_payload["execution"] = {
                                "status": "failed",
                                "mode": "computer_use",
                                "error": str(exc),
                            }
                    else:
                        hybrid_executor = HybridExecutor(repo_path=repo_path or Path.cwd())
                        notifier = ExecutionNotifier(
                            debate_id=debate_id,
                            notify_channel=notify_origin,
                            notify_websocket=notify_origin,
                        )
                        notifier.set_task_descriptions(package.plan.tasks)
                        if parallel_execution:
                            results = run_async(
                                hybrid_executor.execute_plan_parallel(
                                    package.plan.tasks,
                                    set(),
                                    on_task_complete=notifier.on_task_complete,
                                )
                            )
                        else:
                            results = run_async(
                                hybrid_executor.execute_plan(
                                    package.plan.tasks,
                                    set(),
                                    on_task_complete=notifier.on_task_complete,
                                )
                            )
                        if notify_origin:
                            run_async(notifier.send_completion_summary())
                        response_payload["execution"] = {
                            "status": "completed",
                            "results": [r.to_dict() for r in results],
                            "progress": notifier.progress.to_dict(),
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
