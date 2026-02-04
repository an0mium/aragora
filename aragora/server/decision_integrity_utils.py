"""Helpers for emitting decision integrity packages to channels."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_execution_overrides(text: str) -> tuple[str, dict[str, Any]]:
    """Extract execution overrides (e.g., --computer-use) from user text."""
    if not text:
        return text, {}
    tokens = text.split()
    overrides: dict[str, Any] = {}
    cleaned: list[str] = []
    for token in tokens:
        flag = token.lower()
        if flag in {"--computer-use", "--browser", "--ui"}:
            overrides["execution_mode"] = "execute"
            overrides["execution_engine"] = "computer_use"
            continue
        if flag in {"--hybrid"}:
            overrides["execution_mode"] = "execute"
            overrides["execution_engine"] = "hybrid"
            continue
        cleaned.append(token)
    return " ".join(cleaned).strip(), overrides


def _serialize_approval_request(approval_request: Any) -> dict[str, Any]:
    """Serialize ApprovalRequest into a JSON-friendly payload."""
    return {
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
            approval_request.approved_at.isoformat() if approval_request.approved_at else None
        ),
        "rejection_reason": approval_request.rejection_reason,
        "metadata": approval_request.metadata,
    }


# ---------------------------------------------------------------------------
# Internal config holder for build_decision_integrity_payload
# ---------------------------------------------------------------------------


@dataclass
class _IntegrityBuildConfig:
    """Parsed configuration for a decision integrity build."""

    cfg: dict[str, Any]
    execution_mode: str
    execution_engine: str
    workflow_mode: bool
    execute_workflow: bool
    include_receipt: bool
    include_plan: bool
    include_context: bool
    plan_strategy: str
    notify_origin: bool


def _parse_integrity_config(
    decision_integrity: dict[str, Any] | bool | None,
    *,
    notify_origin_override: bool | None = None,
) -> _IntegrityBuildConfig | None:
    """Parse and normalise the ``decision_integrity`` parameter.

    Returns ``None`` when the caller should bail out early.
    """
    if decision_integrity is None:
        return None
    if isinstance(decision_integrity, bool):
        if not decision_integrity:
            return None
        cfg: dict[str, Any] = {}
    elif isinstance(decision_integrity, dict):
        cfg = decision_integrity
    else:
        return None

    execution_mode = str(cfg.get("execution_mode", "plan_only")).lower()
    execution_engine = str(cfg.get("execution_engine", "")).lower()
    if execution_mode in {"hybrid", "computer_use"}:
        execution_engine = execution_mode
        execution_mode = "execute"

    workflow_mode = execution_mode in {"workflow", "workflow_execute", "execute_workflow"}
    execute_workflow = execution_mode in {"workflow_execute", "execute_workflow"}

    include_receipt = bool(cfg.get("include_receipt", True))
    include_plan = bool(cfg.get("include_plan", True))
    include_context = bool(cfg.get("include_context", False))
    plan_strategy = str(cfg.get("plan_strategy", "single_task"))
    notify_origin = (
        bool(cfg.get("notify_origin", True))
        if notify_origin_override is None
        else notify_origin_override
    )

    if execution_mode in {"request_approval", "execute"} or workflow_mode:
        include_plan = True

    if not any([include_receipt, include_plan, include_context]):
        return None

    return _IntegrityBuildConfig(
        cfg=cfg,
        execution_mode=execution_mode,
        execution_engine=execution_engine,
        workflow_mode=workflow_mode,
        execute_workflow=execute_workflow,
        include_receipt=include_receipt,
        include_plan=include_plan,
        include_context=include_context,
        plan_strategy=plan_strategy,
        notify_origin=notify_origin,
    )


def _build_debate_payload(result: Any, debate_id: str | None) -> dict[str, Any]:
    """Construct the debate payload dict from a result object."""
    debate_payload: dict[str, Any] = {}
    if hasattr(result, "to_dict"):
        try:
            debate_payload = result.to_dict()
        except Exception:
            debate_payload = {}
    if not debate_payload:
        debate_payload = {
            "debate_id": getattr(result, "debate_id", "") or "",
            "task": getattr(result, "task", "") or "",
            "final_answer": getattr(result, "final_answer", "") or "",
            "confidence": getattr(result, "confidence", 0.0) or 0.0,
            "consensus_reached": getattr(result, "consensus_reached", False) or False,
            "rounds_used": getattr(result, "rounds_used", 0) or 0,
            "participants": getattr(result, "participants", []) or [],
        }
    if debate_id:
        debate_payload["debate_id"] = debate_id
    return debate_payload


async def _build_package(
    debate_payload: dict[str, Any],
    bc: _IntegrityBuildConfig,
    arena: Any | None,
    document_store: Any | None,
    evidence_store: Any | None,
) -> Any | None:
    """Build the decision integrity package, returning it or ``None``."""
    try:
        from aragora.pipeline.decision_integrity import build_decision_integrity_package
    except Exception as exc:
        logger.debug("Decision integrity pipeline unavailable: %s", exc)
        return None

    continuum_memory = getattr(arena, "continuum_memory", None) if bc.include_context else None
    cross_debate_memory = (
        getattr(arena, "cross_debate_memory", None) if bc.include_context else None
    )
    knowledge_mound = getattr(arena, "knowledge_mound", None) if bc.include_context else None

    try:
        return await build_decision_integrity_package(
            debate_payload,
            include_receipt=bc.include_receipt,
            include_plan=bc.include_plan,
            include_context=bc.include_context,
            plan_strategy=bc.plan_strategy,
            continuum_memory=continuum_memory,
            cross_debate_memory=cross_debate_memory,
            knowledge_mound=knowledge_mound,
            document_store=document_store,
            evidence_store=evidence_store,
        )
    except Exception as exc:
        logger.debug("Decision integrity build failed: %s", exc)
        return None


def _create_decision_plan(
    bc: _IntegrityBuildConfig,
    package: Any,
    debate_payload: dict[str, Any],
    debate_key: str | None,
    arena: Any | None,
) -> Any | None:
    """Create a ``DecisionPlan`` from the package, or return ``None``."""
    if bc.execution_mode not in {"request_approval", "execute"} and not bc.workflow_mode:
        return None
    if package.plan is None:
        logger.debug("Decision integrity execution requested but no plan available.")
        return None

    try:
        from aragora.pipeline.decision_plan import ApprovalMode, DecisionPlanFactory
        from aragora.pipeline.decision_integrity import coerce_debate_result
        from aragora.pipeline.executor import store_plan
        from aragora.pipeline.risk_register import RiskLevel

        approval_mode_raw = str(bc.cfg.get("approval_mode", "risk_based"))
        try:
            approval_mode = ApprovalMode(approval_mode_raw)
        except ValueError:
            approval_mode = ApprovalMode.RISK_BASED

        max_auto_risk_raw = str(bc.cfg.get("max_auto_risk", "low"))
        try:
            max_auto_risk = RiskLevel(max_auto_risk_raw)
        except ValueError:
            max_auto_risk = RiskLevel.LOW

        budget_limit = None
        budget_value = bc.cfg.get("budget_limit_usd")
        if budget_value is not None:
            try:
                budget_limit = float(budget_value)
            except (TypeError, ValueError):
                budget_limit = None

        metadata: dict[str, Any] = {
            "source": "decision_integrity",
            "debate_id": debate_key,
        }
        if isinstance(bc.cfg.get("openclaw_actions"), list):
            metadata["openclaw_actions"] = bc.cfg["openclaw_actions"]
        if isinstance(bc.cfg.get("computer_use_actions"), list):
            metadata["computer_use_actions"] = bc.cfg["computer_use_actions"]
        if isinstance(bc.cfg.get("openclaw_session"), dict):
            metadata["openclaw_session"] = bc.cfg["openclaw_session"]

        repo_root = bc.cfg.get("repo_path") or bc.cfg.get("repo_root")
        if not repo_root:
            repo_root = getattr(arena, "repo_root", None)
        if not repo_root:
            repo_root = os.environ.get("ARAGORA_REPO_ROOT")
        repo_path = Path(repo_root) if repo_root else None

        plan = DecisionPlanFactory.from_debate_result(
            coerce_debate_result(debate_payload),
            budget_limit_usd=budget_limit,
            approval_mode=approval_mode,
            max_auto_risk=max_auto_risk,
            repo_path=repo_path,
            metadata=metadata,
            implement_plan=package.plan,
        )
        store_plan(plan)
        return plan
    except Exception as exc:
        logger.debug("Decision plan creation failed: %s", exc)
        return None


async def _maybe_request_approval(
    bc: _IntegrityBuildConfig,
    plan: Any,
    payload: dict[str, Any],
) -> Any | None:
    """Submit an approval request if the execution mode requires it.

    Returns the ``ApprovalRequest`` (or ``None``).
    """
    request_approval = bc.execution_mode in {"request_approval", "execute"}
    if bc.workflow_mode and plan.requires_human_approval:
        request_approval = True
    if not request_approval:
        return None

    try:
        from aragora.server.handlers.autonomous.approvals import get_approval_flow

        requested_by = str(bc.cfg.get("requested_by") or "system")
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
        approval_flow = get_approval_flow()
        approval_request = await approval_flow.request_approval(
            title=f"Implement debate {plan.debate_id}",
            description="Execute decision implementation plan generated from debate.",
            changes=changes,
            risk_level=str(bc.cfg.get("risk_level", "medium")),
            requested_by=requested_by,
            timeout_seconds=bc.cfg.get("approval_timeout_seconds"),
            metadata={"debate_id": plan.debate_id, "plan_id": plan.id},
        )
        payload["approval"] = _serialize_approval_request(approval_request)

        try:
            from aragora.autonomous.loop_enhancement import ApprovalStatus

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
                from aragora.pipeline.executor import store_plan

                store_plan(plan)
        except Exception:
            pass
        return approval_request
    except Exception as exc:
        logger.debug("Approval request failed: %s", exc)
        return None


async def _execute_plan(
    bc: _IntegrityBuildConfig,
    plan: Any,
    arena: Any | None,
    debate_key: str | None,
    approval_request: Any | None,
) -> dict[str, Any] | None:
    """Execute the plan and return the execution payload, or ``None``."""
    should_execute = bc.execution_mode == "execute" or bc.execute_workflow
    if not should_execute:
        return None

    if os.environ.get("ARAGORA_ENABLE_IMPLEMENTATION_EXECUTION", "0") != "1":
        return {
            "status": "disabled",
            "reason": "Set ARAGORA_ENABLE_IMPLEMENTATION_EXECUTION=1 to enable.",
        }

    if plan.requires_human_approval and not plan.is_approved:
        return {
            "status": "pending_approval",
            "approval_id": approval_request.id if approval_request else None,
        }

    try:
        from aragora.pipeline.execution_notifier import ExecutionNotifier
        from aragora.pipeline.executor import PlanExecutor

        engine = bc.execution_engine or ("workflow" if bc.workflow_mode else "hybrid")
        parallel_execution = bool(bc.cfg.get("parallel_execution", False))

        notifier = None
        on_task_complete = None
        if engine in {"hybrid", "computer_use"}:
            notifier = ExecutionNotifier(
                debate_id=plan.debate_id or str(debate_key or ""),
                plan_id=plan.id,
                notify_channel=bc.notify_origin,
                notify_websocket=bc.notify_origin,
            )
            if engine == "hybrid" and plan.implement_plan is not None:
                notifier.set_task_descriptions(plan.implement_plan.tasks)
            if engine == "computer_use":
                max_steps = bc.cfg.get("computer_use_max_steps", 50)
                try:
                    notifier.progress.total_tasks = int(max_steps)
                except (TypeError, ValueError):
                    notifier.progress.total_tasks = 50
            on_task_complete = notifier.on_task_complete

        executor = PlanExecutor(
            continuum_memory=getattr(arena, "continuum_memory", None),
            knowledge_mound=getattr(arena, "knowledge_mound", None),
            parallel_execution=parallel_execution,
            execution_mode=engine,
        )
        outcome = await executor.execute(
            plan,
            parallel_execution=parallel_execution,
            execution_mode=engine,
            on_task_complete=on_task_complete,
        )
        if notifier and bc.notify_origin:
            await notifier.send_completion_summary()
        result_payload: dict[str, Any] = {
            "status": "completed",
            "mode": engine,
            "outcome": outcome.to_dict(),
        }
        if engine == "computer_use" and notifier:
            result_payload["progress"] = notifier.progress.to_dict()
        return result_payload
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def build_decision_integrity_payload(
    *,
    result: Any,
    debate_id: str | None,
    arena: Any | None,
    decision_integrity: dict[str, Any] | bool | None,
    document_store: Any | None = None,
    evidence_store: Any | None = None,
    notify_origin_override: bool | None = None,
) -> dict[str, Any] | None:
    """Build a decision integrity payload, optionally executing the plan."""
    bc = _parse_integrity_config(decision_integrity, notify_origin_override=notify_origin_override)
    if bc is None:
        return None

    debate_payload = _build_debate_payload(result, debate_id)

    package = await _build_package(debate_payload, bc, arena, document_store, evidence_store)
    if package is None:
        return None

    payload = package.to_dict()

    # Inject execution mode/engine into payload
    if "execution_mode" in bc.cfg or "execution_engine" in bc.cfg:
        payload["execution_mode"] = bc.execution_mode
        effective_engine = bc.execution_engine or ("workflow" if bc.workflow_mode else "")
        if bc.execution_mode in {"execute", "request_approval"} and not effective_engine:
            effective_engine = "hybrid"
        if effective_engine:
            payload["execution_engine"] = effective_engine

    debate_key = (
        payload.get("debate_id")
        or debate_payload.get("debate_id")
        or debate_id
        or getattr(result, "id", None)
    )

    # Plan creation → approval → execution
    plan = _create_decision_plan(bc, package, debate_payload, debate_key, arena)

    if plan is not None:
        payload["decision_plan"] = plan.to_dict()
        payload["plan_id"] = plan.id

        approval_request = await _maybe_request_approval(bc, plan, payload)

        execution_payload = await _execute_plan(bc, plan, arena, debate_key, approval_request)
        if execution_payload is not None:
            key = "workflow_execution" if bc.workflow_mode else "execution"
            payload[key] = execution_payload

    # Route result to originating channel
    if bc.notify_origin and debate_key:
        try:
            from aragora.server.result_router import route_result

            await route_result(
                debate_key,
                {
                    "debate_id": debate_key,
                    "event": "decision_integrity",
                    "package": payload,
                },
            )
        except Exception as exc:
            logger.debug("Decision integrity routing failed: %s", exc)

    return payload


async def maybe_emit_decision_integrity(
    *,
    result: Any,
    debate_id: str | None,
    arena: Any | None,
    decision_integrity: dict[str, Any] | bool | None,
    document_store: Any | None = None,
    evidence_store: Any | None = None,
) -> None:
    """Optionally build and route a decision integrity package."""
    await build_decision_integrity_payload(
        result=result,
        debate_id=debate_id,
        arena=arena,
        decision_integrity=decision_integrity,
        document_store=document_store,
        evidence_store=evidence_store,
    )
