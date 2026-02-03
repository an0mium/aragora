"""Helpers for emitting decision integrity packages to channels."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
    if decision_integrity is None:
        return
    if isinstance(decision_integrity, bool):
        if not decision_integrity:
            return
        cfg: dict[str, Any] = {}
    elif isinstance(decision_integrity, dict):
        cfg = decision_integrity
    else:
        return

    include_receipt = bool(cfg.get("include_receipt", True))
    include_plan = bool(cfg.get("include_plan", True))
    include_context = bool(cfg.get("include_context", False))
    plan_strategy = str(cfg.get("plan_strategy", "single_task"))
    notify_origin = bool(cfg.get("notify_origin", True))

    if not any([include_receipt, include_plan, include_context]):
        return

    try:
        from aragora.pipeline.decision_integrity import build_decision_integrity_package
    except Exception as exc:
        logger.debug("Decision integrity pipeline unavailable: %s", exc)
        return

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

    continuum_memory = getattr(arena, "continuum_memory", None) if include_context else None
    cross_debate_memory = getattr(arena, "cross_debate_memory", None) if include_context else None
    knowledge_mound = getattr(arena, "knowledge_mound", None) if include_context else None

    try:
        package = await build_decision_integrity_package(
            debate_payload,
            include_receipt=include_receipt,
            include_plan=include_plan,
            include_context=include_context,
            plan_strategy=plan_strategy,
            continuum_memory=continuum_memory,
            cross_debate_memory=cross_debate_memory,
            knowledge_mound=knowledge_mound,
            document_store=document_store,
            evidence_store=evidence_store,
        )
    except Exception as exc:
        logger.debug("Decision integrity build failed: %s", exc)
        return

    payload = package.to_dict()
    if not notify_origin:
        return

    try:
        from aragora.server.result_router import route_result

        debate_key = (
            payload.get("debate_id")
            or debate_payload.get("debate_id")
            or debate_id
            or getattr(result, "id", None)
        )
        if debate_key:
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
