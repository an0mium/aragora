"""Decision Integrity pipeline helpers.

Builds a Decision Integrity package from a debate artifact:
- Decision receipt (audit trail)
- Implementation plan (for multi-agent execution)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aragora.core_types import DebateResult
from aragora.gauntlet.receipt import DecisionReceipt
from aragora.implement import create_single_task_plan, generate_implement_plan
from aragora.implement.types import ImplementPlan


@dataclass
class DecisionIntegrityPackage:
    """Bundle of artifacts for decision implementation."""

    debate_id: str
    receipt: DecisionReceipt | None
    plan: ImplementPlan | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "plan": self.plan.to_dict() if self.plan else None,
        }


def _coerce_debate_result(debate: dict[str, Any]) -> DebateResult:
    """Best-effort conversion from stored debate dict to DebateResult."""
    agents = debate.get("agents", []) or []
    if isinstance(agents, str):
        agents = [a.strip() for a in agents.split(",") if a.strip()]

    return DebateResult(
        debate_id=str(debate.get("debate_id") or debate.get("id") or ""),
        task=str(debate.get("task") or debate.get("question") or ""),
        final_answer=str(debate.get("final_answer") or debate.get("conclusion") or ""),
        confidence=float(debate.get("confidence") or 0.0),
        consensus_reached=bool(debate.get("consensus_reached") or False),
        rounds_used=int(debate.get("rounds_used") or debate.get("rounds") or 0),
        rounds_completed=int(debate.get("rounds_completed") or 0),
        status=str(debate.get("status") or ""),
        participants=list(agents),
        metadata=dict(debate.get("metadata") or {}),
    )


async def build_decision_integrity_package(
    debate: dict[str, Any],
    *,
    include_receipt: bool = True,
    include_plan: bool = True,
    plan_strategy: str = "single_task",
    repo_path: Path | None = None,
) -> DecisionIntegrityPackage:
    """Build a Decision Integrity package from a debate payload.

    Args:
        debate: Debate payload (dict) from storage.
        include_receipt: Whether to generate a DecisionReceipt.
        include_plan: Whether to generate an implementation plan.
        plan_strategy: "single_task" (default) or "gemini" (best-effort).
        repo_path: Repository root (defaults to cwd).
    """
    debate_result = _coerce_debate_result(debate)
    receipt = DecisionReceipt.from_debate_result(debate_result) if include_receipt else None

    plan: ImplementPlan | None = None
    if include_plan:
        repo_root = repo_path or Path.cwd()
        design = debate_result.final_answer or debate_result.task
        if plan_strategy == "gemini":
            try:
                plan = await generate_implement_plan(design=design, repo_path=repo_root)
            except Exception:
                plan = create_single_task_plan(design=design, repo_path=repo_root)
        else:
            plan = create_single_task_plan(design=design, repo_path=repo_root)

    return DecisionIntegrityPackage(
        debate_id=debate_result.debate_id or str(debate.get("id") or ""),
        receipt=receipt,
        plan=plan,
    )
