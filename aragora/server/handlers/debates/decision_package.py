"""
Decision Package Assembly Endpoint.

Assembles a unified "decision package" from a completed debate, combining:
- Receipt (verdict, confidence, risk)
- Explanation summary
- Cost breakdown
- Argument map
- Next steps

Routes:
    GET /api/v1/debates/{id}/package          - JSON decision package
    GET /api/v1/debates/{id}/package/markdown  - Markdown export
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)

logger = logging.getLogger(__name__)


def _generate_next_steps(
    verdict: str,
    confidence: float,
    consensus_reached: bool,
    question: str,
) -> list[dict[str, str]]:
    """Generate actionable next steps based on debate outcome.

    Args:
        verdict: The debate verdict (APPROVED, APPROVED_WITH_CONDITIONS, NEEDS_REVIEW).
        confidence: Confidence score 0-1.
        consensus_reached: Whether agents reached consensus.
        question: The original debate question.

    Returns:
        List of next-step dicts with 'action' and 'priority' keys.
    """
    steps: list[dict[str, str]] = []

    if verdict == "APPROVED" and confidence >= 0.8:
        steps.append({"action": "Proceed with implementation", "priority": "high"})
        steps.append(
            {"action": "Document decision rationale for audit trail", "priority": "medium"}
        )
    elif verdict == "APPROVED_WITH_CONDITIONS":
        steps.append({"action": "Address conditions before proceeding", "priority": "high"})
        steps.append(
            {"action": "Schedule follow-up review after conditions met", "priority": "medium"}
        )
        steps.append(
            {"action": "Document conditions and acceptance criteria", "priority": "medium"}
        )
    elif verdict == "NEEDS_REVIEW":
        steps.append({"action": "Escalate to human decision-maker", "priority": "high"})
        steps.append({"action": "Gather additional evidence or expert input", "priority": "high"})
        if not consensus_reached:
            steps.append(
                {
                    "action": "Consider running a follow-up debate with additional agents",
                    "priority": "medium",
                }
            )
    else:
        steps.append(
            {"action": "Review debate results and determine next action", "priority": "medium"}
        )

    if confidence < 0.5:
        steps.append(
            {"action": "Low confidence detected -- seek additional validation", "priority": "high"}
        )

    return steps


def _build_markdown(package: dict[str, Any]) -> str:
    """Render a decision package as markdown.

    Args:
        package: The assembled decision package dict.

    Returns:
        Markdown string.
    """
    lines: list[str] = []
    lines.append(f"# Decision Package: {package.get('debate_id', 'Unknown')}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Question:** {package.get('question', 'N/A')}")
    lines.append(f"- **Verdict:** {package.get('verdict', 'N/A')}")
    lines.append(f"- **Confidence:** {package.get('confidence', 0):.0%}")
    lines.append(f"- **Consensus:** {'Yes' if package.get('consensus_reached') else 'No'}")
    lines.append(f"- **Status:** {package.get('status', 'N/A')}")
    lines.append("")

    # Final answer
    if package.get("final_answer"):
        lines.append("## Final Answer")
        lines.append("")
        lines.append(package["final_answer"])
        lines.append("")

    # Explanation
    if package.get("explanation_summary"):
        lines.append("## Explanation")
        lines.append("")
        lines.append(package["explanation_summary"])
        lines.append("")

    # Cost
    cost = package.get("cost", {})
    if cost:
        lines.append("## Cost Breakdown")
        lines.append("")
        lines.append(f"- **Total:** ${cost.get('total_cost_usd', 0):.4f}")
        per_agent = cost.get("per_agent_cost", {})
        if per_agent:
            for agent, agent_cost in per_agent.items():
                lines.append(f"  - {agent}: ${agent_cost:.4f}")
        lines.append("")

    # Receipt
    receipt = package.get("receipt")
    if receipt:
        lines.append("## Receipt")
        lines.append("")
        lines.append(f"- **Receipt ID:** `{receipt.get('receipt_id', 'N/A')}`")
        lines.append(f"- **Risk Level:** {receipt.get('risk_level', 'N/A')}")
        lines.append(f"- **Checksum:** `{receipt.get('checksum', 'N/A')}`")
        lines.append("")

    # Next steps
    next_steps = package.get("next_steps", [])
    if next_steps:
        lines.append("## Next Steps")
        lines.append("")
        for step in next_steps:
            priority = step.get("priority", "medium").upper()
            lines.append(f"- [{priority}] {step.get('action', '')}")
        lines.append("")

    # Participants
    participants = package.get("participants", [])
    if participants:
        lines.append("## Participants")
        lines.append("")
        for p in participants:
            lines.append(f"- {p}")
        lines.append("")

    # Argument map
    argument_map = package.get("argument_map")
    if argument_map:
        nodes = argument_map.get("nodes", [])
        if nodes:
            lines.append("## Argument Map")
            lines.append("")
            lines.append(f"- **Nodes:** {len(nodes)}")
            lines.append(f"- **Edges:** {len(argument_map.get('edges', []))}")
            lines.append("")

    # Export formats
    lines.append("## Export Formats")
    lines.append("")
    for fmt in package.get("export_formats", []):
        lines.append(f"- {fmt}")
    lines.append("")

    lines.append(f"---\n*Generated at {package.get('assembled_at', 'N/A')}*")
    return "\n".join(lines)


class DecisionPackageHandler(BaseHandler):
    """Handler for decision package assembly endpoints."""

    ROUTES = [
        "/api/v1/debates/*/package",
        "/api/v1/debates/*/package/markdown",
    ]

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        parts = path.split("/")
        # /api/v1/debates/{id}/package -> 6 parts
        if len(parts) == 6 and parts[5] == "package":
            return parts[1] == "api" and parts[2] == "v1" and parts[3] == "debates"
        # /api/v1/debates/{id}/package/markdown -> 7 parts
        if len(parts) == 7 and parts[5] == "package" and parts[6] == "markdown":
            return parts[1] == "api" and parts[2] == "v1" and parts[3] == "debates"
        return False

    def _extract_debate_id(self, path: str) -> str | None:
        """Extract debate ID from path segment 4.

        Path layout: /api/v1/debates/{id}/package[/markdown]
        Indexes:      0  1   2   3       4   5       6
        """
        parts = path.split("/")
        if len(parts) >= 5:
            return parts[4]
        return None

    # ------------------------------------------------------------------
    # GET /api/v1/debates/{id}/package
    # GET /api/v1/debates/{id}/package/markdown
    # ------------------------------------------------------------------

    @require_permission("debates:read")
    def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        """Route to JSON or markdown package assembly."""
        debate_id = self._extract_debate_id(path)
        if not debate_id:
            return error_response("Missing debate ID", 400)

        parts = path.split("/")
        if len(parts) == 7 and parts[6] == "markdown":
            return self._handle_markdown(debate_id)
        return self._handle_json(debate_id)

    # ------------------------------------------------------------------
    # Internal: assemble the package
    # ------------------------------------------------------------------

    def _assemble_package(
        self, debate_id: str
    ) -> tuple[dict[str, Any] | None, HandlerResult | None]:
        """Assemble a decision package for the given debate.

        Returns:
            (package_dict, None) on success, or (None, error_response) on failure.
        """
        storage = self.get_storage()
        if not storage:
            return None, error_response("Storage not available", 503)

        debate = storage.get_debate(debate_id)
        if not debate:
            return None, error_response(f"Debate not found: {debate_id}", 404)

        status = debate.get("status", "unknown")
        if status not in ("completed", "timeout"):
            return None, error_response(
                f"Debate not completed (status: {status}). Package is only available for completed debates.",
                409,
            )

        # Extract result data from debate
        result_data = debate.get("result", {}) or {}

        # -- Receipt (graceful degradation) --
        receipt_dict: dict[str, Any] | None = None
        try:
            from aragora.storage.receipt_store import get_receipt_store

            store = get_receipt_store()
            receipt = store.get_by_gauntlet(f"debate-{debate_id}")
            if receipt:
                receipt_dict = {
                    "receipt_id": receipt.receipt_id,
                    "verdict": receipt.verdict,
                    "confidence": receipt.confidence,
                    "risk_level": receipt.risk_level,
                    "risk_score": getattr(receipt, "risk_score", None),
                    "checksum": receipt.checksum,
                    "created_at": str(receipt.created_at) if receipt.created_at else None,
                }
        except (
            ImportError,
            KeyError,
            ValueError,
            OSError,
            TypeError,
            RuntimeError,
            AttributeError,
        ) as exc:
            logger.debug("Receipt not available for %s: %s", debate_id, exc)

        # -- Verdict & confidence (from receipt or result) --
        verdict = "UNKNOWN"
        confidence = 0.0
        if receipt_dict:
            verdict = receipt_dict.get("verdict", "UNKNOWN")
            confidence = receipt_dict.get("confidence", 0.0)
        elif result_data:
            confidence = result_data.get("confidence", 0.0)
            consensus = result_data.get("consensus_reached", False)
            if consensus and confidence >= 0.7:
                verdict = "APPROVED"
            elif consensus:
                verdict = "APPROVED_WITH_CONDITIONS"
            else:
                verdict = "NEEDS_REVIEW"

        # -- Argument map (graceful degradation) --
        argument_map: dict[str, Any] | None = None
        try:
            messages = debate.get("messages", [])
            if messages:
                from aragora.visualization.mapper import ArgumentCartographer

                cart = ArgumentCartographer()
                cart.set_debate_context(debate_id, debate.get("question", ""))
                for msg in messages:
                    cart.update_from_message(
                        agent=msg.get("agent", msg.get("role", "unknown")),
                        content=msg.get("content", ""),
                        role=msg.get("role", "proposal"),
                        round_num=msg.get("round", 0),
                    )
                map_json = cart.export_json(include_full_content=False)
                argument_map = json.loads(map_json)
        except (
            ImportError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            logger.debug("Argument map not available for %s: %s", debate_id, exc)

        # -- Cost --
        cost = {
            "total_cost_usd": result_data.get("total_cost_usd", 0.0),
            "per_agent_cost": result_data.get("per_agent_cost", {}),
        }

        # -- Next steps --
        consensus_reached = result_data.get("consensus_reached", False)
        question = debate.get("question", "")
        next_steps = _generate_next_steps(verdict, confidence, consensus_reached, question)

        package: dict[str, Any] = {
            "debate_id": debate_id,
            "question": question,
            "status": status,
            "verdict": verdict,
            "confidence": confidence,
            "consensus_reached": consensus_reached,
            "final_answer": result_data.get("final_answer", ""),
            "explanation_summary": result_data.get("explanation_summary", ""),
            "participants": result_data.get("participants", debate.get("agents", [])),
            "receipt": receipt_dict,
            "cost": cost,
            "argument_map": argument_map,
            "next_steps": next_steps,
            "export_formats": ["json", "markdown", "csv", "html", "txt"],
            "assembled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        return package, None

    def _handle_json(self, debate_id: str) -> HandlerResult:
        """Return the decision package as JSON."""
        package, err = self._assemble_package(debate_id)
        if err:
            return err
        return json_response(package)

    def _handle_markdown(self, debate_id: str) -> HandlerResult:
        """Return the decision package as markdown."""
        package, err = self._assemble_package(debate_id)
        if err:
            return err
        md = _build_markdown(package)
        return HandlerResult(
            status_code=200,
            content_type="text/markdown; charset=utf-8",
            body=md.encode("utf-8"),
        )

    def get_storage(self) -> Any:
        """Get debate storage from server context."""
        return self.ctx.get("storage")


__all__ = ["DecisionPackageHandler"]
