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

from aragora.core_types import (
    DebateStatus,
    DebateStatusSource,
    normalize_debate_status,
    normalize_debate_status_source,
)
from aragora.rbac.decorators import require_permission
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)

logger = logging.getLogger(__name__)


def _coerce_non_negative_int(value: Any) -> int:
    """Best-effort int coercion for round/token metadata."""

    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _coerce_float(value: Any, fallback: float = 0.0) -> float:
    """Best-effort float coercion for cost metadata."""

    if isinstance(value, bool):
        return fallback
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return fallback
    return fallback


def _normalize_string_list(values: Any) -> list[str]:
    """Return only string items from a list-like payload."""

    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, str)]


def _normalize_string_dict(values: Any) -> dict[str, str]:
    """Return only string-to-string pairs from dict-like payloads."""

    if not isinstance(values, dict):
        return {}

    normalized: dict[str, str] = {}
    for key, value in values.items():
        if isinstance(key, str) and isinstance(value, str):
            normalized[key] = value
    return normalized


def _normalize_float_dict(values: Any) -> dict[str, float]:
    """Return only string-to-float pairs from dict-like payloads."""

    if not isinstance(values, dict):
        return {}

    normalized: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            continue
        coerced = _coerce_float(value, fallback=float("nan"))
        if coerced == coerced:
            normalized[key] = coerced
    return normalized


def _normalize_provider_routing(value: Any) -> dict[str, Any] | None:
    """Preserve provider-routing metadata in a stable, JSON-safe shape."""

    if not isinstance(value, dict):
        return None

    routing: dict[str, Any] = {}

    if "routing_applied" in value:
        routing["routing_applied"] = bool(value.get("routing_applied"))

    strategy = value.get("routing_strategy")
    if isinstance(strategy, str) and strategy:
        routing["routing_strategy"] = strategy

    routed_agent_names = _normalize_string_list(value.get("routed_agent_names"))
    if routed_agent_names:
        routing["routed_agent_names"] = routed_agent_names

    provider_matches = _normalize_string_dict(value.get("provider_matches"))
    if provider_matches:
        routing["provider_matches"] = provider_matches

    provider_hint_scores = _normalize_float_dict(value.get("provider_hint_scores"))
    if provider_hint_scores:
        routing["provider_hint_scores"] = provider_hint_scores

    return routing or None


def _build_arguments(messages: Any) -> tuple[list[dict[str, Any]], int]:
    """Convert stored debate messages into detail-page argument entries."""

    if not isinstance(messages, list):
        return [], 0

    arguments: list[dict[str, Any]] = []
    rounds = 0
    for message in messages:
        if not isinstance(message, dict):
            continue

        round_num = _coerce_non_negative_int(message.get("round"))
        rounds = max(rounds, round_num)

        agent = message.get("agent") or message.get("author") or message.get("role") or "unknown"
        if not isinstance(agent, str):
            agent = str(agent)

        position = message.get("position") or message.get("role") or ""
        if not isinstance(position, str):
            position = str(position)

        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        arguments.append(
            {
                "agent": agent,
                "round": round_num,
                "position": position,
                "content": content,
            }
        )

    return arguments, rounds


def _build_cost_breakdown(
    per_agent_cost: Any,
    per_agent_tokens: Any | None = None,
) -> list[dict[str, Any]]:
    """Convert per-agent cost maps into the live UI's array form."""

    if not isinstance(per_agent_cost, dict):
        return []

    token_map = per_agent_tokens if isinstance(per_agent_tokens, dict) else {}
    breakdown: list[dict[str, Any]] = []
    for agent_name, cost in per_agent_cost.items():
        breakdown.append(
            {
                "agent": str(agent_name),
                "tokens": _coerce_non_negative_int(token_map.get(agent_name, 0)),
                "cost": float(cost) if isinstance(cost, int | float) else 0.0,
            }
        )

    return breakdown


def _extract_receipt_cost_maps(
    cost_summary: Any,
) -> tuple[float | None, dict[str, float], dict[str, int]]:
    """Normalize receipt cost_summary into the decision package cost fields."""

    if not isinstance(cost_summary, dict):
        return None, {}, {}

    total_cost = (
        _coerce_float(cost_summary.get("total_cost_usd"))
        if "total_cost_usd" in cost_summary
        else None
    )

    per_agent = cost_summary.get("per_agent")
    if not isinstance(per_agent, dict):
        return total_cost, {}, {}

    per_agent_cost: dict[str, float] = {}
    per_agent_tokens: dict[str, int] = {}
    for agent_name, raw_agent_data in per_agent.items():
        agent_key = str(agent_name)
        if isinstance(raw_agent_data, dict):
            if "total_cost_usd" in raw_agent_data:
                per_agent_cost[agent_key] = _coerce_float(raw_agent_data.get("total_cost_usd"))
            elif "cost" in raw_agent_data:
                per_agent_cost[agent_key] = _coerce_float(raw_agent_data.get("cost"))

            if "total_tokens" in raw_agent_data:
                per_agent_tokens[agent_key] = _coerce_non_negative_int(
                    raw_agent_data.get("total_tokens")
                )
            else:
                per_agent_tokens[agent_key] = _coerce_non_negative_int(
                    raw_agent_data.get("total_tokens_in")
                ) + _coerce_non_negative_int(raw_agent_data.get("total_tokens_out"))
        else:
            per_agent_cost[agent_key] = _coerce_float(raw_agent_data)

    return total_cost, per_agent_cost, per_agent_tokens


def _receipt_lookup_candidates(debate_id: str) -> list[str]:
    """Return canonical then legacy receipt lookup keys for a debate."""

    candidates = [debate_id]
    if debate_id and not debate_id.startswith("debate-"):
        candidates.append(f"debate-{debate_id}")
    return candidates


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

    provider_names = _normalize_string_list(package.get("provider_names"))
    provider_routing = package.get("provider_routing")
    if provider_names or isinstance(provider_routing, dict):
        lines.append("## Provider Routing")
        lines.append("")
        if provider_names:
            lines.append(f"- **Providers:** {', '.join(provider_names)}")

        if isinstance(provider_routing, dict):
            strategy = provider_routing.get("routing_strategy")
            if isinstance(strategy, str) and strategy:
                lines.append(f"- **Strategy:** {strategy}")

            routed_agents = _normalize_string_list(provider_routing.get("routed_agent_names"))
            if routed_agents:
                lines.append(f"- **Routed Agents:** {', '.join(routed_agents)}")

            provider_matches = _normalize_string_dict(provider_routing.get("provider_matches"))
            if provider_matches:
                lines.append("- **Agent → Provider:**")
                for agent, provider in provider_matches.items():
                    lines.append(f"  - {agent}: {provider}")

            provider_hint_scores = _normalize_float_dict(
                provider_routing.get("provider_hint_scores")
            )
            if provider_hint_scores:
                lines.append("- **Hint Scores:**")
                for provider, score in provider_hint_scores.items():
                    lines.append(f"  - {provider}: {score:.2f}")
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
        receipt_total_cost: float | None = None
        receipt_per_agent_cost: dict[str, float] = {}
        receipt_per_agent_tokens: dict[str, int] = {}
        try:
            from aragora.storage.receipt_store import get_receipt_store

            store = get_receipt_store()
            receipt = None
            for gauntlet_id in _receipt_lookup_candidates(debate_id):
                receipt = store.get_by_gauntlet(gauntlet_id)
                if receipt:
                    break
            if receipt:
                receipt_created_at = str(receipt.created_at) if receipt.created_at else None
                receipt_cost_summary = getattr(receipt, "cost_summary", None)
                if not isinstance(receipt_cost_summary, dict):
                    receipt_cost_summary = None

                receipt_dict = {
                    "receipt_id": receipt.receipt_id,
                    "verdict": receipt.verdict,
                    "confidence": receipt.confidence,
                    "risk_level": receipt.risk_level,
                    "risk_score": getattr(receipt, "risk_score", None),
                    "checksum": receipt.checksum,
                    "created_at": receipt_created_at,
                    # Frontend compatibility aliases for the live receipt tab.
                    "hash": receipt.checksum,
                    "timestamp": receipt_created_at,
                    "signers": [],
                }
                if receipt_cost_summary:
                    receipt_dict["cost_summary"] = receipt_cost_summary
                    (
                        receipt_total_cost,
                        receipt_per_agent_cost,
                        receipt_per_agent_tokens,
                    ) = _extract_receipt_cost_maps(receipt_cost_summary)
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

        metadata = result_data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        debate_status = normalize_debate_status(
            result_data.get("debate_status") or status,
            default=DebateStatus.COMPLETED,
        ).value
        source_hint = (
            result_data.get("debate_status_source")
            or result_data.get("status_source")
            or metadata.get("debate_status_source")
            or metadata.get("status_source")
            or debate.get("debate_status_source")
            or debate.get("status_source")
            or ("synthetic" if result_data.get("synthetic") or metadata.get("synthetic") else "")
            or result_data.get("mode")
            or metadata.get("mode")
        )
        debate_status_source = normalize_debate_status_source(
            source_hint,
            default=DebateStatusSource.LIVE,
        ).value
        synthetic = debate_status_source == DebateStatusSource.SYNTHETIC.value

        # -- Cost --
        per_agent_cost = result_data.get("per_agent_cost", {})
        if not isinstance(per_agent_cost, dict):
            per_agent_cost = {}
        if not per_agent_cost and receipt_per_agent_cost:
            per_agent_cost = receipt_per_agent_cost

        per_agent_tokens = result_data.get("per_agent_tokens")
        if not isinstance(per_agent_tokens, dict):
            per_agent_tokens = {}
        if not per_agent_tokens and receipt_per_agent_tokens:
            per_agent_tokens = receipt_per_agent_tokens

        if "total_cost_usd" in result_data:
            total_cost_usd = _coerce_float(result_data.get("total_cost_usd"))
            if total_cost_usd <= 0.0 and receipt_total_cost is not None:
                total_cost_usd = receipt_total_cost
        elif receipt_total_cost is not None:
            total_cost_usd = receipt_total_cost
        else:
            total_cost_usd = 0.0

        cost = {
            "total_cost_usd": total_cost_usd,
            "per_agent_cost": per_agent_cost,
        }
        cost_breakdown = _build_cost_breakdown(per_agent_cost, per_agent_tokens)

        # -- Next steps --
        consensus_reached = result_data.get("consensus_reached", False)
        question = debate.get("question", "")
        next_steps = _generate_next_steps(verdict, confidence, consensus_reached, question)
        participants = _normalize_string_list(
            result_data.get("participants", debate.get("agents", []))
        )
        provider_names = _normalize_string_list(
            result_data.get("provider_names", metadata.get("provider_names", []))
        )
        provider_hints = _normalize_string_list(
            result_data.get("provider_hints", metadata.get("provider_hints", []))
        )
        provider_routing = _normalize_provider_routing(
            result_data.get("provider_routing", metadata.get("provider_routing"))
        )
        arguments, rounds = _build_arguments(debate.get("messages", []))
        assembled_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        created_at = debate.get("created_at") or result_data.get("created_at") or assembled_at

        package: dict[str, Any] = {
            "debate_id": debate_id,
            # Frontend-compatible aliases kept alongside the historical contract.
            "id": debate_id,
            "question": question,
            "status": status,
            "debate_status": debate_status,
            "debate_status_source": debate_status_source,
            "synthetic": synthetic,
            "verdict": verdict,
            "confidence": confidence,
            "consensus_reached": consensus_reached,
            "final_answer": result_data.get("final_answer", ""),
            "explanation_summary": result_data.get("explanation_summary", ""),
            "explanation": result_data.get("explanation_summary", ""),
            "participants": participants,
            "agents": participants,
            "provider_names": provider_names,
            "provider_hints": provider_hints,
            "provider_routing": provider_routing,
            "rounds": result_data.get("rounds", rounds),
            "arguments": arguments,
            "receipt": receipt_dict,
            "cost": cost,
            "cost_breakdown": cost_breakdown,
            "total_cost": cost["total_cost_usd"],
            "argument_map": argument_map,
            "next_steps": next_steps,
            "export_formats": ["json", "markdown", "csv", "html", "txt"],
            "created_at": created_at,
            "duration_seconds": result_data.get(
                "duration_seconds", debate.get("duration_seconds", 0.0)
            ),
            "assembled_at": assembled_at,
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
