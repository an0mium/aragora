"""Debate origin message formatting utilities.

Functions for formatting debate results, receipts, and error messages
for display on chat platforms.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import DebateOrigin


def _format_result_message(
    result: dict[str, Any],
    origin: DebateOrigin,
    markdown: bool = True,
    html: bool = False,
) -> str:
    """Format debate result as a message."""
    # Capability events are pre-formatted; return as-is
    if result.get("_capability_event"):
        return result.get("final_answer", "")

    event_type = result.get("event")
    if (
        event_type
        in {
            "decision_integrity",
            "decision_plan",
        }
        or "package" in result
    ):
        package = result.get("package") or result.get("decision_integrity") or {}
        receipt = package.get("receipt") or {}
        plan = package.get("plan") or {}
        context_snapshot = package.get("context_snapshot") or {}
        debate_id = (
            package.get("debate_id") or result.get("debate_id") or origin.debate_id or "unknown"
        )
        topic = package.get("task") or origin.metadata.get("topic", "Unknown topic")
        tasks = plan.get("tasks") or []
        task_count = len(tasks)

        receipt_verdict = receipt.get("verdict")
        receipt_confidence = receipt.get("confidence")
        risk_summary = receipt.get("risk_summary") or {}
        critical = risk_summary.get("critical")
        high = risk_summary.get("high")

        lines: list[str] = []
        title = (
            "Decision Plan Ready"
            if event_type == "decision_plan"
            else "Decision Integrity Package Ready"
        )
        lines.append(f"**{title}**")
        lines.append("")
        lines.append(f"**Topic:** {str(topic)[:200]}")
        lines.append(f"**Debate:** `{str(debate_id)[:12]}...`")

        if receipt_verdict:
            verdict_line = f"**Receipt:** {receipt_verdict}"
            if isinstance(receipt_confidence, (int, float)):
                verdict_line += f" ({receipt_confidence:.0%})"
            if critical is not None or high is not None:
                verdict_line += f" | critical: {critical or 0}, high: {high or 0}"
            lines.append(verdict_line)

        if task_count:
            lines.append(f"**Plan:** {task_count} tasks")
            for task in tasks[:3]:
                desc = str(task.get("description", "")).strip()
                if desc:
                    lines.append(f"- {desc[:160]}")
            if task_count > 3:
                lines.append(f"- ...and {task_count - 3} more")

        total_tokens = context_snapshot.get("total_context_tokens")
        if isinstance(total_tokens, int) and total_tokens > 0:
            lines.append(f"**Context snapshot:** ~{total_tokens} tokens")

        execution = package.get("execution") or package.get("workflow_execution")
        if isinstance(execution, dict):
            status = execution.get("status")
            if status:
                mode = execution.get("mode")
                reason = execution.get("reason")
                status_line = f"**Execution:** {status}"
                if mode:
                    status_line += f" ({mode})"
                if reason:
                    status_line += f" - {reason}"
                lines.append(status_line)

        message = "\n".join(lines)
        if html:
            return message.replace("\n", "<br>")
        return message

    if event_type in {"execution_progress", "execution_complete"}:
        progress = result.get("progress") or result.get("summary") or {}
        pct = progress.get("progress_pct", 0.0)
        completed = progress.get("completed_tasks", 0)
        failed = progress.get("failed_tasks", 0)
        total = progress.get("total_tasks", 0)
        title = "Execution Progress" if event_type == "execution_progress" else "Execution Complete"
        if html:
            return (
                f"<strong>{title}</strong><br>"
                f"Tasks: {completed}/{total} (failed: {failed})<br>"
                f"Progress: {pct}%"
            )
        if markdown:
            return f"**{title}**\n\nTasks: {completed}/{total} (failed: {failed})\nProgress: {pct}%"
        return f"{title}\nTasks: {completed}/{total} (failed: {failed})\nProgress: {pct}%"

    consensus = result.get("consensus_reached", False)
    answer = result.get("final_answer", "No conclusion reached.")
    confidence = result.get("confidence", 0)
    participants = result.get("participants", [])
    topic = result.get("task", origin.metadata.get("topic", "Unknown topic"))

    # Truncate long answers
    if len(answer) > 800:
        answer = answer[:800] + "..."

    if html:
        return f"""
<h2>Debate Complete!</h2>
<p><strong>Topic:</strong> {topic[:200]}</p>
<p><strong>Consensus:</strong> {"Yes" if consensus else "No"}</p>
<p><strong>Confidence:</strong> {confidence:.0%}</p>
<p><strong>Agents:</strong> {", ".join(participants[:5])}</p>
<hr>
<p><strong>Conclusion:</strong></p>
<p>{answer}</p>
"""

    if markdown:
        return f"""**Debate Complete!**

**Topic:** {topic[:200]}

**Consensus:** {"Yes" if consensus else "No"}
**Confidence:** {confidence:.0%}
**Agents:** {", ".join(participants[:5])}

---

**Conclusion:**
{answer}
"""

    # Plain text
    return f"""Debate Complete!

Topic: {topic[:200]}

Consensus: {"Yes" if consensus else "No"}
Confidence: {confidence:.0%}
Agents: {", ".join(participants[:5])}

---

Conclusion:
{answer}
"""


def _format_receipt_summary(receipt: Any, url: str) -> str:
    """Create compact receipt summary for chat platforms.

    Args:
        receipt: DecisionReceipt object
        url: URL to view full receipt

    Returns:
        Formatted summary string
    """
    emoji_map = {
        "APPROVED": "\u2705",
        "APPROVED_WITH_CONDITIONS": "\u26a0\ufe0f",
        "NEEDS_REVIEW": "\U0001f50d",
        "REJECTED": "\u274c",
    }
    emoji = emoji_map.get(receipt.verdict, "\U0001f4cb")

    cost_line = ""
    cost_value = None
    if hasattr(receipt, "cost_usd"):
        try:
            cost_value = float(receipt.cost_usd)
        except (TypeError, ValueError):
            cost_value = None
    if cost_value is not None and cost_value > 0:
        cost_line = f"\n\u2022 Cost: ${cost_value:.4f}"
        if hasattr(receipt, "budget_limit_usd") and receipt.budget_limit_usd:
            try:
                budget_value = float(receipt.budget_limit_usd)
            except (TypeError, ValueError):
                budget_value = None
            if budget_value:
                pct = (cost_value / budget_value) * 100
                cost_line += f" ({pct:.0f}% of budget)"

    return f"""{emoji} **Decision Receipt**
\u2022 Verdict: {receipt.verdict}
\u2022 Confidence: {receipt.confidence:.0%}
\u2022 Findings: {receipt.critical_count} critical, {receipt.high_count} high{cost_line}
\u2022 [View Full Receipt]({url})"""


def format_error_for_chat(error: str, debate_id: str) -> str:
    """Map technical errors to user-friendly messages for chat platforms.

    Converts internal error messages into helpful, non-technical messages
    that guide users on what to do next.

    Args:
        error: The technical error message
        debate_id: The debate ID for reference

    Returns:
        User-friendly error message string
    """
    # Map technical patterns to friendly messages
    error_map = {
        "rate limit": ("Your request is being processed. Results will arrive shortly."),
        "429": (
            "Our AI agents are experiencing high demand. "
            "Your request is queued and will complete shortly."
        ),
        "timeout": ("There was a delay processing your request. Please wait a moment."),
        "timed out": (
            "The analysis is taking longer than expected. Results will be sent when ready."
        ),
        "not found": ("We couldn't find that debate. Please start a new one."),
        "404": ("The requested resource wasn't found. Please try again."),
        "unauthorized": ("Please reconnect the Aragora app to continue."),
        "401": ("Authentication required. Please reconnect the Aragora app."),
        "forbidden": (
            "You don't have permission for this action. Please check with your workspace admin."
        ),
        "403": ("Access denied. Please verify your permissions."),
        "connection": ("We're experiencing connectivity issues. Please try again in a moment."),
        "service unavailable": ("Our service is temporarily busy. Please try again shortly."),
        "503": ("Service temporarily unavailable. Please retry in a few moments."),
        "internal": ("Something went wrong on our end. We're looking into it."),
        "500": ("An unexpected error occurred. Please try again."),
        "budget": (
            "This request would exceed your organization's budget limit. "
            "Please contact your admin to increase the limit."
        ),
        "quota": (
            "You've reached your usage quota for this period. "
            "Usage resets at the start of the next billing cycle."
        ),
        "invalid": ("The request couldn't be processed. Please check your input and try again."),
    }

    error_lower = error.lower()
    for tech_pattern, friendly_msg in error_map.items():
        if tech_pattern in error_lower:
            return f"message: {friendly_msg}\n\n_Debate ID: {debate_id}_"

    # Default fallback
    return (
        f"We encountered an issue processing your request. Please try again.\n\n"
        f"_Debate ID: {debate_id}_"
    )


# ---------------------------------------------------------------------------
# Capability-specific formatters for channel routing
# ---------------------------------------------------------------------------


def format_consensus_event(result: dict[str, Any]) -> str:
    """Format a consensus detection result for channel delivery."""
    consensus = result.get("consensus_reached", False)
    method = result.get("method", "unknown")
    confidence = result.get("confidence", 0.0)
    participants = result.get("participants", [])
    topic = result.get("topic", result.get("task", ""))

    status = "Consensus Reached" if consensus else "No Consensus"
    lines = [f"**{status}**"]
    if topic:
        lines.append(f"**Topic:** {str(topic)[:200]}")
    if isinstance(confidence, (int, float)):
        lines.append(f"**Confidence:** {confidence:.0%}")
    lines.append(f"**Method:** {method}")
    if participants:
        lines.append(f"**Agents:** {', '.join(str(p) for p in participants[:5])}")

    proof = result.get("proof")
    if isinstance(proof, dict):
        hash_val = proof.get("hash", "")[:12]
        if hash_val:
            lines.append(f"**Proof:** `{hash_val}...`")

    return "\n".join(lines)


def format_compliance_event(result: dict[str, Any]) -> str:
    """Format a compliance check result for channel delivery."""
    compliant = result.get("compliant", True)
    score = result.get("score", 1.0)
    frameworks = result.get("frameworks_checked", [])
    issues = result.get("issues", [])

    status = "Compliant" if compliant else "Non-Compliant"
    icon = "passed" if compliant else "FAILED"
    lines = [f"**Compliance Check: {icon}**"]
    lines.append(f"**Status:** {status}")
    if isinstance(score, (int, float)):
        lines.append(f"**Score:** {score:.0%}")
    if frameworks:
        lines.append(f"**Frameworks:** {', '.join(str(f) for f in frameworks[:5])}")

    if issues:
        critical = sum(1 for i in issues if i.get("severity") == "critical")
        high = sum(1 for i in issues if i.get("severity") == "high")
        lines.append(f"**Issues:** {len(issues)} total ({critical} critical, {high} high)")
        for issue in issues[:3]:
            desc = str(issue.get("description", ""))[:120]
            sev = issue.get("severity", "info")
            lines.append(f"  - [{sev}] {desc}")
        if len(issues) > 3:
            lines.append(f"  - ...and {len(issues) - 3} more")

    return "\n".join(lines)


def format_knowledge_event(result: dict[str, Any]) -> str:
    """Format a knowledge mound event for channel delivery."""
    event_type = result.get("km_event", result.get("event", "update"))
    item_count = result.get("item_count", 0)
    topic = result.get("topic", "")
    source = result.get("source", "")

    title_map = {
        "ingestion_complete": "Knowledge Ingested",
        "staleness_detected": "Stale Knowledge Detected",
        "contradiction_found": "Knowledge Contradiction Found",
        "search_complete": "Knowledge Search Complete",
    }
    title = title_map.get(event_type, "Knowledge Update")
    lines = [f"**{title}**"]
    if topic:
        lines.append(f"**Topic:** {str(topic)[:200]}")
    if source:
        lines.append(f"**Source:** {source}")
    if item_count:
        lines.append(f"**Items:** {item_count}")

    items = result.get("items", [])
    for item in items[:3]:
        title_val = item.get("title", item.get("id", ""))
        score = item.get("relevance_score", item.get("score"))
        line = f"  - {str(title_val)[:100]}"
        if isinstance(score, (int, float)):
            line += f" ({score:.0%})"
        lines.append(line)

    return "\n".join(lines)


def format_graph_debate_event(result: dict[str, Any]) -> str:
    """Format a graph debate result for channel delivery."""
    status = result.get("status", "complete")
    node_count = result.get("node_count", 0)
    edge_count = result.get("edge_count", 0)
    topic = result.get("topic", result.get("task", ""))
    conclusion = result.get("conclusion", result.get("final_answer", ""))

    lines = [f"**Graph Debate {status.title()}**"]
    if topic:
        lines.append(f"**Topic:** {str(topic)[:200]}")
    if node_count or edge_count:
        lines.append(f"**Graph:** {node_count} claims, {edge_count} connections")
    if conclusion:
        lines.append(f"**Conclusion:** {str(conclusion)[:300]}")

    return "\n".join(lines)


def format_workflow_event(result: dict[str, Any]) -> str:
    """Format a workflow engine event for channel delivery."""
    event_type = result.get("wf_event", result.get("event", "update"))
    workflow_name = result.get("workflow_name", result.get("name", ""))
    status = result.get("status", "running")

    title_map = {
        "workflow_started": "Workflow Started",
        "step_completed": "Workflow Step Complete",
        "workflow_completed": "Workflow Complete",
        "workflow_failed": "Workflow Failed",
        "approval_required": "Approval Required",
    }
    title = title_map.get(event_type, f"Workflow {status.title()}")
    lines = [f"**{title}**"]
    if workflow_name:
        lines.append(f"**Workflow:** {workflow_name}")

    step = result.get("current_step", result.get("step"))
    if isinstance(step, dict):
        lines.append(f"**Step:** {step.get('name', step.get('id', ''))}")
    elif isinstance(step, str):
        lines.append(f"**Step:** {step}")

    completed = result.get("completed_steps", 0)
    total = result.get("total_steps", 0)
    if total:
        lines.append(f"**Progress:** {completed}/{total}")

    if event_type == "approval_required":
        lines.append("**Action needed:** Please approve or reject this step.")

    return "\n".join(lines)
