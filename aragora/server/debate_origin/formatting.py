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
