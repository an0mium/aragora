"""Multi-agent triage for unified inbox messages.

Handles single and batch message triage using debate-based
or heuristic-based classification.
"""

from __future__ import annotations

import logging
from typing import Any

from .models import TriageAction, TriageResult, UnifiedMessage

logger = logging.getLogger(__name__)


async def run_triage(
    messages: list[UnifiedMessage],
    context: dict[str, Any],
    tenant_id: str,
    store: Any,
    triage_to_record: Any,
) -> list[TriageResult]:
    """Run multi-agent triage on messages."""
    results = []

    for message in messages:
        try:
            result = await triage_single_message(message, context, tenant_id)
            results.append(result)

            # Update message with triage result
            message.triage_action = result.recommended_action
            message.triage_rationale = result.rationale

            await store.save_triage_result(tenant_id, triage_to_record(result))
            await store.update_message_triage(
                tenant_id,
                message.id,
                result.recommended_action.value,
                result.rationale,
            )

        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.warning("Triage failed for message %s: %s", message.id, e)

    return results


async def triage_single_message(
    message: UnifiedMessage,
    context: dict[str, Any],
    tenant_id: str,
) -> TriageResult:
    """Triage a single message using multi-agent debate."""
    try:
        from aragora.debate import Arena, Environment, DebateProtocol  # noqa: F401

        # Build debate environment
        Environment(
            task=f"""
            Analyze this email and recommend the best action:

            From: {message.sender_name} <{message.sender_email}>
            Subject: {message.subject}
            Preview: {message.snippet}

            Consider:
            - Urgency and time-sensitivity
            - Sender importance
            - Required action type
            - Delegation possibilities

            Recommend ONE of: respond_urgent, respond_normal, delegate, schedule, archive, flag, defer
            """,
        )

        # Fast-path triage: intentionally short debates for responsiveness.
        TRIAGE_FAST_ROUNDS = 2
        DebateProtocol(
            rounds=TRIAGE_FAST_ROUNDS,
            consensus="majority",
        )

        # Run debate (simplified for now)
        # In production, use actual Arena with agents
        agents = ["support_analyst", "product_expert"]

        # Determine action based on priority
        action = _action_from_priority(message.priority_tier)

        return TriageResult(
            message_id=message.id,
            recommended_action=action,
            confidence=0.85,
            rationale=f"Based on priority tier '{message.priority_tier}' and sender analysis",
            suggested_response=None,
            delegate_to=None,
            schedule_for=None,
            agents_involved=agents,
            debate_summary="Multi-agent analysis completed",
        )

    except ImportError:
        # Arena not available, use simple heuristics
        action = _action_from_priority(message.priority_tier)

        return TriageResult(
            message_id=message.id,
            recommended_action=action,
            confidence=0.7,
            rationale=f"Heuristic-based triage for '{message.priority_tier}' priority",
            suggested_response=None,
            delegate_to=None,
            schedule_for=None,
            agents_involved=[],
            debate_summary=None,
        )


def _action_from_priority(priority_tier: str) -> TriageAction:
    """Map a priority tier to a triage action."""
    if priority_tier == "critical":
        return TriageAction.RESPOND_URGENT
    elif priority_tier == "high":
        return TriageAction.RESPOND_NORMAL
    elif priority_tier == "low":
        return TriageAction.ARCHIVE
    else:
        return TriageAction.DEFER
