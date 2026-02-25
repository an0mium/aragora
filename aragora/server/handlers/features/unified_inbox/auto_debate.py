"""Auto-spawn debate for high-priority inbox messages.

Connects the Unified Inbox to the debate engine, allowing
multi-agent deliberation on how to handle critical messages.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


async def auto_spawn_debate_for_message(
    message: Any,
    factory: Any,
    tenant_id: str,
) -> dict[str, Any]:
    """Spawn a light debate to triage a high-priority inbox message.

    Creates a 3-round light debate asking agents to recommend how
    to handle the message (respond urgently, respond normally,
    delegate, or schedule later).

    Args:
        message: UnifiedMessage instance with priority_tier, sender_email, subject etc.
        factory: DebateFactory instance for creating arenas.
        tenant_id: Tenant context for the debate.

    Returns:
        Dict with debate_id, final_answer, consensus_reached, confidence.
    """
    from aragora.server.debate_factory import DebateConfig

    # Build triage question from message metadata
    priority_tier = getattr(message, "priority_tier", "medium")
    sender = getattr(message, "sender_email", "unknown sender") or "unknown sender"
    subject = getattr(message, "subject", "no subject") or "no subject"
    snippet = getattr(message, "snippet", "") or ""

    question = (
        f"Analyze this {priority_tier}-priority message from {sender} "
        f're: "{subject}". '
    )
    if snippet:
        question += f'Message preview: "{snippet[:200]}". '
    question += (
        "Recommend one of: "
        "(1) respond urgently, "
        "(2) respond normally, "
        "(3) delegate to team member, "
        "(4) schedule for later."
    )

    debate_id = f"inbox-triage-{uuid4().hex[:12]}"

    config = DebateConfig(
        question=question,
        debate_format="light",
        debate_id=debate_id,
        metadata={
            "source": "unified_inbox",
            "tenant_id": tenant_id,
            "message_id": getattr(message, "id", ""),
            "priority_tier": priority_tier,
        },
    )

    try:
        arena = factory.create_arena(config)
        result = await arena.run()

        final_answer = getattr(result, "final_answer", None) or ""
        consensus_reached = getattr(result, "consensus_reached", False)
        confidence = getattr(result, "confidence", 0.0)

        logger.info(
            "Inbox triage debate %s completed: consensus=%s confidence=%.2f",
            debate_id,
            consensus_reached,
            confidence,
        )

        return {
            "debate_id": debate_id,
            "final_answer": final_answer,
            "consensus_reached": consensus_reached,
            "confidence": confidence,
        }
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        logger.warning("Inbox triage debate failed for %s: %s", debate_id, e)
        return {
            "debate_id": debate_id,
            "final_answer": "",
            "consensus_reached": False,
            "confidence": 0.0,
            "error": "Debate creation failed",
        }
