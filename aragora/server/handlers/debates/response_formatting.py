"""
Response formatting utilities for debate endpoints.

Provides status normalization and debate response formatting for SDK compatibility.
"""

from __future__ import annotations

from aragora.gauntlet.receipt_models import _normalize_receipt_boolean

# Status normalization map: converts internal status values to canonical SDK-compatible values
# Server uses: active, paused, concluded, archived
# SDKs expect: pending, running, completed, failed, cancelled, paused
STATUS_MAP = {
    "active": "running",
    "concluded": "completed",
    "archived": "completed",
    "starting": "created",
    "in_progress": "running",
}

# Reverse map for accepting SDK status values in updates
STATUS_REVERSE_MAP = {
    "running": "active",
    "completed": "concluded",
    "pending": "active",
    "created": "active",
    "in_progress": "active",
}

# Cache TTLs for debates endpoints (in seconds)
CACHE_TTL_DEBATES_LIST = 30  # Short TTL for list (may change frequently)
CACHE_TTL_SEARCH = 60  # Search results cache
CACHE_TTL_CONVERGENCE = 120  # Convergence status (changes less often)
CACHE_TTL_IMPASSE = 120  # Impasse detection


def normalize_status(status: str) -> str:
    """Normalize internal status to SDK-compatible canonical status.

    Args:
        status: Internal status value (active, paused, concluded, archived)

    Returns:
        Canonical status (pending, running, completed, failed, cancelled, paused)
    """
    return STATUS_MAP.get(status, status)


def denormalize_status(status: str) -> str:
    """Convert SDK status to internal status for storage.

    Args:
        status: SDK canonical status (running, completed, etc.)

    Returns:
        Internal status (active, concluded, etc.)
    """
    return STATUS_REVERSE_MAP.get(status, status)


def normalize_debate_response(debate: dict | None) -> dict | None:
    """Normalize debate dict for API response, ensuring SDK compatibility.

    Normalizes status values and ensures both field name variants are present
    for consensus fields (agreement/confidence, conclusion/final_answer).

    Args:
        debate: Raw debate dict from storage

    Returns:
        Normalized debate dict with SDK-compatible fields, or None if input is None
    """
    if debate is None:
        return None

    # Normalize status
    if "status" in debate:
        debate["status"] = normalize_status(debate["status"])
    else:
        debate["status"] = "completed"

    # Ensure debate_id/id aliases exist for SDK/front-end parity
    if "debate_id" in debate and "id" not in debate:
        debate["id"] = debate["debate_id"]
    if "id" in debate and "debate_id" not in debate:
        debate["debate_id"] = debate["id"]

    # Promote consensus_proof into consensus if needed
    if "consensus" not in debate and "consensus_proof" in debate:
        consensus_proof_raw = debate.get("consensus_proof")
        consensus_proof = consensus_proof_raw or {}
        vote_breakdown = consensus_proof.get("vote_breakdown") or {}
        reached = False
        if isinstance(consensus_proof_raw, dict) and "reached" in consensus_proof_raw:
            reached = consensus_proof_raw.get("reached")
        supporting_agents = [
            agent for agent, agreed in vote_breakdown.items() if _normalize_receipt_boolean(agreed)
        ]
        dissenting_agents = [
            agent
            for agent, agreed in vote_breakdown.items()
            if not _normalize_receipt_boolean(agreed)
        ]
        debate["consensus"] = {
            "reached": None if reached is None else _normalize_receipt_boolean(reached),
            "agreement": consensus_proof.get("confidence"),
            "confidence": consensus_proof.get("confidence"),
            "final_answer": consensus_proof.get("final_answer"),
            "conclusion": consensus_proof.get("final_answer"),
            "supporting_agents": supporting_agents,
            "dissenting_agents": dissenting_agents,
        }

    # consensus_reached/concordance helpers for UI
    consensus = debate.get("consensus") or {}
    if isinstance(consensus, dict) and "reached" in consensus:
        reached = consensus.get("reached")
        if reached is not None:
            consensus["reached"] = _normalize_receipt_boolean(reached)
    if "consensus_reached" not in debate:
        debate["consensus_reached"] = _normalize_receipt_boolean(consensus.get("reached"))
    if "confidence" not in debate:
        confidence = consensus.get("confidence", consensus.get("agreement"))
        if confidence is not None:
            debate["confidence"] = confidence

    # rounds_used defaults for list views
    if "rounds_used" not in debate:
        rounds_value = debate.get("rounds")
        if isinstance(rounds_value, int):
            debate["rounds_used"] = rounds_value
        elif isinstance(rounds_value, list):
            debate["rounds_used"] = len(rounds_value)
        else:
            debate["rounds_used"] = 0
    debate.setdefault("duration_seconds", 0)

    # Ensure consensus field aliases (for SDK compatibility)
    # confidence <-> agreement
    if "confidence" in debate and "agreement" not in debate:
        debate["agreement"] = debate["confidence"]
    elif "agreement" in debate and "confidence" not in debate:
        debate["confidence"] = debate["agreement"]

    # conclusion <-> final_answer
    if "conclusion" in debate and "final_answer" not in debate:
        debate["final_answer"] = debate["conclusion"]
    elif "final_answer" in debate and "conclusion" not in debate:
        debate["conclusion"] = debate["final_answer"]

    # winning_proposal alias for final_answer (used by frontend debates list)
    if "winning_proposal" not in debate:
        winning = debate.get("final_answer") or debate.get("conclusion")
        if winning:
            debate["winning_proposal"] = winning

    return debate


__all__ = [
    "CACHE_TTL_CONVERGENCE",
    "CACHE_TTL_DEBATES_LIST",
    "CACHE_TTL_IMPASSE",
    "CACHE_TTL_SEARCH",
    "STATUS_MAP",
    "STATUS_REVERSE_MAP",
    "denormalize_status",
    "normalize_debate_response",
    "normalize_status",
]
