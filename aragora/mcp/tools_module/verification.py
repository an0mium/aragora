"""
MCP Verification Tools.

Formal verification and consensus proofs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


async def get_consensus_proofs_tool(
    debate_id: str = "",
    proof_type: str = "all",
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Retrieve formal verification proofs from debates.

    Args:
        debate_id: Specific debate ID to get proofs for (optional)
        proof_type: Type of proofs ('z3', 'lean', 'all')
        limit: Max proofs to return

    Returns:
        Dict with proofs list and count
    """
    proofs: List[Dict[str, Any]] = []

    # Try to get proofs from debate data
    if not proofs and debate_id:
        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if db:
                debate = db.get(debate_id)
                if debate and "proofs" in debate:
                    for proof in debate["proofs"]:
                        if proof_type == "all" or proof.get("type") == proof_type:
                            proofs.append(proof)
        except Exception as e:
            logger.debug(f"Failed to fetch proofs for debate {debate_id}: {e}")

    proofs = proofs[:limit]

    return {
        "proofs": proofs,
        "count": len(proofs),
        "debate_id": debate_id or "(all debates)",
        "proof_type": proof_type,
    }


async def verify_consensus_tool(
    debate_id: str,
    backend: str = "z3",
) -> Dict[str, Any]:
    """
    Verify the consensus of a completed debate using formal methods.

    Args:
        debate_id: ID of the debate to verify
        backend: Verification backend (z3, lean4)

    Returns:
        Dict with verification result and proof
    """
    if not debate_id:
        return {"error": "debate_id is required"}

    try:
        from aragora.verification.formal import FormalVerificationManager

        manager = FormalVerificationManager()

        # Get debate data
        from aragora.server.storage import get_debates_db

        db = get_debates_db()
        if not db:
            return {"error": "Storage not available"}

        debate = db.get(debate_id)
        if not debate:
            return {"error": f"Debate {debate_id} not found"}

        # Extract consensus claim
        consensus = debate.get("final_answer", "")
        if not consensus:
            return {"error": "Debate has no consensus to verify"}

        # Verify
        result = await manager.attempt_formal_verification(
            claim=consensus,
            context=f"Debate consensus from {debate_id}",
        )

        return {
            "debate_id": debate_id,
            "status": (
                result.status.value if hasattr(result.status, "value") else str(result.status)
            ),
            "is_verified": result.is_verified,
            "language": result.language.value if hasattr(result.language, "value") else backend,
            "formal_statement": result.formal_statement,
            "proof_hash": result.proof_hash,
            "translation_time_ms": result.translation_time_ms,
            "proof_search_time_ms": result.proof_search_time_ms,
        }

    except ImportError:
        return {"error": "Verification module not available"}
    except Exception as e:
        return {"error": f"Verification failed: {e}"}


async def generate_proof_tool(
    claim: str,
    output_format: str = "lean4",
    context: str = "",
) -> Dict[str, Any]:
    """
    Generate a formal proof for a claim without verification.

    Args:
        claim: The claim to translate to formal language
        output_format: Output format (lean4, z3_smt)
        context: Additional context for translation

    Returns:
        Dict with formal statement and confidence
    """
    if not claim:
        return {"error": "claim is required"}

    try:
        from aragora.verification.formal import LeanBackend, Z3Backend

        backend: Union[LeanBackend, Z3Backend]
        if output_format == "lean4":
            backend = LeanBackend()
        else:
            backend = Z3Backend()

        formal_statement = await backend.translate(claim, context)

        return {
            "success": formal_statement is not None,
            "claim": claim,
            "formal_statement": formal_statement,
            "format": output_format,
            "confidence": 0.7 if formal_statement else 0.0,
        }

    except ImportError:
        return {"error": "Verification module not available"}
    except Exception as e:
        return {"error": f"Proof generation failed: {e}"}


__all__ = [
    "get_consensus_proofs_tool",
    "verify_consensus_tool",
    "generate_proof_tool",
]
