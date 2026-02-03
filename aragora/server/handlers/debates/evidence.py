"""
Evidence and analysis operations handler mixin.

Extracted from handler.py for modularity. Provides:
- Get impasse detection
- Get convergence status
- Get evidence citations
- Get comprehensive evidence trail
- Get verification report
- Get debate summary
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from aragora.exceptions import (
    DatabaseError,
    RecordNotFoundError,
    StorageError,
)

from aragora.rbac.decorators import require_permission

from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_storage,
    safe_json_parse,
    ttl_cache,
)
from ..openapi_decorator import api_endpoint
from .response_formatting import (
    CACHE_TTL_CONVERGENCE,
    CACHE_TTL_IMPASSE,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class _DebatesHandlerProtocol(Protocol):
    """Protocol defining the interface expected by EvidenceOperationsMixin.

    This protocol enables proper type checking for mixin classes that
    expect to be mixed into a class providing these methods/attributes.
    """

    ctx: dict[str, Any]

    def get_storage(self) -> Any | None:
        """Get debate storage instance."""
        ...


class EvidenceOperationsMixin:
    """Mixin providing evidence and analysis operations for DebatesHandler."""

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{id}/impasse",
        summary="Detect debate impasse",
        description="Analyze a debate for impasse indicators like repeated critiques and lack of convergence.",
        tags=["Debates", "Analysis"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {
                "description": "Impasse analysis returned",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_id": {"type": "string"},
                                "is_impasse": {"type": "boolean"},
                                "indicators": {
                                    "type": "object",
                                    "properties": {
                                        "repeated_critiques": {"type": "boolean"},
                                        "no_convergence": {"type": "boolean"},
                                        "high_severity_critiques": {"type": "boolean"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "404": {"description": "Debate not found"},
        },
    )
    @require_permission("debates:read")
    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_IMPASSE, key_prefix="debates_impasse", skip_first=True)
    @handle_errors("impasse detection")
    def _get_impasse(self: _DebatesHandlerProtocol, handler: Any, debate_id: str) -> HandlerResult:
        """Detect impasse in a debate."""
        storage = self.get_storage()
        debate = storage.get_debate(debate_id)
        if not debate:
            return error_response(f"Debate not found: {debate_id}", 404)

        # Analyze for impasse indicators
        critiques = debate.get("critiques", [])

        # Simple impasse detection: repetitive critiques without progress
        impasse_indicators = {
            "repeated_critiques": False,
            "no_convergence": not debate.get("consensus_reached", False),
            "high_severity_critiques": any(c.get("severity", 0) > 0.7 for c in critiques),
        }

        is_impasse = sum(impasse_indicators.values()) >= 2

        return json_response(
            {
                "debate_id": debate_id,
                "is_impasse": is_impasse,
                "indicators": impasse_indicators,
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{id}/convergence",
        summary="Get convergence status",
        description="Get the convergence status including similarity scores and consensus state.",
        tags=["Debates", "Analysis"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {
                "description": "Convergence status returned",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_id": {"type": "string"},
                                "convergence_status": {"type": "string"},
                                "convergence_similarity": {"type": "number"},
                                "consensus_reached": {"type": "boolean"},
                                "rounds_used": {"type": "integer"},
                            },
                        },
                    },
                },
            },
            "404": {"description": "Debate not found"},
        },
    )
    @require_permission("debates:read")
    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_CONVERGENCE, key_prefix="debates_convergence", skip_first=True)
    @handle_errors("convergence check")
    def _get_convergence(
        self: _DebatesHandlerProtocol, handler: Any, debate_id: str
    ) -> HandlerResult:
        """Get convergence status for a debate."""
        storage = self.get_storage()
        debate = storage.get_debate(debate_id)
        if not debate:
            return error_response(f"Debate not found: {debate_id}", 404)

        return json_response(
            {
                "debate_id": debate_id,
                "convergence_status": debate.get("convergence_status", "unknown"),
                "convergence_similarity": debate.get("convergence_similarity", 0.0),
                "consensus_reached": debate.get("consensus_reached", False),
                "rounds_used": debate.get("rounds_used", 0),
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{id}/verification-report",
        summary="Get verification report",
        description="Get verification results and bonuses applied during consensus, useful for analyzing claim quality.",
        tags=["Debates", "Verification"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {
                "description": "Verification report returned",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_id": {"type": "string"},
                                "verification_enabled": {"type": "boolean"},
                                "verification_results": {"type": "object"},
                                "verification_bonuses": {"type": "object"},
                                "summary": {"type": "object"},
                                "winner": {"type": "string"},
                                "consensus_reached": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
            "404": {"description": "Debate not found"},
        },
    )
    @require_permission("debates:read")
    @require_storage
    @ttl_cache(
        ttl_seconds=CACHE_TTL_CONVERGENCE, key_prefix="debates_verification", skip_first=True
    )
    @handle_errors("verification report")
    def _get_verification_report(
        self: _DebatesHandlerProtocol, handler: Any, debate_id: str
    ) -> HandlerResult:
        """Get verification report for a debate.

        Returns verification results and bonuses applied during consensus,
        useful for analyzing claim quality and feedback loop effectiveness.
        """
        storage = self.get_storage()
        debate = storage.get_debate(debate_id)
        if not debate:
            return error_response(f"Debate not found: {debate_id}", 404)

        verification_results = debate.get("verification_results", {})
        verification_bonuses = debate.get("verification_bonuses", {})

        # Calculate summary stats
        total_verified = sum(v for v in verification_results.values() if v > 0)
        agents_verified = sum(1 for v in verification_results.values() if v > 0)
        total_bonus = sum(verification_bonuses.values())

        return json_response(
            {
                "debate_id": debate_id,
                "verification_enabled": bool(verification_results),
                "verification_results": verification_results,
                "verification_bonuses": verification_bonuses,
                "summary": {
                    "total_verified_claims": total_verified,
                    "agents_with_verified_claims": agents_verified,
                    "total_bonus_applied": round(total_bonus, 3),
                },
                "winner": debate.get("winner"),
                "consensus_reached": debate.get("consensus_reached", False),
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{id}/summary",
        summary="Get debate summary",
        description="Get a human-readable summary with verdict, key points, confidence assessment, and actionable next steps.",
        tags=["Debates"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {
                "description": "Debate summary returned",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_id": {"type": "string"},
                                "summary": {"type": "object"},
                                "task": {"type": "string"},
                                "consensus_reached": {"type": "boolean"},
                                "confidence": {"type": "number"},
                            },
                        },
                    },
                },
            },
            "404": {"description": "Debate not found"},
        },
    )
    @require_permission("debates:read")
    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_CONVERGENCE, key_prefix="debates_summary", skip_first=True)
    @handle_errors("get summary")
    def _get_summary(self: _DebatesHandlerProtocol, handler: Any, debate_id: str) -> HandlerResult:
        """Get human-readable summary for a debate.

        Returns a structured summary with:
        - One-liner verdict
        - Key points and conclusions
        - Agreement and disagreement areas
        - Confidence assessment
        - Actionable next steps
        """
        from aragora.debate.summarizer import summarize_debate

        storage = self.get_storage()
        debate = storage.get_debate(debate_id)
        if not debate:
            return error_response(f"Debate not found: {debate_id}", 404)

        # Generate summary
        summary = summarize_debate(debate)

        return json_response(
            {
                "debate_id": debate_id,
                "summary": summary.to_dict(),
                "task": debate.get("task", ""),
                "consensus_reached": debate.get("consensus_reached", False),
                "confidence": debate.get("confidence", 0.0),
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{id}/citations",
        summary="Get evidence citations",
        description="Get the grounded verdict including claims, evidence snippets, and citation sources.",
        tags=["Debates", "Evidence"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {
                "description": "Evidence citations returned",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_id": {"type": "string"},
                                "has_citations": {"type": "boolean"},
                                "message": {"type": "string"},
                                "grounded_verdict": {"type": "object"},
                                "grounding_score": {"type": "number"},
                                "confidence": {"type": "number"},
                                "claims": {"type": "array", "items": {"type": "object"}},
                                "all_citations": {"type": "array", "items": {"type": "object"}},
                                "verdict": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "404": {"description": "Debate not found"},
            "500": {"description": "Database error"},
        },
    )
    @require_permission("evidence:read")
    @require_storage
    def _get_citations(
        self: _DebatesHandlerProtocol, handler: Any, debate_id: str
    ) -> HandlerResult:
        """Get evidence citations for a debate.

        Returns the grounded verdict including:
        - Claims extracted from final answer
        - Evidence snippets linked to each claim
        - Overall grounding score
        - Full citation list with sources
        """

        storage = self.get_storage()
        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Check if grounded_verdict is stored
            grounded_verdict_raw = debate.get("grounded_verdict")

            if not grounded_verdict_raw:
                return json_response(
                    {
                        "debate_id": debate_id,
                        "has_citations": False,
                        "message": "No evidence citations available for this debate",
                        "grounded_verdict": None,
                    }
                )

            # Parse grounded_verdict JSON if it's a string
            grounded_verdict = safe_json_parse(grounded_verdict_raw)

            if not grounded_verdict:
                return json_response(
                    {
                        "debate_id": debate_id,
                        "has_citations": False,
                        "message": "Evidence citations could not be parsed",
                        "grounded_verdict": None,
                    }
                )

            return json_response(
                {
                    "debate_id": debate_id,
                    "has_citations": True,
                    "grounding_score": grounded_verdict.get("grounding_score", 0),
                    "confidence": grounded_verdict.get("confidence", 0),
                    "claims": grounded_verdict.get("claims", []),
                    "all_citations": grounded_verdict.get("all_citations", []),
                    "verdict": grounded_verdict.get("verdict", ""),
                }
            )

        except RecordNotFoundError:
            logger.info("Citations failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Failed to get citations for %s: %s: %s",
                debate_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error retrieving citations", 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{id}/evidence",
        summary="Get evidence trail",
        description="Get comprehensive evidence trail combining grounded verdict with related evidence from memory.",
        tags=["Debates", "Evidence"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {
                "description": "Evidence trail returned",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_id": {"type": "string"},
                                "task": {"type": "string"},
                                "has_evidence": {"type": "boolean"},
                                "grounded_verdict": {"type": "object"},
                                "claims": {"type": "array", "items": {"type": "object"}},
                                "citations": {"type": "array", "items": {"type": "object"}},
                                "related_evidence": {"type": "array", "items": {"type": "object"}},
                                "evidence_count": {"type": "integer"},
                            },
                        },
                    },
                },
            },
            "404": {"description": "Debate not found"},
            "500": {"description": "Database error"},
        },
    )
    @require_permission("evidence:read")
    @require_storage
    def _get_evidence(self: _DebatesHandlerProtocol, handler: Any, debate_id: str) -> HandlerResult:
        """Get comprehensive evidence trail for a debate.

        Combines grounded verdict with related evidence from ContinuumMemory.

        Returns:
            - grounded_verdict: Claim analysis with citations
            - related_evidence: Evidence snippets from memory
            - metadata: Search context and quality metrics
        """

        storage = self.get_storage()

        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Get grounded verdict from debate
            grounded_verdict = safe_json_parse(debate.get("grounded_verdict"))

            # Try to get related evidence from ContinuumMemory
            related_evidence = []
            task = debate.get("task", "")

            try:
                continuum = self.ctx.get("continuum_memory")
                if continuum and task and hasattr(continuum, "search"):
                    # Query for evidence-type memories related to this task
                    memories = continuum.search(
                        query=task[:200],
                        limit=10,
                        min_importance=0.3,
                    )

                    # Filter to evidence type
                    for memory in memories:
                        metadata = getattr(memory, "metadata", {}) or {}
                        if metadata.get("type") == "evidence":
                            related_evidence.append(
                                {
                                    "id": getattr(memory, "id", ""),
                                    "content": getattr(memory, "content", ""),
                                    "source": metadata.get("source", "unknown"),
                                    "importance": getattr(memory, "importance", 0.5),
                                    "tier": str(getattr(memory, "tier", "medium")),
                                }
                            )
            except Exception as e:
                logger.debug(f"Could not fetch ContinuumMemory evidence: {e}")

            # Build response
            response: dict[str, Any] = {
                "debate_id": debate_id,
                "task": task,
                "has_evidence": bool(grounded_verdict or related_evidence),
            }

            if grounded_verdict:
                response["grounded_verdict"] = {
                    "grounding_score": grounded_verdict.get("grounding_score", 0),
                    "confidence": grounded_verdict.get("confidence", 0),
                    "claims_count": len(grounded_verdict.get("claims", [])),
                    "citations_count": len(grounded_verdict.get("all_citations", [])),
                    "verdict": grounded_verdict.get("verdict", ""),
                }
                response["claims"] = grounded_verdict.get("claims", [])
                response["citations"] = grounded_verdict.get("all_citations", [])
            else:
                response["grounded_verdict"] = None
                response["claims"] = []
                response["citations"] = []

            response["related_evidence"] = related_evidence
            response["evidence_count"] = len(related_evidence)

            return json_response(response)

        except RecordNotFoundError:
            logger.info("Evidence failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Failed to get evidence for %s: %s: %s",
                debate_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error retrieving evidence", 500)


__all__ = ["EvidenceOperationsMixin"]
