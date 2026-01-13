"""
Belief Network and Reasoning endpoint handlers.

Endpoints:
- GET /api/belief-network/:debate_id/cruxes - Get key claims that impact debate outcome
- GET /api/belief-network/:debate_id/load-bearing-claims - Get high-centrality claims
- GET /api/provenance/:debate_id/claims/:claim_id/support - Get claim verification status
- GET /api/laboratory/emergent-traits - Get emergent traits from agent performance
- GET /api/debate/:debate_id/graph-stats - Get argument graph statistics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_clamped_int_param,
    get_bounded_float_param,
    handle_errors,
)
from .utils.rate_limit import RateLimiter, get_client_ip
from aragora.server.validation import validate_id, validate_debate_id
from aragora.utils.optional_imports import try_import

logger = logging.getLogger(__name__)

# Rate limiter for belief network endpoints (60 requests per minute - read-heavy)
_belief_limiter = RateLimiter(requests_per_minute=60)

# Lazy imports for optional dependencies using centralized utility
_belief_imports, BELIEF_NETWORK_AVAILABLE = try_import(
    "aragora.reasoning.belief", "BeliefNetwork", "BeliefPropagationAnalyzer"
)
BeliefNetwork = _belief_imports["BeliefNetwork"]
BeliefPropagationAnalyzer = _belief_imports["BeliefPropagationAnalyzer"]

_lab_imports, LABORATORY_AVAILABLE = try_import("aragora.agents.laboratory", "PersonaLaboratory")
PersonaLaboratory = _lab_imports["PersonaLaboratory"]

_prov_imports, PROVENANCE_AVAILABLE = try_import(
    "aragora.reasoning.provenance", "ProvenanceTracker"
)
ProvenanceTracker = _prov_imports["ProvenanceTracker"]


class BeliefHandler(BaseHandler):
    """Handler for belief network and reasoning endpoints."""

    ROUTES: list[str] = [
        # Note: /api/laboratory/emergent-traits handled by LaboratoryHandler
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle dynamic routes
        if path.startswith("/api/belief-network/") and path.endswith("/cruxes"):
            return True
        if path.startswith("/api/belief-network/") and path.endswith("/load-bearing-claims"):
            return True
        if "/claims/" in path and path.endswith("/support"):
            return True
        if path.startswith("/api/debate/") and path.endswith("/graph-stats"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route belief network requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _belief_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for belief endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Get nomic_dir from server context
        nomic_dir = self.ctx.get("nomic_dir")
        persona_manager = self.ctx.get("persona_manager")

        # Note: /api/laboratory/emergent-traits handled by LaboratoryHandler

        if path.startswith("/api/belief-network/") and path.endswith("/cruxes"):
            debate_id = self._extract_debate_id(path, 3)
            if debate_id is None:
                return error_response("Invalid debate_id", 400)
            top_k = get_clamped_int_param(query_params, "top_k", 3, min_val=1, max_val=10)
            return self._get_debate_cruxes(nomic_dir, debate_id, top_k)

        if path.startswith("/api/belief-network/") and path.endswith("/load-bearing-claims"):
            debate_id = self._extract_debate_id(path, 3)
            if debate_id is None:
                return error_response("Invalid debate_id", 400)
            limit = get_clamped_int_param(query_params, "limit", 5, min_val=1, max_val=20)
            return self._get_load_bearing_claims(nomic_dir, debate_id, limit)

        if "/claims/" in path and path.endswith("/support"):
            # Pattern: /api/provenance/:debate_id/claims/:claim_id/support
            parts = path.split("/")
            if len(parts) >= 6:
                debate_id = parts[3]
                claim_id = parts[5]
                valid_debate, _ = validate_debate_id(debate_id)
                valid_claim, _ = validate_id(claim_id, "claim ID")
                if not valid_debate or not valid_claim:
                    return error_response("Invalid ID format", 400)
                return self._get_claim_support(nomic_dir, debate_id, claim_id)
            return error_response("Invalid path format", 400)

        if path.startswith("/api/debate/") and path.endswith("/graph-stats"):
            debate_id = self._extract_debate_id(path, 3)
            if debate_id is None:
                return error_response("Invalid debate_id", 400)
            return self._get_debate_graph_stats(nomic_dir, debate_id)

        return None

    def _extract_debate_id(self, path: str, segment_index: int) -> Optional[str]:
        """Extract and validate debate ID from path."""
        parts = path.split("/")
        if len(parts) > segment_index:
            debate_id = parts[segment_index]
            is_valid, _ = validate_debate_id(debate_id)
            if is_valid:
                return debate_id
        return None

    @handle_errors("emergent traits retrieval")
    def _get_emergent_traits(
        self, nomic_dir: Optional[Path], persona_manager, min_confidence: float, limit: int
    ) -> HandlerResult:
        """Get emergent traits detected from agent performance patterns."""
        if not LABORATORY_AVAILABLE:
            return error_response("Persona laboratory not available", 503)

        lab = PersonaLaboratory(
            db_path=str(nomic_dir / "laboratory.db") if nomic_dir else None,
            persona_manager=persona_manager,
        )
        traits = lab.detect_emergent_traits()
        filtered = [t for t in traits if t.confidence >= min_confidence][:limit]
        return json_response(
            {
                "emergent_traits": [
                    {
                        "agent": t.agent_name,
                        "trait": t.trait_name,
                        "domain": t.domain,
                        "confidence": t.confidence,
                        "evidence": t.evidence,
                        "detected_at": t.detected_at,
                    }
                    for t in filtered
                ],
                "count": len(filtered),
                "min_confidence": min_confidence,
            }
        )

    @handle_errors("debate cruxes retrieval")
    def _get_debate_cruxes(
        self, nomic_dir: Optional[Path], debate_id: str, top_k: int
    ) -> HandlerResult:
        """Get key claims that would most impact the debate outcome."""
        if not BELIEF_NETWORK_AVAILABLE:
            return error_response("Belief network not available", 503)

        from aragora.debate.traces import DebateTrace

        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        trace_path = nomic_dir / "traces" / f"{debate_id}.json"
        if not trace_path.exists():
            return error_response("Debate trace not found", 404)

        trace = DebateTrace.load(trace_path)
        result = trace.to_debate_result()  # type: ignore[attr-defined]

        # Build belief network from debate
        network = BeliefNetwork(debate_id=debate_id)
        for msg in result.messages:
            network.add_claim(msg.agent, msg.content[:200], confidence=0.7)

        analyzer = BeliefPropagationAnalyzer(network)
        cruxes = analyzer.identify_debate_cruxes(top_k=top_k)

        return json_response(
            {
                "debate_id": debate_id,
                "cruxes": cruxes,
                "count": len(cruxes),
            }
        )

    @handle_errors("load bearing claims retrieval")
    def _get_load_bearing_claims(
        self, nomic_dir: Optional[Path], debate_id: str, limit: int
    ) -> HandlerResult:
        """Get claims with highest centrality (most load-bearing)."""
        if not BELIEF_NETWORK_AVAILABLE:
            return error_response("Belief network not available", 503)

        from aragora.debate.traces import DebateTrace

        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        trace_path = nomic_dir / "traces" / f"{debate_id}.json"
        if not trace_path.exists():
            return error_response("Debate trace not found", 404)

        trace = DebateTrace.load(trace_path)
        result = trace.to_debate_result()  # type: ignore[attr-defined]

        # Build belief network from debate
        network = BeliefNetwork(debate_id=debate_id)
        for msg in result.messages:
            network.add_claim(msg.agent, msg.content[:200], confidence=0.7)

        load_bearing = network.get_load_bearing_claims(limit=limit)

        return json_response(
            {
                "debate_id": debate_id,
                "load_bearing_claims": [
                    {
                        "claim_id": node.claim_id,
                        "statement": node.claim_statement,
                        "author": node.author,
                        "centrality": centrality,
                    }
                    for node, centrality in load_bearing
                ],
                "count": len(load_bearing),
            }
        )

    @handle_errors("claim support retrieval")
    def _get_claim_support(
        self, nomic_dir: Optional[Path], debate_id: str, claim_id: str
    ) -> HandlerResult:
        """Get verification status of all evidence supporting a claim."""
        if not PROVENANCE_AVAILABLE:
            return error_response("Provenance tracker not available", 503)

        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        provenance_path = nomic_dir / "provenance" / f"{debate_id}.json"
        if not provenance_path.exists():
            return json_response(
                {
                    "debate_id": debate_id,
                    "claim_id": claim_id,
                    "support": None,
                    "message": "No provenance data for this debate",
                }
            )

        tracker = ProvenanceTracker.load(provenance_path)
        support = tracker.get_claim_support(claim_id)

        return json_response(
            {
                "debate_id": debate_id,
                "claim_id": claim_id,
                "support": support,
            }
        )

    @handle_errors("debate graph stats retrieval")
    def _get_debate_graph_stats(self, nomic_dir: Optional[Path], debate_id: str) -> HandlerResult:
        """Get argument graph statistics for a debate."""
        from aragora.visualization.mapper import ArgumentCartographer
        from aragora.debate.traces import DebateTrace

        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        trace_path = nomic_dir / "traces" / f"{debate_id}.json"
        if not trace_path.exists():
            # Try replays directory as fallback
            replay_path = nomic_dir / "replays" / debate_id / "events.jsonl"
            if replay_path.exists():
                cartographer = ArgumentCartographer()
                cartographer.set_debate_context(debate_id, "")
                with replay_path.open() as f:
                    for line in f:
                        if line.strip():
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue  # Skip malformed event lines
                            if event.get("type") == "agent_message":
                                cartographer.update_from_message(
                                    agent=event.get("agent", "unknown"),
                                    content=event.get("data", {}).get("content", ""),
                                    role=event.get("data", {}).get("role", "proposer"),
                                    round_num=event.get("round", 1),
                                )
                            elif event.get("type") == "critique":
                                cartographer.update_from_critique(
                                    critic_agent=event.get("agent", "unknown"),
                                    target_agent=event.get("data", {}).get("target", "unknown"),
                                    severity=event.get("data", {}).get("severity", 0.5),
                                    round_num=event.get("round", 1),
                                    critique_text=event.get("data", {}).get("content", ""),
                                )
                stats = cartographer.get_statistics()
                return json_response(stats)
            else:
                return error_response("Debate not found", 404)

        # Load from trace file
        trace = DebateTrace.load(trace_path)
        result = trace.to_debate_result()  # type: ignore[attr-defined]

        # Build cartographer from debate result
        cartographer = ArgumentCartographer()
        cartographer.set_debate_context(debate_id, result.task or "")

        for msg in result.messages:
            cartographer.update_from_message(
                agent=msg.agent,
                content=msg.content,
                role=msg.role,
                round_num=msg.round,
            )

        for critique in result.critiques:
            cartographer.update_from_critique(
                critic_agent=critique.agent,
                target_agent=critique.target or "",
                severity=critique.severity,
                round_num=getattr(critique, "round", 1),
                critique_text=critique.reasoning,
            )

        stats = cartographer.get_statistics()
        return json_response(stats)
