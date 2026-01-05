"""
Belief Network and Reasoning endpoint handlers.

Endpoints:
- GET /api/belief-network/:debate_id/cruxes - Get key claims that impact debate outcome
- GET /api/belief-network/:debate_id/load-bearing-claims - Get high-centrality claims
- GET /api/provenance/:debate_id/claims/:claim_id/support - Get claim verification status
- GET /api/laboratory/emergent-traits - Get emergent traits from agent performance
- GET /api/debate/:debate_id/graph-stats - Get argument graph statistics
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    get_float_param,
)

logger = logging.getLogger(__name__)

# Safe ID pattern for path segments
SAFE_ID_PATTERN = r'^[a-zA-Z0-9_-]+$'

# Lazy imports for optional dependencies
BELIEF_NETWORK_AVAILABLE = False
LABORATORY_AVAILABLE = False
PROVENANCE_AVAILABLE = False
BeliefNetwork = None
BeliefPropagationAnalyzer = None
PersonaLaboratory = None
ProvenanceTracker = None

try:
    from aragora.reasoning.belief import BeliefNetwork as _BN, BeliefPropagationAnalyzer as _BPA
    BeliefNetwork = _BN
    BeliefPropagationAnalyzer = _BPA
    BELIEF_NETWORK_AVAILABLE = True
except ImportError:
    pass

try:
    from aragora.agents.laboratory import PersonaLaboratory as _PL
    PersonaLaboratory = _PL
    LABORATORY_AVAILABLE = True
except ImportError:
    pass

try:
    from aragora.reasoning.provenance import ProvenanceTracker as _PT
    ProvenanceTracker = _PT
    PROVENANCE_AVAILABLE = True
except ImportError:
    pass


def _safe_error_message(e: Exception, context: str = "") -> str:
    """Return a sanitized error message for client responses."""
    logger.error(f"Error in {context}: {type(e).__name__}: {e}", exc_info=True)
    error_type = type(e).__name__
    if error_type in ("FileNotFoundError", "OSError"):
        return "Resource not found"
    elif error_type in ("json.JSONDecodeError", "ValueError"):
        return "Invalid data format"
    elif error_type in ("TimeoutError", "asyncio.TimeoutError"):
        return "Operation timed out"
    return "An error occurred"


class BeliefHandler(BaseHandler):
    """Handler for belief network and reasoning endpoints."""

    ROUTES = [
        "/api/laboratory/emergent-traits",
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
        # Get nomic_dir from server context
        nomic_dir = self.ctx.get("nomic_dir")
        persona_manager = self.ctx.get("persona_manager")

        if path == "/api/laboratory/emergent-traits":
            min_confidence = get_float_param(query_params, 'min_confidence', 0.5)
            min_confidence = max(0.0, min(1.0, min_confidence))
            limit = get_int_param(query_params, 'limit', 10)
            return self._get_emergent_traits(nomic_dir, persona_manager, min_confidence, min(limit, 50))

        if path.startswith("/api/belief-network/") and path.endswith("/cruxes"):
            debate_id = self._extract_debate_id(path, 3)
            if debate_id is None:
                return error_response("Invalid debate_id", 400)
            top_k = get_int_param(query_params, 'top_k', 3)
            return self._get_debate_cruxes(nomic_dir, debate_id, min(top_k, 10))

        if path.startswith("/api/belief-network/") and path.endswith("/load-bearing-claims"):
            debate_id = self._extract_debate_id(path, 3)
            if debate_id is None:
                return error_response("Invalid debate_id", 400)
            limit = get_int_param(query_params, 'limit', 5)
            return self._get_load_bearing_claims(nomic_dir, debate_id, min(limit, 20))

        if "/claims/" in path and path.endswith("/support"):
            # Pattern: /api/provenance/:debate_id/claims/:claim_id/support
            parts = path.split('/')
            if len(parts) >= 6:
                debate_id = parts[3]
                claim_id = parts[5]
                if not re.match(SAFE_ID_PATTERN, debate_id) or not re.match(SAFE_ID_PATTERN, claim_id):
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
        parts = path.split('/')
        if len(parts) > segment_index:
            debate_id = parts[segment_index]
            if re.match(SAFE_ID_PATTERN, debate_id):
                return debate_id
        return None

    def _get_emergent_traits(
        self, nomic_dir: Optional[Path], persona_manager, min_confidence: float, limit: int
    ) -> HandlerResult:
        """Get emergent traits detected from agent performance patterns."""
        if not LABORATORY_AVAILABLE:
            return error_response("Persona laboratory not available", 503)

        try:
            lab = PersonaLaboratory(
                db_path=str(nomic_dir / "laboratory.db") if nomic_dir else None,
                persona_manager=persona_manager,
            )
            traits = lab.detect_emergent_traits()
            filtered = [t for t in traits if t.confidence >= min_confidence][:limit]
            return json_response({
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
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "emergent_traits"), 500)

    def _get_debate_cruxes(
        self, nomic_dir: Optional[Path], debate_id: str, top_k: int
    ) -> HandlerResult:
        """Get key claims that would most impact the debate outcome."""
        if not BELIEF_NETWORK_AVAILABLE:
            return error_response("Belief network not available", 503)

        try:
            from aragora.debate.traces import DebateTrace

            if not nomic_dir:
                return error_response("Nomic directory not configured", 503)

            trace_path = nomic_dir / "traces" / f"{debate_id}.json"
            if not trace_path.exists():
                return error_response("Debate trace not found", 404)

            trace = DebateTrace.load(trace_path)
            result = trace.to_debate_result()

            # Build belief network from debate
            network = BeliefNetwork(debate_id=debate_id)
            for msg in result.messages:
                network.add_claim(msg.agent, msg.content[:200], confidence=0.7)

            analyzer = BeliefPropagationAnalyzer(network)
            cruxes = analyzer.identify_debate_cruxes(top_k=top_k)

            return json_response({
                "debate_id": debate_id,
                "cruxes": cruxes,
                "count": len(cruxes),
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "debate_cruxes"), 500)

    def _get_load_bearing_claims(
        self, nomic_dir: Optional[Path], debate_id: str, limit: int
    ) -> HandlerResult:
        """Get claims with highest centrality (most load-bearing)."""
        if not BELIEF_NETWORK_AVAILABLE:
            return error_response("Belief network not available", 503)

        try:
            from aragora.debate.traces import DebateTrace

            if not nomic_dir:
                return error_response("Nomic directory not configured", 503)

            trace_path = nomic_dir / "traces" / f"{debate_id}.json"
            if not trace_path.exists():
                return error_response("Debate trace not found", 404)

            trace = DebateTrace.load(trace_path)
            result = trace.to_debate_result()

            # Build belief network from debate
            network = BeliefNetwork(debate_id=debate_id)
            for msg in result.messages:
                network.add_claim(msg.agent, msg.content[:200], confidence=0.7)

            load_bearing = network.get_load_bearing_claims(limit=limit)

            return json_response({
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
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "load_bearing_claims"), 500)

    def _get_claim_support(
        self, nomic_dir: Optional[Path], debate_id: str, claim_id: str
    ) -> HandlerResult:
        """Get verification status of all evidence supporting a claim."""
        if not PROVENANCE_AVAILABLE:
            return error_response("Provenance tracker not available", 503)

        try:
            if not nomic_dir:
                return error_response("Nomic directory not configured", 503)

            provenance_path = nomic_dir / "provenance" / f"{debate_id}.json"
            if not provenance_path.exists():
                return json_response({
                    "debate_id": debate_id,
                    "claim_id": claim_id,
                    "support": None,
                    "message": "No provenance data for this debate"
                })

            tracker = ProvenanceTracker.load(provenance_path)
            support = tracker.get_claim_support(claim_id)

            return json_response({
                "debate_id": debate_id,
                "claim_id": claim_id,
                "support": support,
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "claim_support"), 500)

    def _get_debate_graph_stats(
        self, nomic_dir: Optional[Path], debate_id: str
    ) -> HandlerResult:
        """Get argument graph statistics for a debate."""
        try:
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
                                event = json.loads(line)
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
            result = trace.to_debate_result()

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
                    round_num=getattr(critique, 'round', 1),
                    critique_text=critique.reasoning,
                )

            stats = cartographer.get_statistics()
            return json_response(stats)

        except Exception as e:
            return error_response(_safe_error_message(e, "debate_graph_stats"), 500)
