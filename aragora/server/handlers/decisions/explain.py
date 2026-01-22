"""
Decision Explainability endpoint handler.

Endpoint:
- GET /api/decisions/:request_id/explain - Get comprehensive decision explanation

This endpoint aggregates data from:
- ConsensusProof (votes, claims, dissents, tensions)
- BeliefNetwork (cruxes, load-bearing claims)
- ProvenanceChain (evidence sources)
- Audit trail (timing, participants)
"""

from __future__ import annotations

__all__ = ["DecisionExplainHandler"]

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

from aragora.server.validation import validate_id
from aragora.utils.optional_imports import try_import

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_bounded_string_param,
    handle_errors,
    json_response,
)
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter (30 requests per minute - explanation is read-heavy)
_explain_limiter = RateLimiter(requests_per_minute=30)

# Lazy imports for optional dependencies
_consensus_imports, CONSENSUS_AVAILABLE = try_import(
    "aragora.debate.consensus", "ConsensusProof", "VoteType"
)
ConsensusProof = _consensus_imports["ConsensusProof"]
VoteType = _consensus_imports["VoteType"]

_belief_imports, BELIEF_AVAILABLE = try_import(
    "aragora.reasoning.belief", "BeliefNetwork", "BeliefPropagationAnalyzer"
)
BeliefNetwork = _belief_imports["BeliefNetwork"]
BeliefPropagationAnalyzer = _belief_imports["BeliefPropagationAnalyzer"]

_claims_imports, CLAIMS_AVAILABLE = try_import("aragora.reasoning.claims", "ClaimsKernel")
ClaimsKernel = _claims_imports["ClaimsKernel"]


def _enum_to_value(obj: Any) -> Any:
    """Convert Enum values to their string representation."""
    if isinstance(obj, Enum):
        return obj.value
    return obj


def _serialize_dataclass(obj: Any) -> dict:
    """Serialize a dataclass to dict, handling Enum values."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            if isinstance(value, Enum):
                result[field_name] = value.value
            elif isinstance(value, list):
                result[field_name] = [_serialize_dataclass(item) for item in value]
            elif isinstance(value, datetime):
                result[field_name] = value.isoformat()
            elif hasattr(value, "__dataclass_fields__"):
                result[field_name] = _serialize_dataclass(value)
            else:
                result[field_name] = value
        return result
    return obj


class DecisionExplainHandler(BaseHandler):
    """Handler for decision explainability endpoints."""

    ROUTES: list[str] = [
        "/api/decisions/*/explain",
        "/api/v1/decisions/*/explain",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        # Handle /api/decisions/:request_id/explain pattern
        if path.startswith("/api/decisions/") and path.endswith("/explain"):
            return True
        if path.startswith("/api/v1/decisions/") and path.endswith("/explain"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route decision explain requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _explain_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for explain endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Get nomic_dir from server context
        nomic_dir = self.ctx.get("nomic_dir")

        # Extract request_id from path
        # Pattern: /api/decisions/:request_id/explain or /api/v1/decisions/:request_id/explain
        parts = path.split("/")
        if path.startswith("/api/v1/"):
            request_id = parts[4] if len(parts) > 4 else None
        else:
            request_id = parts[3] if len(parts) > 3 else None

        if not request_id:
            return error_response("Invalid request_id", 400)

        is_valid, err_msg = validate_id(request_id, "request ID")
        if not is_valid:
            return error_response(err_msg or "Invalid request ID format", 400)

        # Get format parameter
        format_type = get_bounded_string_param(
            query_params, "format", "json", max_length=10
        ).lower()

        return self._explain_decision(nomic_dir, request_id, format_type)

    @handle_errors("decision explanation")
    def _explain_decision(
        self, nomic_dir: Optional[Path], request_id: str, format_type: str
    ) -> HandlerResult:
        """Generate comprehensive decision explanation.

        Aggregates:
        - Decision summary (answer, confidence, consensus)
        - Key claims with evidence strength
        - Vote records with reasoning
        - Dissenting views and alternative perspectives
        - Unresolved tensions and tradeoffs
        - Audit trail (timing, agents, rounds)

        Returns JSON, Markdown, or HTML based on format parameter.
        """
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        # Try to load decision data from multiple sources
        explanation = self._build_explanation(nomic_dir, request_id)

        if explanation is None:
            return error_response(f"Decision {request_id} not found", 404)

        # Format response
        if format_type == "md" or format_type == "markdown":
            return self._format_markdown(explanation)
        elif format_type == "html":
            return self._format_html(explanation)
        else:
            return json_response(explanation)

    def _build_explanation(self, nomic_dir: Path, request_id: str) -> Optional[dict[str, Any]]:
        """Build comprehensive explanation from available data sources."""
        explanation: dict[str, Any] = {
            "request_id": request_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Try to load from trace file
        trace_path = nomic_dir / "traces" / f"{request_id}.json"
        debate_result = None

        if trace_path.exists():
            try:
                from aragora.debate.traces import DebateTrace

                trace = DebateTrace.load(trace_path)
                debate_result = trace.to_debate_result()
            except Exception as e:
                logger.warning(f"Failed to load trace for {request_id}: {e}")

        # Try to load from replays
        if debate_result is None:
            replay_path = nomic_dir / "replays" / request_id / "events.jsonl"
            if replay_path.exists():
                debate_result = self._load_from_replay(replay_path)

        if debate_result is None:
            # Try decision cache or storage
            debate_result = self._load_from_storage(request_id)

        if debate_result is None:
            return None

        # Build summary section
        explanation["summary"] = self._build_summary(debate_result)

        # Build reasoning section
        explanation["reasoning"] = self._build_reasoning(nomic_dir, request_id, debate_result)

        # Build votes section
        explanation["votes"] = self._build_votes(debate_result)

        # Build dissent section
        explanation["dissent"] = self._build_dissent(debate_result)

        # Build tensions section
        explanation["tensions"] = self._build_tensions(debate_result)

        # Build audit trail
        explanation["audit_trail"] = self._build_audit_trail(debate_result)

        return explanation

    def _build_summary(self, result: Any) -> dict[str, Any]:
        """Build decision summary."""
        summary = {
            "answer": getattr(result, "final_answer", None) or getattr(result, "answer", ""),
            "confidence": getattr(result, "confidence", 0.0),
            "consensus_reached": getattr(result, "consensus_reached", False),
            "agreement_ratio": 0.0,
            "has_strong_consensus": False,
        }

        # Calculate agreement ratio from consensus proof if available
        consensus_proof = getattr(result, "consensus_proof", None)
        if consensus_proof:
            summary["agreement_ratio"] = getattr(consensus_proof, "agreement_ratio", 0.0)
            summary["has_strong_consensus"] = getattr(
                consensus_proof, "has_strong_consensus", False
            )

        return summary

    def _build_reasoning(self, nomic_dir: Path, request_id: str, result: Any) -> dict[str, Any]:
        """Build reasoning section with key claims and evidence."""
        reasoning: dict[str, Any] = {
            "key_claims": [],
            "supporting_evidence": [],
            "crux_claims": [],
        }

        # Extract claims from consensus proof
        consensus_proof = getattr(result, "consensus_proof", None)
        if consensus_proof and CONSENSUS_AVAILABLE:
            claims = getattr(consensus_proof, "claims", [])
            for claim in claims[:10]:  # Top 10 claims
                claim_dict = _serialize_dataclass(claim)
                claim_dict["strength"] = getattr(claim, "net_evidence_strength", 0.5)
                reasoning["key_claims"].append(claim_dict)

            # Extract evidence
            evidence_chain = getattr(consensus_proof, "evidence_chain", [])
            for evidence in evidence_chain[:20]:  # Top 20 evidence items
                reasoning["supporting_evidence"].append(_serialize_dataclass(evidence))

        # Try to get crux claims from belief network
        if BELIEF_AVAILABLE:
            try:
                network = BeliefNetwork(debate_id=request_id)
                messages = getattr(result, "messages", [])
                for msg in messages:
                    content = getattr(msg, "content", "")
                    agent = getattr(msg, "agent", "unknown")
                    network.add_claim(agent, content[:200], confidence=0.7)

                analyzer = BeliefPropagationAnalyzer(network)
                cruxes = analyzer.identify_debate_cruxes(top_k=5)
                reasoning["crux_claims"] = cruxes
            except Exception as e:
                logger.debug(f"Failed to build belief network: {e}")

        return reasoning

    def _build_votes(self, result: Any) -> list[dict[str, Any]]:
        """Build votes section with agent reasoning."""
        votes = []

        consensus_proof = getattr(result, "consensus_proof", None)
        if consensus_proof:
            proof_votes = getattr(consensus_proof, "votes", [])
            for vote in proof_votes:
                vote_dict = _serialize_dataclass(vote)
                votes.append(vote_dict)

        # If no votes from consensus proof, try to infer from agent contributions
        if not votes:
            contributions = getattr(result, "agent_contributions", [])
            for contrib in contributions:
                if isinstance(contrib, dict):
                    votes.append(
                        {
                            "agent": contrib.get("agent", "unknown"),
                            "vote": "agree",  # Inferred
                            "confidence": 0.7,
                            "reasoning": contrib.get("response", "")[:500],
                        }
                    )

        return votes

    def _build_dissent(self, result: Any) -> dict[str, Any]:
        """Build dissent section with minority views."""
        dissent = {
            "dissenting_agents": [],
            "reasons": [],
            "alternative_views": [],
            "severity": 0.0,
        }

        consensus_proof = getattr(result, "consensus_proof", None)
        if consensus_proof:
            dissent["dissenting_agents"] = getattr(consensus_proof, "dissenting_agents", [])

            dissent_records = getattr(consensus_proof, "dissents", [])
            total_severity = 0.0
            for record in dissent_records:
                reasons = getattr(record, "reasons", [])
                dissent["reasons"].extend(reasons)

                alt_view = getattr(record, "alternative_view", None)
                if alt_view:
                    dissent["alternative_views"].append(
                        {
                            "agent": getattr(record, "agent", "unknown"),
                            "view": alt_view,
                            "suggested_resolution": getattr(record, "suggested_resolution", None),
                        }
                    )
                total_severity += getattr(record, "severity", 0.5)

            if dissent_records:
                dissent["severity"] = total_severity / len(dissent_records)

        return dissent

    def _build_tensions(self, result: Any) -> list[dict[str, Any]]:
        """Build tensions section with unresolved tradeoffs."""
        tensions = []

        consensus_proof = getattr(result, "consensus_proof", None)
        if consensus_proof:
            unresolved = getattr(consensus_proof, "unresolved_tensions", [])
            for tension in unresolved:
                tensions.append(_serialize_dataclass(tension))

        return tensions

    def _build_audit_trail(self, result: Any) -> dict[str, Any]:
        """Build audit trail section."""
        audit: dict[str, Any] = {
            "created_at": None,
            "duration_seconds": getattr(result, "duration_seconds", 0.0),
            "rounds_completed": getattr(result, "rounds_used", 0),
            "agents_involved": [],
            "checksum": None,
        }

        # Get timing info
        completed_at = getattr(result, "completed_at", None)
        if completed_at:
            if isinstance(completed_at, datetime):
                audit["created_at"] = completed_at.isoformat()
            else:
                audit["created_at"] = str(completed_at)

        # Get participants
        participants = getattr(result, "participants", [])
        if participants:
            audit["agents_involved"] = participants
        else:
            # Try to extract from messages
            messages = getattr(result, "messages", [])
            agents = set()
            for msg in messages:
                agent = getattr(msg, "agent", None)
                if agent:
                    agents.add(agent)
            audit["agents_involved"] = list(agents)

        # Get checksum from consensus proof
        consensus_proof = getattr(result, "consensus_proof", None)
        if consensus_proof:
            audit["checksum"] = getattr(consensus_proof, "checksum", None)
            audit["rounds_completed"] = getattr(
                consensus_proof, "rounds_to_consensus", audit["rounds_completed"]
            )

        return audit

    def _load_from_replay(self, replay_path: Path) -> Optional[Any]:
        """Load decision result from replay events."""
        try:
            from aragora.core import DebateResult, Message

            messages = []
            final_answer = None
            consensus_reached = False
            confidence = 0.0

            with replay_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "agent_message":
                        data = event.get("data", {})
                        messages.append(
                            Message(
                                agent=event.get("agent", "unknown"),
                                content=data.get("content", ""),
                                role=data.get("role", "proposer"),
                                round=event.get("round", 1),
                            )
                        )

                    elif event_type == "consensus":
                        data = event.get("data", {})
                        consensus_reached = data.get("reached", False)
                        confidence = data.get("confidence", 0.0)
                        final_answer = data.get("answer", "")

                    elif event_type == "debate_end":
                        data = event.get("data", {})
                        if not final_answer:
                            final_answer = data.get("answer", "")

            if messages:
                return DebateResult(
                    task=replay_path.parent.name,
                    final_answer=final_answer or "",
                    consensus_reached=consensus_reached,
                    confidence=confidence,
                    messages=messages,
                    rounds_used=max((m.round for m in messages), default=1),
                    participants=list({m.agent for m in messages}),
                )
        except Exception as e:
            logger.warning(f"Failed to load replay: {e}")

        return None

    def _load_from_storage(self, request_id: str) -> Optional[Any]:
        """Load decision result from storage or cache."""
        try:
            from aragora.core.decision_cache import get_decision_cache

            cache = get_decision_cache()
            if cache:
                # Note: This is a simplified lookup - actual implementation
                # would need to index by request_id
                pass
        except ImportError:
            pass

        return None

    def _format_markdown(self, explanation: dict[str, Any]) -> HandlerResult:
        """Format explanation as Markdown."""
        lines = [
            "# Decision Explanation",
            f"**Request ID:** {explanation['request_id']}",
            f"**Generated:** {explanation['generated_at']}",
            "",
        ]

        # Summary
        summary = explanation.get("summary", {})
        lines.extend(
            [
                "## Summary",
                f"**Answer:** {summary.get('answer', 'N/A')}",
                f"**Confidence:** {summary.get('confidence', 0):.1%}",
                f"**Consensus Reached:** {'Yes' if summary.get('consensus_reached') else 'No'}",
                f"**Agreement Ratio:** {summary.get('agreement_ratio', 0):.1%}",
                "",
            ]
        )

        # Key Claims
        reasoning = explanation.get("reasoning", {})
        claims = reasoning.get("key_claims", [])
        if claims:
            lines.extend(["## Key Claims", ""])
            for i, claim in enumerate(claims[:5], 1):
                statement = claim.get("statement", "")[:200]
                author = claim.get("author", "unknown")
                strength = claim.get("strength", 0.5)
                lines.append(f"{i}. **{author}:** {statement} (strength: {strength:.2f})")
            lines.append("")

        # Crux Claims
        cruxes = reasoning.get("crux_claims", [])
        if cruxes:
            lines.extend(["## Crux Claims (Decision-Pivotal)", ""])
            for crux in cruxes[:3]:
                claim_id = crux.get("claim_id", "")
                score = crux.get("crux_score", 0)
                lines.append(f"- {claim_id} (impact: {score:.2f})")
            lines.append("")

        # Votes
        votes = explanation.get("votes", [])
        if votes:
            lines.extend(["## Vote Record", ""])
            for vote in votes:
                agent = vote.get("agent", "unknown")
                vote_type = vote.get("vote", "unknown")
                confidence = vote.get("confidence", 0)
                reasoning_text = vote.get("reasoning", "")[:100]
                lines.append(f"- **{agent}**: {vote_type} ({confidence:.0%}) - {reasoning_text}")
            lines.append("")

        # Dissent
        dissent = explanation.get("dissent", {})
        dissenting = dissent.get("dissenting_agents", [])
        if dissenting:
            lines.extend(["## Dissenting Views", ""])
            lines.append(f"**Dissenting Agents:** {', '.join(dissenting)}")
            lines.append(f"**Average Severity:** {dissent.get('severity', 0):.1%}")
            lines.append("")
            for alt in dissent.get("alternative_views", [])[:3]:
                lines.append(f"- **{alt.get('agent', 'unknown')}:** {alt.get('view', '')[:200]}")
            lines.append("")

        # Tensions
        tensions = explanation.get("tensions", [])
        if tensions:
            lines.extend(["## Unresolved Tensions", ""])
            for tension in tensions[:3]:
                desc = tension.get("description", "")
                impact = tension.get("impact", "")
                lines.append(f"- {desc}")
                if impact:
                    lines.append(f"  - Impact: {impact}")
            lines.append("")

        # Audit Trail
        audit = explanation.get("audit_trail", {})
        lines.extend(
            [
                "## Audit Trail",
                f"- **Duration:** {audit.get('duration_seconds', 0):.2f}s",
                f"- **Rounds:** {audit.get('rounds_completed', 0)}",
                f"- **Agents:** {', '.join(audit.get('agents_involved', []))}",
            ]
        )
        if audit.get("checksum"):
            lines.append(f"- **Checksum:** {audit['checksum']}")

        content = "\n".join(lines)
        return HandlerResult(
            status=200,
            body=content.encode("utf-8"),
            headers={"Content-Type": "text/markdown; charset=utf-8"},
        )

    def _format_html(self, explanation: dict[str, Any]) -> HandlerResult:
        """Format explanation as HTML."""
        summary = explanation.get("summary", {})
        reasoning = explanation.get("reasoning", {})
        votes = explanation.get("votes", [])
        dissent = explanation.get("dissent", {})
        tensions = explanation.get("tensions", [])
        audit = explanation.get("audit_trail", {})

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Decision Explanation - {explanation['request_id']}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #1a1a1a; }}
        h2 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
        .summary {{ background: #f7f7f7; padding: 16px; border-radius: 8px; margin: 16px 0; }}
        .confidence {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
        .consensus-yes {{ color: #22c55e; }}
        .consensus-no {{ color: #ef4444; }}
        .claim {{ background: #fff; border: 1px solid #ddd; padding: 12px; margin: 8px 0; border-radius: 4px; }}
        .claim-author {{ font-weight: bold; color: #666; }}
        .claim-strength {{ float: right; color: #888; }}
        .vote {{ padding: 8px 0; border-bottom: 1px solid #eee; }}
        .vote-agent {{ font-weight: bold; }}
        .vote-type {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        .vote-agree {{ background: #dcfce7; color: #166534; }}
        .vote-disagree {{ background: #fee2e2; color: #991b1b; }}
        .dissent {{ background: #fef3c7; padding: 16px; border-radius: 8px; margin: 16px 0; }}
        .tension {{ background: #f0f9ff; padding: 12px; margin: 8px 0; border-radius: 4px; border-left: 4px solid #0ea5e9; }}
        .audit {{ font-family: monospace; background: #1a1a1a; color: #fff; padding: 16px; border-radius: 8px; }}
        .audit dt {{ color: #888; }}
        .audit dd {{ margin: 0 0 8px 0; }}
    </style>
</head>
<body>
    <h1>Decision Explanation</h1>
    <p><strong>Request ID:</strong> {explanation['request_id']}<br>
    <strong>Generated:</strong> {explanation['generated_at']}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Answer:</strong> {summary.get('answer', 'N/A')}</p>
        <p class="confidence">{summary.get('confidence', 0):.1%} Confidence</p>
        <p>Consensus: <span class="{'consensus-yes' if summary.get('consensus_reached') else 'consensus-no'}">
            {'Reached' if summary.get('consensus_reached') else 'Not Reached'}
        </span></p>
        <p>Agreement Ratio: {summary.get('agreement_ratio', 0):.1%}</p>
    </div>

    <h2>Key Claims</h2>
"""

        for claim in reasoning.get("key_claims", [])[:5]:
            html += f"""
    <div class="claim">
        <span class="claim-author">{claim.get('author', 'unknown')}:</span>
        <span class="claim-strength">Strength: {claim.get('strength', 0.5):.2f}</span>
        <p>{claim.get('statement', '')[:300]}</p>
    </div>
"""

        html += "<h2>Vote Record</h2>"
        for vote in votes:
            vote_type = vote.get("vote", "unknown")
            vote_class = "vote-agree" if vote_type == "agree" else "vote-disagree"
            html += f"""
    <div class="vote">
        <span class="vote-agent">{vote.get('agent', 'unknown')}</span>
        <span class="vote-type {vote_class}">{vote_type}</span>
        ({vote.get('confidence', 0):.0%})
        <p>{vote.get('reasoning', '')[:150]}</p>
    </div>
"""

        if dissent.get("dissenting_agents"):
            html += f"""
    <div class="dissent">
        <h2>Dissenting Views</h2>
        <p><strong>Dissenting Agents:</strong> {', '.join(dissent.get('dissenting_agents', []))}</p>
        <p><strong>Average Severity:</strong> {dissent.get('severity', 0):.1%}</p>
"""
            for alt in dissent.get("alternative_views", [])[:3]:
                html += f"<p><strong>{alt.get('agent', 'unknown')}:</strong> {alt.get('view', '')[:200]}</p>"
            html += "</div>"

        if tensions:
            html += "<h2>Unresolved Tensions</h2>"
            for tension in tensions[:3]:
                html += f"""
    <div class="tension">
        <p>{tension.get('description', '')}</p>
        <p><em>Impact: {tension.get('impact', 'Unknown')}</em></p>
    </div>
"""

        html += f"""
    <h2>Audit Trail</h2>
    <dl class="audit">
        <dt>Duration</dt><dd>{audit.get('duration_seconds', 0):.2f}s</dd>
        <dt>Rounds</dt><dd>{audit.get('rounds_completed', 0)}</dd>
        <dt>Agents</dt><dd>{', '.join(audit.get('agents_involved', []))}</dd>
        <dt>Checksum</dt><dd>{audit.get('checksum', 'N/A')}</dd>
    </dl>
</body>
</html>
"""

        return HandlerResult(
            status=200,
            body=html.encode("utf-8"),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )
