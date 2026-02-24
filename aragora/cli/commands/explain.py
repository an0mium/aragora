"""CLI command: aragora explain <debate_id> -- explain a debate decision.

Retrieves and displays decision explanations for a completed debate,
including evidence chains, vote pivots, and counterfactual analysis.

Tries API-first via AragoraClient, falls back to local ExplanationBuilder.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def add_explain_parser(subparsers: Any) -> None:
    """Register the 'explain' subcommand."""
    parser = subparsers.add_parser(
        "explain",
        help="Explain a debate decision (evidence chains, vote pivots, counterfactuals)",
        description="""
Show a structured explanation of how a debate reached its decision.

Displays:
  - Decision summary and confidence
  - Evidence chain (key arguments and their sources)
  - Vote pivots (which votes changed the outcome)
  - Counterfactuals (what would have changed the result)

Examples:
  aragora explain abc123
  aragora explain abc123 --format json
  aragora explain abc123 --verbose
  aragora explain abc123 --api-url http://localhost:8080
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "debate_id",
        help="The debate ID to explain",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show full detail (untruncated evidence, belief changes, confidence attribution)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="API server URL (default: ARAGORA_API_URL or http://localhost:8080)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for server authentication",
    )
    parser.set_defaults(func=cmd_explain)


def cmd_explain(args: argparse.Namespace) -> int:
    """Handle 'explain' command."""
    debate_id = args.debate_id
    output_format = getattr(args, "output_format", "text")
    verbose = getattr(args, "verbose", False)

    # Try API-first approach
    explanation = _try_api_explanation(debate_id, args)

    # Fallback to local builder
    if explanation is None:
        explanation = _try_local_explanation(debate_id)

    if explanation is None:
        print(
            f"Error: Could not retrieve explanation for debate {debate_id}.",
            file=sys.stderr,
        )
        print(
            "Ensure the API server is running or the debate exists locally.",
            file=sys.stderr,
        )
        return 1

    if output_format == "json":
        print(json.dumps(explanation, indent=2, default=str))
    else:
        _print_text_explanation(debate_id, explanation, verbose=verbose)

    return 0


def _try_api_explanation(debate_id: str, args: argparse.Namespace) -> dict[str, Any] | None:
    """Try to get explanation via the API client."""
    try:
        import os

        from aragora.client.client import AragoraClient

        api_url = getattr(args, "api_url", None) or os.environ.get(
            "ARAGORA_API_URL", "http://localhost:8080"
        )
        api_key = getattr(args, "api_key", None) or os.environ.get("ARAGORA_API_KEY")

        client = AragoraClient(base_url=api_url, api_key=api_key)
        result = client.explainability.get_explanation(debate_id)

        return {
            "debate_id": result.debate_id,
            "decision": result.decision,
            "confidence": result.confidence,
            "summary": result.summary,
            "factors": [
                {
                    "name": f.name,
                    "description": f.description,
                    "weight": f.weight,
                    "evidence": f.evidence,
                    "source_agents": f.source_agents,
                }
                for f in (result.factors or [])
            ],
            "evidence_chain": [
                {
                    "content": e.content,
                    "source": e.source,
                    "confidence": e.confidence,
                    "round_number": e.round_number,
                    "agent_id": e.agent_id,
                }
                for e in (result.evidence_chain or [])
            ],
            "vote_pivots": [
                {
                    "agent_id": v.agent_id,
                    "vote_value": v.vote_value,
                    "confidence": v.confidence,
                    "influence_score": v.influence_score,
                    "reasoning": v.reasoning,
                    "changed_outcome": v.changed_outcome,
                }
                for v in (result.vote_pivots or [])
            ],
            "counterfactuals": [
                {
                    "scenario": c.scenario,
                    "description": c.description,
                    "alternative_outcome": c.alternative_outcome,
                    "probability": c.probability,
                    "key_differences": c.key_differences,
                }
                for c in (result.counterfactuals or [])
            ],
        }
    except ImportError:
        logger.debug("AragoraClient not available")
        return None
    except (OSError, ConnectionError, RuntimeError, ValueError, KeyError) as e:
        logger.debug("API explanation failed: %s", e)
        return None


def _try_local_explanation(debate_id: str) -> dict[str, Any] | None:
    """Try to build explanation locally from stored debate data."""
    try:
        from aragora.explainability.builder import ExplanationBuilder

        builder = ExplanationBuilder()

        # Try to load debate from local storage
        debate_result = _load_local_debate(debate_id)
        if debate_result is None:
            return None

        import asyncio

        decision = asyncio.run(builder.build(debate_result))

        # Generate a summary from the builder
        summary = builder.generate_summary(decision)

        # Map Decision entity fields to the unified dict format.
        # The Decision dataclass uses:
        #   conclusion, confidence, evidence_chain (list[EvidenceLink]),
        #   vote_pivots (list[VotePivot]), confidence_attribution, counterfactuals
        return {
            "debate_id": decision.debate_id,
            "decision": decision.conclusion,
            "confidence": decision.confidence,
            "consensus_reached": decision.consensus_reached,
            "consensus_type": decision.consensus_type,
            "rounds_used": decision.rounds_used,
            "agents_participated": decision.agents_participated,
            "task": decision.task,
            "domain": decision.domain,
            "summary": summary,
            "factors": [
                {
                    "name": attr.factor,
                    "description": attr.explanation,
                    "weight": attr.contribution,
                    "evidence": [],
                    "source_agents": [],
                    "raw_value": attr.raw_value,
                }
                for attr in decision.confidence_attribution
            ],
            "evidence_chain": [
                {
                    "content": e.content,
                    "source": e.source,
                    "confidence": e.relevance_score,
                    "round_number": e.metadata.get("round", 0),
                    "agent_id": e.source,
                    "grounding_type": e.grounding_type,
                    "quality_scores": e.quality_scores,
                    "cited_by": e.cited_by,
                }
                for e in decision.evidence_chain
            ],
            "vote_pivots": [
                {
                    "agent_id": v.agent,
                    "vote_value": v.choice,
                    "confidence": v.confidence,
                    "influence_score": v.influence_score,
                    "reasoning": v.reasoning_summary,
                    "changed_outcome": v.flip_detected,
                    "weight": v.weight,
                    "elo_rating": v.elo_rating,
                    "calibration_adjustment": v.calibration_adjustment,
                }
                for v in decision.vote_pivots
            ],
            "counterfactuals": [
                {
                    "scenario": c.condition,
                    "description": c.outcome_change,
                    "alternative_outcome": c.outcome_change,
                    "probability": c.likelihood,
                    "sensitivity": c.sensitivity,
                    "key_differences": [f"Affected agents: {', '.join(c.affected_agents)}"]
                    if c.affected_agents
                    else [],
                }
                for c in decision.counterfactuals
            ],
            "belief_changes": [
                {
                    "agent": b.agent,
                    "round": b.round,
                    "topic": b.topic,
                    "prior_belief": b.prior_belief,
                    "posterior_belief": b.posterior_belief,
                    "prior_confidence": b.prior_confidence,
                    "posterior_confidence": b.posterior_confidence,
                    "confidence_delta": b.confidence_delta,
                    "trigger": b.trigger,
                    "trigger_source": b.trigger_source,
                }
                for b in decision.belief_changes
            ],
            "evidence_quality_score": decision.evidence_quality_score,
            "agent_agreement_score": decision.agent_agreement_score,
            "belief_stability_score": decision.belief_stability_score,
        }
    except ImportError:
        logger.debug("ExplanationBuilder not available")
        return None
    except (ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
        logger.debug("Local explanation failed: %s", e)
        return None


def _load_local_debate(debate_id: str) -> Any:
    """Try to load a debate result from local storage."""
    try:
        from aragora.memory.store import CritiqueStore

        store = CritiqueStore()
        results = store.get_debates_with_critiques_batch([debate_id])
        return results[0] if results else None
    except (ImportError, OSError, KeyError, ValueError):
        pass

    # Try Knowledge Mound
    try:
        from aragora.knowledge.mound.adapters.receipt_adapter import get_receipt_adapter

        adapter = get_receipt_adapter()
        data = adapter.get_ingestion_result(debate_id)
        if data:
            return data
    except (ImportError, OSError, KeyError, ValueError, AttributeError):
        pass

    return None


def _print_text_explanation(
    debate_id: str, explanation: dict[str, Any], *, verbose: bool = False
) -> None:
    """Print a human-readable explanation to stdout.

    Args:
        debate_id: The debate identifier.
        explanation: Explanation dict (unified format from API or local builder).
        verbose: When True, show full untruncated content and extra sections
                 (belief changes, confidence attribution, quality scores).
    """
    print(f"\nDecision Explanation: {debate_id}")
    print("=" * 60)

    # Summary
    decision = explanation.get("decision", "")
    confidence = explanation.get("confidence", 0.0)
    summary = explanation.get("summary", "")

    print(f"\nDecision:   {decision}")
    print(f"Confidence: {confidence:.1%}")

    # Verbose: show extra metadata
    if verbose:
        consensus = explanation.get("consensus_reached")
        if consensus is not None:
            print(f"Consensus:  {'Reached' if consensus else 'Not reached'}")
        consensus_type = explanation.get("consensus_type")
        if consensus_type:
            print(f"Type:       {consensus_type}")
        rounds_used = explanation.get("rounds_used")
        if rounds_used:
            print(f"Rounds:     {rounds_used}")
        agents = explanation.get("agents_participated", [])
        if agents:
            print(f"Agents:     {', '.join(agents)}")
        task = explanation.get("task")
        if task:
            print(f"Task:       {task}")
        domain = explanation.get("domain")
        if domain and domain != "general":
            print(f"Domain:     {domain}")

    if summary:
        print(f"\nSummary:\n  {summary}")

    # Factors (confidence attribution)
    factors = explanation.get("factors", [])
    if factors:
        print(f"\n--- Contributing Factors ({len(factors)}) ---")
        for i, factor in enumerate(factors, 1):
            name = factor.get("name", "Unknown")
            weight = factor.get("weight", 0.0)
            desc = factor.get("description", "")
            agents = ", ".join(factor.get("source_agents", []))
            print(f"\n  {i}. {name} (weight: {weight:.2f})")
            if desc:
                print(f"     {desc}")
            if agents:
                print(f"     Sources: {agents}")
            if verbose:
                raw_value = factor.get("raw_value")
                if raw_value is not None:
                    print(f"     Raw value: {raw_value}")

    # Evidence chain
    evidence = explanation.get("evidence_chain", [])
    if evidence:
        print(f"\n--- Evidence Chain ({len(evidence)}) ---")
        for e in evidence:
            agent = e.get("agent_id", "unknown")
            rnd = e.get("round_number", 0)
            conf = e.get("confidence", 0.0)
            content = e.get("content", "")
            if verbose:
                # Full content in verbose mode
                preview = content
            else:
                # Truncate long evidence
                preview = content[:120] + "..." if len(content) > 120 else content
            print(f"\n  [{agent} R{rnd}] (conf: {conf:.0%})")
            print(f"    {preview}")
            if verbose:
                grounding = e.get("grounding_type")
                if grounding:
                    print(f"    Type: {grounding}")
                quality = e.get("quality_scores", {})
                if quality:
                    scores_str = ", ".join(f"{k}: {v:.2f}" for k, v in quality.items())
                    print(f"    Quality: {scores_str}")
                cited = e.get("cited_by", [])
                if cited:
                    print(f"    Cited by: {', '.join(cited)}")

    # Vote pivots
    pivots = explanation.get("vote_pivots", [])
    if pivots:
        print(f"\n--- Vote Pivots ({len(pivots)}) ---")
        for v in pivots:
            agent = v.get("agent_id", "unknown")
            vote = v.get("vote_value", "")
            influence = v.get("influence_score", 0.0)
            reasoning = v.get("reasoning", "")
            changed = " [CHANGED OUTCOME]" if v.get("changed_outcome") else ""
            print(f"\n  {agent}: {vote} (influence: {influence:.2f}){changed}")
            if verbose:
                # Full reasoning in verbose mode
                if reasoning:
                    print(f"    Reasoning: {reasoning}")
                weight = v.get("weight")
                if weight is not None:
                    print(f"    Weight: {weight:.2f}")
                elo = v.get("elo_rating")
                if elo is not None:
                    print(f"    ELO: {elo:.0f}")
                cal = v.get("calibration_adjustment")
                if cal is not None:
                    print(f"    Calibration adj: {cal:.3f}")
            else:
                if reasoning:
                    print(f"    Reasoning: {reasoning[:100]}")

    # Counterfactuals
    counterfactuals = explanation.get("counterfactuals", [])
    if counterfactuals:
        print(f"\n--- Counterfactuals ({len(counterfactuals)}) ---")
        for c in counterfactuals:
            scenario = c.get("scenario", "")
            alt = c.get("alternative_outcome", "")
            prob = c.get("probability", 0.0)
            print(f"\n  If: {scenario}")
            print(f"  Then: {alt} (probability: {prob:.0%})")
            if verbose:
                sensitivity = c.get("sensitivity")
                if sensitivity is not None:
                    print(f"  Sensitivity: {sensitivity:.2f}")
            diffs = c.get("key_differences", [])
            limit = len(diffs) if verbose else 3
            for d in diffs[:limit]:
                print(f"    - {d}")

    # Belief changes (verbose only)
    belief_changes = explanation.get("belief_changes", [])
    if verbose and belief_changes:
        print(f"\n--- Belief Changes ({len(belief_changes)}) ---")
        for b in belief_changes:
            agent = b.get("agent", "unknown")
            rnd = b.get("round", 0)
            prior = b.get("prior_belief", "")
            posterior = b.get("posterior_belief", "")
            delta = b.get("confidence_delta", 0.0)
            trigger = b.get("trigger", "")
            print(f"\n  [{agent} R{rnd}] {prior} -> {posterior} (delta: {delta:+.2f})")
            if trigger:
                print(f"    Trigger: {trigger}")

    # Summary metrics (verbose only)
    if verbose:
        eq = explanation.get("evidence_quality_score")
        aa = explanation.get("agent_agreement_score")
        bs = explanation.get("belief_stability_score")
        if eq is not None or aa is not None or bs is not None:
            print("\n--- Summary Metrics ---")
            if eq is not None:
                print(f"  Evidence quality:  {eq:.2f}")
            if aa is not None:
                print(f"  Agent agreement:   {aa:.2f}")
            if bs is not None:
                print(f"  Belief stability:  {bs:.2f}")

    print("\n" + "=" * 60)
