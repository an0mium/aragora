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
        _print_text_explanation(debate_id, explanation)

    return 0


def _try_api_explanation(
    debate_id: str, args: argparse.Namespace
) -> dict[str, Any] | None:
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

        return {
            "debate_id": debate_id,
            "decision": getattr(decision, "outcome", ""),
            "confidence": getattr(decision, "confidence", 0.0),
            "summary": getattr(decision, "summary", ""),
            "factors": [
                {
                    "name": getattr(f, "name", ""),
                    "description": getattr(f, "description", ""),
                    "weight": getattr(f, "weight", 0.0),
                    "evidence": getattr(f, "evidence", []),
                    "source_agents": getattr(f, "source_agents", []),
                }
                for f in getattr(decision, "factors", [])
            ],
            "evidence_chain": [
                {
                    "content": getattr(e, "content", ""),
                    "source": getattr(e, "source", ""),
                    "confidence": getattr(e, "confidence", 0.0),
                    "round_number": getattr(e, "round_number", 0),
                    "agent_id": getattr(e, "agent_id", ""),
                }
                for e in getattr(decision, "evidence_chain", [])
            ],
            "vote_pivots": [
                {
                    "agent_id": getattr(v, "agent_id", ""),
                    "vote_value": getattr(v, "vote_value", ""),
                    "confidence": getattr(v, "confidence", 0.0),
                    "influence_score": getattr(v, "influence_score", 0.0),
                    "reasoning": getattr(v, "reasoning", ""),
                    "changed_outcome": getattr(v, "changed_outcome", False),
                }
                for v in getattr(decision, "vote_pivots", [])
            ],
            "counterfactuals": [
                {
                    "scenario": getattr(c, "scenario", ""),
                    "description": getattr(c, "description", ""),
                    "alternative_outcome": getattr(c, "alternative_outcome", ""),
                    "probability": getattr(c, "probability", 0.0),
                    "key_differences": getattr(c, "key_differences", []),
                }
                for c in getattr(decision, "counterfactuals", [])
            ],
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
        result = store.get_debate(debate_id)
        return result
    except (ImportError, OSError, KeyError, ValueError):
        pass

    # Try Knowledge Mound
    try:
        from aragora.knowledge.mound.adapters.receipt_adapter import get_receipt_adapter

        adapter = get_receipt_adapter()
        data = adapter.query(debate_id=debate_id)
        if data:
            return data[0] if isinstance(data, list) else data
    except (ImportError, OSError, KeyError, ValueError, AttributeError):
        pass

    return None


def _print_text_explanation(debate_id: str, explanation: dict[str, Any]) -> None:
    """Print a human-readable explanation to stdout."""
    print(f"\nDecision Explanation: {debate_id}")
    print("=" * 60)

    # Summary
    decision = explanation.get("decision", "")
    confidence = explanation.get("confidence", 0.0)
    summary = explanation.get("summary", "")

    print(f"\nDecision:   {decision}")
    print(f"Confidence: {confidence:.1%}")
    if summary:
        print(f"\nSummary:\n  {summary}")

    # Factors
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

    # Evidence chain
    evidence = explanation.get("evidence_chain", [])
    if evidence:
        print(f"\n--- Evidence Chain ({len(evidence)}) ---")
        for e in evidence:
            agent = e.get("agent_id", "unknown")
            rnd = e.get("round_number", 0)
            conf = e.get("confidence", 0.0)
            content = e.get("content", "")
            # Truncate long evidence
            preview = content[:120] + "..." if len(content) > 120 else content
            print(f"\n  [{agent} R{rnd}] (conf: {conf:.0%})")
            print(f"    {preview}")

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
            diffs = c.get("key_differences", [])
            for d in diffs[:3]:
                print(f"    - {d}")

    print("\n" + "=" * 60)
