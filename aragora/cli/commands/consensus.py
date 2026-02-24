"""CLI command: aragora consensus -- analyze consensus from debate output.

Commands for consensus detection and analysis:
- detect: Analyze proposals for consensus (from file, stdin, or inline)
- status: Get consensus status for an existing debate by ID

Tries API-first via AragoraClient, falls back to local ConsensusBuilder.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def add_consensus_parser(subparsers: Any) -> None:
    """Register the 'consensus' subcommand."""
    parser = subparsers.add_parser(
        "consensus",
        help="Analyze consensus from debate proposals or check debate consensus status",
        description="""
Detect and analyze consensus across agent proposals.

Subcommands:
  detect  Analyze proposals for consensus
  status  Get consensus status for an existing debate

Examples:
  aragora consensus detect --task "Choose a database" --file proposals.json
  aragora consensus detect --task "Choose a database" --proposals '["Use PostgreSQL", "Use PostgreSQL with Redis cache"]'
  echo '{"task":"Pick a framework","proposals":[{"agent":"a","content":"Use React"}]}' | aragora consensus detect --stdin
  aragora consensus status abc123
  aragora consensus status abc123 --format json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    consensus_sub = parser.add_subparsers(dest="consensus_command")

    # --- detect ---
    detect_parser = consensus_sub.add_parser(
        "detect",
        help="Analyze proposals for consensus",
    )
    detect_parser.add_argument(
        "--task",
        help="The debate task/question being analyzed",
    )
    detect_parser.add_argument(
        "--file",
        "-f",
        help="Path to JSON file with proposals (list of {agent, content} dicts)",
    )
    detect_parser.add_argument(
        "--proposals",
        help="Inline JSON array of proposal strings or {agent, content} objects",
    )
    detect_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON input from stdin ({task, proposals} object)",
    )
    detect_parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for consensus (default: 0.7)",
    )
    detect_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json",
    )
    detect_parser.add_argument(
        "--api-url",
        default=None,
        help="API server URL (default: ARAGORA_API_URL or http://localhost:8080)",
    )
    detect_parser.add_argument(
        "--api-key",
        default=None,
        help="API key for server authentication",
    )
    detect_parser.set_defaults(func=cmd_consensus_detect)

    # --- status ---
    status_parser = consensus_sub.add_parser(
        "status",
        help="Get consensus status for an existing debate",
    )
    status_parser.add_argument(
        "debate_id",
        help="The debate ID to check consensus for",
    )
    status_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json",
    )
    status_parser.add_argument(
        "--api-url",
        default=None,
        help="API server URL (default: ARAGORA_API_URL or http://localhost:8080)",
    )
    status_parser.add_argument(
        "--api-key",
        default=None,
        help="API key for server authentication",
    )
    status_parser.set_defaults(func=cmd_consensus_status)

    # Default: show help
    parser.set_defaults(func=cmd_consensus, _parser=parser)


def cmd_consensus(args: argparse.Namespace) -> None:
    """Handle 'consensus' command -- route to subcommands or show help."""
    subcommand = getattr(args, "consensus_command", None)

    if subcommand == "detect":
        cmd_consensus_detect(args)
    elif subcommand == "status":
        cmd_consensus_status(args)
    else:
        parser = getattr(args, "_parser", None)
        if parser:
            parser.print_help()
        else:
            print("Usage: aragora consensus {detect,status} ...")
            print("Run 'aragora consensus --help' for details.")


def cmd_consensus_detect(args: argparse.Namespace) -> int:
    """Handle 'consensus detect' command."""
    task = getattr(args, "task", None)
    proposals_json = getattr(args, "proposals", None)
    file_path = getattr(args, "file", None)
    use_stdin = getattr(args, "stdin", False)
    threshold = getattr(args, "threshold", 0.7)
    output_format = getattr(args, "output_format", "text")

    # Load proposals from the appropriate source
    proposals: list[dict[str, Any]] = []

    if use_stdin:
        try:
            raw = sys.stdin.read()
            data = json.loads(raw)
            if isinstance(data, dict):
                task = task or data.get("task", "")
                raw_proposals = data.get("proposals", [])
            elif isinstance(data, list):
                raw_proposals = data
            else:
                print("Error: stdin must contain a JSON object or array", file=sys.stderr)
                return 1
            proposals = _normalize_proposals(raw_proposals)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON on stdin: {e}", file=sys.stderr)
            return 1

    elif file_path:
        try:
            from pathlib import Path

            raw = Path(file_path).read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                task = task or data.get("task", "")
                raw_proposals = data.get("proposals", [])
            elif isinstance(data, list):
                raw_proposals = data
            else:
                print("Error: File must contain a JSON object or array", file=sys.stderr)
                return 1
            proposals = _normalize_proposals(raw_proposals)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error: Could not read file: {e}", file=sys.stderr)
            return 1

    elif proposals_json:
        try:
            raw_proposals = json.loads(proposals_json)
            proposals = _normalize_proposals(raw_proposals)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in --proposals: {e}", file=sys.stderr)
            return 1
    else:
        print(
            "Error: Provide proposals via --file, --proposals, or --stdin",
            file=sys.stderr,
        )
        return 1

    if not task:
        print("Error: --task is required (or include 'task' in JSON input)", file=sys.stderr)
        return 1

    if not proposals:
        print("Error: No valid proposals found", file=sys.stderr)
        return 1

    # Try API-first
    result = _try_api_detect(task, proposals, threshold, args)

    # Fallback to local detection
    if result is None:
        result = _try_local_detect(task, proposals, threshold)

    if result is None:
        print("Error: Consensus detection failed. Check that the module is available.", file=sys.stderr)
        return 1

    if output_format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_detect_result(result)

    return 0


def cmd_consensus_status(args: argparse.Namespace) -> int:
    """Handle 'consensus status' command."""
    debate_id = getattr(args, "debate_id", None)
    output_format = getattr(args, "output_format", "text")

    if not debate_id:
        print("Error: debate_id is required", file=sys.stderr)
        return 1

    # Try API-first
    result = _try_api_status(debate_id, args)

    if result is None:
        print(
            f"Error: Could not retrieve consensus status for debate {debate_id}.",
            file=sys.stderr,
        )
        print(
            "Ensure the API server is running or the debate exists.",
            file=sys.stderr,
        )
        return 1

    if output_format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_status_result(debate_id, result)

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_proposals(raw: list) -> list[dict[str, Any]]:
    """Normalize a list of proposals to [{agent, content}] format."""
    result: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            result.append({"agent": f"agent-{i + 1}", "content": item})
        elif isinstance(item, dict):
            result.append({
                "agent": item.get("agent", f"agent-{i + 1}"),
                "content": item.get("content", ""),
                "round": item.get("round", 0),
            })
    return [p for p in result if p.get("content")]


def _try_api_detect(
    task: str,
    proposals: list[dict[str, Any]],
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Try to detect consensus via the API."""
    try:
        import os

        from aragora.client.client import AragoraClient

        api_url = getattr(args, "api_url", None) or os.environ.get(
            "ARAGORA_API_URL", "http://localhost:8080"
        )
        api_key = getattr(args, "api_key", None) or os.environ.get("ARAGORA_API_KEY")

        client = AragoraClient(base_url=api_url, api_key=api_key)
        result = client.request(
            "POST",
            "/api/v1/consensus/detect",
            json={"task": task, "proposals": proposals, "threshold": threshold},
        )
        # Unwrap {data: ...} envelope
        if isinstance(result, dict) and "data" in result:
            return result["data"]
        return result
    except ImportError:
        logger.debug("AragoraClient not available")
        return None
    except (OSError, ConnectionError, RuntimeError, ValueError, KeyError) as e:
        logger.debug("API consensus detect failed: %s", e)
        return None


def _try_local_detect(
    task: str,
    proposals: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any] | None:
    """Detect consensus locally using ConsensusBuilder."""
    try:
        import hashlib

        from aragora.debate.consensus import ConsensusBuilder, VoteType

        debate_id = "detect-" + hashlib.sha256(task.encode()).hexdigest()[:12]
        builder = ConsensusBuilder(debate_id=debate_id, task=task)

        for proposal in proposals:
            agent = proposal.get("agent", "unknown")
            content = proposal.get("content", "")
            round_num = proposal.get("round", 0)
            if not content:
                continue
            claim = builder.add_claim(
                statement=content[:500],
                author=agent,
                confidence=0.6,
                round_num=round_num,
            )
            builder.add_evidence(
                claim_id=claim.claim_id,
                source=agent,
                content=content,
                evidence_type="argument",
                supports=True,
                strength=0.6,
            )

        agents = list({p.get("agent", "unknown") for p in proposals if p.get("content")})
        contents = [p.get("content", "") for p in proposals if p.get("content")]

        # Calculate agreement
        if len(contents) >= 2:
            agreement_scores = []
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    words_a = set(w.lower() for w in contents[i].split() if len(w) > 4)
                    words_b = set(w.lower() for w in contents[j].split() if len(w) > 4)
                    if words_a and words_b:
                        overlap = len(words_a & words_b)
                        union = len(words_a | words_b)
                        agreement_scores.append(overlap / union if union else 0.0)
            avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0
        else:
            avg_agreement = 1.0

        confidence = min(avg_agreement * 1.2, 1.0)
        consensus_reached = confidence >= threshold

        for agent in agents:
            vote_type = VoteType.AGREE if consensus_reached else VoteType.CONDITIONAL
            builder.record_vote(
                agent=agent,
                vote=vote_type,
                confidence=confidence,
                reasoning="Agreed with consensus" if consensus_reached else "Partial agreement",
            )

        final_claim = contents[0][:500] if contents else task
        proof = builder.build(
            final_claim=final_claim,
            confidence=confidence,
            consensus_reached=consensus_reached,
            reasoning_summary=(
                f"Analyzed {len(proposals)} proposals from {len(agents)} agents. "
                f"Average agreement: {avg_agreement:.0%}. "
                f"{'Consensus reached' if consensus_reached else 'Consensus not reached'} "
                f"(threshold: {threshold:.0%})."
            ),
            rounds=max((p.get("round", 0) for p in proposals), default=0),
        )

        return {
            "debate_id": debate_id,
            "consensus_reached": consensus_reached,
            "confidence": round(confidence, 4),
            "threshold": threshold,
            "agreement_ratio": round(proof.agreement_ratio, 4),
            "has_strong_consensus": proof.has_strong_consensus,
            "final_claim": proof.final_claim,
            "reasoning_summary": proof.reasoning_summary,
            "supporting_agents": proof.supporting_agents,
            "dissenting_agents": proof.dissenting_agents,
            "claims_count": len(proof.claims),
            "evidence_count": len(proof.evidence_chain),
            "checksum": proof.checksum,
        }

    except ImportError:
        logger.debug("ConsensusBuilder not available")
        return None
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug("Local consensus detection failed: %s", e)
        return None


def _try_api_status(
    debate_id: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Try to get consensus status via the API."""
    try:
        import os

        from aragora.client.client import AragoraClient

        api_url = getattr(args, "api_url", None) or os.environ.get(
            "ARAGORA_API_URL", "http://localhost:8080"
        )
        api_key = getattr(args, "api_key", None) or os.environ.get("ARAGORA_API_KEY")

        client = AragoraClient(base_url=api_url, api_key=api_key)
        result = client.request("GET", f"/api/v1/consensus/status/{debate_id}")
        if isinstance(result, dict) and "data" in result:
            return result["data"]
        return result
    except ImportError:
        logger.debug("AragoraClient not available")
        return None
    except (OSError, ConnectionError, RuntimeError, ValueError, KeyError) as e:
        logger.debug("API consensus status failed: %s", e)
        return None


def _print_detect_result(result: dict[str, Any]) -> None:
    """Print consensus detection result as human-readable text."""
    print("\nConsensus Detection Result")
    print("=" * 60)

    reached = result.get("consensus_reached", False)
    confidence = result.get("confidence", 0.0)
    threshold = result.get("threshold", 0.7)

    status_str = "REACHED" if reached else "NOT REACHED"
    print(f"\nStatus:     {status_str}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Threshold:  {threshold:.1%}")
    print(f"Agreement:  {result.get('agreement_ratio', 0.0):.1%}")

    if result.get("has_strong_consensus"):
        print("Strength:   STRONG (>80% agreement, >70% confidence)")

    print("\nFinal Claim:")
    claim = result.get("final_claim", "")
    if len(claim) > 200:
        print(f"  {claim[:200]}...")
    else:
        print(f"  {claim}")

    supporting = result.get("supporting_agents", [])
    dissenting = result.get("dissenting_agents", [])
    if supporting:
        print(f"\nSupporting: {', '.join(supporting)}")
    if dissenting:
        print(f"Dissenting: {', '.join(dissenting)}")

    print(f"\nClaims:     {result.get('claims_count', 0)}")
    print(f"Evidence:   {result.get('evidence_count', 0)}")

    summary = result.get("reasoning_summary", "")
    if summary:
        print(f"\nSummary:\n  {summary}")

    print(f"\nChecksum:   {result.get('checksum', 'N/A')}")
    print("=" * 60)


def _print_status_result(debate_id: str, result: dict[str, Any]) -> None:
    """Print consensus status as human-readable text."""
    print(f"\nConsensus Status: {debate_id}")
    print("=" * 60)

    reached = result.get("consensus_reached", False)
    confidence = result.get("confidence", 0.0)

    status_str = "REACHED" if reached else "NOT REACHED"
    print(f"\nStatus:     {status_str}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Agreement:  {result.get('agreement_ratio', 0.0):.1%}")

    if result.get("has_strong_consensus"):
        print("Strength:   STRONG")

    claim = result.get("final_claim", "")
    if claim:
        print("\nFinal Claim:")
        if len(claim) > 200:
            print(f"  {claim[:200]}...")
        else:
            print(f"  {claim}")

    supporting = result.get("supporting_agents", [])
    dissenting = result.get("dissenting_agents", [])
    if supporting:
        print(f"\nSupporting: {', '.join(supporting)}")
    if dissenting:
        print(f"Dissenting: {', '.join(dissenting)}")

    print(f"\nClaims:     {result.get('claims_count', 0)}")
    print(f"Dissents:   {result.get('dissents_count', 0)}")
    print(f"Tensions:   {result.get('unresolved_tensions_count', 0)}")

    # Partial consensus summary
    partial = result.get("partial_consensus", {})
    if partial:
        agreed = partial.get("agreed_count", 0)
        total = len(partial.get("items", []))
        if total > 0:
            print(f"\nPartial Consensus: {agreed}/{total} items agreed")

    print(f"\nChecksum:   {result.get('checksum', 'N/A')}")
    print("=" * 60)


__all__ = [
    "add_consensus_parser",
    "cmd_consensus",
    "cmd_consensus_detect",
    "cmd_consensus_status",
]
