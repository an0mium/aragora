"""
CLI export command - export debate artifacts to various formats.

Extracted from main.py for modularity.
Supports HTML, JSON, and Markdown export formats.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def create_demo_artifact():
    """Create a demo artifact for testing exports."""
    from aragora.core import DebateResult, Message, Critique
    from aragora.export.artifact import ArtifactBuilder

    demo_result = DebateResult(
        task="Design a distributed rate limiter for a high-traffic API",
        final_answer="""## Recommended Architecture

1. **Token Bucket Algorithm** - Use a distributed token bucket with Redis as the backing store
2. **Sliding Window Counters** - Combine with sliding window for burst handling
3. **Consistent Hashing** - Distribute rate limit state across multiple nodes
4. **Circuit Breaker** - Implement fallback when rate limit service is unavailable

### Key Implementation Details:
- Use Redis MULTI/EXEC for atomic operations
- Implement local caching with 100ms TTL for hot keys
- Add monitoring for rate limit violations
- Include graceful degradation mode""",
        confidence=0.85,
        consensus_reached=True,
        rounds_used=2,
        duration_seconds=45.3,
        messages=[
            Message(role="proposer", agent="codex", content="Token bucket with Redis...", round=0),
            Message(role="proposer", agent="claude", content="Consider sliding window...", round=0),
            Message(
                role="critic", agent="claude", content="Redis single point of failure...", round=1
            ),
            Message(
                role="synthesizer",
                agent="codex",
                content="Combined approach with fallback...",
                round=2,
            ),
        ],
        critiques=[
            Critique(
                agent="claude",
                target_agent="codex",
                target_content="Redis proposal",
                issues=["Single point of failure", "Network latency concerns"],
                suggestions=["Add local caching", "Implement circuit breaker"],
                severity=0.4,
                reasoning="Good base but needs resilience",
            ),
        ],
    )

    artifact = (
        ArtifactBuilder()
        .from_result(demo_result)
        .with_verification("claim-1", "Token bucket is O(1)", "verified", "simulation")
        .build()
    )

    return artifact


def load_artifact_from_debate(debate_id: str, db_path: Optional[str] = None):
    """Load an artifact from a debate trace database."""
    from aragora.debate.traces import DebateReplayer
    from aragora.export.artifact import DebateArtifact, ConsensusProof

    replayer = DebateReplayer.from_database(f"trace-{debate_id}", db_path or "aragora_traces.db")
    trace = replayer.trace

    artifact = DebateArtifact(
        debate_id=trace.debate_id,
        task=trace.task,
        trace_data={"events": [e.to_dict() for e in trace.events]},
        agents=trace.agents,
        duration_seconds=trace.duration_ms / 1000 if trace.duration_ms else 0,
    )

    if trace.final_result:
        artifact.consensus_proof = ConsensusProof(
            reached=trace.final_result.get("consensus_reached", False),
            confidence=trace.final_result.get("confidence", 0),
            vote_breakdown={},
            final_answer=trace.final_result.get("final_answer", ""),
            rounds_used=trace.final_result.get("rounds_used", 0),
        )

    return artifact


def export_to_html(artifact, output_dir: Path) -> Path:
    """Export artifact to HTML format."""
    from aragora.export.static_html import StaticHTMLExporter

    exporter = StaticHTMLExporter(artifact)
    filepath = output_dir / f"debate_{artifact.artifact_id}.html"
    exporter.save(filepath)
    return filepath


def export_to_json(artifact, output_dir: Path) -> Path:
    """Export artifact to JSON format."""
    filepath = output_dir / f"debate_{artifact.artifact_id}.json"
    artifact.save(filepath)
    return filepath


def export_to_markdown(artifact, output_dir: Path) -> Path:
    """Export artifact to Markdown format."""
    from aragora.cli.publish import generate_markdown_report
    from aragora.core import DebateResult

    # Reconstruct minimal result for markdown generator
    result = DebateResult(
        id=artifact.artifact_id,
        task=artifact.task,
        final_answer=artifact.consensus_proof.final_answer if artifact.consensus_proof else "",
        confidence=artifact.consensus_proof.confidence if artifact.consensus_proof else 0,
        consensus_reached=artifact.consensus_proof.reached if artifact.consensus_proof else False,
        rounds_used=artifact.rounds,
        duration_seconds=artifact.duration_seconds,
        messages=[],
        critiques=[],
    )

    md_content = generate_markdown_report(result)
    filepath = output_dir / f"debate_{artifact.artifact_id}.md"
    filepath.write_text(md_content)
    return filepath


def main(args: argparse.Namespace) -> None:
    """Handle 'export' command - export debate artifacts."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get or create artifact
    if args.demo:
        artifact = create_demo_artifact()
    elif args.debate_id:
        try:
            artifact = load_artifact_from_debate(args.debate_id, getattr(args, "db", None))
        except Exception as e:
            print(f"Error loading debate: {e}")
            print("Use --demo for a sample export, or ensure the debate ID exists.")
            return
    else:
        print("Please provide a debate ID (--debate-id) or use --demo for a sample export.")
        return

    # Export to requested format
    format_type = args.format.lower()

    if format_type == "html":
        filepath = export_to_html(artifact, output_dir)
        print(f"HTML export saved: {filepath}")
    elif format_type == "json":
        filepath = export_to_json(artifact, output_dir)
        print(f"JSON export saved: {filepath}")
    elif format_type == "md":
        filepath = export_to_markdown(artifact, output_dir)
        print(f"Markdown export saved: {filepath}")
    else:
        print(f"Unknown format: {format_type}. Use html, json, or md.")
        return

    print(f"\nArtifact ID: {artifact.artifact_id}")
    print(f"Content Hash: {artifact.content_hash}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export debate artifacts")
    parser.add_argument("--output", "-o", default="./exports", help="Output directory")
    parser.add_argument("--format", "-f", default="html", choices=["html", "json", "md"])
    parser.add_argument("--debate-id", help="Debate ID to export")
    parser.add_argument("--demo", action="store_true", help="Create demo export")
    parser.add_argument("--db", help="Database path")
    main(parser.parse_args())
