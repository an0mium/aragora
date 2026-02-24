"""
Document Analysis Example
=========================

Runs multi-agent analysis on a document, where agents debate key
findings, risks, and recommendations. Outputs a structured JSON report.

Requirements:
    - pip install aragora
    - ANTHROPIC_API_KEY or OPENAI_API_KEY set in environment

Usage:
    python examples/document_analysis/main.py path/to/document.txt
    python examples/document_analysis/main.py report.md --output analysis.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# --- API key check -------------------------------------------------------


def _check_api_keys() -> None:
    """Exit early with a helpful message if no API keys are configured."""
    keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
    if not any(os.environ.get(k) for k in keys):
        print(
            "ERROR: No API key found. Set at least one of:\n"
            "  ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY\n"
            "See the README for setup instructions.",
            file=sys.stderr,
        )
        sys.exit(1)


# --- Document loading -----------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml"}
MAX_DOCUMENT_CHARS = 15_000  # Keep within context window limits


def load_document(path: str) -> tuple[str, str]:
    """Load a document and return (content, filename).

    Validates file existence, extension, and size.
    """
    doc_path = Path(path)
    if not doc_path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    if doc_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(
            f"WARNING: Unsupported extension '{doc_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            file=sys.stderr,
        )

    content = doc_path.read_text(encoding="utf-8", errors="replace")
    if len(content) > MAX_DOCUMENT_CHARS:
        print(
            f"NOTE: Document truncated from {len(content)} to "
            f"{MAX_DOCUMENT_CHARS} chars for analysis.",
            file=sys.stderr,
        )
        content = content[:MAX_DOCUMENT_CHARS]

    return content, doc_path.name


# --- Debate setup ---------------------------------------------------------


def build_analysis_task(content: str, filename: str) -> str:
    """Build the debate task prompt for document analysis."""
    return (
        f"Analyze the following document ({filename}) and provide:\n"
        "1. KEY FINDINGS: The 3-5 most important facts or conclusions\n"
        "2. RISKS: Potential risks, gaps, or concerns identified\n"
        "3. RECOMMENDATIONS: Concrete next steps or improvements\n"
        "4. CONFIDENCE ASSESSMENT: How reliable are the document's claims?\n\n"
        "Be specific and cite relevant passages from the document.\n\n"
        f"--- DOCUMENT START ---\n{content}\n--- DOCUMENT END ---"
    )


async def run_analysis(content: str, filename: str) -> dict:
    """Run multi-agent document analysis and return structured report."""
    from aragora import Arena, Environment, DebateProtocol

    env = Environment(
        task=build_analysis_task(content, filename),
        context=(
            "You are analyzing a document for a business stakeholder. "
            "Focus on actionable insights and be precise about risks."
        ),
        roles=["proposer", "critic", "synthesizer"],
        max_rounds=3,
    )

    protocol = DebateProtocol(
        rounds=3,
        consensus="majority",
        use_structured_phases=False,
    )

    # Run the multi-agent debate
    arena = Arena(env, protocol=protocol)
    result = await arena.run()

    # Build the structured report
    report = {
        "document": filename,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "analysis": {
            "summary": result.final_answer,
            "consensus_reached": result.consensus_reached,
            "confidence": result.confidence,
            "consensus_strength": result.consensus_strength,
        },
        "debate_metadata": {
            "rounds_used": result.rounds_used,
            "participants": result.participants,
            "winner": result.winner,
            "convergence_status": result.convergence_status,
        },
    }

    # Include dissenting views if consensus was not reached
    if not result.consensus_reached and result.dissenting_views:
        report["dissenting_views"] = result.dissenting_views

    # Include individual agent proposals for transparency
    if result.proposals:
        report["agent_proposals"] = {
            agent: proposal[:500] for agent, proposal in result.proposals.items()
        }

    return report


# --- CLI entry point ------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent document analysis via Aragora")
    parser.add_argument("document", help="Path to the document to analyze")
    parser.add_argument(
        "--output",
        "-o",
        help="Write JSON report to this file (default: stdout)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: true)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    _check_api_keys()

    # Load the document
    content, filename = load_document(args.document)
    print(
        f"Analyzing '{filename}' ({len(content)} chars) with multi-agent debate...", file=sys.stderr
    )

    # Run the analysis
    report = await run_analysis(content, filename)

    # Output the report
    indent = 2 if args.pretty else None
    json_output = json.dumps(report, indent=indent, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(json_output, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    asyncio.run(main())
