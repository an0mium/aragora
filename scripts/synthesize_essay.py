#!/usr/bin/env python3
"""
Essay Synthesis CLI - Synthesize essays from ChatGPT/Claude conversation exports.

This script provides a command-line interface to the essay synthesis workflow.

Usage:
    # Basic usage - load exports and generate essay
    python scripts/synthesize_essay.py \
        --exports /path/to/exports \
        --title "AI, Evolution, and the Myth of Final States" \
        --output output/essay.md

    # With seed essay to weave in
    python scripts/synthesize_essay.py \
        --exports /path/to/exports \
        --seed-essay drafts/seed_essay.md \
        --title "My Essay Title" \
        --output output/essay.md

    # Full options
    python scripts/synthesize_essay.py \
        --exports /path/to/exports \
        --seed-essay drafts/seed_essay.md \
        --title "My Essay Title" \
        --thesis "My central argument is..." \
        --output output/essay.md \
        --debate-rounds 3 \
        --enable-attribution \
        --target-words 50000 \
        --verbose

Steps to use:
    1. Export your ChatGPT data: Settings > Data Controls > Export Data
    2. Export your Claude data: Settings > Export Data
    3. Place all JSON exports in a directory
    4. Run this script pointing to that directory
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.pipelines.essay_workflow import (
    EssayWorkflow,
    WorkflowConfig,
    WorkflowResult,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthesize essays from ChatGPT/Claude conversation exports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    parser.add_argument(
        "--exports", "-e",
        type=Path,
        required=True,
        help="Path to exports directory or single export file",
    )
    parser.add_argument(
        "--seed-essay", "-s",
        type=Path,
        help="Path to seed essay (markdown file) to weave into synthesis",
    )

    # Output options
    parser.add_argument(
        "--title", "-t",
        type=str,
        required=True,
        help="Essay title",
    )
    parser.add_argument(
        "--thesis",
        type=str,
        help="Optional thesis statement",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output/essay.md"),
        help="Output file path (default: output/essay.md)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    # Processing options
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=3,
        help="Number of debate rounds for stress-testing claims (default: 3)",
    )
    parser.add_argument(
        "--no-debate",
        action="store_true",
        help="Disable multi-agent debate",
    )
    parser.add_argument(
        "--enable-attribution",
        action="store_true",
        help="Enable scholarly attribution search (requires API keys)",
    )
    parser.add_argument(
        "--claims-to-debate",
        type=int,
        default=10,
        help="Number of top claims to debate (default: 10)",
    )

    # Essay options
    parser.add_argument(
        "--target-words",
        type=int,
        default=50000,
        help="Target word count for essay (default: 50000)",
    )
    parser.add_argument(
        "--voice-style",
        choices=["analytical", "conversational", "academic"],
        default="analytical",
        help="Writing voice style (default: analytical)",
    )

    # Seed filtering options
    parser.add_argument(
        "--max-claims",
        type=int,
        default=300,
        help="Maximum claims to keep after seed filtering (default: 300)",
    )
    parser.add_argument(
        "--seed-min-score",
        type=float,
        default=0.05,
        help="Minimum seed relevance score (default: 0.05)",
    )

    # Additional options
    parser.add_argument(
        "--export-package",
        type=Path,
        help="Export synthesis package (JSON) for LLM processing",
    )
    parser.add_argument(
        "--export-all",
        type=Path,
        help="Export all outputs to directory (claims, clusters, outline, thread, etc.)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running",
    )

    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> bool:
    """Validate input arguments."""
    # Check exports path
    if not args.exports.exists():
        print(f"Error: Exports path not found: {args.exports}")
        return False

    # Check seed essay if provided
    if args.seed_essay and not args.seed_essay.exists():
        print(f"Error: Seed essay not found: {args.seed_essay}")
        return False

    return True


def export_all_outputs(workflow, result: WorkflowResult, output_dir: Path) -> None:
    """Export all synthesis outputs to a directory."""
    from dataclasses import asdict

    output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    # Export claims
    write_json(
        output_dir / "claims.json",
        {"claims": [c.to_dict() for c in result.claims]}
    )

    # Export clusters
    write_json(
        output_dir / "clusters.json",
        {"clusters": [c.to_dict() for c in result.clusters]}
    )

    # Export outline
    write_json(output_dir / "outline.json", result.outline.to_dict())

    # Export outline as markdown
    outline_md = render_outline_markdown(result.outline, result.claims)
    (output_dir / "outline.md").write_text(outline_md, encoding="utf-8")

    # Export thread skeleton
    thread = workflow.generate_thread_skeleton(result.outline)
    (output_dir / "thread_skeleton.md").write_text(thread, encoding="utf-8")

    # Export debate results
    write_json(
        output_dir / "debate_results.json",
        {"debates": [d.to_dict() for d in result.debate_results]}
    )

    # Export steelman claims
    write_json(
        output_dir / "steelman_claims.json",
        {"steelman_claims": result.steelman_claims}
    )

    # Export seed scores if available
    if result.seed_essay:
        _, scores = workflow.filter_claims_by_seed(max_claims=50)
        write_json(output_dir / "seed_scores.json", {"top_scores": scores})
        (output_dir / "seed.txt").write_text(result.seed_essay.content, encoding="utf-8")

    # Export statistics
    write_json(output_dir / "stats.json", result.statistics)

    # Export full synthesis package
    workflow.export_synthesis_package(output_dir / "essay_package.json", outline=result.outline)

    # Export essay
    (output_dir / "essay.md").write_text(result.final_essay, encoding="utf-8")


def render_outline_markdown(outline, claims: list) -> str:
    """Render outline as markdown."""
    claims_by_prefix = {c.claim[:50]: c.claim for c in claims}

    lines = [
        f"# {outline.title}",
        "",
        f"**Thesis:** {outline.thesis}",
        "",
        f"**Target word count:** {outline.target_word_count:,}",
        "",
        "## Sections",
        "",
    ]

    for section in outline.sections:
        lines.append(f"### {section.title}")
        if section.claims_referenced:
            lines.append("")
            lines.append("Key claims:")
            for prefix in section.claims_referenced[:6]:
                claim = claims_by_prefix.get(prefix, prefix)
                lines.append(f"- {claim}")
        if section.subsections:
            lines.append("")
            lines.append("Subsections:")
            for sub in section.subsections:
                lines.append(f"- {sub.title}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def print_summary(result: WorkflowResult) -> None:
    """Print workflow summary."""
    print("\n" + "=" * 60)
    print("ESSAY SYNTHESIS COMPLETE")
    print("=" * 60)
    print(f"\nTitle: {result.title}")
    print(f"Thesis: {result.thesis[:200]}...")
    print(f"\nStatistics:")
    for key, value in result.statistics.items():
        print(f"  - {key}: {value}")
    print(f"\nGenerated: {result.generated_at}")
    print("=" * 60)


async def run_workflow(args: argparse.Namespace) -> WorkflowResult | None:
    """Run the essay synthesis workflow."""
    # Create config
    config = WorkflowConfig(
        enable_debate=not args.no_debate,
        debate_rounds=args.debate_rounds,
        claims_to_debate=args.claims_to_debate,
        enable_attribution=args.enable_attribution,
        target_word_count=args.target_words,
        voice_style=args.voice_style,
        output_format=args.format,
        enable_seed_filtering=bool(args.seed_essay),
        seed_min_score=args.seed_min_score,
        max_claims=args.max_claims,
    )

    # Create workflow
    workflow = EssayWorkflow(config=config)

    # Load exports
    print(f"\nLoading exports from: {args.exports}")
    exports = await workflow.load_exports(args.exports)
    print(f"  Loaded {len(exports)} export files")

    total_convs = sum(e.conversation_count for e in exports)
    total_words = sum(e.total_words for e in exports)
    print(f"  Total conversations: {total_convs}")
    print(f"  Total words: {total_words:,}")

    if args.dry_run:
        print("\n[Dry run - stopping before processing]")
        return None

    # Set seed essay if provided
    if args.seed_essay:
        print(f"\nLoading seed essay: {args.seed_essay}")
        seed = await workflow.set_seed_essay(args.seed_essay)
        print(f"  Title: {seed.title}")
        print(f"  Themes: {', '.join(seed.themes[:5])}")
        print(f"  Keywords extracted: {len(seed.keywords)}")

    # Run workflow
    print("\nRunning essay synthesis workflow...")
    result = await workflow.run(
        title=args.title,
        thesis=args.thesis,
    )

    # Report seed filtering summary if seed was provided
    if args.seed_essay:
        filtered, scores = workflow.filter_claims_by_seed(
            min_score=args.seed_min_score,
            max_claims=args.max_claims,
        )
        print(f"  Seed filtering kept {len(filtered)} claims")

    # Export essay
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workflow.export_essay(args.output, result)
    print(f"\nExported essay to: {args.output}")

    # Export synthesis package if requested
    if args.export_package:
        workflow.export_synthesis_package(args.export_package, outline=result.outline)
        print(f"Exported synthesis package to: {args.export_package}")

    # Export all files if requested
    if args.export_all:
        export_all_outputs(workflow, result, args.export_all)
        print(f"Exported all outputs to: {args.export_all}")

    return result


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    print("\n" + "=" * 60)
    print("ESSAY SYNTHESIS FROM CONVERSATION EXPORTS")
    print("=" * 60)

    # Validate inputs
    if not validate_inputs(args):
        return 1

    # Run async workflow
    try:
        result = asyncio.run(run_workflow(args))

        if result:
            print_summary(result)

        return 0

    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return 130

    except Exception as e:
        logging.exception("Workflow failed")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
