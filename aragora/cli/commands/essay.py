"""CLI command for essay refinement and scoring.

Usage::

    aragora essay refine --input ideas.md --output essay.md
    aragora essay refine --input ideas.md --dry-run
    aragora essay score --input draft.md
    aragora essay score --input draft.md --rubric custom.yaml
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from aragora.agents.base import create_agent
from aragora.essay.pipeline import EssayRefinementPipeline
from aragora.essay.rubric import evaluate_essay, load_rubric

logger = logging.getLogger(__name__)


def _read_input(path: str) -> str:
    """Read input file and return its contents."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as e:
        print(f"[!] Could not read input file '{path}': {e}")
        sys.exit(1)


def _print_score_breakdown(score: Any) -> None:
    """Print a human-readable score breakdown."""
    dimensions = [
        ("thesis_clarity", "Thesis Clarity"),
        ("argument_coherence", "Argument Coherence"),
        ("evidence_grounding", "Evidence Grounding"),
        ("rhetorical_force", "Rhetorical Force"),
        ("concision", "Concision"),
        ("factual_accuracy", "Factual Accuracy"),
        ("originality", "Originality"),
    ]

    print("\n  Score Breakdown:")
    for attr, label in dimensions:
        value = getattr(score, attr, None)
        if value is not None:
            bar = "#" * int(value * 20)
            print(f"    {label:<22} {value:.2f}  [{bar:<20}]")

    overall = getattr(score, "overall", None)
    if overall is not None:
        print(f"\n  Overall Score:         {overall:.2f}")

    severity_notes = getattr(score, "severity_notes", [])
    if severity_notes:
        print(f"\n  Issues ({len(severity_notes)}):")
        for note in severity_notes:
            print(f"    - {note}")

    suggestions = getattr(score, "suggestions", [])
    if suggestions:
        print(f"\n  Suggestions ({len(suggestions)}):")
        for s in suggestions:
            print(f"    - {s}")

    weakest = getattr(score, "weakest_paragraph", None)
    if weakest:
        print(f"\n  Weakest Paragraph:\n    {weakest[:200]}")


def _cmd_refine(args: Any) -> None:
    """Handle 'essay refine' subcommand."""
    input_path = getattr(args, "input", None)
    output_path = getattr(args, "output", None)
    rounds = getattr(args, "rounds", 3)
    models_str = getattr(args, "models", None)
    target_words = getattr(args, "target_words", 1200)
    voice_notes = getattr(args, "voice_notes", None) or ""
    rubric = getattr(args, "rubric", None)
    dry_run = getattr(args, "dry_run", False)
    resume = getattr(args, "resume", False)

    raw_ideas = _read_input(input_path)

    models = None
    if models_str:
        models = [m.strip() for m in models_str.split(",") if m.strip()]

    print("\n" + "=" * 60)
    print("  ARAGORA ESSAY REFINE")
    print("=" * 60)
    print(f"\n  Input:        {input_path}")
    print(f"  Rounds:       {rounds}")
    print(f"  Target words: {target_words}")
    if models:
        print(f"  Models:       {', '.join(models)}")
    if voice_notes:
        print(f"  Voice notes:  {voice_notes[:60]}")
    if dry_run:
        print("  Mode:         dry-run")
    if resume:
        print("  Resume:       enabled")

    pipeline_kwargs: dict[str, Any] = {
        "max_rounds": rounds,
        "target_words": target_words,
        "voice_notes": voice_notes,
    }
    if models:
        pipeline_kwargs["models"] = models
    if rubric:
        pipeline_kwargs["rubric_path"] = rubric

    pipeline = EssayRefinementPipeline(**pipeline_kwargs)

    print("\n[*] Running essay refinement pipeline...\n")

    try:
        result = asyncio.run(pipeline.run(raw_ideas, dry_run=dry_run))
    except (RuntimeError, ValueError, TypeError, ImportError) as e:
        print(f"\n[!] Essay refinement failed: {e}")
        sys.exit(1)

    if dry_run:
        thesis = result.get("thesis", "")
        outline = result.get("outline", "")
        print("  [dry-run] Extraction complete. No drafts produced.\n")
        if thesis:
            print(f"  Thesis:\n    {thesis}\n")
        if outline:
            print("  Outline:")
            for line in outline.splitlines()[:15]:
                print(f"    {line}")
        print("\n  Run without --dry-run to produce a full essay.")
        return

    final_essay = result.get("final_essay", "")
    final_score = result.get("final_score")
    rounds_used = result.get("rounds_used", rounds)

    print(f"\n  Rounds used:  {rounds_used}")

    if final_score is not None:
        _print_score_breakdown(final_score)

    print("\n" + "=" * 60)
    print("  FINAL ESSAY")
    print("=" * 60)
    print()
    print(final_essay)
    print("\n" + "=" * 60)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(final_essay, encoding="utf-8")
        print(f"\nEssay saved to: {out}")

    print("\nNext steps:")
    print("  aragora essay score --input <output>  # Re-score the saved draft")
    print("  aragora essay refine --input <output> --rounds 1  # Further polish")


def _cmd_score(args: Any) -> None:
    """Handle 'essay score' subcommand."""
    input_path = getattr(args, "input", None)
    rubric_path = getattr(args, "rubric", None)
    models_str = getattr(args, "models", None)

    draft = _read_input(input_path)

    model = "anthropic-api"
    if models_str:
        first = [m.strip() for m in models_str.split(",") if m.strip()]
        if first:
            model = first[0]

    print("\n" + "=" * 60)
    print("  ARAGORA ESSAY SCORE")
    print("=" * 60)
    print(f"\n  Input:   {input_path}")
    print(f"  Model:   {model}")
    if rubric_path:
        print(f"  Rubric:  {rubric_path}")

    print("\n[*] Evaluating essay...\n")

    try:
        rubric = load_rubric(rubric_path)
        judge = create_agent(model, name="scorer", role="critic")
        score = asyncio.run(evaluate_essay(draft, judge, rubric=rubric))
    except (RuntimeError, ValueError, TypeError, ImportError) as e:
        print(f"\n[!] Essay scoring failed: {e}")
        sys.exit(1)

    _print_score_breakdown(score)
    print("\n" + "=" * 60)


def essay_command(args: Any) -> None:
    """Dispatch 'essay' subcommands."""
    subcommand = getattr(args, "essay_subcommand", None)

    if subcommand == "refine":
        _cmd_refine(args)
    elif subcommand == "score":
        _cmd_score(args)
    else:
        print("Usage: aragora essay <subcommand> [options]")
        print("Subcommands: refine, score")
        print("Run 'aragora essay --help' for more information.")
        sys.exit(1)
