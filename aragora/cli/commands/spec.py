"""
Prompt-to-spec CLI command: transform vague ideas into structured specifications.

Guides users through the prompt-to-spec pipeline:
1. Decomposes a vague prompt into structured intent
2. Generates clarifying questions (interactive or auto-answered)
3. Researches relevant context from Knowledge Mound
4. Builds a validated specification with acceptance criteria
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)


async def _run_spec_pipeline(
    prompt: str,
    *,
    depth: str = "quick",
    skip_research: bool = False,
    skip_interrogation: bool = False,
    profile: str = "founder",
    output_format: str = "text",
) -> dict[str, Any]:
    """Run the prompt-to-spec pipeline and return the result."""
    from aragora.prompt_engine.conductor import ConductorConfig, PromptConductor
    from aragora.prompt_engine.types import InterrogationDepth

    depth_map = {
        "quick": InterrogationDepth.QUICK,
        "thorough": InterrogationDepth.THOROUGH,
        "exhaustive": InterrogationDepth.EXHAUSTIVE,
    }

    config = ConductorConfig.from_profile(profile)
    config.interrogation_depth = depth_map.get(depth, InterrogationDepth.QUICK)
    config.skip_research = skip_research
    config.skip_interrogation = skip_interrogation

    # Use a fast agent for CLI to keep latency bounded
    try:
        from aragora.agents.api_agents.openai import OpenAIAPIAgent

        agent = OpenAIAPIAgent(name="spec-agent", model="gpt-4o-mini", role="proposer")
    except (ImportError, RuntimeError, ValueError):
        agent = None

    conductor = PromptConductor(config=config, agent=agent)
    result = await conductor.run(prompt)

    return {
        "specification": result.specification.to_dict()
        if hasattr(result.specification, "to_dict")
        else result.specification,
        "intent": result.intent.to_dict() if hasattr(result.intent, "to_dict") else result.intent,
        "research": result.research.to_dict()
        if result.research and hasattr(result.research, "to_dict")
        else result.research,
        "questions": [q.to_dict() if hasattr(q, "to_dict") else q for q in result.questions],
        "stages_completed": result.stages_completed,
        "auto_approved": result.auto_approved,
        "timing": result.timing.to_dict() if hasattr(result, "timing") else None,
    }


def _print_timing_summary(timing: dict[str, Any]) -> None:
    """Render a concise latency profile for human-readable output."""
    total_ms = float(timing.get("total_duration_ms", 0.0) or 0.0)
    target_ms = float(timing.get("target_duration_ms", 0.0) or 0.0)
    coverage_pct = float(timing.get("tracking_coverage_pct", 0.0) or 0.0)

    print("\n  Latency Profile:")
    print(f"  Total:        {total_ms:.1f}ms / target {target_ms:.1f}ms")
    print(f"  Coverage:     {coverage_pct:.1f}% instrumented")

    stage_breakdown = timing.get("stage_breakdown", [])
    if stage_breakdown:
        print("  Stage Split:")
        for item in stage_breakdown[:4]:
            stage = item.get("stage", "unknown")
            duration_ms = float(item.get("duration_ms", 0.0) or 0.0)
            share_pct = float(item.get("share_of_total_pct", 0.0) or 0.0)
            print(f"    - {stage}: {duration_ms:.1f}ms ({share_pct:.1f}%)")

    targets = timing.get("optimization_targets", [])
    if targets:
        print("  Optimization Targets:")
        for target in targets[:3]:
            operation = target.get("operation", "unknown")
            duration_ms = float(target.get("duration_ms", 0.0) or 0.0)
            share_pct = float(target.get("share_of_total_pct", 0.0) or 0.0)
            hint = target.get("optimization_hint", "")
            print(f"    - {operation}: {duration_ms:.1f}ms ({share_pct:.1f}%)")
            if hint:
                print(f"      {hint}")


def _print_spec_result(result: dict[str, Any], output_format: str = "text") -> None:
    """Display the spec result in the requested format."""
    if output_format == "json":
        print(json.dumps(result, indent=2, default=str))
        return

    spec = result.get("specification")
    intent = result.get("intent")
    research = result.get("research")
    timing = result.get("timing") or {}

    print("\n" + "=" * 60)
    print("  SPECIFICATION")
    print("=" * 60)

    if intent:
        intent_type = getattr(intent, "intent_type", None) or intent.get("intent_type", "unknown")
        scope = getattr(intent, "scope_estimate", None) or intent.get("scope_estimate", "unknown")
        print(f"\n  Intent Type:  {intent_type}")
        print(f"  Scope:        {scope}")

        domains = getattr(intent, "domains", None) or intent.get("domains", [])
        if domains:
            print(f"  Domains:      {', '.join(str(d) for d in domains)}")

        ambiguities = getattr(intent, "ambiguities", None) or intent.get("ambiguities", [])
        if ambiguities:
            print(f"  Ambiguities:  {len(ambiguities)} detected")

    if spec:
        problem = getattr(spec, "problem_statement", None) or spec.get("problem_statement", "")
        if problem:
            print(f"\n  Problem:\n  {problem[:500]}")

        solution = getattr(spec, "proposed_solution", None) or spec.get("proposed_solution", "")
        if solution:
            print(f"\n  Solution:\n  {solution[:800]}")

        criteria = getattr(spec, "success_criteria", None) or spec.get("success_criteria", [])
        if criteria:
            print(f"\n  Success Criteria ({len(criteria)}):")
            for i, c in enumerate(criteria[:5], 1):
                desc = getattr(c, "description", None) or (
                    c.get("description") if isinstance(c, dict) else str(c)
                )
                print(f"    {i}. {desc}")

        risks = getattr(spec, "risks", None) or spec.get("risks", [])
        if risks:
            print(f"\n  Risks ({len(risks)}):")
            for r in risks[:3]:
                desc = getattr(r, "description", None) or (
                    r.get("description") if isinstance(r, dict) else str(r)
                )
                print(f"    - {desc}")

        effort = getattr(spec, "estimated_effort", None) or spec.get("estimated_effort", "")
        if effort:
            print(f"\n  Estimated Effort: {effort}")

        confidence = getattr(spec, "confidence", None) or spec.get("confidence")
        if confidence is not None:
            print(f"  Confidence:       {confidence:.0%}")

    if research:
        evidence = getattr(research, "evidence_links", None) or research.get("evidence_links", [])
        if evidence:
            print(f"\n  Evidence: {len(evidence)} source(s) found")

    if timing:
        total_duration_ms = timing.get("total_duration_ms")
        if total_duration_ms is not None:
            print(f"\n  Latency Total: {total_duration_ms:.1f}ms")

        slowest_stage = timing.get("slowest_stage") or {}
        slowest_stage_name = slowest_stage.get("stage")
        if slowest_stage_name:
            print(
                "  Slowest Stage: "
                f"{slowest_stage_name} ({slowest_stage.get('duration_ms', 0.0):.1f}ms)"
            )

        top_operations = timing.get("top_operations", [])
        if top_operations:
            print("  Slowest Operations:")
            for item in top_operations[:3]:
                print(f"    - {item['operation']}: {item['duration_ms']:.1f}ms")

    print("\n" + "=" * 60)


def cmd_spec(args: argparse.Namespace) -> None:
    """Handle the 'spec' command."""
    prompt = getattr(args, "prompt", None)
    if not prompt:
        print("Usage: aragora spec 'your idea or task description'")
        sys.exit(1)

    depth = getattr(args, "depth", "quick")
    profile = getattr(args, "profile", "founder")
    skip_research = getattr(args, "skip_research", False)
    skip_interrogation = getattr(args, "skip_interrogation", False)
    output_format = getattr(args, "format", "text")
    dry_run = getattr(args, "dry_run", False)

    print("\n" + "=" * 60)
    print("  ARAGORA SPEC")
    print("  Prompt-to-specification pipeline")
    print("=" * 60)
    print(f"\n  Prompt:  {prompt}")
    print(f"  Depth:   {depth}")
    print(f"  Profile: {profile}")

    if dry_run:
        print("\n  [dry-run] Would run: decompose -> interrogate -> research -> specify")
        print("  Use without --dry-run to execute.")
        return

    print("\n[*] Running prompt-to-spec pipeline...\n")

    start_time = time.monotonic()
    try:
        result = asyncio.run(
            _run_spec_pipeline(
                prompt,
                depth=depth,
                skip_research=skip_research,
                skip_interrogation=skip_interrogation,
                profile=profile,
                output_format=output_format,
            )
        )
    except (RuntimeError, ValueError, TypeError, ImportError) as e:
        print(f"\n[!] Spec pipeline failed: {e}")
        sys.exit(1)

    elapsed = time.monotonic() - start_time
    print(f"  Elapsed: {elapsed:.1f}s")

    _print_spec_result(result, output_format)

    # Save spec artifact
    output_path = getattr(args, "output", None)
    if output_path:
        import json as json_mod
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json_mod.dump(result, f, indent=2, default=str)
        print(f"\nSpec saved to: {path}")

    print("\nNext steps:")
    print("  aragora decide 'task' --spec <file>  # Execute from spec")
    print("  aragora ask 'task'                    # Debate the approach")
