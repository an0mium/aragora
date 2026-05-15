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
from pathlib import Path
from typing import Any

from aragora.agents.errors import AgentCircuitOpenError

logger = logging.getLogger(__name__)


async def _run_spec_via_orchestrator(
    prompt: str,
    *,
    depth: str = "quick",
    profile: str = "founder",
    skip_research: bool = False,
) -> dict[str, Any]:
    """Run the spec pipeline through UnifiedOrchestrator for full backbone tracking."""
    from aragora.pipeline.unified_orchestrator import OrchestratorConfig, UnifiedOrchestrator

    cfg = OrchestratorConfig(
        preset_name=profile,
        execution_mode="openclaw",
        skip_execution=True,
    )
    orchestrator = UnifiedOrchestrator()
    result = await orchestrator.run(prompt, config=cfg)

    return {
        "specification": None,
        "intent": None,
        "research": result.research_context,
        "questions": [],
        "stages_completed": result.stages_completed,
        "auto_approved": False,
        "timing": None,
        "run_id": result.run_id,
        "debate_result": result.debate_result,
        "spec_bundle": result.spec_bundle,
    }


async def _run_spec_pipeline(
    prompt: str,
    *,
    depth: str = "quick",
    skip_research: bool = False,
    skip_interrogation: bool = False,
    profile: str = "founder",
    output_format: str = "text",
    use_orchestrator: bool = False,
) -> dict[str, Any]:
    """Run the prompt-to-spec pipeline and return the result."""
    if use_orchestrator:
        return await _run_spec_via_orchestrator(
            prompt,
            depth=depth,
            profile=profile,
            skip_research=skip_research,
        )

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
    timing = getattr(result, "timing", None)

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
        "timing": timing.to_dict() if timing is not None and hasattr(timing, "to_dict") else timing,
    }


def _build_spec_fallback(prompt: str, *, reason: str) -> dict[str, Any]:
    """Create a truthful starter spec when the live spec agent is unavailable."""
    normalized_prompt = str(prompt or "").strip() or "Prompt-to-spec fallback"
    title = normalized_prompt.rstrip(" ?!.") or "Prompt-to-spec fallback"
    if len(title) > 72:
        title = title[:69].rstrip() + "..."

    return {
        "specification": {
            "title": title,
            "problem_statement": normalized_prompt,
            "proposed_solution": (
                "Start from a bounded manual spec so the next command can validate the "
                "decision, collect missing constraints, and continue without pretending "
                "the live prompt-to-spec pipeline succeeded."
            ),
            "success_criteria": [
                {
                    "description": "Capture the exact decision or change in one sentence.",
                },
                {
                    "description": "Name at least one measurable verification step or success signal.",
                },
                {
                    "description": "List the main constraint, risk, or rollback trigger before execution.",
                },
            ],
            "risks": [
                {
                    "description": (
                        "The dedicated spec agent is temporarily unavailable in this environment."
                    ),
                    "likelihood": "medium",
                    "impact": "medium",
                    "mitigation": (
                        "Retry `aragora spec` after the circuit recovers, or continue with this "
                        "manual starter spec and tighten it before execution."
                    ),
                }
            ],
            "estimated_effort": "small",
            "confidence": 0.2,
        },
        "intent": None,
        "research": None,
        "questions": [],
        "stages_completed": ["fallback_spec"],
        "auto_approved": False,
        "timing": None,
        "pipeline": "spec_fallback",
        "fallback_reason": reason,
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
    print(_render_spec_result(result, output_format))


def _render_spec_result(result: dict[str, Any], output_format: str = "text") -> str:
    """Render the spec result in the requested format."""
    if output_format == "json":
        return json.dumps(result, indent=2, default=str)

    spec = result.get("specification")
    intent = result.get("intent")
    research = result.get("research")
    timing = result.get("timing") or {}
    pipeline = str(result.get("pipeline", "") or "").strip()
    fallback_reason = str(result.get("fallback_reason", "") or "").strip()

    lines = ["", "=" * 60, "  SPECIFICATION", "=" * 60]

    if intent:
        intent_type = getattr(intent, "intent_type", None) or intent.get("intent_type", "unknown")
        scope = getattr(intent, "scope_estimate", None) or intent.get("scope_estimate", "unknown")
        lines.extend(
            [
                "",
                f"  Intent Type:  {intent_type}",
                f"  Scope:        {scope}",
            ]
        )

        domains = getattr(intent, "domains", None) or intent.get("domains", [])
        if domains:
            lines.append(f"  Domains:      {', '.join(str(d) for d in domains)}")

        ambiguities = getattr(intent, "ambiguities", None) or intent.get("ambiguities", [])
        if ambiguities:
            lines.append(f"  Ambiguities:  {len(ambiguities)} detected")

    if spec:
        problem = getattr(spec, "problem_statement", None) or spec.get("problem_statement", "")
        if problem:
            lines.extend(["", "  Problem:", f"  {problem[:500]}"])

        solution = getattr(spec, "proposed_solution", None) or spec.get("proposed_solution", "")
        if solution:
            lines.extend(["", "  Solution:", f"  {solution[:800]}"])

        criteria = getattr(spec, "success_criteria", None) or spec.get("success_criteria", [])
        if criteria:
            lines.extend(["", f"  Success Criteria ({len(criteria)}):"])
            for i, c in enumerate(criteria[:5], 1):
                desc = getattr(c, "description", None) or (
                    c.get("description") if isinstance(c, dict) else str(c)
                )
                lines.append(f"    {i}. {desc}")

        risks = getattr(spec, "risks", None) or spec.get("risks", [])
        if risks:
            lines.extend(["", f"  Risks ({len(risks)}):"])
            for r in risks[:3]:
                desc = getattr(r, "description", None) or (
                    r.get("description") if isinstance(r, dict) else str(r)
                )
                lines.append(f"    - {desc}")

        effort = getattr(spec, "estimated_effort", None) or spec.get("estimated_effort", "")
        if effort:
            lines.extend(["", f"  Estimated Effort: {effort}"])

        confidence = getattr(spec, "confidence", None) or spec.get("confidence")
        if confidence is not None:
            lines.append(f"  Confidence:       {confidence:.0%}")

    if research:
        evidence = getattr(research, "evidence_links", None) or research.get("evidence_links", [])
        if evidence:
            lines.extend(["", f"  Evidence: {len(evidence)} source(s) found"])

    if timing:
        total_duration_ms = timing.get("total_duration_ms")
        if total_duration_ms is not None:
            lines.extend(["", f"  Latency Total: {total_duration_ms:.1f}ms"])

        slowest_stage = timing.get("slowest_stage") or {}
        slowest_stage_name = slowest_stage.get("stage")
        if slowest_stage_name:
            lines.append(
                "  Slowest Stage: "
                f"{slowest_stage_name} ({slowest_stage.get('duration_ms', 0.0):.1f}ms)"
            )

        top_operations = timing.get("top_operations", [])
        if top_operations:
            lines.append("  Slowest Operations:")
            for item in top_operations[:3]:
                lines.append(f"    - {item['operation']}: {item['duration_ms']:.1f}ms")

    if pipeline:
        lines.extend(["", f"  Pipeline:   {pipeline}"])
    if fallback_reason:
        lines.append(f"  Note:       Fallback spec ({fallback_reason})")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def _save_spec_result(result: dict[str, Any], output_path: str, output_format: str) -> Path:
    """Persist the rendered spec result to the requested output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_spec_result(result, output_format) + "\n")
    return path


def _spec_result_task_text(prompt: str, result: dict[str, Any]) -> str:
    """Build deterministic decomposition input from a spec command result."""
    spec = result.get("specification") or {}
    sections: list[str] = []
    if isinstance(spec, dict):
        for key in ("title", "problem_statement", "proposed_solution", "raw"):
            value = str(spec.get(key) or "").strip()
            if value:
                sections.append(value)
        criteria = spec.get("success_criteria") or []
        if isinstance(criteria, list) and criteria:
            rendered: list[str] = []
            for item in criteria:
                if isinstance(item, dict):
                    text = str(item.get("description") or "").strip()
                else:
                    text = str(item or "").strip()
                if text:
                    rendered.append(f"- {text}")
            if rendered:
                sections.append("Success criteria:\n" + "\n".join(rendered))
    if sections:
        return "\n\n".join(sections)
    return prompt


def _write_mission_from_spec_result(
    *,
    prompt: str,
    result: dict[str, Any],
    output_path: str,
) -> Path:
    """Convert the spec output into a conductor mission YAML file."""
    from aragora.nomic.mission_bridge import decomposition_to_mission, write_mission_yaml
    from aragora.nomic.task_decomposer import TaskDecomposer

    task_text = _spec_result_task_text(prompt, result)
    decomp = TaskDecomposer().analyze(task_text)
    mission = decomposition_to_mission(
        decomp,
        objective=prompt,
        stop_condition=(
            "Stop when every lane reaches a draft PR, a precise blocker report, or a handoff."
        ),
    )
    return write_mission_yaml(mission, output_path)


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
    use_orchestrator = getattr(args, "orchestrator", False)
    to_mission = getattr(args, "to_mission", None)

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
                use_orchestrator=use_orchestrator,
            )
        )
    except AgentCircuitOpenError as e:
        agent_name = (getattr(e, "agent_name", None) or "spec-agent").strip() or "spec-agent"
        reason = f"{agent_name} circuit breaker open"
        logger.warning("spec_command_circuit_open", extra={"agent_name": agent_name})
        result = _build_spec_fallback(prompt, reason=reason)
    except (RuntimeError, ValueError, TypeError, ImportError) as e:
        print(f"\n[!] Spec pipeline failed: {e}")
        sys.exit(1)

    elapsed = time.monotonic() - start_time
    print(f"  Elapsed: {elapsed:.1f}s")

    _print_spec_result(result, output_format)

    # Save spec artifact
    output_path = getattr(args, "output", None)
    if output_path:
        path = _save_spec_result(result, output_path, output_format)
        print(f"\nSpec saved to: {path}")

    if to_mission:
        mission_path = _write_mission_from_spec_result(
            prompt=prompt,
            result=result,
            output_path=to_mission,
        )
        print(f"\nConductor mission saved to: {mission_path}")
        print(f"Run: python3 scripts/goal_conductor.py run-once --mission {mission_path} --json")

    print("\nNext steps:")
    print("  aragora decide 'task' --spec <file>  # Execute from spec")
    print("  aragora ask 'task'                    # Debate the approach")
