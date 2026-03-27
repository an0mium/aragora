#!/usr/bin/env python3
"""Profile prompt-engine latency and highlight optimization targets."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aragora.prompt_engine import ConductorConfig, PromptConductor
from aragora.prompt_engine.timing import PipelineTiming, optimization_hint
from aragora.prompt_engine.types import InterrogationDepth

DEFAULT_MOCK_STAGE_LATENCIES_MS = {
    "decompose": 0.0,
    "interrogate": 0.0,
    "research": 0.0,
    "specify": 0.0,
}


class MockPromptEngineAgent:
    """Deterministic agent for repeatable local profiling."""

    def __init__(self, stage_latencies_ms: dict[str, float] | None = None) -> None:
        self._stage_latencies_ms = dict(DEFAULT_MOCK_STAGE_LATENCIES_MS)
        if stage_latencies_ms:
            self._stage_latencies_ms.update(stage_latencies_ms)

    async def generate(self, prompt: str, context: list[Any] | None = None) -> str:
        stage = self._detect_stage(prompt)
        delay_ms = self._stage_latencies_ms.get(stage, 0.0)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        return self._response_for(stage)

    @staticmethod
    def _detect_stage(prompt: str) -> str:
        if "structured intent" in prompt and "intent_type" in prompt:
            return "decompose"
        if "clarifying questions" in prompt and '"questions"' in prompt:
            return "interrogate"
        if "research report" in prompt and "current_state" in prompt:
            return "research"
        if "implementation specification" in prompt and "file_changes" in prompt:
            return "specify"
        return "unknown"

    @staticmethod
    def _response_for(stage: str) -> str:
        if stage == "decompose":
            return json.dumps(
                {
                    "intent_type": "improvement",
                    "summary": "Improve the prompt-to-spec pipeline latency profile.",
                    "domains": ["pipeline", "observability"],
                    "ambiguities": [
                        {
                            "description": "Whether to prioritize API or CLI latency first.",
                            "impact": "Changes where optimization work starts.",
                            "options": ["api", "cli"],
                            "recommended": None,
                        }
                    ],
                    "assumptions": [
                        {
                            "description": "The user needs actionable latency attribution.",
                            "confidence": 0.9,
                            "alternative": "Only a coarse total duration is needed.",
                        }
                    ],
                    "scope_estimate": "medium",
                }
            )
        if stage == "interrogate":
            return json.dumps(
                {
                    "questions": [
                        {
                            "question": "Which interface matters most for latency?",
                            "why_it_matters": "Determines where to instrument first.",
                            "options": [
                                {
                                    "label": "API",
                                    "description": "Optimize the server entrypoint first.",
                                    "tradeoffs": "May not help local CLI workflows immediately.",
                                },
                                {
                                    "label": "CLI",
                                    "description": "Optimize interactive command latency first.",
                                    "tradeoffs": "Server clients benefit later.",
                                },
                            ],
                            "default": "API",
                        }
                    ]
                }
            )
        if stage == "research":
            return json.dumps(
                {
                    "summary": "Prompt-engine profiling complete.",
                    "current_state": "The pipeline already records per-stage timings internally.",
                    "related_decisions": [],
                    "competitive_analysis": "Most latency sits in LLM calls plus optional context lookup.",
                    "recommendations": [
                        "Expose timing payloads to API and CLI consumers.",
                        "Prioritize the slowest LLM round trips first.",
                    ],
                    "evidence": [],
                }
            )
        if stage == "specify":
            return json.dumps(
                {
                    "title": "Profile prompt-engine latency",
                    "problem_statement": "Prompt-to-spec runs do not expose enough timing detail.",
                    "proposed_solution": "Add pipeline timing payloads and a profiling entrypoint.",
                    "alternatives_considered": ["Only log total duration"],
                    "file_changes": [
                        {
                            "path": "aragora/prompt_engine/timing.py",
                            "action": "modify",
                            "description": "Expose richer timing breakdowns.",
                            "estimated_lines": 80,
                        }
                    ],
                    "dependencies": [],
                    "risks": [
                        {
                            "description": "Payload growth for timing-heavy responses.",
                            "likelihood": "low",
                            "impact": "low",
                            "mitigation": "Keep summaries concise and JSON-native.",
                        }
                    ],
                    "success_criteria": [
                        {
                            "description": "Slow stages and operations are visible per run.",
                            "measurement": "timing.optimization_targets",
                            "target": "non-empty for realistic runs",
                        }
                    ],
                    "estimated_effort": "medium",
                    "confidence": 0.92,
                }
            )
        return "{}"


def _parse_stage_latencies(raw: str | None) -> dict[str, float]:
    """Parse stage latency overrides like 'decompose=10,specify=50'."""
    if not raw:
        return {}

    values: dict[str, float] = {}
    for item in raw.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid stage latency override: {chunk}")
        stage, value = chunk.split("=", 1)
        values[stage.strip()] = float(value.strip())
    return values


def _summarize(values: list[float]) -> dict[str, float]:
    """Return min/mean/median/max for a non-empty sample set."""
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 2),
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "max": round(max(values), 2),
    }


def build_profile_report(
    *,
    prompt: str,
    timings: list[PipelineTiming],
    profile: str,
    depth: str,
    skip_research: bool,
    skip_interrogation: bool,
    mode: str,
) -> dict[str, Any]:
    """Aggregate one or more prompt-engine timing samples."""
    stage_samples: dict[str, list[float]] = defaultdict(list)
    operation_samples: dict[str, list[float]] = defaultdict(list)
    operation_meta: dict[str, dict[str, str]] = {}

    for timing in timings:
        for stage, duration_ms in timing.stage_durations_ms.items():
            stage_samples[stage].append(duration_ms)
        for operation in timing.operation_timings:
            operation_samples[operation.operation].append(operation.duration_ms)
            operation_meta.setdefault(
                operation.operation,
                {
                    "stage": operation.stage,
                    "category": operation.category,
                },
            )

    top_operations = sorted(
        operation_samples.items(),
        key=lambda item: statistics.fmean(item[1]),
        reverse=True,
    )

    aggregate = {
        "total_duration_ms": _summarize([timing.total_duration_ms for timing in timings]),
        "stage_duration_ms": {
            stage: _summarize(values)
            for stage, values in sorted(
                stage_samples.items(),
                key=lambda item: statistics.fmean(item[1]),
                reverse=True,
            )
        },
        "top_operations": [
            {
                "operation": operation,
                "stage": operation_meta[operation]["stage"],
                "category": operation_meta[operation]["category"],
                "mean_duration_ms": round(statistics.fmean(values), 2),
                "median_duration_ms": round(statistics.median(values), 2),
                "optimization_hint": optimization_hint(operation_meta[operation]["category"]),
            }
            for operation, values in top_operations[:5]
        ],
    }

    return {
        "prompt": prompt,
        "profile": profile,
        "depth": depth,
        "mode": mode,
        "skip_research": skip_research,
        "skip_interrogation": skip_interrogation,
        "runs": len(timings),
        "samples": [timing.to_dict() for timing in timings],
        "aggregate": aggregate,
    }


def format_profile_report(report: dict[str, Any]) -> str:
    """Render a concise human-readable baseline report."""
    aggregate = report["aggregate"]
    lines = [
        "Prompt Engine Latency Profile",
        "=" * 30,
        f"Prompt: {report['prompt']}",
        f"Mode: {report['mode']}",
        f"Runs: {report['runs']}",
        "",
        "Total duration (ms):",
        (
            "  min={min:.2f} mean={mean:.2f} median={median:.2f} max={max:.2f}".format(
                **aggregate["total_duration_ms"]
            )
        ),
        "",
        "Stage baselines:",
    ]

    for stage, summary in aggregate["stage_duration_ms"].items():
        lines.append(
            "  - {stage}: min={min:.2f} mean={mean:.2f} median={median:.2f} max={max:.2f}".format(
                stage=stage,
                **summary,
            )
        )

    lines.append("")
    lines.append("Top operations:")
    for item in aggregate["top_operations"]:
        lines.append(
            "  - {operation}: mean={mean_duration_ms:.2f}ms median={median_duration_ms:.2f}ms"
            " [{category}]".format(**item)
        )
        lines.append(f"    {item['optimization_hint']}")

    return "\n".join(lines)


async def _run_profile(args: argparse.Namespace) -> dict[str, Any]:
    depth_map = {
        "quick": InterrogationDepth.QUICK,
        "thorough": InterrogationDepth.THOROUGH,
        "exhaustive": InterrogationDepth.EXHAUSTIVE,
    }

    config = ConductorConfig.from_profile(args.profile)
    config.interrogation_depth = depth_map[args.depth]
    config.skip_research = args.skip_research
    config.skip_interrogation = args.skip_interrogation

    mode = "live-agent"
    agent: Any | None = None
    if args.mock_agent:
        mode = "mock-agent"
        agent = MockPromptEngineAgent(_parse_stage_latencies(args.mock_stage_latencies))

    timings: list[PipelineTiming] = []
    for _ in range(args.runs):
        conductor = PromptConductor(config=config, agent=agent)
        result = await conductor.run(args.prompt)
        timings.append(result.timing)

    return build_profile_report(
        prompt=args.prompt,
        timings=timings,
        profile=args.profile,
        depth=args.depth,
        skip_research=args.skip_research,
        skip_interrogation=args.skip_interrogation,
        mode=mode,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Prompt to profile through the prompt-engine pipeline")
    parser.add_argument("--profile", default="founder", help="Prompt-engine profile to use")
    parser.add_argument(
        "--depth",
        choices=("quick", "thorough", "exhaustive"),
        default="quick",
        help="Interrogation depth",
    )
    parser.add_argument("--skip-research", action="store_true", help="Skip the research stage")
    parser.add_argument(
        "--skip-interrogation",
        action="store_true",
        help="Skip clarifying question generation",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of profiling runs")
    parser.add_argument(
        "--mock-agent",
        action="store_true",
        help="Use a deterministic local mock agent instead of live APIs",
    )
    parser.add_argument(
        "--mock-stage-latencies",
        help="Comma-separated per-stage mock latency overrides in ms",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--output", help="Optional file path for the JSON report")
    args = parser.parse_args(argv)

    report = asyncio.run(_run_profile(args))
    output = json.dumps(report, indent=2) if args.json else format_profile_report(report)
    print(output)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
