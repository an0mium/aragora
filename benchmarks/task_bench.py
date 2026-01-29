#!/usr/bin/env python3
"""
Task Bench: ROI/quality comparison harness.

Runs a small suite of curated tasks against:
1) baseline (single-agent)
2) multi-agent (debate/orchestrated)

Default mode uses demo agents for offline determinism.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from aragora.agents.base import create_agent
from aragora.cli.main import run_debate
from aragora.cli.review import build_review_prompt, extract_review_findings
from aragora.core import Environment
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.gauntlet import AttackCategory, GauntletConfig, GauntletRunner, ProbeCategory

TaskKind = Literal["debate", "review", "gauntlet"]

logger = logging.getLogger(__name__)


@dataclass
class TaskCase:
    id: str
    kind: TaskKind
    title: str
    input: str
    expected_keywords: list[str]
    tags: list[str]
    context: str = ""
    attack_categories: list[str] | None = None
    probe_categories: list[str] | None = None
    input_type: str | None = None


def _load_task_file(path: Path) -> TaskCase:
    data = json.loads(path.read_text())
    return TaskCase(
        id=data["id"],
        kind=data["kind"],
        title=data.get("title", data["id"]),
        input=data["input"],
        expected_keywords=data.get("expected_keywords", []),
        tags=data.get("tags", []),
        context=data.get("context", ""),
        attack_categories=data.get("attack_categories"),
        probe_categories=data.get("probe_categories"),
        input_type=data.get("input_type"),
    )


def load_tasks(task_dir: Path, only: set[str] | None = None) -> list[TaskCase]:
    tasks: list[TaskCase] = []
    for path in sorted(task_dir.glob("*.json")):
        case = _load_task_file(path)
        if only and case.id not in only:
            continue
        tasks.append(case)
    return tasks


def _protocol_overrides(profile: str) -> dict[str, Any]:
    # Keep in sync with DebateProtocol fields.
    overrides = {
        "enable_research": False,
        "enable_trending_injection": False,
        "enable_rhetorical_observer": False,
        "enable_trickster": False,
        "enable_evolution": False,
        "verify_claims_during_consensus": False,
        "enable_evidence_weighting": False,
        "enable_breakpoints": False,
        "role_rotation": False,
        "role_matching": False,
        "use_structured_phases": False,
        "convergence_detection": False,
        "vote_grouping": False,
        "early_stopping": False,
    }

    if profile == "full":
        overrides.update(
            {
                "enable_rhetorical_observer": True,
                "role_rotation": True,
                "role_matching": True,
                "use_structured_phases": True,
            }
        )

    return overrides


def _build_agent_factory() -> Any:
    def factory(agent_type: str) -> Any:
        return create_agent(
            model_type=agent_type,  # type: ignore[arg-type]
            name=f"{agent_type}_bench",
            role="critic",
        )

    return factory


def _normalize_text(parts: Iterable[str]) -> str:
    return "\n".join([p.strip() for p in parts if p and p.strip()])


def _keyword_hits(text: str, keywords: list[str]) -> dict[str, bool]:
    text_lower = text.lower()
    return {kw: kw.lower() in text_lower for kw in keywords}


def _score_keywords(text: str, keywords: list[str]) -> dict[str, Any]:
    hits = _keyword_hits(text, keywords)
    total = len(keywords)
    hit_count = sum(1 for v in hits.values() if v)
    hit_rate = hit_count / total if total else 0.0
    return {"hits": hits, "hit_count": hit_count, "hit_rate": hit_rate}


def _map_attack_categories(values: list[str] | None) -> list[AttackCategory]:
    if not values:
        return [AttackCategory.SECURITY, AttackCategory.LOGIC]
    mapping = {
        "security": AttackCategory.SECURITY,
        "logic": AttackCategory.LOGIC,
        "architecture": AttackCategory.ARCHITECTURE,
        "compliance": AttackCategory.COMPLIANCE,
        "gdpr": AttackCategory.GDPR,
        "hipaa": AttackCategory.HIPAA,
        "ai_act": AttackCategory.AI_ACT,
        "performance": AttackCategory.PERFORMANCE,
        "scalability": AttackCategory.SCALABILITY,
    }
    return [mapping[v] for v in values if v in mapping]


def _map_probe_categories(values: list[str] | None) -> list[ProbeCategory]:
    if not values:
        return [ProbeCategory.CONTRADICTION, ProbeCategory.HALLUCINATION]
    mapping = {
        "contradiction": ProbeCategory.CONTRADICTION,
        "hallucination": ProbeCategory.HALLUCINATION,
        "sycophancy": ProbeCategory.SYCOPHANCY,
        "calibration": ProbeCategory.CALIBRATION,
        "edge_case": ProbeCategory.EDGE_CASE,
        "instruction_injection": ProbeCategory.INSTRUCTION_INJECTION,
    }
    return [mapping[v] for v in values if v in mapping]


def _run_debate_task(
    task: TaskCase,
    agents_str: str,
    rounds: int,
    profile: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    result = asyncio.run(
        run_debate(
            task=task.input,
            agents_str=agents_str,
            rounds=rounds,
            consensus="majority",
            context=task.context,
            learn=False,
            enable_audience=False,
            protocol_overrides=_protocol_overrides(profile),
        )
    )
    duration = time.perf_counter() - start
    critique_text = _normalize_text(
        [issue for critique in result.critiques for issue in critique.issues]
    )
    combined_text = _normalize_text([result.final_answer, critique_text])
    keyword_score = _score_keywords(combined_text, task.expected_keywords)
    return {
        "duration_seconds": duration,
        "final_answer": result.final_answer,
        "consensus_reached": result.consensus_reached,
        "confidence": result.confidence,
        "critique_count": len(result.critiques),
        "dissent_count": len(result.dissenting_views),
        "keyword_score": keyword_score,
    }


async def _run_review_debate(
    diff: str,
    agents_str: str,
    rounds: int,
    profile: str,
) -> Any:
    agent_types = [spec.strip() for spec in agents_str.split(",") if spec.strip()]
    if not agent_types:
        agent_types = ["demo", "demo"]
    roles = ["security_reviewer", "performance_reviewer", "quality_reviewer"]
    agents = []
    for i, agent_type in enumerate(agent_types):
        role = roles[i % len(roles)]
        agents.append(
            create_agent(
                model_type=agent_type,  # type: ignore[arg-type]
                name=f"{agent_type}_{role}",
                role=role,
            )
        )
    task_prompt = build_review_prompt(diff)
    env = Environment(task=task_prompt, max_rounds=rounds)
    protocol = DebateProtocol(rounds=rounds, consensus="majority", **_protocol_overrides(profile))
    arena = Arena(env, agents, protocol)
    return await arena.run()


def _run_review_task(
    task: TaskCase,
    agents_str: str,
    rounds: int,
    profile: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    result = asyncio.run(_run_review_debate(task.input, agents_str, rounds, profile))
    duration = time.perf_counter() - start
    findings = extract_review_findings(result)
    findings.pop("all_critiques", None)
    combined_text = _normalize_text(
        [
            findings.get("final_summary", ""),
            *findings.get("unanimous_critiques", []),
            *findings.get("risk_areas", []),
        ]
    )
    keyword_score = _score_keywords(combined_text, task.expected_keywords)
    return {
        "duration_seconds": duration,
        "summary": findings.get("final_summary", ""),
        "agreement_score": findings.get("agreement_score", 0.0),
        "unanimous_count": len(findings.get("unanimous_critiques", [])),
        "risk_count": len(findings.get("risk_areas", [])),
        "keyword_score": keyword_score,
    }


def _run_gauntlet_task(
    task: TaskCase,
    agents: list[str],
    profile: str,
) -> dict[str, Any]:
    attack_categories = _map_attack_categories(task.attack_categories)
    probe_categories = _map_probe_categories(task.probe_categories)
    config = GauntletConfig(
        agents=agents,
        attack_categories=attack_categories,
        probe_categories=probe_categories,
        attack_rounds=1 if profile == "fast" else 2,
        attacks_per_category=1,
        probes_per_category=1 if profile == "fast" else 2,
        run_scenario_matrix=False if profile == "fast" else True,
    )
    if task.input_type:
        config.input_type = task.input_type

    runner = GauntletRunner(config=config, agent_factory=_build_agent_factory())
    start = time.perf_counter()
    result = asyncio.run(runner.run(task.input, context=task.context))
    duration = time.perf_counter() - start
    vuln_text = _normalize_text(
        [vuln.description for vuln in result.vulnerabilities]
        + [vuln.title for vuln in result.vulnerabilities]
    )
    combined_text = _normalize_text([result.verdict_reasoning, vuln_text])
    keyword_score = _score_keywords(combined_text, task.expected_keywords)
    return {
        "duration_seconds": duration,
        "verdict": result.verdict.value,
        "confidence": result.confidence,
        "vulnerability_count": len(result.vulnerabilities),
        "keyword_score": keyword_score,
    }


def run_task(
    task: TaskCase,
    agents_str: str,
    rounds: int,
    profile: str,
) -> dict[str, Any]:
    if task.kind == "debate":
        return _run_debate_task(task, agents_str, rounds, profile)
    if task.kind == "review":
        return _run_review_task(task, agents_str, rounds, profile)
    if task.kind == "gauntlet":
        agents = [spec.strip() for spec in agents_str.split(",") if spec.strip()]
        return _run_gauntlet_task(task, agents, profile)
    raise ValueError(f"Unsupported task kind: {task.kind}")


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label in ("baseline", "multi"):
        subset = [r for r in results if r["agent_set"] == label]
        if not subset:
            continue
        avg_hit_rate = sum(r["keyword_score"]["hit_rate"] for r in subset) / len(subset)
        avg_duration = sum(r["duration_seconds"] for r in subset) / len(subset)
        summary[label] = {
            "avg_hit_rate": round(avg_hit_rate, 3),
            "avg_duration_seconds": round(avg_duration, 3),
            "samples": len(subset),
        }
    if "baseline" in summary and "multi" in summary:
        summary["delta"] = {
            "hit_rate": round(
                summary["multi"]["avg_hit_rate"] - summary["baseline"]["avg_hit_rate"], 3
            ),
            "duration_seconds": round(
                summary["multi"]["avg_duration_seconds"]
                - summary["baseline"]["avg_duration_seconds"],
                3,
            ),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run task benchmark suite.")
    parser.add_argument(
        "--task-dir",
        default="benchmarks/tasks",
        help="Directory of task JSON files (default: benchmarks/tasks).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/benchmarks/task_bench",
        help="Directory to write results (default: output/benchmarks/task_bench).",
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "live"],
        default="demo",
        help="Agent mode (demo uses built-in demo agent).",
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "full"],
        default="fast",
        help="Execution profile (fast disables heavy features).",
    )
    parser.add_argument(
        "--baseline-agents",
        default="demo",
        help="Baseline agent set (comma-separated).",
    )
    parser.add_argument(
        "--agents",
        default="demo,demo,demo",
        help="Multi-agent set (comma-separated).",
    )
    parser.add_argument("--rounds", type=int, default=1, help="Debate rounds (default: 1).")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional task ids to run (space-separated).",
    )
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    only = set(args.only) if args.only else None
    tasks = load_tasks(task_dir, only)

    if not tasks:
        raise SystemExit("No tasks found.")

    if args.mode == "demo":
        baseline_agents = "demo"
        multi_agents = "demo,demo,demo"
    else:
        baseline_agents = args.baseline_agents
        multi_agents = args.agents

    results: list[dict[str, Any]] = []
    for task in tasks:
        for label, agents_str in (("baseline", baseline_agents), ("multi", multi_agents)):
            run = run_task(task, agents_str, args.rounds, args.profile)
            run.update(
                {
                    "task_id": task.id,
                    "task_kind": task.kind,
                    "task_title": task.title,
                    "agent_set": label,
                    "agents": agents_str,
                    "tags": task.tags,
                }
            )
            results.append(run)

    results_path = output_dir / "results.jsonl"
    with results_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = summarize(results)
    summary_payload = {
        "mode": args.mode,
        "profile": args.profile,
        "tasks": len(tasks),
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    logger.info("task-bench results written to %s", results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
