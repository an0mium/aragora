#!/usr/bin/env python3
"""
Deterministic Gauntlet evaluation harness.

Runs GauntletRunner against curated fixtures using a stubbed agent runner.
Outputs latency, coverage, and quality metrics without requiring API keys.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aragora.gauntlet import GauntletRunner, GauntletConfig, AttackCategory


@dataclass
class Fixture:
    """Evaluation fixture definition."""

    fixture_id: str
    input_content: str
    context: str = ""
    attack_rounds: int = 1
    attack_categories: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=lambda: ["mock-agent-1", "mock-agent-2"])
    expected: dict[str, int] = field(default_factory=dict)
    attack_responses: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: Path) -> "Fixture":
        fixture_id = payload.get("id") or source.stem
        input_content = payload.get("input_content", "").strip()
        if not input_content:
            raise ValueError(f"Fixture {source} is missing input_content")
        return cls(
            fixture_id=fixture_id,
            input_content=input_content,
            context=payload.get("context", ""),
            attack_rounds=int(payload.get("attack_rounds", 1)),
            attack_categories=payload.get("attack_categories", []),
            agents=payload.get("agents", ["mock-agent-1", "mock-agent-2"]),
            expected=payload.get("expected", {}),
            attack_responses=payload.get("attack_responses", []),
        )


class StubAgent:
    """Minimal agent used for deterministic evaluation runs."""

    def __init__(self, name: str):
        self.name = name


class ResponseQueue:
    """Deterministic response sequence for stubbed agent runs."""

    def __init__(self, responses: list[str]):
        self.responses = responses or ["No critical vulnerabilities found."]
        self.index = 0
        self.estimated_tokens = 0

    def next_response(self) -> str:
        response = self.responses[self.index % len(self.responses)]
        self.index += 1
        self.estimated_tokens += max(1, len(response.split()))
        return response


def _convert_attack_categories(names: list[str]) -> list[AttackCategory]:
    categories = []
    for name in names:
        try:
            categories.append(AttackCategory(name))
        except ValueError:
            continue
    return categories or [AttackCategory.SECURITY, AttackCategory.LOGIC]


def _severity_counts(result) -> dict[str, int]:
    risk = result.risk_summary
    return {
        "critical": int(risk.critical),
        "high": int(risk.high),
        "medium": int(risk.medium),
        "low": int(risk.low),
        "info": int(risk.info),
        "total": int(risk.total),
    }


def _precision_recall(expected_total: int, found_total: int) -> tuple[float, float]:
    if expected_total == 0 and found_total == 0:
        return 1.0, 1.0
    if expected_total == 0 and found_total > 0:
        return 0.0, 0.0
    matched = min(expected_total, found_total)
    precision = matched / found_total if found_total > 0 else 0.0
    recall = matched / expected_total if expected_total > 0 else 0.0
    return precision, recall


async def evaluate_fixture(
    fixture: Fixture,
    cost_per_1k_tokens: float = 0.0,
) -> dict[str, Any]:
    response_queue = ResponseQueue(fixture.attack_responses)

    def agent_factory(agent_name: str) -> StubAgent:
        return StubAgent(agent_name)

    async def run_agent_fn(agent: StubAgent, prompt: str) -> str:
        return response_queue.next_response()

    config = GauntletConfig(
        attack_categories=_convert_attack_categories(fixture.attack_categories),
        attack_rounds=fixture.attack_rounds,
        probe_categories=[],
        run_scenario_matrix=False,
        agents=fixture.agents,
    )

    runner = GauntletRunner(
        config=config,
        agent_factory=agent_factory,
        run_agent_fn=run_agent_fn,
    )

    start = time.time()
    result = await runner.run(fixture.input_content, fixture.context)
    elapsed = time.time() - start

    found = _severity_counts(result)
    expected = {k: int(v) for k, v in fixture.expected.items()}
    expected_total = sum(expected.values())
    precision, recall = _precision_recall(expected_total, found["total"])
    quality = (precision + recall) / 2 if expected_total > 0 else precision

    cost = (response_queue.estimated_tokens / 1000.0) * cost_per_1k_tokens

    return {
        "fixture_id": fixture.fixture_id,
        "latency_seconds": round(elapsed, 4),
        "expected": expected,
        "found": found,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "quality_score": round(quality, 4),
        "coverage_score": round(result.attack_summary.coverage_score, 4),
        "robustness_score": round(result.attack_summary.robustness_score, 4),
        "estimated_tokens": response_queue.estimated_tokens,
        "estimated_cost": round(cost, 6),
        "verdict": result.verdict.value,
    }


def load_fixtures(fixtures_dir: Path) -> list[Fixture]:
    fixtures = []
    for path in sorted(fixtures_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        fixtures.append(Fixture.from_dict(payload, path))
    return fixtures


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    total_latency = sum(r["latency_seconds"] for r in results)
    avg_quality = sum(r["quality_score"] for r in results) / len(results)
    avg_coverage = sum(r["coverage_score"] for r in results) / len(results)
    avg_robustness = sum(r["robustness_score"] for r in results) / len(results)
    total_tokens = sum(r["estimated_tokens"] for r in results)
    total_cost = sum(r["estimated_cost"] for r in results)
    return {
        "fixtures_run": len(results),
        "avg_latency_seconds": round(total_latency / len(results), 4),
        "avg_quality_score": round(avg_quality, 4),
        "avg_coverage_score": round(avg_coverage, 4),
        "avg_robustness_score": round(avg_robustness, 4),
        "estimated_tokens": total_tokens,
        "estimated_cost": round(total_cost, 6),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Gauntlet deterministically")
    parser.add_argument(
        "--fixtures-dir",
        default="benchmarks/fixtures/gauntlet",
        help="Directory containing fixture JSON files",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path for evaluation results",
    )
    parser.add_argument(
        "--cost-per-1k",
        type=float,
        default=0.0,
        help="Estimated cost per 1k tokens (optional)",
    )
    args = parser.parse_args()

    fixtures_dir = Path(args.fixtures_dir)
    if not fixtures_dir.exists():
        raise FileNotFoundError(f"Fixtures directory not found: {fixtures_dir}")

    fixtures = load_fixtures(fixtures_dir)
    if not fixtures:
        raise RuntimeError(f"No fixture files found in {fixtures_dir}")

    results = []
    for fixture in fixtures:
        result = await evaluate_fixture(fixture, cost_per_1k_tokens=args.cost_per_1k)
        results.append(result)

    summary = summarize(results)

    print("\nGAUNTLET EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key:>22}: {value}")

    print("\nDETAILS")
    print("-" * 60)
    for result in results:
        print(f"{result['fixture_id']}: quality={result['quality_score']}, "
              f"latency={result['latency_seconds']}s, "
              f"coverage={result['coverage_score']}, "
              f"verdict={result['verdict']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "summary": summary,
            "results": results,
        }, indent=2))
        print(f"\nSaved results to {output_path}")

    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
