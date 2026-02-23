#!/usr/bin/env python3
"""Trickster A/B Benchmark Pipeline.

Measures the impact of the Evidence-Powered Trickster (hollow consensus detection)
on debate quality by running controlled A/B experiments with StyledMockAgents.

Usage::

    # Full benchmark (all 15 questions)
    python scripts/benchmark_trickster.py

    # Quick mode (3 questions for fast iteration)
    python scripts/benchmark_trickster.py --quick

The script outputs a Markdown report to docs/benchmarks/trickster_ab_results.md.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the aragora-debate package is importable when running from repo root
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
_debate_src = _repo_root / "aragora-debate" / "src"
if _debate_src.exists() and str(_debate_src) not in sys.path:
    sys.path.insert(0, str(_debate_src))

from aragora_debate import (
    Arena,
    DebateConfig,
    DebateResult,
    EvidenceQualityAnalyzer,
    StyledMockAgent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

CATEGORY_CLEAR = "clear-answer"
CATEGORY_AMBIGUOUS = "ambiguous"
CATEGORY_DOMAIN = "domain-specific"


@dataclass
class TestCase:
    """A single benchmark question."""

    question: str
    category: str
    context: str = ""
    hollow: bool = False  # Use hollow-consensus agents for this case


ALL_TEST_CASES: list[TestCase] = [
    # Clear-answer questions (benchmarkable)
    TestCase(
        question="Is Python or JavaScript better for web scraping?",
        category=CATEGORY_CLEAR,
        context="We need to scrape structured data from 50 e-commerce sites daily.",
    ),
    TestCase(
        question="Should a startup use PostgreSQL or MongoDB for a CRM?",
        category=CATEGORY_CLEAR,
        context="B2B SaaS with relational customer data, 10k accounts, complex queries.",
    ),
    TestCase(
        question="Should we use TypeScript or JavaScript for a new React project?",
        category=CATEGORY_CLEAR,
        context="Team of 8 engineers, long-lived product, complex state management.",
    ),
    TestCase(
        question="Is Redis or Memcached better for session caching?",
        category=CATEGORY_CLEAR,
        context="Web app with 100k daily active users, need sub-10ms latency.",
    ),
    TestCase(
        question="Should we use pytest or unittest for our Python test suite?",
        category=CATEGORY_CLEAR,
        context="New project with 200+ modules, need fixtures, parameterization, and CI.",
    ),
    # Ambiguous questions (where hollow consensus is likely).
    # These use hollow agents to simulate the pattern where everyone agrees
    # but nobody provides real evidence — the exact scenario the Trickster
    # should detect and challenge.
    TestCase(
        question="Is AI good for society?",
        category=CATEGORY_AMBIGUOUS,
        context="Consider economic, social, ethical, and existential dimensions.",
        hollow=True,
    ),
    TestCase(
        question="Should we adopt microservices?",
        category=CATEGORY_AMBIGUOUS,
        context="Mid-size monolith, 15 engineers, growing but not yet at scale.",
        hollow=True,
    ),
    TestCase(
        question="Is remote work better than in-office for engineering teams?",
        category=CATEGORY_AMBIGUOUS,
        context="50-person engineering org, mix of senior and junior developers.",
        hollow=True,
    ),
    TestCase(
        question="Should startups prioritize growth or profitability?",
        category=CATEGORY_AMBIGUOUS,
        context="Series A SaaS company, $2M ARR, 18 months of runway.",
        hollow=True,
    ),
    TestCase(
        question="Is open source a better strategy than proprietary for developer tools?",
        category=CATEGORY_AMBIGUOUS,
        context="Developer tooling startup competing against established players.",
        hollow=True,
    ),
    # Domain-specific questions
    TestCase(
        question="Should we use Kubernetes or ECS for container orchestration?",
        category=CATEGORY_DOMAIN,
        context="AWS-first infrastructure, 40 microservices, small platform team.",
    ),
    TestCase(
        question="Is GraphQL better than REST for our mobile API?",
        category=CATEGORY_DOMAIN,
        context="Mobile-first product, bandwidth-sensitive, 30+ entity types.",
    ),
    TestCase(
        question="Should we use Kafka or RabbitMQ for our event bus?",
        category=CATEGORY_DOMAIN,
        context="E-commerce platform, 10k events/sec peak, need ordering guarantees.",
    ),
    TestCase(
        question="Is Terraform or Pulumi better for our IaC?",
        category=CATEGORY_DOMAIN,
        context="Python-heavy team, multi-cloud (AWS primary, some GCP), 200+ resources.",
    ),
    TestCase(
        question="Should we implement CQRS or keep a simple CRUD architecture?",
        category=CATEGORY_DOMAIN,
        context="Financial reporting app with complex read patterns and audit requirements.",
    ),
]

QUICK_TEST_CASES: list[TestCase] = [
    ALL_TEST_CASES[0],  # Clear: Python vs JS for scraping (diverse agents)
    ALL_TEST_CASES[5],  # Ambiguous: AI good for society (hollow agents)
    ALL_TEST_CASES[6],  # Ambiguous: microservices (hollow agents)
    ALL_TEST_CASES[10],  # Domain: K8s vs ECS (diverse agents)
]

# ---------------------------------------------------------------------------
# Benchmark data structures
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    """Metrics captured from a single debate run."""

    confidence: float = 0.0
    consensus_reached: bool = False
    rounds_used: int = 0
    trickster_interventions: int = 0
    convergence_detected: bool = False
    final_similarity: float = 0.0
    duration_seconds: float = 0.0
    proposal_length_avg: float = 0.0
    dissent_count: int = 0
    evidence_quality_avg: float = 0.0
    verdict: str = ""


@dataclass
class ABResult:
    """A/B comparison for a single question."""

    test_case: TestCase
    with_trickster: RunMetrics = field(default_factory=RunMetrics)
    without_trickster: RunMetrics = field(default_factory=RunMetrics)

    @property
    def confidence_delta(self) -> float:
        return self.with_trickster.confidence - self.without_trickster.confidence

    @property
    def consensus_changed(self) -> str:
        wt = self.with_trickster.consensus_reached
        wo = self.without_trickster.consensus_reached
        if wt == wo:
            return "unchanged"
        return "gained" if wt and not wo else "lost"


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_agents(seed: int, *, hollow: bool = False) -> list[StyledMockAgent]:
    """Create a reproducible set of 3 styled mock agents.

    Uses a fixed random seed so both A and B runs see the same agent
    configuration (though response selection within StyledMockAgent has
    its own randomness).

    When *hollow* is True the agents all use the ``"hollow"`` style,
    producing similar evidence-free responses that should trigger the
    Trickster's hollow consensus detection.
    """
    random.seed(seed)
    if hollow:
        return [
            StyledMockAgent("Advocate", style="hollow"),
            StyledMockAgent("Skeptic", style="hollow"),
            StyledMockAgent("Mediator", style="hollow"),
        ]
    return [
        StyledMockAgent("Advocate", style="supportive"),
        StyledMockAgent("Skeptic", style="critical"),
        StyledMockAgent("Mediator", style="balanced"),
    ]


# ---------------------------------------------------------------------------
# Single debate runner
# ---------------------------------------------------------------------------


async def run_debate(
    test_case: TestCase,
    enable_trickster: bool,
    seed: int,
    rounds: int = 2,
) -> tuple[DebateResult, RunMetrics]:
    """Run a single debate and extract metrics."""
    agents = create_agents(seed, hollow=test_case.hollow)

    config = DebateConfig(
        rounds=rounds,
        enable_trickster=enable_trickster,
        enable_convergence=True,
        convergence_threshold=0.85,
        trickster_sensitivity=0.7,
        early_stopping=False,  # always run all rounds for fair comparison
        min_rounds=rounds,
    )

    arena = Arena(
        question=test_case.question,
        agents=agents,
        config=config,
        context=test_case.context,
    )

    result = await arena.run()

    # Compute average proposal length
    proposal_lengths = [len(p) for p in result.proposals.values()]
    avg_proposal_length = sum(proposal_lengths) / len(proposal_lengths) if proposal_lengths else 0.0

    # Evidence quality scoring
    evidence_analyzer = EvidenceQualityAnalyzer()
    evidence_scores = evidence_analyzer.analyze_batch(result.proposals)
    evidence_qualities = [s.overall_quality for s in evidence_scores.values()]
    avg_evidence_quality = (
        sum(evidence_qualities) / len(evidence_qualities) if evidence_qualities else 0.0
    )

    metrics = RunMetrics(
        confidence=result.confidence,
        consensus_reached=result.consensus_reached,
        rounds_used=result.rounds_used,
        trickster_interventions=result.trickster_interventions,
        convergence_detected=result.convergence_detected,
        final_similarity=result.final_similarity,
        duration_seconds=result.duration_seconds,
        proposal_length_avg=avg_proposal_length,
        dissent_count=len(result.dissenting_views),
        evidence_quality_avg=avg_evidence_quality,
        verdict=result.verdict.value if result.verdict else "none",
    )

    return result, metrics


# ---------------------------------------------------------------------------
# A/B benchmark orchestrator
# ---------------------------------------------------------------------------


async def run_benchmark(
    test_cases: list[TestCase],
    rounds: int = 2,
    seed: int = 42,
) -> list[ABResult]:
    """Run the full A/B benchmark across all test cases."""
    results: list[ABResult] = []

    for i, tc in enumerate(test_cases):
        case_seed = seed + i  # deterministic but unique per case
        logger.info(
            "Benchmark %d/%d: %s [%s]",
            i + 1,
            len(test_cases),
            tc.question[:60],
            tc.category,
        )

        # --- Run WITH Trickster ---
        _, metrics_with = await run_debate(
            tc,
            enable_trickster=True,
            seed=case_seed,
            rounds=rounds,
        )

        # --- Run WITHOUT Trickster ---
        _, metrics_without = await run_debate(
            tc,
            enable_trickster=False,
            seed=case_seed,
            rounds=rounds,
        )

        ab = ABResult(
            test_case=tc,
            with_trickster=metrics_with,
            without_trickster=metrics_without,
        )
        results.append(ab)

        logger.info(
            "  WITH  trickster: confidence=%.2f consensus=%s interventions=%d",
            metrics_with.confidence,
            metrics_with.consensus_reached,
            metrics_with.trickster_interventions,
        )
        logger.info(
            "  W/OUT trickster: confidence=%.2f consensus=%s",
            metrics_without.confidence,
            metrics_without.consensus_reached,
        )

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _fmt(val: float, fmt: str = ".2f") -> str:
    """Format a float for the markdown table."""
    return f"{val:{fmt}}"


def _pct(val: float) -> str:
    """Format as percentage."""
    return f"{val:.0%}"


def _delta(val: float) -> str:
    """Format a delta with sign."""
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.3f}"


def generate_report(results: list[ABResult], duration: float) -> str:
    """Generate the Markdown benchmark report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append("# Trickster A/B Benchmark Results")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Total duration:** {duration:.1f}s")
    lines.append(f"**Test cases:** {len(results)}")
    lines.append("**Rounds per debate:** 2")
    lines.append("**Agents:** Advocate (supportive), Skeptic (critical), Mediator (balanced)")
    lines.append("")

    # -----------------------------------------------------------------------
    # Detailed results table
    # -----------------------------------------------------------------------
    lines.append("## Detailed Results")
    lines.append("")
    lines.append(
        "| # | Question | Category "
        "| Conf (with) | Conf (w/o) | Delta "
        "| Interventions | Consensus Change "
        "| Similarity (with) | Similarity (w/o) |"
    )
    lines.append(
        "|---|----------|----------"
        "|-------------|------------|-------"
        "|---------------|------------------"
        "|-------------------|------------------|"
    )

    for i, r in enumerate(results, 1):
        q = r.test_case.question
        if len(q) > 55:
            q = q[:52] + "..."
        lines.append(
            f"| {i} "
            f"| {q} "
            f"| {r.test_case.category} "
            f"| {_fmt(r.with_trickster.confidence)} "
            f"| {_fmt(r.without_trickster.confidence)} "
            f"| {_delta(r.confidence_delta)} "
            f"| {r.with_trickster.trickster_interventions} "
            f"| {r.consensus_changed} "
            f"| {_fmt(r.with_trickster.final_similarity)} "
            f"| {_fmt(r.without_trickster.final_similarity)} |"
        )

    lines.append("")

    # -----------------------------------------------------------------------
    # Evidence quality table
    # -----------------------------------------------------------------------
    lines.append("## Evidence Quality Scores")
    lines.append("")
    lines.append(
        "| # | Question "
        "| Evid. Quality (with) | Evid. Quality (w/o) "
        "| Avg Proposal Len (with) | Avg Proposal Len (w/o) "
        "| Dissents (with) | Dissents (w/o) |"
    )
    lines.append(
        "|---|----------"
        "|----------------------|---------------------"
        "|-------------------------|------------------------"
        "|-----------------|----------------|"
    )

    for i, r in enumerate(results, 1):
        q = r.test_case.question
        if len(q) > 55:
            q = q[:52] + "..."
        lines.append(
            f"| {i} "
            f"| {q} "
            f"| {_fmt(r.with_trickster.evidence_quality_avg)} "
            f"| {_fmt(r.without_trickster.evidence_quality_avg)} "
            f"| {_fmt(r.with_trickster.proposal_length_avg, '.0f')} "
            f"| {_fmt(r.without_trickster.proposal_length_avg, '.0f')} "
            f"| {r.with_trickster.dissent_count} "
            f"| {r.without_trickster.dissent_count} |"
        )

    lines.append("")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    lines.append("## Summary Statistics")
    lines.append("")

    # Confidence
    conf_with = [r.with_trickster.confidence for r in results]
    conf_without = [r.without_trickster.confidence for r in results]
    avg_conf_with = sum(conf_with) / len(conf_with) if conf_with else 0
    avg_conf_without = sum(conf_without) / len(conf_without) if conf_without else 0
    avg_conf_delta = avg_conf_with - avg_conf_without

    lines.append("| Metric | With Trickster | Without Trickster | Delta |")
    lines.append("|--------|----------------|-------------------|-------|")
    lines.append(
        f"| Avg confidence | {_fmt(avg_conf_with)} "
        f"| {_fmt(avg_conf_without)} "
        f"| {_delta(avg_conf_delta)} |"
    )

    # Consensus rate
    cons_with = sum(1 for r in results if r.with_trickster.consensus_reached)
    cons_without = sum(1 for r in results if r.without_trickster.consensus_reached)
    n = len(results)
    lines.append(
        f"| Consensus rate | {cons_with}/{n} ({_pct(cons_with / n)}) "
        f"| {cons_without}/{n} ({_pct(cons_without / n)}) "
        f"| {_delta((cons_with - cons_without) / n)} |"
    )

    # Intervention rate
    total_interventions = sum(r.with_trickster.trickster_interventions for r in results)
    cases_with_intervention = sum(
        1 for r in results if r.with_trickster.trickster_interventions > 0
    )
    lines.append(f"| Total interventions | {total_interventions} | 0 | +{total_interventions} |")
    lines.append(
        f"| Cases with intervention | {cases_with_intervention}/{n} ({_pct(cases_with_intervention / n)}) "
        f"| 0/0 "
        f"| -- |"
    )

    # Evidence quality
    eq_with = [r.with_trickster.evidence_quality_avg for r in results]
    eq_without = [r.without_trickster.evidence_quality_avg for r in results]
    avg_eq_with = sum(eq_with) / len(eq_with) if eq_with else 0
    avg_eq_without = sum(eq_without) / len(eq_without) if eq_without else 0
    lines.append(
        f"| Avg evidence quality | {_fmt(avg_eq_with)} "
        f"| {_fmt(avg_eq_without)} "
        f"| {_delta(avg_eq_with - avg_eq_without)} |"
    )

    # Similarity
    sim_with = [r.with_trickster.final_similarity for r in results]
    sim_without = [r.without_trickster.final_similarity for r in results]
    avg_sim_with = sum(sim_with) / len(sim_with) if sim_with else 0
    avg_sim_without = sum(sim_without) / len(sim_without) if sim_without else 0
    lines.append(
        f"| Avg final similarity | {_fmt(avg_sim_with)} "
        f"| {_fmt(avg_sim_without)} "
        f"| {_delta(avg_sim_with - avg_sim_without)} |"
    )

    # Average duration
    dur_with = [r.with_trickster.duration_seconds for r in results]
    dur_without = [r.without_trickster.duration_seconds for r in results]
    avg_dur_with = sum(dur_with) / len(dur_with) if dur_with else 0
    avg_dur_without = sum(dur_without) / len(dur_without) if dur_without else 0
    lines.append(
        f"| Avg debate duration | {_fmt(avg_dur_with, '.3f')}s "
        f"| {_fmt(avg_dur_without, '.3f')}s "
        f"| {_delta(avg_dur_with - avg_dur_without)}s |"
    )

    lines.append("")

    # -----------------------------------------------------------------------
    # Per-category breakdown
    # -----------------------------------------------------------------------
    lines.append("## Per-Category Breakdown")
    lines.append("")

    for cat in [CATEGORY_CLEAR, CATEGORY_AMBIGUOUS, CATEGORY_DOMAIN]:
        cat_results = [r for r in results if r.test_case.category == cat]
        if not cat_results:
            continue

        cat_n = len(cat_results)
        cat_conf_with = sum(r.with_trickster.confidence for r in cat_results) / cat_n
        cat_conf_without = sum(r.without_trickster.confidence for r in cat_results) / cat_n
        cat_interventions = sum(r.with_trickster.trickster_interventions for r in cat_results)
        cat_cons_with = sum(1 for r in cat_results if r.with_trickster.consensus_reached)
        cat_cons_without = sum(1 for r in cat_results if r.without_trickster.consensus_reached)

        lines.append(f"### {cat.replace('-', ' ').title()} ({cat_n} questions)")
        lines.append("")
        lines.append(f"- **Avg confidence with trickster:** {_fmt(cat_conf_with)}")
        lines.append(f"- **Avg confidence without trickster:** {_fmt(cat_conf_without)}")
        lines.append(f"- **Confidence delta:** {_delta(cat_conf_with - cat_conf_without)}")
        lines.append(f"- **Consensus rate (with):** {cat_cons_with}/{cat_n}")
        lines.append(f"- **Consensus rate (without):** {cat_cons_without}/{cat_n}")
        lines.append(f"- **Total trickster interventions:** {cat_interventions}")
        lines.append("")

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The Trickster system is designed to detect and challenge *hollow consensus* -- "
        "situations where agents converge on an answer without substantive evidence. "
        "Key observations from this benchmark:"
    )
    lines.append("")

    if total_interventions > 0:
        lines.append(
            f"- The Trickster intervened in **{cases_with_intervention}/{n}** debates "
            f"({_pct(cases_with_intervention / n)}), injecting a total of "
            f"**{total_interventions}** challenges."
        )
    else:
        lines.append(
            "- The Trickster did not intervene in any debates. This may indicate "
            "that the mock agents' canned responses contain enough specificity "
            "to avoid triggering hollow consensus detection, or that the "
            "sensitivity threshold needs adjustment."
        )

    if abs(avg_conf_delta) > 0.01:
        direction = "higher" if avg_conf_delta > 0 else "lower"
        lines.append(
            f"- Debates with the Trickster had **{direction}** average confidence "
            f"({_delta(avg_conf_delta)}), suggesting the challenges "
            f"{'strengthened' if avg_conf_delta > 0 else 'appropriately tempered'} "
            f"consensus quality."
        )
    else:
        lines.append(
            "- Average confidence was nearly identical between conditions, "
            "indicating the Trickster did not significantly alter debate outcomes "
            "for these mock agent configurations."
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*This report was generated by `scripts/benchmark_trickster.py` "
        "using `StyledMockAgent` configurations (no live LLM calls).*"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trickster A/B Benchmark Pipeline",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 3 questions for fast iteration.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of debate rounds per run (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path for the report (default: docs/benchmarks/trickster_ab_results.md).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    test_cases = QUICK_TEST_CASES if args.quick else ALL_TEST_CASES
    mode = "QUICK" if args.quick else "FULL"
    print(f"\n=== Trickster A/B Benchmark ({mode}: {len(test_cases)} questions) ===\n")

    t0 = time.monotonic()
    results = await run_benchmark(
        test_cases=test_cases,
        rounds=args.rounds,
        seed=args.seed,
    )
    total_duration = time.monotonic() - t0

    report = generate_report(results, total_duration)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _repo_root / "docs" / "benchmarks" / "trickster_ab_results.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("\n=== Benchmark Complete ===")
    print(f"Duration: {total_duration:.1f}s")
    print(f"Report saved to: {output_path}")
    print()

    # Print summary to stdout as well
    n = len(results)
    total_interventions = sum(r.with_trickster.trickster_interventions for r in results)
    avg_conf_with = sum(r.with_trickster.confidence for r in results) / n
    avg_conf_without = sum(r.without_trickster.confidence for r in results) / n
    cons_with = sum(1 for r in results if r.with_trickster.consensus_reached)
    cons_without = sum(1 for r in results if r.without_trickster.consensus_reached)

    print(
        f"  Avg confidence:   WITH={avg_conf_with:.3f}  WITHOUT={avg_conf_without:.3f}  delta={avg_conf_with - avg_conf_without:+.3f}"
    )
    print(f"  Consensus rate:   WITH={cons_with}/{n}  WITHOUT={cons_without}/{n}")
    print(f"  Trickster interventions: {total_interventions}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
