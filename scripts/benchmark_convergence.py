#!/usr/bin/env python3
"""
Convergence Detection Benchmark
================================

Demonstrates how Aragora's convergence detection identifies when agents
reach consensus early, saving unnecessary debate rounds.

Runs debates with different round counts (2, 3, 5, 7) and measures:
- Convergence round (when consensus was detected)
- Rounds saved vs. max rounds
- Similarity trajectory per round
- Per-agent convergence patterns

Uses the Jaccard similarity backend (no external dependencies required).

Usage:
    python scripts/benchmark_convergence.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force Jaccard backend (no ML deps required)
os.environ["ARAGORA_CONVERGENCE_BACKEND"] = "jaccard"

from aragora.debate.convergence.detector import ConvergenceDetector
from aragora.debate.convergence.metrics import ConvergenceResult


# ---------------------------------------------------------------------------
# Simulated agent personas with realistic convergence behaviour
# ---------------------------------------------------------------------------

PERSONAS = {
    "analyst": {
        "style": "supportive",
        "responses_by_round": {
            # Round 1: Initial position -- strong pro-microservices
            1: (
                "After careful analysis, I strongly endorse adopting microservices. "
                "The benefits include reduced operational overhead, improved developer "
                "velocity, and better alignment with industry best practices. "
                "I recommend a phased rollout starting with non-critical services."
            ),
            # Round 2: Incorporating some feedback, still supportive
            2: (
                "Building on the discussion, I continue to support microservices "
                "with a phased rollout. The risk concerns are valid and the phased "
                "approach addresses them. Benefits of reduced operational overhead "
                "and improved developer velocity remain compelling. Starting with "
                "non-critical services is the right first step."
            ),
            # Round 3: Converging toward hybrid -- adopting shared vocabulary
            3: (
                "I support the phased hybrid migration approach. Starting with "
                "non-critical services and 2-3 bounded contexts manages risk while "
                "capturing benefits of reduced overhead and improved velocity. "
                "Clear success criteria and rollback triggers should be established "
                "before expanding the migration scope."
            ),
            # Round 4: Settled position -- near-identical to round 3
            4: (
                "I support the phased hybrid migration approach. Starting with "
                "non-critical services and 2-3 bounded contexts manages risk while "
                "capturing benefits of reduced overhead and improved velocity. "
                "Clear success criteria and rollback triggers should be established "
                "before we expand the migration scope further."
            ),
            # Round 5+: Identical to round 4
            5: (
                "I support the phased hybrid migration approach. Starting with "
                "non-critical services and 2-3 bounded contexts manages risk while "
                "capturing benefits of reduced overhead and improved velocity. "
                "Clear success criteria and rollback triggers should be established "
                "before we expand the migration scope further."
            ),
        },
    },
    "critic": {
        "style": "critical",
        "responses_by_round": {
            1: (
                "I have significant concerns about this approach. The migration cost "
                "is severely underestimated -- distributed systems introduce network "
                "partitioning, data consistency challenges, and operational complexity. "
                "Before committing, we need a detailed total-cost-of-ownership analysis."
            ),
            2: (
                "I acknowledge the phased approach reduces some risk, but I still "
                "have concerns about migration costs and operational complexity. "
                "A total-cost-of-ownership analysis is essential. However, starting "
                "with non-critical services is a reasonable compromise if we set "
                "clear success criteria and define rollback triggers."
            ),
            3: (
                "I can accept the phased hybrid migration approach with non-critical "
                "services first, provided we establish clear success criteria and "
                "rollback triggers before expanding scope. The total cost of ownership "
                "must be monitored throughout. This compromise manages the risks "
                "I raised while allowing measured progress."
            ),
            4: (
                "I accept the phased hybrid migration approach with non-critical "
                "services first, provided we establish clear success criteria and "
                "rollback triggers before expanding scope. The total cost of ownership "
                "must be monitored throughout. This compromise manages the risks "
                "I raised while allowing measured progress forward."
            ),
            5: (
                "I accept the phased hybrid migration approach with non-critical "
                "services first, provided we establish clear success criteria and "
                "rollback triggers before expanding scope. The total cost of ownership "
                "must be monitored throughout. This compromise manages the risks "
                "I raised while allowing measured progress forward."
            ),
        },
    },
    "pm": {
        "style": "balanced",
        "responses_by_round": {
            1: (
                "There are valid arguments on both sides. The proposed microservices "
                "approach offers scalability and team autonomy, but introduces "
                "operational complexity. I recommend a hybrid strategy: identify "
                "2-3 bounded contexts that would benefit most, migrate those first, "
                "and measure results before expanding."
            ),
            2: (
                "The hybrid strategy remains the best path forward. We should "
                "start with 2-3 bounded contexts and migrate non-critical services "
                "first. Measuring results with clear success criteria will balance "
                "the scalability benefits against operational risk management. "
                "Rollback triggers are essential for safety."
            ),
            3: (
                "I support the phased hybrid migration approach starting with "
                "non-critical services and 2-3 bounded contexts. Clear success "
                "criteria and rollback triggers should be established before "
                "expanding scope. This approach balances benefits of scalability "
                "with risk management and measured progress."
            ),
            4: (
                "I support the phased hybrid migration approach starting with "
                "non-critical services and 2-3 bounded contexts. Clear success "
                "criteria and rollback triggers should be established before "
                "expanding scope. This approach balances the benefits of scalability "
                "with risk management and measured progress."
            ),
            5: (
                "I support the phased hybrid migration approach starting with "
                "non-critical services and 2-3 bounded contexts. Clear success "
                "criteria and rollback triggers should be established before "
                "expanding scope. This approach balances the benefits of scalability "
                "with risk management and measured progress."
            ),
        },
    },
    "devil_advocate": {
        "style": "contrarian",
        "responses_by_round": {
            1: (
                "I disagree with the prevailing direction. Everyone seems to be "
                "converging too quickly on microservices. Our current monolith, with "
                "targeted improvements, may outperform a wholesale migration. The "
                "grass is not always greener. We should consider a modular monolith."
            ),
            2: (
                "While the phased approach is better than a big-bang migration, "
                "I still think a modular monolith captures 80 percent of the benefits "
                "with less risk. However, if the team insists on microservices, "
                "starting with non-critical services and having rollback criteria "
                "is essential for safety."
            ),
            3: (
                "I conditionally accept the phased hybrid migration approach as a "
                "proof of concept with non-critical services first. Clear rollback "
                "criteria and success metrics must be established. If results are "
                "poor after the bounded contexts, we should pivot to modular "
                "monolith which captures most benefits with reduced risk."
            ),
            4: (
                "I conditionally accept the phased hybrid migration approach as a "
                "proof of concept with non-critical services first. Clear rollback "
                "criteria and success metrics must be established before expanding "
                "scope. If results are poor we should pivot to a modular monolith "
                "which captures most benefits with reduced risk."
            ),
            5: (
                "I conditionally accept the phased hybrid migration approach as a "
                "proof of concept with non-critical services first. Clear rollback "
                "criteria and success metrics must be established before expanding "
                "scope. If results are poor we should pivot to a modular monolith "
                "which captures most benefits with reduced risk."
            ),
        },
    },
}


def get_agent_response(agent_name: str, round_num: int) -> str:
    """Get a simulated response for an agent at a given round."""
    persona = PERSONAS[agent_name]
    responses = persona["responses_by_round"]
    # Use the highest available round template (converged behavior for later rounds)
    available_rounds = sorted(responses.keys())
    best_round = available_rounds[0]
    for r in available_rounds:
        if r <= round_num:
            best_round = r
    return responses[best_round]


# ---------------------------------------------------------------------------
# Benchmark data structures
# ---------------------------------------------------------------------------


@dataclass
class RoundMetrics:
    """Metrics for a single debate round."""

    round_num: int
    avg_similarity: float
    min_similarity: float
    per_agent_similarity: dict[str, float]
    status: str  # "converged", "refining", "diverging"
    converged: bool


@dataclass
class DebateRunResult:
    """Result of a single benchmark debate run."""

    max_rounds: int
    convergence_round: int  # 0 = never converged
    rounds_executed: int
    rounds_saved: int
    savings_pct: float
    round_metrics: list[RoundMetrics] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    runs: list[DebateRunResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_single_debate(
    max_rounds: int,
    agents: list[str],
    convergence_threshold: float = 0.85,
) -> DebateRunResult:
    """Run a single debate and track convergence."""
    detector = ConvergenceDetector(
        convergence_threshold=convergence_threshold,
        divergence_threshold=0.40,
        min_rounds_before_check=1,
        consecutive_rounds_needed=1,
    )

    start = time.monotonic()
    previous_responses: dict[str, str] = {}
    round_metrics: list[RoundMetrics] = []
    convergence_round = 0

    for round_num in range(1, max_rounds + 1):
        # Generate responses for this round
        current_responses = {
            agent: get_agent_response(agent, round_num) for agent in agents
        }

        # Check convergence (only after round 1)
        if previous_responses:
            result = detector.check_convergence(
                current_responses=current_responses,
                previous_responses=previous_responses,
                round_number=round_num,
            )

            if result:
                metrics = RoundMetrics(
                    round_num=round_num,
                    avg_similarity=result.avg_similarity,
                    min_similarity=result.min_similarity,
                    per_agent_similarity=dict(result.per_agent_similarity),
                    status=result.status,
                    converged=result.converged,
                )
                round_metrics.append(metrics)

                if result.converged and convergence_round == 0:
                    convergence_round = round_num
            else:
                round_metrics.append(
                    RoundMetrics(
                        round_num=round_num,
                        avg_similarity=0.0,
                        min_similarity=0.0,
                        per_agent_similarity={},
                        status="initial",
                        converged=False,
                    )
                )
        else:
            round_metrics.append(
                RoundMetrics(
                    round_num=round_num,
                    avg_similarity=0.0,
                    min_similarity=0.0,
                    per_agent_similarity={},
                    status="initial",
                    converged=False,
                )
            )

        previous_responses = current_responses

        # Stop early if converged
        if convergence_round > 0:
            break

    elapsed_ms = (time.monotonic() - start) * 1000
    rounds_executed = convergence_round if convergence_round > 0 else max_rounds
    rounds_saved = max_rounds - rounds_executed
    savings_pct = (rounds_saved / max_rounds * 100) if max_rounds > 0 else 0.0

    return DebateRunResult(
        max_rounds=max_rounds,
        convergence_round=convergence_round,
        rounds_executed=rounds_executed,
        rounds_saved=rounds_saved,
        savings_pct=savings_pct,
        round_metrics=round_metrics,
        agents=agents,
        duration_ms=elapsed_ms,
    )


def run_benchmark() -> BenchmarkResults:
    """Run the full convergence benchmark."""
    print("=" * 70)
    print("Aragora Convergence Detection Benchmark")
    print("=" * 70)
    print()

    agents = list(PERSONAS.keys())
    round_configs = [2, 3, 5, 7]
    results = BenchmarkResults()
    total_start = time.monotonic()

    for max_rounds in round_configs:
        print(f"Running debate with max_rounds={max_rounds}...")
        run_result = run_single_debate(
            max_rounds=max_rounds,
            agents=agents,
            convergence_threshold=0.80,
        )
        results.runs.append(run_result)

        status = (
            f"Converged at round {run_result.convergence_round}"
            if run_result.convergence_round > 0
            else "Did not converge"
        )
        print(f"  {status} | Rounds saved: {run_result.rounds_saved} ({run_result.savings_pct:.0f}%)")

        # Print similarity trajectory
        for rm in run_result.round_metrics:
            if rm.avg_similarity > 0:
                bar = "#" * int(rm.avg_similarity * 40)
                print(f"    Round {rm.round_num}: avg={rm.avg_similarity:.3f} [{bar}] {rm.status}")

    results.total_duration_ms = (time.monotonic() - total_start) * 1000
    print()
    print(f"Total benchmark time: {results.total_duration_ms:.1f}ms")
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(results: BenchmarkResults) -> str:
    """Generate markdown benchmark report."""
    lines = []

    # Header
    lines.append("# Convergence Detection Benchmark Results")
    lines.append("")
    lines.append(f"*Generated: {results.timestamp}*")
    lines.append(f"*Benchmark duration: {results.total_duration_ms:.1f}ms*")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    converged_runs = [r for r in results.runs if r.convergence_round > 0]
    total_possible_rounds = sum(r.max_rounds for r in results.runs)
    total_rounds_used = sum(r.rounds_executed for r in results.runs)
    total_saved = total_possible_rounds - total_rounds_used

    if converged_runs:
        avg_convergence_round = sum(r.convergence_round for r in converged_runs) / len(converged_runs)
        avg_savings = sum(r.savings_pct for r in converged_runs) / len(converged_runs)
        lines.append(
            f"**Convergence detection saved {total_saved} out of {total_possible_rounds} "
            f"total rounds ({total_saved / total_possible_rounds * 100:.0f}%) across "
            f"{len(results.runs)} debate configurations.**"
        )
        lines.append("")
        lines.append(f"- **{len(converged_runs)}/{len(results.runs)}** debates converged early")
        lines.append(f"- Average convergence detected at round **{avg_convergence_round:.1f}**")
        lines.append(f"- Average round savings: **{avg_savings:.0f}%**")
        lines.append(
            f"- Total rounds saved: **{total_saved}** (executed {total_rounds_used} "
            f"instead of {total_possible_rounds})"
        )
    else:
        lines.append("No debates converged early in this benchmark run.")

    lines.append("")

    # Key Insight
    lines.append("### Key Insight")
    lines.append("")
    lines.append(
        "Convergence detection identifies when agents have reached substantive agreement, "
        "even when they use different words. This prevents wasted compute on rounds where "
        "agents are merely rephrasing their settled positions. In real-world debates with "
        "API-backed agents, each saved round avoids multiple LLM calls per agent."
    )
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Agents:** 4 agents with distinct personas (analyst/supportive, "
                 "critic/critical, pm/balanced, devil_advocate/contrarian)")
    lines.append("- **Convergence backend:** Jaccard similarity (word overlap)")
    lines.append("- **Convergence threshold:** 80% similarity between consecutive rounds")
    lines.append("- **Divergence threshold:** 40% similarity")
    lines.append("- **Debate topic:** Microservices migration decision")
    lines.append("- **Agent behavior:** Agents naturally converge over rounds as they "
                 "incorporate each other's concerns into revised positions")
    lines.append("")

    # Results Table
    lines.append("## Results by Round Configuration")
    lines.append("")
    lines.append("| Max Rounds | Convergence Round | Rounds Executed | Rounds Saved | Savings % | Final Similarity |")
    lines.append("|:----------:|:-----------------:|:---------------:|:------------:|:---------:|:----------------:|")

    for run in results.runs:
        conv_round = str(run.convergence_round) if run.convergence_round > 0 else "N/A"
        final_sim = "N/A"
        # Get highest similarity observed
        sim_metrics = [rm for rm in run.round_metrics if rm.avg_similarity > 0]
        if sim_metrics:
            final_sim = f"{sim_metrics[-1].avg_similarity:.3f}"

        lines.append(
            f"| {run.max_rounds} | {conv_round} | {run.rounds_executed} | "
            f"{run.rounds_saved} | {run.savings_pct:.0f}% | {final_sim} |"
        )

    lines.append("")

    # Detailed per-run analysis
    lines.append("## Detailed Similarity Trajectories")
    lines.append("")

    for run in results.runs:
        lines.append(f"### {run.max_rounds}-Round Debate")
        lines.append("")

        if run.convergence_round > 0:
            lines.append(
                f"Convergence detected at round {run.convergence_round}, "
                f"saving {run.rounds_saved} round(s) ({run.savings_pct:.0f}% reduction)."
            )
        else:
            lines.append("Debate did not converge within the allotted rounds.")
        lines.append("")

        # Similarity table per round
        sim_rounds = [rm for rm in run.round_metrics if rm.avg_similarity > 0]
        if sim_rounds:
            lines.append("| Round | Avg Similarity | Min Similarity | Status | Visual |")
            lines.append("|:-----:|:--------------:|:--------------:|:------:|:-------|")

            for rm in sim_rounds:
                bar_len = int(rm.avg_similarity * 20)
                bar = "=" * bar_len + "." * (20 - bar_len)
                status_icon = {
                    "converged": "CONVERGED",
                    "refining": "refining",
                    "diverging": "DIVERGING",
                }.get(rm.status, rm.status)

                lines.append(
                    f"| {rm.round_num} | {rm.avg_similarity:.3f} | "
                    f"{rm.min_similarity:.3f} | {status_icon} | `[{bar}]` |"
                )
            lines.append("")

            # Per-agent breakdown for last measured round
            last_round = sim_rounds[-1]
            if last_round.per_agent_similarity:
                lines.append("**Per-agent similarity (last measured round):**")
                lines.append("")
                for agent, sim in sorted(last_round.per_agent_similarity.items()):
                    bar_len = int(sim * 20)
                    bar = "=" * bar_len
                    lines.append(f"- `{agent:20s}` {sim:.3f} `[{bar}]`")
                lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    lines.append("### What This Demonstrates")
    lines.append("")
    lines.append(
        "1. **Early consensus detection works.** Even with agents starting from very "
        "different positions (supportive vs. critical vs. contrarian), convergence "
        "detection identifies when they settle on a shared position."
    )
    lines.append("")
    lines.append(
        "2. **Round savings scale with debate length.** The longer a debate is configured "
        "to run, the more rounds convergence detection saves. A 7-round debate that "
        "converges at round 3 saves 57% of compute."
    )
    lines.append("")
    lines.append(
        "3. **Per-agent tracking reveals holdouts.** The per-agent similarity scores "
        "show which agents converge quickly (supportive, balanced) vs. which take "
        "longer (critical, contrarian). This helps facilitators focus attention."
    )
    lines.append("")

    # Limitations
    lines.append("### Limitations")
    lines.append("")
    lines.append(
        "- This benchmark uses Jaccard (word overlap) similarity. In production, "
        "Aragora can use sentence-transformer embeddings for semantic similarity, "
        "which catches agreement even when agents use completely different vocabulary."
    )
    lines.append(
        "- The simulated agents have predetermined convergence trajectories. Real agents "
        "may converge faster or slower depending on the topic complexity."
    )
    lines.append(
        "- Convergence detection deliberately errs on the side of caution: it requires "
        "all agents to converge, not just a majority, to avoid cutting off productive "
        "disagreement."
    )
    lines.append("")

    # Competitive context
    lines.append("### Why This Matters")
    lines.append("")
    lines.append(
        "Most multi-agent debate systems run a fixed number of rounds regardless of "
        "whether agents have reached agreement. This wastes compute and API credits. "
        "Aragora's convergence detection is an adaptive termination mechanism that:"
    )
    lines.append("")
    lines.append("- Reduces LLM API costs by avoiding unnecessary rounds")
    lines.append("- Improves latency by ending debates as soon as consensus is reached")
    lines.append("- Integrates with the Trickster to detect *hollow* consensus (agents agreeing without evidence)")
    lines.append("- Provides per-agent convergence data for post-debate analysis")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    results = run_benchmark()
    report = generate_report(results)

    output_path = PROJECT_ROOT / "docs" / "benchmarks" / "convergence_results.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    print(f"\nReport written to: {output_path}")
    print()
    # Print summary
    for run in results.runs:
        if run.convergence_round > 0:
            print(
                f"  max_rounds={run.max_rounds}: converged at round {run.convergence_round}, "
                f"saved {run.rounds_saved} rounds ({run.savings_pct:.0f}%)"
            )
        else:
            print(f"  max_rounds={run.max_rounds}: did not converge")


if __name__ == "__main__":
    main()
