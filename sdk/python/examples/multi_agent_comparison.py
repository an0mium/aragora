"""
Multi-Agent Comparison Example

Demonstrates running the same debate with different agent combinations
and comparing their performance, consensus patterns, and ELO changes.

Usage:
    python examples/multi_agent_comparison.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from aragora_sdk import AragoraAsyncClient

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DebateResult:
    """Result of a single debate run."""

    debate_id: str
    agents: list[str]
    final_answer: str
    confidence: float
    rounds_to_consensus: int
    duration_seconds: float
    elo_changes: dict[str, int]


# =============================================================================
# List Available Agents
# =============================================================================


async def list_agents(client: AragoraAsyncClient) -> list[dict[str, Any]]:
    """List available agents and their current ELO ratings."""
    print("=== Available Agents ===\n")

    agents = await client.agents.list()

    print(f"Found {len(agents.get('agents', []))} agents:\n")

    for agent in agents.get("agents", [])[:10]:
        name = agent.get("name", "Unknown")
        elo = agent.get("elo", 1500)
        specialties = agent.get("specialties", [])
        status = agent.get("status", "unknown")

        status_icon = {"active": "[OK]", "degraded": "[!]", "offline": "[X]"}.get(status, "[?]")

        print(f"  {status_icon} {name:<15} ELO: {elo:>4}")
        if specialties:
            print(f"      Specialties: {', '.join(specialties[:3])}")

    return agents.get("agents", [])


# =============================================================================
# Run Comparative Debates
# =============================================================================


async def run_debate_with_agents(
    client: AragoraAsyncClient,
    task: str,
    agents: list[str],
    label: str,
) -> DebateResult:
    """Run a single debate with specified agents."""
    print(f"\n--- Running {label} ---")
    print(f"Agents: {', '.join(agents)}")

    import time

    start_time = time.time()

    # Get initial ELO ratings
    initial_elos = {}
    for agent_name in agents:
        try:
            agent_info = await client.agents.get(agent_name)
            initial_elos[agent_name] = agent_info.get("elo", 1500)
        except Exception:
            initial_elos[agent_name] = 1500

    # Create and run debate
    debate = await client.debates.create(
        task=task,
        agents=agents,
        rounds=3,
        consensus="weighted",
    )

    debate_id = debate["debate_id"]
    print(f"  Debate ID: {debate_id}")

    # Wait for completion
    while debate.get("status") in ("running", "pending"):
        await asyncio.sleep(2)
        debate = await client.debates.get(debate_id)

    duration = time.time() - start_time

    # Get final ELO ratings
    elo_changes = {}
    for agent_name in agents:
        try:
            agent_info = await client.agents.get(agent_name)
            final_elo = agent_info.get("elo", 1500)
            elo_changes[agent_name] = final_elo - initial_elos[agent_name]
        except Exception:
            elo_changes[agent_name] = 0

    # Extract results
    consensus = debate.get("consensus", {})
    result = DebateResult(
        debate_id=debate_id,
        agents=agents,
        final_answer=consensus.get("final_answer", "N/A"),
        confidence=consensus.get("confidence", 0),
        rounds_to_consensus=len(debate.get("rounds", [])),
        duration_seconds=duration,
        elo_changes=elo_changes,
    )

    print(f"  Completed in {duration:.1f}s")
    print(f"  Confidence: {result.confidence:.1%}")

    return result


async def run_comparative_debates(
    client: AragoraAsyncClient,
) -> list[DebateResult]:
    """Run the same task with different agent combinations."""
    print("\n=== Running Comparative Debates ===\n")

    task = "What is the most important factor when designing a REST API?"

    # Define agent combinations to test
    combinations = [
        (["claude", "gpt-4"], "Claude + GPT-4"),
        (["claude", "gemini"], "Claude + Gemini"),
        (["gpt-4", "gemini"], "GPT-4 + Gemini"),
        (["claude", "gpt-4", "gemini"], "All Three"),
    ]

    results = []
    for agents, label in combinations:
        result = await run_debate_with_agents(client, task, agents, label)
        results.append(result)

    return results


# =============================================================================
# Analyze Results
# =============================================================================


def analyze_results(results: list[DebateResult]) -> None:
    """Analyze and compare debate results."""
    print("\n=== Comparison Analysis ===\n")

    # Confidence comparison
    print("Confidence Comparison:")
    print("-" * 50)
    for result in sorted(results, key=lambda r: r.confidence, reverse=True):
        agents_str = " + ".join(result.agents)
        bar = "#" * int(result.confidence * 20)
        print(f"  {agents_str:<25} {result.confidence:>5.1%} {bar}")

    # Duration comparison
    print("\nDuration Comparison:")
    print("-" * 50)
    for result in sorted(results, key=lambda r: r.duration_seconds):
        agents_str = " + ".join(result.agents)
        print(f"  {agents_str:<25} {result.duration_seconds:>6.1f}s")

    # Answer similarity
    print("\nAnswer Comparison:")
    print("-" * 50)
    for result in results:
        agents_str = " + ".join(result.agents)
        answer_preview = (
            result.final_answer[:50] + "..."
            if len(result.final_answer) > 50
            else result.final_answer
        )
        print(f"  {agents_str}:")
        print(f"    {answer_preview}")

    # ELO changes
    print("\nELO Changes:")
    print("-" * 50)
    all_elo_changes: dict[str, list[int]] = {}
    for result in results:
        for agent, change in result.elo_changes.items():
            if agent not in all_elo_changes:
                all_elo_changes[agent] = []
            all_elo_changes[agent].append(change)

    for agent, changes in all_elo_changes.items():
        avg_change = sum(changes) / len(changes)
        sign = "+" if avg_change >= 0 else ""
        print(f"  {agent:<15} {sign}{avg_change:.1f} (across {len(changes)} debates)")


# =============================================================================
# ELO Leaderboard
# =============================================================================


async def show_leaderboard(client: AragoraAsyncClient) -> None:
    """Show the current agent leaderboard."""
    print("\n=== Agent Leaderboard ===\n")

    leaderboard = await client.rankings.get_leaderboard(limit=10)

    print("Top 10 Agents by ELO:")
    print("-" * 50)

    for i, entry in enumerate(leaderboard.get("rankings", []), 1):
        agent = entry.get("agent", "Unknown")
        elo = entry.get("elo", 1500)
        wins = entry.get("wins", 0)
        losses = entry.get("losses", 0)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "  ")
        print(f"  {medal} {i:>2}. {agent:<15} ELO: {elo:>4}  W/L: {wins}/{losses} ({win_rate:.0%})")


# =============================================================================
# Agent Comparison Report
# =============================================================================


async def generate_comparison_report(
    client: AragoraAsyncClient,
    results: list[DebateResult],
) -> None:
    """Generate a detailed comparison report."""
    print("\n=== Comparison Report ===\n")

    # Find best performers
    best_confidence = max(results, key=lambda r: r.confidence)
    fastest = min(results, key=lambda r: r.duration_seconds)

    print("Best Performers:")
    print(
        f"  Highest confidence: {' + '.join(best_confidence.agents)} ({best_confidence.confidence:.1%})"
    )
    print(f"  Fastest completion: {' + '.join(fastest.agents)} ({fastest.duration_seconds:.1f}s)")

    # Calculate statistics
    avg_confidence = sum(r.confidence for r in results) / len(results)
    avg_duration = sum(r.duration_seconds for r in results) / len(results)

    print("\nAverages:")
    print(f"  Confidence: {avg_confidence:.1%}")
    print(f"  Duration: {avg_duration:.1f}s")

    # Check answer consistency
    unique_answers = len({r.final_answer[:100] for r in results})
    print("\nAnswer Consistency:")
    print(f"  Unique answers: {unique_answers} / {len(results)}")

    if unique_answers == 1:
        print("  All agent combinations reached the same conclusion!")
    else:
        print("  Agent combinations reached different conclusions.")

    # Recommendations
    print("\nRecommendations:")
    if best_confidence.confidence > 0.9:
        print(f"  - High confidence debates: Use {' + '.join(best_confidence.agents)}")
    if fastest.duration_seconds < avg_duration * 0.8:
        print(f"  - Quick decisions: Use {' + '.join(fastest.agents)}")
    if len(best_confidence.agents) == 3:
        print("  - Complex decisions benefit from more agents")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run multi-agent comparison demonstration."""
    print("Aragora SDK Multi-Agent Comparison")
    print("=" * 60)

    # Check if we should run actual examples
    run_examples = os.environ.get("RUN_EXAMPLES", "false").lower() == "true"

    if not run_examples:
        print("\nMulti-agent comparison features:")
        print("  1. List agents and their ELO ratings")
        print("  2. Run same debate with different agent combos")
        print("  3. Compare confidence, duration, answers")
        print("  4. Track ELO changes across debates")
        print("  5. View agent leaderboard")
        print("  6. Generate comparison reports")
        print("\nSet RUN_EXAMPLES=true to run actual API examples.")
        return

    async with AragoraAsyncClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    ) as client:
        # List available agents
        await list_agents(client)

        # Run comparative debates
        results = await run_comparative_debates(client)

        # Analyze results
        analyze_results(results)

        # Show leaderboard
        await show_leaderboard(client)

        # Generate report
        await generate_comparison_report(client, results)

    print("\n" + "=" * 60)
    print("Multi-agent comparison complete!")
    print("\nKey Insights:")
    print("  - Different agent combos can yield different results")
    print("  - More agents often = higher confidence but longer time")
    print("  - ELO tracks agent performance over time")
    print("  - Use comparisons to find optimal agent mix for your use case")


if __name__ == "__main__":
    asyncio.run(main())
