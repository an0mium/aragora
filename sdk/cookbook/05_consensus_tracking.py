#!/usr/bin/env python3
"""
05_consensus_tracking.py - Monitor consensus evolution and voting patterns.

This example demonstrates how to:
- Track consensus evolution across debate rounds
- Analyze voting patterns and convergence
- Detect when consensus is likely to form

Usage:
    python 05_consensus_tracking.py --dry-run
    python 05_consensus_tracking.py --topic "Should we adopt GraphQL?"
"""

import argparse
import asyncio
from aragora_sdk import ArenaClient, DebateConfig, Agent
from aragora_sdk.consensus import ConsensusTracker, VotePattern, ConvergenceMetrics


class ConsensusMonitor:
    """Monitor and analyze consensus evolution during debates."""

    def __init__(self):
        self.round_metrics = []
        self.vote_history = []

    def on_round_complete(self, metrics: ConvergenceMetrics) -> None:
        """Called after each debate round with convergence metrics."""
        self.round_metrics.append(metrics)

        print(f"\n--- Round {len(self.round_metrics)} Metrics ---")
        print(f"Agreement level: {metrics.agreement_level:.2%}")
        print(f"Semantic similarity: {metrics.semantic_similarity:.2%}")
        print(f"Position drift: {metrics.position_drift:.2%}")
        print(f"Convergence trend: {'converging' if metrics.is_converging else 'diverging'}")

    def on_vote(self, pattern: VotePattern) -> None:
        """Called when voting occurs to track patterns."""
        self.vote_history.append(pattern)

        print(f"\n[VOTE] {pattern.agent} -> {pattern.position}")
        print(f"  Confidence: {pattern.confidence:.2%}")
        print(f"  Reasoning: {pattern.reasoning[:80]}...")

    def analyze_convergence(self) -> dict:
        """Analyze overall convergence patterns."""
        if not self.round_metrics:
            return {"status": "no_data"}

        # Calculate convergence velocity
        agreements = [m.agreement_level for m in self.round_metrics]
        if len(agreements) >= 2:
            velocity = agreements[-1] - agreements[0]
        else:
            velocity = 0

        # Find dominant positions
        position_counts = {}
        for vote in self.vote_history:
            pos = vote.position
            position_counts[pos] = position_counts.get(pos, 0) + 1

        dominant = (
            max(position_counts.items(), key=lambda x: x[1]) if position_counts else (None, 0)
        )

        return {
            "total_rounds": len(self.round_metrics),
            "final_agreement": agreements[-1] if agreements else 0,
            "convergence_velocity": velocity,
            "dominant_position": dominant[0],
            "dominant_vote_count": dominant[1],
            "is_converging": velocity > 0,
        }


async def run_consensus_tracking(topic: str, dry_run: bool = False) -> dict:
    """Run a debate while tracking consensus evolution."""

    monitor = ConsensusMonitor()

    if dry_run:
        print(f"[DRY RUN] Topic: {topic}")
        print("[DRY RUN] Simulating consensus tracking...")

        # Simulate convergence data
        from aragora_sdk.consensus import ConvergenceMetrics

        for i in range(3):
            metrics = ConvergenceMetrics(
                agreement_level=0.5 + (i * 0.15),
                semantic_similarity=0.6 + (i * 0.1),
                position_drift=0.3 - (i * 0.1),
                is_converging=True,
            )
            monitor.on_round_complete(metrics)

        analysis = monitor.analyze_convergence()
        print("\n=== Convergence Analysis ===")
        print(f"Converging: {analysis['is_converging']}")
        print(f"Final agreement: {analysis['final_agreement']:.2%}")
        return {"status": "dry_run", "analysis": analysis}

    # Initialize with consensus tracking
    client = ArenaClient()
    tracker = ConsensusTracker(
        on_round_complete=monitor.on_round_complete,
        on_vote=monitor.on_vote,
    )

    agents = [
        Agent(name="claude", model="claude-sonnet-4-20250514"),
        Agent(name="gpt", model="gpt-4o"),
        Agent(name="gemini", model="gemini-2.0-flash"),
    ]

    config = DebateConfig(
        topic=topic,
        agents=agents,
        rounds=4,
        consensus_tracker=tracker,
        # Stop early if consensus is clear
        early_consensus_threshold=0.85,
    )

    result = await client.run_debate(config)
    analysis = monitor.analyze_convergence()

    print("\n=== Final Analysis ===")
    print(f"Consensus reached: {result.consensus_reached}")
    print(f"Convergence velocity: {analysis['convergence_velocity']:.2%}")
    print(f"Dominant position: {analysis['dominant_position']}")

    return {**result.to_dict(), "convergence_analysis": analysis}


def main():
    parser = argparse.ArgumentParser(description="Track consensus evolution")
    parser.add_argument(
        "--topic", default="Should we adopt GraphQL over REST?", help="Topic for the debate"
    )
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    args = parser.parse_args()

    result = asyncio.run(run_consensus_tracking(args.topic, args.dry_run))
    return result


if __name__ == "__main__":
    main()
