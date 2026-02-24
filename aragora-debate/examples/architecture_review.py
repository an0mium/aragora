#!/usr/bin/env python3
"""Example: Architecture review debate — Kafka vs RabbitMQ.

Three AI agents debate whether to use Kafka or RabbitMQ for an event-driven
order processing system.  Each agent has a different stance, forcing genuine
adversarial engagement rather than polite agreement.

Usage:
    # Requires ANTHROPIC_API_KEY and OPENAI_API_KEY
    python examples/architecture_review.py

    # Or run the built-in mock mode (no API keys needed):
    python examples/architecture_review.py --mock
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import os

# Allow running from the repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aragora_debate import (
    Agent,
    Arena,
    Critique,
    DebateConfig,
    Message,
    ReceiptBuilder,
    Vote,
)
from aragora_debate.types import ConsensusMethod


# ---------------------------------------------------------------------------
# Mock agent (no API keys required)
# ---------------------------------------------------------------------------

MOCK_PROPOSALS = {
    "kafka-advocate": {
        1: (
            "Kafka is the right choice for order processing at scale.\n\n"
            "1. **Throughput**: Kafka handles millions of messages/sec with "
            "sequential disk I/O. Our projected load of 50K orders/hour is "
            "well within a single-broker capacity.\n\n"
            "2. **Durability**: Messages are persisted to disk and replicated. "
            "Order events are never lost, even during broker failures.\n\n"
            "3. **Replay**: Consumer groups can replay events for debugging, "
            "analytics, or rebuilding downstream state — critical for financial "
            "reconciliation.\n\n"
            "4. **Ecosystem**: Kafka Connect and Kafka Streams provide built-in "
            "ETL and stream processing without additional infrastructure."
        ),
        2: (
            "Addressing the critiques:\n\n"
            "1. **Operational complexity**: Kafka on managed services (Confluent "
            "Cloud, AWS MSK) eliminates ZooKeeper management. Our team doesn't "
            "need to run brokers directly.\n\n"
            "2. **Latency**: For order processing, p99 latency of 5-10ms is "
            "acceptable. We're not building a trading system.\n\n"
            "3. **The replay capability alone justifies Kafka** — when an order "
            "processing bug is discovered, we can replay events to fix state "
            "rather than running manual database migrations."
        ),
    },
    "rabbitmq-advocate": {
        1: (
            "RabbitMQ is the pragmatic choice for our order processing system.\n\n"
            "1. **Simplicity**: RabbitMQ's routing model (exchanges, queues, "
            "bindings) maps directly to order workflows — new orders go to "
            "the processing queue, failures to the DLX, notifications fan out.\n\n"
            "2. **Low latency**: Sub-millisecond message delivery for the "
            "order → payment → fulfillment pipeline.\n\n"
            "3. **Operational maturity**: Our team has 3 years of RabbitMQ "
            "experience. Kafka would require hiring or significant training.\n\n"
            "4. **Resource efficiency**: RabbitMQ runs on 2 nodes with 4GB RAM. "
            "Kafka's minimum viable cluster is 3 brokers + ZooKeeper."
        ),
        2: (
            "The Kafka advocate overstates our scale requirements:\n\n"
            "1. **50K orders/hour is ~14 orders/second**. RabbitMQ handles "
            "40K+ messages/second on modest hardware. We're at 0.03% capacity.\n\n"
            "2. **Replay is a feature, not a requirement**: We can achieve "
            "the same with idempotent consumers + event store pattern if needed.\n\n"
            "3. **Managed Kafka still costs 3-5x more** than managed RabbitMQ "
            "(CloudAMQP) for equivalent throughput. At our scale, we'd be "
            "paying for capacity we'll never use."
        ),
    },
    "neutral-analyst": {
        1: (
            "Both Kafka and RabbitMQ can handle this workload. The decision "
            "should be driven by non-functional requirements:\n\n"
            "1. **If event replay is a hard requirement** (audit compliance, "
            "financial reconciliation), Kafka is the only option that provides "
            "this natively.\n\n"
            "2. **If time-to-production matters most**, RabbitMQ wins — the "
            "team already knows it, and the routing model fits order workflows.\n\n"
            "3. **If we expect 10x growth** in 18 months, Kafka provides more "
            "headroom without re-architecture.\n\n"
            "Recommendation: Start with RabbitMQ for the MVP, but design the "
            "consumer interface to be broker-agnostic so we can migrate later."
        ),
        2: (
            "After reviewing both positions, I want to highlight the key tradeoff:\n\n"
            "**The RabbitMQ advocate's cost analysis is compelling** — at 14 msg/sec, "
            "we're dramatically over-provisioning with Kafka.\n\n"
            "**But the Kafka advocate's replay argument is underappreciated** — "
            "in financial systems, the ability to replay events is worth the "
            "extra cost because the alternative (manual data fixes) is far more "
            "expensive when things go wrong.\n\n"
            "**My updated recommendation**: RabbitMQ + event store pattern. "
            "This gives us replay capability without Kafka's operational cost, "
            "and the team can build it with existing skills."
        ),
    },
}


class MockAgent(Agent):
    """Simulated agent using pre-written responses for demo purposes."""

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        round_num = 1
        if context:
            round_num = max((m.round for m in context), default=0) + 1
        proposals = MOCK_PROPOSALS.get(self.name, {})
        return proposals.get(round_num, proposals.get(1, "I agree with the analysis presented."))

    async def critique(self, proposal: str, task: str, **kw) -> Critique:
        target = kw.get("target_agent", "unknown")
        if self.name == "kafka-advocate" and "RabbitMQ" in proposal:
            return Critique(
                agent=self.name,
                target_agent=target,
                target_content=proposal,
                issues=[
                    "Ignores replay requirements for financial compliance",
                    "Cost comparison doesn't include incident recovery time",
                ],
                suggestions=["Quantify the cost of a single data reconciliation incident"],
                severity=6.0,
            )
        elif self.name == "rabbitmq-advocate" and "Kafka" in proposal:
            return Critique(
                agent=self.name,
                target_agent=target,
                target_content=proposal,
                issues=[
                    "Over-engineers for current scale (14 msg/sec vs millions)",
                    "Operational complexity risk with a team of 4 engineers",
                ],
                suggestions=["Compare managed service costs at our actual volume"],
                severity=7.0,
            )
        return Critique(
            agent=self.name,
            target_agent=target,
            target_content=proposal,
            issues=["Could be more specific about migration timeline"],
            suggestions=["Add concrete cost estimates"],
            severity=3.0,
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        if self.name == "kafka-advocate":
            return Vote(
                agent=self.name,
                choice="kafka-advocate",
                reasoning="Replay is non-negotiable for financial systems",
                confidence=0.8,
            )
        elif self.name == "rabbitmq-advocate":
            return Vote(
                agent=self.name,
                choice="neutral-analyst",
                reasoning="The hybrid approach addresses my concerns",
                confidence=0.7,
            )
        else:
            return Vote(
                agent=self.name,
                choice="neutral-analyst",
                reasoning="Pragmatic middle ground with migration path",
                confidence=0.85,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(use_mock: bool = True) -> None:
    if use_mock:
        agents = [
            MockAgent("kafka-advocate", stance="affirmative"),
            MockAgent("rabbitmq-advocate", stance="negative"),
            MockAgent("neutral-analyst", stance="neutral"),
        ]
    else:
        # Replace with real agent implementations
        raise NotImplementedError(
            "Set ANTHROPIC_API_KEY / OPENAI_API_KEY and implement real agents, or use --mock mode."
        )

    arena = Arena(
        question=(
            "Should we use Apache Kafka or RabbitMQ for our new event-driven "
            "order processing system? We process ~50K orders/hour, need "
            "exactly-once delivery guarantees, and have a team of 4 backend engineers."
        ),
        agents=agents,
        config=DebateConfig(
            rounds=2,
            consensus_method=ConsensusMethod.SUPERMAJORITY,
            early_stopping=True,
        ),
        context=(
            "Current stack: Python 3.12, PostgreSQL, Redis, Docker/K8s. "
            "Team has 3 years RabbitMQ experience, no Kafka experience. "
            "System must comply with PCI DSS for payment processing."
        ),
    )

    print("=" * 60)
    print("ARCHITECTURE REVIEW: Kafka vs RabbitMQ")
    print("=" * 60)
    print()

    result = await arena.run()

    # Print debate transcript
    current_round = 0
    for msg in result.messages:
        if msg.round != current_round:
            current_round = msg.round
            print(f"\n--- Round {current_round} ---\n")
        role_label = {"proposer": "PROPOSE", "critic": "CRITIQUE", "voter": "VOTE"}.get(
            msg.role, msg.role.upper()
        )
        print(f"[{role_label}] {msg.agent}:")
        print(f"  {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
        print()

    # Print receipt
    print("=" * 60)
    print("DECISION RECEIPT")
    print("=" * 60)
    assert result.receipt is not None
    print(result.receipt.to_markdown())

    # Export JSON
    print("\n--- JSON Receipt ---")
    print(ReceiptBuilder.to_json(result.receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Architecture review debate demo")
    parser.add_argument(
        "--mock", action="store_true", default=True, help="Use mock agents (no API keys needed)"
    )
    parser.add_argument(
        "--live", action="store_true", help="Use real LLM agents (requires API keys)"
    )
    args = parser.parse_args()
    asyncio.run(main(use_mock=not args.live))
