"""Demo: Trickster + convergence detection with event callbacks.

No API keys needed — uses StyledMockAgent for deterministic output.

Run::

    python examples/trickster_demo.py
"""

from __future__ import annotations

import asyncio

from aragora_debate.arena import Arena
from aragora_debate.events import DebateEvent, EventType
from aragora_debate.styled_mock import StyledMockAgent
from aragora_debate.types import DebateConfig


def on_event(event: DebateEvent) -> None:
    """Print interesting events as they happen."""
    if event.event_type == EventType.DEBATE_START:
        agents = event.data.get("agents", [])
        print(f"\n=== Debate started with {len(agents)} agents ===")

    elif event.event_type == EventType.ROUND_START:
        print(f"\n--- Round {event.round_num} ---")

    elif event.event_type == EventType.PROPOSAL:
        length = event.data.get("content_length", 0)
        print(f"  [{event.agent}] proposed ({length} chars)")

    elif event.event_type == EventType.CONVERGENCE_DETECTED:
        sim = event.data.get("similarity", 0)
        print(f"  ** Convergence detected: {sim:.0%} similarity **")

    elif event.event_type == EventType.TRICKSTER_INTERVENTION:
        targets = event.data.get("targets", [])
        itype = event.data.get("type", "challenge")
        print(f"  ** Trickster ({itype}) -> {', '.join(targets)} **")

    elif event.event_type == EventType.CONSENSUS_CHECK:
        reached = event.data.get("reached", False)
        conf = event.data.get("confidence", 0)
        status = "REACHED" if reached else "not reached"
        print(f"  Consensus: {status} ({conf:.0%} confidence)")

    elif event.event_type == EventType.DEBATE_END:
        print(f"\n=== Debate complete ===")


async def main() -> None:
    agents = [
        StyledMockAgent("analyst", style="supportive"),
        StyledMockAgent("critic", style="critical"),
        StyledMockAgent("moderator", style="balanced"),
    ]

    config = DebateConfig(
        rounds=3,
        early_stopping=True,
        enable_trickster=True,
        enable_convergence=True,
        trickster_sensitivity=0.7,
        convergence_threshold=0.85,
    )

    arena = Arena(
        question="Should we adopt event-driven architecture for our payment system?",
        agents=agents,
        config=config,
        on_event=on_event,
    )

    result = await arena.run()

    print(f"\nResult: {result.status}")
    print(f"Rounds used: {result.rounds_used}")
    print(f"Consensus: {'yes' if result.consensus_reached else 'no'} ({result.confidence:.0%})")
    print(
        f"Convergence detected: {result.convergence_detected} "
        f"(similarity: {result.final_similarity:.0%})"
    )
    print(f"Trickster interventions: {result.trickster_interventions}")

    if result.receipt:
        print(f"\n{result.receipt.to_markdown()}")


if __name__ == "__main__":
    asyncio.run(main())
