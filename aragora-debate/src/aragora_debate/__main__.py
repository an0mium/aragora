"""Run a demo adversarial debate with zero API keys.

Usage::

    python -m aragora_debate
    python -m aragora_debate --topic "Should we use Kubernetes?"
    python -m aragora_debate --topic "Kafka vs RabbitMQ?" --rounds 2
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from aragora_debate.styled_mock import StyledMockAgent
from aragora_debate.arena import Arena
from aragora_debate.types import DebateConfig

# ---------------------------------------------------------------------------
# ANSI helpers (no external deps)
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_COLORS = {
    "blue": "\033[34m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
}

_AGENT_COLORS = ["blue", "red", "yellow", "green", "cyan", "magenta"]


def _c(text: str, color: str, bold: bool = False) -> str:
    prefix = _COLORS.get(color, "")
    if bold:
        prefix = _BOLD + prefix
    return f"{prefix}{text}{_RESET}"


def _header(text: str) -> str:
    return f"\n{_BOLD}{text}{_RESET}\n" + "─" * len(text)


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

async def _run_demo(topic: str, rounds: int) -> None:
    agents = [
        StyledMockAgent("analyst", style="supportive"),
        StyledMockAgent("critic", style="critical"),
        StyledMockAgent("moderator", style="balanced"),
    ]
    color_map = {a.name: _AGENT_COLORS[i] for i, a in enumerate(agents)}

    print(_c("aragora-debate", "cyan", bold=True))
    print(_c("Adversarial multi-model debate engine", "cyan"))
    print()
    print(f"  Topic:   {_BOLD}{topic}{_RESET}")
    print(f"  Agents:  {', '.join(_c(a.name, color_map[a.name]) for a in agents)}")
    print(f"  Rounds:  {rounds}")
    print(f"  Method:  majority consensus")

    config = DebateConfig(rounds=rounds, early_stopping=True)
    arena = Arena(question=topic, agents=agents, config=config)
    result = await arena.run()

    # --- Show round-by-round highlights ---
    shown_rounds: set[int] = set()
    for msg in result.messages:
        if msg.round not in shown_rounds and msg.role == "proposer":
            shown_rounds.add(msg.round)
            print(_header(f"Round {msg.round}"))

        if msg.role == "proposer":
            agent_label = _c(msg.agent, color_map.get(msg.agent, "cyan"), bold=True)
            print(f"\n  {agent_label} proposes:")
            content = msg.content[:300]
            if len(msg.content) > 300:
                content += "..."
            for line in content.split("\n"):
                print(f"    {_DIM}{line}{_RESET}")

    # --- Critiques summary ---
    if result.critiques:
        print(_header("Critiques"))
        for crit in result.critiques[:6]:
            src = _c(crit.agent, color_map.get(crit.agent, "cyan"))
            tgt = _c(crit.target_agent, color_map.get(crit.target_agent, "cyan"))
            print(f"  {src} → {tgt}  (severity {crit.severity}/10)")
            for issue in crit.issues[:2]:
                print(f"    • {_DIM}{issue}{_RESET}")

    # --- Votes ---
    if result.votes:
        print(_header("Votes"))
        for vote in result.votes:
            voter = _c(vote.agent, color_map.get(vote.agent, "cyan"))
            chosen = _c(vote.choice, color_map.get(vote.choice, "cyan"), bold=True)
            print(f"  {voter} → {chosen}  (confidence {vote.confidence:.0%})")

    # --- Receipt ---
    print(_header("Decision Receipt"))
    if result.receipt:
        print()
        print(result.receipt.to_markdown())
    else:
        print(f"  Status: {result.status}")
        print(f"  Confidence: {result.confidence:.0%}")

    print()
    status_color = "green" if result.consensus_reached else "yellow"
    print(
        _c(
            f"✓ Debate complete in {result.duration_seconds:.2f}s "
            f"({result.rounds_used} round{'s' if result.rounds_used != 1 else ''})",
            status_color,
        )
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m aragora_debate",
        description="Run an adversarial multi-agent debate (no API keys needed)",
    )
    parser.add_argument(
        "--topic",
        default="Should we use microservices or a monolith?",
        help="The question to debate (default: microservices vs monolith)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of debate rounds (default: 2)",
    )
    args = parser.parse_args()
    asyncio.run(_run_demo(args.topic, args.rounds))


if __name__ == "__main__":
    main()
