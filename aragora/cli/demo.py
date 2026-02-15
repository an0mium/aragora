"""
CLI demo command -- run a self-contained adversarial debate in one command.

No API keys required.  Uses StyledMockAgent from the aragora-debate package
to simulate a realistic multi-perspective debate with real-time output.

Usage:
    aragora demo                          # Default: microservices debate
    aragora demo --topic "Should we use Kubernetes?"
    aragora demo --list                   # Show available demos
    aragora demo --server                 # Start offline server and open browser
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

try:
    from aragora_debate.arena import Arena
    from aragora_debate.events import EventType
    from aragora_debate.styled_mock import StyledMockAgent
    from aragora_debate.types import DebateConfig, DebateResult

    HAS_ARAGORA_DEBATE = True
except ImportError:
    HAS_ARAGORA_DEBATE = False
    Arena = None  # type: ignore[assignment,misc]
    EventType = None  # type: ignore[assignment,misc]
    StyledMockAgent = None  # type: ignore[assignment,misc]
    DebateConfig = None  # type: ignore[assignment,misc]
    DebateResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Pre-configured demo scenarios
# ---------------------------------------------------------------------------

DEMO_TASKS: dict[str, dict[str, Any]] = {
    "microservices": {
        "topic": "Should we adopt microservices or keep our monolith?",
        "description": "Classic architecture decision with strong arguments on both sides",
    },
    "rate-limiter": {
        "topic": "Design a distributed rate limiter that handles 1M requests/second across multiple regions",
        "description": "System design challenge requiring distributed consensus",
    },
    "auth": {
        "topic": "Design a secure authentication system with passwordless login and MFA support",
        "description": "Security-critical design with usability tradeoffs",
    },
    "cache": {
        "topic": "Design a cache invalidation strategy for a social media feed with 100M users",
        "description": "Classic distributed systems problem",
    },
    "kubernetes": {
        "topic": "Should we migrate from VMs to Kubernetes for our production workloads?",
        "description": "Infrastructure modernization with operational complexity tradeoffs",
    },
}

_DEFAULT_DEMO = "microservices"


def list_demos() -> list[str]:
    """Return available demo names."""
    return list(DEMO_TASKS.keys())


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

_PHASE_LABELS = {
    EventType.DEBATE_START: "DEBATE STARTING",
    EventType.ROUND_START: "ROUND",
    EventType.ROUND_END: "ROUND COMPLETE",
    EventType.CONSENSUS_CHECK: "CONSENSUS CHECK",
    EventType.DEBATE_END: "DEBATE COMPLETE",
}


def _print_banner(topic: str, agents: list[str]) -> None:
    """Print the welcome banner."""
    print()
    print("=" * 64)
    print("  ARAGORA DEMO -- Adversarial Decision Stress-Test")
    print("=" * 64)
    print()
    print("  Aragora orchestrates multiple AI agents to adversarially vet")
    print("  decisions, producing audit-ready decision receipts.")
    print()
    print("  What you are about to see:")
    print("    1. Agents propose positions on a question")
    print("    2. Agents critique each other's proposals")
    print("    3. Agents vote on the strongest position")
    print("    4. Consensus is evaluated and a receipt is generated")
    print()
    print(f"  Topic:  {topic}")
    print(f"  Agents: {', '.join(agents)}")
    print("  Rounds: 2")
    print("  Mode:   Offline (mock agents, no API keys needed)")
    print()
    print("-" * 64)


def _print_proposal(agent: str, style: str, content: str) -> None:
    """Print a proposal from an agent."""
    label = f"[{agent.upper()}] ({style})"
    print(f"\n  {label}")
    # Wrap content at ~72 chars with indent
    for line in _wrap(content, width=58):
        print(f"    {line}")


def _print_critique(agent: str, target: str, issues: list[str]) -> None:
    """Print a critique summary."""
    print(f"\n  [{agent.upper()}] critiques {target}:")
    for issue in issues[:3]:
        print(f"    - {issue}")


def _print_vote(agent: str, choice: str, confidence: float) -> None:
    """Print a vote."""
    print(f"    {agent} -> {choice} ({confidence:.0%} confidence)")


def _wrap(text: str, width: int = 72) -> list[str]:
    """Simple word wrap without textwrap import."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines or [""]


def _build_receipt_data(result: DebateResult, elapsed: float) -> dict[str, Any]:
    """Build a receipt dict from a DebateResult for saving/rendering."""
    verdict_str = "consensus"
    if result.verdict:
        verdict_str = result.verdict.value

    receipt_id = ""
    artifact_hash = ""
    signature_algorithm = ""
    consensus_proof: dict[str, Any] = {}

    if result.receipt:
        receipt_id = result.receipt.receipt_id
        artifact_hash = result.receipt.signature or ""
        signature_algorithm = result.receipt.signature_algorithm or ""
        if result.receipt.consensus:
            consensus_proof = {
                "reached": result.receipt.consensus.reached,
                "method": result.receipt.consensus.method.value,
                "confidence": result.receipt.consensus.confidence,
                "supporting_agents": result.receipt.consensus.supporting_agents,
                "dissenting_agents": result.receipt.consensus.dissenting_agents,
            }

    dissent_items = []
    if result.dissenting_views:
        for view in result.dissenting_views:
            dissent_items.append(str(view))

    summary = result.final_answer or ""

    return {
        "receipt_id": receipt_id,
        "question": result.task,
        "verdict": verdict_str,
        "confidence": result.confidence,
        "agents": result.participants,
        "rounds": result.rounds_used,
        "summary": summary,
        "dissent": dissent_items,
        "dissenting_views": dissent_items,
        "consensus_proof": consensus_proof,
        "artifact_hash": artifact_hash,
        "signature_algorithm": signature_algorithm,
        "elapsed_seconds": elapsed,
        "mode": "demo (offline)",
        "proposals": result.proposals,
    }


def _save_demo_receipt(
    result: DebateResult,
    elapsed: float,
    output_path: str,
) -> str:
    """Save a demo receipt to file. Returns the path written."""
    import json

    receipt_data = _build_receipt_data(result, elapsed)
    path = Path(output_path)

    if path.suffix.lower() in (".html", ".htm"):
        from aragora.cli.receipt_formatter import receipt_to_html

        path.write_text(receipt_to_html(receipt_data))
    elif path.suffix.lower() == ".md":
        from aragora.cli.receipt_formatter import receipt_to_markdown

        path.write_text(receipt_to_markdown(receipt_data))
    else:
        path.write_text(json.dumps(receipt_data, indent=2, default=str))

    return str(path)


def _print_result(result: DebateResult, elapsed: float) -> None:
    """Print the final debate result summary."""
    print()
    print("=" * 64)
    print("  DECISION SUMMARY")
    print("=" * 64)

    # Verdict
    verdict_str = "Consensus Reached" if result.consensus_reached else "No Consensus"
    if result.verdict:
        verdict_str = result.verdict.value.replace("_", " ").title()
    print(f"\n  Verdict:    {verdict_str}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Rounds:     {result.rounds_used}")
    print(f"  Duration:   {elapsed:.2f}s")

    # Winning position
    if result.final_answer:
        print()
        print("  WINNING POSITION:")
        print("  " + "-" * 40)
        for line in _wrap(result.final_answer, width=58):
            print(f"    {line}")

    # Key arguments (from proposals)
    if result.proposals:
        print()
        print("  KEY ARGUMENTS:")
        print("  " + "-" * 40)
        for agent_name, proposal_text in result.proposals.items():
            summary = proposal_text[:120]
            if len(proposal_text) > 120:
                summary += "..."
            print(f"    {agent_name}: {summary}")

    # Dissenting views
    if result.dissenting_views:
        print()
        print("  DISSENTING VIEWS:")
        print("  " + "-" * 40)
        for view in result.dissenting_views:
            print(f"    - {view}")

    # Decision receipt
    if result.receipt:
        print()
        print("  DECISION RECEIPT:")
        print("  " + "-" * 40)
        print(f"    ID:        {result.receipt.receipt_id}")
        print(f"    Integrity: {result.receipt.signature_algorithm}")
        sig = result.receipt.signature or ""
        print(f"    Hash:      {sig[:40]}...")

    # Action items
    print()
    print("  SUGGESTED NEXT STEPS:")
    print("  " + "-" * 40)
    if result.consensus_reached:
        print("    1. Review the winning position with stakeholders")
        print("    2. Address dissenting views before proceeding")
        print("    3. Create an implementation plan")
        print("    4. Set up monitoring and success criteria")
    else:
        print("    1. Gather more information on contested points")
        print("    2. Run a deeper debate with more rounds")
        print("    3. Consider a phased approach to reduce risk")
        print("    4. Consult domain experts on unresolved concerns")

    print()
    print("=" * 64)
    print()
    print("  Try with real AI agents:")
    print("    aragora ask 'Your question' --agents anthropic-api,openai-api")
    print()
    print("  Full decision pipeline:")
    print("    aragora decide 'Your question'")
    print()
    print("  Learn more:")
    print("    aragora doctor            # Check system health")
    print("    aragora quickstart --demo  # Guided onboarding")
    print()


# ---------------------------------------------------------------------------
# Core demo runner
# ---------------------------------------------------------------------------

_AGENT_CONFIGS = [
    ("Analyst", "supportive"),
    ("Critic", "critical"),
    ("Synthesizer", "balanced"),
    ("Devil's Advocate", "contrarian"),
]


async def _run_demo_debate(topic: str) -> tuple[DebateResult, float]:
    """Run a demo debate and return (result, elapsed_seconds)."""
    agents = [
        StyledMockAgent(name, style=style)  # type: ignore[arg-type]
        for name, style in _AGENT_CONFIGS
    ]
    agent_names = [a.name for a in agents]

    _print_banner(topic, agent_names)

    # Track phases for printing
    proposals_shown: set[str] = set()
    critiques_shown: set[str] = set()
    votes_shown: set[str] = set()

    config = DebateConfig(
        rounds=2,
        early_stopping=False,
    )

    arena = Arena(
        question=topic,
        agents=agents,
        config=config,
    )

    start = time.monotonic()
    result = await arena.run()
    elapsed = time.monotonic() - start

    # Print the debate progression from recorded messages
    current_round = 0
    for msg in result.messages:
        # Round header
        if msg.round != current_round:
            current_round = msg.round
            print(f"\n{'  ':>2}--- Round {current_round} {'---':->50}")

        if msg.role == "proposer" and msg.agent not in proposals_shown:
            proposals_shown.add(msg.agent)
            # Find the agent's style
            style = "balanced"
            for name, s in _AGENT_CONFIGS:
                if name == msg.agent:
                    style = s
                    break
            _print_proposal(msg.agent, style, msg.content)

        elif msg.role == "critic":
            key = f"{msg.agent}->{msg.round}"
            if key not in critiques_shown:
                critiques_shown.add(key)
                # Extract critique info from the content
                issues = []
                for line in msg.content.split("\n"):
                    line = line.strip()
                    if line.startswith("- "):
                        issues.append(line[2:])
                if not issues:
                    issues = [msg.content[:100]]
                # Find target from critique objects
                target = "other"
                for c in result.critiques:
                    if c.agent == msg.agent and c.content == msg.content:
                        target = c.target_agent
                        issues = c.issues[:3]
                        break
                _print_critique(msg.agent, target, issues)

        elif msg.role == "voter" and msg.agent not in votes_shown:
            if not votes_shown:
                print(f"\n  VOTES (Round {msg.round}):")
            votes_shown.add(msg.agent)
            # Find vote confidence
            conf = 0.7
            for v in result.votes:
                if v.agent == msg.agent:
                    conf = v.confidence
                    choice = v.choice
                    _print_vote(msg.agent, choice, conf)
                    break

    _print_result(result, elapsed)
    return result, elapsed


def _run_server_demo() -> None:
    """Start the server in offline/demo mode."""
    import subprocess

    print()
    print("=" * 64)
    print("  ARAGORA SERVER DEMO")
    print("=" * 64)
    print()
    print("  Starting server in offline mode (no API keys needed)...")
    print("  The web UI will be available at: http://localhost:8080")
    print()
    print("  Press Ctrl+C to stop the server.")
    print()
    print("=" * 64)
    print()

    try:
        subprocess.run(
            [sys.executable, "-m", "aragora.cli.main", "serve", "--offline"],
            check=False,
        )
    except KeyboardInterrupt:
        print("\n  Server stopped.")


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def run_demo(
    demo_name: str,
    receipt_path: str | None = None,
) -> DebateResult | None:
    """Run a specific demo by name. Returns the DebateResult or None.

    Args:
        demo_name: Name of the demo scenario (e.g. "microservices").
        receipt_path: If provided, save a decision receipt to this file path.
    """
    if demo_name not in DEMO_TASKS:
        print(f"Unknown demo: {demo_name}")
        print(f"Available demos: {', '.join(DEMO_TASKS.keys())}")
        return None

    topic = DEMO_TASKS[demo_name]["topic"]
    result, elapsed = asyncio.run(_run_demo_debate(topic))

    if receipt_path:
        saved = _save_demo_receipt(result, elapsed, receipt_path)
        print(f"\n  Receipt saved to: {saved}")

    return result


def main(args: argparse.Namespace) -> None:
    """Handle 'demo' command."""
    if not HAS_ARAGORA_DEBATE:
        print("Error: aragora-debate package is not installed.")
        print("Install it with: pip install aragora-debate")
        sys.exit(1)

    # --list flag
    if getattr(args, "list_demos", False):
        print("\nAvailable demos:")
        for name, info in DEMO_TASKS.items():
            marker = " (default)" if name == _DEFAULT_DEMO else ""
            print(f"  {name:<16} {info['description']}{marker}")
        print("\nRun with: aragora demo <name>")
        return

    # --server flag
    if getattr(args, "server", False):
        _run_server_demo()
        return

    receipt_path = getattr(args, "receipt", None)

    # Custom topic via --topic
    custom_topic = getattr(args, "topic", None)
    if custom_topic:
        result, elapsed = asyncio.run(_run_demo_debate(custom_topic))
        if receipt_path:
            saved = _save_demo_receipt(result, elapsed, receipt_path)
            print(f"\n  Receipt saved to: {saved}")
        return

    # Named demo (or default)
    demo_name = getattr(args, "name", None) or _DEFAULT_DEMO
    run_demo(demo_name, receipt_path=receipt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Aragora demos")
    parser.add_argument(
        "name",
        nargs="?",
        default=_DEFAULT_DEMO,
        help=f"Demo name (available: {', '.join(DEMO_TASKS.keys())})",
    )
    parser.add_argument(
        "--topic",
        "-t",
        help="Custom topic to debate (overrides named demo)",
    )
    parser.add_argument(
        "--list",
        dest="list_demos",
        action="store_true",
        help="List available demo scenarios",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start server in offline demo mode",
    )
    main(parser.parse_args())
