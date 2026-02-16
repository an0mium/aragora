"""
Quickstart CLI command: zero-to-receipt onboarding in one command.

Guides new users through their first adversarial debate:
1. Checks for API keys (loads .env if present)
2. Accepts a question via --question or interactive prompt (default provided)
3. Runs a 2-round demo debate with auto-detected or mock agents
4. Displays verdict with confidence and elapsed time
5. Generates HTML receipt and opens in browser (unless --no-browser)
6. Optionally saves the decision receipt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

_DEFAULT_QUESTION = "Should we adopt microservices or keep our monolith?"


def add_quickstart_parser(subparsers: Any) -> None:
    """Register the 'quickstart' subcommand."""
    qs_parser = subparsers.add_parser(
        "quickstart",
        help="Guided zero-to-receipt first debate (new user onboarding)",
        description="""
Run your first adversarial debate in under 60 seconds.

Automatically detects available API keys, picks agents, runs a fast
2-round debate, and opens the decision receipt in your browser.
No configuration needed.

Examples:
  aragora quickstart --demo                              # Zero-config demo
  aragora quickstart --question "Should we use Kubernetes?"
  aragora quickstart --question "Migrate to TypeScript?" --output receipt.json
  aragora quickstart --demo --no-browser                 # CI/headless mode
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    qs_parser.add_argument(
        "--question",
        "-q",
        help="The question to debate (uses a default if omitted with --demo)",
    )
    qs_parser.add_argument(
        "--output",
        "-o",
        help="Save receipt to file (supports .json, .md, .html)",
    )
    qs_parser.add_argument(
        "--demo",
        action="store_true",
        help="Use mock agents (no API keys required)",
    )
    qs_parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=2,
        help="Number of debate rounds (default: 2)",
    )
    qs_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "md", "html"],
        default="json",
        help="Receipt output format (default: json)",
    )
    qs_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open receipt in browser (for CI/headless environments)",
    )
    qs_parser.set_defaults(func=cmd_quickstart)


def _load_dotenv() -> bool:
    """Try to load .env file from cwd or parent. Returns True if loaded."""
    for candidate in [Path.cwd() / ".env", Path.cwd().parent / ".env"]:
        if candidate.is_file():
            try:
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        if key and key not in os.environ:
                            os.environ[key] = value
                return True
            except OSError:
                pass
    return False


def _detect_agents() -> list[tuple[str, str | None]]:
    """Detect available agents based on API keys.

    Returns list of (provider, model) tuples.
    """
    agents: list[tuple[str, str | None]] = []

    if os.environ.get("ANTHROPIC_API_KEY"):
        agents.append(("anthropic-api", "claude-sonnet-4-5-20250929"))
    if os.environ.get("OPENAI_API_KEY"):
        agents.append(("openai-api", "gpt-4o"))
    if os.environ.get("GEMINI_API_KEY"):
        agents.append(("gemini", None))
    if os.environ.get("MISTRAL_API_KEY"):
        agents.append(("mistral", None))
    if os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"):
        agents.append(("grok", None))
    if os.environ.get("OPENROUTER_API_KEY"):
        agents.append(("deepseek", None))

    return agents


def _get_question(args: argparse.Namespace) -> str | None:
    """Get the debate question from args, default, or interactive prompt."""
    if args.question:
        return args.question

    # In demo mode, use the default question instead of prompting
    if getattr(args, "demo", False):
        return _DEFAULT_QUESTION

    # Interactive prompt
    try:
        print("\nWhat question should the agents debate?")
        print("(Example: 'Should we migrate from REST to GraphQL?')\n")
        question = input("> ").strip()
        return question if question else None
    except (EOFError, KeyboardInterrupt):
        return None


def _save_receipt(receipt_data: dict[str, Any], path: str, fmt: str) -> None:
    """Save receipt to file in the specified format."""
    from aragora.cli.receipt_formatter import receipt_to_html, receipt_to_markdown

    output_path = Path(path)

    if fmt == "json" or output_path.suffix == ".json":
        output_path.write_text(json.dumps(receipt_data, indent=2, default=str))
    elif fmt == "md" or output_path.suffix == ".md":
        md = receipt_to_markdown(receipt_data)
        output_path.write_text(md)
    elif fmt == "html" or output_path.suffix == ".html":
        html = receipt_to_html(receipt_data)
        output_path.write_text(html)
    else:
        output_path.write_text(json.dumps(receipt_data, indent=2, default=str))

    print(f"\nReceipt saved to: {output_path}")


def _open_receipt_in_browser(receipt_data: dict[str, Any]) -> str | None:
    """Generate HTML receipt and open in browser.

    Returns the path to the saved HTML file, or None on failure.
    """
    from aragora.cli.receipt_formatter import receipt_to_html

    try:
        html = receipt_to_html(receipt_data)
        # Create a persistent temp file (not auto-deleted)
        fd, path = tempfile.mkstemp(suffix=".html", prefix="aragora-receipt-")
        with os.fdopen(fd, "w") as f:
            f.write(html)
        webbrowser.open(f"file://{path}")
        return path
    except (OSError, RuntimeError, ValueError) as e:
        logger.debug("Failed to open receipt in browser: %s", e)
        return None


async def _run_demo_debate(question: str, rounds: int) -> dict[str, Any]:
    """Run a debate with mock agents (no API keys needed)."""
    from aragora_debate.arena import Arena
    from aragora_debate.styled_mock import StyledMockAgent
    from aragora_debate.types import Agent as DebateAgent, DebateConfig

    agents: list[DebateAgent] = [
        StyledMockAgent("analyst", style="supportive"),
        StyledMockAgent("critic", style="critical"),
        StyledMockAgent("synthesizer", style="balanced"),
    ]
    arena = Arena(question=question, agents=agents, config=DebateConfig(rounds=rounds))
    result = await arena.run()

    return {
        "question": question,
        "verdict": result.verdict.value if hasattr(result, "verdict") and hasattr(result.verdict, "value") else str(result.verdict) if hasattr(result, "verdict") else "consensus",
        "confidence": result.confidence if hasattr(result, "confidence") else 0.85,
        "rounds": rounds,
        "agents": [a.name for a in agents],
        "summary": result.receipt.to_markdown() if hasattr(result, "receipt") else str(result),
        "dissent": [],
        "mode": "demo",
    }


async def _run_live_debate(
    question: str,
    agents_list: list[tuple[str, str | None]],
    rounds: int,
) -> dict[str, Any]:
    """Run a debate with live API agents."""
    from aragora.agents.base import AgentType, create_agent
    from aragora.core import Environment
    from aragora.debate.orchestrator import Arena, DebateProtocol
    from aragora.memory.store import CritiqueStore

    env = Environment(task=question)
    protocol = DebateProtocol(rounds=rounds, consensus="majority")
    store = CritiqueStore()

    agents = []
    agent_names = []
    for provider, model in agents_list[:4]:  # Cap at 4 agents for quickstart
        agent = create_agent(cast(AgentType, provider), model=model)
        agents.append(agent)
        agent_names.append(provider)

    arena = Arena(env, agents, protocol, insight_store=store)
    result = await arena.run()

    verdict = "consensus"
    confidence = 0.0
    summary = ""
    dissent: list[dict[str, str]] = []

    if hasattr(result, "verdict"):
        verdict = result.verdict
    if hasattr(result, "confidence"):
        confidence = result.confidence
    if hasattr(result, "summary"):
        _summary_attr = result.summary
        summary = _summary_attr() if callable(_summary_attr) else _summary_attr
    elif hasattr(result, "final_summary"):
        summary = result.final_summary

    return {
        "question": question,
        "verdict": verdict,
        "confidence": confidence,
        "rounds": rounds,
        "agents": agent_names,
        "summary": summary,
        "dissent": dissent,
        "mode": "live",
    }


def cmd_quickstart(args: argparse.Namespace) -> None:
    """Handle the 'quickstart' command."""
    print("\n" + "=" * 60)
    print("  ARAGORA QUICKSTART")
    print("  Zero-to-receipt adversarial debate")
    print("=" * 60)

    # Step 1: Load .env
    loaded = _load_dotenv()
    if loaded:
        print("\n[+] Loaded .env configuration")

    # Step 2: Get question
    question = _get_question(args)
    if not question:
        print("\nNo question provided. Exiting.")
        sys.exit(1)

    print(f"\nQuestion: {question}")

    # Step 3: Detect agents
    use_demo = getattr(args, "demo", False)
    rounds = getattr(args, "rounds", 2)

    if use_demo:
        print("\n[*] Demo mode: using mock agents (no API keys needed)")
        print("    Agents: analyst (supportive), critic (critical), synthesizer (balanced)")
    else:
        detected = _detect_agents()
        if not detected:
            print("\n[!] No API keys found. Falling back to demo mode.")
            print("    Set ANTHROPIC_API_KEY or OPENAI_API_KEY for live debates.")
            print("    Agents: analyst (supportive), critic (critical), synthesizer (balanced)")
            use_demo = True
        else:
            providers = [p for p, _ in detected[:4]]
            print(f"\n[+] Detected agents: {', '.join(providers)}")

    print(f"[*] Running {rounds}-round debate...\n")

    # Step 4: Run debate
    start_time = time.monotonic()
    try:
        if use_demo:
            result = asyncio.run(_run_demo_debate(question, rounds))
        else:
            result = asyncio.run(_run_live_debate(question, detected[:4], rounds))
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
        logger.debug("Debate failed: %s", e)
        print(f"\n[!] Debate failed: {e}")
        print("    Try: aragora quickstart --demo")
        sys.exit(1)

    elapsed = time.monotonic() - start_time
    result["elapsed_seconds"] = elapsed

    # Step 5: Display results
    print("=" * 60)
    print("  RESULT")
    print("=" * 60)
    verdict_display = str(result['verdict']).replace('_', ' ').title()
    print(f"\n  Verdict:    {verdict_display}")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"  Agents:     {', '.join(result['agents'])}")
    print(f"  Rounds:     {result['rounds']}")
    print(f"  Elapsed:    {elapsed:.1f}s")

    if result.get("summary"):
        print(f"\n  Summary:\n  {result['summary'][:500]}")

    if result.get("dissent"):
        print("\n  Dissent:")
        for d in result["dissent"]:
            print(f"    - {d.get('agent', '?')}: {d.get('reason', 'N/A')}")

    print("\n" + "=" * 60)

    # Step 6: Save receipt
    output_path = getattr(args, "output", None)
    fmt = getattr(args, "format", "json")

    if output_path:
        _save_receipt(result, output_path, fmt)

    # Step 7: Open receipt in browser
    no_browser = getattr(args, "no_browser", False)
    if not no_browser:
        receipt_path = _open_receipt_in_browser(result)
        if receipt_path:
            print(f"\nReceipt opened in browser. Saved to {receipt_path}")
        else:
            print("\nCould not open browser. Save receipt with --output receipt.html")
    elif not output_path:
        print("\nTip: Save receipt with --output receipt.html")

    print("\nNext steps:")
    print("  aragora ask 'Your question' --agents anthropic-api,openai-api  # Full debate")
    print("  aragora decide 'Your question'                                  # Full pipeline")
    print("  aragora doctor                                                  # System health")
