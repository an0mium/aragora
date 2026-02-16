"""
SME Starter Pack: guided onboarding from install to decision receipt.

Chains the golden path into one guided flow:
  1. Initialize workspace (scaffolding + config)
  2. Run an offline demo debate (no API keys required)
  3. Save and open a decision receipt
  4. Print clear next steps for going live

Usage:
    aragora starter                          # Full guided flow
    aragora starter --question "My topic"    # Custom question
    aragora starter --no-browser             # Skip opening browser
    aragora starter --output receipt.html    # Custom receipt path
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import webbrowser
from pathlib import Path
from typing import Any


def add_starter_parser(subparsers: Any) -> None:
    """Register the 'starter' subcommand."""
    starter_parser = subparsers.add_parser(
        "starter",
        help="SME Starter Pack -- install to decision receipt in 15 minutes",
        description="""
SME Starter Pack: guided onboarding for new users.

Walks you through the complete golden path in one command:
  1. Initializes an Aragora workspace in the current directory
  2. Runs an offline adversarial debate (no API keys needed)
  3. Generates a decision receipt and opens it in your browser
  4. Shows you exactly what to do next

No API keys, no server, no configuration required.

Examples:
  aragora starter                                          # Full guided flow
  aragora starter --question "Should we use Kubernetes?"   # Custom topic
  aragora starter --output decision.html                   # Save receipt to file
  aragora starter --no-browser                             # CI/headless mode
  aragora starter --skip-init                              # Skip workspace setup
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    starter_parser.add_argument(
        "--question",
        "-q",
        help="Custom question to debate (default: architecture decision)",
    )
    starter_parser.add_argument(
        "--output",
        "-o",
        help="Save receipt to this file path (.json, .html, .md)",
    )
    starter_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open receipt in browser (for CI/headless environments)",
    )
    starter_parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip workspace initialization (if already set up)",
    )
    starter_parser.add_argument(
        "--demo-name",
        choices=["microservices", "rate-limiter", "auth", "cache", "kubernetes"],
        help="Named demo scenario (alternative to --question)",
    )
    starter_parser.set_defaults(func=cmd_starter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEP_WIDTH = 64


def _banner() -> None:
    """Print the starter pack welcome banner."""
    print()
    print("=" * _STEP_WIDTH)
    print("  ARAGORA SME STARTER PACK")
    print("  From install to decision receipt in 15 minutes")
    print("=" * _STEP_WIDTH)
    print()
    print("  Aragora orchestrates multiple AI agents to adversarially")
    print("  vet decisions, producing audit-ready decision receipts.")
    print()
    print("  This guided flow will:")
    print("    Step 1.  Set up your workspace")
    print("    Step 2.  Run an adversarial debate (offline, no API keys)")
    print("    Step 3.  Generate your first decision receipt")
    print("    Step 4.  Show you how to go live with real AI agents")
    print()
    print("-" * _STEP_WIDTH)


def _step_header(step: int, total: int, title: str) -> None:
    """Print a numbered step header."""
    print()
    print(f"  STEP {step}/{total}: {title}")
    print("  " + "-" * (_STEP_WIDTH - 4))


def _detect_api_keys() -> list[str]:
    """Detect which API keys are available."""
    key_map = {
        "ANTHROPIC_API_KEY": "Anthropic (Claude)",
        "OPENAI_API_KEY": "OpenAI (GPT)",
        "GEMINI_API_KEY": "Google (Gemini)",
        "MISTRAL_API_KEY": "Mistral",
        "XAI_API_KEY": "xAI (Grok)",
        "OPENROUTER_API_KEY": "OpenRouter (fallback)",
    }
    found = []
    for env_var, name in key_map.items():
        if os.environ.get(env_var):
            found.append(name)
    return found


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


def cmd_starter(args: argparse.Namespace) -> None:
    """Handle the 'starter' command -- SME Starter Pack guided flow."""
    _banner()

    skip_init = getattr(args, "skip_init", False)
    no_browser = getattr(args, "no_browser", False)
    output_path = getattr(args, "output", None)
    custom_question = getattr(args, "question", None)
    demo_name = getattr(args, "demo_name", None)

    total_steps = 4

    # ------------------------------------------------------------------
    # Step 1: Initialize workspace
    # ------------------------------------------------------------------
    _step_header(1, total_steps, "WORKSPACE SETUP")

    if skip_init:
        print("    Skipped (--skip-init)")
    else:
        from aragora.cli.init import init_project

        cwd = Path.cwd()
        aragora_dir = cwd / ".aragora"
        config_file = cwd / ".aragora.yaml"

        if aragora_dir.exists() and config_file.exists():
            print("    Workspace already initialized.")
            print(f"    Config: {config_file}")
            print(f"    Data:   {aragora_dir}")
        else:
            result = init_project(force=False, with_git=True)
            if result["files"]:
                for f in result["files"]:
                    print(f"    Created: {f}")
            if result["directories"]:
                for d in result["directories"]:
                    print(f"    Created: {d}")
            print("    Workspace ready.")

    # Detect API keys
    detected_keys = _detect_api_keys()
    if detected_keys:
        print(f"\n    Detected API keys: {', '.join(detected_keys)}")
    else:
        print("\n    No API keys detected (that's fine -- demo mode needs none).")

    # ------------------------------------------------------------------
    # Step 2: Run demo debate
    # ------------------------------------------------------------------
    _step_header(2, total_steps, "ADVERSARIAL DEBATE")

    # Resolve the debate topic
    from aragora.cli.demo import DEMO_TASKS, _AGENT_CONFIGS

    if custom_question:
        topic = custom_question
        print(f"    Topic: {topic}")
    elif demo_name and demo_name in DEMO_TASKS:
        topic = DEMO_TASKS[demo_name]["topic"]
        print(f"    Scenario: {demo_name}")
        print(f"    Topic: {topic}")
    else:
        topic = DEMO_TASKS["microservices"]["topic"]
        print("    Scenario: microservices (default)")
        print(f"    Topic: {topic}")

    agent_names = [name for name, _ in _AGENT_CONFIGS]
    print(f"    Agents: {', '.join(agent_names)}")
    print("    Rounds: 2")
    print("    Mode: Offline (mock agents, no API keys)")
    print()

    # Run the debate
    from aragora.cli.demo import _run_demo_debate

    try:
        result, elapsed = asyncio.run(_run_demo_debate(topic))
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
        print(f"\n    Debate failed: {e}")
        print("    Try: aragora demo --list")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Generate receipt
    # ------------------------------------------------------------------
    _step_header(3, total_steps, "DECISION RECEIPT")

    from aragora.cli.demo import _build_receipt_data
    from aragora.cli.receipt_formatter import receipt_to_html

    receipt_data = _build_receipt_data(result, elapsed)

    # Determine receipt output path
    receipt_file = output_path
    if not receipt_file:
        # Default to .aragora/receipts/starter-receipt.html
        receipts_dir = Path.cwd() / ".aragora" / "receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)
        receipt_file = str(receipts_dir / "starter-receipt.html")

    # Save receipt
    receipt_path_obj = Path(receipt_file)
    if receipt_path_obj.suffix.lower() in (".html", ".htm"):
        html = receipt_to_html(receipt_data)
        receipt_path_obj.parent.mkdir(parents=True, exist_ok=True)
        receipt_path_obj.write_text(html)
    elif receipt_path_obj.suffix.lower() == ".md":
        from aragora.cli.receipt_formatter import receipt_to_markdown

        receipt_path_obj.parent.mkdir(parents=True, exist_ok=True)
        receipt_path_obj.write_text(receipt_to_markdown(receipt_data))
    else:
        receipt_path_obj.parent.mkdir(parents=True, exist_ok=True)
        receipt_path_obj.write_text(json.dumps(receipt_data, indent=2, default=str))

    print(f"    Receipt saved: {receipt_file}")

    # Open in browser
    if not no_browser:
        try:
            abs_path = str(receipt_path_obj.resolve())
            webbrowser.open(f"file://{abs_path}")
            print("    Receipt opened in browser.")
        except (OSError, webbrowser.Error):
            print("    Could not open browser. View the receipt file manually.")

    # Print receipt summary in terminal
    verdict = receipt_data.get("verdict", "N/A")
    confidence = receipt_data.get("confidence", 0)
    receipt_id = receipt_data.get("receipt_id", "")
    print()
    print(f"    Verdict:    {verdict}")
    print(f"    Confidence: {confidence:.0%}")
    print(f"    Agents:     {', '.join(receipt_data.get('agents', []))}")
    print(f"    Rounds:     {receipt_data.get('rounds', 0)}")
    print(f"    Duration:   {elapsed:.2f}s")
    if receipt_id:
        print(f"    Receipt ID: {receipt_id}")

    # ------------------------------------------------------------------
    # Step 4: Next steps
    # ------------------------------------------------------------------
    _step_header(4, total_steps, "WHAT TO DO NEXT")

    print()
    print("    You just ran your first adversarial decision stress-test!")
    print("    The receipt above is what Aragora produces for every decision.")
    print()

    if detected_keys:
        print("    READY TO GO LIVE (API keys detected):")
        print("    " + "-" * 44)
        print()
        print("    Run a real debate with AI agents:")
        print('      aragora decide "Should we migrate to Kubernetes?"')
        print()
        print("    Or use the full ask command for more control:")
        print('      aragora ask "Your question" --agents anthropic-api,openai-api')
        print()
    else:
        print("    TO GO LIVE, add at least one API key:")
        print("    " + "-" * 44)
        print()
        print("    Option A: Set environment variables")
        print("      export ANTHROPIC_API_KEY=sk-ant-...")
        print("      export OPENAI_API_KEY=sk-...")
        print()
        print("    Option B: Run the setup wizard")
        print("      aragora setup")
        print()
        print("    Then run a real debate:")
        print('      aragora decide "Should we migrate to Kubernetes?"')
        print()

    print("    MORE COMMANDS TO EXPLORE:")
    print("    " + "-" * 44)
    print()
    print("    aragora demo --list              # See all demo scenarios")
    print("    aragora demo --topic 'My topic'  # Demo with custom topic")
    print("    aragora doctor                   # Check system health")
    print("    aragora quickstart               # Another guided onboarding")
    print("    aragora review --demo            # AI code review demo")
    print("    aragora receipt view <file>       # Open any receipt in browser")
    print("    aragora serve                    # Start the API server")
    print()
    print("=" * _STEP_WIDTH)
    print("  Starter Pack complete. Your receipt is at:")
    print(f"  {receipt_file}")
    print("=" * _STEP_WIDTH)
    print()
