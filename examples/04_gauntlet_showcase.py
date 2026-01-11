#!/usr/bin/env python3
"""
Gauntlet Showcase - AI Decision Assurance in Action
====================================================

This example demonstrates Aragora's Gauntlet system - adversarial validation
that stress-tests decisions, architectures, and strategies through multi-agent
debate with regulatory and security personas.

It also showcases a real debate from 10 AI models discussing Aragora's
strategic positioning as an "AI Decision Assurance Engine."

Time: 1-2 minutes (viewing pre-computed results)
Requirements: None (uses stored debate results)

Usage:
    python examples/04_gauntlet_showcase.py

This showcase demonstrates:
    1. Decision Receipt generation - audit-ready output
    2. Regulatory personas (GDPR, HIPAA, AI Act)
    3. Security personas (Red Team, Threat Modeler)
    4. Viewing a real 10-agent strategic debate
"""

import json
import sys
from pathlib import Path

# Add aragora to path if running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))


def show_personas():
    """Show available Gauntlet personas."""
    print("\n" + "=" * 60)
    print("GAUNTLET: Adversarial Validation Personas")
    print("=" * 60)

    try:
        from aragora.gauntlet.personas import list_personas, get_persona

        print("\nAvailable Regulatory & Security Personas:\n")
        for persona_name in list_personas():
            persona = get_persona(persona_name)
            print(f"  {persona_name.upper()}")
            print(f"    {persona.description[:80]}...")
            print(f"    Attacks: {len(persona.attack_prompts)}")
            print()

    except ImportError as e:
        print(f"\n  (Personas module not available: {e})")


def show_decision_receipt_demo():
    """Demonstrate Decision Receipt generation."""
    print("\n" + "=" * 60)
    print("DECISION RECEIPT: Audit-Ready Output")
    print("=" * 60)

    from aragora.gauntlet.receipt import DecisionReceipt, ConsensusProof

    # Create a sample receipt
    receipt = DecisionReceipt(
        receipt_id="receipt-demo-001",
        gauntlet_id="gauntlet-demo-001",
        timestamp="2026-01-11T12:00:00Z",
        input_summary="Strategic proposal: Launch AI-first compliance platform...",
        input_hash="a1b2c3d4e5f6...",
        risk_summary={"critical": 0, "high": 2, "medium": 5, "low": 3},
        attacks_attempted=15,
        attacks_successful=2,
        probes_run=8,
        vulnerabilities_found=10,
        verdict="CONDITIONAL",
        confidence=0.78,
        robustness_score=0.87,
        verdict_reasoning="Strategy is sound but requires addressing identified competitive risks.",
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.78,
            supporting_agents=["anthropic", "openai", "gemini", "grok"],
            dissenting_agents=["mistral"],
            method="adversarial_validation",
        ),
    )

    print("\n  Receipt generated with:")
    print(f"    - Receipt ID: {receipt.receipt_id}")
    print(f"    - Verdict: {receipt.verdict}")
    print(f"    - Confidence: {receipt.confidence:.0%}")
    print(f"    - Robustness Score: {receipt.robustness_score:.0%}")
    print(f"    - Artifact Hash: {receipt.artifact_hash[:16]}...")

    print("\n  Export formats available:")
    print("    - receipt.to_json()     -> JSON for APIs/storage")
    print("    - receipt.to_markdown() -> Markdown for docs/PRs")
    print("    - receipt.to_html()     -> HTML for printing/sharing")
    print("    - receipt.to_pdf(path)  -> PDF for compliance (requires weasyprint)")

    # Show snippet of markdown
    print("\n  Sample Markdown Output:")
    print("  " + "-" * 40)
    md_lines = receipt.to_markdown().split("\n")[:12]
    for line in md_lines:
        print(f"  {line}")
    print("  ...")


def show_epic_debate():
    """Show the epic strategic debate results."""
    print("\n" + "=" * 60)
    print("EPIC DEBATE: 10 AI Models on Aragora Positioning")
    print("=" * 60)

    debate_path = Path(__file__).parent.parent / ".nomic" / "epic_strategic_debate"

    if not debate_path.exists():
        print("\n  (Demo debate files not found)")
        print("  Run a strategic debate with: aragora ask 'Position Aragora' --agents all")
        return

    # Load the debate result
    result_file = debate_path / "debate_result.json"
    if result_file.exists():
        with open(result_file) as f:
            debate = json.load(f)

        print(f"\n  Task: {debate.get('task', 'N/A')[:70]}...")
        print(f"  Agents: {len(debate.get('agents', []))} AI models")
        print(f"  Phases: {len(debate.get('phases', {}))}")

        # Show agents
        agents = debate.get("agents", [])
        if agents:
            print("\n  Participating Models:")
            for agent in agents[:6]:
                if isinstance(agent, dict):
                    name = agent.get("name", "unknown")
                    model = agent.get("model", "")
                    print(f"    - {name}: {model[:40]}")
                else:
                    print(f"    - {agent}")
            if len(agents) > 6:
                print(f"    ... and {len(agents) - 6} more")

        # Show final consensus if available
        consensus = debate.get("final_consensus", {})
        if consensus:
            print("\n  Final Consensus:")
            headline = consensus.get("headline", "N/A")
            confidence = consensus.get("confidence", 0)
            print(f"    Headline: {headline[:60]}...")
            print(f"    Confidence: {confidence:.0%}" if isinstance(confidence, float) else f"    Confidence: {confidence}")

        # Show key insights
        insights = debate.get("key_insights", [])
        if insights:
            print("\n  Key Insights:")
            for i, insight in enumerate(insights[:3], 1):
                if isinstance(insight, str):
                    print(f"    {i}. {insight[:70]}...")

    # Check for summary
    summary_file = debate_path / "debate_summary.md"
    if summary_file.exists():
        print(f"\n  Full Summary: {summary_file}")

    transcript_file = debate_path / "debate_transcript.txt"
    if transcript_file.exists():
        print(f"  Full Transcript: {transcript_file}")

    audio_file = debate_path / "debate_full.mp3"
    if audio_file.exists():
        size_mb = audio_file.stat().st_size / (1024 * 1024)
        print(f"  Audio Recording: {audio_file} ({size_mb:.1f} MB)")


def show_cli_usage():
    """Show Gauntlet CLI usage."""
    print("\n" + "=" * 60)
    print("GAUNTLET CLI: Command-Line Usage")
    print("=" * 60)

    print("""
  Run adversarial validation from the command line:

  # Quick stress-test of a spec
  aragora gauntlet spec.md --profile quick

  # Thorough architecture review
  aragora gauntlet architecture.md -t architecture --profile thorough

  # Code review with formal verification
  aragora gauntlet main.py -t code --profile code --verify

  # Policy review with audit trail
  aragora gauntlet policy.yaml -t policy -o receipt.html

  Profiles:
    quick    - Fast validation (2-3 minutes)
    thorough - Comprehensive analysis (10-15 minutes)
    code     - Code-specific checks (security, bugs, style)
    policy   - Policy-specific compliance checks
""")


def main():
    """Run the Gauntlet showcase."""
    print("\n" + "#" * 60)
    print("# ARAGORA GAUNTLET SHOWCASE")
    print("# AI Decision Assurance Engine")
    print("#" * 60)

    show_personas()
    show_decision_receipt_demo()
    show_epic_debate()
    show_cli_usage()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
  1. Run a gauntlet on your own spec:
     aragora gauntlet my-spec.md

  2. Generate a Decision Receipt:
     aragora gauntlet arch.md -o decision-receipt.html

  3. Explore regulatory personas:
     python -c "from aragora.gauntlet.personas import *; help(GDPRPersona)"

  4. Integrate in CI/CD:
     aragora gauntlet spec.md && echo "PASSED" || echo "FAILED"
""")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
