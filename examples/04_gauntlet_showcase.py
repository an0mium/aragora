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

    try:
        from aragora.gauntlet.personas import list_personas, get_persona

        for persona_name in list_personas():
            get_persona(persona_name)

    except ImportError:
        pass


def show_decision_receipt_demo():
    """Demonstrate Decision Receipt generation."""

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

    # Show snippet of markdown
    md_lines = receipt.to_markdown().split("\n")[:12]
    for line in md_lines:
        pass


def show_epic_debate():
    """Show the epic strategic debate results."""

    debate_path = Path(__file__).parent.parent / ".nomic" / "epic_strategic_debate"

    if not debate_path.exists():
        return

    # Load the debate result
    result_file = debate_path / "debate_result.json"
    if result_file.exists():
        with open(result_file) as f:
            debate = json.load(f)

        # Show agents
        agents = debate.get("agents", [])
        if agents:
            for agent in agents[:6]:
                if isinstance(agent, dict):
                    agent.get("name", "unknown")
                    agent.get("model", "")
                else:
                    pass
            if len(agents) > 6:
                pass

        # Show final consensus if available
        consensus = debate.get("final_consensus", {})
        if consensus:
            consensus.get("headline", "N/A")
            consensus.get("confidence", 0)

        # Show key insights
        insights = debate.get("key_insights", [])
        if insights:
            for i, insight in enumerate(insights[:3], 1):
                if isinstance(insight, str):
                    pass

    # Check for summary
    summary_file = debate_path / "debate_summary.md"
    if summary_file.exists():
        pass

    transcript_file = debate_path / "debate_transcript.txt"
    if transcript_file.exists():
        pass

    audio_file = debate_path / "debate_full.mp3"
    if audio_file.exists():
        audio_file.stat().st_size / (1024 * 1024)


def show_cli_usage():
    """Show Gauntlet CLI usage."""


def main():
    """Run the Gauntlet showcase."""

    show_personas()
    show_decision_receipt_demo()
    show_epic_debate()
    show_cli_usage()


if __name__ == "__main__":
    main()
