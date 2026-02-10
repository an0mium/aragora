"""
CrewAI + Aragora Verification Example

Demonstrates using Aragora as a verification layer for CrewAI agent outputs.
After your crew completes its task, Aragora runs adversarial stress-testing
and produces a cryptographic decision receipt.

Requirements:
    pip install aragora-sdk crewai

Usage:
    export ARAGORA_API_KEY=your-key
    export OPENAI_API_KEY=your-key   # For CrewAI agents
    python main.py
"""

from __future__ import annotations

import os
import sys

from aragora_sdk import AragoraClient


def verify_crew_output(
    output: str,
    *,
    context: str = "",
    attack_rounds: int = 3,
    export_sarif: bool = False,
) -> dict:
    """
    Verify CrewAI output through Aragora's adversarial testing.

    Args:
        output: The crew's output to verify.
        context: Additional context about the task.
        attack_rounds: Number of adversarial attack/defend cycles.
        export_sarif: Whether to export findings as SARIF.

    Returns:
        Dict with verdict, findings, and receipt hash.
    """
    client = AragoraClient(
        base_url=os.getenv("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.getenv("ARAGORA_API_KEY"),
    )

    # Run adversarial validation
    task_prompt = f"Verify this agent analysis for accuracy, bias, and completeness:\n\n{output}"
    if context:
        task_prompt += f"\n\nOriginal task context: {context}"

    result = client.gauntlet.run(task=task_prompt, attack_rounds=attack_rounds)

    # Get the cryptographic receipt
    receipt = client.gauntlet.get_receipt(result["gauntlet_id"])

    # Optionally export as SARIF for CI/CD
    if export_sarif:
        sarif = client.gauntlet.export_receipt(result["gauntlet_id"], format="sarif")
        with open("crew-verification.sarif", "w") as f:
            f.write(sarif)
        print(f"SARIF exported to crew-verification.sarif")

    return {
        "verdict": result["verdict"],
        "gauntlet_id": result["gauntlet_id"],
        "receipt_hash": receipt["hash"],
        "findings": client.gauntlet.get_findings(result["gauntlet_id"]),
    }


def main():
    """Run a CrewAI workflow and verify the output."""
    try:
        from crewai import Agent, Crew, Task
    except ImportError:
        print("crewai not installed. Showing verification with mock output.")
        print()

        # Simulate crew output for demonstration
        mock_output = (
            "Based on our analysis, we recommend migrating to a microservices "
            "architecture. Key benefits include independent scaling, technology "
            "diversity, and fault isolation. Estimated migration timeline: 6 months. "
            "Risk assessment: Medium. Main risks are data consistency across services "
            "and increased operational complexity."
        )

        print("Crew output (simulated):")
        print(f"  {mock_output[:100]}...")
        print()
        print("Verifying with Aragora...")

        result = verify_crew_output(
            output=mock_output,
            context="Architecture decision for a 50-person SaaS startup",
        )

        print(f"  Verdict: {result['verdict']}")
        print(f"  Receipt: {result['receipt_hash']}")
        if result["findings"]:
            print(f"  Findings ({len(result['findings'])}):")
            for f in result["findings"][:3]:
                print(f"    - [{f.get('severity', 'INFO')}] {f.get('title', 'N/A')}")
        return

    # Real CrewAI workflow
    researcher = Agent(
        role="Technical Researcher",
        goal="Research architecture patterns and provide data-driven recommendations",
        backstory="Senior software architect with 15 years of experience",
    )

    analyst = Agent(
        role="Risk Analyst",
        goal="Identify risks, costs, and potential failure modes",
        backstory="Enterprise risk management specialist",
    )

    task = Task(
        description=(
            "Analyze whether our 50-person SaaS startup should migrate from a "
            "monolithic Django application to microservices. Consider: team size, "
            "current pain points (slow deployments, coupled releases), growth "
            "trajectory (3x in 18 months), and budget constraints."
        ),
        expected_output="Architecture recommendation with risk assessment",
        agent=researcher,
    )

    crew = Crew(agents=[researcher, analyst], tasks=[task], verbose=True)
    result = crew.kickoff()

    # Verify the crew's output
    print("\n--- Aragora Verification ---")
    verification = verify_crew_output(
        output=str(result),
        context="Architecture decision for SaaS startup",
        export_sarif=True,
    )

    print(f"Verdict: {verification['verdict']}")
    print(f"Receipt: {verification['receipt_hash']}")

    if verification["verdict"] != "PASS":
        print("Issues found:")
        for finding in verification["findings"]:
            print(f"  [{finding.get('severity')}] {finding.get('title')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
