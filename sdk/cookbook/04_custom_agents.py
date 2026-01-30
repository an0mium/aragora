#!/usr/bin/env python3
"""
04_custom_agents.py - Define custom agent personas and behavior.

This example shows how to create specialized agents with:
- Custom personas and expertise areas
- Behavioral configuration
- Vote weighting based on expertise

Usage:
    python 04_custom_agents.py --dry-run
    python 04_custom_agents.py --topic "How should we architect our microservices?"
"""

import argparse
import asyncio
from aragora_sdk import ArenaClient, DebateConfig, Agent, AgentPersona


async def run_custom_agents_debate(topic: str, dry_run: bool = False) -> dict:
    """Run a debate with custom-configured agents."""

    # Define specialized agent personas
    security_expert = Agent(
        name="security_analyst",
        model="claude-sonnet-4-20250514",
        persona=AgentPersona(
            role="Security Expert",
            expertise=["cybersecurity", "threat modeling", "compliance"],
            tone="cautious and thorough",
            priorities=["security", "compliance", "risk mitigation"],
        ),
        # Higher weight for security-related topics
        vote_weight=1.5,
    )

    performance_expert = Agent(
        name="performance_engineer",
        model="gpt-4o",
        persona=AgentPersona(
            role="Performance Engineer",
            expertise=["scalability", "optimization", "distributed systems"],
            tone="data-driven and pragmatic",
            priorities=["performance", "efficiency", "scalability"],
        ),
        vote_weight=1.0,
    )

    ux_advocate = Agent(
        name="ux_advocate",
        model="gemini-2.0-flash",
        persona=AgentPersona(
            role="User Experience Advocate",
            expertise=["user research", "accessibility", "usability"],
            tone="empathetic and user-focused",
            priorities=["user experience", "accessibility", "simplicity"],
        ),
        vote_weight=1.0,
    )

    cost_analyst = Agent(
        name="cost_analyst",
        model="mistral-large-latest",
        persona=AgentPersona(
            role="Cost Analyst",
            expertise=["budgeting", "ROI analysis", "resource optimization"],
            tone="analytical and budget-conscious",
            priorities=["cost efficiency", "ROI", "sustainability"],
        ),
        vote_weight=0.8,
    )

    agents = [security_expert, performance_expert, ux_advocate, cost_analyst]

    if dry_run:
        print(f"[DRY RUN] Topic: {topic}")
        print("[DRY RUN] Custom agents configured:")
        for agent in agents:
            print(f"  - {agent.name}: {agent.persona.role} (weight: {agent.vote_weight})")
        return {"status": "dry_run", "agents": [a.name for a in agents]}

    # Configure the debate
    client = ArenaClient()
    config = DebateConfig(
        topic=topic,
        agents=agents,
        rounds=3,
        consensus_threshold=0.65,
        # Enable persona-aware prompting
        use_personas=True,
    )

    # Run the debate
    result = await client.run_debate(config)

    print("\n=== Results ===")
    print(f"Decision: {result.decision}")
    print("\nAgent contributions:")
    for contribution in result.contributions:
        print(f"  {contribution.agent}: {contribution.key_points[0][:60]}...")

    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Run a debate with custom agents")
    parser.add_argument(
        "--topic", default="How should we architect our microservices?", help="Topic for the debate"
    )
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    args = parser.parse_args()

    result = asyncio.run(run_custom_agents_debate(args.topic, args.dry_run))
    return result


if __name__ == "__main__":
    main()
