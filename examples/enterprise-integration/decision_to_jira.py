"""Example: Debate → DecisionPlan → Jira Issue Creation.

Demonstrates the full pipeline from debate to automatic Jira issue creation
via the DecisionBridge integration.

Prerequisites:
    pip install aragora
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    # Jira credentials (for actual issue creation):
    export JIRA_URL=https://your-org.atlassian.net
    export JIRA_EMAIL=you@company.com
    export JIRA_API_TOKEN=...

Usage:
    python examples/enterprise-integration/decision_to_jira.py
"""

from __future__ import annotations

import asyncio

from aragora import Arena, Environment
from aragora.agents.base import create_agent
from aragora.debate.protocol import DebateProtocol
from aragora.integrations.decision_bridge import DecisionBridge
from aragora.pipeline.decision_plan.factory import DecisionPlanFactory


async def main() -> None:
    # Create agents for the debate
    agents = [
        create_agent("anthropic-api", name="architect", role="proposer"),
        create_agent("openai-api", name="reviewer", role="critic"),
        create_agent("anthropic-api", name="lead", role="synthesizer"),
    ]

    # Run a debate about a technical decision
    env = Environment(
        task="Design a caching strategy for our API. Consider Redis vs in-memory vs CDN approaches.",
        max_rounds=3,
    )

    protocol = DebateProtocol(rounds=3, consensus="majority")
    arena = Arena(env, agents, protocol)
    result = await arena.run()

    print(f"Debate completed: consensus={result.consensus_reached}")
    print(f"Decision: {result.final_answer[:200]}...")

    # Convert debate result to a DecisionPlan
    plan = DecisionPlanFactory.from_debate_result(result)
    print(f"\nDecision Plan: {plan.title}")
    print(f"  Tasks: {len(plan.implement_plan.tasks)}")
    print(f"  Risk level: {plan.risk_register.overall_risk.value}")

    # Route to Jira (set integrations in metadata)
    plan.metadata["integrations"] = ["jira"]

    bridge = DecisionBridge()
    bridge_result = await bridge.handle_decision_plan(plan)

    print(f"\nJira issues created: {len(bridge_result.jira_issues)}")
    for issue in bridge_result.jira_issues:
        print(f"  - {issue.get('key', 'N/A')}: {issue.get('summary', '')}")

    if bridge_result.errors:
        print(f"\nErrors: {bridge_result.errors}")


if __name__ == "__main__":
    asyncio.run(main())
