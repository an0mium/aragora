"""
Agent Selection Example

Demonstrates how to list available agents, view their capabilities,
check ELO rankings, and compare agent performance.

Usage:
    python examples/agent_selection.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import os

from aragora_sdk import AragoraClient


def main() -> None:
    client = AragoraClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    )

    # List all available agents
    print("=== Available Agents ===\n")
    agents = client.agents.list()

    for agent in agents.get("agents", []):
        name = agent.get("name", agent["id"])
        provider = agent.get("provider", "unknown")
        capabilities = ", ".join(agent.get("capabilities", ["general"]))
        elo = agent.get("elo_rating", "N/A")
        status = agent.get("status", "unknown")
        print(f"{name} ({agent['id']})")
        print(f"  Provider: {provider}")
        print(f"  Capabilities: {capabilities}")
        print(f"  ELO Rating: {elo}")
        print(f"  Status: {status}")
        print()

    # View the leaderboard
    print("=== Agent Leaderboard ===\n")
    leaderboard = client.agents.leaderboard()

    for i, entry in enumerate(leaderboard.get("rankings", [])[:10], 1):
        agent_id = entry.get("agent_id", "unknown")
        elo = entry.get("elo_rating", "N/A")
        wins = entry.get("wins", 0)
        losses = entry.get("losses", 0)
        print(f"  {i}. {agent_id} — ELO: {elo}, W/L: {wins}/{losses}")

    # Compare specific agents
    print("\n=== Agent Comparison ===\n")
    comparison = client.agents.compare(["claude", "gpt-4"])

    for agent_data in comparison.get("agents", []):
        agent_id = agent_data.get("agent_id", "unknown")
        print(f"{agent_id}:")
        stats = agent_data.get("stats", {})
        print(f"  Win rate: {stats.get('win_rate', 'N/A')}")
        print(f"  Avg confidence: {stats.get('avg_confidence', 'N/A')}")
        print(f"  Debates participated: {stats.get('debate_count', 'N/A')}")
        print()

    # Get detailed profile for a specific agent
    print("=== Agent Profile: claude ===\n")
    profile = client.agents.get_profile("claude")
    print(f"  Name: {profile.get('name', 'N/A')}")
    print(f"  Provider: {profile.get('provider', 'N/A')}")
    print(f"  ELO Rating: {profile.get('elo_rating', 'N/A')}")
    print(f"  Total debates: {profile.get('total_debates', 'N/A')}")
    print(f"  Specialties: {', '.join(profile.get('specialties', []))}")


if __name__ == "__main__":
    main()
