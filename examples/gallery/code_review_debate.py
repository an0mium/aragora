#!/usr/bin/env python3
"""
Code Review Debate Example
==========================

Multi-agent code review where AI agents analyze code for security,
performance, and best practices from different perspectives.

Use case: Get comprehensive code review feedback from multiple AI perspectives.

Time: ~3-5 minutes
Requirements: At least one API key

Usage:
    python examples/gallery/code_review_debate.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from aragora import Arena, Environment, DebateProtocol
from aragora.agents.base import create_agent
from aragora.templates import CODE_REVIEW_TEMPLATE


# Code sample to review
CODE_SAMPLE = '''
def process_user_data(user_input: str, db_connection) -> dict:
    """Process user input and store in database."""
    # Parse the input
    data = eval(user_input)  # Quick parsing

    # Build query
    query = f"INSERT INTO users (name, email) VALUES ('{data['name']}', '{data['email']}')"

    # Execute
    db_connection.execute(query)

    # Return result
    return {"status": "ok", "data": data, "password": data.get("password")}
'''


async def run_code_review_debate():
    """Run a multi-agent code review debate."""

    print("\n" + "=" * 60)
    print("ARAGORA: Code Review Debate")
    print("=" * 60)

    # Create specialized agents
    agent_configs = [
        ("anthropic-api", "security_reviewer"),
        ("openai-api", "performance_reviewer"),
        ("gemini", "maintainability_reviewer"),
    ]

    agents = []
    for agent_type, role in agent_configs:
        try:
            agent = create_agent(model_type=agent_type, name=f"{role}", role=role)  # type: ignore
            agents.append(agent)
            print(f"  + {agent.name} ready")
        except Exception as e:
            print(f"  - {agent_type} unavailable: {str(e)[:40]}")

    if len(agents) < 2:
        print("\nError: Need at least 2 agents. Check API keys.")
        return None

    # Use code review template
    env = Environment(
        task=f"""Review the following Python code for security vulnerabilities,
performance issues, and best practice violations.

```python
{CODE_SAMPLE}
```

Provide:
1. Critical issues that must be fixed
2. Severity rating (Critical/High/Medium/Low)
3. Specific fix recommendations
4. Security implications""",
        context="Production code review for a web application",
    )

    protocol = DebateProtocol(
        rounds=2,
        consensus="unanimous",
        enable_calibration=True,
    )

    print(f"\nReviewing code with {len(agents)} agents...")

    arena = Arena(env, agents, protocol)
    result = await arena.run()

    print(f"\n{'='*60}")
    print("CODE REVIEW RESULTS")
    print(f"{'='*60}")
    print(f"Consensus: {'Yes' if result.consensus_reached else 'No'}")
    print(f"Confidence: {result.confidence:.0%}")

    print(f"\n--- Review Summary ---")
    answer = result.final_answer
    print(answer[:1500] if len(answer) > 1500 else answer)

    return result


if __name__ == "__main__":
    result = asyncio.run(run_code_review_debate())
    if result and result.consensus_reached:
        print("\n[SUCCESS] Code review completed with consensus!")
