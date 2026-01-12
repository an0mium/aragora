#!/usr/bin/env python3
"""Simple test debate without heavy imports."""

import os
import sys

# Set environment before importing heavy libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp/sentence_transformers'

import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Step 1: Importing core modules...", flush=True)
from aragora.core import Environment
print("Step 2: Environment imported", flush=True)

from aragora.debate.protocol import DebateProtocol
print("Step 3: DebateProtocol imported", flush=True)

from aragora.agents.registry import AgentRegistry
print("Step 4: AgentRegistry imported", flush=True)

from aragora.agents.api_agents.gemini import GeminiAgent
print("Step 5: GeminiAgent imported", flush=True)

from aragora.agents.api_agents.grok import GrokAgent
print("Step 6: GrokAgent imported", flush=True)

print("Step 7: About to import Arena...", flush=True)
from aragora.debate.orchestrator import Arena
print("Step 8: Arena imported", flush=True)


async def main():
    print("Step 9: Inside main()", flush=True)

    # Create environment
    env = Environment(
        task="What is 2+2? Explain your reasoning briefly.",
        context="This is a simple test debate.",
    )
    print("Step 10: Environment created", flush=True)

    # Create protocol
    protocol = DebateProtocol(
        rounds=1,
        consensus="majority",
        convergence_detection=False,
        early_stopping=False,
    )
    print("Step 11: Protocol created", flush=True)

    # Create agents
    print("Step 12: Creating Gemini agent...", flush=True)
    agent1 = AgentRegistry.create(
        "gemini",
        name="agent1",
        model="gemini-3-pro-preview",
        role="proposer",
        use_cache=False,
        timeout=60,
    )
    print("Step 13: Gemini agent created", flush=True)

    print("Step 14: Creating Grok agent...", flush=True)
    agent2 = AgentRegistry.create(
        "grok",
        name="agent2",
        model="grok-4",
        role="critic",
        use_cache=False,
        timeout=60,
    )
    print("Step 15: Grok agent created", flush=True)

    agents = [agent1, agent2]

    # Create arena
    print("Step 16: Creating Arena...", flush=True)
    arena = Arena(
        environment=env,
        agents=agents,
        protocol=protocol,
    )
    print("Step 17: Arena created", flush=True)

    # Run debate
    print("Step 18: Running debate...", flush=True)
    result = await arena.run()
    print("Step 19: Debate complete", flush=True)

    print(f"\nResult: {result.final_answer}", flush=True)
    return result


if __name__ == "__main__":
    print("Starting test...", flush=True)
    result = asyncio.run(main())
    print("Test completed.", flush=True)
