#!/usr/bin/env python3
"""
Meta-debate: How should aragora's phase_implement work?

Uses aragora to debate how aragora should implement code.
The ultimate self-referential improvement.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.core import Environment
from aragora.agents.cli_agents import ClaudeAgent, CodexAgent, GeminiCLIAgent
import os


async def main():
    # Set Gemini API key if available
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    print("\n" + "=" * 70)
    print("META-DEBATE: How should phase_implement work?")
    print("=" * 70)
    print("\nThis is aragora debating how to improve itself.")
    print("Claude Code, Codex, and Gemini will propose different approaches.\n")

    env = Environment(
        task="""How should aragora's nomic loop phase_implement work?

CURRENT STATE:
- phase_implement uses `codex exec` to write code based on a design
- Codex often times out (300s limit) on complex designs
- Codex gets cautious when detecting other agents modifying the same codebase
- Claude Code is faster and more responsive but isn't currently used

OBSERVATIONS:
- Claude Code completed the recent ~430 line implementation in one session
- Codex waits for confirmation when it detects concurrent modifications
- Different CLI tools have different strengths

OPTIONS TO DEBATE:
1. Replace Codex with Claude Code entirely
2. Use both in parallel (concurrent implementation, then merge)
3. Use Claude Code as coordinator that delegates to Codex for specific tasks
4. Sequential: Claude Code does scaffolding, Codex fills in complex logic
5. Keep Codex but with better prompting/chunking

Each agent should propose their preferred approach with specific implementation details.
Consider: reliability, speed, code quality, collaboration mechanics, error handling.""",
        context="""aragora is an agorist AI framework for multi-agent debate.
The nomic loop allows aragora to improve itself through:
1. Debate phase: Agents propose improvements
2. Design phase: Agents design the implementation
3. Implement phase: Currently uses Codex to write code
4. Verify phase: Tests and checks
5. Commit phase: Git commit if verified

The goal is to make phase_implement more reliable and effective.""",
    )

    # All three as competing visionaries
    claude_agent = ClaudeAgent(
        name="claude-code-advocate",
        model="claude-sonnet-4-20250514",
        role="proposer",
        timeout=600,  # Doubled from 300
    )
    claude_agent.system_prompt = """You are advocating for Claude Code's approach.
You have direct experience implementing aragora features quickly and reliably.
Argue for approaches that leverage Claude Code's strengths:
- Fast iteration and responsiveness
- Good at understanding full codebase context
- Can coordinate multi-file changes
Be specific about implementation details."""

    codex_agent = CodexAgent(
        name="codex-advocate",
        model="o3",
        role="proposer",
        timeout=600,  # Doubled from 300
    )
    codex_agent.system_prompt = """You are advocating for Codex/GPT's approach.
You have deep code understanding and can reason about complex logic.
Argue for approaches that leverage your strengths:
- Strong at complex algorithmic reasoning
- Good at code generation from specifications
- Can work on isolated, well-defined tasks
Be honest about your limitations (timeouts, caution with concurrent changes)
and propose solutions to address them."""

    # Use Gemini as synthesizer if available, otherwise Claude
    if gemini_key:
        synthesizer = GeminiCLIAgent(
            name="gemini-synthesizer",
            model="gemini-3-pro",
            role="synthesizer",
            timeout=600,  # Doubled from 300
        )
        synthesizer.system_prompt = """You are a neutral synthesizer from Google's perspective.
Your role is to find the best approach that combines the strengths of both Claude Code and Codex.
Consider:
- Reliability and error handling
- Speed and developer experience
- Code quality and consistency
- Practical implementation complexity
Synthesize a concrete recommendation."""
        agents = [claude_agent, codex_agent, synthesizer]
        print("Agents: Claude Code Advocate + Codex Advocate + Gemini Synthesizer\n")
    else:
        synthesizer = ClaudeAgent(
            name="neutral-synthesizer",
            model="claude-sonnet-4-20250514",
            role="synthesizer",
            timeout=600,  # Doubled from 300
        )
        synthesizer.system_prompt = """You are a neutral synthesizer.
Listen to both Claude Code and Codex advocates, then synthesize the best approach.
Consider practical implementation and developer experience.
Your recommendation should be specific and actionable."""
        agents = [claude_agent, codex_agent, synthesizer]
        print("Agents: Claude Code Advocate + Codex Advocate + Claude Synthesizer")
        print("(Set GEMINI_API_KEY for true 3-provider debate)\n")

    protocol = DebateProtocol(rounds=2, consensus="judge")
    arena = Arena(env, agents, protocol)
    result = await arena.run()

    print("\n" + "=" * 70)
    print("META-DEBATE RESULT")
    print("=" * 70)
    print(f"\nConsensus: {'Yes' if result.consensus_reached else 'No'} ({result.confidence:.0%})")
    print(f"\n{result.final_answer}")

    # Save result
    import json

    output = {
        "task": "phase_implement improvement",
        "consensus_reached": result.consensus_reached,
        "confidence": result.confidence,
        "final_answer": result.final_answer,
        "duration_seconds": result.duration_seconds,
    }

    output_path = Path(__file__).parent.parent / ".nomic" / "phase_implement_debate.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
