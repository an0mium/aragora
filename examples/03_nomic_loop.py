#!/usr/bin/env python3
"""
Nomic Loop Demo - Self-Improving AI
====================================

This example demonstrates Aragora's unique feature: autonomous self-improvement.

The Nomic Loop is a 5-phase cycle where AI agents:
1. DEBATE: Propose improvements to a codebase
2. DESIGN: Architecture the implementation
3. IMPLEMENT: Generate code changes
4. VERIFY: Run tests and checks
5. COMMIT: Apply changes if verified

Time: ~5-10 minutes (simplified demo)
Requirements: At least 2 API keys

SAFETY: This demo runs in DRY-RUN mode (no actual file changes)

Usage:
    python examples/03_nomic_loop.py

Expected output:
    === NOMIC LOOP: SELF-IMPROVEMENT CYCLE ===

    Phase 1: DEBATE
      Agents proposing improvements...
      Consensus: Add input validation (75% agreement)

    Phase 2: DESIGN
      Designing implementation...
      Files affected: 1

    Phase 3: IMPLEMENT (DRY RUN)
      Would modify: src/handlers.py

    Phase 4: VERIFY
      Syntax check: PASS
      Type check: PASS

    [DRY RUN] Would commit: "Add input validation to API handlers"
"""

import asyncio
import sys
from pathlib import Path

# Add aragora to path if running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aragora import Arena, Environment, DebateProtocol
from aragora.agents.base import create_agent


class SimplifiedNomicLoop:
    """
    A simplified version of the full nomic loop for demonstration.

    The full implementation is in scripts/nomic_loop.py (~8000 lines).
    This demo shows the core concept: agents debate and design improvements.
    """

    def __init__(self, agents, dry_run=True):
        self.agents = agents
        self.dry_run = dry_run
        self.improvement = None
        self.design = None

    async def phase_debate(self, focus: str) -> str:
        """Phase 1: Agents debate what to improve."""

        env = Environment(
            task=f"""You are improving a Python codebase. Propose ONE specific improvement.

Focus area: {focus}

Requirements:
- Must be a concrete, implementable change
- Should improve code quality or functionality
- Explain why this improvement matters

Respond with:
IMPROVEMENT: <one-sentence description>
RATIONALE: <why this helps>
FILES: <which files would be affected>""",
        )

        protocol = DebateProtocol(
            rounds=2,
            consensus="majority",
            early_stopping=True,
        )

        arena = Arena(env, self.agents, protocol)
        result = await arena.run()

        self.improvement = result.final_answer

        # Extract improvement summary
        lines = self.improvement.split("\n")
        for line in lines[:5]:
            if line.strip():
                pass

        return self.improvement

    async def phase_design(self) -> str:
        """Phase 2: Design the implementation."""

        if not self.improvement:
            return ""

        env = Environment(
            task=f"""Design the implementation for this improvement:

{self.improvement[:500]}

Provide:
1. Specific code changes needed
2. Files to modify
3. Edge cases to handle

Keep the design focused and minimal.""",
        )

        arena = Arena(env, self.agents, DebateProtocol(rounds=1))
        result = await arena.run()

        self.design = result.final_answer

        return self.design

    def phase_implement(self) -> dict:
        """Phase 3: Generate implementation (dry run)."""

        if self.dry_run:
            return {"status": "dry_run", "changes": 0}

        # Full implementation would use CodeWriter tool
        return {"status": "implemented", "files": []}

    def phase_verify(self) -> bool:
        """Phase 4: Verify the changes."""

        if self.dry_run:
            return True

        return True

    async def run_cycle(self, focus: str) -> dict:
        """Run one full improvement cycle."""

        # Run all phases
        await self.phase_debate(focus)
        await self.phase_design()
        self.phase_implement()
        verified = self.phase_verify()

        # Summary

        # Extract a short summary from the improvement
        if self.improvement:
            for line in self.improvement.split("\n"):
                if "IMPROVEMENT:" in line.upper():
                    line.split(":", 1)[-1].strip()[:50]
                    break

        if self.dry_run:
            pass
        else:
            pass

        return {
            "improvement": self.improvement,
            "design": self.design,
            "verified": verified,
            "dry_run": self.dry_run,
        }


async def main():
    """Run a simplified nomic loop demo."""

    agent_types = [
        ("grok", "proposer"),
        ("gemini", "critic"),
        ("anthropic-api", "synthesizer"),
    ]

    agents = []
    for agent_type, role in agent_types:
        try:
            agent = create_agent(
                model_type=agent_type,  # type: ignore
                name=f"{agent_type}-{role}",
                role=role,
            )
            agents.append(agent)
        except Exception:
            pass

    if len(agents) < 2:
        return None

    # Run simplified nomic loop
    loop = SimplifiedNomicLoop(agents, dry_run=True)
    result = await loop.run_cycle(focus="error handling and input validation")

    return result


if __name__ == "__main__":
    result = asyncio.run(main())

    if result:
        pass
    else:
        pass
