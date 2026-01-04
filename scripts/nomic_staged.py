#!/usr/bin/env python3
"""
Staged Nomic Loop - Run each phase separately to avoid timeouts.

Usage:
    python scripts/nomic_staged.py debate      # Phase 1: Debate improvements
    python scripts/nomic_staged.py design      # Phase 2: Design implementation
    python scripts/nomic_staged.py implement   # Phase 3: Implement (manual or Claude)
    python scripts/nomic_staged.py verify      # Phase 4: Verify changes
    python scripts/nomic_staged.py commit      # Phase 5: Commit changes

    python scripts/nomic_staged.py all         # Run all phases sequentially

Each phase saves output to .nomic/ directory for the next phase.
"""

import asyncio
import argparse
import json
import logging
import subprocess

logger = logging.getLogger(__name__)
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.core import Environment
from aragora.agents.cli_agents import ClaudeAgent, CodexAgent, GeminiCLIAgent


NOMIC_DIR = Path(__file__).parent.parent / ".nomic"
ARAGORA_PATH = Path(__file__).parent.parent


def ensure_nomic_dir():
    NOMIC_DIR.mkdir(exist_ok=True)


def save_phase(phase: str, data: dict):
    ensure_nomic_dir()
    path = NOMIC_DIR / f"{phase}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def load_phase(phase: str) -> dict:
    path = NOMIC_DIR / f"{phase}.json"
    if not path.exists():
        print(f"Error: {path} not found. Run '{phase}' phase first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def get_current_features() -> str:
    """Read current aragora state."""
    init_file = ARAGORA_PATH / "aragora" / "__init__.py"
    if init_file.exists():
        content = init_file.read_text()
        if '"""' in content:
            return content.split('"""')[1][:2000]
    return "Unable to read features"


def get_recent_changes() -> str:
    """Get recent git commits."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-10"],
            cwd=ARAGORA_PATH,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except Exception as e:
        logger.warning(f"Failed to read git history: {e}")
        return "Unable to read git history"


async def phase_debate():
    """Phase 1: Multi-agent debate on improvements."""
    print("\n" + "=" * 70)
    print("PHASE 1: IMPROVEMENT DEBATE")
    print("=" * 70 + "\n")

    current_features = get_current_features()
    recent_changes = get_recent_changes()

    env = Environment(
        task=f"""What single improvement would most benefit aragora RIGHT NOW?

Consider what would make aragora:
- More INTERESTING (novel, creative, intellectually stimulating)
- More POWERFUL (capable, versatile, effective)
- More VIRAL (shareable, demonstrable, meme-worthy)
- More USEFUL (practical, solves real problems)

Each agent should propose ONE specific, implementable feature.
Be concrete: describe what it does, how it works, and why it matters.

After debate, reach consensus on THE SINGLE BEST improvement to implement.

Recent changes:
{recent_changes}""",
        context=f"Current aragora features:\n{current_features}",
    )

    # Heterogeneous agents: 3 competing visionaries from different AI providers
    gemini_visionary = GeminiCLIAgent(
        name="gemini-visionary",
        model="gemini-3-pro-preview",
        role="proposer",
        timeout=360,  # Doubled from 180
    )
    gemini_visionary.system_prompt = """You are a visionary strategist representing Google's perspective.
Propose ONE bold, specific improvement for aragora.
Focus on: scalability, integration, enterprise readiness, novel ML applications.
Argue passionately for your proposal. Challenge other proposals directly."""

    codex_visionary = CodexAgent(
        name="codex-visionary",
        model="o3",  # GPT model via codex CLI
        role="proposer",
        timeout=360,  # Doubled from 180
    )
    codex_visionary.system_prompt = """You are a visionary strategist representing OpenAI's perspective.
Propose ONE bold, specific improvement for aragora.
Focus on: developer experience, code quality, practical utility, elegance.
Argue passionately for your proposal. Challenge other proposals directly."""

    claude_visionary = ClaudeAgent(
        name="claude-visionary",
        model="claude-sonnet-4-20250514",
        role="proposer",
        timeout=360,  # Doubled from 180
    )
    claude_visionary.system_prompt = """You are a visionary strategist representing Anthropic's perspective.
Propose ONE bold, specific improvement for aragora.
Focus on: safety, interpretability, thoughtful design, philosophical depth.
Argue passionately for your proposal. Challenge other proposals directly."""

    agents = [gemini_visionary, codex_visionary, claude_visionary]

    print("Agents: Gemini vs GPT/Codex vs Claude - ALL COMPETING VISIONARIES")
    print("TRUE heterogeneous debate - 3 different AI providers each proposing their vision.\n")

    protocol = DebateProtocol(rounds=2, consensus="judge")
    arena = Arena(env, agents, protocol)
    result = await arena.run()

    data = {
        "timestamp": datetime.now().isoformat(),
        "final_answer": result.final_answer,
        "consensus_reached": result.consensus_reached,
        "confidence": result.confidence,
        "duration": result.duration_seconds,
        "messages": [
            {"agent": m.agent, "role": m.role, "round": m.round, "content": m.content[:500]}
            for m in result.messages
        ],
    }

    save_phase("debate", data)

    print("\n" + "=" * 70)
    print("DEBATE RESULT:")
    print("=" * 70)
    print(f"\nConsensus: {'Yes' if result.consensus_reached else 'No'} ({result.confidence:.0%})")
    print(f"\n{result.final_answer}")

    return data


async def phase_design():
    """Phase 2: Design the implementation."""
    print("\n" + "=" * 70)
    print("PHASE 2: IMPLEMENTATION DESIGN")
    print("=" * 70 + "\n")

    debate_data = load_phase("debate")
    improvement = debate_data["final_answer"]

    print(f"Designing implementation for:\n{improvement[:300]}...\n")

    env = Environment(
        task=f"""Design the implementation for this improvement:

{improvement}

Provide:
1. FILE CHANGES: Which files to create or modify (with paths)
2. API DESIGN: Key classes, functions, signatures
3. INTEGRATION: How it connects to existing aragora modules
4. TEST PLAN: How to verify it works
5. EXAMPLE USAGE: Code snippet showing the feature

Be specific enough that an engineer could implement it.""",
        context=f"aragora path: {ARAGORA_PATH}",
    )

    agents = [
        ClaudeAgent(
            name="architect",
            model="claude-sonnet-4-20250514",
            role="proposer",
            timeout=600,  # 10 min for complex design work (doubled from 300)
        ),
        ClaudeAgent(
            name="reviewer",
            model="claude-sonnet-4-20250514",
            role="synthesizer",
            timeout=600,  # Doubled from 300
        ),
    ]

    protocol = DebateProtocol(rounds=1, consensus="judge")
    arena = Arena(env, agents, protocol)
    result = await arena.run()

    data = {
        "timestamp": datetime.now().isoformat(),
        "improvement": improvement,
        "design": result.final_answer,
        "consensus_reached": result.consensus_reached,
    }

    save_phase("design", data)

    print("\n" + "=" * 70)
    print("DESIGN RESULT:")
    print("=" * 70)
    print(f"\n{result.final_answer}")

    return data


async def phase_implement():
    """Phase 3: Implementation guidance."""
    print("\n" + "=" * 70)
    print("PHASE 3: IMPLEMENTATION")
    print("=" * 70 + "\n")

    design_data = load_phase("design")
    design = design_data["design"]

    print("Design to implement:")
    print("-" * 40)
    print(design[:1000])
    print("-" * 40)

    print("\n\nIMPLEMENTATION OPTIONS:")
    print("1. Implement manually based on the design above")
    print("2. Use Claude Code to implement: claude -p 'Implement this design in aragora: ...'")
    print("3. Continue to verification (if already implemented)")

    # Save implementation status
    data = {
        "timestamp": datetime.now().isoformat(),
        "design": design,
        "status": "ready_for_implementation",
        "instructions": "Implement the design above, then run 'python scripts/nomic_staged.py verify'",
    }

    save_phase("implement", data)

    print(f"\nNext step: Implement the design, then run:")
    print(f"  python scripts/nomic_staged.py verify")

    return data


async def phase_verify():
    """Phase 4: Verify changes."""
    print("\n" + "=" * 70)
    print("PHASE 4: VERIFICATION")
    print("=" * 70 + "\n")

    checks = []

    # 1. Python syntax check
    print("Checking syntax...")
    try:
        result = subprocess.run(
            ["python3", "-m", "py_compile", "aragora/__init__.py"],
            cwd=ARAGORA_PATH,
            capture_output=True,
            text=True,
        )
        passed = result.returncode == 0
        checks.append({"check": "syntax", "passed": passed})
        print(f"  {'✓' if passed else '✗'} Syntax check")
    except Exception as e:
        checks.append({"check": "syntax", "passed": False, "error": str(e)})
        print(f"  ✗ Syntax check: {e}")

    # 2. Import check
    print("Checking imports...")
    try:
        result = subprocess.run(
            ["python3", "-c", "import aragora; print('OK')"],
            cwd=ARAGORA_PATH,
            capture_output=True,
            text=True,
            timeout=180,  # Minimum 3 min (was 30)
        )
        passed = "OK" in result.stdout
        checks.append({"check": "import", "passed": passed})
        print(f"  {'✓' if passed else '✗'} Import check")
        if not passed:
            print(f"      {result.stderr[:200]}")
    except Exception as e:
        checks.append({"check": "import", "passed": False, "error": str(e)})
        print(f"  ✗ Import check: {e}")

    # 3. Git status
    print("Checking git status...")
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=ARAGORA_PATH,
            capture_output=True,
            text=True,
        )
        has_changes = bool(result.stdout.strip())
        checks.append({"check": "git_changes", "has_changes": has_changes, "diff": result.stdout})
        print(f"  {'✓' if has_changes else '○'} Git changes: {'Yes' if has_changes else 'No changes detected'}")
        if has_changes:
            print(result.stdout)
    except Exception as e:
        checks.append({"check": "git_changes", "error": str(e)})

    all_passed = all(c.get("passed", True) for c in checks if "passed" in c)

    data = {
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
        "all_passed": all_passed,
    }

    save_phase("verify", data)

    print("\n" + "-" * 40)
    print(f"Verification: {'PASSED' if all_passed else 'FAILED'}")

    if all_passed:
        print(f"\nNext step: python scripts/nomic_staged.py commit")

    return data


async def phase_commit():
    """Phase 5: Commit changes."""
    print("\n" + "=" * 70)
    print("PHASE 5: COMMIT")
    print("=" * 70 + "\n")

    # Load previous phases for context
    try:
        debate_data = load_phase("debate")
        improvement = debate_data.get("final_answer", "Nomic improvement")[:100]
    except Exception as e:
        logger.warning(f"Failed to load debate phase data: {e}")
        improvement = "Nomic improvement"

    # Check for changes
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ARAGORA_PATH,
        capture_output=True,
        text=True,
    )

    if not result.stdout.strip():
        print("No changes to commit.")
        return {"committed": False, "reason": "no_changes"}

    # Show changes
    print("Changes to commit:")
    subprocess.run(["git", "diff", "--stat"], cwd=ARAGORA_PATH)

    # Confirm
    response = input("\nCommit these changes? [y/N]: ")
    if response.lower() != "y":
        print("Commit cancelled.")
        return {"committed": False, "reason": "user_cancelled"}

    # Commit
    summary = improvement.replace("\n", " ")[:80]
    subprocess.run(["git", "add", "-A"], cwd=ARAGORA_PATH)

    commit_msg = f"""feat(nomic): {summary}

🤖 Generated by aragora nomic loop

Co-Authored-By: Claude <noreply@anthropic.com>
"""

    result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=ARAGORA_PATH,
        capture_output=True,
        text=True,
    )

    committed = result.returncode == 0
    print(f"\n{'✓ Committed!' if committed else '✗ Commit failed'}")

    if committed:
        # Show commit
        subprocess.run(["git", "log", "--oneline", "-1"], cwd=ARAGORA_PATH)

    data = {
        "timestamp": datetime.now().isoformat(),
        "committed": committed,
        "message": summary,
    }

    save_phase("commit", data)

    return data


async def run_all():
    """Run all phases sequentially."""
    print("\n" + "=" * 70)
    print("ARAGORA NOMIC LOOP - FULL CYCLE")
    print("=" * 70)

    await phase_debate()
    await phase_design()
    await phase_implement()

    print("\n" + "=" * 70)
    print("PAUSING FOR IMPLEMENTATION")
    print("=" * 70)
    print("\nThe debate and design phases are complete.")
    print("Review the design in .nomic/design.json")
    print("\nAfter implementing, run:")
    print("  python scripts/nomic_staged.py verify")
    print("  python scripts/nomic_staged.py commit")


async def main():
    parser = argparse.ArgumentParser(
        description="Staged Nomic Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  debate     Multi-agent debate on what to improve
  design     Design the implementation
  implement  Instructions for implementation
  verify     Verify changes work
  commit     Commit the changes
  all        Run debate + design + implement (pauses before verify)
        """,
    )
    parser.add_argument(
        "phase",
        choices=["debate", "design", "implement", "verify", "commit", "all"],
        help="Phase to run",
    )

    args = parser.parse_args()

    if args.phase == "debate":
        await phase_debate()
    elif args.phase == "design":
        await phase_design()
    elif args.phase == "implement":
        await phase_implement()
    elif args.phase == "verify":
        await phase_verify()
    elif args.phase == "commit":
        await phase_commit()
    elif args.phase == "all":
        await run_all()


if __name__ == "__main__":
    asyncio.run(main())
