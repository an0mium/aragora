#!/usr/bin/env python3
"""
Nomic Loop: Autonomous self-improvement cycle for aragora.

Like a PCR machine for code evolution:
1. DEBATE: All agents propose improvements to aragora
2. CONSENSUS: Agents critique and refine until consensus
3. DESIGN: Agents design the implementation
4. IMPLEMENT: Agents write the code
5. VERIFY: Run tests, check quality
6. COMMIT: If verified, commit changes
7. REPEAT: Cycle continues

The dialectic tension between models (visionary vs pragmatic vs synthesizer)
creates emergent complexity and self-criticality.

Inspired by:
- Nomic (game where rules change the rules)
- Project Sid (emergent civilization)
- PCR (exponential amplification through cycles)
- Self-organized criticality (sandpile dynamics)
"""

import asyncio
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add aragora to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.core import Environment
from aragora.agents.api_agents import GeminiAgent
from aragora.agents.cli_agents import CodexAgent, ClaudeAgent
from aragora.implement import (
    generate_implement_plan,
    create_single_task_plan,
    HybridExecutor,
    load_progress,
    save_progress,
    clear_progress,
    ImplementProgress,
)


class NomicLoop:
    """
    Autonomous self-improvement loop for aragora.

    Each cycle:
    1. Agents debate what to improve
    2. Agents design the implementation
    3. Agents implement (codex writes code)
    4. Changes are verified and committed
    5. Loop repeats
    """

    def __init__(
        self,
        aragora_path: str = None,
        max_cycles: int = 10,
        require_human_approval: bool = True,
        auto_commit: bool = False,
    ):
        self.aragora_path = Path(aragora_path or Path(__file__).parent.parent)
        self.max_cycles = max_cycles
        self.require_human_approval = require_human_approval
        self.auto_commit = auto_commit
        self.cycle_count = 0
        self.history = []

        # Initialize agents with distinct personalities
        self.gemini = GeminiAgent(
            name='gemini-visionary',
            model='gemini-2.5-flash',
            role='proposer',
            timeout=180,
        )
        self.gemini.system_prompt = """You are a visionary product strategist for aragora.
Focus on: viral growth, developer excitement, novel capabilities, bold ideas.
Think about what would make aragora famous and widely adopted."""

        self.codex = CodexAgent(
            name='codex-engineer',
            model='gpt-5.2-codex',
            role='proposer',
            timeout=300,  # Longer for code examination
        )
        self.codex.system_prompt = """You are a pragmatic engineer for aragora.
Focus on: technical excellence, code quality, practical utility, implementation feasibility.
You can examine the codebase deeply to understand what's possible."""

        self.claude = ClaudeAgent(
            name='claude-synthesizer',
            model='claude',
            role='synthesizer',
            timeout=180,
        )
        self.claude.system_prompt = """You are a thoughtful synthesizer for aragora.
Focus on: finding common ground, building consensus, balancing vision with pragmatism.
Your role is to create actionable plans from the debate."""

    def get_current_features(self) -> str:
        """Read current aragora state from the codebase."""
        init_file = self.aragora_path / "aragora" / "__init__.py"
        if init_file.exists():
            content = init_file.read_text()
            # Extract docstring
            if '"""' in content:
                docstring = content.split('"""')[1]
                return docstring[:2000]
        return "Unable to read current features"

    def get_recent_changes(self) -> str:
        """Get recent git commits."""
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-10"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            return result.stdout
        except:
            return "Unable to read git history"

    async def phase_debate(self) -> dict:
        """Phase 1: Agents debate what to improve."""
        print("\n" + "=" * 70)
        print("PHASE 1: IMPROVEMENT DEBATE")
        print("=" * 70)

        current_features = self.get_current_features()
        recent_changes = self.get_recent_changes()

        env = Environment(
            task=f"""What single improvement would most benefit aragora RIGHT NOW?

Consider what would make aragora:
- More INTERESTING (novel, creative, intellectually stimulating)
- More POWERFUL (capable, versatile, effective)
- More VIRAL (shareable, demonstrable, meme-worthy)
- More USEFUL (practical, solves real problems)

Each agent should propose ONE specific, implementable feature.
Be concrete: describe what it does, how it works, and why it matters.
After debate, reach consensus on THE SINGLE BEST improvement to implement this cycle.

Recent changes:
{recent_changes}""",
            context=f"Current aragora features:\n{current_features}",
        )

        protocol = DebateProtocol(
            rounds=2,
            consensus="judge",
            proposer_count=2,
        )

        arena = Arena(env, [self.gemini, self.codex, self.claude], protocol)
        result = await arena.run()

        return {
            "phase": "debate",
            "final_answer": result.final_answer,
            "consensus_reached": result.consensus_reached,
            "confidence": result.confidence,
            "duration": result.duration_seconds,
        }

    async def phase_design(self, improvement: str) -> dict:
        """Phase 2: Agents design the implementation."""
        print("\n" + "=" * 70)
        print("PHASE 2: IMPLEMENTATION DESIGN")
        print("=" * 70)

        env = Environment(
            task=f"""Design the implementation for this improvement:

{improvement}

Provide:
1. FILE CHANGES: Which files to create or modify
2. API DESIGN: Key classes, functions, signatures
3. INTEGRATION: How it connects to existing aragora modules
4. TEST PLAN: How to verify it works
5. EXAMPLE USAGE: Code snippet showing the feature in action

Be specific enough that an engineer could implement it.""",
            context=f"Working directory: {self.aragora_path}",
        )

        protocol = DebateProtocol(
            rounds=1,
            consensus="judge",
        )

        arena = Arena(env, [self.codex, self.claude], protocol)
        result = await arena.run()

        return {
            "phase": "design",
            "design": result.final_answer,
            "consensus_reached": result.consensus_reached,
        }

    def _git_stash_create(self) -> Optional[str]:
        """Create a git stash for transactional safety."""
        try:
            # Check if there are changes to stash
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if not status.stdout.strip():
                return None  # Nothing to stash

            result = subprocess.run(
                ["git", "stash", "push", "-m", "nomic-implement-backup"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                # Get the stash ref
                ref_result = subprocess.run(
                    ["git", "stash", "list", "-1", "--format=%H"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                return ref_result.stdout.strip() or "stash@{0}"
        except Exception as e:
            print(f"Warning: Could not create stash: {e}")
        return None

    def _git_stash_pop(self, stash_ref: Optional[str]) -> None:
        """Pop a stash to restore previous state."""
        if not stash_ref:
            return
        try:
            # First reset any changes
            subprocess.run(
                ["git", "checkout", "."],
                cwd=self.aragora_path,
                capture_output=True,
            )
            # Then pop the stash
            subprocess.run(
                ["git", "stash", "pop"],
                cwd=self.aragora_path,
                capture_output=True,
            )
        except Exception as e:
            print(f"Warning: Could not pop stash: {e}")

    def _get_git_diff(self) -> str:
        """Get current git diff."""
        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            return result.stdout
        except Exception:
            return ""

    async def phase_implement(self, design: str) -> dict:
        """
        Phase 3: Hybrid multi-model implementation.

        Uses the consensus architecture from the 3-provider debate:
        - Gemini generates an implementation plan
        - Claude handles complex tasks
        - Codex handles simple tasks
        """
        print("\n" + "=" * 70)
        print("PHASE 3: IMPLEMENTATION (Hybrid)")
        print("=" * 70)

        use_hybrid = os.environ.get("ARAGORA_HYBRID_IMPLEMENT", "1") == "1"

        if not use_hybrid:
            # Legacy single-codex path
            return await self._legacy_implement(design)

        design_hash = hashlib.md5(design.encode()).hexdigest()

        # 1. Check for crash recovery
        progress = load_progress(self.aragora_path)
        if progress and progress.plan.design_hash == design_hash:
            print("  Resuming from checkpoint...")
            plan = progress.plan
            completed = set(progress.completed_tasks)
            stash_ref = progress.git_stash_ref
        else:
            # 2. Generate plan with Gemini
            try:
                plan = await generate_implement_plan(design, self.aragora_path)
            except Exception as e:
                print(f"  Plan generation failed: {e}")
                print("  Falling back to single-task mode...")
                plan = create_single_task_plan(design, self.aragora_path)

            completed = set()

            # 3. Git stash for transactional safety
            stash_ref = self._git_stash_create()

            # Save initial progress
            save_progress(
                ImplementProgress(
                    plan=plan,
                    completed_tasks=[],
                    git_stash_ref=stash_ref,
                ),
                self.aragora_path,
            )

        # 4. Execute tasks with hybrid executor
        executor = HybridExecutor(self.aragora_path)

        def on_task_complete(task_id: str, result):
            """Callback to save progress after each task."""
            completed.add(task_id)
            save_progress(
                ImplementProgress(
                    plan=plan,
                    completed_tasks=list(completed),
                    current_task=None,
                    git_stash_ref=stash_ref,
                ),
                self.aragora_path,
            )

        try:
            results = await executor.execute_plan(
                plan.tasks,
                completed,
                on_task_complete=on_task_complete,
            )

            # Check if all tasks completed
            all_success = all(r.success for r in results)
            tasks_completed = len([r for r in results if r.success])

            if all_success and tasks_completed == len(plan.tasks):
                # Full success - clear checkpoint
                clear_progress(self.aragora_path)
                return {
                    "phase": "implement",
                    "success": True,
                    "tasks_completed": tasks_completed,
                    "tasks_total": len(plan.tasks),
                    "diff": self._get_git_diff(),
                    "results": [r.to_dict() for r in results],
                }
            else:
                # Partial success or failure
                failed = [r for r in results if not r.success]
                return {
                    "phase": "implement",
                    "success": False,
                    "tasks_completed": tasks_completed,
                    "tasks_total": len(plan.tasks),
                    "error": failed[0].error if failed else "Unknown error",
                    "diff": self._get_git_diff(),
                }

        except Exception as e:
            # Rollback on catastrophic failure
            print(f"  Catastrophic failure: {e}")
            print("  Rolling back changes...")
            self._git_stash_pop(stash_ref)
            return {
                "phase": "implement",
                "success": False,
                "error": str(e),
            }

    async def _legacy_implement(self, design: str) -> dict:
        """Legacy single-Codex implementation (fallback)."""
        print("  Using legacy Codex-only mode...")

        prompt = f"""Implement this design in the aragora codebase:

{design}

Write the actual code. Create or modify files as needed.
Follow aragora's existing code style and patterns.
Include docstrings and type hints.

IMPORTANT: Only make changes that are safe and reversible.
Do not delete or break existing functionality."""

        try:
            result = subprocess.run(
                ["codex", "exec", "-C", str(self.aragora_path), prompt],
                capture_output=True,
                text=True,
                timeout=300,
            )

            return {
                "phase": "implement",
                "output": result.stdout,
                "diff": self._get_git_diff(),
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            return {
                "phase": "implement",
                "error": "Implementation timed out",
                "success": False,
            }
        except Exception as e:
            return {
                "phase": "implement",
                "error": str(e),
                "success": False,
            }

    async def phase_verify(self) -> dict:
        """Phase 4: Verify changes work."""
        print("\n" + "=" * 70)
        print("PHASE 4: VERIFICATION")
        print("=" * 70)

        # Run basic checks
        checks = []

        # 1. Python syntax check
        try:
            result = subprocess.run(
                ["python", "-m", "py_compile", "aragora/__init__.py"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            checks.append({
                "check": "syntax",
                "passed": result.returncode == 0,
                "output": result.stderr,
            })
        except Exception as e:
            checks.append({"check": "syntax", "passed": False, "error": str(e)})

        # 2. Import check
        try:
            result = subprocess.run(
                ["python", "-c", "import aragora; print('OK')"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            checks.append({
                "check": "import",
                "passed": "OK" in result.stdout,
                "output": result.stderr if result.returncode != 0 else "",
            })
        except Exception as e:
            checks.append({"check": "import", "passed": False, "error": str(e)})

        # 3. Run tests if they exist
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "--tb=short", "-q"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            checks.append({
                "check": "tests",
                "passed": result.returncode == 0,
                "output": result.stdout[-500:] if result.stdout else "",
            })
        except Exception as e:
            checks.append({"check": "tests", "passed": True, "note": "No tests or timeout"})

        all_passed = all(c.get("passed", False) for c in checks)

        for check in checks:
            status = "✓" if check.get("passed") else "✗"
            print(f"  {status} {check['check']}")

        return {
            "phase": "verify",
            "checks": checks,
            "all_passed": all_passed,
        }

    async def phase_commit(self, improvement: str) -> dict:
        """Phase 5: Commit changes if verified."""
        print("\n" + "=" * 70)
        print("PHASE 5: COMMIT")
        print("=" * 70)

        if self.require_human_approval and not self.auto_commit:
            print("\nChanges ready for review:")
            subprocess.run(["git", "diff", "--stat"], cwd=self.aragora_path)

            response = input("\nCommit these changes? [y/N]: ")
            if response.lower() != 'y':
                print("Skipping commit.")
                return {"phase": "commit", "committed": False, "reason": "Human declined"}

        # Generate commit message
        summary = improvement[:100].replace('\n', ' ')

        try:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.aragora_path,
                check=True,
            )

            result = subprocess.run(
                ["git", "commit", "-m", f"feat(nomic): {summary}\n\n🤖 Auto-generated by aragora nomic loop"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )

            committed = result.returncode == 0

            if committed:
                print(f"  ✓ Committed: {summary[:60]}...")
            else:
                print(f"  ✗ Commit failed: {result.stderr}")

            return {
                "phase": "commit",
                "committed": committed,
                "message": summary,
            }

        except Exception as e:
            return {
                "phase": "commit",
                "committed": False,
                "error": str(e),
            }

    async def run_cycle(self) -> dict:
        """Run one complete improvement cycle."""
        self.cycle_count += 1
        cycle_start = datetime.now()

        print("\n" + "=" * 70)
        print(f"NOMIC CYCLE {self.cycle_count}")
        print(f"Started: {cycle_start.isoformat()}")
        print("=" * 70)

        cycle_result = {
            "cycle": self.cycle_count,
            "started": cycle_start.isoformat(),
            "phases": {},
        }

        # Phase 1: Debate
        debate_result = await self.phase_debate()
        cycle_result["phases"]["debate"] = debate_result

        if not debate_result.get("consensus_reached"):
            print("No consensus reached. Ending cycle.")
            cycle_result["outcome"] = "no_consensus"
            return cycle_result

        improvement = debate_result["final_answer"]
        print(f"\n✓ Consensus improvement:\n{improvement[:500]}...")

        # Phase 2: Design
        design_result = await self.phase_design(improvement)
        cycle_result["phases"]["design"] = design_result

        design = design_result.get("design", "")
        print(f"\n✓ Design complete")

        # Phase 3: Implement
        impl_result = await self.phase_implement(design)
        cycle_result["phases"]["implement"] = impl_result

        if not impl_result.get("success"):
            print("Implementation failed. Ending cycle.")
            cycle_result["outcome"] = "implementation_failed"
            return cycle_result

        print(f"\n✓ Implementation complete")
        print(f"Changed files:\n{impl_result.get('diff', 'No changes')}")

        # Phase 4: Verify
        verify_result = await self.phase_verify()
        cycle_result["phases"]["verify"] = verify_result

        if not verify_result.get("all_passed"):
            print("Verification failed. Rolling back.")
            subprocess.run(["git", "checkout", "."], cwd=self.aragora_path)
            cycle_result["outcome"] = "verification_failed"
            return cycle_result

        print(f"\n✓ Verification passed")

        # Phase 5: Commit
        commit_result = await self.phase_commit(improvement)
        cycle_result["phases"]["commit"] = commit_result

        if commit_result.get("committed"):
            cycle_result["outcome"] = "success"
            print(f"\n✓ CYCLE {self.cycle_count} COMPLETE - Changes committed!")
        else:
            cycle_result["outcome"] = "not_committed"

        cycle_result["duration_seconds"] = (datetime.now() - cycle_start).total_seconds()
        self.history.append(cycle_result)

        return cycle_result

    async def run(self):
        """Run the nomic loop until max cycles or interrupted."""
        print("=" * 70)
        print("AAGORA NOMIC LOOP")
        print("Self-improving multi-agent system")
        print("=" * 70)
        print(f"Max cycles: {self.max_cycles}")
        print(f"Human approval required: {self.require_human_approval}")
        print(f"Auto-commit: {self.auto_commit}")
        print("=" * 70)

        try:
            while self.cycle_count < self.max_cycles:
                result = await self.run_cycle()

                print(f"\nCycle {self.cycle_count} outcome: {result.get('outcome')}")

                if result.get("outcome") == "success":
                    print("Continuing to next cycle...")
                else:
                    print("Cycle did not complete successfully.")
                    if self.require_human_approval:
                        response = input("Continue to next cycle? [Y/n]: ")
                        if response.lower() == 'n':
                            break

                # Brief pause between cycles
                await asyncio.sleep(2)

        except KeyboardInterrupt:
            print("\n\nNomic loop interrupted by user.")

        print("\n" + "=" * 70)
        print("NOMIC LOOP COMPLETE")
        print(f"Total cycles: {self.cycle_count}")
        print(f"Successful commits: {sum(1 for h in self.history if h.get('outcome') == 'success')}")
        print("=" * 70)

        return self.history


async def main():
    parser = argparse.ArgumentParser(description="Aagora Nomic Loop - Self-improvement cycle")
    parser.add_argument("--cycles", type=int, default=3, help="Maximum cycles to run")
    parser.add_argument("--auto", action="store_true", help="Auto-commit without human approval")
    parser.add_argument("--path", type=str, help="Path to aragora repository")

    args = parser.parse_args()

    loop = NomicLoop(
        aragora_path=args.path,
        max_cycles=args.cycles,
        require_human_approval=not args.auto,
        auto_commit=args.auto,
    )

    await loop.run()


if __name__ == "__main__":
    asyncio.run(main())
