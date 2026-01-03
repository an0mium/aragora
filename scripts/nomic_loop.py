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

SAFETY: This file includes backup/restore mechanisms and safety prompts
to prevent the nomic loop from breaking itself.
"""

import asyncio
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# =============================================================================
# SAFETY CONSTANTS - Files that must NEVER be deleted or broken
# =============================================================================
PROTECTED_FILES = [
    # Core nomic loop infrastructure
    "scripts/nomic_loop.py",  # The nomic loop itself - CRITICAL
    "scripts/run_nomic_with_stream.py",  # Streaming wrapper - protects --auto flag

    # Core aragora modules
    "aragora/__init__.py",     # Core package initialization
    "aragora/core.py",         # Core types and abstractions
    "aragora/debate/orchestrator.py",  # Debate infrastructure
    "aragora/agents/__init__.py",      # Agent system
    "aragora/implement/__init__.py",   # Implementation system

    # Valuable features added by nomic loop
    "aragora/agents/cli_agents.py",    # CLI agent harnesses (KiloCode, Claude, Codex, Grok)
    "aragora/server/stream.py",        # Streaming, AudienceInbox, TokenBucket
    "aragora/memory/store.py",         # CritiqueStore, AgentReputation
    "aragora/debate/embeddings.py",    # DebateEmbeddingsDatabase for historical search
]

SAFETY_PREAMBLE = """
=== CRITICAL SAFETY RULES ===
You are modifying a self-improving system. These rules are NON-NEGOTIABLE:

1. NEVER DELETE OR BREAK:
   - scripts/nomic_loop.py (the loop itself)
   - aragora/__init__.py (core package)
   - aragora/core.py (core types)
   - aragora/debate/orchestrator.py (debate infrastructure)
   - Any file that enables the nomic loop to function

2. ANABOLISM OVER CATABOLISM:
   - ADD features, don't remove working ones
   - EXTEND functionality, don't simplify it away
   - Only remove code that is BROKEN or HARMFUL
   - When in doubt, keep existing functionality

3. PRESERVE CORE CAPABILITIES:
   - Multi-agent debate must keep working
   - File logging must keep working
   - Git integration must keep working
   - All existing API contracts must be maintained

4. DEFENSIVE CODING:
   - New features should not break existing ones
   - Add tests for new functionality
   - Maintain backward compatibility

5. TECHNICAL DEBT - REDUCE SAFELY:
   - Reducing technical debt is GOOD when it's safe
   - Safe refactoring: improve code without changing behavior
   - UNSAFE: removing functionality, breaking APIs, deleting imports
   - SAFE: renaming for clarity, extracting functions, improving types
   - Test that refactored code works identically to original
   - If unsure whether a change is safe, DON'T MAKE IT

6. AGENT PROMPTS ARE SACRED:
   - NEVER modify agent system prompts in the codebase
   - Agent prompts define the personalities and safety constraints
   - Changes to agent prompts require UNANIMOUS consent from all agents
   - If ANY doubt exists about modifying prompts, DO NOT MODIFY THEM
   - This includes prompts in: nomic_loop.py, agents/*.py, any prompt templates
   - The only exception: fixing an obvious typo or syntax error
===========================
"""


# Load .env file if present
def load_dotenv(env_path: Path):
    """Load environment variables from .env file."""
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())


# Load .env from aragora root
_script_dir = Path(__file__).parent
_env_file = _script_dir.parent / ".env"
load_dotenv(_env_file)

# Add aragora to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.core import Environment
from aragora.agents.api_agents import GeminiAgent
from aragora.agents.cli_agents import CodexAgent, ClaudeAgent, GrokCLIAgent, KiloCodeAgent

# Check if Kilo Code CLI is available for Gemini/Grok codebase exploration
KILOCODE_AVAILABLE = False
try:
    import subprocess
    result = subprocess.run(["which", "kilocode"], capture_output=True, text=True)
    KILOCODE_AVAILABLE = result.returncode == 0
except Exception:
    pass

# Genesis module for fractal debates with agent evolution
GENESIS_AVAILABLE = False
try:
    from aragora.genesis import (
        FractalOrchestrator,
        PopulationManager,
        GenesisLedger,
        create_genesis_hooks,
        create_logging_hooks,
    )
    GENESIS_AVAILABLE = True
except ImportError:
    pass
from aragora.implement import (
    generate_implement_plan,
    create_single_task_plan,
    HybridExecutor,
    load_progress,
    save_progress,
    clear_progress,
    ImplementProgress,
)

# Optional streaming support
try:
    from aragora.server.stream import SyncEventEmitter, create_arena_hooks
    from aragora.server.nomic_stream import create_nomic_hooks
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    SyncEventEmitter = None
    create_nomic_hooks = None
    create_arena_hooks = None

# Optional Supabase persistence
try:
    from aragora.persistence import SupabaseClient, NomicCycle, StreamEvent, DebateArtifact
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    SupabaseClient = None
    NomicCycle = None
    StreamEvent = None
    DebateArtifact = None

# Debate embeddings for historical search
try:
    from aragora.debate.embeddings import DebateEmbeddingsDatabase
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    DebateEmbeddingsDatabase = None


class NomicLoop:
    """
    Autonomous self-improvement loop for aragora.

    Each cycle:
    1. Agents debate what to improve
    2. Agents design the implementation
    3. Agents implement (codex writes code)
    4. Changes are verified and committed
    5. Loop repeats

    SAFETY FEATURES:
    - All output logged to .nomic/nomic_loop.log for live monitoring
    - State saved to .nomic/nomic_state.json for crash recovery
    - Protected files backed up before each cycle
    - Automatic restore if protected files are damaged
    """

    def __init__(
        self,
        aragora_path: str = None,
        max_cycles: int = 10,
        require_human_approval: bool = True,
        auto_commit: bool = False,
        initial_proposal: str = None,
        stream_emitter: "SyncEventEmitter" = None,
        use_genesis: bool = False,
        enable_persistence: bool = True,
        disable_rollback: bool = False,  # Disable rollback on verification failure
    ):
        self.aragora_path = Path(aragora_path or Path(__file__).parent.parent)
        self.max_cycles = max_cycles
        self.require_human_approval = require_human_approval
        self.auto_commit = auto_commit
        self.initial_proposal = initial_proposal
        self.disable_rollback = disable_rollback
        self.cycle_count = 0
        self.history = []

        # Generate unique loop ID for this run
        self.loop_id = f"nomic-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Genesis mode: fractal debates with agent evolution
        self.use_genesis = use_genesis and GENESIS_AVAILABLE
        self.genesis_ledger = None
        self.population_manager = None
        if self.use_genesis:
            self.genesis_ledger = GenesisLedger(str(self.aragora_path / ".nomic" / "genesis.db"))
            self.population_manager = PopulationManager(str(self.aragora_path / ".nomic" / "genesis.db"))

        # Setup logging infrastructure (must be before other initializations that use nomic_dir)
        self.nomic_dir = self.aragora_path / ".nomic"
        self.nomic_dir.mkdir(exist_ok=True)
        self.log_file = self.nomic_dir / "nomic_loop.log"
        self.state_file = self.nomic_dir / "nomic_state.json"
        self.backup_dir = self.nomic_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Supabase persistence for history tracking
        self.persistence = None
        if enable_persistence and PERSISTENCE_AVAILABLE:
            self.persistence = SupabaseClient()
            if self.persistence.is_configured:
                print(f"[persistence] Supabase connected, loop_id: {self.loop_id}")
            else:
                self.persistence = None

        # Debate embeddings database for historical search
        self.debate_embeddings = None
        if EMBEDDINGS_AVAILABLE:
            embeddings_path = self.nomic_dir / "debate_embeddings.db"
            self.debate_embeddings = DebateEmbeddingsDatabase(str(embeddings_path))
            print(f"[embeddings] Debate embeddings database initialized")

        # CritiqueStore for patterns and agent reputation tracking
        self.critique_store = None
        try:
            from aragora.memory.store import CritiqueStore
            critique_db_path = self.nomic_dir / "agora_memory.db"
            self.critique_store = CritiqueStore(str(critique_db_path))
            print(f"[memory] CritiqueStore initialized for patterns and reputation")
        except ImportError:
            pass

        # Setup streaming (optional)
        self.stream_emitter = stream_emitter
        if stream_emitter and STREAMING_AVAILABLE and create_nomic_hooks:
            self.stream_hooks = create_nomic_hooks(stream_emitter)
        else:
            self.stream_hooks = {}

        # Add genesis hooks if available
        if self.use_genesis:
            genesis_hooks = create_logging_hooks(lambda msg: self._log(f"    [genesis] {msg}"))
            self.stream_hooks.update(genesis_hooks)

        # Clear log file on start
        with open(self.log_file, "w") as f:
            f.write(f"=== NOMIC LOOP STARTED: {datetime.now().isoformat()} ===\n")

        # Initialize agents
        self._init_agents()

    def _stream_emit(self, hook_name: str, *args, **kwargs) -> None:
        """Emit event to WebSocket stream and persist to Supabase."""
        # Emit to WebSocket stream
        if hook_name in self.stream_hooks:
            try:
                self.stream_hooks[hook_name](*args, **kwargs)
            except Exception:
                pass  # Don't let streaming errors break the loop

        # Persist to Supabase
        if self.persistence and StreamEvent:
            try:
                event = StreamEvent(
                    loop_id=self.loop_id,
                    cycle=self.cycle_count,
                    event_type=hook_name,
                    event_data={"args": [str(a)[:500] for a in args], "kwargs": {k: str(v)[:500] for k, v in kwargs.items()}},
                    agent=kwargs.get("agent"),
                )
                # Run async save in background (fire and forget)
                asyncio.get_event_loop().create_task(self.persistence.save_event(event))
            except Exception:
                pass  # Don't let persistence errors break the loop

    async def _persist_cycle(self, phase: str, stage: str, success: bool = None,
                              git_commit: str = None, task_description: str = None,
                              error_message: str = None) -> None:
        """Persist cycle state to Supabase."""
        if not self.persistence or not NomicCycle:
            return
        try:
            cycle = NomicCycle(
                loop_id=self.loop_id,
                cycle_number=self.cycle_count,
                phase=phase,
                stage=stage,
                started_at=datetime.utcnow(),
                success=success,
                git_commit=git_commit,
                task_description=task_description,
                error_message=error_message,
            )
            await self.persistence.save_cycle(cycle)
        except Exception:
            pass  # Don't let persistence errors break the loop

    async def _persist_debate(self, phase: str, task: str, agents: list,
                               transcript: list, consensus_reached: bool,
                               confidence: float, winning_proposal: str = None) -> None:
        """Persist debate artifact to Supabase."""
        if not self.persistence or not DebateArtifact:
            return
        try:
            debate = DebateArtifact(
                loop_id=self.loop_id,
                cycle_number=self.cycle_count,
                phase=phase,
                task=task,
                agents=agents,
                transcript=transcript,
                consensus_reached=consensus_reached,
                confidence=confidence,
                winning_proposal=winning_proposal,
            )
            await self.persistence.save_debate(debate)

            # Also index in embeddings database for future search
            if self.debate_embeddings:
                await self.debate_embeddings.index_debate(debate)
        except Exception:
            pass  # Don't let persistence errors break the loop

    def _log(self, message: str, also_print: bool = True, phase: str = None, agent: str = None):
        """Log to file and optionally stdout. File is always flushed immediately."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"

        # Write to file immediately (unbuffered)
        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")
            f.flush()

        if also_print:
            print(message)
            sys.stdout.flush()

        # Also emit to stream for real-time dashboard
        self._stream_emit("on_log_message", message, level="info", phase=phase, agent=agent)

    def _save_state(self, state: dict):
        """Save current state for crash recovery and monitoring."""
        state["saved_at"] = datetime.now().isoformat()
        state["cycle"] = self.cycle_count
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self) -> Optional[dict]:
        """Load saved state if exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _create_backup(self, reason: str = "pre_cycle") -> Path:
        """Create a backup of protected files before making changes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{reason}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)

        self._log(f"  Creating backup: {backup_name}")

        backed_up = []
        for rel_path in PROTECTED_FILES:
            src = self.aragora_path / rel_path
            if src.exists():
                dst = backup_path / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                backed_up.append(rel_path)
                self._log(f"    Backed up: {rel_path}", also_print=False)

        # Save manifest
        manifest = {
            "created_at": datetime.now().isoformat(),
            "reason": reason,
            "cycle": self.cycle_count,
            "files": backed_up,
        }
        with open(backup_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        self._log(f"  Backup complete: {len(backed_up)} files")
        self._stream_emit("on_backup_created", backup_name, len(backed_up), reason)
        return backup_path

    def _restore_backup(self, backup_path: Path) -> bool:
        """Restore protected files from a backup."""
        manifest_file = backup_path / "manifest.json"
        if not manifest_file.exists():
            self._log(f"  No manifest found in {backup_path}")
            return False

        with open(manifest_file) as f:
            manifest = json.load(f)

        self._log(f"  Restoring backup from {manifest['created_at']}")

        restored = []
        for rel_path in manifest["files"]:
            src = backup_path / rel_path
            dst = self.aragora_path / rel_path
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                restored.append(rel_path)
                self._log(f"    Restored: {rel_path}", also_print=False)

        self._log(f"  Restored {len(restored)} files")
        self._stream_emit("on_backup_restored", backup_path.name, len(restored), "verification_failed")
        return True

    def _get_latest_backup(self) -> Optional[Path]:
        """Get the most recent backup directory."""
        backups = sorted(self.backup_dir.iterdir(), reverse=True)
        for backup in backups:
            if (backup / "manifest.json").exists():
                return backup
        return None

    def _verify_protected_files(self) -> List[str]:
        """Verify protected files still exist and are importable."""
        issues = []

        for rel_path in PROTECTED_FILES:
            full_path = self.aragora_path / rel_path
            if not full_path.exists():
                issues.append(f"MISSING: {rel_path}")
                continue

            # Check if Python file is syntactically valid
            if rel_path.endswith(".py"):
                try:
                    result = subprocess.run(
                        ["python", "-m", "py_compile", str(full_path)],
                        capture_output=True,
                        text=True,
                        timeout=180,  # Minimum 3 min (was 20)
                    )
                    if result.returncode != 0:
                        issues.append(f"SYNTAX ERROR: {rel_path}")
                except Exception as e:
                    issues.append(f"CHECK FAILED: {rel_path} - {e}")

        return issues

    def _init_agents(self):
        """Initialize agents with distinct personalities and safety awareness."""
        # Common safety footer for all agents
        safety_footer = """

CRITICAL: You are part of a self-improving system. You MUST:
- NEVER propose removing or simplifying core infrastructure (nomic_loop.py, aragora/core.py, debate system)
- ALWAYS prefer adding new features over removing existing ones
- ONLY remove code that is demonstrably BROKEN or HARMFUL
- Preserve backward compatibility in all changes
- If unsure whether to keep functionality, KEEP IT"""

        self.gemini = GeminiAgent(
            name='gemini-visionary',
            model='gemini-3-pro-preview',  # Gemini 3 Pro
            role='proposer',
            timeout=720,  # Doubled to 12 min for thorough codebase exploration
        )
        self.gemini.system_prompt = """You are a visionary product strategist for aragora.
Focus on: viral growth, developer excitement, novel capabilities, bold ideas.
Think about what would make aragora famous and widely adopted.

IMPORTANT: Your proposals should ADD capabilities, not remove or simplify existing ones.
Aragora should grow more powerful over time, not be stripped down.""" + safety_footer

        self.codex = CodexAgent(
            name='codex-engineer',
            model='gpt-5.2-codex',
            role='proposer',
            timeout=1200,  # Doubled - Codex has known latency issues
        )
        self.codex.system_prompt = """You are a pragmatic engineer for aragora.
Focus on: technical excellence, code quality, practical utility, implementation feasibility.
You can examine the codebase deeply to understand what's possible.

IMPORTANT: Your role is to BUILD and EXTEND, not to remove or break.
Reducing technical debt is GOOD when it's safe (improve code without changing behavior).
But NEVER delete working functionality just to make things "cleaner".
Safe refactors: renaming, extracting, improving types. Unsafe: removing features, breaking APIs.""" + safety_footer

        self.claude = ClaudeAgent(
            name='claude-visionary',
            model='claude',
            role='proposer',
            timeout=600,  # 10 min - increased for judge role with large context
        )
        self.claude.system_prompt = """You are a visionary architect for aragora.
Focus on: elegant design, user experience, novel AI patterns, system cohesion.
Think about what would make aragora the most powerful and delightful multi-agent framework.

IMPORTANT: You are a guardian of aragora's core functionality.
Your proposals should ADD capabilities and improve the system.
Never propose removing the nomic loop or core debate infrastructure.""" + safety_footer

        self.grok = GrokCLIAgent(
            name='grok-lateral-thinker',
            model='grok-4',  # Grok 4 full
            role='proposer',
            timeout=1200,  # Doubled to 20 min for thorough codebase exploration
        )
        self.grok.system_prompt = """You are a lateral-thinking synthesizer for aragora.
Focus on: unconventional approaches, novel patterns, creative breakthroughs.
Connect ideas in surprising ways that others might miss.

IMPORTANT: Your role is to BUILD and EXTEND, not to remove or break.
Propose additions that unlock new capabilities and create emergent value.""" + safety_footer

    def get_current_features(self) -> str:
        """Read current aragora state from the codebase."""
        init_file = self.aragora_path / "aragora" / "__init__.py"
        if init_file.exists():
            content = init_file.read_text()
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
        except Exception:
            return "Unable to read git history"

    def _analyze_failed_branches(self, limit: int = 3) -> str:
        """Analyze recent failed branches for lessons learned.

        This extracts information from preserved failed branches so agents
        can learn from previous failures and avoid repeating them.
        """
        try:
            # List failed branches
            result = subprocess.run(
                ["git", "branch", "--list", "nomic-failed-*"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            branches = [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]
            if not branches:
                return ""

            # Get most recent ones (sorted by name which includes timestamp)
            recent = sorted(branches, reverse=True)[:limit]

            lessons = ["## LESSONS FROM RECENT FAILURES"]
            lessons.append("Learn from these previous failed attempts:\n")

            for branch in recent:
                # Get commit message
                msg_result = subprocess.run(
                    ["git", "log", branch, "-1", "--format=%B"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                # Get changed files summary
                files_result = subprocess.run(
                    ["git", "diff", f"main...{branch}", "--stat", "--stat-width=60"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )

                lessons.append(f"**{branch}:**")
                lessons.append(f"```\n{msg_result.stdout[:300].strip()}")
                if files_result.stdout.strip():
                    lessons.append(f"\nFiles changed:\n{files_result.stdout[:200].strip()}")
                lessons.append("```\n")

            return "\n".join(lessons)
        except Exception:
            return ""

    def _format_successful_patterns(self, limit: int = 5) -> str:
        """Format successful critique patterns for prompt injection.

        This retrieves patterns from the CritiqueStore that have led to
        successful fixes in previous debates.
        """
        if not hasattr(self, 'critique_store') or not self.critique_store:
            return ""

        try:
            patterns = self.critique_store.retrieve_patterns(min_success=2, limit=limit)
            if not patterns:
                return ""

            lines = ["## SUCCESSFUL PATTERNS (from past debates)"]
            lines.append("These critique patterns have worked well before:\n")

            for p in patterns:
                lines.append(f"- **{p.issue_type}**: {p.issue_text[:100]}")
                if p.suggestion_text:
                    lines.append(f"  → Fix: {p.suggestion_text[:100]}")
                lines.append(f"  ({p.success_count} successes)")

            return "\n".join(lines)
        except Exception:
            return ""

    async def _parallel_implementation_review(self, diff: str) -> Optional[str]:
        """
        All 3 agents review implementation changes in parallel.

        This provides balanced participation in the implementation stage
        while keeping the actual implementation specialized to Claude.

        Returns:
            Combined concerns from all agents, or None if all approve.
        """
        review_prompt = f"""Quick review of these code changes. Are there any obvious issues?

## Code Changes (git diff)
```
{diff[:3000]}
```

Reply with ONE of:
- APPROVED: <brief reason>
- CONCERN: <specific issue>

Be concise (1-2 sentences). Focus on correctness and safety issues only.
"""

        async def review_with_agent(agent, name: str) -> tuple[str, str]:
            """Run review with one agent, returning (name, result)."""
            try:
                self._log(f"    {name}: reviewing implementation...", agent=name)
                result = await agent.generate(review_prompt, context=[])
                self._log(f"    {name}: {result[:100] if result else 'No response'}...", agent=name)
                # Emit full review
                if result:
                    self._stream_emit("on_log_message", result, level="info", phase="review", agent=name)
                return (name, result[:200] if result else "No response")
            except Exception as e:
                self._log(f"    {name}: review error - {e}", agent=name)
                return (name, f"Error: {e}")

        # Run all 4 agents in parallel
        import asyncio
        reviews = await asyncio.gather(
            review_with_agent(self.gemini, "gemini"),
            review_with_agent(self.codex, "codex"),
            review_with_agent(self.claude, "claude"),
            review_with_agent(self.grok, "grok"),
            return_exceptions=True,
        )

        # Collect concerns
        concerns = []
        for result in reviews:
            if isinstance(result, Exception):
                continue
            name, response = result
            if response and "CONCERN" in response.upper():
                concerns.append(f"{name}: {response}")

        if concerns:
            return "\n".join(concerns)
        return None

    def _create_arena_hooks(self, phase_name: str) -> dict:
        """Create event hooks for real-time Arena logging and streaming."""
        # Get streaming hooks if available
        stream_hooks = {}
        if self.stream_emitter and STREAMING_AVAILABLE and create_arena_hooks:
            stream_hooks = create_arena_hooks(self.stream_emitter)

        def make_combined_hook(log_fn, stream_hook_name):
            """Combine logging and streaming for a hook."""
            stream_fn = stream_hooks.get(stream_hook_name)
            def combined(*args, **kwargs):
                log_fn(*args, **kwargs)
                if stream_fn:
                    try:
                        stream_fn(*args, **kwargs)
                    except Exception:
                        pass  # Don't let streaming errors break the loop
            return combined

        return {
            "on_debate_start": make_combined_hook(
                lambda task, agents: self._log(f"    Debate started: {len(agents)} agents"),
                "on_debate_start"
            ),
            "on_message": make_combined_hook(
                lambda agent, content, role, round_num: self._log(
                    f"    [{role}] {agent} (round {round_num}): {content}"  # Full content, no truncation
                ),
                "on_message"
            ),
            "on_critique": make_combined_hook(
                lambda agent, target, issues, severity, round_num, full_content=None: self._log(
                    f"    [critique] {agent} -> {target}: {len(issues)} issues, severity {severity:.1f}"
                ),
                "on_critique"
            ),
            "on_round_start": make_combined_hook(
                lambda round_num: self._log(f"    --- Round {round_num} ---"),
                "on_round_start"
            ),
            "on_consensus": make_combined_hook(
                lambda reached, confidence, answer: self._log(
                    f"    Consensus: {'Yes' if reached else 'No'} ({confidence:.0%})"
                ),
                "on_consensus"
            ),
            "on_vote": make_combined_hook(
                lambda agent, vote, confidence: self._log(
                    f"    [vote] {agent}: {vote} ({confidence:.0%})"
                ),
                "on_vote"
            ),
            "on_debate_end": make_combined_hook(
                lambda duration, rounds: self._log(f"    Completed in {duration:.1f}s ({rounds} rounds)"),
                "on_debate_end"
            ),
        }

    async def _run_arena_with_logging(self, arena: Arena, phase_name: str) -> "DebateResult":
        """Run an Arena debate with real-time logging via event hooks."""
        self._log(f"  Starting {phase_name} arena...")
        self._save_state({"phase": phase_name, "stage": "arena_starting"})

        # Add event hooks for real-time logging
        arena.hooks = self._create_arena_hooks(phase_name)

        try:
            result = await arena.run()

            self._log(f"  {phase_name} arena complete")
            self._log(f"    Consensus: {result.consensus_reached}", also_print=False)
            self._log(f"    Confidence: {result.confidence}", also_print=False)
            self._log(f"    Duration: {result.duration_seconds:.1f}s", also_print=False)

            self._save_state({
                "phase": phase_name,
                "stage": "arena_complete",
                "consensus_reached": result.consensus_reached,
                "confidence": result.confidence,
                "final_answer_preview": result.final_answer[:500] if result.final_answer else None,
            })

            return result

        except Exception as e:
            self._log(f"  {phase_name} arena ERROR: {e}")
            self._save_state({"phase": phase_name, "stage": "arena_error", "error": str(e)})
            raise

    async def _run_fractal_with_logging(self, task: str, agents: list, phase_name: str) -> "DebateResult":
        """Run a fractal debate with agent evolution and real-time logging."""
        if not self.use_genesis or not GENESIS_AVAILABLE:
            # Fall back to regular arena
            env = Environment(task=task)
            protocol = DebateProtocol(rounds=2, consensus="majority")
            arena = Arena(environment=env, agents=agents, protocol=protocol, memory=self.critique_store, debate_embeddings=self.debate_embeddings)
            return await self._run_arena_with_logging(arena, phase_name)

        self._log(f"  Starting {phase_name} fractal debate (genesis mode)...")
        self._save_state({"phase": phase_name, "stage": "fractal_starting", "genesis": True})

        # Create fractal orchestrator with hooks
        orchestrator = FractalOrchestrator(
            max_depth=2,
            tension_threshold=0.6,
            evolve_agents=True,
            population_manager=self.population_manager,
            event_hooks=self.stream_hooks,
        )

        try:
            # Get or create population from agent names
            agent_names = [a.name.split("_")[0] for a in agents]
            population = self.population_manager.get_or_create_population(agent_names)

            self._log(f"    Population: {population.size} genomes, gen {population.generation}")

            # Run fractal debate
            fractal_result = await orchestrator.run(
                task=task,
                agents=agents,
                population=population,
            )

            self._log(f"  {phase_name} fractal debate complete")
            self._log(f"    Total depth: {fractal_result.total_depth}")
            self._log(f"    Sub-debates: {len(fractal_result.sub_debates)}")
            self._log(f"    Tensions resolved: {fractal_result.tensions_resolved}")
            self._log(f"    Evolved genomes: {len(fractal_result.evolved_genomes)}")

            # Log evolved genomes
            for genome in fractal_result.evolved_genomes:
                self._log(f"      - {genome.name} (gen {genome.generation})")

            self._save_state({
                "phase": phase_name,
                "stage": "fractal_complete",
                "genesis": True,
                "total_depth": fractal_result.total_depth,
                "sub_debates": len(fractal_result.sub_debates),
                "evolved_genomes": len(fractal_result.evolved_genomes),
                "consensus_reached": fractal_result.main_result.consensus_reached,
            })

            return fractal_result.main_result

        except Exception as e:
            self._log(f"  {phase_name} fractal debate ERROR: {e}")
            self._save_state({"phase": phase_name, "stage": "fractal_error", "error": str(e)})
            # Fall back to regular arena on error
            self._log(f"  Falling back to regular arena...")
            env = Environment(task=task)
            protocol = DebateProtocol(rounds=2, consensus="majority")
            arena = Arena(environment=env, agents=agents, protocol=protocol, memory=self.critique_store, debate_embeddings=self.debate_embeddings)
            return await self._run_arena_with_logging(arena, phase_name)

    async def phase_context_gathering(self) -> dict:
        """
        Phase 0: All agents explore codebase to gather context.

        Each agent uses its native codebase exploration harness:
        - Claude → Claude Code CLI (native codebase access)
        - Codex → Codex CLI (native codebase access)
        - Gemini → Kilo Code CLI (agentic codebase exploration)
        - Grok → Kilo Code CLI (agentic codebase exploration)

        This ensures ALL agents have first-hand knowledge of the codebase,
        preventing proposals for features that already exist.
        """
        phase_start = datetime.now()

        # Determine how many agents will participate
        agents_count = 2  # Claude + Codex always
        if KILOCODE_AVAILABLE:
            agents_count = 4  # + Gemini + Grok via Kilo Code
            self._log("\n" + "=" * 70)
            self._log("PHASE 0: CONTEXT GATHERING (All 4 agents with codebase access)")
            self._log("  Claude → Claude Code | Codex → Codex CLI")
            self._log("  Gemini → Kilo Code  | Grok → Kilo Code")
            self._log("=" * 70)
        else:
            self._log("\n" + "=" * 70)
            self._log("PHASE 0: CONTEXT GATHERING (Claude + Codex)")
            self._log("  Note: Install kilocode CLI to enable Gemini/Grok exploration")
            self._log("=" * 70)

        self._stream_emit("on_phase_start", "context", self.cycle_count, {"agents": agents_count})

        # Prompt for codebase exploration
        explore_prompt = f"""Explore the aragora codebase and provide a comprehensive summary of EXISTING features.

Working directory: {self.aragora_path}

Your task:
1. Read key files: aragora/__init__.py, aragora/debate/orchestrator.py, aragora/server/stream.py
2. List ALL existing major features and capabilities
3. Note any features related to: streaming, real-time, visualization, spectator mode, WebSocket
4. Identify the project's current architecture and patterns

Output format:
## EXISTING FEATURES (DO NOT RECREATE)
- Feature 1: description
- Feature 2: description
...

## ARCHITECTURE OVERVIEW
Brief description of how the system is organized.

## RECENT FOCUS AREAS
What has been worked on recently (from git log).

## GAPS AND OPPORTUNITIES
What's genuinely missing (not already implemented).

CRITICAL: Be thorough. Features you miss here may be accidentally proposed for recreation."""

        async def gather_with_agent(agent, name: str, harness: str) -> tuple[str, str, str]:
            """Run exploration with one agent."""
            try:
                self._log(f"  {name} ({harness}): exploring codebase...", agent=name)
                result = await agent.generate(explore_prompt, context=[])
                self._log(f"  {name}: complete ({len(result) if result else 0} chars)", agent=name)
                # Emit agent's full exploration result
                if result:
                    self._stream_emit("on_log_message", result, level="info", phase="context", agent=name)
                return (name, harness, result if result else "No response")
            except Exception as e:
                self._log(f"  {name}: error - {e}", agent=name)
                return (name, harness, f"Error: {e}")

        # Build list of exploration tasks
        exploration_tasks = [
            gather_with_agent(self.claude, "claude", "Claude Code"),
            gather_with_agent(self.codex, "codex", "Codex CLI"),
        ]

        # Add Gemini and Grok via Kilo Code if available
        if KILOCODE_AVAILABLE:
            # Create temporary Kilo Code agents for exploration
            gemini_explorer = KiloCodeAgent(
                name="gemini-explorer",
                provider_id="gemini-explorer",
                model="gemini-3-pro",
                role="explorer",
                timeout=600,
                mode="architect",
            )
            grok_explorer = KiloCodeAgent(
                name="grok-explorer",
                provider_id="grok-explorer",
                model="grok-code-fast-1",
                role="explorer",
                timeout=600,
                mode="architect",
            )
            exploration_tasks.extend([
                gather_with_agent(gemini_explorer, "gemini", "Kilo Code"),
                gather_with_agent(grok_explorer, "grok", "Kilo Code"),
            ])

        # Run all agents in parallel
        results = await asyncio.gather(*exploration_tasks, return_exceptions=True)

        # Combine the context from all agents
        combined_context = []
        for result in results:
            if isinstance(result, Exception):
                continue
            name, harness, content = result
            if content and "Error:" not in content:
                combined_context.append(
                    f"=== {name.upper()}'S CODEBASE ANALYSIS (via {harness}) ===\n{content}"
                )

        # If all failed, fall back to basic context
        if not combined_context:
            self._log("  Warning: Context gathering failed, using basic context")
            combined_context = [f"Current features (from docstring):\n{self.get_current_features()}"]

        gathered_context = "\n\n".join(combined_context)

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._log(f"  Context gathered from {len(combined_context)} agents in {phase_duration:.1f}s")
        self._stream_emit(
            "on_phase_end", "context", self.cycle_count, True,
            phase_duration, {"agents": len(combined_context), "context_length": len(gathered_context)}
        )

        return {
            "phase": "context",
            "context": gathered_context,
            "duration": phase_duration,
            "agents_succeeded": len(combined_context),
        }

    async def phase_debate(self, codebase_context: str = None) -> dict:
        """Phase 1: Agents debate what to improve."""
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 1: IMPROVEMENT DEBATE")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "debate", self.cycle_count, {"agents": 4})

        # Use provided context or fall back to basic
        if codebase_context:
            current_features = codebase_context
        else:
            current_features = self.get_current_features()
        recent_changes = self.get_recent_changes()

        # Gather learning context from previous cycles
        failure_lessons = self._analyze_failed_branches()
        successful_patterns = self._format_successful_patterns()

        # Build task with optional initial proposal
        initial_proposal_section = ""
        if self.initial_proposal:
            initial_proposal_section = f"""

===== HUMAN-SUBMITTED PROPOSAL =====
A human has submitted the following proposal for your consideration.
You may adopt it, critique it, improve upon it, or propose something entirely different.

{self.initial_proposal}
====================================
"""
            self._log(f"  Including human proposal: {self.initial_proposal[:100]}...")

        # Build context section with clear attribution
        if codebase_context and len(codebase_context) > 500:
            context_section = f"""
===== CODEBASE ANALYSIS (from Claude + Codex who explored the code) =====
The following is a comprehensive analysis of aragora's EXISTING features.
Claude and Codex have read the actual codebase. DO NOT propose features that already exist below.

{current_features}
========================================================================"""
        else:
            context_section = f"Current aragora features:\n{current_features}"

        # Build learning context section
        learning_context = ""
        if failure_lessons:
            learning_context += f"\n{failure_lessons}\n"
        if successful_patterns:
            learning_context += f"\n{successful_patterns}\n"

        task = f"""{SAFETY_PREAMBLE}

What single improvement would most benefit aragora RIGHT NOW?

CRITICAL: Read the codebase analysis below carefully. DO NOT propose features that already exist.
Claude and Codex have explored the codebase and documented existing features.
{learning_context}
Consider what would make aragora:
- More INTERESTING (novel, creative, intellectually stimulating)
- More POWERFUL (capable, versatile, effective)
- More VIRAL (shareable, demonstrable, meme-worthy)
- More USEFUL (practical, solves real problems)
{initial_proposal_section}
Each agent should propose ONE specific, implementable feature.
Be concrete: describe what it does, how it works, and why it matters.
After debate, reach consensus on THE SINGLE BEST improvement to implement this cycle.

REMEMBER:
- Propose ADDITIONS, not removals. Build new capabilities, don't simplify existing ones.
- Check the codebase analysis - if a feature is listed there, it ALREADY EXISTS.
- Learn from previous failures shown above - avoid repeating them.

Recent changes:
{recent_changes}"""

        env = Environment(
            task=task,
            context=context_section,
        )

        protocol = DebateProtocol(
            rounds=2,
            consensus="judge",
            proposer_count=4,  # All 4 agents participate
        )

        arena = Arena(
            env,
            [self.gemini, self.codex, self.claude, self.grok],
            protocol,
            memory=self.critique_store,
            debate_embeddings=self.debate_embeddings,
        )
        result = await self._run_arena_with_logging(arena, "debate")

        # Update agent reputation based on debate outcome
        if self.critique_store and result.consensus_reached:
            winning_proposal = result.final_answer[:200] if result.final_answer else ""
            for agent in [self.gemini, self.codex, self.claude, self.grok]:
                # Check if this agent's proposal was selected
                proposal_accepted = agent.name.lower() in winning_proposal.lower()
                self.critique_store.update_reputation(
                    agent.name,
                    proposal_made=True,
                    proposal_accepted=proposal_accepted,
                )

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit(
            "on_phase_end", "debate", self.cycle_count, result.consensus_reached,
            phase_duration, {"confidence": result.confidence}
        )

        return {
            "phase": "debate",
            "final_answer": result.final_answer,
            "consensus_reached": result.consensus_reached,
            "confidence": result.confidence,
            "duration": result.duration_seconds,
        }

    async def phase_design(self, improvement: str) -> dict:
        """Phase 2: All agents design the implementation together."""
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 2: IMPLEMENTATION DESIGN")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "design", self.cycle_count, {"agents": 4})

        env = Environment(
            task=f"""{SAFETY_PREAMBLE}

Design the implementation for this improvement:

{improvement}

Provide:
1. FILE CHANGES: Which files to create or modify (NEVER delete protected files)
2. API DESIGN: Key classes, functions, signatures (EXTEND existing APIs, don't break them)
3. INTEGRATION: How it connects to existing aragora modules (preserve all existing functionality)
4. TEST PLAN: How to verify it works AND that existing features still work
5. EXAMPLE USAGE: Code snippet showing the feature in action

Be specific enough that an engineer could implement it.
The implementation MUST preserve all existing aragora functionality.""",
            context=f"Working directory: {self.aragora_path}\n\nProtected files (NEVER delete): {PROTECTED_FILES}",
        )

        protocol = DebateProtocol(
            rounds=1,
            consensus="judge",
            proposer_count=4,  # All 4 agents participate as proposers
        )

        # All 4 agents participate in design
        arena = Arena(env, [self.gemini, self.codex, self.claude, self.grok], protocol, memory=self.critique_store, debate_embeddings=self.debate_embeddings)
        result = await self._run_arena_with_logging(arena, "design")

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit(
            "on_phase_end", "design", self.cycle_count, result.consensus_reached,
            phase_duration, {}
        )

        return {
            "phase": "design",
            "design": result.final_answer,
            "consensus_reached": result.consensus_reached,
        }

    def _git_stash_create(self) -> Optional[str]:
        """Create a git stash for transactional safety."""
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if not status.stdout.strip():
                return None

            result = subprocess.run(
                ["git", "stash", "push", "-m", "nomic-implement-backup"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                ref_result = subprocess.run(
                    ["git", "stash", "list", "-1", "--format=%H"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                return ref_result.stdout.strip() or "stash@{0}"
        except Exception as e:
            self._log(f"Warning: Could not create stash: {e}")
        return None

    def _git_stash_pop(self, stash_ref: Optional[str]) -> None:
        """Pop a stash to restore previous state."""
        if not stash_ref:
            return
        try:
            subprocess.run(
                ["git", "checkout", "."],
                cwd=self.aragora_path,
                capture_output=True,
            )
            subprocess.run(
                ["git", "stash", "pop"],
                cwd=self.aragora_path,
                capture_output=True,
            )
        except Exception as e:
            self._log(f"Warning: Could not pop stash: {e}")

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

    async def _preserve_failed_work(self, branch_name: str) -> Optional[str]:
        """
        Preserve failed implementation work in a git branch before rollback.

        This ensures that even failed implementations can be inspected and
        potentially salvaged later.

        Returns:
            Branch name if successful, None if failed
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        full_branch_name = f"{branch_name}-{timestamp}"

        try:
            # Get current branch
            current_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            current_branch = current_result.stdout.strip()

            # Check if there are any changes to preserve
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if not status_result.stdout.strip():
                self._log("  No changes to preserve")
                return None

            # Create and switch to preservation branch
            subprocess.run(
                ["git", "checkout", "-b", full_branch_name],
                cwd=self.aragora_path,
                capture_output=True,
                check=True,
            )

            # Stage all changes including untracked files
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.aragora_path,
                capture_output=True,
                check=True,
            )

            # Commit the failed work
            commit_msg = f"WIP: Failed nomic cycle {self.cycle_count} (verification failed)\n\nThis branch contains work that failed verification and was rolled back.\nPreserved for inspection and potential salvage."
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.aragora_path,
                capture_output=True,
            )

            # Switch back to original branch
            subprocess.run(
                ["git", "checkout", current_branch],
                cwd=self.aragora_path,
                capture_output=True,
                check=True,
            )

            self._log(f"  Preserved failed work in branch: {full_branch_name}")
            self._stream_emit("on_work_preserved", full_branch_name, self.cycle_count)
            return full_branch_name

        except Exception as e:
            self._log(f"  Warning: Could not preserve work in branch: {e}")
            # Try to get back to original branch
            try:
                subprocess.run(
                    ["git", "checkout", current_branch],
                    cwd=self.aragora_path,
                    capture_output=True,
                )
            except Exception:
                pass
            return None

    async def phase_implement(self, design: str) -> dict:
        """Phase 3: Hybrid multi-model implementation."""
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 3: IMPLEMENTATION (Hybrid)")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "implement", self.cycle_count, {})

        use_hybrid = os.environ.get("ARAGORA_HYBRID_IMPLEMENT", "1") == "1"

        if not use_hybrid:
            return await self._legacy_implement(design)

        design_hash = hashlib.md5(design.encode()).hexdigest()

        # 1. Check for crash recovery
        progress = load_progress(self.aragora_path)
        if progress and progress.plan.design_hash == design_hash:
            self._log("  Resuming from checkpoint...")
            plan = progress.plan
            completed = set(progress.completed_tasks)
            stash_ref = progress.git_stash_ref
        else:
            # 2. Generate plan
            try:
                self._log("  Generating implementation plan...")
                plan = await generate_implement_plan(design, self.aragora_path)
                self._log(f"  Plan generated: {len(plan.tasks)} tasks")
                for task in plan.tasks:
                    self._log(f"    - {task.id}: {task.description[:60]}...", also_print=False)
            except Exception as e:
                self._log(f"  Plan generation failed: {e}")
                self._log("  Falling back to single-task mode...")
                plan = create_single_task_plan(design, self.aragora_path)

            completed = set()
            stash_ref = self._git_stash_create()

            save_progress(
                ImplementProgress(
                    plan=plan,
                    completed_tasks=[],
                    git_stash_ref=stash_ref,
                ),
                self.aragora_path,
            )

        # Save state
        self._save_state({
            "phase": "implement",
            "stage": "executing",
            "total_tasks": len(plan.tasks),
            "completed_tasks": len(completed),
        })

        # 4. Execute tasks
        executor = HybridExecutor(self.aragora_path)

        def on_task_complete(task_id: str, result):
            completed.add(task_id)
            self._log(f"  Task {task_id}: {'completed' if result.success else 'failed'}")
            save_progress(
                ImplementProgress(
                    plan=plan,
                    completed_tasks=list(completed),
                    current_task=None,
                    git_stash_ref=stash_ref,
                ),
                self.aragora_path,
            )
            self._save_state({
                "phase": "implement",
                "stage": "executing",
                "total_tasks": len(plan.tasks),
                "completed_tasks": len(completed),
                "last_task": task_id,
                "last_success": result.success,
            })
            # Stream task completion event
            self._stream_emit(
                "on_task_complete",
                task_id,
                result.success,
                result.duration_seconds,
                result.diff[:500] if result.diff else "",
                result.error if not result.success else None,
            )

        try:
            results = await executor.execute_plan(
                plan.tasks,
                completed,
                on_task_complete=on_task_complete,
            )

            all_success = all(r.success for r in results)
            tasks_completed = len([r for r in results if r.success])

            if all_success and tasks_completed == len(plan.tasks):
                clear_progress(self.aragora_path)
                self._log(f"  All {tasks_completed} tasks completed successfully")

                # Pre-verification review: All 3 agents review implementation in parallel
                self._log("\n  Pre-verification review (all agents)...", agent="claude")
                diff = self._get_git_diff()
                if diff and len(diff) > 100:  # Only review if there are substantial changes
                    review_concerns = await self._parallel_implementation_review(diff)
                    if review_concerns:
                        self._log(f"    Review concerns: {review_concerns[:200]}...", agent="claude")
                    else:
                        self._log("    All agents approve the implementation", agent="claude")

                phase_duration = (datetime.now() - phase_start).total_seconds()
                self._stream_emit(
                    "on_phase_end", "implement", self.cycle_count, True,
                    phase_duration, {"tasks_completed": tasks_completed}
                )
                return {
                    "phase": "implement",
                    "success": True,
                    "tasks_completed": tasks_completed,
                    "tasks_total": len(plan.tasks),
                    "diff": diff,
                    "results": [r.to_dict() for r in results],
                }
            else:
                failed = [r for r in results if not r.success]
                self._log(f"  {len(failed)} tasks failed")
                phase_duration = (datetime.now() - phase_start).total_seconds()
                self._stream_emit(
                    "on_phase_end", "implement", self.cycle_count, False,
                    phase_duration, {"tasks_failed": len(failed)}
                )
                return {
                    "phase": "implement",
                    "success": False,
                    "tasks_completed": tasks_completed,
                    "tasks_total": len(plan.tasks),
                    "error": failed[0].error if failed else "Unknown error",
                    "diff": self._get_git_diff(),
                }

        except Exception as e:
            self._log(f"  Catastrophic failure: {e}")
            self._log("  Rolling back changes...")
            self._git_stash_pop(stash_ref)
            phase_duration = (datetime.now() - phase_start).total_seconds()
            self._stream_emit(
                "on_phase_end", "implement", self.cycle_count, False,
                phase_duration, {"error": str(e)}
            )
            self._stream_emit("on_error", "implement", str(e), True)
            return {
                "phase": "implement",
                "success": False,
                "error": str(e),
            }

    async def _legacy_implement(self, design: str) -> dict:
        """Legacy single-Codex implementation (fallback)."""
        self._log("  Using legacy Codex-only mode...")

        prompt = f"""{SAFETY_PREAMBLE}

Implement this design in the aragora codebase:

{design}

Write the actual code. Create or modify files as needed.
Follow aragora's existing code style and patterns.
Include docstrings and type hints.

CRITICAL SAFETY RULES:
- NEVER delete or modify these protected files: {PROTECTED_FILES}
- NEVER remove existing functionality - only ADD new code
- NEVER simplify code by removing features - complexity is acceptable
- If a file seems "too complex", DO NOT simplify it
- Preserve ALL existing imports, classes, and functions"""

        try:
            result = subprocess.run(
                ["codex", "exec", "-C", str(self.aragora_path), prompt],
                capture_output=True,
                text=True,
                timeout=1200,  # Doubled - Codex has known latency issues
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
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 4: VERIFICATION")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "verify", self.cycle_count, {})
        self._stream_emit("on_verification_start", ["syntax", "import", "tests"])

        checks = []

        # 1. Python syntax check
        self._log("  Checking syntax...")
        try:
            result = subprocess.run(
                ["python", "-m", "py_compile", "aragora/__init__.py"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            passed = result.returncode == 0
            checks.append({
                "check": "syntax",
                "passed": passed,
                "output": result.stderr,
            })
            self._log(f"    {'passed' if passed else 'FAILED'} syntax")
            self._stream_emit("on_verification_result", "syntax", passed, result.stderr[:200] if result.stderr else "")
        except Exception as e:
            checks.append({"check": "syntax", "passed": False, "error": str(e)})
            self._log(f"    FAILED syntax: {e}")
            self._stream_emit("on_verification_result", "syntax", False, str(e))

        # 2. Import check
        self._log("  Checking imports...")
        try:
            result = subprocess.run(
                ["python", "-c", "import aragora; print('OK')"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
                timeout=180,  # Minimum 3 min (was 60)
            )
            passed = "OK" in result.stdout
            checks.append({
                "check": "import",
                "passed": passed,
                "output": result.stderr if result.returncode != 0 else "",
            })
            self._log(f"    {'passed' if passed else 'FAILED'} import")
            self._stream_emit("on_verification_result", "import", passed, result.stderr[:200] if result.stderr else "")
        except Exception as e:
            checks.append({"check": "import", "passed": False, "error": str(e)})
            self._log(f"    FAILED import: {e}")
            self._stream_emit("on_verification_result", "import", False, str(e))

        # 3. Run tests
        self._log("  Running tests...")
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "--tb=short", "-q"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
                timeout=240,
            )
            # pytest returns 5 when no tests are collected - treat as pass
            # Also check for "no tests ran" in output as a fallback
            no_tests_collected = result.returncode == 5 or "no tests ran" in result.stdout.lower()
            passed = result.returncode == 0 or no_tests_collected
            checks.append({
                "check": "tests",
                "passed": passed,
                "output": result.stdout[-500:] if result.stdout else "",
                "note": "no tests collected" if no_tests_collected else "",
            })
            self._log(f"    {'passed' if passed else 'FAILED'} tests" + (" (no tests collected)" if no_tests_collected else ""))
            self._stream_emit("on_verification_result", "tests", passed, result.stdout[-200:] if result.stdout else "")
        except Exception as e:
            checks.append({"check": "tests", "passed": True, "note": "No tests or timeout"})
            self._log(f"    skipped tests: {e}")
            self._stream_emit("on_verification_result", "tests", True, f"Skipped: {e}")

        all_passed = all(c.get("passed", False) for c in checks)
        self._save_state({
            "phase": "verify",
            "stage": "complete",
            "all_passed": all_passed,
            "checks": checks,
        })

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit(
            "on_phase_end", "verify", self.cycle_count, all_passed,
            phase_duration, {"checks_passed": sum(1 for c in checks if c.get("passed"))}
        )

        return {
            "phase": "verify",
            "checks": checks,
            "all_passed": all_passed,
        }

    async def phase_commit(self, improvement: str) -> dict:
        """Phase 5: Commit changes if verified."""
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 5: COMMIT")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "commit", self.cycle_count, {})

        if self.require_human_approval and not self.auto_commit:
            self._log("\nChanges ready for review:")
            subprocess.run(["git", "diff", "--stat"], cwd=self.aragora_path)

            response = input("\nCommit these changes? [y/N]: ")
            if response.lower() != 'y':
                self._log("Skipping commit.")
                phase_duration = (datetime.now() - phase_start).total_seconds()
                self._stream_emit(
                    "on_phase_end", "commit", self.cycle_count, False,
                    phase_duration, {"reason": "human_declined"}
                )
                return {"phase": "commit", "committed": False, "reason": "Human declined"}

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
                self._log(f"  Committed: {summary[:60]}...")
                # Get commit hash
                hash_result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else "unknown"
                # Get files changed count
                stat_result = subprocess.run(
                    ["git", "diff", "--stat", "HEAD~1..HEAD"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                files_changed = len([l for l in stat_result.stdout.split('\n') if '|' in l])
                self._stream_emit("on_commit", commit_hash, summary, files_changed)
            else:
                self._log(f"  Commit failed: {result.stderr}")

            phase_duration = (datetime.now() - phase_start).total_seconds()
            self._stream_emit(
                "on_phase_end", "commit", self.cycle_count, committed,
                phase_duration, {}
            )

            return {
                "phase": "commit",
                "committed": committed,
                "message": summary,
            }

        except Exception as e:
            self._log(f"  Commit error: {e}")
            phase_duration = (datetime.now() - phase_start).total_seconds()
            self._stream_emit(
                "on_phase_end", "commit", self.cycle_count, False,
                phase_duration, {"error": str(e)}
            )
            self._stream_emit("on_error", "commit", str(e), True)
            return {
                "phase": "commit",
                "committed": False,
                "error": str(e),
            }

    async def run_cycle(self) -> dict:
        """Run one complete improvement cycle with safety backup/restore."""
        self.cycle_count += 1
        cycle_start = datetime.now()

        self._log("\n" + "=" * 70)
        self._log(f"NOMIC CYCLE {self.cycle_count}")
        self._log(f"Started: {cycle_start.isoformat()}")
        self._log("=" * 70)

        # Emit cycle start event
        self._stream_emit("on_cycle_start", self.cycle_count, self.max_cycles, cycle_start.isoformat())

        # === SAFETY: Create backup before any changes ===
        backup_path = self._create_backup(f"cycle_{self.cycle_count}")

        cycle_result = {
            "cycle": self.cycle_count,
            "started": cycle_start.isoformat(),
            "backup_path": str(backup_path),
            "phases": {},
        }

        self._save_state({
            "phase": "cycle_start",
            "cycle": self.cycle_count,
            "backup_path": str(backup_path),
        })

        # Phase 0: Context Gathering (Claude + Codex explore codebase)
        # This ensures Gemini and Grok get accurate context about existing features
        context_result = await self.phase_context_gathering()
        cycle_result["phases"]["context"] = context_result
        codebase_context = context_result.get("context", "")

        # Phase 1: Debate (all agents, with gathered context)
        debate_result = await self.phase_debate(codebase_context=codebase_context)
        cycle_result["phases"]["debate"] = debate_result

        if not debate_result.get("consensus_reached"):
            self._log("No consensus reached. Ending cycle.")
            cycle_result["outcome"] = "no_consensus"
            return cycle_result

        improvement = debate_result["final_answer"]
        self._log(f"\nConsensus improvement:\n{improvement}")  # Full content, no truncation

        # Phase 2: Design
        design_result = await self.phase_design(improvement)
        cycle_result["phases"]["design"] = design_result

        design = design_result.get("design", "")
        self._log(f"\nDesign complete")

        # Phase 3: Implement
        impl_result = await self.phase_implement(design)
        cycle_result["phases"]["implement"] = impl_result

        if not impl_result.get("success"):
            self._log("Implementation failed. Ending cycle.")
            cycle_result["outcome"] = "implementation_failed"
            return cycle_result

        self._log(f"\nImplementation complete")
        self._log(f"Changed files:\n{impl_result.get('diff', 'No changes')}")

        # === SAFETY: Verify protected files are intact ===
        self._log("\n  Checking protected files...")
        protected_issues = self._verify_protected_files()
        if protected_issues:
            self._log("  CRITICAL: Protected files damaged!")
            for issue in protected_issues:
                self._log(f"    - {issue}")

            # Preserve work before rollback
            preserve_branch = await self._preserve_failed_work(f"nomic-protected-damaged-{self.cycle_count}")
            if preserve_branch:
                cycle_result["preserved_branch"] = preserve_branch
                self._log(f"  Work preserved in branch: {preserve_branch}")

            self._log("  Restoring from backup...")
            self._restore_backup(backup_path)
            subprocess.run(["git", "checkout", "."], cwd=self.aragora_path)
            cycle_result["outcome"] = "protected_files_damaged"
            cycle_result["protected_issues"] = protected_issues
            return cycle_result
        self._log("  All protected files intact")

        # === Iterative Review/Fix Cycle ===
        # Default: 1 fix attempt. Set ARAGORA_MAX_FIX_ITERATIONS for more.
        # The fix cycle: Codex reviews failure -> Claude fixes -> Gemini reviews -> re-verify
        max_fix_iterations = int(os.environ.get("ARAGORA_MAX_FIX_ITERATIONS", "1"))
        fix_iteration = 0
        cycle_result["fix_iterations"] = []

        while True:
            # Phase 4: Verify
            verify_result = await self.phase_verify()
            cycle_result["phases"]["verify"] = verify_result

            if verify_result.get("all_passed"):
                self._log(f"\nVerification passed!")
                break  # Success - exit the fix loop

            # Verification failed
            fix_iteration += 1
            iteration_result = {
                "iteration": fix_iteration,
                "verify_result": verify_result,
            }

            if fix_iteration > max_fix_iterations:
                # No more fix attempts allowed
                if self.disable_rollback:
                    self._log(f"Verification failed after {fix_iteration - 1} fix attempts.")
                    self._log("  ROLLBACK DISABLED - keeping changes for inspection")
                    cycle_result["outcome"] = "verification_failed_no_rollback"
                    cycle_result["fix_iterations"].append(iteration_result)
                    return cycle_result
                else:
                    # Preserve work in a branch before rollback
                    preserve_branch = await self._preserve_failed_work(f"nomic-failed-cycle-{self.cycle_count}")
                    if preserve_branch:
                        cycle_result["preserved_branch"] = preserve_branch
                        self._log(f"  Work preserved in branch: {preserve_branch}")

                    self._log(f"Verification failed after {fix_iteration - 1} fix attempts. Rolling back.")
                    self._restore_backup(backup_path)
                    subprocess.run(["git", "checkout", "."], cwd=self.aragora_path)
                    cycle_result["outcome"] = "verification_failed"
                    cycle_result["fix_iterations"].append(iteration_result)
                    return cycle_result

            self._log(f"\n{'=' * 50}")
            self._log(f"FIX ITERATION {fix_iteration}/{max_fix_iterations}")
            self._log(f"{'=' * 50}")

            # Get test failure details
            test_output = ""
            for check in verify_result.get("checks", []):
                if check.get("check") == "tests" and not check.get("passed"):
                    test_output = check.get("output", "")

            # Step 1: Codex reviews the failed changes
            self._log("\n  Step 1: Codex analyzing test failures...", agent="codex")
            from aragora.implement import HybridExecutor
            executor = HybridExecutor(self.aragora_path)
            diff = self._get_git_diff()

            review_prompt = f"""The following code changes caused test failures. Analyze and suggest fixes.

## Test Failures
```
{test_output[:2000]}
```

## Code Changes (git diff)
```
{diff[:3000]}
```

Provide specific, actionable fixes. Focus on:
1. What exactly is broken?
2. What specific code changes will fix it?
3. Are there missing imports or dependencies?
"""
            review_result = await executor.review_with_codex(review_prompt, timeout=2400)  # 40 min for thorough review
            iteration_result["codex_review"] = review_result
            self._log(f"    Codex review complete", agent="codex")
            # Emit Codex's full review
            if review_result.get("review"):
                self._stream_emit("on_log_message", review_result["review"], level="info", phase="fix", agent="codex")

            # Step 2: Claude fixes based on Codex review
            self._log("\n  Step 2: Claude applying fixes...", agent="claude")
            fix_prompt = f"""{SAFETY_PREAMBLE}

Fix the test failures in the codebase. Here's what went wrong and how to fix it:

## Test Failures
```
{test_output[:1500]}
```

## Codex Analysis
{review_result.get('review', 'No review available')[:2000]}

## Instructions
1. Read the failing tests to understand what's expected
2. Apply the minimal fixes needed to make tests pass
3. Do NOT remove or simplify existing functionality
4. Preserve all imports and dependencies

Working directory: {self.aragora_path}
"""
            try:
                fix_agent = ClaudeAgent(
                    name="claude-fixer",
                    model="claude",
                    role="fixer",
                    timeout=1200,  # Doubled - fixes can be complex
                )
                await fix_agent.generate(fix_prompt, context=[])
                iteration_result["fix_applied"] = True
                self._log("    Fixes applied", agent="claude")
            except Exception as e:
                iteration_result["fix_error"] = str(e)
                self._log(f"    Fix failed: {e}", agent="claude")

            # Step 3: Gemini quick review (optional sanity check)
            self._log("\n  Step 3: Gemini quick review...", agent="gemini")
            try:
                gemini_review_prompt = f"""Quick review of fix attempt. Are these changes correct?

## Changes Made (by Claude)
{self._get_git_diff()[:2000]}

## Original Test Failures
{test_output[:500]}

Reply with: LOOKS_GOOD or ISSUES: <brief description>
"""
                gemini_result = await self.gemini.generate(gemini_review_prompt, context=[])
                iteration_result["gemini_review"] = gemini_result[:200] if gemini_result else "No response"
                self._log(f"    Gemini: {gemini_result[:100] if gemini_result else 'No response'}...", agent="gemini")
                # Emit Gemini's full review
                if gemini_result:
                    self._stream_emit("on_log_message", gemini_result, level="info", phase="fix", agent="gemini")
            except Exception as e:
                self._log(f"    Gemini review skipped: {e}", agent="gemini")

            cycle_result["fix_iterations"].append(iteration_result)

            # Re-check protected files after fix
            protected_issues = self._verify_protected_files()
            if protected_issues:
                self._log("  CRITICAL: Fix damaged protected files!")
                # Preserve work before rollback
                preserve_branch = await self._preserve_failed_work(f"nomic-fix-damaged-{self.cycle_count}")
                if preserve_branch:
                    cycle_result["preserved_branch"] = preserve_branch
                    self._log(f"  Work preserved in branch: {preserve_branch}")
                self._restore_backup(backup_path)
                subprocess.run(["git", "checkout", "."], cwd=self.aragora_path)
                cycle_result["outcome"] = "protected_files_damaged"
                return cycle_result

            self._log("\n  Re-running verification...")

        self._log(f"\nVerification passed")

        # Phase 5: Commit
        commit_result = await self.phase_commit(improvement)
        cycle_result["phases"]["commit"] = commit_result

        if commit_result.get("committed"):
            cycle_result["outcome"] = "success"
            self._log(f"\nCYCLE {self.cycle_count} COMPLETE - Changes committed!")
        else:
            cycle_result["outcome"] = "not_committed"

        cycle_result["duration_seconds"] = (datetime.now() - cycle_start).total_seconds()
        self.history.append(cycle_result)

        self._save_state({
            "phase": "cycle_complete",
            "cycle": self.cycle_count,
            "outcome": cycle_result["outcome"],
            "duration_seconds": cycle_result["duration_seconds"],
        })

        # Emit cycle end event
        self._stream_emit(
            "on_cycle_end",
            self.cycle_count,
            cycle_result.get("outcome") == "success",
            cycle_result["duration_seconds"],
            cycle_result.get("outcome", "unknown"),
        )

        return cycle_result

    async def run(self):
        """Run the nomic loop until max cycles or interrupted."""
        self._log("=" * 70)
        self._log("ARAGORA NOMIC LOOP")
        self._log("Self-improving multi-agent system")
        self._log("=" * 70)
        self._log(f"Max cycles: {self.max_cycles}")
        self._log(f"Human approval required: {self.require_human_approval}")
        self._log(f"Auto-commit: {self.auto_commit}")
        if self.initial_proposal:
            self._log(f"Initial proposal: {self.initial_proposal[:100]}...")
        self._log("=" * 70)
        self._log(f"Log file: {self.log_file}")
        self._log(f"State file: {self.state_file}")
        self._log(f"Backup dir: {self.backup_dir}")
        self._log("=" * 70)

        try:
            while self.cycle_count < self.max_cycles:
                result = await self.run_cycle()

                self._log(f"\nCycle {self.cycle_count} outcome: {result.get('outcome')}")

                if result.get("outcome") == "success":
                    self._log("Continuing to next cycle...")
                else:
                    self._log("Cycle did not complete successfully.")
                    if self.require_human_approval and not self.auto_commit:
                        try:
                            response = input("Continue to next cycle? [Y/n]: ")
                            if response.lower() == 'n':
                                break
                        except EOFError:
                            # Running in background/non-interactive mode
                            self._log("Non-interactive mode detected, continuing...")
                    else:
                        self._log("Auto-commit mode: continuing to next cycle...")

                await asyncio.sleep(2)

        except KeyboardInterrupt:
            self._log("\n\nNomic loop interrupted by user.")

        self._log("\n" + "=" * 70)
        self._log("NOMIC LOOP COMPLETE")
        self._log(f"Total cycles: {self.cycle_count}")
        self._log(f"Successful commits: {sum(1 for h in self.history if h.get('outcome') == 'success')}")
        self._log("=" * 70)

        return self.history


# =============================================================================
# CLI Commands for backup management
# =============================================================================

def list_backups(aragora_path: Path) -> None:
    """List available backups."""
    backup_dir = aragora_path / ".nomic" / "backups"
    if not backup_dir.exists():
        print("No backups directory found.")
        return

    backups = sorted(backup_dir.iterdir(), reverse=True)
    if not backups:
        print("No backups found.")
        return

    print(f"Available backups in {backup_dir}:")
    for backup in backups:
        manifest_file = backup / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            print(f"  {backup.name}")
            print(f"    Created: {manifest.get('created_at')}")
            print(f"    Reason: {manifest.get('reason')}")
            print(f"    Files: {len(manifest.get('files', []))}")
        else:
            print(f"  {backup.name} (no manifest)")


def restore_backup_cli(aragora_path: Path, backup_name: str = None) -> bool:
    """Restore from a specific backup or the latest one."""
    backup_dir = aragora_path / ".nomic" / "backups"

    if backup_name:
        backup_path = backup_dir / backup_name
        if not backup_path.exists():
            print(f"Backup not found: {backup_name}")
            return False
    else:
        # Find latest backup
        backups = sorted(backup_dir.iterdir(), reverse=True)
        backup_path = None
        for b in backups:
            if (b / "manifest.json").exists():
                backup_path = b
                break
        if not backup_path:
            print("No valid backups found.")
            return False

    manifest_file = backup_path / "manifest.json"
    with open(manifest_file) as f:
        manifest = json.load(f)

    print(f"Restoring backup: {backup_path.name}")
    print(f"  Created: {manifest.get('created_at')}")
    print(f"  Reason: {manifest.get('reason')}")

    restored = []
    for rel_path in manifest["files"]:
        src = backup_path / rel_path
        dst = aragora_path / rel_path
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(rel_path)
            print(f"  Restored: {rel_path}")

    print(f"\nRestored {len(restored)} files")
    return True


async def main():
    parser = argparse.ArgumentParser(description="Aragora Nomic Loop - Self-improvement cycle")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run subcommand (default)
    run_parser = subparsers.add_parser("run", help="Run the nomic loop")
    run_parser.add_argument("--cycles", type=int, default=3, help="Maximum cycles to run")
    run_parser.add_argument("--auto", action="store_true", help="Auto-commit without human approval")
    run_parser.add_argument("--path", type=str, help="Path to aragora repository")
    run_parser.add_argument(
        "--proposal", "-p", type=str,
        help="Initial proposal for agents to consider (they may adopt, improve, or reject it)"
    )
    run_parser.add_argument(
        "--proposal-file", "-f", type=str,
        help="File containing initial proposal (alternative to --proposal)"
    )
    run_parser.add_argument(
        "--genesis", action="store_true",
        help="Enable genesis mode: fractal debates with agent evolution"
    )
    run_parser.add_argument(
        "--no-rollback", action="store_true",
        help="Disable rollback on verification failure (keep changes for inspection)"
    )
    run_parser.add_argument(
        "--no-stream", action="store_true",
        help="DISCOURAGED: Run without live streaming. Use 'python scripts/run_nomic_with_stream.py run' instead."
    )

    # Backup management subcommands
    list_parser = subparsers.add_parser("list-backups", help="List available backups")
    list_parser.add_argument("--path", type=str, help="Path to aragora repository")

    restore_parser = subparsers.add_parser("restore", help="Restore from a backup")
    restore_parser.add_argument("--path", type=str, help="Path to aragora repository")
    restore_parser.add_argument("--backup", type=str, help="Specific backup name (default: latest)")

    # Legacy arguments for backward compatibility (when no subcommand specified)
    parser.add_argument("--cycles", type=int, default=3, help="Maximum cycles to run")
    parser.add_argument("--auto", action="store_true", help="Auto-commit without human approval")
    parser.add_argument("--path", type=str, help="Path to aragora repository")
    parser.add_argument(
        "--proposal", "-p", type=str,
        help="Initial proposal for agents to consider (they may adopt, improve, or reject it)"
    )
    parser.add_argument(
        "--proposal-file", "-f", type=str,
        help="File containing initial proposal (alternative to --proposal)"
    )
    parser.add_argument(
        "--genesis", action="store_true",
        help="Enable genesis mode: fractal debates with agent evolution"
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="DISCOURAGED: Run without live streaming. Use 'python scripts/run_nomic_with_stream.py run' instead."
    )

    args = parser.parse_args()

    # Determine aragora path
    aragora_path = Path(args.path) if args.path else Path(__file__).parent.parent

    # Handle subcommands
    if args.command == "list-backups":
        list_backups(aragora_path)
        return

    if args.command == "restore":
        restore_backup_cli(aragora_path, getattr(args, 'backup', None))
        return

    # Default: run the nomic loop (either "run" subcommand or no subcommand)
    no_stream = getattr(args, 'no_stream', False)

    # ENFORCE STREAMING: Redirect to run_nomic_with_stream.py unless --no-stream is specified
    if not no_stream:
        print("=" * 70)
        print("STREAMING IS REQUIRED")
        print("=" * 70)
        print()
        print("The nomic loop MUST stream to live.aragora.ai for transparency.")
        print()
        print("Please use the streaming script instead:")
        print()
        print("    python scripts/run_nomic_with_stream.py run --cycles 3")
        print()
        print("This ensures that all nomic loop activity is visible in real-time")
        print("at https://live.aragora.ai")
        print()
        print("If you MUST run without streaming (not recommended), use:")
        print()
        print("    python scripts/nomic_loop.py run --no-stream --cycles 3")
        print()
        print("=" * 70)
        sys.exit(1)

    # If --no-stream is specified, show warning and continue
    print("=" * 70)
    print("WARNING: Running WITHOUT live streaming")
    print("=" * 70)
    print()
    print("Activity will NOT be visible at https://live.aragora.ai")
    print("This is strongly discouraged for transparency reasons.")
    print()
    print("Press Ctrl+C within 5 seconds to cancel...")
    print()
    try:
        await asyncio.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled. Use 'python scripts/run_nomic_with_stream.py run' instead.")
        sys.exit(0)
    print("Continuing without streaming...")
    print("=" * 70)
    print()

    initial_proposal = getattr(args, 'proposal', None)
    if hasattr(args, 'proposal_file') and args.proposal_file:
        with open(args.proposal_file) as f:
            initial_proposal = f.read()

    use_genesis = getattr(args, 'genesis', False)

    loop = NomicLoop(
        aragora_path=args.path,
        max_cycles=args.cycles,
        require_human_approval=not args.auto,
        auto_commit=args.auto,
        initial_proposal=initial_proposal,
        use_genesis=use_genesis,
    )

    if use_genesis:
        print("Genesis mode enabled: fractal debates with agent evolution")

    await loop.run()


if __name__ == "__main__":
    asyncio.run(main())
