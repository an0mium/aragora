"""
Context gathering phase for nomic loop.

Phase 0: Gather codebase understanding
- All agents explore codebase to gather context
- Each agent uses its native codebase exploration harness
- Prevents proposals for features that already exist
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from . import ContextResult

# Optional metrics recording (imported lazily to avoid circular imports)
_metrics_recorder: Optional[Callable[[str, str, float], None]] = None
_agent_metrics_recorder: Optional[Callable[[str, str, float], None]] = None


def set_metrics_recorder(
    phase_recorder: Optional[Callable[[str, str, float], None]] = None,
    agent_recorder: Optional[Callable[[str, str, float], None]] = None,
) -> None:
    """Set the metrics recorder callbacks for profiling.

    Args:
        phase_recorder: Callback(phase, outcome, duration_seconds)
        agent_recorder: Callback(phase, agent, duration_seconds)
    """
    global _metrics_recorder, _agent_metrics_recorder
    _metrics_recorder = phase_recorder
    _agent_metrics_recorder = agent_recorder


class ContextPhase:
    """
    Handles context gathering from multiple agents.

    Each agent uses its native codebase exploration harness:
    - Claude → Claude Code CLI (native codebase access)
    - Codex → Codex CLI (native codebase access)
    - Gemini → Kilo Code CLI (agentic codebase exploration)
    - Grok → Kilo Code CLI (agentic codebase exploration)
    """

    def __init__(
        self,
        aragora_path: Path,
        claude_agent: Any,
        codex_agent: Any,
        kilocode_available: bool = False,
        skip_kilocode: bool = False,
        kilocode_agent_factory: Optional[Callable[..., Any]] = None,
        cycle_count: int = 0,
        log_fn: Optional[Callable[[str], None]] = None,
        stream_emit_fn: Optional[Callable[..., None]] = None,
        get_features_fn: Optional[Callable[[], str]] = None,
    ):
        """
        Initialize the context gathering phase.

        Args:
            aragora_path: Path to the aragora project root
            claude_agent: Claude agent instance
            codex_agent: Codex agent instance
            kilocode_available: Whether KiloCode is available
            skip_kilocode: Whether to skip KiloCode for context gathering
            kilocode_agent_factory: Factory to create KiloCode agents
            cycle_count: Current cycle number
            log_fn: Function to log messages
            stream_emit_fn: Function to emit streaming events
            get_features_fn: Function to get current features as fallback
        """
        self.aragora_path = aragora_path
        self.claude = claude_agent
        self.codex = codex_agent
        self.kilocode_available = kilocode_available
        self.skip_kilocode = skip_kilocode
        self.kilocode_agent_factory = kilocode_agent_factory
        self.cycle_count = cycle_count
        self._log = log_fn or print
        self._stream_emit = stream_emit_fn or (lambda *args: None)
        self._get_features = get_features_fn or (lambda: "No features available")

    async def execute(self) -> ContextResult:
        """
        Execute the context gathering phase.

        Returns:
            ContextResult with gathered codebase context
        """
        phase_start = time.perf_counter()
        phase_start_dt = datetime.now()

        # Determine how many agents will participate
        use_kilocode = self.kilocode_available and not self.skip_kilocode
        agents_count = 2  # Claude + Codex always
        if use_kilocode:
            agents_count = 4  # + Gemini + Grok via Kilo Code
            self._log("\n" + "=" * 70)
            self._log("PHASE 0: CONTEXT GATHERING (All 4 agents with codebase access)")
            self._log("  Claude → Claude Code | Codex → Codex CLI")
            self._log("  Gemini → Kilo Code  | Grok → Kilo Code")
            self._log("=" * 70)
        else:
            self._log("\n" + "=" * 70)
            self._log("PHASE 0: CONTEXT GATHERING (Claude + Codex)")
            if self.kilocode_available and self.skip_kilocode:
                self._log("  Note: KiloCode skipped (timeouts); Gemini/Grok join in debates")
            else:
                self._log("  Note: Install kilocode CLI to enable Gemini/Grok exploration")
            self._log("=" * 70)

        self._stream_emit("on_phase_start", "context", self.cycle_count, {"agents": agents_count})

        # Build exploration prompt
        explore_prompt = self._build_explore_prompt()

        # Build list of exploration tasks
        exploration_tasks = [
            self._gather_with_agent(self.claude, "claude", "Claude Code"),
            self._gather_with_agent(self.codex, "codex", "Codex CLI"),
        ]

        # Add Gemini and Grok via Kilo Code if available (and not skipped)
        if use_kilocode and self.kilocode_agent_factory:
            gemini_explorer = self.kilocode_agent_factory(
                name="gemini-explorer",
                provider_id="gemini-explorer",
                model="gemini-3-pro-preview",
                role="explorer",
                timeout=600,
                mode="architect",
            )
            grok_explorer = self.kilocode_agent_factory(
                name="grok-explorer",
                provider_id="grok-explorer",
                model="grok-4",
                role="explorer",
                timeout=600,
                mode="architect",
            )
            exploration_tasks.extend(
                [
                    self._gather_with_agent(gemini_explorer, "gemini", "Kilo Code"),
                    self._gather_with_agent(grok_explorer, "grok", "Kilo Code"),
                ]
            )

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
            combined_context = [f"Current features (from docstring):\n{self._get_features()}"]

        gathered_context = "\n\n".join(combined_context)

        phase_duration = time.perf_counter() - phase_start
        success = len(combined_context) > 0
        self._log(
            f"  Context gathered from {len(combined_context)} agents in {phase_duration:.1f}s"
        )
        self._stream_emit(
            "on_phase_end",
            "context",
            self.cycle_count,
            success,
            phase_duration,
            {"agents": len(combined_context), "context_length": len(gathered_context)},
        )

        # Record metrics if configured
        if _metrics_recorder:
            _metrics_recorder("context", "success" if success else "failure", phase_duration)

        return ContextResult(
            success=success,
            data={"agents_succeeded": len(combined_context)},
            duration_seconds=phase_duration,
            codebase_summary=gathered_context,
            recent_changes="",  # Can be populated if needed
            open_issues=[],
        )

    def _build_explore_prompt(self) -> str:
        """Build the codebase exploration prompt."""
        return f"""Explore the aragora codebase and provide a comprehensive summary of EXISTING features.

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

    async def _gather_with_agent(self, agent: Any, name: str, harness: str) -> Tuple[str, str, str]:
        """Run exploration with one agent."""
        agent_start = time.perf_counter()
        try:
            self._log(f"  {name} ({harness}): exploring codebase...", agent=name)
            prompt = self._build_explore_prompt()
            result = await agent.generate(prompt, context=[])
            self._log(f"  {name}: complete ({len(result) if result else 0} chars)", agent=name)
            # Emit agent's full exploration result
            if result:
                self._stream_emit(
                    "on_log_message", result, level="info", phase="context", agent=name
                )
            return (name, harness, result if result else "No response")
        except Exception as e:
            self._log(f"  {name}: error - {e}", agent=name)
            return (name, harness, f"Error: {e}")
        finally:
            # Record per-agent metrics
            if _agent_metrics_recorder:
                agent_duration = time.perf_counter() - agent_start
                _agent_metrics_recorder("context", name, agent_duration)


__all__ = ["ContextPhase", "set_metrics_recorder"]
