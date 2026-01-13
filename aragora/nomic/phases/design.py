"""
Design phase for nomic loop.

Phase 2: Implementation design
- Multi-agent design collaboration
- Gemini as design lead, others as critics
- Deadlock resolution via counterfactual branching
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import DesignResult


# Default protected files
DEFAULT_PROTECTED_FILES = [
    "CLAUDE.md",
    "core.py",
    "aragora/__init__.py",
    ".env",
    "scripts/nomic_loop.py",
]

SAFETY_PREAMBLE = """SAFETY RULES:
1. NEVER delete or modify protected files
2. EXTEND existing APIs, don't break them
3. Focus on MINIMAL viable implementation
4. Prefer simple, direct solutions"""


@dataclass
class DesignConfig:
    """Configuration for design phase."""
    rounds: int = 2
    consensus_mode: str = "judge"
    judge_selection: str = "elo_ranked"
    proposer_count: int = 1
    early_stopping: bool = True
    early_stop_threshold: float = 0.66
    min_rounds_before_early_stop: int = 1
    protected_files: List[str] = None

    def __post_init__(self):
        if self.protected_files is None:
            self.protected_files = DEFAULT_PROTECTED_FILES


@dataclass
class BeliefContext:
    """Belief analysis context from debate phase."""
    contested_count: int = 0
    crux_count: int = 0
    posteriors: Dict[str, Any] = None
    convergence_achieved: bool = False

    def to_string(self) -> str:
        """Convert to context string for design prompt."""
        if self.contested_count == 0 and self.crux_count == 0:
            return ""

        lines = ["\n## UNCERTAINTY FROM DEBATE"]
        lines.append(
            f"The debate identified {self.contested_count} contested claims "
            f"and {self.crux_count} crux (high-impact disputed) claims."
        )

        if self.crux_count > 0:
            lines.append("Your design MUST address these disputed points explicitly.")

        if self.posteriors:
            lines.append("Key uncertainty areas:")
            for claim_id, dist in list(self.posteriors.items())[:3]:
                if isinstance(dist, dict) and dist.get("entropy", 0) > 0.5:
                    lines.append(f"  - {claim_id}: entropy={dist.get('entropy', 0):.2f}")

        if self.convergence_achieved:
            lines.append("\n[Belief network converged - good consensus foundation]")
        else:
            lines.append("\n[Belief network did NOT converge - proceed with caution]")

        return "\n".join(lines)


class DesignPhase:
    """
    Handles the implementation design phase.

    Orchestrates multi-agent design collaboration where Gemini leads
    and other agents critique to produce a detailed implementation plan.
    """

    def __init__(
        self,
        aragora_path: Path,
        agents: List[Any],
        arena_factory: Callable[..., Any],
        environment_factory: Callable[..., Any],
        protocol_factory: Callable[..., Any],
        config: Optional[DesignConfig] = None,
        nomic_integration: Optional[Any] = None,
        deep_audit_fn: Optional[Callable[..., Any]] = None,
        arbitrate_fn: Optional[Callable[..., Any]] = None,
        max_cycle_seconds: int = 3600,
        cycle_count: int = 0,
        log_fn: Optional[Callable[[str], None]] = None,
        stream_emit_fn: Optional[Callable[..., None]] = None,
        record_replay_fn: Optional[Callable[..., None]] = None,
    ):
        """
        Initialize the design phase.

        Args:
            aragora_path: Path to the aragora project root
            agents: List of agent instances for design
            arena_factory: Factory to create Arena instances
            environment_factory: Factory to create Environment instances
            protocol_factory: Factory to create DebateProtocol instances
            config: Design configuration
            nomic_integration: Optional NomicIntegration for probing/checkpointing
            deep_audit_fn: Optional function for deep audit
            arbitrate_fn: Optional function for design arbitration
            max_cycle_seconds: Maximum cycle time budget
            cycle_count: Current cycle number
            log_fn: Function to log messages
            stream_emit_fn: Function to emit streaming events
            record_replay_fn: Function to record replay events
        """
        self.aragora_path = aragora_path
        self.agents = agents
        self._arena_factory = arena_factory
        self._environment_factory = environment_factory
        self._protocol_factory = protocol_factory
        self.config = config or DesignConfig()
        self.nomic_integration = nomic_integration
        self._deep_audit = deep_audit_fn
        self._arbitrate = arbitrate_fn
        self.max_cycle_seconds = max_cycle_seconds
        self.cycle_count = cycle_count
        self._log = log_fn or print
        self._stream_emit = stream_emit_fn or (lambda *args: None)
        self._record_replay = record_replay_fn or (lambda *args: None)

    async def execute(
        self,
        improvement: str,
        belief_context: Optional[BeliefContext] = None,
        learning_context: str = "",
        arena_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DesignResult:
        """
        Execute the design phase.

        Args:
            improvement: The improvement proposal from debate phase
            belief_context: Optional belief analysis from debate
            learning_context: Optional learning context (patterns, etc.)
            arena_kwargs: Additional kwargs for Arena creation

        Returns:
            DesignResult with design outcome
        """
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 2: IMPLEMENTATION DESIGN")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "design", self.cycle_count, {"agents": len(self.agents)})
        self._record_replay("phase", "system", "design")

        # Check if deep audit needed
        if self._deep_audit:
            audit_result = await self._check_deep_audit(improvement)
            if audit_result and not audit_result.get("approved", True):
                return self._rejected_result(phase_start, audit_result)

        # Build design prompt
        design_prompt = self._build_design_prompt(
            improvement,
            belief_context or BeliefContext(),
            learning_context,
        )

        # Create environment
        env = self._environment_factory(
            task=design_prompt,
            context=f"Working directory: {self.aragora_path}\n\nProtected files (NEVER delete): {self.config.protected_files}",
        )

        # Create protocol
        protocol = self._protocol_factory(
            rounds=self.config.rounds,
            consensus=self.config.consensus_mode,
            judge_selection=self.config.judge_selection,
            proposer_count=self.config.proposer_count,
            early_stopping=self.config.early_stopping,
            early_stop_threshold=self.config.early_stop_threshold,
            min_rounds_before_early_stop=self.config.min_rounds_before_early_stop,
        )

        self._log("  [hybrid] Lead agent as design lead, others as critics")

        # Probe agents for reliability
        agent_weights = await self._probe_agents()

        # Create arena
        arena = self._arena_factory(
            env,
            self.agents,
            protocol,
            agent_weights=agent_weights,
            **(arena_kwargs or {}),
        )

        # Run design debate
        result = await arena.run()

        # Handle no consensus with fast-track arbitration
        if not result.consensus_reached:
            result = await self._handle_no_consensus(
                result, arena, improvement, phase_start
            )

        # Extract proposals and votes
        individual_proposals = self._extract_proposals(result)
        vote_counts = self._count_votes(result)

        # Checkpoint
        await self._checkpoint(result, agent_weights)

        # Build result
        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit(
            "on_phase_end", "design", self.cycle_count, result.consensus_reached,
            phase_duration, {}
        )

        # Identify affected files
        files_affected = self._extract_files_from_design(result.final_answer or "")

        return DesignResult(
            success=result.consensus_reached,
            data={
                "individual_proposals": individual_proposals,
                "vote_counts": vote_counts,
                "agent_weights": agent_weights,
            },
            duration_seconds=phase_duration,
            design=result.final_answer or "",
            files_affected=files_affected,
            complexity_estimate=self._estimate_complexity(result.final_answer or ""),
        )

    async def _check_deep_audit(self, improvement: str) -> Optional[Dict]:
        """Check if deep audit is needed and run it."""
        try:
            should_audit, reason = self._deep_audit("check", improvement, phase="design")
            if should_audit:
                self._log(f"  [deep-audit] {reason}")
                return await self._deep_audit("run", improvement)
        except Exception as e:
            self._log(f"  [deep-audit] Check failed: {e}")
        return None

    def _rejected_result(self, phase_start: datetime, audit_result: Dict) -> DesignResult:
        """Create a rejected result from deep audit."""
        self._log("  [deep-audit] Design rejected - returning issues for rework")
        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit("on_phase_end", "design", self.cycle_count, False, phase_duration)
        return DesignResult(
            success=False,
            error="Rejected by deep audit",
            data={
                "rejected_by_deep_audit": True,
                "issues": audit_result.get("unanimous_issues", []),
            },
            duration_seconds=phase_duration,
            design="",
            files_affected=[],
            complexity_estimate="blocked",
        )

    def _build_design_prompt(
        self,
        improvement: str,
        belief: BeliefContext,
        learning: str,
    ) -> str:
        """Build the design task prompt."""
        belief_text = belief.to_string()
        combined_learning = f"{learning}\n{belief_text}" if belief_text else learning

        return f"""{SAFETY_PREAMBLE}

## DESIGN TASK
Create a detailed implementation design for this improvement:

{improvement}

## CONTEXT FROM PRIOR LEARNING
{combined_learning}

## REQUIRED DESIGN SECTIONS

### 1. FILE CHANGES (Required)
List EVERY file to create or modify. Be specific:
- `aragora/path/file.py` - Create new | Modify existing
- Estimated lines of code per file
- NEVER delete or modify protected files: {self.config.protected_files}

### 2. API DESIGN (Required)
Define the public interface:
```python
class ClassName:
    def method_name(self, param: Type) -> ReturnType:
        '''Docstring explaining purpose.'''
        ...
```
- EXTEND existing APIs, don't break them

### 3. INTEGRATION POINTS (Required)
How does this connect to existing modules?
- Which existing classes/functions does it use?
- Which existing classes/functions call it?

### 4. TEST PLAN (Required)
Concrete test cases:
- Unit tests: `test_feature_basic()`, `test_feature_edge_case()`
- Integration tests if needed

### 5. EXAMPLE USAGE (Required)
Working code snippet showing the feature in action.

## REQUIRED: Viability Checklist
- [ ] At least 3 specific file changes with estimated line counts
- [ ] At least 2 function/class signatures with parameters
- [ ] At least 1 integration point showing how components connect
- [ ] 1 concrete example of how to use the feature

Designs missing any of these will be automatically rejected."""

    async def _probe_agents(self) -> Dict[str, float]:
        """Probe agents for reliability weights."""
        if not self.nomic_integration:
            return {}

        try:
            self._log("  [integration] Probing agents for reliability...")
            weights = await self.nomic_integration.probe_agents(
                self.agents,
                probe_count=2,
                min_weight=0.5,
            )
            reliable = sum(1 for w in weights.values() if w >= 0.7)
            self._log(f"  [integration] Agent weights: {reliable}/{len(self.agents)} reliable")
            return weights
        except Exception as e:
            self._log(f"  [integration] Probing failed: {e}")
            return {}

    async def _handle_no_consensus(
        self,
        result: Any,
        arena: Any,
        improvement: str,
        phase_start: datetime,
    ) -> Any:
        """Handle case when no consensus is reached."""
        elapsed = (datetime.now() - phase_start).total_seconds()
        remaining = self.max_cycle_seconds - elapsed

        # Fast-track arbitration if low on time
        if remaining < 600 and self._arbitrate:
            self._log(f"  [fast-track] Time budget critical ({remaining:.0f}s left)")
            proposals = self._extract_proposals(result)
            if len(proposals) >= 2:
                arbitrated = await self._arbitrate(proposals, improvement)
                if arbitrated:
                    result.final_answer = arbitrated
                    result.consensus_reached = True
                    result.confidence = 0.7
                    self._log(f"  [fast-track] Judge selected design")
                    return result

        # Try counterfactual resolution via nomic integration
        if self.nomic_integration:
            result = await self._counterfactual_resolution(result, arena)

        return result

    async def _counterfactual_resolution(self, result: Any, arena: Any) -> Any:
        """Attempt counterfactual resolution for deadlock."""
        try:
            self._log("  [deadlock] No design consensus - attempting resolution...")
            post_analysis = await self.nomic_integration.full_post_debate_analysis(
                result=result,
                arena=arena,
                claims_kernel=None,
                changed_files=None,
            )

            conditional = post_analysis.get("conditional")
            if conditional:
                self._log("  [deadlock] Resolved with conditional consensus")
                if hasattr(conditional, 'synthesized_answer') and conditional.synthesized_answer:
                    result.final_answer = conditional.synthesized_answer
                    result.consensus_reached = True
                    result.confidence = getattr(conditional, 'confidence', 0.7)
                elif hasattr(conditional, 'if_true_conclusion'):
                    result.final_answer = self._build_conditional_design(conditional)
                    result.consensus_reached = True
                    result.confidence = max(
                        conditional.if_true_confidence or 0.5,
                        conditional.if_false_confidence or 0.5
                    )
        except Exception as e:
            self._log(f"  [deadlock] Resolution failed: {e}")

        return result

    def _build_conditional_design(self, conditional: Any) -> str:
        """Build conditional design from branches."""
        pivot = getattr(conditional, 'pivot_claim', None)
        pivot_text = pivot.text if pivot and hasattr(pivot, 'text') else "the disputed assumption"
        return (
            f"## CONDITIONAL DESIGN\n\n"
            f"**Key Decision Point:** {pivot_text}\n\n"
            f"### If True:\n{conditional.if_true_conclusion}\n\n"
            f"### If False:\n{conditional.if_false_conclusion}\n\n"
            f"**Recommended Path:** "
            f"{'True assumption' if conditional.preferred_world else 'False assumption'}"
        )

    def _extract_proposals(self, result: Any) -> Dict[str, str]:
        """Extract individual proposals from messages."""
        proposals = {}
        if hasattr(result, 'messages'):
            for msg in result.messages:
                if msg.role == "proposer" and msg.content:
                    proposals[msg.agent] = msg.content
        return proposals

    def _count_votes(self, result: Any) -> Dict[str, int]:
        """Count votes from result."""
        counts: Dict[str, int] = {}
        if hasattr(result, 'votes'):
            for vote in result.votes:
                choice = vote.choice
                counts[choice] = counts.get(choice, 0) + 1
        return counts

    async def _checkpoint(self, result: Any, agent_weights: Dict) -> None:
        """Checkpoint the design phase."""
        if not self.nomic_integration:
            return

        try:
            await self.nomic_integration.checkpoint(
                phase="design",
                state={"design": result.final_answer, "agent_weights": agent_weights},
                cycle=self.cycle_count,
            )
        except Exception as e:
            self._log(f"  [integration] Checkpoint failed: {e}")

    def _extract_files_from_design(self, design: str) -> List[str]:
        """Extract file paths mentioned in the design."""
        import re
        files = []
        # Match patterns like `aragora/path/file.py` or aragora/path/file.py
        pattern = r'`?(aragora/[a-zA-Z0-9_/]+\.py)`?'
        matches = re.findall(pattern, design)
        files.extend(matches)
        return list(set(files))[:10]  # Limit to 10 files

    def _estimate_complexity(self, design: str) -> str:
        """Estimate implementation complexity."""
        if not design:
            return "unknown"

        lines_mentioned = design.lower().count("lines")
        file_count = len(self._extract_files_from_design(design))

        if file_count > 5 or "complex" in design.lower():
            return "high"
        elif file_count > 2 or lines_mentioned > 3:
            return "medium"
        else:
            return "low"


__all__ = ["DesignPhase", "DesignConfig", "BeliefContext"]
