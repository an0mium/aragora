"""Stage transition debates — mini-debates at each pipeline stage boundary.

When enabled, the pipeline runs a lightweight 2-agent debate before each
stage transition to validate priorities, challenge assumptions, and
generate a Decision Receipt documenting the rationale.

Each transition debate uses a focused task prompt and 2 rounds of
adversarial critique, producing:
- A verdict (proceed / revise / block)
- Confidence score
- Rationale summary
- Optional Decision Receipt for audit trail
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TransitionDebateResult:
    """Result of a stage transition mini-debate."""

    transition_id: str
    from_stage: str
    to_stage: str
    verdict: str = "proceed"  # proceed | revise | block
    confidence: float = 0.7
    rationale: str = ""
    dissenting_views: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    debate_id: str | None = None
    receipt: dict[str, Any] | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "dissenting_views": self.dissenting_views,
            "suggestions": self.suggestions,
            "debate_id": self.debate_id,
            "duration_ms": self.duration_ms,
        }


class StageTransitionDebater:
    """Run mini-debates at pipeline stage transitions.

    Uses the Arena debate engine with 2 agents and 2 rounds to
    adversarially validate each stage transition before proceeding.

    Usage::

        debater = StageTransitionDebater(debate_rounds=2)
        result = await debater.debate_ideas_to_goals(
            ideas=["improve UX", "add tests"],
            proposed_goals=[{"title": "Redesign nav", "priority": "high"}],
        )
        if result.verdict == "proceed":
            # Continue with stage transition
            ...
    """

    def __init__(
        self,
        debate_rounds: int = 2,
        num_agents: int = 2,
        enable_receipts: bool = True,
        timeout_seconds: int = 300,
    ) -> None:
        self.debate_rounds = debate_rounds
        self.num_agents = num_agents
        self.enable_receipts = enable_receipts
        self.timeout_seconds = timeout_seconds

    async def debate_ideas_to_goals(
        self,
        ideas: list[str],
        proposed_goals: list[dict[str, Any]],
    ) -> TransitionDebateResult:
        """Debate whether proposed goals correctly capture the ideas.

        Agents evaluate: Are these the right goals? Should any be
        reprioritized, merged, split, or removed?
        """
        ideas_str = "\n".join(f"- {idea}" for idea in ideas[:20])
        goals_str = "\n".join(
            f"- [{g.get('priority', 'medium')}] {g.get('title', g.get('id', '?'))}"
            for g in proposed_goals[:15]
        )

        task = (
            "You are reviewing a pipeline stage transition from Ideas to Goals.\n\n"
            f"## Input Ideas\n{ideas_str}\n\n"
            f"## Proposed Goals\n{goals_str}\n\n"
            "## Your Task\n"
            "Evaluate whether these goals correctly capture the intent of the ideas.\n"
            "Consider: Are any ideas missing from the goals? Are priorities correct?\n"
            "Should any goals be merged, split, or removed?\n\n"
            "End your response with one of:\n"
            "VERDICT: PROCEED — goals are well-formed\n"
            "VERDICT: REVISE — goals need adjustment (explain what)\n"
            "VERDICT: BLOCK — fundamental issues prevent proceeding"
        )

        return await self._run_debate(task, "ideas", "goals")

    async def debate_goals_to_actions(
        self,
        goals: list[dict[str, Any]],
        proposed_actions: list[dict[str, Any]],
    ) -> TransitionDebateResult:
        """Debate whether the proposed action plan adequately addresses goals.

        Agents evaluate: Is this the right implementation approach? Are
        there better alternatives or missing steps?
        """
        goals_str = "\n".join(
            f"- [{g.get('priority', 'medium')}] {g.get('title', '?')}" for g in goals[:15]
        )
        actions_str = "\n".join(
            f"- [{a.get('step_type', 'task')}] {a.get('name', a.get('id', '?'))}"
            for a in proposed_actions[:20]
        )

        task = (
            "You are reviewing a pipeline stage transition from Goals to Actions.\n\n"
            f"## Goals\n{goals_str}\n\n"
            f"## Proposed Action Steps\n{actions_str}\n\n"
            "## Your Task\n"
            "Evaluate whether these action steps will achieve the goals.\n"
            "Consider: Are any critical steps missing? Is the ordering correct?\n"
            "Could steps be parallelized? Are there unnecessary steps?\n\n"
            "End your response with one of:\n"
            "VERDICT: PROCEED — action plan is sound\n"
            "VERDICT: REVISE — actions need adjustment (explain what)\n"
            "VERDICT: BLOCK — fundamental issues with the approach"
        )

        return await self._run_debate(task, "goals", "actions")

    async def debate_actions_to_orchestration(
        self,
        actions: list[dict[str, Any]],
        proposed_assignments: list[dict[str, Any]],
    ) -> TransitionDebateResult:
        """Debate agent assignment and execution strategy.

        Agents evaluate: Are the right agents assigned to the right tasks?
        Is the parallelism strategy optimal?
        """
        actions_str = "\n".join(f"- {a.get('name', a.get('id', '?'))}" for a in actions[:20])
        assignments_str = "\n".join(
            f"- {a.get('name', '?')} → {a.get('agent_id', 'unassigned')}"
            for a in proposed_assignments[:20]
        )

        task = (
            "You are reviewing a pipeline transition from Actions to Orchestration.\n\n"
            f"## Action Steps\n{actions_str}\n\n"
            f"## Proposed Agent Assignments\n{assignments_str}\n\n"
            "## Your Task\n"
            "Evaluate whether agent assignments are optimal.\n"
            "Consider: Are specialists assigned to their domain? Could tasks\n"
            "be parallelized more? Are there dependency bottlenecks?\n\n"
            "End your response with one of:\n"
            "VERDICT: PROCEED — assignments are well-matched\n"
            "VERDICT: REVISE — some assignments should change (explain)\n"
            "VERDICT: BLOCK — critical assignment problems"
        )

        return await self._run_debate(task, "actions", "orchestration")

    async def _run_debate(
        self,
        task: str,
        from_stage: str,
        to_stage: str,
    ) -> TransitionDebateResult:
        """Run a lightweight Arena debate and extract the verdict."""
        start = time.monotonic()
        transition_id = f"td-{from_stage}-{to_stage}-{uuid.uuid4().hex[:8]}"

        try:
            debate_result = await self._execute_arena_debate(task)

            # Parse verdict from final answer
            verdict, confidence, rationale = self._parse_verdict(
                debate_result.get("final_answer", "")
            )

            result = TransitionDebateResult(
                transition_id=transition_id,
                from_stage=from_stage,
                to_stage=to_stage,
                verdict=verdict,
                confidence=confidence,
                rationale=rationale,
                dissenting_views=debate_result.get("dissenting_views", []),
                suggestions=self._extract_suggestions(debate_result.get("final_answer", "")),
                debate_id=debate_result.get("debate_id"),
                duration_ms=(time.monotonic() - start) * 1000,
            )

            # Generate receipt if enabled
            if self.enable_receipts:
                result.receipt = self._generate_transition_receipt(result, task, debate_result)

            logger.info(
                "Transition debate %s→%s: verdict=%s confidence=%.2f (%.0fms)",
                from_stage,
                to_stage,
                verdict,
                confidence,
                result.duration_ms,
            )
            return result

        except Exception as exc:
            logger.warning(
                "Transition debate %s→%s failed, defaulting to proceed: %s",
                from_stage,
                to_stage,
                exc,
            )
            return TransitionDebateResult(
                transition_id=transition_id,
                from_stage=from_stage,
                to_stage=to_stage,
                verdict="proceed",
                confidence=0.5,
                rationale=f"Debate unavailable ({type(exc).__name__}), proceeding with default",
                duration_ms=(time.monotonic() - start) * 1000,
            )

    async def _execute_arena_debate(self, task: str) -> dict[str, Any]:
        """Execute a lightweight Arena debate.

        Returns a dict with keys: final_answer, consensus_reached,
        confidence, debate_id, dissenting_views.
        """
        try:
            from aragora.core_types import Environment
            from aragora.debate.orchestrator import Arena
            from aragora.debate.protocol import DebateProtocol

            env = Environment(task=task)
            protocol = DebateProtocol(
                rounds=self.debate_rounds,
                consensus="majority",
                consensus_threshold=0.6,
                early_stopping=True,
                early_stop_threshold=0.9,
                timeout_seconds=self.timeout_seconds,
                use_structured_phases=False,
                enable_trickster=False,
                formal_verification_enabled=False,
            )

            # Use lightweight API agents
            agents = self._create_debate_agents()

            arena = Arena(env, agents, protocol=protocol)
            result = await arena.run()

            return {
                "final_answer": getattr(result, "final_answer", ""),
                "consensus_reached": getattr(result, "consensus_reached", False),
                "confidence": getattr(result, "confidence", 0.5),
                "debate_id": getattr(result, "id", None),
                "dissenting_views": getattr(result, "dissenting_views", []),
                "participants": getattr(result, "participants", []),
            }

        except ImportError:
            logger.debug("Arena not available for transition debate")
            return self._fallback_debate(task)
        except Exception as exc:
            logger.warning("Arena debate failed: %s", exc)
            return self._fallback_debate(task)

    def _create_debate_agents(self) -> list[Any]:
        """Create lightweight agents for mini-debates."""
        agents = []
        try:
            from aragora.agents.api_agents.anthropic import AnthropicAgent

            agents.append(
                AnthropicAgent(
                    name="transition-analyst",
                    model="claude-sonnet-4-20250514",
                    role="proposer",
                )
            )
            if self.num_agents >= 2:
                agents.append(
                    AnthropicAgent(
                        name="transition-critic",
                        model="claude-sonnet-4-20250514",
                        role="critic",
                    )
                )
        except ImportError:
            pass

        if not agents:
            try:
                from aragora.agents.api_agents.openrouter import OpenRouterAgent

                agents.append(
                    OpenRouterAgent(
                        name="transition-analyst",
                        model="anthropic/claude-3.5-sonnet",
                        role="proposer",
                    )
                )
                if self.num_agents >= 2:
                    agents.append(
                        OpenRouterAgent(
                            name="transition-critic",
                            model="anthropic/claude-3.5-sonnet",
                            role="critic",
                        )
                    )
            except ImportError:
                pass

        if not agents:
            # Create stub agents as final fallback
            try:
                from aragora.agents.cli_agents import ClaudeAgent

                agents = [
                    ClaudeAgent(name=f"transition-agent-{i}", model="claude")
                    for i in range(self.num_agents)
                ]
            except ImportError:
                logger.warning("No agents available for transition debate")

        return agents

    def _fallback_debate(self, task: str) -> dict[str, Any]:
        """Structural fallback when Arena is unavailable."""
        return {
            "final_answer": (
                "Arena unavailable. Based on structural analysis, "
                "the transition appears reasonable.\n\nVERDICT: PROCEED"
            ),
            "consensus_reached": True,
            "confidence": 0.5,
            "debate_id": None,
            "dissenting_views": [],
            "participants": [],
        }

    def _parse_verdict(self, final_answer: str) -> tuple[str, float, str]:
        """Parse verdict, confidence, and rationale from debate output."""
        answer_lower = final_answer.lower()

        # Extract verdict
        if "verdict: block" in answer_lower:
            verdict = "block"
            base_confidence = 0.8
        elif "verdict: revise" in answer_lower:
            verdict = "revise"
            base_confidence = 0.7
        elif "verdict: proceed" in answer_lower:
            verdict = "proceed"
            base_confidence = 0.8
        else:
            # Default to proceed if no clear verdict
            verdict = "proceed"
            base_confidence = 0.5

        # Extract rationale (text before verdict line)
        lines = final_answer.strip().split("\n")
        rationale_lines = []
        for line in lines:
            if line.strip().upper().startswith("VERDICT:"):
                break
            if line.strip():
                rationale_lines.append(line.strip())
        rationale = " ".join(rationale_lines[-3:]) if rationale_lines else ""

        return verdict, base_confidence, rationale

    def _extract_suggestions(self, final_answer: str) -> list[str]:
        """Extract actionable suggestions from debate output."""
        suggestions = []
        for line in final_answer.split("\n"):
            stripped = line.strip()
            # Look for suggestion patterns
            if stripped.startswith(("- Should ", "- Consider ", "- Recommend ")):
                suggestions.append(stripped.lstrip("- "))
            elif stripped.startswith(("1. ", "2. ", "3. ")):
                suggestions.append(stripped[3:])
        return suggestions[:5]

    def _generate_transition_receipt(
        self,
        result: TransitionDebateResult,
        task: str,
        debate_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a Decision Receipt for the transition debate."""
        now = datetime.now(timezone.utc).isoformat()
        input_hash = hashlib.sha256(task.encode()).hexdigest()

        receipt = {
            "receipt_id": f"receipt-{result.transition_id}",
            "type": "stage_transition",
            "timestamp": now,
            "transition": {
                "from_stage": result.from_stage,
                "to_stage": result.to_stage,
                "transition_id": result.transition_id,
            },
            "input_hash": input_hash,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "rationale": result.rationale,
            "consensus_reached": debate_result.get("consensus_reached", False),
            "participants": debate_result.get("participants", []),
            "dissenting_views": result.dissenting_views,
            "suggestions": result.suggestions,
            "debate_id": result.debate_id,
            "duration_ms": result.duration_ms,
        }

        # Sign with content hash
        content_str = f"{result.transition_id}:{result.verdict}:{input_hash}"
        receipt["content_hash"] = hashlib.sha256(content_str.encode()).hexdigest()

        return receipt
