"""Natural Language to Workflow Builder.

Converts natural language descriptions into executable workflow definitions
using template matching, pattern matching, and task decomposition.

Supports two modes:
- Quick: Fast heuristic matching, no API calls
- Debate: Multi-agent decomposition via TaskDecomposer
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from aragora.workflow.types import (
    StepDefinition,
    TransitionRule,
    WorkflowDefinition,
)

logger = logging.getLogger(__name__)

# Stopwords for tokenization
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        "their",
    }
)

# Keyword groups for pattern matching
_PATTERN_KEYWORDS: dict[str, list[str]] = {
    "hive_mind": ["parallel", "simultaneously", "at the same time", "concurrently", "all at once"],
    "review_cycle": ["review", "iterate", "refine", "cycle", "revise", "feedback loop"],
    "map_reduce": ["split", "divide", "map reduce", "aggregate", "partition", "combine"],
    "hierarchical": ["hierarchy", "delegate", "manager", "subordinate", "top-down"],
    "dialectic": ["thesis", "antithesis", "dialectic", "synthesis", "opposing views"],
    "sequential": ["sequence", "pipeline", "chain", "step by step", "then", "after that"],
    "ensemble": ["ensemble", "vote", "majority", "consensus", "poll"],
}

# Keyword groups for inferring step types
_STEP_TYPE_KEYWORDS: dict[str, list[str]] = {
    "debate": ["debate", "discuss", "argue", "deliberate", "weigh"],
    "human_checkpoint": ["review", "approve", "check", "sign off", "manual"],
    "content_extraction": ["extract", "parse", "scrape", "pull data", "harvest"],
    "connector": ["search", "fetch", "query", "api call", "webhook", "send"],
    "memory_write": ["remember", "store", "save", "persist", "record"],
    "memory_read": ["recall", "retrieve", "lookup", "look up", "find in memory"],
    "decision": ["decide", "choose", "if", "branch", "pick"],
    "loop": ["loop", "repeat", "iterate", "until", "while"],
    "parallel": ["parallel", "concurrent", "fan out", "fork"],
    "verification": ["verify", "test", "validate", "check quality"],
    "implementation": ["implement", "code", "build", "develop", "write code"],
}


@dataclass
class NLBuildConfig:
    """Configuration for NL workflow generation."""

    mode: str = "quick"  # "quick" or "debate"
    max_steps: int = 20
    template_match_threshold: float = 0.3
    auto_layout: bool = True
    default_category: str = "general"


@dataclass
class NLBuildResult:
    """Result of NL-to-workflow generation."""

    workflow: WorkflowDefinition | None = None
    source: str = ""  # "template_match", "pattern_match", "decomposition", "none"
    confidence: float = 0.0
    matched_template_id: str | None = None
    matched_pattern: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.workflow is not None,
            "source": self.source,
            "confidence": self.confidence,
            "matched_template_id": self.matched_template_id,
            "matched_pattern": self.matched_pattern,
            "warnings": self.warnings,
            "workflow": self.workflow.to_dict() if self.workflow else None,
        }


def _tokenize(text: str) -> set[str]:
    """Lowercase split, remove stopwords and short tokens."""
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Compute Jaccard coefficient between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)


class NLWorkflowBuilder:
    """Converts natural language descriptions to workflow definitions."""

    def __init__(self, config: NLBuildConfig | None = None):
        self._config = config or NLBuildConfig()

    async def build(
        self,
        description: str,
        category: str | None = None,
        agents: list[str] | None = None,
    ) -> NLBuildResult:
        """Build a workflow from natural language description.

        Pipeline:
        1. Try template matching (fast, high confidence)
        2. Try pattern matching (medium confidence)
        3. Fall back to task decomposition (slower, lower confidence)
        """
        category = category or self._config.default_category

        # Step 1: Template matching
        result = self._match_template(description)
        if result.workflow:
            return result

        # Step 2: Pattern matching
        result = self._match_pattern(description, agents)
        if result.workflow:
            return result

        # Step 3: Task decomposition
        if self._config.mode == "debate":
            result = await self._decompose_with_debate(description, category, agents)
        else:
            result = self._decompose_quick(description, category, agents)

        return result

    def build_quick(
        self,
        description: str,
        category: str | None = None,
    ) -> NLBuildResult:
        """Synchronous quick build -- template/pattern match only, no API calls."""
        result = self._match_template(description)
        if result.workflow:
            return result
        result = self._match_pattern(description)
        if result.workflow:
            return result
        result = self._decompose_quick(description, category)
        return result

    # -----------------------------------------------------------------
    # Template matching
    # -----------------------------------------------------------------

    def _match_template(self, description: str) -> NLBuildResult:
        """Match description against registered workflow templates."""
        try:
            from aragora.workflow.templates import WORKFLOW_TEMPLATES
        except ImportError:
            return NLBuildResult(source="none")

        desc_tokens = _tokenize(description)
        if not desc_tokens:
            return NLBuildResult(source="none")

        best_score = 0.0
        best_id: str | None = None
        best_template: dict[str, Any] | None = None

        for template_id, template in WORKFLOW_TEMPLATES.items():
            template_text = (
                f"{template_id} {template.get('name', '')} {template.get('description', '')}"
            )
            template_tokens = _tokenize(template_text)
            score = _jaccard_similarity(desc_tokens, template_tokens)
            if score > best_score:
                best_score = score
                best_id = template_id
                best_template = template

        if best_score >= self._config.template_match_threshold and best_template is not None:
            try:
                workflow = WorkflowDefinition.from_dict(best_template)
                # Give it a fresh ID
                workflow = workflow.clone(
                    new_id=f"wf_{uuid.uuid4().hex[:12]}",
                    new_name=workflow.name,
                )
                return NLBuildResult(
                    workflow=workflow,
                    source="template_match",
                    confidence=min(best_score * 1.5, 1.0),
                    matched_template_id=best_id,
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Template instantiation failed for %s: %s", best_id, exc)

        return NLBuildResult(source="none")

    # -----------------------------------------------------------------
    # Pattern matching
    # -----------------------------------------------------------------

    def _match_pattern(self, description: str, agents: list[str] | None = None) -> NLBuildResult:
        """Match description against workflow pattern keywords."""
        desc_lower = description.lower()

        best_pattern: str | None = None
        best_count = 0

        for pattern_name, keywords in _PATTERN_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in desc_lower)
            if count > best_count:
                best_count = count
                best_pattern = pattern_name

        if best_pattern and best_count >= 1:
            try:
                from aragora.workflow.patterns.base import PatternType
                from aragora.workflow.patterns import create_pattern

                pattern_type = PatternType(best_pattern)
                kwargs: dict[str, Any] = {
                    "name": f"NL: {description[:60]}",
                    "task": description,
                }
                if agents:
                    kwargs["agents"] = agents

                pattern = create_pattern(pattern_type, **kwargs)
                workflow = pattern.create(**kwargs)

                return NLBuildResult(
                    workflow=workflow,
                    source="pattern_match",
                    confidence=min(best_count * 0.25, 0.9),
                    matched_pattern=best_pattern,
                )
            except (ValueError, TypeError, ImportError) as exc:
                logger.debug("Pattern creation failed for %s: %s", best_pattern, exc)

        return NLBuildResult(source="none")

    # -----------------------------------------------------------------
    # Task decomposition (quick)
    # -----------------------------------------------------------------

    def _decompose_quick(
        self,
        description: str,
        category: str | None = None,
        agents: list[str] | None = None,
    ) -> NLBuildResult:
        """Decompose description into steps using the TaskDecomposer heuristic."""
        try:
            from aragora.nomic.task_decomposer import TaskDecomposer
        except ImportError:
            return self._fallback_decompose(description, category, agents)

        try:
            decomposer = TaskDecomposer()
            decomposition = decomposer.analyze(description)
        except (TypeError, ValueError, RuntimeError) as exc:
            logger.debug("TaskDecomposer.analyze failed: %s", exc)
            return self._fallback_decompose(description, category, agents)

        if not decomposition.subtasks:
            return self._fallback_decompose(description, category, agents)

        steps = []
        for subtask in decomposition.subtasks[: self._config.max_steps]:
            step_type = self._infer_step_type(subtask.title + " " + subtask.description)
            steps.append(
                StepDefinition(
                    id=subtask.id,
                    name=subtask.title,
                    step_type=step_type,
                    config={"task": subtask.description},
                    description=subtask.description,
                )
            )

        transitions = self._build_transitions(steps, decomposition)
        workflow = self._assemble_workflow(
            name=f"NL: {description[:60]}",
            steps=steps,
            transitions=transitions,
            category=category,
        )

        return NLBuildResult(
            workflow=workflow,
            source="decomposition",
            confidence=min(decomposition.complexity_score / 10.0, 0.8),
        )

    # -----------------------------------------------------------------
    # Task decomposition (debate)
    # -----------------------------------------------------------------

    async def _decompose_with_debate(
        self,
        description: str,
        category: str | None = None,
        agents: list[str] | None = None,
    ) -> NLBuildResult:
        """Decompose using debate-based analysis."""
        try:
            from aragora.nomic.task_decomposer import TaskDecomposer

            decomposer = TaskDecomposer()
            decomposition = await decomposer.analyze_with_debate(description)
        except (ImportError, TypeError, ValueError, RuntimeError) as exc:
            logger.debug("Debate decomposition failed, falling back to quick: %s", exc)
            return self._decompose_quick(description, category, agents)

        if not decomposition.subtasks:
            return self._decompose_quick(description, category, agents)

        steps = []
        for subtask in decomposition.subtasks[: self._config.max_steps]:
            step_type = self._infer_step_type(subtask.title + " " + subtask.description)
            steps.append(
                StepDefinition(
                    id=subtask.id,
                    name=subtask.title,
                    step_type=step_type,
                    config={"task": subtask.description},
                    description=subtask.description,
                )
            )

        transitions = self._build_transitions(steps, decomposition)
        workflow = self._assemble_workflow(
            name=f"NL: {description[:60]}",
            steps=steps,
            transitions=transitions,
            category=category,
        )

        return NLBuildResult(
            workflow=workflow,
            source="decomposition",
            confidence=min(decomposition.complexity_score / 10.0, 0.85),
        )

    # -----------------------------------------------------------------
    # Fallback decomposition
    # -----------------------------------------------------------------

    def _fallback_decompose(
        self,
        description: str,
        category: str | None = None,
        agents: list[str] | None = None,
    ) -> NLBuildResult:
        """Create a simple single-agent step when decomposition is unavailable."""
        step_id = f"step_{uuid.uuid4().hex[:8]}"
        step = StepDefinition(
            id=step_id,
            name=description[:80],
            step_type="agent",
            config={"task": description, "agent_type": agents[0] if agents else "claude"},
            description=description,
        )
        workflow = self._assemble_workflow(
            name=f"NL: {description[:60]}",
            steps=[step],
            transitions=[],
            category=category,
        )
        return NLBuildResult(
            workflow=workflow,
            source="decomposition",
            confidence=0.3,
            warnings=["TaskDecomposer unavailable; created single-step workflow"],
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _infer_step_type(text: str) -> str:
        """Infer the best step type from a text description."""
        text_lower = text.lower()
        best_type = "agent"
        best_count = 0

        for step_type, keywords in _STEP_TYPE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_type = step_type

        return best_type

    @staticmethod
    def _build_transitions(
        steps: list[StepDefinition],
        decomposition: Any,
    ) -> list[TransitionRule]:
        """Build TransitionRules from SubTask dependencies."""
        step_ids = {s.id for s in steps}
        transitions: list[TransitionRule] = []

        for subtask in decomposition.subtasks:
            if subtask.id not in step_ids:
                continue
            for dep_id in subtask.dependencies:
                if dep_id in step_ids:
                    transitions.append(
                        TransitionRule(
                            id=f"tr_{dep_id}_to_{subtask.id}_{uuid.uuid4().hex[:6]}",
                            from_step=dep_id,
                            to_step=subtask.id,
                            condition="True",
                        )
                    )

        # If no transitions from decomposition, create a sequential chain
        if not transitions and len(steps) > 1:
            for i in range(len(steps) - 1):
                transitions.append(
                    TransitionRule(
                        id=f"tr_{steps[i].id}_to_{steps[i + 1].id}_{uuid.uuid4().hex[:6]}",
                        from_step=steps[i].id,
                        to_step=steps[i + 1].id,
                        condition="True",
                    )
                )

        return transitions

    def _assemble_workflow(
        self,
        name: str,
        steps: list[StepDefinition],
        transitions: list[TransitionRule],
        category: str | None = None,
    ) -> WorkflowDefinition:
        """Assemble a WorkflowDefinition with optional auto-layout."""
        from aragora.workflow.types import WorkflowCategory

        cat_value = category or self._config.default_category
        try:
            wf_category = WorkflowCategory(cat_value)
        except ValueError:
            wf_category = WorkflowCategory.GENERAL

        workflow = WorkflowDefinition(
            id=f"wf_{uuid.uuid4().hex[:12]}",
            name=name,
            description=f"Auto-generated workflow: {name}",
            steps=steps,
            transitions=transitions,
            category=wf_category,
        )

        if self._config.auto_layout and steps:
            try:
                from aragora.workflow.layout import flow_layout

                step_dicts = [{"id": s.id, "type": s.step_type} for s in steps]
                tr_dicts = [{"from_step": t.from_step, "to_step": t.to_step} for t in transitions]
                positions = flow_layout(step_dicts, tr_dicts)
                for pos in positions:
                    step = workflow.get_step(pos.step_id)
                    if step:
                        step.visual.position.x = pos.x
                        step.visual.position.y = pos.y
            except ImportError:
                pass

        return workflow
