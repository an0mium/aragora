"""Task decomposition for Nomic Loop.

Analyzes task complexity and decomposes large tasks into smaller subtasks
for parallel or sequential processing.

Supports two decomposition modes:
1. Heuristic: Fast pattern-matching for concrete goals with file mentions
2. Debate: Multi-agent Arena debate for abstract high-level goals

Integrates with workflow patterns for execution strategies.

Oracle-driven validation:
- File-independence validation ensures sibling subtasks don't share files
- Oracle checks (syntax, existence) validate task scope coherence
- Decomposition quality scoring measures independence, granularity, coverage
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.core import DebateResult, Environment

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """A subtask extracted from a larger task.

    Supports hierarchical goal trees via parent_id/depth:
    - parent_id: ID of the parent subtask (None for root-level)
    - depth: Nesting level (0 = root, 1 = child of root, etc.)
    - children: Populated by TaskDecomposition.build_tree()
    """

    id: str
    title: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    estimated_complexity: str = "low"  # low, medium, high
    file_scope: list[str] = field(default_factory=list)
    success_criteria: dict[str, Any] = field(default_factory=dict)
    """Measurable success criteria, e.g. {"test_pass_rate": ">0.95", "lint_errors": "==0"}.
    Keys map to MetricSnapshot fields. Values are targets like ">0.9", "==0", "<=10"."""
    parent_id: str | None = None
    depth: int = 0
    children: list[SubTask] = field(default_factory=list)


@dataclass
class FileConflict:
    """A file-scope conflict between two sibling subtasks.

    Indicates that both subtasks list the same file in their file_scope,
    meaning they cannot safely execute in parallel without coordination.
    """

    file_path: str
    subtask_ids: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"FileConflict({self.file_path!r}, subtasks={self.subtask_ids})"


@dataclass
class OracleResult:
    """Result of oracle validation on a subtask's file scope.

    Oracle checks are lightweight pre-execution validations:
    - Python syntax check via ast.parse
    - File existence and readability
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)


@dataclass
class DecompositionQuality:
    """Quality score for a set of decomposed subtasks.

    Factors:
    - score: overall quality 0.0-1.0 (higher is better)
    - file_conflicts: number of file-scope overlaps between sibling subtasks
    - avg_scope_size: average number of files per subtask
    - coverage_ratio: fraction of original goal's file scope covered by subtasks
    """

    score: float
    file_conflicts: int
    avg_scope_size: float
    coverage_ratio: float


@dataclass
class TaskDecomposition:
    """Result of task decomposition analysis.

    Supports both flat (subtasks list) and hierarchical (tree) views.
    Call build_tree() to populate children on SubTask objects.
    """

    original_task: str
    complexity_score: int  # 1-10
    complexity_level: str  # low, medium, high
    should_decompose: bool
    subtasks: list[SubTask] = field(default_factory=list)
    rationale: str = ""
    recommend_debate: bool = False
    """Set to True when the goal is abstract (high complexity, no file hints)
    and would benefit from debate-based decomposition. Callers can check this
    to decide whether to use ``analyze_with_debate()`` instead of heuristic."""

    def build_tree(self) -> list[SubTask]:
        """Build hierarchical tree from flat subtask list using parent_id.

        Returns root-level subtasks with children populated recursively.
        The flat subtasks list is not modified.
        """
        by_id: dict[str, SubTask] = {s.id: s for s in self.subtasks}
        roots: list[SubTask] = []
        for subtask in self.subtasks:
            subtask.children = []  # Reset before building
        for subtask in self.subtasks:
            if subtask.parent_id and subtask.parent_id in by_id:
                by_id[subtask.parent_id].children.append(subtask)
            else:
                roots.append(subtask)
        return roots

    def get_roots(self) -> list[SubTask]:
        """Get root-level subtasks (parent_id is None)."""
        return [s for s in self.subtasks if s.parent_id is None]

    def get_children(self, parent_id: str) -> list[SubTask]:
        """Get direct children of a subtask."""
        return [s for s in self.subtasks if s.parent_id == parent_id]

    def max_depth(self) -> int:
        """Get maximum depth in the goal tree."""
        return max((s.depth for s in self.subtasks), default=0)

    def flatten_tree(self, roots: list[SubTask] | None = None) -> list[SubTask]:
        """Flatten a tree back into an ordered list (depth-first)."""
        if roots is None:
            roots = self.build_tree()
        result: list[SubTask] = []
        for root in roots:
            result.append(root)
            result.extend(self.flatten_tree(root.children))
        return result


@dataclass
class DecomposerConfig:
    """Configuration for TaskDecomposer."""

    complexity_threshold: int = 5  # Score above which decomposition is triggered
    max_subtasks: int = 5
    min_subtasks: int = 2
    max_depth: int = 3  # Maximum recursive decomposition depth
    file_complexity_weight: float = 0.3
    concept_complexity_weight: float = 0.4
    length_complexity_weight: float = 0.3
    # Debate-based decomposition settings
    debate_rounds: int = 2  # Rounds for goal decomposition debate
    debate_timeout: int = 120  # Timeout in seconds for debate
    # Trickster: detect hollow consensus in decomposition debates
    enable_trickster: bool = True
    trickster_sensitivity: float = 0.7
    # Convergence detection for semantic consensus
    enable_convergence: bool = True
    # Automatically use debate mode for abstract goals scoring >= 7
    # with no file hints, instead of heuristic decomposition
    auto_debate_abstract: bool = True


# Keywords that indicate different complexity areas
COMPLEXITY_INDICATORS = {
    "high": [
        "refactor",
        "migrate",
        "redesign",
        "overhaul",
        "rewrite",
        "architectural",
        "system-wide",
        "cross-cutting",
        "harden",
        "consolidate",
    ],
    "medium": [
        "integrate",
        "implement",
        "add",
        "create",
        "build",
        "enhance",
        "extend",
        "improve",
        "optimize",
        "adapter",
        "comprehensive",
        "coverage",
        "module",
        "pipeline",
    ],
    "low": [
        "fix",
        "update",
        "tweak",
        "adjust",
        "document",
        "comment",
        "rename",
    ],
}

# Concept areas that suggest decomposition
DECOMPOSITION_CONCEPTS = [
    "database",
    "api",
    "frontend",
    "backend",
    "test",
    "tests",
    "testing",
    "integration",
    "security",
    "performance",
    "documentation",
    "configuration",
    "deployment",
    "authentication",
    "compliance",
    "templates",
    "agents",
    "workflow",
    "connectors",
    "storage",
    "memory",
    "debate",
    "analytics",
    "vertical",
    "audit",
    "cli",
    "sdk",
    "orchestrator",
    "pipeline",
    "validation",
    "gauntlet",
    "handler",
    "server",
    "resilience",
    "observability",
]


class TaskDecomposer:
    """Analyzes tasks and decomposes complex ones into subtasks.

    Uses heuristics based on:
    - Number of files mentioned
    - Complexity keywords present
    - Length of task description
    - Concept breadth (how many different areas touched)

    Example:
        decomposer = TaskDecomposer()
        result = decomposer.analyze("Refactor the authentication system")

        if result.should_decompose:
            for subtask in result.subtasks:
                print(f"  - {subtask.title}")
    """

    def __init__(
        self,
        config: DecomposerConfig | None = None,
        extract_subtasks_fn: Callable[[str], list[dict]] | None = None,
    ):
        """Initialize the decomposer.

        Args:
            config: Decomposition configuration
            extract_subtasks_fn: Optional function to extract subtasks using AI
        """
        self.config = config or DecomposerConfig()
        self._extract_subtasks_fn = extract_subtasks_fn
        self._concept_pattern = re.compile(
            r"\b(" + "|".join(DECOMPOSITION_CONCEPTS) + r")\b",
            re.IGNORECASE,
        )

    def analyze(
        self,
        task_description: str,
        debate_result: DebateResult | None = None,
        depth: int = 0,
    ) -> TaskDecomposition:
        """Analyze a task and determine if decomposition is needed.

        Args:
            task_description: The task or improvement proposal
            debate_result: Optional debate result for additional context
            depth: Current recursion depth (0 = top-level)

        Returns:
            TaskDecomposition with analysis and optional subtasks
        """
        if not task_description:
            return TaskDecomposition(
                original_task="",
                complexity_score=0,
                complexity_level="low",
                should_decompose=False,
                rationale="Empty task",
            )

        # Enforce depth limit to prevent unbounded recursive decomposition
        if depth >= self.config.max_depth:
            logger.info(
                "decomposition_depth_limit_reached depth=%s max=%s", depth, self.config.max_depth
            )
            complexity_score = self._calculate_complexity(task_description, debate_result)
            return TaskDecomposition(
                original_task=task_description,
                complexity_score=complexity_score,
                complexity_level=self._score_to_level(complexity_score),
                should_decompose=False,
                rationale=f"Max decomposition depth ({self.config.max_depth}) reached",
            )

        # Calculate complexity score
        complexity_score = self._calculate_complexity(task_description, debate_result)
        complexity_level = self._score_to_level(complexity_score)

        # If the goal is vague (below decomposition threshold), try semantic
        # expansion to produce concrete subtasks from templates and track configs.
        # This handles abstract goals like "maximize utility for SMEs" that lack
        # file mentions and specific keywords but are still genuinely complex.
        # Skip expansion for goals that are specific but just score low on
        # complexity (e.g. "add retry logic to connectors" — actionable as-is).
        if complexity_score < self.config.complexity_threshold and not self._is_specific_goal(
            task_description
        ):
            expanded = self._expand_vague_goal(task_description)
            if expanded is not None:
                logger.info(
                    "vague_goal_expanded original_score=%s subtasks=%s depth=%s",
                    complexity_score,
                    len(expanded.subtasks),
                    depth,
                )
                return expanded

        # Determine if decomposition is needed
        should_decompose = complexity_score >= self.config.complexity_threshold

        # Build rationale
        rationale = self._build_rationale(task_description, complexity_score, should_decompose)

        # Determine if debate mode is recommended for this goal:
        # abstract goals (high complexity, no file hints) benefit from
        # multi-agent debate rather than heuristic decomposition.
        recommend_debate = (
            self.config.auto_debate_abstract
            and complexity_score >= 7
            and not self._has_file_hints(task_description)
        )

        result = TaskDecomposition(
            original_task=task_description,
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            should_decompose=should_decompose,
            rationale=rationale,
            recommend_debate=recommend_debate,
        )

        # Extract subtasks if decomposition is needed
        if should_decompose:
            result.subtasks = self._generate_subtasks(task_description, debate_result)
            logger.info(
                "task_decomposed complexity=%s subtasks=%s depth=%s",
                complexity_score,
                len(result.subtasks),
                depth,
            )
        else:
            logger.debug(
                "task_not_decomposed complexity=%s threshold=%s",
                complexity_score,
                self.config.complexity_threshold,
            )

        return result

    def _calculate_complexity(
        self,
        task: str,
        debate_result: DebateResult | None = None,
    ) -> int:
        """Calculate complexity score (1-10) for a task.

        Scoring based on:
        - File mentions (30% weight)
        - Complexity keywords (40% weight)
        - Task length (30% weight)
        """
        task_lower = task.lower()

        # File complexity (0-3 points)
        file_count = len(re.findall(r"\b\w+\.(py|ts|tsx|js|jsx|md)\b", task_lower))
        file_score = min(file_count, 3)

        # Keyword complexity (0-4 points)
        keyword_score: float = 0.0
        for indicator in COMPLEXITY_INDICATORS["high"]:
            if indicator in task_lower:
                keyword_score += 1.5
        for indicator in COMPLEXITY_INDICATORS["medium"]:
            if indicator in task_lower:
                keyword_score += 0.5
        keyword_score = min(keyword_score, 4)

        # Length complexity (0-3 points)
        word_count = len(task.split())
        length_score = min(word_count / 30, 3)

        # Concept breadth (0-3 bonus points)
        concepts = (
            self._concept_pattern.findall(task_lower) if hasattr(self, "_concept_pattern") else []
        )
        unique_concepts = set(c.lower() for c in concepts)
        concept_score = min(len(unique_concepts), 3)

        # Multi-clause goals (commas, "and", semicolons indicate compound tasks)
        clause_count = len(re.split(r",\s+and\s+|\band\b|;\s*", task)) - 1
        clause_score = min(clause_count, 2)

        # Vagueness bonus: goals that lack specifics are high-level strategic
        # objectives that inherently require decomposition.  A goal with no file
        # mentions, no technical keywords, and no concept terms is almost
        # certainly a broad directive like "maximize utility for SMEs".
        vagueness_bonus = 0.0
        # Check for specific path references that indicate a targeted goal
        has_path_ref = bool(
            re.search(r"aragora/\w+|tests/\w+|sdk/\w+|scripts/\w+|src/\w+", task_lower)
        )
        if file_score == 0 and not has_path_ref:
            # Check for strategic/broad language that signals high-level goals
            strategic_terms = {
                "maximize",
                "minimise",
                "minimize",
                "optimise",
                "optimize",
                "ensure",
                "improve",
                "enhance",
                "increase",
                "reduce",
                "accelerate",
                "streamline",
                "transform",
                "scale",
                "grow",
                "utility",
                "value",
                "experience",
                "strategy",
                "vision",
                "roadmap",
                "impact",
                "outcome",
                "business",
                "customer",
                "user",
                "market",
                "revenue",
                "adoption",
                "engagement",
            }
            strategic_matches = sum(1 for term in strategic_terms if term in task_lower)
            if strategic_matches >= 1:
                # At least one strategic term + no file refs = high-level goal
                # Scale down bonus when many keywords are present (more concrete)
                base_bonus = 2.0 + min(strategic_matches - 1, 2) * 0.5
                specificity_discount = min(keyword_score * 0.3, 1.5)
                vagueness_bonus = max(base_bonus - specificity_discount, 0.5)

        # Abstract/meta goal detection: goals that are exploratory, superlative,
        # or interrogative inherently require multi-step investigation and should
        # score high even without file mentions or technical keywords.
        abstract_bonus = 0.0

        # Superlative / exploratory action words that signal open-ended investigation
        _ABSTRACT_ACTION_WORDS = {
            "find",
            "discover",
            "identify",
            "investigate",
            "analyze",
            "analyse",
            "evaluate",
            "assess",
            "diagnose",
            "audit",
            "review",
            "survey",
            "explore",
            "determine",
            "prioritize",
            "rank",
        }
        _SUPERLATIVE_WORDS = {
            "best",
            "worst",
            "highest",
            "lowest",
            "most",
            "least",
            "biggest",
            "smallest",
            "top",
            "critical",
            "impactful",
            "important",
            "urgent",
            "fragile",
            "vulnerable",
            "risky",
        }
        _BROAD_SCOPE_WORDS = {
            "codebase",
            "system",
            "architecture",
            "project",
            "entire",
            "overall",
            "across",
            "everywhere",
            "all",
            "whole",
            "global",
        }

        # Strip punctuation from words for matching (e.g. "system?" -> "system")
        words_set = set(re.sub(r"[^\w]", "", w) for w in task_lower.split())
        has_abstract_action = bool(words_set & _ABSTRACT_ACTION_WORDS)
        has_superlative = bool(words_set & _SUPERLATIVE_WORDS)
        has_broad_scope = bool(words_set & _BROAD_SCOPE_WORDS)

        # Goals with abstract action + superlative (e.g. "find the highest-impact bug")
        if has_abstract_action and has_superlative and has_broad_scope:
            # All three signals: truly high-level investigative task
            abstract_bonus += 4.0
        elif has_abstract_action and has_superlative:
            abstract_bonus += 3.5
        elif has_abstract_action and has_broad_scope:
            abstract_bonus += 3.0
        elif has_superlative and has_broad_scope:
            abstract_bonus += 2.5
        elif has_abstract_action or has_superlative:
            abstract_bonus += 1.5

        # Strategic improvement verbs + domain concepts but no file targets:
        # e.g. "improve test coverage", "optimize performance", "enhance security"
        # These are broad directives requiring codebase-wide analysis.
        _STRATEGIC_IMPROVEMENT_VERBS = {
            "improve",
            "optimize",
            "optimise",
            "enhance",
            "increase",
            "reduce",
            "boost",
            "strengthen",
            "maximize",
            "minimise",
            "minimize",
        }
        has_strategic_verb = bool(words_set & _STRATEGIC_IMPROVEMENT_VERBS)
        has_concept = concept_score > 0
        if has_strategic_verb and has_concept and file_score == 0 and not has_path_ref:
            abstract_bonus += 2.0

        # Question-form goals (contain "?" or start with interrogative words)
        is_question = "?" in task
        interrogative_starts = {"what", "where", "which", "how", "why", "who"}
        first_word = task_lower.split()[0] if task_lower.split() else ""
        if is_question or first_word in interrogative_starts:
            abstract_bonus += 2.0

        # Broad scope words without file paths: "improve performance across the codebase"
        if has_broad_scope and file_score == 0 and not has_path_ref:
            abstract_bonus += 1.5

        # Discount the abstract bonus if the goal also has concrete file refs
        # (e.g. "find the best way to refactor auth.py" is more concrete)
        if file_score > 0 or has_path_ref:
            abstract_bonus *= 0.3

        # Combine scores with weights
        total = (
            file_score * self.config.file_complexity_weight * 10 / 3
            + keyword_score * self.config.concept_complexity_weight * 10 / 4
            + length_score * self.config.length_complexity_weight * 10 / 3
            + concept_score * 0.8
            + clause_score * 0.5
            + vagueness_bonus
            + abstract_bonus
        )

        # Add bonus for debate context if available
        if debate_result:
            consensus_text = getattr(debate_result, "consensus_text", "") or ""
            if len(consensus_text) > 500:
                total += 1

        return max(1, min(10, round(total)))

    _SPECIFIC_ACTION_VERBS = {
        "add",
        "remove",
        "fix",
        "update",
        "refactor",
        "implement",
        "replace",
        "rename",
        "extract",
        "move",
        "split",
        "merge",
        "delete",
        "create",
        "migrate",
        "convert",
        "wrap",
        "inject",
        "enable",
        "disable",
        "improve",
        "enhance",
        "optimize",
        "increase",
        "reduce",
        "test",
    }

    _SPECIFIC_TECHNICAL_TERMS = {
        "retry",
        "backoff",
        "timeout",
        "cache",
        "queue",
        "pool",
        "lock",
        "mutex",
        "batch",
        "stream",
        "parse",
        "serialize",
        "validate",
        "sanitize",
        "encrypt",
        "decrypt",
        "hash",
        "compress",
        "paginate",
        "throttle",
        "debounce",
        "middleware",
        "decorator",
        "hook",
        "callback",
        "handler",
        "endpoint",
        "route",
        "model",
        "schema",
        "coverage",
        "benchmark",
        "lint",
        "type-check",
        "typecheck",
        "migration",
        "fixture",
        "mock",
        "stub",
        "factory",
        "singleton",
    }

    def _is_specific_goal(self, goal: str) -> bool:
        """Check if a goal is specific enough to skip vague expansion.

        A goal is specific if it contains concrete action verbs AND technical
        terms that indicate the user knows exactly what they want, even if it
        doesn't mention file paths (which would raise the complexity score).
        """
        words = set(goal.lower().split())
        has_action = bool(words & self._SPECIFIC_ACTION_VERBS)
        has_technical = bool(words & self._SPECIFIC_TECHNICAL_TERMS)
        # Also check for module/area references (connectors, agents, etc.)
        has_module = bool(words & {c.lower() for c in DECOMPOSITION_CONCEPTS})
        # Specific if it has an action verb + either technical term or module ref
        return has_action and (has_technical or has_module)

    def _has_file_hints(self, goal: str) -> bool:
        """Check if a goal references specific files or directories.

        Used to determine if a goal is abstract (no file refs) vs concrete.
        """
        goal_lower = goal.lower()
        has_file_ext = bool(re.search(r"\b\w+\.(py|ts|tsx|js|jsx|md)\b", goal_lower))
        has_path_ref = bool(
            re.search(r"aragora/\w+|tests/\w+|sdk/\w+|scripts/\w+|src/\w+", goal_lower)
        )
        return has_file_ext or has_path_ref

    def _score_to_level(self, score: int) -> str:
        """Convert numeric score to complexity level."""
        if score <= 3:
            return "low"
        elif score <= 6:
            return "medium"
        else:
            return "high"

    def _build_rationale(
        self,
        task: str,
        score: int,
        should_decompose: bool,
    ) -> str:
        """Build explanation for decomposition decision."""
        task_lower = task.lower()

        reasons = []

        # Check for high complexity indicators
        high_keywords = [k for k in COMPLEXITY_INDICATORS["high"] if k in task_lower]
        if high_keywords:
            reasons.append(f"high-complexity keywords: {', '.join(high_keywords)}")

        # Check file count
        file_count = len(re.findall(r"\b\w+\.(py|ts|tsx|js|jsx|md)\b", task_lower))
        if file_count >= 3:
            reasons.append(f"touches {file_count} files")

        # Check concept breadth
        concepts = self._concept_pattern.findall(task_lower)
        unique_concepts = list(set(c.lower() for c in concepts))
        if len(unique_concepts) >= 2:
            reasons.append(f"spans concepts: {', '.join(unique_concepts)}")

        if should_decompose:
            return f"Decomposition recommended (score={score}): " + "; ".join(
                reasons or ["complexity exceeds threshold"]
            )
        else:
            return f"No decomposition needed (score={score})"

    def _generate_subtasks(
        self,
        task: str,
        debate_result: DebateResult | None = None,
    ) -> list[SubTask]:
        """Generate subtasks for a complex task.

        Uses heuristics to identify natural decomposition points.
        """
        subtasks: list[SubTask] = []

        # If AI extraction is available, use it
        if self._extract_subtasks_fn:
            try:
                extracted = self._extract_subtasks_fn(task)
                for i, st in enumerate(extracted[: self.config.max_subtasks]):
                    subtasks.append(
                        SubTask(
                            id=f"subtask_{i + 1}",
                            title=st.get("title", f"Subtask {i + 1}"),
                            description=st.get("description", ""),
                            dependencies=st.get("dependencies", []),
                            estimated_complexity=st.get("complexity", "medium"),
                            file_scope=st.get("files", []),
                        )
                    )
                if subtasks:
                    return subtasks
            except (RuntimeError, ValueError, KeyError) as e:
                logger.debug("AI subtask extraction failed: %s", e)

        # Fall back to heuristic decomposition
        return self._heuristic_decomposition(task, debate_result)

    def _heuristic_decomposition(
        self,
        task: str,
        debate_result: DebateResult | None = None,
    ) -> list[SubTask]:
        """Generate subtasks using heuristics.

        Looks for:
        1. Different concept areas mentioned
        2. Sequential steps implied
        3. File groupings
        """
        subtasks: list[SubTask] = []
        task_lower = task.lower()

        # Find concept areas in the task
        concepts = self._concept_pattern.findall(task_lower)
        unique_concepts = list(set(c.lower() for c in concepts))

        # Create subtasks for each major concept area
        for i, concept in enumerate(unique_concepts[: self.config.max_subtasks]):
            subtask_id = f"subtask_{i + 1}"

            # Extract relevant sentences for this concept
            sentences = task.split(".")
            relevant = [s.strip() for s in sentences if concept in s.lower()]
            description = ". ".join(relevant) if relevant else f"Handle {concept} changes"

            subtasks.append(
                SubTask(
                    id=subtask_id,
                    title=f"{concept.title()} Changes",
                    description=description,
                    dependencies=[f"subtask_{j + 1}" for j in range(i)],
                    estimated_complexity=self._estimate_concept_complexity(concept),
                    file_scope=self._find_files_for_concept(concept, task),
                )
            )

        # If no concepts found, create generic phases
        if not subtasks:
            subtasks = self._create_generic_phases(task)

        return subtasks[: self.config.max_subtasks]

    def _estimate_concept_complexity(self, concept: str) -> str:
        """Estimate complexity for a concept area."""
        high_complexity = {"database", "security", "architecture", "migration"}
        medium_complexity = {"api", "backend", "frontend", "performance"}

        if concept in high_complexity:
            return "high"
        elif concept in medium_complexity:
            return "medium"
        else:
            return "low"

    def _find_files_for_concept(self, concept: str, task: str) -> list[str]:
        """Find files mentioned in the task that relate to a concept."""
        files: list[str] = []

        # Map concepts to likely file patterns
        concept_patterns = {
            "database": r"(store|storage|db|model)\.py",
            "api": r"(handler|endpoint|route|api)\.py",
            "frontend": r"\.(tsx?|jsx?)$",
            "backend": r"(server|service|worker)\.py",
            "testing": r"test_\w+\.py",
            "security": r"(auth|security|rbac)\.py",
        }

        pattern = concept_patterns.get(concept, r"\.py$")
        matches = re.findall(rf"[\w/]+{pattern}", task, re.IGNORECASE)
        files.extend(matches)

        return list(set(files))[:5]

    def _create_generic_phases(self, task: str) -> list[SubTask]:
        """Create generic implementation phases when no concepts found."""
        return [
            SubTask(
                id="subtask_1",
                title="Analysis & Design",
                description="Analyze requirements and design the solution",
                dependencies=[],
                estimated_complexity="low",
            ),
            SubTask(
                id="subtask_2",
                title="Core Implementation",
                description="Implement the main functionality",
                dependencies=["subtask_1"],
                estimated_complexity="medium",
            ),
            SubTask(
                id="subtask_3",
                title="Testing & Integration",
                description="Write tests and integrate with existing code",
                dependencies=["subtask_2"],
                estimated_complexity="low",
            ),
        ]

    # =========================================================================
    # KM-informed subtask enrichment (async overlay)
    # =========================================================================

    async def enrich_subtasks_from_km(
        self,
        task: str,
        subtasks: list[SubTask],
    ) -> list[SubTask]:
        """Enrich subtasks with learnings from past Nomic cycles.

        Queries NomicCycleAdapter for similar past decompositions and
        recurring failures, then:
        - Adds failure warnings to success_criteria
        - Suggests additional subtasks learned from past cycles

        This is an async overlay — analyze() stays sync.

        Args:
            task: The original task description
            subtasks: Existing subtasks from analyze()

        Returns:
            Enriched list of subtasks (may include additions)
        """
        try:
            from aragora.knowledge.mound.adapters.nomic_cycle_adapter import (
                get_nomic_cycle_adapter,
            )

            adapter = get_nomic_cycle_adapter()

            # Query recurring failures relevant to this task
            try:
                failures = await adapter.find_recurring_failures(min_occurrences=2, limit=5)
                task_lower = task.lower()
                for failure in failures:
                    # Check if failure is relevant to this task's domain
                    pattern = failure.get("pattern", "").lower()
                    affected = failure.get("affected_tracks", [])

                    # Match if failure pattern shares words with task
                    pattern_words = set(pattern.split())
                    task_words = set(task_lower.split())
                    overlap = pattern_words & task_words
                    relevant_domain = any(track in task_lower for track in affected)

                    if overlap or relevant_domain:
                        # Add warning to all subtasks' success_criteria
                        warning = f"avoid: {failure['pattern'][:80]}"
                        for subtask in subtasks:
                            if "km_warnings" not in subtask.success_criteria:
                                subtask.success_criteria["km_warnings"] = []
                            if warning not in subtask.success_criteria["km_warnings"]:
                                subtask.success_criteria["km_warnings"].append(warning)

                if failures:
                    logger.info(
                        "km_enrichment_failures injected=%d warnings for task=%s",
                        len(failures),
                        task[:50],
                    )
            except (RuntimeError, ValueError, OSError) as e:
                logger.debug("KM failure query failed: %s", e)

            # Query high-ROI patterns to suggest focus areas
            try:
                high_roi = await adapter.find_high_roi_goal_types(limit=3)
                existing_titles = {s.title.lower() for s in subtasks}

                for roi in high_roi:
                    if roi.get("avg_improvement_score", 0) < 0.5:
                        continue

                    pattern = roi.get("pattern", "")
                    # Only add if not already covered by existing subtasks
                    if not any(
                        word in title
                        for title in existing_titles
                        for word in pattern.split()
                        if len(word) > 3
                    ):
                        # Add as a suggested subtask (capped at max_subtasks)
                        if len(subtasks) < self.config.max_subtasks:
                            example = roi.get("example_objectives", [""])[0]
                            subtasks.append(
                                SubTask(
                                    id=f"subtask_{len(subtasks) + 1}",
                                    title=f"KM-suggested: {pattern[:40]}",
                                    description=(
                                        f"Based on past success pattern "
                                        f"(avg improvement: {roi['avg_improvement_score']:.2f}). "
                                        f"Example: {example[:100]}"
                                    ),
                                    dependencies=[],
                                    estimated_complexity="medium",
                                    success_criteria={
                                        "km_source": "high_roi_pattern",
                                        "historical_improvement": roi["avg_improvement_score"],
                                    },
                                )
                            )

                if high_roi:
                    logger.info(
                        "km_enrichment_roi suggestions=%d for task=%s",
                        len(high_roi),
                        task[:50],
                    )
            except (RuntimeError, ValueError, OSError) as e:
                logger.debug("KM high-ROI query failed: %s", e)

        except ImportError:
            logger.debug("NomicCycleAdapter not available for KM enrichment")
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("KM enrichment failed: %s", e)

        return subtasks

    # =========================================================================
    # Oracle-driven validation (Tier 3)
    # =========================================================================

    def validate_file_independence(self, subtasks: list[SubTask]) -> list[FileConflict]:
        """Check that no two sibling subtasks share the same file in their file_scope.

        When subtasks share files, they cannot safely execute in parallel.
        This method detects overlaps and returns a list of conflicts so the
        caller can merge conflicting subtasks or mark them as sequential.

        Args:
            subtasks: List of subtasks to check for file overlaps.

        Returns:
            List of FileConflict objects describing which files overlap
            and which subtasks are involved. Empty list means all subtasks
            are file-independent and safe for parallel execution.
        """
        # Build a mapping from file -> list of subtask IDs that reference it
        file_to_subtasks: dict[str, list[str]] = {}
        for subtask in subtasks:
            for file_path in subtask.file_scope:
                normalized = file_path.rstrip("/")
                if normalized not in file_to_subtasks:
                    file_to_subtasks[normalized] = []
                file_to_subtasks[normalized].append(subtask.id)

        # Collect conflicts (files referenced by 2+ subtasks)
        conflicts: list[FileConflict] = []
        for file_path, subtask_ids in file_to_subtasks.items():
            if len(subtask_ids) > 1:
                conflicts.append(FileConflict(file_path=file_path, subtask_ids=list(subtask_ids)))

        if conflicts:
            logger.info(
                "file_independence_conflicts count=%d files=%s",
                len(conflicts),
                [c.file_path for c in conflicts],
            )
        else:
            logger.debug(
                "file_independence_ok subtasks=%d all_independent",
                len(subtasks),
            )

        return conflicts

    def validate_with_oracle(
        self,
        subtask: SubTask,
        worktree_path: str | None = None,
    ) -> OracleResult:
        """Run lightweight oracle checks on a subtask's file scope.

        Validates that the files referenced in the subtask's file_scope are
        coherent before agent execution begins. Checks include:
        - File existence and readability
        - Python syntax validation via ``ast.parse``

        This is an optional enhancement -- failures here indicate the task
        scope may reference non-existent or malformed files, but do not
        block execution.

        Args:
            subtask: The subtask whose file_scope to validate.
            worktree_path: Optional worktree root to resolve files against.
                If not provided, files are checked relative to cwd.

        Returns:
            OracleResult with validation status, errors, and checked files.
        """
        errors: list[str] = []
        checked_files: list[str] = []
        base_path = worktree_path or os.getcwd()

        for file_path in subtask.file_scope:
            # Skip directory-only scopes (e.g. "aragora/debate/")
            if file_path.endswith("/"):
                continue

            full_path = os.path.join(base_path, file_path)
            checked_files.append(file_path)

            # Check existence and readability
            if not os.path.isfile(full_path):
                errors.append(f"File not found: {file_path}")
                continue

            if not os.access(full_path, os.R_OK):
                errors.append(f"File not readable: {file_path}")
                continue

            # Python syntax check
            if file_path.endswith(".py"):
                try:
                    result = subprocess.run(  # noqa: S603 -- subprocess with fixed args, no shell
                        [
                            sys.executable,
                            "-c",
                            f"import ast; ast.parse(open({full_path!r}).read())",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode != 0:
                        stderr = result.stderr.strip()
                        errors.append(f"Syntax error in {file_path}: {stderr[:200]}")
                except subprocess.TimeoutExpired:
                    errors.append(f"Syntax check timed out: {file_path}")
                except OSError as e:
                    errors.append(f"Syntax check failed for {file_path}: {e}")

        valid = len(errors) == 0
        if not valid:
            logger.info(
                "oracle_validation_failed subtask=%s errors=%d: %s",
                subtask.id,
                len(errors),
                errors[:3],
            )
        else:
            logger.debug(
                "oracle_validation_passed subtask=%s checked=%d files",
                subtask.id,
                len(checked_files),
            )

        return OracleResult(valid=valid, errors=errors, checked_files=checked_files)

    def score_decomposition(
        self,
        subtasks: list[SubTask],
        original_file_scope: list[str] | None = None,
    ) -> DecompositionQuality:
        """Score the quality of a decomposition.

        Evaluates the subtask set on three dimensions:
        - **File independence** (no overlaps): penalizes shared files
        - **Granularity** (1-5 files per task): rewards focused subtasks
        - **Coverage** (union of scopes covers original goal): rewards completeness

        Args:
            subtasks: List of subtasks to score.
            original_file_scope: Optional list of files from the original goal.
                Used to compute coverage_ratio. If not provided, coverage
                is estimated from the subtask union.

        Returns:
            DecompositionQuality with score (0.0-1.0) and component metrics.
        """
        if not subtasks:
            return DecompositionQuality(
                score=0.0, file_conflicts=0, avg_scope_size=0.0, coverage_ratio=0.0
            )

        # File independence score
        conflicts = self.validate_file_independence(subtasks)
        file_conflicts = len(conflicts)
        # Penalize: 0 conflicts = 1.0, each conflict reduces by 0.15
        independence_score = max(0.0, 1.0 - file_conflicts * 0.15)

        # Granularity score: ideal is 1-5 files per subtask
        scope_sizes = [len(s.file_scope) for s in subtasks]
        avg_scope_size = sum(scope_sizes) / len(scope_sizes) if scope_sizes else 0.0
        # Score: 1-5 files = 1.0, 0 files = 0.5 (no scope defined), >5 = decreasing
        if avg_scope_size == 0:
            granularity_score = 0.5  # No file scope defined is neutral
        elif 1 <= avg_scope_size <= 5:
            granularity_score = 1.0
        elif avg_scope_size > 5:
            # Use a steeper drop for oversized scopes so broad subtasks are
            # materially penalized in the combined quality score.
            granularity_score = max(0.1, 1.0 - (avg_scope_size - 5) * 0.15)
        else:
            granularity_score = 0.5

        # Coverage score
        all_subtask_files: set[str] = set()
        for s in subtasks:
            all_subtask_files.update(f.rstrip("/") for f in s.file_scope)

        if original_file_scope:
            original_set = {f.rstrip("/") for f in original_file_scope}
            if original_set:
                covered = all_subtask_files & original_set
                coverage_ratio = len(covered) / len(original_set)
            else:
                coverage_ratio = 1.0
        else:
            # Without an original scope, use subtask count as proxy
            # More subtasks with files = better coverage
            has_scope = sum(1 for s in subtasks if s.file_scope)
            coverage_ratio = has_scope / len(subtasks) if subtasks else 0.0

        # Weighted combination
        score = independence_score * 0.4 + granularity_score * 0.3 + coverage_ratio * 0.3

        logger.info(
            "decomposition_quality score=%.2f independence=%.2f granularity=%.2f "
            "coverage=%.2f conflicts=%d avg_scope=%.1f",
            score,
            independence_score,
            granularity_score,
            coverage_ratio,
            file_conflicts,
            avg_scope_size,
        )

        return DecompositionQuality(
            score=round(score, 3),
            file_conflicts=file_conflicts,
            avg_scope_size=round(avg_scope_size, 2),
            coverage_ratio=round(coverage_ratio, 3),
        )

    # =========================================================================
    # Codebase module mapping for relevance scoring
    # =========================================================================

    _CODEBASE_MODULES: dict[str, list[str]] = {
        "debate": ["aragora/debate/"],
        "agents": ["aragora/agents/"],
        "analytics": ["aragora/analytics/"],
        "audit": ["aragora/audit/"],
        "billing": ["aragora/billing/"],
        "cli": ["aragora/cli/"],
        "compliance": ["aragora/compliance/"],
        "connectors": ["aragora/connectors/"],
        "gateway": ["aragora/gateway/"],
        "knowledge": ["aragora/knowledge/"],
        "memory": ["aragora/memory/"],
        "nomic": ["aragora/nomic/"],
        "pipeline": ["aragora/pipeline/"],
        "rbac": ["aragora/rbac/"],
        "security": ["aragora/security/"],
        "server": ["aragora/server/"],
        "skills": ["aragora/skills/"],
        "storage": ["aragora/storage/"],
        "workflow": ["aragora/workflow/"],
        "frontend": ["aragora/live/src/"],
        "sdk": ["sdk/"],
        "tests": ["tests/"],
    }

    def _score_codebase_relevance(self, goal: str) -> list[str]:
        """Find relevant codebase directories for a goal using keyword matching.

        Parses the goal against the CLAUDE.md module table to suggest file
        scopes for matched templates, turning abstract matches into
        actionable subtasks with file_scope populated.

        Args:
            goal: The goal string to find relevant directories for

        Returns:
            List of up to 5 relevant codebase directory paths
        """
        goal_lower = goal.lower()
        relevant: list[str] = []
        for module, paths in self._CODEBASE_MODULES.items():
            if module in goal_lower:
                relevant.extend(paths)
        return relevant[:5]

    # =========================================================================
    # Vague goal expansion (cross-references templates + track configs)
    # =========================================================================

    def _expand_vague_goal(self, goal: str) -> TaskDecomposition | None:
        """Expand a vague goal into concrete subtasks using templates and tracks.

        When a goal like "maximize utility for SMEs" scores low on the heuristic
        complexity check (no file mentions, few keywords), this method cross-
        references deliberation templates and development track configs to
        generate concrete, actionable subtasks.

        Args:
            goal: The vague goal string

        Returns:
            TaskDecomposition with expanded subtasks, or None if expansion
            didn't produce useful results
        """
        subtasks: list[SubTask] = []
        matched_sources: list[str] = []

        # Compute codebase-relevant directories from the goal text
        goal_relevant_paths = self._score_codebase_relevance(goal)

        # 1. Cross-reference against deliberation templates
        try:
            from aragora.deliberation.templates.registry import match_templates

            matched = match_templates(goal, limit=3)
            for i, template in enumerate(matched):
                # Derive file_scope from template tags + codebase module mapping
                tag_paths: list[str] = []
                for tag in template.tags:
                    tag_lower = tag.lower()
                    if tag_lower in self._CODEBASE_MODULES:
                        tag_paths.extend(self._CODEBASE_MODULES[tag_lower])
                # Combine tag-derived paths with goal-derived paths, deduplicate
                combined_paths = list(dict.fromkeys(tag_paths + goal_relevant_paths))[:5]

                subtasks.append(
                    SubTask(
                        id=f"subtask_{len(subtasks) + 1}",
                        title=f"{template.name.replace('_', ' ').title()}",
                        description=(
                            f"{template.description}. "
                            f"Suggested personas: {', '.join(template.personas[:3])}."
                            if template.personas
                            else template.description
                        ),
                        dependencies=[f"subtask_{len(subtasks)}"] if subtasks else [],
                        estimated_complexity="medium",
                        file_scope=combined_paths,
                    )
                )
                matched_sources.append(f"template:{template.name}")
        except ImportError:
            logger.debug("Deliberation templates not available for expansion")

        # 2. Cross-reference against development track configs
        try:
            from aragora.nomic.autonomous_orchestrator import (
                DEFAULT_TRACK_CONFIGS,
                Track,
            )

            goal_lower = goal.lower()
            track_keywords = {
                Track.SME: ["sme", "small business", "dashboard", "user experience", "utility"],
                Track.DEVELOPER: ["sdk", "api", "developer", "documentation", "package"],
                Track.SELF_HOSTED: ["deploy", "docker", "self-hosted", "ops", "backup"],
                Track.QA: ["test", "quality", "coverage", "ci", "reliability"],
                Track.CORE: ["debate", "agent", "consensus", "engine", "core"],
                Track.SECURITY: ["security", "auth", "vulnerability", "harden", "owasp"],
            }
            # Count how many tracks match explicitly
            matched_tracks = [
                track
                for track in DEFAULT_TRACK_CONFIGS
                if any(kw in goal_lower for kw in track_keywords.get(track, []))
            ]
            # If 0-1 tracks match, the goal is so broad it affects all tracks.
            # Strategic terms like "maximize", "improve", "optimize" are
            # inherently cross-cutting — include all tracks.
            broad_terms = {
                "maximize",
                "minimise",
                "minimize",
                "improve",
                "enhance",
                "optimize",
                "optimise",
                "scale",
                "transform",
                "grow",
                "utility",
                "value",
                "business",
            }
            is_broad = any(t in goal_lower for t in broad_terms)
            # Also check if the goal mentions a specific path/directory
            has_path = bool(re.search(r"aragora/\w+|tests/\w+|sdk/\w+|scripts/\w+", goal_lower))
            if len(matched_tracks) == 0 and is_broad and not has_path:
                # Truly broad goal with no specific track or path — all tracks
                tracks_to_expand = list(DEFAULT_TRACK_CONFIGS.keys())[:4]
            elif len(matched_tracks) == 1 and is_broad and not has_path:
                # Broad but slightly focused — matched track + 2 adjacent
                all_tracks = list(DEFAULT_TRACK_CONFIGS.keys())
                idx = all_tracks.index(matched_tracks[0])
                extra = [t for i, t in enumerate(all_tracks) if i != idx][:2]
                tracks_to_expand = matched_tracks + extra
            else:
                tracks_to_expand = matched_tracks

            for track in tracks_to_expand:
                config = DEFAULT_TRACK_CONFIGS[track]
                folders_str = ", ".join(config.folders[:3])
                subtasks.append(
                    SubTask(
                        id=f"subtask_{len(subtasks) + 1}",
                        title=f"Improve {config.name} Track",
                        description=(
                            f"Enhance capabilities in the {config.name} track. "
                            f"Key folders: {folders_str}. "
                            f"Preferred agents: {', '.join(config.agent_types)}."
                        ),
                        dependencies=[],
                        estimated_complexity="medium",
                        file_scope=config.folders[:3],
                    )
                )
                matched_sources.append(f"track:{track.value}")
        except ImportError:
            logger.debug("Track configs not available for expansion")

        # Only return expansion if we found meaningful matches
        if len(subtasks) < 2:
            return None

        # Cap at max_subtasks
        subtasks = subtasks[: self.config.max_subtasks]

        rationale = (
            f"Vague goal expanded via semantic matching (sources: {', '.join(matched_sources)})"
        )

        return TaskDecomposition(
            original_task=goal,
            complexity_score=5,  # Elevated: vague goals are inherently complex
            complexity_level="medium",
            should_decompose=True,
            subtasks=subtasks,
            rationale=rationale,
        )

    # =========================================================================
    # Debate-based decomposition (for abstract high-level goals)
    # =========================================================================

    async def analyze_with_debate(
        self,
        goal: str,
        agents: list[Any] | None = None,
        context: str = "",
        depth: int = 0,
    ) -> TaskDecomposition:
        """Analyze an abstract goal using multi-agent debate.

        Uses Arena debate to decompose high-level goals like "Maximize utility
        for SME businesses" into concrete, actionable subtasks. Multiple agents
        debate what improvements would best serve the goal and reach consensus.

        This is more powerful than heuristic decomposition for abstract goals
        but uses more tokens and takes longer.

        Args:
            goal: High-level goal to decompose (can be abstract)
            agents: Optional list of agents to use in debate. If not provided,
                   will use default API agents.
            context: Optional additional context about the codebase or project
            depth: Current recursion depth (0 = top-level)

        Returns:
            TaskDecomposition with debate-derived subtasks

        Example:
            decomposer = TaskDecomposer()
            result = await decomposer.analyze_with_debate(
                "Maximize utility for SME businesses"
            )
            for subtask in result.subtasks:
                print(f"  - {subtask.title}: {subtask.description}")
        """
        # Enforce depth limit
        if depth >= self.config.max_depth:
            logger.info(
                "debate_decomposition_depth_limit depth=%s max=%s", depth, self.config.max_depth
            )
            return self.analyze(goal, depth=self.config.max_depth)
        from aragora.core import Environment
        from aragora.debate.protocol import DebateProtocol

        # Build the debate task - ask agents to decompose the goal
        debate_task = self._build_debate_task(goal, context)

        # Get agents if not provided
        if agents is None:
            agents = await self._get_default_agents()

        # Configure debate protocol for decomposition with Trickster
        # and convergence detection for higher-quality consensus
        protocol = DebateProtocol(
            rounds=self.config.debate_rounds,
            consensus="majority",
            timeout_seconds=self.config.debate_timeout,
            enable_trickster=self.config.enable_trickster,
            trickster_sensitivity=self.config.trickster_sensitivity,
            convergence_detection=self.config.enable_convergence,
        )

        # Create environment
        env = Environment(
            task=debate_task,
            context=context,
            max_rounds=self.config.debate_rounds,
            require_consensus=True,
            consensus_threshold=0.6,
        )

        logger.info("debate_decomposition_started goal=%s...", goal[:50])

        # Try to run debate, with OpenRouter fallback on API errors
        result = await self._run_debate_with_fallback(env, agents, protocol, goal, context)

        if result is None:
            # All attempts failed, fall back to heuristic
            logger.warning("debate_decomposition_all_failed falling back to heuristic")
            return self.analyze(goal, depth=depth)

        # Parse subtasks from final answer (consensus text)
        subtasks = self._parse_debate_subtasks(result.final_answer or "")

        if not subtasks:
            logger.warning("debate_decomposition_empty falling back to heuristic")
            subtasks = self._create_generic_phases(goal)

        logger.info(
            f"debate_decomposition_completed subtasks={len(subtasks)} "
            f"confidence={result.confidence:.2f}"
        )

        return TaskDecomposition(
            original_task=goal,
            complexity_score=8,  # Debate implies high complexity
            complexity_level="high",
            should_decompose=True,
            subtasks=subtasks[: self.config.max_subtasks],
            rationale=f"Debate decomposition (confidence={result.confidence:.2f}): "
            + (result.final_answer or "")[:200],
        )

    async def _run_debate_with_fallback(
        self,
        env: Environment,
        agents: list[Any],
        protocol: Any,
        goal: str,
        context: str,
    ) -> Any | None:
        """Run debate with OpenRouter fallback on API errors or poor output.

        The fallback triggers when:
        1. AgentAPIError, AgentRateLimitError, or similar billing errors occur
        2. The debate returns but the output doesn't contain valid subtasks

        Returns:
            DebateResult if successful with valid subtasks, None if all attempts failed
        """
        from aragora.agents.errors.exceptions import (
            AgentAPIError,
            AgentError,
            AgentRateLimitError,
        )
        from aragora.debate.orchestrator import Arena

        result = None
        should_fallback = False
        fallback_reason = ""

        # First attempt with provided agents
        try:
            arena = Arena(env, agents, protocol)
            result = await arena.run()

            # Check if the result has valid, parseable subtasks
            if result and result.final_answer:
                subtasks = self._parse_debate_subtasks(result.final_answer)
                if subtasks:
                    logger.info(
                        f"debate_primary_succeeded subtasks={len(subtasks)} "
                        f"confidence={result.confidence:.2f}"
                    )
                    return result
                else:
                    # Debate completed but output is not useful
                    should_fallback = True
                    fallback_reason = "output has no parseable subtasks"
            else:
                should_fallback = True
                fallback_reason = "no final answer"

        except AgentRateLimitError as e:
            should_fallback = True
            fallback_reason = f"rate limit: {e}"
        except AgentAPIError as e:
            error_msg = str(e).lower()
            # Check for billing/quota errors that warrant fallback
            if any(
                keyword in error_msg
                for keyword in [
                    "credit",
                    "balance",
                    "quota",
                    "rate limit",
                    "billing",
                    "insufficient",
                ]
            ):
                should_fallback = True
                fallback_reason = f"billing error: {e}"
            else:
                # Other API errors might not benefit from fallback
                logger.exception("debate_api_error error=%s", e)
                return None
        except AgentError as e:
            # Generic agent error - try fallback
            should_fallback = True
            fallback_reason = f"agent error: {e}"
        except (RuntimeError, OSError, ConnectionError, TimeoutError) as e:
            # Check if exception message indicates billing/API issues
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in [
                    "credit",
                    "balance",
                    "quota",
                    "rate limit",
                    "billing",
                    "insufficient",
                    "401",
                    "403",
                ]
            ):
                should_fallback = True
                fallback_reason = f"api error: {e}"
            else:
                logger.exception("debate_failed error=%s", e)
                return None

        if not should_fallback:
            return result

        # Fallback: try with OpenRouter agents
        logger.warning("debate_fallback_triggered reason=%s", fallback_reason)

        try:
            fallback_agents = await self._get_openrouter_agents()
            if not fallback_agents:
                logger.warning("debate_no_fallback_agents OpenRouter not available")
                # Return original result if we have one (better than nothing)
                return result

            logger.info("debate_fallback_started agents=%s", len(fallback_agents))

            # Rebuild environment and protocol for fresh debate
            from aragora.core import Environment
            from aragora.debate.protocol import DebateProtocol

            fallback_env = Environment(
                task=self._build_debate_task(goal, context),
                context=context,
                max_rounds=self.config.debate_rounds,
                require_consensus=True,
                consensus_threshold=0.6,
            )
            fallback_protocol = DebateProtocol(
                rounds=self.config.debate_rounds,
                consensus="majority",
                timeout_seconds=self.config.debate_timeout,
                enable_trickster=self.config.enable_trickster,
                trickster_sensitivity=self.config.trickster_sensitivity,
                convergence_detection=self.config.enable_convergence,
            )

            arena = Arena(fallback_env, fallback_agents, fallback_protocol)
            fallback_result = await arena.run()

            # Check if fallback result is better
            if fallback_result and fallback_result.final_answer:
                subtasks = self._parse_debate_subtasks(fallback_result.final_answer)
                if subtasks:
                    logger.info(
                        f"debate_fallback_succeeded subtasks={len(subtasks)} "
                        f"confidence={fallback_result.confidence:.2f}"
                    )
                    return fallback_result

            logger.warning("debate_fallback_no_subtasks returning original result")
            return result or fallback_result

        except (RuntimeError, OSError, ConnectionError, TimeoutError) as e:
            logger.exception("debate_fallback_failed error=%s", e)
            # Return original result if we have one
            return result

    async def _get_openrouter_agents(self) -> list[Any]:
        """Get OpenRouter agents for fallback."""
        from aragora.config.secrets import get_secret

        openrouter_key = get_secret("OPENROUTER_API_KEY")
        if not openrouter_key:
            return []

        try:
            from aragora.agents.api_agents.openrouter import OpenRouterAgent

            # Set the API key in environment for OpenRouterAgent
            import os

            os.environ["OPENROUTER_API_KEY"] = openrouter_key

            return [
                OpenRouterAgent(
                    name="or-claude",
                    model="anthropic/claude-sonnet-4",
                ),
                OpenRouterAgent(
                    name="or-gpt",
                    model="openai/gpt-4o",
                ),
            ]
        except (ImportError, RuntimeError, OSError) as e:
            logger.warning("openrouter_agents_failed error=%s", e)
            return []

    def _build_debate_task(self, goal: str, context: str = "") -> str:
        """Build the debate task prompt for goal decomposition."""
        # Always include aragora codebase structure
        codebase_context = """
CODEBASE STRUCTURE (Aragora project):
- aragora/workflow/templates/ - Workflow template definitions
- aragora/workflow/engine.py - Workflow execution engine
- aragora/workflow/types.py - WorkflowDefinition, StepDefinition types
- aragora/server/handlers/ - HTTP API handlers
- aragora/live/ - Next.js frontend (in aragora/live/src/)
- aragora/nomic/ - Nomic loop and autonomous orchestration
- tests/ - Test files (tests/workflow/, tests/nomic/, etc.)
- docs/ - Documentation

FILE PATH CONVENTIONS:
- Python backend: aragora/module/file.py (NOT src/)
- TypeScript frontend: aragora/live/src/components/, aragora/live/src/app/
- Tests: tests/module/test_file.py
- Workflows: aragora/workflow/templates/category/template.py
"""
        user_context = f"\n\nAdditional Context:\n{context}" if context else ""

        return f"""Decompose this high-level goal into 3-5 concrete, actionable subtasks.

GOAL: {goal}
{codebase_context}{user_context}

For each subtask, provide:
1. A clear title (2-5 words)
2. A specific description of what needs to be done
3. Estimated complexity (low/medium/high)
4. Files or areas likely affected (use ACTUAL aragora paths, NOT src/)
5. Dependencies on other subtasks (if any)

Format your response as a JSON array:
```json
[
  {{
    "title": "Subtask Title",
    "description": "Specific description of what to implement",
    "complexity": "medium",
    "files": ["aragora/path/to/file.py", "aragora/live/src/file.tsx"],
    "dependencies": []
  }},
  ...
]
```

Focus on:
- Concrete, implementable tasks (not abstract goals)
- Clear boundaries between subtasks
- Parallelizable work where possible
- Use the ACTUAL aragora file paths shown above

Prioritize by impact: which improvements would provide the most value?"""

    def _parse_debate_subtasks(self, consensus_text: str) -> list[SubTask]:
        """Parse subtasks from debate consensus text."""
        subtasks: list[SubTask] = []

        # Try to extract JSON from the consensus
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", consensus_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON array directly
            json_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", consensus_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.debug("No JSON found in debate consensus")
                return subtasks

        try:
            parsed = json.loads(json_str)
            if not isinstance(parsed, list):
                parsed = [parsed]

            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue

                subtasks.append(
                    SubTask(
                        id=f"subtask_{i + 1}",
                        title=item.get("title", f"Subtask {i + 1}"),
                        description=item.get("description", ""),
                        dependencies=item.get("dependencies", []),
                        estimated_complexity=item.get("complexity", "medium"),
                        file_scope=item.get("files", []),
                    )
                )

        except json.JSONDecodeError as e:
            logger.debug("Failed to parse debate JSON: %s", e)

        return subtasks

    async def _get_default_agents(self) -> list[Any]:
        """Get default agents for debate decomposition.

        Uses aragora.config.secrets to load API keys from AWS Secrets Manager
        or environment variables.
        """
        from aragora.config.secrets import get_secret
        from aragora.agents.api_agents.base import APIAgent

        agents: list[APIAgent] = []
        errors: list[str] = []

        # Try Anthropic agents first (pass API key explicitly)
        anthropic_key = get_secret("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

                agents.extend(
                    [
                        AnthropicAPIAgent(
                            name="claude-strategist",
                            model="claude-sonnet-4-20250514",
                            api_key=anthropic_key,
                        ),
                        AnthropicAPIAgent(
                            name="claude-architect",
                            model="claude-sonnet-4-20250514",
                            api_key=anthropic_key,
                        ),
                    ]
                )
            except (ImportError, RuntimeError, OSError) as e:
                errors.append(f"Anthropic: {e}")

        # Try OpenAI agents (pass API key explicitly)
        openai_key = get_secret("OPENAI_API_KEY")
        if openai_key:
            try:
                from aragora.agents.api_agents.openai import OpenAIAPIAgent

                agents.append(
                    OpenAIAPIAgent(name="gpt-analyst", model="gpt-4o", api_key=openai_key)
                )
            except (ImportError, RuntimeError, OSError) as e:
                errors.append(f"OpenAI: {e}")

        # Try OpenRouter as fallback (pass API key explicitly)
        openrouter_key = get_secret("OPENROUTER_API_KEY")
        if not agents and openrouter_key:
            try:
                from aragora.agents.api_agents.openrouter import OpenRouterAgent

                # OpenRouterAgent uses OPENROUTER_API_KEY from environment
                agents.extend(
                    [
                        OpenRouterAgent(
                            name="or-claude",
                            model="anthropic/claude-sonnet-4",
                        ),
                        OpenRouterAgent(
                            name="or-gpt",
                            model="openai/gpt-4o",
                        ),
                    ]
                )
            except (ImportError, RuntimeError, OSError) as e:
                errors.append(f"OpenRouter: {e}")

        if not agents:
            raise RuntimeError(
                "No API agents available for debate decomposition.\n"
                "Required: ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY\n"
                "For AWS Secrets Manager: set ARAGORA_USE_SECRETS_MANAGER=true\n"
                f"Errors: {'; '.join(errors) if errors else 'No API keys found'}"
            )

        logger.info("debate_agents_loaded count=%s", len(agents))
        return agents


# Module-level singleton
_decomposer: TaskDecomposer | None = None


def get_task_decomposer() -> TaskDecomposer:
    """Get or create the singleton TaskDecomposer instance."""
    global _decomposer
    if _decomposer is None:
        _decomposer = TaskDecomposer()
    return _decomposer


def analyze_task(task: str) -> TaskDecomposition:
    """Convenience function to analyze a task."""
    return get_task_decomposer().analyze(task)
