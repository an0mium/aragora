"""Task decomposition for Nomic Loop.

Analyzes task complexity and decomposes large tasks into smaller subtasks
for parallel or sequential processing.

Supports two decomposition modes:
1. Heuristic: Fast pattern-matching for concrete goals with file mentions
2. Debate: Multi-agent Arena debate for abstract high-level goals

Integrates with workflow patterns for execution strategies.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.core import DebateResult

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """A subtask extracted from a larger task."""

    id: str
    title: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    estimated_complexity: str = "low"  # low, medium, high
    file_scope: List[str] = field(default_factory=list)


@dataclass
class TaskDecomposition:
    """Result of task decomposition analysis."""

    original_task: str
    complexity_score: int  # 1-10
    complexity_level: str  # low, medium, high
    should_decompose: bool
    subtasks: List[SubTask] = field(default_factory=list)
    rationale: str = ""


@dataclass
class DecomposerConfig:
    """Configuration for TaskDecomposer."""

    complexity_threshold: int = 5  # Score above which decomposition is triggered
    max_subtasks: int = 5
    min_subtasks: int = 2
    file_complexity_weight: float = 0.3
    concept_complexity_weight: float = 0.4
    length_complexity_weight: float = 0.3
    # Debate-based decomposition settings
    debate_rounds: int = 2  # Rounds for goal decomposition debate
    debate_timeout: int = 120  # Timeout in seconds for debate


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
    ],
    "medium": [
        "integrate",
        "implement",
        "add",
        "create",
        "build",
        "enhance",
        "extend",
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
    "testing",
    "security",
    "performance",
    "documentation",
    "configuration",
    "deployment",
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
        config: Optional[DecomposerConfig] = None,
        extract_subtasks_fn: Optional[Callable[[str], List[Dict]]] = None,
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
        debate_result: Optional["DebateResult"] = None,
    ) -> TaskDecomposition:
        """Analyze a task and determine if decomposition is needed.

        Args:
            task_description: The task or improvement proposal
            debate_result: Optional debate result for additional context

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

        # Calculate complexity score
        complexity_score = self._calculate_complexity(task_description, debate_result)
        complexity_level = self._score_to_level(complexity_score)

        # Determine if decomposition is needed
        should_decompose = complexity_score >= self.config.complexity_threshold

        # Build rationale
        rationale = self._build_rationale(task_description, complexity_score, should_decompose)

        result = TaskDecomposition(
            original_task=task_description,
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            should_decompose=should_decompose,
            rationale=rationale,
        )

        # Extract subtasks if decomposition is needed
        if should_decompose:
            result.subtasks = self._generate_subtasks(task_description, debate_result)
            logger.info(
                f"task_decomposed complexity={complexity_score} " f"subtasks={len(result.subtasks)}"
            )
        else:
            logger.debug(
                f"task_not_decomposed complexity={complexity_score} "
                f"threshold={self.config.complexity_threshold}"
            )

        return result

    def _calculate_complexity(
        self,
        task: str,
        debate_result: Optional["DebateResult"] = None,
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
        length_score = min(word_count / 100, 3)

        # Combine scores with weights
        total = (
            file_score * self.config.file_complexity_weight * 10 / 3
            + keyword_score * self.config.concept_complexity_weight * 10 / 4
            + length_score * self.config.length_complexity_weight * 10 / 3
        )

        # Add bonus for debate context if available
        if debate_result:
            consensus_text = getattr(debate_result, "consensus_text", "") or ""
            if len(consensus_text) > 500:
                total += 1

        return max(1, min(10, round(total)))

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
        debate_result: Optional["DebateResult"] = None,
    ) -> List[SubTask]:
        """Generate subtasks for a complex task.

        Uses heuristics to identify natural decomposition points.
        """
        subtasks: List[SubTask] = []

        # If AI extraction is available, use it
        if self._extract_subtasks_fn:
            try:
                extracted = self._extract_subtasks_fn(task)
                for i, st in enumerate(extracted[: self.config.max_subtasks]):
                    subtasks.append(
                        SubTask(
                            id=f"subtask_{i+1}",
                            title=st.get("title", f"Subtask {i+1}"),
                            description=st.get("description", ""),
                            dependencies=st.get("dependencies", []),
                            estimated_complexity=st.get("complexity", "medium"),
                            file_scope=st.get("files", []),
                        )
                    )
                if subtasks:
                    return subtasks
            except Exception as e:
                logger.debug(f"AI subtask extraction failed: {e}")

        # Fall back to heuristic decomposition
        return self._heuristic_decomposition(task, debate_result)

    def _heuristic_decomposition(
        self,
        task: str,
        debate_result: Optional["DebateResult"] = None,
    ) -> List[SubTask]:
        """Generate subtasks using heuristics.

        Looks for:
        1. Different concept areas mentioned
        2. Sequential steps implied
        3. File groupings
        """
        subtasks: List[SubTask] = []
        task_lower = task.lower()

        # Find concept areas in the task
        concepts = self._concept_pattern.findall(task_lower)
        unique_concepts = list(set(c.lower() for c in concepts))

        # Create subtasks for each major concept area
        for i, concept in enumerate(unique_concepts[: self.config.max_subtasks]):
            subtask_id = f"subtask_{i+1}"

            # Extract relevant sentences for this concept
            sentences = task.split(".")
            relevant = [s.strip() for s in sentences if concept in s.lower()]
            description = ". ".join(relevant) if relevant else f"Handle {concept} changes"

            subtasks.append(
                SubTask(
                    id=subtask_id,
                    title=f"{concept.title()} Changes",
                    description=description,
                    dependencies=[f"subtask_{j+1}" for j in range(i)],
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

    def _find_files_for_concept(self, concept: str, task: str) -> List[str]:
        """Find files mentioned in the task that relate to a concept."""
        files: List[str] = []

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

    def _create_generic_phases(self, task: str) -> List[SubTask]:
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
    # Debate-based decomposition (for abstract high-level goals)
    # =========================================================================

    async def analyze_with_debate(
        self,
        goal: str,
        agents: Optional[List[Any]] = None,
        context: str = "",
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
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        # Build the debate task - ask agents to decompose the goal
        debate_task = self._build_debate_task(goal, context)

        # Get agents if not provided
        if agents is None:
            agents = await self._get_default_agents()

        # Configure debate protocol for decomposition
        protocol = DebateProtocol(
            rounds=self.config.debate_rounds,
            consensus="majority",
            timeout_seconds=self.config.debate_timeout,
        )

        # Create environment
        env = Environment(
            task=debate_task,
            context=context,
            max_rounds=self.config.debate_rounds,
            require_consensus=True,
            consensus_threshold=0.6,
        )

        logger.info(f"debate_decomposition_started goal={goal[:50]}...")

        try:
            # Run the debate
            arena = Arena(env, agents, protocol)
            result = await arena.run()

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

        except Exception as e:
            logger.exception(f"debate_decomposition_failed error={e}")
            # Fall back to heuristic analysis
            return self.analyze(goal)

    def _build_debate_task(self, goal: str, context: str = "") -> str:
        """Build the debate task prompt for goal decomposition."""
        context_section = f"\n\nContext:\n{context}" if context else ""

        return f"""Decompose this high-level goal into 3-5 concrete, actionable subtasks.

GOAL: {goal}
{context_section}

For each subtask, provide:
1. A clear title (2-5 words)
2. A specific description of what needs to be done
3. Estimated complexity (low/medium/high)
4. Files or areas likely affected
5. Dependencies on other subtasks (if any)

Format your response as a JSON array:
```json
[
  {{
    "title": "Subtask Title",
    "description": "Specific description of what to implement",
    "complexity": "medium",
    "files": ["path/to/file.py", "another/file.tsx"],
    "dependencies": []
  }},
  ...
]
```

Focus on:
- Concrete, implementable tasks (not abstract goals)
- Clear boundaries between subtasks
- Parallelizable work where possible
- Practical files and areas in a typical codebase

Prioritize by impact: which improvements would provide the most value?"""

    def _parse_debate_subtasks(self, consensus_text: str) -> List[SubTask]:
        """Parse subtasks from debate consensus text."""
        subtasks: List[SubTask] = []

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
            logger.debug(f"Failed to parse debate JSON: {e}")

        return subtasks

    async def _get_default_agents(self) -> List[Any]:
        """Get default agents for debate decomposition.

        Uses aragora.config.secrets to load API keys from AWS Secrets Manager
        or environment variables.
        """
        from aragora.config.secrets import get_secret

        agents = []
        errors = []

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
            except Exception as e:
                errors.append(f"Anthropic: {e}")

        # Try OpenAI agents (pass API key explicitly)
        openai_key = get_secret("OPENAI_API_KEY")
        if openai_key:
            try:
                from aragora.agents.api_agents.openai import OpenAIAPIAgent

                agents.append(
                    OpenAIAPIAgent(name="gpt-analyst", model="gpt-4o", api_key=openai_key)
                )
            except Exception as e:
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
                            model="anthropic/claude-3.5-sonnet",
                        ),
                        OpenRouterAgent(
                            name="or-gpt",
                            model="openai/gpt-4o",
                        ),
                    ]
                )
            except Exception as e:
                errors.append(f"OpenRouter: {e}")

        if not agents:
            raise RuntimeError(
                "No API agents available for debate decomposition.\n"
                "Required: ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY\n"
                "For AWS Secrets Manager: set ARAGORA_USE_SECRETS_MANAGER=true\n"
                f"Errors: {'; '.join(errors) if errors else 'No API keys found'}"
            )

        logger.info(f"debate_agents_loaded count={len(agents)}")
        return agents


# Module-level singleton
_decomposer: Optional[TaskDecomposer] = None


def get_task_decomposer() -> TaskDecomposer:
    """Get or create the singleton TaskDecomposer instance."""
    global _decomposer
    if _decomposer is None:
        _decomposer = TaskDecomposer()
    return _decomposer


def analyze_task(task: str) -> TaskDecomposition:
    """Convenience function to analyze a task."""
    return get_task_decomposer().analyze(task)
