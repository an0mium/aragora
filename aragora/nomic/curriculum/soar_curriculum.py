"""
SOAR-Inspired Curriculum Generation.

Self-Organizing Adversarial Reasoning generates intermediate problems
("stepping stones") that help bridge capability gaps. Instead of jumping
directly from easy to hard problems, SOAR creates a curriculum of
intermediate challenges.

Key concepts:
1. Stepping Stones: Problems of intermediate difficulty
2. Skill Decomposition: Breaking complex skills into components
3. Adversarial Generation: Creating challenges that expose weaknesses
4. Self-Organization: Curriculum adapts based on performance

Integration with Nomic Loop:
- Each cycle can generate stepping stones for failed improvements
- Curriculum guides which improvements to attempt next
- Performance on stepping stones informs difficulty estimation

Usage:
    generator = SteppingStoneGenerator()
    planner = CurriculumPlanner(generator=generator)

    # When a Nomic cycle fails on a complex improvement:
    curriculum = await planner.create_curriculum(
        target_skill="Implement distributed caching",
        current_level=0.3,  # 30% success rate on similar tasks
        target_level=0.8,
    )

    for stone in curriculum.stepping_stones:
        # Attempt intermediate challenges before retrying main goal
        result = await attempt_improvement(stone.task)
        planner.record_result(stone, result)
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class SkillCategory(str, Enum):
    """Categories of skills that can be improved."""

    CODE_GENERATION = "code_generation"
    CODE_REFACTORING = "code_refactoring"
    BUG_FIXING = "bug_fixing"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"
    SECURITY = "security"
    INTEGRATION = "integration"
    REASONING = "reasoning"


@dataclass
class SkillProfile:
    """Profile of current skill levels across categories."""

    levels: dict[SkillCategory, float] = field(default_factory=dict)
    history: list[tuple[SkillCategory, float, datetime]] = field(default_factory=list)

    def get_level(self, category: SkillCategory) -> float:
        """Get current level for a category (0.0 to 1.0)."""
        return self.levels.get(category, 0.5)

    def update_level(self, category: SkillCategory, new_level: float) -> None:
        """Update level for a category with history tracking."""
        old_level = self.levels.get(category, 0.5)
        self.levels[category] = max(0.0, min(1.0, new_level))
        self.history.append((category, old_level, datetime.now()))

    def weakness_categories(self, threshold: float = 0.5) -> list[SkillCategory]:
        """Get categories below threshold."""
        return [cat for cat, level in self.levels.items() if level < threshold]


@dataclass
class SteppingStone:
    """An intermediate challenge problem.

    Attributes:
        id: Unique identifier
        task: Description of the intermediate task
        difficulty: Estimated difficulty (0.0 to 1.0)
        skills_required: List of skill categories needed
        prerequisites: IDs of stepping stones that should be completed first
        hints: Optional hints for completing the task
        validation_criteria: How to verify successful completion
        metadata: Additional context
    """

    id: str
    task: str
    difficulty: float
    skills_required: list[SkillCategory]
    prerequisites: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    validation_criteria: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id(task: str) -> str:
        """Generate deterministic ID from task description."""
        hash_input = task.encode("utf-8")
        return f"ss_{hashlib.sha256(hash_input).hexdigest()[:12]}"


@dataclass
class SteppingStoneResult:
    """Result of attempting a stepping stone."""

    stone_id: str
    success: bool
    completion_score: float  # 0.0 to 1.0
    time_taken: float  # seconds
    errors_encountered: list[str] = field(default_factory=list)
    skills_demonstrated: list[SkillCategory] = field(default_factory=list)


@dataclass
class Curriculum:
    """A planned sequence of stepping stones.

    Attributes:
        id: Unique curriculum identifier
        target_task: The ultimate goal
        target_difficulty: Difficulty of the target task
        stepping_stones: Ordered list of intermediate challenges
        current_index: Current position in curriculum
        created_at: When curriculum was generated
        results: Results from completed stepping stones
    """

    id: str
    target_task: str
    target_difficulty: float
    stepping_stones: list[SteppingStone]
    current_index: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    results: list[SteppingStoneResult] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if all stepping stones have been attempted."""
        return self.current_index >= len(self.stepping_stones)

    @property
    def success_rate(self) -> float:
        """Calculate success rate on completed stepping stones."""
        if not self.results:
            return 0.0
        successes = sum(1 for r in self.results if r.success)
        return successes / len(self.results)

    def next_stone(self) -> SteppingStone | None:
        """Get the next stepping stone to attempt."""
        if self.is_complete:
            return None
        return self.stepping_stones[self.current_index]


class TaskDecomposer(Protocol):
    """Protocol for decomposing complex tasks into sub-tasks."""

    async def decompose(
        self,
        task: str,
        num_subtasks: int = 3,
    ) -> list[str]:
        """Decompose a task into simpler sub-tasks."""
        ...


class DifficultyEstimator:
    """Estimates difficulty of tasks based on various factors.

    Factors considered:
    - Length and complexity of task description
    - Number of skills required
    - Presence of specific difficulty indicators
    - Historical performance on similar tasks
    """

    def __init__(self):
        """Initialize the difficulty estimator."""
        self._history: dict[str, float] = {}  # task_hash -> difficulty
        self._skill_weights: dict[SkillCategory, float] = {
            SkillCategory.CODE_GENERATION: 0.4,
            SkillCategory.CODE_REFACTORING: 0.5,
            SkillCategory.BUG_FIXING: 0.6,
            SkillCategory.ARCHITECTURE: 0.7,
            SkillCategory.TESTING: 0.5,
            SkillCategory.DOCUMENTATION: 0.3,
            SkillCategory.PERFORMANCE: 0.7,
            SkillCategory.SECURITY: 0.8,
            SkillCategory.INTEGRATION: 0.6,
            SkillCategory.REASONING: 0.6,
        }

    def estimate(
        self,
        task: str,
        skills_required: list[SkillCategory] | None = None,
    ) -> float:
        """Estimate task difficulty.

        Args:
            task: Task description
            skills_required: Optional list of required skills

        Returns:
            Difficulty score from 0.0 (trivial) to 1.0 (extremely hard)
        """
        # Base difficulty from task length (proxy for complexity)
        words = task.split()
        length_factor = min(1.0, len(words) / 100)

        # Skill-based difficulty
        skill_factor = 0.5
        if skills_required:
            skill_difficulties = [self._skill_weights.get(s, 0.5) for s in skills_required]
            skill_factor = max(skill_difficulties) if skill_difficulties else 0.5

        # Keyword-based difficulty indicators
        hard_indicators = [
            "complex",
            "distributed",
            "concurrent",
            "optimize",
            "security",
            "scale",
            "refactor",
            "migrate",
            "architecture",
        ]
        easy_indicators = [
            "simple",
            "basic",
            "add",
            "update",
            "fix typo",
            "comment",
            "rename",
            "format",
        ]

        task_lower = task.lower()
        hard_count = sum(1 for w in hard_indicators if w in task_lower)
        easy_count = sum(1 for w in easy_indicators if w in task_lower)

        keyword_factor = 0.5 + (hard_count * 0.1) - (easy_count * 0.1)
        keyword_factor = max(0.0, min(1.0, keyword_factor))

        # Combine factors
        difficulty = 0.3 * length_factor + 0.4 * skill_factor + 0.3 * keyword_factor
        return max(0.0, min(1.0, difficulty))

    def update_from_result(
        self,
        task: str,
        actual_difficulty: float,
    ) -> None:
        """Update estimates based on actual performance.

        Args:
            task: The task that was attempted
            actual_difficulty: Observed difficulty (inverse of success rate)
        """
        task_hash = hashlib.sha256(task.encode()).hexdigest()[:16]
        self._history[task_hash] = actual_difficulty


class SteppingStoneGenerator:
    """Generates intermediate stepping stone problems.

    Uses task decomposition and difficulty interpolation to create
    a sequence of problems bridging current ability to target challenge.
    """

    def __init__(
        self,
        difficulty_estimator: DifficultyEstimator | None = None,
        decomposer: TaskDecomposer | None = None,
    ):
        """Initialize the generator.

        Args:
            difficulty_estimator: Estimator for task difficulty
            decomposer: Optional custom task decomposer
        """
        self.difficulty_estimator = difficulty_estimator or DifficultyEstimator()
        self.decomposer = decomposer

    async def generate_stones(
        self,
        target_task: str,
        current_level: float,
        target_level: float,
        num_stones: int = 3,
        skill_profile: SkillProfile | None = None,
    ) -> list[SteppingStone]:
        """Generate stepping stones from current to target level.

        Creates intermediate challenges with gradually increasing difficulty.

        Args:
            target_task: The ultimate goal task
            current_level: Current ability level (0.0 to 1.0)
            target_level: Target ability level
            num_stones: Number of intermediate stones to generate
            skill_profile: Optional profile of current skills

        Returns:
            List of stepping stones in increasing difficulty order
        """
        estimated_difficulty = self.difficulty_estimator.estimate(target_task)
        # Use the higher of estimated difficulty or target_level as the ceiling
        target_difficulty = max(estimated_difficulty, target_level)

        # Calculate difficulty gap from current level to target
        difficulty_gap = target_difficulty - current_level
        if difficulty_gap <= 0:
            # Already capable, return empty curriculum
            return []

        step_size = difficulty_gap / (num_stones + 1)
        target_difficulties = [current_level + step_size * (i + 1) for i in range(num_stones)]

        # Identify skills needed for target task
        skills_required = self._infer_skills(target_task)

        # Generate stepping stones
        stones = []
        for i, diff in enumerate(target_difficulties):
            stone = await self._generate_stone(
                target_task=target_task,
                target_difficulty=diff,
                stone_number=i + 1,
                total_stones=num_stones,
                skills_required=skills_required,
                skill_profile=skill_profile,
            )
            stones.append(stone)

        # Set prerequisites (each stone depends on previous)
        for i in range(1, len(stones)):
            stones[i].prerequisites.append(stones[i - 1].id)

        return stones

    async def _generate_stone(
        self,
        target_task: str,
        target_difficulty: float,
        stone_number: int,
        total_stones: int,
        skills_required: list[SkillCategory],
        skill_profile: SkillProfile | None,
    ) -> SteppingStone:
        """Generate a single stepping stone.

        Args:
            target_task: Ultimate goal
            target_difficulty: Target difficulty for this stone
            stone_number: Position in sequence (1-indexed)
            total_stones: Total number of stones
            skills_required: Skills needed for target
            skill_profile: Current skill levels

        Returns:
            Generated SteppingStone
        """
        # Simplify task based on difficulty level
        simplification_level = 1.0 - target_difficulty

        # Create task variation
        task = self._simplify_task(
            target_task,
            simplification_level,
            stone_number,
            total_stones,
        )

        # Include all skills detected for the target task
        # All stepping stones need to build toward the same skill set
        stone_skills = skills_required.copy()

        # Generate hints based on skill profile weaknesses
        hints = []
        if skill_profile:
            weak_skills = skill_profile.weakness_categories()
            relevant_weak = [s for s in weak_skills if s in stone_skills]
            for skill in relevant_weak[:2]:
                hints.append(f"Focus on {skill.value} aspects")

        # Create validation criteria
        validation = [
            "Code compiles without errors",
            "Basic functionality works as expected",
        ]
        if target_difficulty > 0.5:
            validation.append("Handles edge cases appropriately")
        if target_difficulty > 0.7:
            validation.append("Includes appropriate tests")

        stone_id = SteppingStone.generate_id(task)

        return SteppingStone(
            id=stone_id,
            task=task,
            difficulty=target_difficulty,
            skills_required=stone_skills,
            hints=hints,
            validation_criteria=validation,
            metadata={
                "generated_from": target_task[:100],
                "stone_number": stone_number,
                "total_stones": total_stones,
            },
        )

    def _simplify_task(
        self,
        task: str,
        simplification_level: float,
        stone_number: int,
        total_stones: int,
    ) -> str:
        """Create a simplified version of the task.

        Args:
            task: Original task
            simplification_level: How much to simplify (0=same, 1=trivial)
            stone_number: Position in sequence
            total_stones: Total number of stones

        Returns:
            Simplified task description
        """
        # Determine simplification strategy
        if simplification_level > 0.7:
            # Very simple: just understand/plan
            prefix = f"[Stone {stone_number}/{total_stones}] Analyze and plan: "
            return prefix + self._truncate_task(task, 100)
        elif simplification_level > 0.4:
            # Moderate: implement core feature only
            prefix = f"[Stone {stone_number}/{total_stones}] Implement basic version: "
            return prefix + self._truncate_task(task, 150)
        else:
            # Close to target: nearly full implementation
            prefix = f"[Stone {stone_number}/{total_stones}] Implement with constraints: "
            return prefix + task

    def _truncate_task(self, task: str, max_len: int) -> str:
        """Truncate task description to max length."""
        if len(task) <= max_len:
            return task
        return task[: max_len - 3] + "..."

    def _infer_skills(self, task: str) -> list[SkillCategory]:
        """Infer required skills from task description.

        Args:
            task: Task description

        Returns:
            List of likely required skills
        """
        task_lower = task.lower()
        skills = []

        skill_keywords = {
            SkillCategory.CODE_GENERATION: ["create", "implement", "write", "build"],
            SkillCategory.CODE_REFACTORING: ["refactor", "restructure", "clean", "simplify"],
            SkillCategory.BUG_FIXING: ["fix", "bug", "error", "issue", "debug"],
            SkillCategory.ARCHITECTURE: ["architect", "design", "structure", "pattern"],
            SkillCategory.TESTING: ["test", "spec", "verify", "validate"],
            SkillCategory.DOCUMENTATION: ["document", "readme", "comment", "explain"],
            SkillCategory.PERFORMANCE: ["optimize", "performance", "speed", "efficient"],
            SkillCategory.SECURITY: ["security", "auth", "encrypt", "safe"],
            SkillCategory.INTEGRATION: ["integrate", "connect", "api", "external"],
            SkillCategory.REASONING: ["analyze", "evaluate", "decide", "compare"],
        }

        for skill, keywords in skill_keywords.items():
            if any(kw in task_lower for kw in keywords):
                skills.append(skill)

        # Default to code generation if no skills detected
        if not skills:
            skills.append(SkillCategory.CODE_GENERATION)

        return skills


class CurriculumPlanner:
    """Plans and manages learning curricula.

    Coordinates stepping stone generation, tracks progress,
    and adapts curriculum based on performance.
    """

    def __init__(
        self,
        generator: SteppingStoneGenerator | None = None,
        storage_path: Path | None = None,
    ):
        """Initialize the planner.

        Args:
            generator: Stepping stone generator
            storage_path: Optional path for persisting curricula
        """
        self.generator = generator or SteppingStoneGenerator()
        self.storage_path = storage_path
        self.active_curricula: dict[str, Curriculum] = {}
        self.skill_profile = SkillProfile()

    async def create_curriculum(
        self,
        target_task: str,
        current_level: float = 0.5,
        target_level: float = 0.8,
        num_stones: int = 3,
    ) -> Curriculum:
        """Create a new learning curriculum.

        Args:
            target_task: The ultimate goal
            current_level: Current ability estimate
            target_level: Desired ability level
            num_stones: Number of intermediate challenges

        Returns:
            New Curriculum with stepping stones
        """
        stones = await self.generator.generate_stones(
            target_task=target_task,
            current_level=current_level,
            target_level=target_level,
            num_stones=num_stones,
            skill_profile=self.skill_profile,
        )

        target_difficulty = self.generator.difficulty_estimator.estimate(target_task)

        curriculum = Curriculum(
            id=str(uuid.uuid4()),
            target_task=target_task,
            target_difficulty=target_difficulty,
            stepping_stones=stones,
        )

        self.active_curricula[curriculum.id] = curriculum

        if self.storage_path:
            self._save_curriculum(curriculum)

        return curriculum

    def record_result(
        self,
        curriculum_id: str,
        result: SteppingStoneResult,
    ) -> None:
        """Record result of attempting a stepping stone.

        Args:
            curriculum_id: ID of the curriculum
            result: Result of the attempt
        """
        curriculum = self.active_curricula.get(curriculum_id)
        if not curriculum:
            logger.warning(f"Curriculum not found: {curriculum_id}")
            return

        curriculum.results.append(result)
        curriculum.current_index += 1

        # Update skill profile based on result
        for skill in result.skills_demonstrated:
            current = self.skill_profile.get_level(skill)
            if result.success:
                # Increase skill level on success
                new_level = current + 0.1 * result.completion_score
            else:
                # Slight decrease on failure
                new_level = current - 0.05
            self.skill_profile.update_level(skill, new_level)

        # Update difficulty estimates
        stone = next(
            (s for s in curriculum.stepping_stones if s.id == result.stone_id),
            None,
        )
        if stone:
            actual_difficulty = 1.0 - result.completion_score
            self.generator.difficulty_estimator.update_from_result(
                stone.task,
                actual_difficulty,
            )

        if self.storage_path:
            self._save_curriculum(curriculum)

    def should_attempt_target(self, curriculum_id: str) -> bool:
        """Check if target task should be attempted.

        Based on curriculum completion and success rate.

        Args:
            curriculum_id: ID of the curriculum

        Returns:
            True if ready to attempt target task
        """
        curriculum = self.active_curricula.get(curriculum_id)
        if not curriculum:
            return False

        # Must complete curriculum first
        if not curriculum.is_complete:
            return False

        # Check success rate threshold
        return curriculum.success_rate >= 0.6

    def get_next_action(self, curriculum_id: str) -> SteppingStone | str | None:
        """Get the next action for a curriculum.

        Returns either:
        - A SteppingStone to attempt
        - The target task string (if ready)
        - None (if curriculum failed)

        Args:
            curriculum_id: ID of the curriculum

        Returns:
            Next SteppingStone, target task, or None
        """
        curriculum = self.active_curricula.get(curriculum_id)
        if not curriculum:
            return None

        if not curriculum.is_complete:
            return curriculum.next_stone()

        if self.should_attempt_target(curriculum_id):
            return curriculum.target_task

        # Curriculum complete but insufficient success rate
        return None

    def _save_curriculum(self, curriculum: Curriculum) -> None:
        """Save curriculum to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        filepath = self.storage_path / f"{curriculum.id}.json"

        data = {
            "id": curriculum.id,
            "target_task": curriculum.target_task,
            "target_difficulty": curriculum.target_difficulty,
            "current_index": curriculum.current_index,
            "created_at": curriculum.created_at.isoformat(),
            "stepping_stones": [
                {
                    "id": s.id,
                    "task": s.task,
                    "difficulty": s.difficulty,
                    "skills_required": [sk.value for sk in s.skills_required],
                    "prerequisites": s.prerequisites,
                    "hints": s.hints,
                    "validation_criteria": s.validation_criteria,
                    "metadata": s.metadata,
                }
                for s in curriculum.stepping_stones
            ],
            "results": [
                {
                    "stone_id": r.stone_id,
                    "success": r.success,
                    "completion_score": r.completion_score,
                    "time_taken": r.time_taken,
                    "errors_encountered": r.errors_encountered,
                    "skills_demonstrated": [s.value for s in r.skills_demonstrated],
                }
                for r in curriculum.results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


async def generate_curriculum(
    target_task: str,
    current_level: float = 0.5,
    num_stones: int = 3,
) -> Curriculum:
    """Convenience function to generate a curriculum.

    Args:
        target_task: The ultimate goal
        current_level: Current ability estimate
        num_stones: Number of intermediate challenges

    Returns:
        Generated Curriculum
    """
    planner = CurriculumPlanner()
    return await planner.create_curriculum(
        target_task=target_task,
        current_level=current_level,
        num_stones=num_stones,
    )
