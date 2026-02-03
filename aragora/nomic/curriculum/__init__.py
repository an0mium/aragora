"""
SOAR-Inspired Curriculum Generation for Nomic Loop.

Self-Organizing Adversarial Reasoning (SOAR) generates intermediate
"stepping stone" problems that bridge the gap between current capabilities
and target challenges.

This module provides:
- SteppingStoneGenerator: Creates intermediate challenge problems
- CurriculumPlanner: Plans learning progression
- DifficultyEstimator: Estimates problem difficulty
- CurriculumAwareFeedbackLoop: Integration with autonomous orchestrator
"""

from aragora.nomic.curriculum.soar_curriculum import (
    Curriculum,
    CurriculumPlanner,
    DifficultyEstimator,
    SkillCategory,
    SkillProfile,
    SteppingStone,
    SteppingStoneGenerator,
    SteppingStoneResult,
    generate_curriculum,
)
from aragora.nomic.curriculum.integration import (
    CurriculumAwareFeedbackLoop,
    CurriculumConfig,
    integrate_curriculum_with_orchestrator,
)

__all__ = [
    # Core types
    "Curriculum",
    "CurriculumPlanner",
    "DifficultyEstimator",
    "SkillCategory",
    "SkillProfile",
    "SteppingStone",
    "SteppingStoneGenerator",
    "SteppingStoneResult",
    "generate_curriculum",
    # Integration
    "CurriculumAwareFeedbackLoop",
    "CurriculumConfig",
    "integrate_curriculum_with_orchestrator",
]
