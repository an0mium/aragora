"""Interrogation Engine -- vague prompts to structured specs."""

from aragora.interrogation.crystallizer import Crystallizer, Spec, Requirement, RequirementLevel
from aragora.interrogation.decomposer import InterrogationDecomposer, Dimension, DecompositionResult
from aragora.interrogation.engine import (
    InterrogationEngine,
    InterrogationResult,
    InterrogationState,
)
from aragora.interrogation.questioner import InterrogationQuestioner, Question, QuestionSet
from aragora.interrogation.researcher import (
    InterrogationResearcher,
    ResearchResult,
    Finding,
    ResearchSource,
)

__all__ = [
    "Crystallizer",
    "Spec",
    "Requirement",
    "RequirementLevel",
    "InterrogationDecomposer",
    "Dimension",
    "DecompositionResult",
    "InterrogationEngine",
    "InterrogationResult",
    "InterrogationState",
    "InterrogationQuestioner",
    "Question",
    "QuestionSet",
    "InterrogationResearcher",
    "ResearchResult",
    "Finding",
    "ResearchSource",
]
