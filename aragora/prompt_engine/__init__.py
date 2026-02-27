"""Prompt-to-Spec Engine.

Transforms vague user prompts into structured specifications through:
1. Decomposition - classify intent, detect ambiguities
2. Interrogation - generate clarifying questions
3. Research - investigate current state and context
4. Specification - produce formal implementation spec

Usage:
    from aragora.prompt_engine import PromptConductor

    conductor = PromptConductor()
    result = await conductor.run("I want to improve performance")
    print(result.specification.title)
"""

from aragora.prompt_engine.conductor import (
    ConductorConfig,
    ConductorResult,
    PromptConductor,
)
from aragora.prompt_engine.decomposer import PromptDecomposer
from aragora.prompt_engine.interrogator import PromptInterrogator
from aragora.prompt_engine.researcher import PromptResearcher
from aragora.prompt_engine.spec_builder import SpecBuilder
from aragora.prompt_engine.types import (
    PROFILE_DEFAULTS,
    Ambiguity,
    Assumption,
    AutonomyLevel,
    ClarifyingQuestion,
    EvidenceLink,
    IntentType,
    InterrogationDepth,
    PromptIntent,
    QuestionOption,
    ResearchReport,
    ScopeEstimate,
    SpecFile,
    SpecProvenance,
    SpecRisk,
    Specification,
    SuccessCriterion,
    UserProfile,
)

__all__ = [
    "Ambiguity",
    "Assumption",
    "AutonomyLevel",
    "ClarifyingQuestion",
    "ConductorConfig",
    "ConductorResult",
    "EvidenceLink",
    "IntentType",
    "InterrogationDepth",
    "PROFILE_DEFAULTS",
    "PromptConductor",
    "PromptDecomposer",
    "PromptIntent",
    "PromptInterrogator",
    "PromptResearcher",
    "QuestionOption",
    "ResearchReport",
    "ScopeEstimate",
    "SpecBuilder",
    "SpecFile",
    "SpecProvenance",
    "SpecRisk",
    "Specification",
    "SuccessCriterion",
    "UserProfile",
]
