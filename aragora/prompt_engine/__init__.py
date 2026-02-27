"""Prompt Engine: Transforms vague prompts into validated specifications.

Coordinates decomposition, interrogation, research, and specification building
into a single coherent flow with configurable autonomy levels.

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
from aragora.prompt_engine.spec_validator import (
    SpecValidator,
    ValidationResult,
    ValidatorRole,
)
from aragora.prompt_engine.types import (
    PROFILE_DEFAULTS,
    Ambiguity,
    AmbiguityLevel,
    Assumption,
    AutonomyLevel,
    ClarifyingQuestion,
    EvidenceLink,
    IntentType,
    InterrogationDepth,
    PromptIntent,
    QuestionOption,
    ResearchReport,
    RiskItem,
    ScopeEstimate,
    SpecFile,
    SpecProvenance,
    SpecRisk,
    Specification,
    SpecificationStatus,
    SuccessCriterion,
    UserProfile,
)

__all__ = [
    "Ambiguity",
    "AmbiguityLevel",
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
    "RiskItem",
    "ScopeEstimate",
    "SpecBuilder",
    "SpecValidator",
    "SpecFile",
    "SpecProvenance",
    "SpecRisk",
    "Specification",
    "SpecificationStatus",
    "SuccessCriterion",
    "UserProfile",
    "ValidationResult",
    "ValidatorRole",
]
