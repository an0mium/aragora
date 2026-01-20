"""
Aragora Explainability Module.

Provides decision explanation capabilities for understanding
why and how debates reached their conclusions.

Components:
- Decision: Entity aggregating all explainability data
- ExplanationBuilder: Constructs Decision entities from results

Usage:
    from aragora.explainability import Decision, ExplanationBuilder

    builder = ExplanationBuilder()
    decision = await builder.build(debate_result)
    summary = builder.generate_summary(decision)
"""

from aragora.explainability.builder import ExplanationBuilder
from aragora.explainability.decision import (
    BeliefChange,
    ConfidenceAttribution,
    Counterfactual,
    Decision,
    EvidenceLink,
    InfluenceType,
    VotePivot,
)

__all__ = [
    # Main classes
    "Decision",
    "ExplanationBuilder",
    # Data classes
    "EvidenceLink",
    "VotePivot",
    "BeliefChange",
    "ConfidenceAttribution",
    "Counterfactual",
    # Enums
    "InfluenceType",
]
