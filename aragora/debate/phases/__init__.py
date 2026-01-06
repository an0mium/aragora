"""
Debate phase modules for orchestrator decomposition.

This package contains extracted phase-specific logic from the Arena class
to reduce the orchestrator's complexity and improve maintainability.

Phases:
- context_init: Context initialization (Phase 0)
- analytics_phase: Post-consensus analytics (Phases 4-6)
- feedback_phase: Post-debate feedback loops (Phase 7)
- voting: Vote collection and aggregation
- critique: Critique selection and routing
- judgment: Judge selection and final decisions
- roles_manager: Role and stance assignment
- spectator: Event emission and spectator notifications

Context:
- DebateContext: Shared state container for debate execution
"""

from aragora.debate.context import DebateContext
from aragora.debate.phases.context_init import ContextInitializer
from aragora.debate.phases.analytics_phase import AnalyticsPhase
from aragora.debate.phases.feedback_phase import FeedbackPhase
from aragora.debate.phases.voting import VotingPhase
from aragora.debate.phases.critique import CritiquePhase
from aragora.debate.phases.judgment import JudgmentPhase
from aragora.debate.phases.roles_manager import RolesManager
from aragora.debate.phases.spectator import SpectatorMixin

__all__ = [
    "DebateContext",
    "ContextInitializer",
    "AnalyticsPhase",
    "FeedbackPhase",
    "VotingPhase",
    "CritiquePhase",
    "JudgmentPhase",
    "RolesManager",
    "SpectatorMixin",
]
