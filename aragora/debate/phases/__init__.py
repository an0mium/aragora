"""
Debate phase modules for orchestrator decomposition.

This package contains extracted phase-specific logic from the Arena class
to reduce the orchestrator's complexity and improve maintainability.

Phases:
- context_init: Context initialization (Phase 0)
- proposal_phase: Initial proposal generation (Phase 1)
- debate_rounds: Critique/Revision loop (Phase 2)
- consensus_phase: Voting and consensus resolution (Phase 3)
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
from aragora.debate.phases.analytics_phase import AnalyticsPhase
from aragora.debate.phases.belief_analysis import (
    BeliefAnalysisResult,
    DebateBeliefAnalyzer,
)
from aragora.debate.phases.consensus_phase import ConsensusPhase
from aragora.debate.phases.consensus_verification import ConsensusVerifier
from aragora.debate.phases.context_init import ContextInitializer
from aragora.debate.phases.critique import CritiquePhase
from aragora.debate.phases.debate_rounds import DebateRoundsPhase
from aragora.debate.phases.feedback_phase import FeedbackPhase
from aragora.debate.phases.judgment import JudgmentPhase
from aragora.debate.phases.metrics import MetricsHelper, build_relationship_updates
from aragora.debate.phases.proposal_phase import ProposalPhase
from aragora.debate.phases.roles_manager import RolesManager
from aragora.debate.phases.spectator import SpectatorMixin
from aragora.debate.phases.vote_aggregator import AggregatedVotes, VoteAggregator
from aragora.debate.phases.vote_processor import VoteProcessor
from aragora.debate.phases.vote_weighter import (
    VoteWeighter,
    VoteWeighterConfig,
    VoteWeighterDeps,
)
from aragora.debate.phases.voting import (
    VoteWeightCalculator,
    VotingPhase,
    WeightedVoteResult,
)
from aragora.debate.phases.weight_calculator import WeightCalculator

__all__ = [
    "DebateContext",
    "ContextInitializer",
    "ProposalPhase",
    "DebateRoundsPhase",
    "ConsensusPhase",
    "AnalyticsPhase",
    "FeedbackPhase",
    "VotingPhase",
    "VoteWeightCalculator",
    "WeightedVoteResult",
    "DebateBeliefAnalyzer",
    "BeliefAnalysisResult",
    "CritiquePhase",
    "JudgmentPhase",
    "RolesManager",
    "SpectatorMixin",
    "MetricsHelper",
    "build_relationship_updates",
    "ConsensusVerifier",
    "VoteAggregator",
    "AggregatedVotes",
    "WeightCalculator",
    "VoteProcessor",
    "VoteWeighter",
    "VoteWeighterConfig",
    "VoteWeighterDeps",
]
