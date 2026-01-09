"""
Debate orchestration module.
"""

from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.debate.graph import (
    DebateGraph,
    DebateNode,
    Branch,
    BranchPolicy,
    BranchReason,
    MergeStrategy,
    MergeResult,
    ConvergenceScorer,
    GraphReplayBuilder,
    GraphDebateOrchestrator,
    NodeType,
)
from aragora.debate.scenarios import (
    ScenarioMatrix,
    Scenario,
    ScenarioType,
    ScenarioResult,
    MatrixResult,
    MatrixDebateRunner,
    ScenarioComparator,
    OutcomeCategory,
    create_scale_scenarios,
    create_risk_scenarios,
    create_time_horizon_scenarios,
)
from aragora.debate.counterfactual import (
    CounterfactualOrchestrator,
    CounterfactualBranch,
    ConditionalConsensus,
    PivotClaim,
    ImpactDetector,
    CounterfactualIntegration,
    CounterfactualStatus,
    BranchComparison,
    explore_counterfactual,
)
from aragora.debate.checkpoint import (
    CheckpointManager,
    DebateCheckpoint,
    CheckpointStore,
    FileCheckpointStore,
    S3CheckpointStore,
    GitCheckpointStore,
    CheckpointConfig,
    CheckpointStatus,
    ResumedDebate,
    AgentState,
    CheckpointWebhook,
    checkpoint_debate,
)
from aragora.debate.blackbox import (
    BlackboxRecorder,
    BlackboxEvent,
    BlackboxSnapshot,
    get_blackbox,
    close_blackbox,
)
from aragora.debate.immune_system import (
    TransparentImmuneSystem,
    HealthEvent,
    HealthStatus,
    AgentStatus,
    AgentHealthState,
    get_immune_system,
    reset_immune_system,
)
from aragora.debate.wisdom_injector import (
    WisdomInjector,
    WisdomSubmission,
    WisdomInjection,
    get_wisdom_injector,
    close_wisdom_injector,
)
from aragora.debate.complexity_governor import (
    AdaptiveComplexityGovernor,
    GovernorConstraints,
    StressLevel,
    get_complexity_governor,
    reset_complexity_governor,
)
from aragora.debate.cognitive_limiter import (
    CognitiveLoadLimiter,
    CognitiveBudget,
    limit_debate_context,
    STRESS_BUDGETS,
)
from aragora.debate.recovery_narrator import (
    RecoveryNarrator,
    RecoveryNarrative,
    get_narrator,
    reset_narrator,
    setup_narrator_with_immune_system,
)
from aragora.debate.rhetorical_observer import (
    RhetoricalAnalysisObserver,
    RhetoricalObservation,
    RhetoricalPattern,
    get_rhetorical_observer,
    reset_rhetorical_observer,
)
from aragora.debate.chaos_theater import (
    ChaosDirector,
    TheaterResponse,
    FailureType,
    DramaLevel,
    get_chaos_director,
    theatrical_timeout,
    theatrical_error,
)

__all__ = [
    "Arena",
    "DebateProtocol",
    # Graph-based debates
    "DebateGraph",
    "DebateNode",
    "Branch",
    "BranchPolicy",
    "BranchReason",
    "MergeStrategy",
    "MergeResult",
    "ConvergenceScorer",
    "GraphReplayBuilder",
    "GraphDebateOrchestrator",
    "NodeType",
    # Scenario Matrix
    "ScenarioMatrix",
    "Scenario",
    "ScenarioType",
    "ScenarioResult",
    "MatrixResult",
    "MatrixDebateRunner",
    "ScenarioComparator",
    "OutcomeCategory",
    "create_scale_scenarios",
    "create_risk_scenarios",
    "create_time_horizon_scenarios",
    # Counterfactual Branching
    "CounterfactualOrchestrator",
    "CounterfactualBranch",
    "ConditionalConsensus",
    "PivotClaim",
    "ImpactDetector",
    "CounterfactualIntegration",
    "CounterfactualStatus",
    "BranchComparison",
    "explore_counterfactual",
    # Checkpointing
    "CheckpointManager",
    "DebateCheckpoint",
    "CheckpointStore",
    "FileCheckpointStore",
    "S3CheckpointStore",
    "GitCheckpointStore",
    "CheckpointConfig",
    "CheckpointStatus",
    "ResumedDebate",
    "AgentState",
    "CheckpointWebhook",
    "checkpoint_debate",
    # Blackbox Protocol
    "BlackboxRecorder",
    "BlackboxEvent",
    "BlackboxSnapshot",
    "get_blackbox",
    "close_blackbox",
    # Transparent Immune System
    "TransparentImmuneSystem",
    "HealthEvent",
    "HealthStatus",
    "AgentStatus",
    "AgentHealthState",
    "get_immune_system",
    "reset_immune_system",
    # Wisdom Injector
    "WisdomInjector",
    "WisdomSubmission",
    "WisdomInjection",
    "get_wisdom_injector",
    "close_wisdom_injector",
    # Complexity Governor
    "AdaptiveComplexityGovernor",
    "GovernorConstraints",
    "StressLevel",
    "get_complexity_governor",
    "reset_complexity_governor",
    # Cognitive Load Limiter
    "CognitiveLoadLimiter",
    "CognitiveBudget",
    "limit_debate_context",
    "STRESS_BUDGETS",
    # Recovery Narrator
    "RecoveryNarrator",
    "RecoveryNarrative",
    "get_narrator",
    "reset_narrator",
    "setup_narrator_with_immune_system",
    # Rhetorical Observer
    "RhetoricalAnalysisObserver",
    "RhetoricalObservation",
    "RhetoricalPattern",
    "get_rhetorical_observer",
    "reset_rhetorical_observer",
    # Chaos Theater
    "ChaosDirector",
    "TheaterResponse",
    "FailureType",
    "DramaLevel",
    "get_chaos_director",
    "theatrical_timeout",
    "theatrical_error",
]
