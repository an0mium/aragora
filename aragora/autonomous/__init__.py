"""
Autonomous Operations Module.

Provides infrastructure for self-improving, continuously learning,
and proactive intelligent behavior.

Phase 5 Components:
- 5.1 Nomic Loop Enhancement: Self-improvement automation, verification, rollback
- 5.2 Continuous Learning: Real-time ELO, calibration, pattern extraction
- 5.3 Proactive Intelligence: Scheduled triggers, alerts, monitoring, anomalies

Usage:
    from aragora.autonomous import (
        # Loop Enhancement
        SelfImprovementManager,
        CodeVerifier,
        RollbackManager,
        ApprovalFlow,

        # Continuous Learning
        ContinuousLearner,
        EloUpdater,
        PatternExtractor,
        KnowledgeDecayManager,

        # Proactive Intelligence
        ScheduledTrigger,
        AlertAnalyzer,
        TrendMonitor,
        AnomalyDetector,
    )
"""

from aragora.autonomous.loop_enhancement import (
    SelfImprovementManager,
    CodeVerifier,
    RollbackManager,
    ApprovalFlow,
    ApprovalStatus,
)

from aragora.autonomous.continuous_learning import (
    ContinuousLearner,
    EloUpdater,
    PatternExtractor,
    KnowledgeDecayManager,
    LearningEvent,
)

from aragora.autonomous.proactive_intelligence import (
    ScheduledTrigger,
    AlertAnalyzer,
    TrendMonitor,
    AnomalyDetector,
    AlertSeverity,
    TrendDirection,
)

__all__ = [
    # Loop Enhancement
    "SelfImprovementManager",
    "CodeVerifier",
    "RollbackManager",
    "ApprovalFlow",
    "ApprovalStatus",
    # Continuous Learning
    "ContinuousLearner",
    "EloUpdater",
    "PatternExtractor",
    "KnowledgeDecayManager",
    "LearningEvent",
    # Proactive Intelligence
    "ScheduledTrigger",
    "AlertAnalyzer",
    "TrendMonitor",
    "AnomalyDetector",
    "AlertSeverity",
    "TrendDirection",
]
