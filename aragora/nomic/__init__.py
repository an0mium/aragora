"""
Nomic Loop Integration Module.

Provides integration hub for connecting all aragora features
with the nomic loop self-improvement cycle.

The nomic loop is a 5-phase cycle:
1. Debate - Multi-agent debate on improvements
2. Design - Design the implementation
3. Implement - Write the code changes
4. Verify - Test and validate changes
5. Commit - Commit approved changes

This module integrates:
- Bayesian belief propagation for debate analysis
- Capability probing for agent reliability
- Evidence staleness detection
- Counterfactual branching for deadlocks
- Checkpointing for crash recovery
"""

from aragora.nomic.integration import (
    NomicIntegration,
    BeliefAnalysis,
    AgentReliability,
    StalenessReport,
    PhaseCheckpoint,
    create_nomic_integration,
)

__all__ = [
    "NomicIntegration",
    "BeliefAnalysis",
    "AgentReliability",
    "StalenessReport",
    "PhaseCheckpoint",
    "create_nomic_integration",
]
