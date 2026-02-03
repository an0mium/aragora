"""
Molecules: Durable Chained Workflows.

Inspired by Gastown's Molecules pattern, this module provides workflows
that survive agent restarts with per-step persistence. Each step is
individually checkpointed, ensuring no work is lost.

Key concepts:
- Molecule: A durable workflow containing multiple steps
- MoleculeStep: A single step with checkpoint support
- MoleculeEngine: Executes molecules with recovery support
- StepExecutor: Pluggable step execution strategy

Usage:
    engine = MoleculeEngine(bead_store)

    # Define a molecule
    molecule = Molecule.create(
        name="deploy_feature",
        steps=[
            MoleculeStep.create("run_tests", "test", {"command": "pytest"}),
            MoleculeStep.create("build", "build", {"command": "make build"}),
            MoleculeStep.create("deploy", "deploy", {"env": "staging"}),
        ]
    )

    # Execute with automatic checkpointing
    result = await engine.execute(molecule)

    # Resume after crash
    result = await engine.resume("molecule-123")
"""

# Base types and dataclasses
from aragora.nomic.molecules.base import (
    # Type aliases
    ConsensusType,
    # Exceptions
    TransactionError,
    TransactionRollbackError,
    CyclicDependencyError,
    DeadlockError,
    DependencyValidationError,
    # Enums
    TransactionState,
    StepStatus,
    MoleculeStatus,
    # Dataclasses
    CompensatingAction,
    MoleculeTransaction,
    MoleculeStep,
    Molecule,
    MoleculeResult,
)

# Proposal phase executors
from aragora.nomic.molecules.proposal import (
    StepExecutor,
    AgentStepExecutor,
)

# Critique phase executors
from aragora.nomic.molecules.critique import (
    DebateStepExecutor,
)

# Execution phase executors and engine
from aragora.nomic.molecules.execution import (
    ShellStepExecutor,
    ParallelStepExecutor,
    ConditionalStepExecutor,
    MoleculeEngine,
)

# Verification components
from aragora.nomic.molecules.verification import (
    DependencyGraph,
    ResourceLock,
    DeadlockDetector,
    get_deadlock_detector,
    reset_deadlock_detector,
)

# Registry and factory functions
from aragora.nomic.molecules.registry import (
    EscalationLevel,
    EscalationContext,
    EscalationStepExecutor,
    create_escalation_molecule,
    create_conditional_escalation_molecule,
    get_molecule_engine,
    reset_molecule_engine,
)


__all__ = [
    # Type aliases
    "ConsensusType",
    # Exceptions
    "TransactionError",
    "TransactionRollbackError",
    "CyclicDependencyError",
    "DeadlockError",
    "DependencyValidationError",
    # Enums
    "TransactionState",
    "StepStatus",
    "MoleculeStatus",
    "EscalationLevel",
    # Dataclasses
    "CompensatingAction",
    "MoleculeTransaction",
    "MoleculeStep",
    "Molecule",
    "MoleculeResult",
    "EscalationContext",
    # Step executors
    "StepExecutor",
    "AgentStepExecutor",
    "DebateStepExecutor",
    "ShellStepExecutor",
    "ParallelStepExecutor",
    "ConditionalStepExecutor",
    "EscalationStepExecutor",
    # Engine
    "MoleculeEngine",
    # Verification components
    "DependencyGraph",
    "ResourceLock",
    "DeadlockDetector",
    "get_deadlock_detector",
    "reset_deadlock_detector",
    # Factory functions
    "create_escalation_molecule",
    "create_conditional_escalation_molecule",
    "get_molecule_engine",
    "reset_molecule_engine",
]
