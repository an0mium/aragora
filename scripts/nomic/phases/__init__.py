"""
Phase implementations for nomic loop.

Each phase handles a specific step of the self-improvement cycle:
- context: Phase 0 - Gather codebase understanding
- debate: Phase 1 - Agents propose improvements
- design: Phase 2 - Architecture planning
- implement: Phase 3 - Code generation
- verify: Phase 4 - Tests and quality checks
- commit: Phase 5 - Git commit if verified

Note: Phase implementations are currently in the main NomicLoop class.
This package provides type definitions and helpers for future extraction.
"""

from typing import TypedDict, Optional, List


class PhaseResult(TypedDict, total=False):
    """Result from a phase execution."""
    success: bool
    error: Optional[str]
    data: dict
    duration_seconds: float


class ContextResult(PhaseResult):
    """Result from context gathering phase."""
    codebase_summary: str
    recent_changes: str
    open_issues: List[str]


class DebateResult(PhaseResult):
    """Result from debate phase."""
    improvement: str
    consensus_reached: bool
    confidence: float
    votes: List[tuple]


class DesignResult(PhaseResult):
    """Result from design phase."""
    design: str
    files_affected: List[str]
    complexity_estimate: str


class ImplementResult(PhaseResult):
    """Result from implement phase."""
    files_modified: List[str]
    diff_summary: str


class VerifyResult(PhaseResult):
    """Result from verify phase."""
    tests_passed: bool
    test_output: str
    syntax_valid: bool


class CommitResult(PhaseResult):
    """Result from commit phase."""
    commit_hash: Optional[str]
    committed: bool


# Phase implementations
from .verify import VerifyPhase
from .commit import CommitPhase
from .context import ContextPhase
from .implement import ImplementPhase
from .debate import DebatePhase, DebateConfig, LearningContext, PostDebateHooks
from .design import DesignPhase, DesignConfig, BeliefContext

__all__ = [
    # Result types
    "PhaseResult",
    "ContextResult",
    "DebateResult",
    "DesignResult",
    "ImplementResult",
    "VerifyResult",
    "CommitResult",
    # Phase implementations
    "ContextPhase",
    "DebatePhase",
    "DesignPhase",
    "ImplementPhase",
    "VerifyPhase",
    "CommitPhase",
    # Config/helper classes
    "DebateConfig",
    "DesignConfig",
    "LearningContext",
    "BeliefContext",
    "PostDebateHooks",
]
