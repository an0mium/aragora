"""
Nomic Loop: Autonomous self-improvement cycle for aragora.

This package provides modular components for the nomic loop:
- recovery: Phase error recovery and retry logic
- circuit_breaker: Agent failure detection and cooldown
- safety: Protected file checksums, backups, and rollback
- git: Git operations for version control
- config: Configuration constants and environment loading

Like a PCR machine for code evolution:
1. DEBATE: All agents propose improvements to aragora
2. CONSENSUS: Agents critique and refine until consensus
3. DESIGN: Agents design the implementation
4. IMPLEMENT: Agents write the code
5. VERIFY: Run tests, check quality
6. COMMIT: If verified, commit changes
7. REPEAT: Cycle continues

The dialectic tension between models (visionary vs pragmatic vs synthesizer)
creates emergent complexity and self-criticality.
"""

from .recovery import PhaseError, PhaseRecovery
from .circuit_breaker import AgentCircuitBreaker
from .deadlock import DeadlockManager, DeadlockState, DeadlockResolution
from .formatters import ContextFormatter, FormatterDependencies
from .deep_audit import DeepAuditRunner, AuditResult
from .disagreement import DisagreementHandler, DisagreementActions
from .arena_factory import ArenaFactory, ArenaFactoryDependencies, ArenaConfig
from .post_processing import PostDebateProcessor, ProcessingDependencies, ProcessingContext
from .error_taxonomy import (
    ErrorType,
    Severity,
    ErrorCategory,
    ErrorPattern,
    classify_error,
    extract_test_failures,
    format_learning_summary,
)
from .config import (
    NOMIC_AUTO_COMMIT,
    NOMIC_AUTO_CONTINUE,
    NOMIC_MAX_CYCLE_SECONDS,
    NOMIC_STALL_THRESHOLD,
    NOMIC_FIX_DEADLINE_BUFFER,
    NOMIC_FIX_ITERATION_BUDGET,
    NOMIC_AUTO_CHECKPOINT,
    NOMIC_USE_PERFORMANCE_SELECTION,
    NOMIC_TRICKSTER_ENABLED,
    NOMIC_TRICKSTER_SENSITIVITY,
    NOMIC_CALIBRATION_ENABLED,
    NOMIC_OUTCOME_TRACKING,
    load_dotenv,
)

__all__ = [
    # Recovery
    "PhaseError",
    "PhaseRecovery",
    # Circuit Breaker
    "AgentCircuitBreaker",
    # Deadlock
    "DeadlockManager",
    "DeadlockState",
    "DeadlockResolution",
    # Formatters
    "ContextFormatter",
    "FormatterDependencies",
    # Deep Audit
    "DeepAuditRunner",
    "AuditResult",
    # Disagreement
    "DisagreementHandler",
    "DisagreementActions",
    # Arena Factory (Wave 3)
    "ArenaFactory",
    "ArenaFactoryDependencies",
    "ArenaConfig",
    # Post-Processing (Wave 3)
    "PostDebateProcessor",
    "ProcessingDependencies",
    "ProcessingContext",
    # Error Taxonomy (learning)
    "ErrorType",
    "Severity",
    "ErrorCategory",
    "ErrorPattern",
    "classify_error",
    "extract_test_failures",
    "format_learning_summary",
    # Config
    "NOMIC_AUTO_COMMIT",
    "NOMIC_AUTO_CONTINUE",
    "NOMIC_MAX_CYCLE_SECONDS",
    "NOMIC_STALL_THRESHOLD",
    "NOMIC_FIX_DEADLINE_BUFFER",
    "NOMIC_FIX_ITERATION_BUDGET",
    "NOMIC_AUTO_CHECKPOINT",
    "NOMIC_USE_PERFORMANCE_SELECTION",
    "NOMIC_TRICKSTER_ENABLED",
    "NOMIC_TRICKSTER_SENSITIVITY",
    "NOMIC_CALIBRATION_ENABLED",
    "NOMIC_OUTCOME_TRACKING",
    "load_dotenv",
]
