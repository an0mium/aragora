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

import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict

logger = logging.getLogger(__name__)


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


class PhaseValidationError(Exception):
    """Raised when phase result validation fails."""

    def __init__(self, phase: str, message: str, recoverable: bool = True):
        self.phase = phase
        self.recoverable = recoverable
        super().__init__(f"[{phase}] Validation failed: {message}")


class PhaseValidator:
    """
    Validates phase results to ensure safe state transitions.

    Prevents cycles from crashing due to:
    - Empty or None results
    - Missing required fields
    - Invalid data types
    - Logical inconsistencies
    """

    # Required fields per phase
    REQUIRED_FIELDS: Dict[str, List[str]] = {
        "context": ["success"],
        "debate": ["success", "consensus_reached"],
        "design": ["success"],
        "implement": ["success"],
        "verify": ["success"],
        "commit": ["success"],
    }

    # Fields that must be non-empty strings if present
    NON_EMPTY_STRING_FIELDS: Dict[str, List[str]] = {
        "context": ["codebase_summary"],
        "debate": ["improvement"],
        "design": ["design"],
        "implement": [],
        "verify": ["test_output"],
        "commit": [],
    }

    @classmethod
    def validate(cls, phase: str, result: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a phase result.

        Args:
            phase: Phase name
            result: Phase result to validate

        Returns:
            (is_valid, error_message or None)
        """
        if result is None:
            return False, "Result is None"

        if not isinstance(result, dict):
            return False, f"Result is not a dict: {type(result)}"

        # Check required fields
        required = cls.REQUIRED_FIELDS.get(phase, [])
        for field in required:
            if field not in result:
                return False, f"Missing required field: {field}"

        # Check non-empty strings for successful results
        if result.get("success", False):
            non_empty = cls.NON_EMPTY_STRING_FIELDS.get(phase, [])
            for field in non_empty:
                if field in result:
                    value = result[field]
                    if value is None or (isinstance(value, str) and not value.strip()):
                        return False, f"Field '{field}' is empty but result marked successful"

        # Phase-specific validations
        if phase == "debate":
            if result.get("consensus_reached") and not result.get("improvement"):
                return False, "Consensus reached but no improvement specified"
            # Validate confidence is in range
            confidence = result.get("confidence", 0)
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                logger.warning(f"Invalid confidence value: {confidence}, clamping to [0,1]")

        if phase == "design":
            files = result.get("files_affected", [])
            if result.get("success") and not files:
                logger.warning("Design succeeded but no files_affected specified")

        if phase == "verify":
            if result.get("tests_passed") is None and result.get("success"):
                logger.warning("Verify phase succeeded but tests_passed not set")

        return True, None

    @classmethod
    def validate_or_raise(cls, phase: str, result: Any) -> None:
        """Validate and raise PhaseValidationError if invalid."""
        is_valid, error = cls.validate(phase, result)
        if not is_valid:
            raise PhaseValidationError(phase, error)

    @classmethod
    def safe_get(cls, result: Any, field: str, default: Any = None) -> Any:
        """Safely get a field from a result, handling None/non-dict cases."""
        if result is None or not isinstance(result, dict):
            return default
        return result.get(field, default)

    @classmethod
    def normalize_result(cls, phase: str, result: Any) -> Dict[str, Any]:
        """
        Normalize a result to ensure it has expected structure.

        Args:
            phase: Phase name
            result: Raw result

        Returns:
            Normalized result dict with defaults filled in
        """
        if result is None:
            result = {}
        if not isinstance(result, dict):
            result = {"error": str(result), "success": False}

        # Ensure success field exists
        if "success" not in result:
            result["success"] = False

        # Phase-specific defaults
        if phase == "context":
            result.setdefault("codebase_summary", "")
            result.setdefault("recent_changes", "")
            result.setdefault("open_issues", [])

        if phase == "debate":
            result.setdefault("consensus_reached", False)
            result.setdefault("improvement", "")
            result.setdefault("confidence", 0.0)
            result.setdefault("votes", [])
            # Clamp confidence to valid range
            conf = result.get("confidence", 0)
            if isinstance(conf, (int, float)):
                result["confidence"] = max(0.0, min(1.0, float(conf)))

        if phase == "design":
            result.setdefault("design", "")
            result.setdefault("files_affected", [])
            result.setdefault("complexity_estimate", "unknown")

        if phase == "implement":
            result.setdefault("files_modified", [])
            result.setdefault("diff_summary", "")

        if phase == "verify":
            result.setdefault("tests_passed", False)
            result.setdefault("test_output", "")
            result.setdefault("syntax_valid", True)

        if phase == "commit":
            result.setdefault("commit_hash", None)
            result.setdefault("committed", False)

        return result


def validate_agents_list(agents: List[Any], min_agents: int = 1) -> Tuple[bool, str]:
    """
    Validate that agents list is properly formed.

    Args:
        agents: List of agent instances
        min_agents: Minimum required agents

    Returns:
        (is_valid, error_message)
    """
    if agents is None:
        return False, "Agents list is None"
    if not isinstance(agents, list):
        return False, f"Agents is not a list: {type(agents)}"
    if len(agents) < min_agents:
        return False, f"Need at least {min_agents} agent(s), got {len(agents)}"

    # Check each agent has required methods
    for i, agent in enumerate(agents):
        if agent is None:
            return False, f"Agent at index {i} is None"
        if not hasattr(agent, "generate"):
            return False, f"Agent at index {i} missing 'generate' method"

    return True, ""


# Phase implementations
from .commit import CommitPhase
from .context import ContextPhase, set_metrics_recorder
from .debate import DebateConfig, DebatePhase, LearningContext, PostDebateHooks
from .design import BeliefContext, DesignConfig, DesignPhase
from .implement import ImplementPhase
from .scope_limiter import ScopeEvaluation, ScopeLimiter, check_design_scope
from .verify import VerifyPhase

__all__ = [
    # Result types
    "PhaseResult",
    "ContextResult",
    "DebateResult",
    "DesignResult",
    "ImplementResult",
    "VerifyResult",
    "CommitResult",
    # Validation
    "PhaseValidator",
    "PhaseValidationError",
    "validate_agents_list",
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
    # Scope limiting
    "ScopeLimiter",
    "ScopeEvaluation",
    "check_design_scope",
    # Metrics integration
    "set_metrics_recorder",
]
