"""
Nomic Loop Approval Gates.

Safety gates that require explicit approval before advancing to critical phases.
Gates provide structured decision points with audit trails.

Gate Types:
- DesignGate: Requires approval before implementation begins
- TestQualityGate: Validates test quality thresholds after verification
- CommitGate: Structured approval before committing changes
"""

import hashlib
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Simple Gate Functions (for backward compatibility with tests)
# =============================================================================

# Default protected files
DEFAULT_PROTECTED_FILES = [
    "CLAUDE.md",
    "core.py",
    "aragora/__init__.py",
    ".env",
    ".env.local",
    "scripts/nomic_loop.py",
]

# Default dangerous patterns to detect
DEFAULT_DANGEROUS_PATTERNS = [
    "eval(",
    "exec(",
    "os.system(",
    "subprocess.call(",
    "__import__(",
    "compile(",
]


@dataclass
class GateConfig:
    """Configuration for simple gate functions."""

    protected_files: List[str] = field(default_factory=lambda: DEFAULT_PROTECTED_FILES.copy())
    dangerous_patterns: List[str] = field(default_factory=lambda: DEFAULT_DANGEROUS_PATTERNS.copy())
    max_files: int = 20
    max_lines: int = 1000
    max_duration_seconds: int = 600


# Module-level config (can be overridden in tests)
_gate_config: Optional[GateConfig] = None


def _get_config() -> GateConfig:
    """Get the current gate configuration."""
    global _gate_config
    if _gate_config is None:
        _gate_config = GateConfig()
    return _gate_config


def is_protected_file(file_path: str) -> bool:
    """
    Check if a file is protected from modification.

    Args:
        file_path: Path to the file (relative or absolute)

    Returns:
        True if the file is protected
    """
    config = _get_config()
    path = Path(file_path)

    # Normalize the path
    name = path.name
    path_str = str(path)

    for protected in config.protected_files:
        # Check if the protected file matches the name or is in the path
        if name == protected or protected in path_str:
            return True
        # Check for .env pattern matching
        if protected.startswith(".env") and name.startswith(".env"):
            return True

    return False


def check_change_volume(
    files_changed: List[str],
    max_files: Optional[int] = None,
    lines_added: int = 0,
    lines_removed: int = 0,
    max_lines: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Check if the change volume is within acceptable limits.

    Args:
        files_changed: List of file paths being changed
        max_files: Maximum allowed files (uses config default if None)
        lines_added: Number of lines added
        lines_removed: Number of lines removed
        max_lines: Maximum allowed lines changed (uses config default if None)

    Returns:
        Dict with 'allowed' bool and 'reason' if blocked
    """
    config = _get_config()
    max_files = max_files if max_files is not None else config.max_files
    max_lines = max_lines if max_lines is not None else config.max_lines

    # Check file count
    if len(files_changed) > max_files:
        return {
            "allowed": False,
            "reason": f"Too many files changed ({len(files_changed)} > {max_files})",
        }

    # Check line count
    total_lines = lines_added + lines_removed
    if total_lines > max_lines:
        return {
            "allowed": False,
            "reason": f"Too many lines changed ({total_lines} > {max_lines})",
        }

    return {"allowed": True, "reason": ""}


def check_dangerous_patterns(code: str) -> Dict[str, Any]:
    """
    Check for dangerous code patterns.

    Args:
        code: Source code to check

    Returns:
        Dict with 'safe' bool and 'patterns_found' list
    """
    config = _get_config()
    patterns_found = []

    for pattern in config.dangerous_patterns:
        if pattern in code:
            # Extract pattern name (e.g., "eval(" -> "eval")
            pattern_name = pattern.rstrip("(")
            patterns_found.append(pattern_name)

    return {
        "safe": len(patterns_found) == 0,
        "patterns_found": patterns_found,
    }


def check_resource_limits(
    estimated_duration_seconds: int,
    max_duration_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Check if resource usage is within limits.

    Args:
        estimated_duration_seconds: Estimated operation duration
        max_duration_seconds: Maximum allowed duration (uses config default if None)

    Returns:
        Dict with 'allowed' bool and 'reason' if blocked
    """
    config = _get_config()
    max_duration = (
        max_duration_seconds if max_duration_seconds is not None else config.max_duration_seconds
    )

    if estimated_duration_seconds > max_duration:
        return {
            "allowed": False,
            "reason": f"Operation time exceeds limit ({estimated_duration_seconds}s > {max_duration}s)",
        }

    return {"allowed": True, "reason": ""}


def check_all_gates(changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check all gates for a set of changes.

    Args:
        changes: Dict with keys:
            - files_changed: List of file paths
            - lines_added: Number of lines added
            - lines_removed: Number of lines removed
            - code_content: Code to check for dangerous patterns
            - estimated_duration: Estimated operation time in seconds

    Returns:
        Dict with 'allowed' bool and 'blocked_by' list of gate names
    """
    blocked_by = []

    # Check protected files
    for file_path in changes.get("files_changed", []):
        if is_protected_file(file_path):
            blocked_by.append(f"protected_file:{file_path}")

    # Check change volume
    volume_result = check_change_volume(
        files_changed=changes.get("files_changed", []),
        lines_added=changes.get("lines_added", 0),
        lines_removed=changes.get("lines_removed", 0),
    )
    if not volume_result["allowed"]:
        blocked_by.append(f"change_volume:{volume_result['reason']}")

    # Check dangerous patterns
    code_content = changes.get("code_content", "")
    if code_content:
        pattern_result = check_dangerous_patterns(code_content)
        if not pattern_result["safe"]:
            blocked_by.append(f"dangerous_patterns:{','.join(pattern_result['patterns_found'])}")

    # Check resource limits
    estimated_duration = changes.get("estimated_duration", 0)
    if estimated_duration:
        resource_result = check_resource_limits(estimated_duration)
        if not resource_result["allowed"]:
            blocked_by.append(f"resource_limits:{resource_result['reason']}")

    return {
        "allowed": len(blocked_by) == 0,
        "blocked_by": blocked_by,
    }


# =============================================================================
# Approval Gate Classes (structured decision points)
# =============================================================================


class ApprovalStatus(Enum):
    """Status of a gate approval."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"  # When gates are disabled


class GateType(Enum):
    """Types of approval gates."""

    DESIGN = "design"
    TEST_QUALITY = "test_quality"
    COMMIT = "commit"


@dataclass
class ApprovalDecision:
    """Record of an approval decision."""

    gate_type: GateType
    status: ApprovalStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)
    approver: str = "unknown"  # human, auto, system
    artifact_hash: str = ""  # Hash of what was approved
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "gate_type": self.gate_type.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "approver": self.approver,
            "artifact_hash": self.artifact_hash,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class ApprovalRequired(Exception):
    """Raised when a gate requires approval before proceeding."""

    def __init__(
        self,
        gate_type: GateType,
        message: str,
        artifact: Optional[str] = None,
        recoverable: bool = True,
    ):
        self.gate_type = gate_type
        self.artifact = artifact
        self.recoverable = recoverable
        super().__init__(f"[{gate_type.value}] Approval required: {message}")


class ApprovalGate(ABC):
    """
    Base class for approval gates.

    Gates check conditions and require explicit approval before
    allowing the nomic loop to proceed to the next phase.
    """

    def __init__(
        self,
        gate_type: GateType,
        enabled: bool = True,
        auto_approve_dev: bool = True,
        approval_callback: Optional[Callable[[str], bool]] = None,
    ):
        """
        Initialize the gate.

        Args:
            gate_type: Type of this gate
            enabled: Whether the gate is active
            auto_approve_dev: Auto-approve in development mode
            approval_callback: Optional callback for custom approval logic
        """
        self.gate_type = gate_type
        self.enabled = enabled
        self.auto_approve_dev = auto_approve_dev
        self._approval_callback = approval_callback
        self._decisions: List[ApprovalDecision] = []

    @property
    def is_dev_mode(self) -> bool:
        """Check if running in development mode."""
        return os.environ.get("ARAGORA_DEV_MODE", "0") == "1"

    @property
    def skip_gates(self) -> bool:
        """Check if gates should be skipped."""
        return os.environ.get("ARAGORA_SKIP_GATES", "0") == "1"

    def hash_artifact(self, artifact: str) -> str:
        """Create hash of artifact for tracking."""
        return hashlib.sha256(artifact.encode()).hexdigest()[:16]

    @abstractmethod
    async def check(self, artifact: Any, context: Dict[str, Any]) -> ApprovalDecision:
        """
        Check if the gate should allow proceeding.

        Args:
            artifact: The artifact to evaluate (design, test results, etc.)
            context: Additional context for the decision

        Returns:
            ApprovalDecision with the outcome
        """
        pass

    async def require_approval(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ApprovalDecision:
        """
        Require approval before proceeding.

        Args:
            artifact: What needs approval
            context: Additional context

        Returns:
            ApprovalDecision if approved

        Raises:
            ApprovalRequired: If approval is needed but not granted
        """
        context = context or {}

        # Check if gates are disabled
        if not self.enabled or self.skip_gates:
            decision = ApprovalDecision(
                gate_type=self.gate_type,
                status=ApprovalStatus.SKIPPED,
                approver="system",
                reason="Gates disabled",
            )
            self._decisions.append(decision)
            logger.debug(f"Gate {self.gate_type.value} skipped (disabled)")
            return decision

        # Check gate conditions
        decision = await self.check(artifact, context)
        self._decisions.append(decision)

        if decision.status == ApprovalStatus.REJECTED:
            raise ApprovalRequired(
                gate_type=self.gate_type,
                message=decision.reason,
                artifact=str(artifact)[:500],
            )

        return decision

    def get_decisions(self) -> List[ApprovalDecision]:
        """Get all decisions made by this gate."""
        return self._decisions.copy()


class DesignGate(ApprovalGate):
    """
    Gate that requires approval before implementation begins.

    Validates design documents and requires explicit sign-off
    before code changes are made.
    """

    def __init__(
        self,
        enabled: bool = True,
        auto_approve_dev: bool = True,
        max_complexity_score: float = 0.8,
        require_files_list: bool = True,
    ):
        """
        Initialize design gate.

        Args:
            enabled: Whether gate is active
            auto_approve_dev: Auto-approve in dev mode
            max_complexity_score: Maximum allowed complexity (0-1)
            require_files_list: Whether design must list affected files
        """
        super().__init__(
            gate_type=GateType.DESIGN,
            enabled=enabled,
            auto_approve_dev=auto_approve_dev,
        )
        self.max_complexity_score = max_complexity_score
        self.require_files_list = require_files_list

    async def check(self, artifact: Any, context: Dict[str, Any]) -> ApprovalDecision:
        """
        Check if design should be approved.

        Args:
            artifact: Design document string
            context: Should contain 'complexity_score', 'files_affected'

        Returns:
            ApprovalDecision
        """
        design = str(artifact) if artifact else ""
        artifact_hash = self.hash_artifact(design)

        # Check complexity
        complexity = context.get("complexity_score", 0.5)
        if complexity > self.max_complexity_score:
            return ApprovalDecision(
                gate_type=self.gate_type,
                status=ApprovalStatus.REJECTED,
                artifact_hash=artifact_hash,
                reason=f"Design complexity ({complexity:.2f}) exceeds threshold ({self.max_complexity_score})",
                metadata={"complexity_score": complexity},
            )

        # Check files list
        files_affected = context.get("files_affected", [])
        if self.require_files_list and not files_affected:
            return ApprovalDecision(
                gate_type=self.gate_type,
                status=ApprovalStatus.REJECTED,
                artifact_hash=artifact_hash,
                reason="Design must specify affected files",
            )

        # Auto-approve in dev mode
        if self.auto_approve_dev and self.is_dev_mode:
            return ApprovalDecision(
                gate_type=self.gate_type,
                status=ApprovalStatus.APPROVED,
                approver="auto_dev",
                artifact_hash=artifact_hash,
                reason="Auto-approved in dev mode",
                metadata={"complexity_score": complexity, "files_affected": files_affected},
            )

        # Request interactive approval
        approved = await self._request_interactive_approval(design, context)

        return ApprovalDecision(
            gate_type=self.gate_type,
            status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
            approver="human" if approved else "human_declined",
            artifact_hash=artifact_hash,
            reason="Human approved" if approved else "Human declined",
            metadata={"complexity_score": complexity, "files_affected": files_affected},
        )

    async def _request_interactive_approval(self, design: str, context: Dict[str, Any]) -> bool:
        """Request interactive approval from user."""
        # Check for non-interactive mode
        if not sys.stdin.isatty():
            logger.warning("Design gate requires approval but running non-interactively")
            # Check for auto-approval flag
            if os.environ.get("NOMIC_AUTO_APPROVE", "0") == "1":
                logger.info("Auto-approving design (NOMIC_AUTO_APPROVE=1)")
                return True
            return False

        # Show design summary
        print("\n" + "=" * 70)
        print("DESIGN APPROVAL REQUIRED")
        print("=" * 70)
        print(f"\nComplexity Score: {context.get('complexity_score', 'N/A')}")
        print(f"Files Affected: {context.get('files_affected', [])}")
        print("\nDesign Summary:")
        print("-" * 40)
        # Show first 1000 chars of design
        print(design[:1000])
        if len(design) > 1000:
            print(f"... ({len(design) - 1000} more characters)")
        print("-" * 40)

        response = input("\nApprove this design? [y/N]: ")
        return response.lower() == "y"


class TestQualityGate(ApprovalGate):
    """
    Gate that validates test quality after verification.

    Ensures tests pass quality thresholds before allowing commit.
    """

    def __init__(
        self,
        enabled: bool = True,
        require_all_tests_pass: bool = True,
        min_coverage: float = 0.0,  # 0 = no coverage requirement
        max_new_warnings: int = 0,
    ):
        """
        Initialize test quality gate.

        Args:
            enabled: Whether gate is active
            require_all_tests_pass: Require all tests to pass
            min_coverage: Minimum coverage percentage (0-100)
            max_new_warnings: Maximum allowed new warnings
        """
        super().__init__(
            gate_type=GateType.TEST_QUALITY,
            enabled=enabled,
            auto_approve_dev=False,  # Never auto-approve test quality
        )
        self.require_all_tests_pass = require_all_tests_pass
        self.min_coverage = min_coverage
        self.max_new_warnings = max_new_warnings

    async def check(self, artifact: Any, context: Dict[str, Any]) -> ApprovalDecision:
        """
        Check if test quality meets thresholds.

        Args:
            artifact: VerifyResult or test output
            context: Should contain 'tests_passed', 'coverage', 'warnings_count'

        Returns:
            ApprovalDecision
        """
        tests_passed = context.get("tests_passed", False)
        coverage = context.get("coverage", 0.0)
        warnings_count = context.get("warnings_count", 0)
        test_output = str(artifact) if artifact else ""
        artifact_hash = self.hash_artifact(test_output)

        rejection_reasons = []

        # Check tests passed
        if self.require_all_tests_pass and not tests_passed:
            rejection_reasons.append("Not all tests passed")

        # Check coverage
        if self.min_coverage > 0 and coverage < self.min_coverage:
            rejection_reasons.append(
                f"Coverage ({coverage:.1f}%) below threshold ({self.min_coverage}%)"
            )

        # Check warnings
        if warnings_count > self.max_new_warnings:
            rejection_reasons.append(
                f"New warnings ({warnings_count}) exceed limit ({self.max_new_warnings})"
            )

        if rejection_reasons:
            return ApprovalDecision(
                gate_type=self.gate_type,
                status=ApprovalStatus.REJECTED,
                artifact_hash=artifact_hash,
                reason="; ".join(rejection_reasons),
                metadata={
                    "tests_passed": tests_passed,
                    "coverage": coverage,
                    "warnings_count": warnings_count,
                },
            )

        return ApprovalDecision(
            gate_type=self.gate_type,
            status=ApprovalStatus.APPROVED,
            approver="auto_quality",
            artifact_hash=artifact_hash,
            reason="All quality thresholds met",
            metadata={
                "tests_passed": tests_passed,
                "coverage": coverage,
                "warnings_count": warnings_count,
            },
        )


class CommitGate(ApprovalGate):
    """
    Gate that provides structured approval before commit.

    Enhances the existing commit approval with:
    - Structured diff view
    - Rollback information
    - Web UI option (via callback)
    """

    def __init__(
        self,
        enabled: bool = True,
        aragora_path: Optional[Path] = None,
        web_ui_callback: Optional[Callable[[Dict], bool]] = None,
    ):
        """
        Initialize commit gate.

        Args:
            enabled: Whether gate is active
            aragora_path: Path to aragora root for git operations
            web_ui_callback: Optional callback for web-based approval
        """
        super().__init__(
            gate_type=GateType.COMMIT,
            enabled=enabled,
            auto_approve_dev=False,  # Never auto-approve commits
        )
        self.aragora_path = aragora_path or Path.cwd()
        self.web_ui_callback = web_ui_callback

    async def check(self, artifact: Any, context: Dict[str, Any]) -> ApprovalDecision:
        """
        Check if commit should be approved.

        Args:
            artifact: Commit message or diff summary
            context: Should contain 'files_changed', 'improvement_summary'

        Returns:
            ApprovalDecision
        """
        commit_info = str(artifact) if artifact else ""
        artifact_hash = self.hash_artifact(commit_info)
        files_changed = context.get("files_changed", [])
        improvement = context.get("improvement_summary", "")

        # Check for auto-commit mode
        if os.environ.get("NOMIC_AUTO_COMMIT", "0") == "1":
            return ApprovalDecision(
                gate_type=self.gate_type,
                status=ApprovalStatus.APPROVED,
                approver="auto_commit",
                artifact_hash=artifact_hash,
                reason="Auto-commit enabled",
                metadata={"files_changed": files_changed},
            )

        # Try web UI callback first
        if self.web_ui_callback:
            try:
                approval_data = {
                    "commit_info": commit_info,
                    "files_changed": files_changed,
                    "improvement": improvement,
                    "artifact_hash": artifact_hash,
                }
                approved = self.web_ui_callback(approval_data)
                return ApprovalDecision(
                    gate_type=self.gate_type,
                    status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
                    approver="web_ui",
                    artifact_hash=artifact_hash,
                    reason="Web UI approval" if approved else "Web UI rejected",
                    metadata={"files_changed": files_changed},
                )
            except Exception as e:
                logger.warning(f"Web UI callback failed: {e}, falling back to CLI")

        # Fall back to CLI approval
        approved = await self._request_cli_approval(commit_info, files_changed, improvement)

        return ApprovalDecision(
            gate_type=self.gate_type,
            status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
            approver="human_cli" if approved else "human_declined",
            artifact_hash=artifact_hash,
            reason="CLI approval" if approved else "CLI rejected",
            metadata={"files_changed": files_changed},
        )

    async def _request_cli_approval(
        self,
        commit_info: str,
        files_changed: List[str],
        improvement: str,
    ) -> bool:
        """Request CLI approval for commit."""
        if not sys.stdin.isatty():
            logger.warning("Commit gate requires approval but running non-interactively")
            return False

        print("\n" + "=" * 70)
        print("COMMIT APPROVAL REQUIRED")
        print("=" * 70)
        print(f"\nImprovement: {improvement[:200]}...")
        print(f"\nFiles Changed ({len(files_changed)}):")
        for f in files_changed[:10]:
            print(f"  - {f}")
        if len(files_changed) > 10:
            print(f"  ... and {len(files_changed) - 10} more")
        print("\nCommit Info:")
        print("-" * 40)
        print(commit_info[:500])
        print("-" * 40)

        response = input("\nCommit these changes? [y/N]: ")
        return response.lower() == "y"


# Gate factory for common configurations
def create_standard_gates(
    aragora_path: Optional[Path] = None,
    dev_mode: bool = False,
) -> Dict[GateType, ApprovalGate]:
    """
    Create standard gate configuration.

    Args:
        aragora_path: Path to aragora root
        dev_mode: Whether to enable dev mode auto-approval

    Returns:
        Dictionary of gate type to gate instance
    """
    return {
        GateType.DESIGN: DesignGate(
            enabled=True,
            auto_approve_dev=dev_mode,
            max_complexity_score=0.8,
        ),
        GateType.TEST_QUALITY: TestQualityGate(
            enabled=True,
            require_all_tests_pass=True,
            min_coverage=0.0,  # Coverage check disabled by default
        ),
        GateType.COMMIT: CommitGate(
            enabled=True,
            aragora_path=aragora_path,
        ),
    }


__all__ = [
    # Simple gate functions
    "GateConfig",
    "is_protected_file",
    "check_change_volume",
    "check_dangerous_patterns",
    "check_resource_limits",
    "check_all_gates",
    # Approval gate classes
    "ApprovalGate",
    "ApprovalDecision",
    "ApprovalRequired",
    "ApprovalStatus",
    "GateType",
    "DesignGate",
    "TestQualityGate",
    "CommitGate",
    "create_standard_gates",
]
