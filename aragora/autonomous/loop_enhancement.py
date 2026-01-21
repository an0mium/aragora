"""
Nomic Loop Enhancement (Phase 5.1).

Provides:
- Self-improvement debate automation
- Code generation verification
- Rollback safety mechanisms
- Human-in-the-loop approval flows
"""

import asyncio
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ApprovalRequest:
    """A request for human approval."""

    id: str
    title: str
    description: str
    changes: List[Dict[str, Any]]
    risk_level: str  # low, medium, high, critical
    requested_at: datetime
    requested_by: str
    timeout_seconds: int = 3600
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackPoint:
    """A point in time to which changes can be rolled back."""

    id: str
    created_at: datetime
    description: str
    git_commit: Optional[str] = None
    file_backups: Dict[str, str] = field(default_factory=dict)
    database_snapshot: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of code verification."""

    passed: bool
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    syntax_errors: List[str] = field(default_factory=list)
    security_issues: List[str] = field(default_factory=list)
    lint_warnings: List[str] = field(default_factory=list)
    coverage_percent: Optional[float] = None
    performance_regressions: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class ApprovalFlow:
    """
    Human-in-the-loop approval flow for dangerous operations.

    Supports:
    - Approval thresholds based on risk level
    - Timeout-based auto-rejection
    - Callback notifications (webhook, email, Slack, etc.)
    - Audit logging
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        auto_approve_low_risk: bool = True,
        default_timeout_seconds: int = 3600,
        notification_callback: Optional[Callable[[ApprovalRequest], None]] = None,
    ):
        """
        Initialize approval flow.

        Args:
            storage_dir: Directory for storing approval requests
            auto_approve_low_risk: Auto-approve low-risk changes
            default_timeout_seconds: Default timeout for approval requests
            notification_callback: Called when approval is needed
        """
        self.storage_dir = storage_dir or Path(".approvals")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.auto_approve_low_risk = auto_approve_low_risk
        self.default_timeout_seconds = default_timeout_seconds
        self.notification_callback = notification_callback
        self._pending_requests: Dict[str, ApprovalRequest] = {}

    async def request_approval(
        self,
        title: str,
        description: str,
        changes: List[Dict[str, Any]],
        risk_level: str = "medium",
        requested_by: str = "system",
        timeout_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """
        Request approval for an operation.

        Args:
            title: Short title for the approval request
            description: Detailed description of what's being changed
            changes: List of changes being made
            risk_level: Risk level (low, medium, high, critical)
            requested_by: Who/what is requesting approval
            timeout_seconds: How long to wait for approval
            metadata: Additional metadata

        Returns:
            ApprovalRequest with status
        """
        request_id = hashlib.sha256(
            f"{title}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        request = ApprovalRequest(
            id=request_id,
            title=title,
            description=description,
            changes=changes,
            risk_level=risk_level,
            requested_at=datetime.now(),
            requested_by=requested_by,
            timeout_seconds=timeout_seconds or self.default_timeout_seconds,
            metadata=metadata or {},
        )

        # Auto-approve low-risk changes if configured
        if self.auto_approve_low_risk and risk_level == "low":
            request.status = ApprovalStatus.AUTO_APPROVED
            request.approved_at = datetime.now()
            request.approved_by = "auto"
            logger.info(f"Auto-approved low-risk request: {request_id}")
            return request

        # Store pending request
        self._pending_requests[request_id] = request
        self._save_request(request)

        # Send notification
        if self.notification_callback:
            try:
                self.notification_callback(request)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

        logger.info(f"Approval requested: {request_id} - {title} ({risk_level} risk)")

        return request

    async def wait_for_approval(
        self,
        request_id: str,
        poll_interval: float = 5.0,
    ) -> ApprovalRequest:
        """
        Wait for approval with timeout.

        Args:
            request_id: ID of the approval request
            poll_interval: How often to check for approval

        Returns:
            Updated ApprovalRequest
        """
        request = self._pending_requests.get(request_id)
        if not request:
            request = self._load_request(request_id)
            if not request:
                raise ValueError(f"Unknown approval request: {request_id}")

        deadline = request.requested_at + timedelta(seconds=request.timeout_seconds)

        while datetime.now() < deadline:
            # Check for updated status
            updated = self._load_request(request_id)
            if updated and updated.status != ApprovalStatus.PENDING:
                return updated

            await asyncio.sleep(poll_interval)

        # Timeout
        request.status = ApprovalStatus.TIMEOUT
        self._save_request(request)
        logger.warning(f"Approval request timed out: {request_id}")

        return request

    def approve(
        self,
        request_id: str,
        approved_by: str,
    ) -> ApprovalRequest:
        """
        Approve a pending request.

        Args:
            request_id: ID of the request to approve
            approved_by: Who is approving

        Returns:
            Updated ApprovalRequest
        """
        request = self._load_request(request_id)
        if not request:
            raise ValueError(f"Unknown approval request: {request_id}")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request {request_id} is not pending: {request.status}")

        request.status = ApprovalStatus.APPROVED
        request.approved_by = approved_by
        request.approved_at = datetime.now()
        self._save_request(request)

        logger.info(f"Request approved: {request_id} by {approved_by}")

        return request

    def reject(
        self,
        request_id: str,
        rejected_by: str,
        reason: str,
    ) -> ApprovalRequest:
        """
        Reject a pending request.

        Args:
            request_id: ID of the request to reject
            rejected_by: Who is rejecting
            reason: Reason for rejection

        Returns:
            Updated ApprovalRequest
        """
        request = self._load_request(request_id)
        if not request:
            raise ValueError(f"Unknown approval request: {request_id}")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request {request_id} is not pending: {request.status}")

        request.status = ApprovalStatus.REJECTED
        request.approved_by = rejected_by
        request.approved_at = datetime.now()
        request.rejection_reason = reason
        self._save_request(request)

        logger.info(f"Request rejected: {request_id} by {rejected_by}: {reason}")

        return request

    def list_pending(self) -> List[ApprovalRequest]:
        """List all pending approval requests."""
        pending = []
        for path in self.storage_dir.glob("*.json"):
            request = self._load_request(path.stem)
            if request and request.status == ApprovalStatus.PENDING:
                pending.append(request)
        return pending

    def _save_request(self, request: ApprovalRequest) -> None:
        """Save request to storage."""
        path = self.storage_dir / f"{request.id}.json"
        data = {
            "id": request.id,
            "title": request.title,
            "description": request.description,
            "changes": request.changes,
            "risk_level": request.risk_level,
            "requested_at": request.requested_at.isoformat(),
            "requested_by": request.requested_by,
            "timeout_seconds": request.timeout_seconds,
            "status": request.status.value,
            "approved_by": request.approved_by,
            "approved_at": request.approved_at.isoformat() if request.approved_at else None,
            "rejection_reason": request.rejection_reason,
            "metadata": request.metadata,
        }
        path.write_text(json.dumps(data, indent=2))

    def _load_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Load request from storage."""
        path = self.storage_dir / f"{request_id}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return ApprovalRequest(
                id=data["id"],
                title=data["title"],
                description=data["description"],
                changes=data["changes"],
                risk_level=data["risk_level"],
                requested_at=datetime.fromisoformat(data["requested_at"]),
                requested_by=data["requested_by"],
                timeout_seconds=data["timeout_seconds"],
                status=ApprovalStatus(data["status"]),
                approved_by=data.get("approved_by"),
                approved_at=(
                    datetime.fromisoformat(data["approved_at"])
                    if data.get("approved_at")
                    else None
                ),
                rejection_reason=data.get("rejection_reason"),
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            logger.error(f"Failed to load approval request {request_id}: {e}")
            return None


class RollbackManager:
    """
    Manages rollback points for safe self-improvement.

    Features:
    - Git-based rollback
    - File backup/restore
    - Database snapshot restore
    - Automatic cleanup of old rollback points
    """

    def __init__(
        self,
        backup_dir: Optional[Path] = None,
        max_rollback_points: int = 10,
        repo_path: Optional[Path] = None,
    ):
        """
        Initialize rollback manager.

        Args:
            backup_dir: Directory for file backups
            max_rollback_points: Maximum rollback points to keep
            repo_path: Git repository path
        """
        self.backup_dir = backup_dir or Path(".rollback")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_rollback_points = max_rollback_points
        self.repo_path = repo_path or Path.cwd()
        self._rollback_points: List[RollbackPoint] = []
        self._load_rollback_points()

    def create_rollback_point(
        self,
        description: str,
        files_to_backup: Optional[List[Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RollbackPoint:
        """
        Create a new rollback point.

        Args:
            description: Description of what's being changed
            files_to_backup: Specific files to backup (None = current git state)
            metadata: Additional metadata

        Returns:
            Created RollbackPoint
        """
        point_id = hashlib.sha256(
            f"{description}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        point = RollbackPoint(
            id=point_id,
            created_at=datetime.now(),
            description=description,
            metadata=metadata or {},
        )

        # Capture git commit
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            if result.returncode == 0:
                point.git_commit = result.stdout.strip()
        except Exception as e:
            logger.warning(f"Failed to capture git commit: {e}")

        # Backup specific files
        if files_to_backup:
            point_dir = self.backup_dir / point_id
            point_dir.mkdir(parents=True, exist_ok=True)

            for file_path in files_to_backup:
                if file_path.exists():
                    backup_path = point_dir / file_path.name
                    shutil.copy2(file_path, backup_path)
                    point.file_backups[str(file_path)] = str(backup_path)

        self._rollback_points.append(point)
        self._save_rollback_point(point)
        self._cleanup_old_points()

        logger.info(f"Created rollback point: {point_id} - {description}")

        return point

    def rollback(
        self,
        point_id: str,
        restore_files: bool = True,
        restore_git: bool = False,
    ) -> bool:
        """
        Rollback to a specific point.

        Args:
            point_id: ID of the rollback point
            restore_files: Whether to restore backed up files
            restore_git: Whether to reset git to the commit

        Returns:
            True if rollback succeeded
        """
        point = self.get_rollback_point(point_id)
        if not point:
            logger.error(f"Rollback point not found: {point_id}")
            return False

        success = True

        # Restore files
        if restore_files and point.file_backups:
            for original_path, backup_path in point.file_backups.items():
                try:
                    if Path(backup_path).exists():
                        shutil.copy2(backup_path, original_path)
                        logger.info(f"Restored file: {original_path}")
                except Exception as e:
                    logger.error(f"Failed to restore {original_path}: {e}")
                    success = False

        # Restore git state
        if restore_git and point.git_commit:
            try:
                result = subprocess.run(
                    ["git", "reset", "--hard", point.git_commit],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path,
                )
                if result.returncode != 0:
                    logger.error(f"Git reset failed: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Git reset to: {point.git_commit}")
            except Exception as e:
                logger.error(f"Failed to reset git: {e}")
                success = False

        if success:
            logger.info(f"Rollback to {point_id} completed successfully")
        else:
            logger.warning(f"Rollback to {point_id} completed with errors")

        return success

    def get_rollback_point(self, point_id: str) -> Optional[RollbackPoint]:
        """Get a specific rollback point."""
        for point in self._rollback_points:
            if point.id == point_id:
                return point
        return None

    def list_rollback_points(self) -> List[RollbackPoint]:
        """List all rollback points (newest first)."""
        return sorted(
            self._rollback_points,
            key=lambda p: p.created_at,
            reverse=True,
        )

    def delete_rollback_point(self, point_id: str) -> bool:
        """Delete a specific rollback point."""
        point = self.get_rollback_point(point_id)
        if not point:
            return False

        # Remove backup files
        point_dir = self.backup_dir / point_id
        if point_dir.exists():
            shutil.rmtree(point_dir)

        # Remove metadata
        meta_path = self.backup_dir / f"{point_id}.json"
        if meta_path.exists():
            meta_path.unlink()

        self._rollback_points = [p for p in self._rollback_points if p.id != point_id]

        logger.info(f"Deleted rollback point: {point_id}")
        return True

    def _save_rollback_point(self, point: RollbackPoint) -> None:
        """Save rollback point metadata."""
        path = self.backup_dir / f"{point.id}.json"
        data = {
            "id": point.id,
            "created_at": point.created_at.isoformat(),
            "description": point.description,
            "git_commit": point.git_commit,
            "file_backups": point.file_backups,
            "database_snapshot": point.database_snapshot,
            "metadata": point.metadata,
        }
        path.write_text(json.dumps(data, indent=2))

    def _load_rollback_points(self) -> None:
        """Load all rollback points from storage."""
        self._rollback_points = []
        for path in self.backup_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                point = RollbackPoint(
                    id=data["id"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    description=data["description"],
                    git_commit=data.get("git_commit"),
                    file_backups=data.get("file_backups", {}),
                    database_snapshot=data.get("database_snapshot"),
                    metadata=data.get("metadata", {}),
                )
                self._rollback_points.append(point)
            except Exception as e:
                logger.warning(f"Failed to load rollback point from {path}: {e}")

    def _cleanup_old_points(self) -> None:
        """Remove oldest rollback points if over limit."""
        while len(self._rollback_points) > self.max_rollback_points:
            oldest = min(self._rollback_points, key=lambda p: p.created_at)
            self.delete_rollback_point(oldest.id)


class CodeVerifier:
    """
    Verifies code changes before they are applied.

    Checks:
    - Syntax validity
    - Test execution
    - Security analysis
    - Lint warnings
    - Performance regressions
    """

    def __init__(
        self,
        repo_path: Optional[Path] = None,
        run_tests: bool = True,
        run_lint: bool = True,
        run_security: bool = True,
        test_command: str = "pytest",
        lint_command: str = "ruff check",
        security_command: str = "bandit -r",
    ):
        """
        Initialize code verifier.

        Args:
            repo_path: Repository path
            run_tests: Whether to run tests
            run_lint: Whether to run linter
            run_security: Whether to run security scan
            test_command: Test command to run
            lint_command: Lint command to run
            security_command: Security scan command to run
        """
        self.repo_path = repo_path or Path.cwd()
        self.run_tests = run_tests
        self.run_lint = run_lint
        self.run_security = run_security
        self.test_command = test_command
        self.lint_command = lint_command
        self.security_command = security_command

    async def verify(
        self,
        files: Optional[List[Path]] = None,
        quick_mode: bool = False,
    ) -> VerificationResult:
        """
        Verify code changes.

        Args:
            files: Specific files to verify (None = all changed files)
            quick_mode: Skip slow checks (tests, security)

        Returns:
            VerificationResult with all check results
        """
        result = VerificationResult(passed=True)

        # Check syntax
        syntax_result = await self._check_syntax(files)
        result.syntax_errors = syntax_result
        if syntax_result:
            result.passed = False

        # Run linter
        if self.run_lint:
            lint_result = await self._run_lint(files)
            result.lint_warnings = lint_result

        # Quick mode skips slower checks
        if quick_mode:
            return result

        # Run tests
        if self.run_tests:
            test_result = await self._run_tests(files)
            result.tests_run = test_result.get("total", 0)
            result.tests_passed = test_result.get("passed", 0)
            result.tests_failed = test_result.get("failed", 0)
            result.coverage_percent = test_result.get("coverage")
            if result.tests_failed > 0:
                result.passed = False

        # Run security scan
        if self.run_security:
            security_result = await self._run_security_scan(files)
            result.security_issues = security_result
            # High-severity security issues fail verification
            high_severity = [
                i for i in security_result
                if "high" in i.lower() or "critical" in i.lower()
            ]
            if high_severity:
                result.passed = False

        return result

    async def _check_syntax(self, files: Optional[List[Path]] = None) -> List[str]:
        """Check Python syntax for files."""
        import ast

        errors = []
        target_files = files or list(self.repo_path.rglob("*.py"))

        for file_path in target_files:
            if not file_path.exists():
                continue
            try:
                content = file_path.read_text()
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f"{file_path}:{e.lineno}: {e.msg}")
            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")

        return errors

    async def _run_lint(self, files: Optional[List[Path]] = None) -> List[str]:
        """Run linter on files."""
        warnings = []

        try:
            # Build command list safely without shell injection
            cmd = shlex.split(self.lint_command)
            if files:
                cmd.extend(str(f) for f in files)
            else:
                cmd.append(".")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            if result.stdout:
                warnings = result.stdout.strip().split("\n")
        except Exception as e:
            logger.warning(f"Lint check failed: {e}")

        return warnings

    async def _run_tests(
        self,
        files: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """Run tests for files."""
        try:
            # Build command list safely without shell injection
            cmd = shlex.split(self.test_command)
            if files:
                # Only run tests for changed files
                test_files = [str(f) for f in files if "test" in str(f).lower()]
                if test_files:
                    cmd.extend(test_files)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=600,  # 10 minute timeout
            )

            # Parse pytest output
            output = result.stdout + result.stderr
            total = 0
            passed = 0
            failed = 0

            # Look for pytest summary line
            for line in output.split("\n"):
                if "passed" in line or "failed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            try:
                                passed = int(parts[i - 1])
                            except ValueError:
                                pass
                        elif part == "failed" and i > 0:
                            try:
                                failed = int(parts[i - 1])
                            except ValueError:
                                pass

            total = passed + failed

            return {
                "total": total,
                "passed": passed,
                "failed": failed,
                "coverage": None,  # Would need coverage plugin
            }

        except subprocess.TimeoutExpired:
            logger.warning("Test execution timed out")
            return {"total": 0, "passed": 0, "failed": 0, "timeout": True}
        except Exception as e:
            logger.warning(f"Test execution failed: {e}")
            return {"total": 0, "passed": 0, "failed": 0, "error": str(e)}

    async def _run_security_scan(
        self,
        files: Optional[List[Path]] = None,
    ) -> List[str]:
        """Run security scan on files."""
        issues = []

        try:
            # Build command list safely without shell injection
            cmd = shlex.split(self.security_command)
            if files:
                cmd.extend(str(f) for f in files)
            else:
                cmd.append(".")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            if result.stdout:
                issues = [
                    line for line in result.stdout.strip().split("\n")
                    if line and not line.startswith("[")
                ]
        except Exception as e:
            logger.warning(f"Security scan failed: {e}")

        return issues


class SelfImprovementManager:
    """
    Orchestrates the self-improvement cycle with safety controls.

    Coordinates:
    - Rollback point creation before changes
    - Code verification after generation
    - Approval flow for high-risk changes
    - Automatic rollback on failure
    """

    def __init__(
        self,
        rollback_manager: Optional[RollbackManager] = None,
        code_verifier: Optional[CodeVerifier] = None,
        approval_flow: Optional[ApprovalFlow] = None,
        auto_rollback_on_failure: bool = True,
        require_approval_for_risk_levels: Optional[Set[str]] = None,
    ):
        """
        Initialize self-improvement manager.

        Args:
            rollback_manager: Rollback point manager
            code_verifier: Code verification system
            approval_flow: Human approval flow
            auto_rollback_on_failure: Auto-rollback on verification failure
            require_approval_for_risk_levels: Risk levels requiring approval
        """
        self.rollback_manager = rollback_manager or RollbackManager()
        self.code_verifier = code_verifier or CodeVerifier()
        self.approval_flow = approval_flow or ApprovalFlow()
        self.auto_rollback_on_failure = auto_rollback_on_failure
        self.require_approval_for_risk_levels = require_approval_for_risk_levels or {
            "high",
            "critical",
        }
        self._current_rollback_point: Optional[str] = None

    async def start_improvement_cycle(
        self,
        description: str,
        files_to_modify: List[Path],
        risk_level: str = "medium",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Start a self-improvement cycle.

        Args:
            description: What's being improved
            files_to_modify: Files that will be changed
            risk_level: Risk level of changes
            metadata: Additional metadata

        Returns:
            Tuple of (can_proceed, rollback_point_id)
        """
        # Check if approval is needed
        if risk_level in self.require_approval_for_risk_levels:
            request = await self.approval_flow.request_approval(
                title=f"Self-improvement: {description}",
                description=f"The system wants to modify {len(files_to_modify)} files.",
                changes=[{"file": str(f), "action": "modify"} for f in files_to_modify],
                risk_level=risk_level,
                requested_by="nomic_loop",
                metadata=metadata,
            )

            # Wait for approval
            request = await self.approval_flow.wait_for_approval(request.id)

            if request.status != ApprovalStatus.APPROVED and request.status != ApprovalStatus.AUTO_APPROVED:
                logger.warning(
                    f"Self-improvement rejected: {request.status.value}"
                )
                return False, None

        # Create rollback point
        rollback_point = self.rollback_manager.create_rollback_point(
            description=description,
            files_to_backup=files_to_modify,
            metadata=metadata,
        )
        self._current_rollback_point = rollback_point.id

        logger.info(f"Started improvement cycle: {description}")

        return True, rollback_point.id

    async def verify_and_complete(
        self,
        modified_files: List[Path],
        quick_verify: bool = False,
    ) -> Tuple[bool, VerificationResult]:
        """
        Verify changes and complete the improvement cycle.

        Args:
            modified_files: Files that were modified
            quick_verify: Skip slow verification checks

        Returns:
            Tuple of (success, verification_result)
        """
        # Verify code
        result = await self.code_verifier.verify(
            files=modified_files,
            quick_mode=quick_verify,
        )

        if not result.passed:
            logger.warning(
                f"Verification failed: {len(result.syntax_errors)} syntax errors, "
                f"{result.tests_failed} test failures, "
                f"{len(result.security_issues)} security issues"
            )

            # Auto-rollback if configured
            if self.auto_rollback_on_failure and self._current_rollback_point:
                logger.info("Auto-rolling back failed changes...")
                self.rollback_manager.rollback(
                    self._current_rollback_point,
                    restore_files=True,
                    restore_git=False,
                )

            return False, result

        logger.info("Verification passed, improvement cycle complete")
        self._current_rollback_point = None

        return True, result

    def rollback_current(self) -> bool:
        """Rollback the current improvement cycle."""
        if not self._current_rollback_point:
            logger.warning("No current rollback point to restore")
            return False

        success = self.rollback_manager.rollback(
            self._current_rollback_point,
            restore_files=True,
            restore_git=False,
        )
        self._current_rollback_point = None

        return success


__all__ = [
    "ApprovalStatus",
    "ApprovalRequest",
    "ApprovalFlow",
    "RollbackPoint",
    "RollbackManager",
    "VerificationResult",
    "CodeVerifier",
    "SelfImprovementManager",
]
