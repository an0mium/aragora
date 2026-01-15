"""
Commit phase for nomic loop.

Phase 5: Commit changes if verified
- Human approval check (if required)
- Git add and commit
- Commit hash tracking
- Structured commit gate with audit trail
"""

# Import automation flag from environment
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

from . import CommitResult

if TYPE_CHECKING:
    from aragora.nomic.gates import CommitGate

NOMIC_AUTO_COMMIT = os.environ.get("NOMIC_AUTO_COMMIT", "0") == "1"


class CommitPhase:
    """
    Handles committing verified changes to git.

    Supports human approval workflow and auto-commit mode.
    """

    def __init__(
        self,
        aragora_path: Path,
        require_human_approval: bool = True,
        auto_commit: bool = False,
        cycle_count: int = 0,
        log_fn: Optional[Callable[[str], None]] = None,
        stream_emit_fn: Optional[Callable[..., None]] = None,
        commit_gate: Optional["CommitGate"] = None,
    ):
        """
        Initialize the commit phase.

        Args:
            aragora_path: Path to the aragora project root
            require_human_approval: Whether to require human approval for commits
            auto_commit: Whether to auto-commit without prompting
            cycle_count: Current cycle number
            log_fn: Function to log messages
            stream_emit_fn: Function to emit streaming events
            commit_gate: Optional CommitGate for structured approval with audit trail
        """
        self.aragora_path = aragora_path
        self.require_human_approval = require_human_approval
        self.auto_commit = auto_commit
        self.cycle_count = cycle_count
        self._log = log_fn or print
        self._stream_emit = stream_emit_fn or (lambda *args: None)
        self._commit_gate = commit_gate

    async def execute(self, improvement: str) -> CommitResult:
        """
        Execute the commit phase.

        Args:
            improvement: Description of the improvement being committed

        Returns:
            CommitResult with commit status and hash
        """
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 5: COMMIT")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "commit", self.cycle_count, {})

        # Get changed files for gate context
        changed_files = self._get_changed_files()

        # === SAFETY: Commit approval gate ===
        gate_decision = None
        if self._commit_gate:
            try:
                from aragora.nomic.gates import ApprovalRequired, ApprovalStatus

                gate_context = {
                    "files_changed": changed_files,
                    "improvement_summary": improvement[:200],
                }

                # Get diff for commit info
                diff_result = subprocess.run(
                    ["git", "diff", "--stat"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                commit_info = diff_result.stdout if diff_result.returncode == 0 else ""

                gate_decision = await self._commit_gate.require_approval(commit_info, gate_context)

                if gate_decision.status == ApprovalStatus.APPROVED:
                    self._log(f"  [gate] Commit approved by {gate_decision.approver}")
                elif gate_decision.status == ApprovalStatus.SKIPPED:
                    self._log("  [gate] Commit gate skipped (disabled)")

            except ApprovalRequired as e:
                self._log(f"  [gate] Commit approval denied: {e}")
                phase_duration = (datetime.now() - phase_start).total_seconds()
                self._stream_emit(
                    "on_phase_end",
                    "commit",
                    self.cycle_count,
                    False,
                    phase_duration,
                    {"reason": "gate_declined", "gate": "commit"},
                )
                return CommitResult(
                    success=False,
                    data={"reason": "Commit gate declined", "gate": "commit"},
                    duration_seconds=phase_duration,
                    commit_hash=None,
                    committed=False,
                )
            except Exception as gate_error:
                self._log(f"  [gate] Gate check error: {gate_error}, falling back to legacy")
                # Fall through to legacy approval

        # Legacy approval check if gate not used or failed
        if not gate_decision and self.require_human_approval and not self.auto_commit:
            approval = self._get_approval()
            if not approval:
                phase_duration = (datetime.now() - phase_start).total_seconds()
                self._stream_emit(
                    "on_phase_end",
                    "commit",
                    self.cycle_count,
                    False,
                    phase_duration,
                    {"reason": "human_declined"},
                )
                return CommitResult(
                    success=False,
                    data={"reason": "Human declined"},
                    duration_seconds=phase_duration,
                    commit_hash=None,
                    committed=False,
                )

        summary = improvement.replace("\n", " ")

        try:
            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.aragora_path,
                check=True,
            )

            # Commit
            result = subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"feat(nomic): {summary}\n\n🤖 Auto-generated by aragora nomic loop",
                ],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )

            committed = result.returncode == 0
            commit_hash = None

            if committed:
                self._log(f"  Committed: {summary}")
                # Get commit hash
                hash_result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                commit_hash = (
                    hash_result.stdout.strip() if hash_result.returncode == 0 else "unknown"
                )

                # Get files changed count
                stat_result = subprocess.run(
                    ["git", "diff", "--stat", "HEAD~1..HEAD"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                files_changed = len(
                    [line for line in stat_result.stdout.split("\n") if "|" in line]
                )
                self._stream_emit("on_commit", commit_hash, summary, files_changed)
            else:
                self._log(f"  Commit failed: {result.stderr}")

            phase_duration = (datetime.now() - phase_start).total_seconds()
            self._stream_emit(
                "on_phase_end", "commit", self.cycle_count, committed, phase_duration, {}
            )

            return CommitResult(
                success=committed,
                data={"message": summary},
                duration_seconds=phase_duration,
                commit_hash=commit_hash,
                committed=committed,
            )

        except Exception as e:
            self._log(f"  Commit error: {e}")
            phase_duration = (datetime.now() - phase_start).total_seconds()
            self._stream_emit(
                "on_phase_end", "commit", self.cycle_count, False, phase_duration, {"error": str(e)}
            )
            self._stream_emit("on_error", "commit", str(e), True)
            return CommitResult(
                success=False,
                error=str(e),
                data={},
                duration_seconds=phase_duration,
                commit_hash=None,
                committed=False,
            )

    def _get_approval(self) -> bool:
        """Get human approval for commit."""
        self._log("\nChanges ready for review:")
        subprocess.run(["git", "diff", "--stat"], cwd=self.aragora_path)

        # Check if auto-commit is enabled or running non-interactively
        if NOMIC_AUTO_COMMIT:
            self._log("\n[commit] Auto-committing (NOMIC_AUTO_COMMIT=1)")
            return True
        elif not sys.stdin.isatty():
            self._log("\n[commit] Non-interactive mode detected, auto-commit disabled")
            self._log("[commit] Set NOMIC_AUTO_COMMIT=1 to allow unattended commits")
            return False
        else:
            # Interactive mode: prompt for approval
            response = input("\nCommit these changes? [y/N]: ")
            return response.lower() == "y"

    def _get_changed_files(self) -> List[str]:
        """Get list of changed files."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        except Exception as e:
            self._log(f"  [git] Failed to get changed files: {e}")
        return []


__all__ = ["CommitPhase"]
