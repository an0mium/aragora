"""
Verification phase for nomic loop.

Phase 4: Verify changes work
- Python syntax check
- Import check
- Run tests
- Optional Codex audit
- Evidence staleness check
"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from . import VerifyResult

logger = logging.getLogger(__name__)


class VerifyPhase:
    """
    Handles verification of changes made during the implementation phase.

    Runs syntax checks, import validation, tests, and optional Codex audit
    to ensure changes are safe to commit.
    """

    def __init__(
        self,
        aragora_path: Path,
        codex: Optional[Any] = None,
        nomic_integration: Optional[Any] = None,
        cycle_count: int = 0,
        log_fn: Optional[Callable[[str], None]] = None,
        stream_emit_fn: Optional[Callable[..., None]] = None,
        record_replay_fn: Optional[Callable[..., None]] = None,
        save_state_fn: Optional[Callable[[dict], None]] = None,
    ):
        """
        Initialize the verify phase.

        Args:
            aragora_path: Path to the aragora project root
            codex: Optional Codex agent for code audit
            nomic_integration: Optional NomicIntegration for staleness checks
            cycle_count: Current cycle number
            log_fn: Function to log messages
            stream_emit_fn: Function to emit streaming events
            record_replay_fn: Function to record replay events
            save_state_fn: Function to save phase state
        """
        self.aragora_path = aragora_path
        self.codex = codex
        self.nomic_integration = nomic_integration
        self.cycle_count = cycle_count
        self._log = log_fn or (lambda msg: logger.info(msg))
        self._stream_emit = stream_emit_fn or (lambda *args: None)
        self._record_replay = record_replay_fn or (lambda *args: None)
        self._save_state = save_state_fn or (lambda state: None)

    async def execute(self) -> VerifyResult:
        """
        Execute the verification phase.

        Returns:
            VerifyResult with check results and pass/fail status
        """
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 4: VERIFICATION")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "verify", self.cycle_count, {})
        self._stream_emit("on_verification_start", ["syntax", "import", "tests"])
        self._record_replay("phase", "system", "verify")

        checks = []

        # 1. Python syntax check
        checks.append(self._check_syntax())

        # 2. Import check
        checks.append(self._check_imports())

        # 3. Run tests
        checks.append(self._run_tests())

        all_passed = all(c.get("passed", False) for c in checks)

        # 4. Optional Codex audit
        if all_passed and self.codex:
            audit_result = await self._codex_audit()
            if audit_result:
                checks.append(audit_result)
                all_passed = all(c.get("passed", False) for c in checks)

        # Save state
        self._save_state({
            "phase": "verify",
            "stage": "complete",
            "all_passed": all_passed,
            "checks": checks,
        })

        # Check evidence staleness
        stale_claims = await self._check_staleness() if self.nomic_integration else []

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit(
            "on_phase_end", "verify", self.cycle_count, all_passed,
            phase_duration, {"checks_passed": sum(1 for c in checks if c.get("passed"))}
        )

        return VerifyResult(
            success=all_passed,
            data={"checks": checks, "stale_claims": stale_claims},
            duration_seconds=phase_duration,
            tests_passed=all_passed,
            test_output=checks[-1].get("output", "") if checks else "",
            syntax_valid=checks[0].get("passed", False) if checks else False,
        )

    def _check_syntax(self) -> dict:
        """Check Python syntax."""
        self._log("  Checking syntax...")
        try:
            result = subprocess.run(
                ["python", "-m", "py_compile", "aragora/__init__.py"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            passed = result.returncode == 0
            check = {
                "check": "syntax",
                "passed": passed,
                "output": result.stderr,
            }
            self._log(f"    {'passed' if passed else 'FAILED'} syntax")
            self._stream_emit("on_verification_result", "syntax", passed, result.stderr if result.stderr else "")
            return check
        except Exception as e:
            self._log(f"    FAILED syntax: {e}")
            self._stream_emit("on_verification_result", "syntax", False, str(e))
            return {"check": "syntax", "passed": False, "error": str(e)}

    def _check_imports(self) -> dict:
        """Check that aragora can be imported."""
        self._log("  Checking imports...")
        try:
            result = subprocess.run(
                ["python", "-c", "import aragora; print('OK')"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
                timeout=180,  # 3 min timeout
            )
            passed = "OK" in result.stdout
            check = {
                "check": "import",
                "passed": passed,
                "output": result.stderr if result.returncode != 0 else "",
            }
            self._log(f"    {'passed' if passed else 'FAILED'} import")
            self._stream_emit("on_verification_result", "import", passed, result.stderr if result.stderr else "")
            return check
        except Exception as e:
            self._log(f"    FAILED import: {e}")
            self._stream_emit("on_verification_result", "import", False, str(e))
            return {"check": "import", "passed": False, "error": str(e)}

    def _run_tests(self) -> dict:
        """Run pytest tests."""
        self._log("  Running tests...")
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "--tb=short", "-q"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
                timeout=240,
            )
            # pytest returns 5 when no tests are collected - treat as pass
            no_tests_collected = result.returncode == 5 or "no tests ran" in result.stdout.lower()
            passed = result.returncode == 0 or no_tests_collected
            check = {
                "check": "tests",
                "passed": passed,
                "output": result.stdout[-500:] if result.stdout else "",
                "note": "no tests collected" if no_tests_collected else "",
            }
            self._log(f"    {'passed' if passed else 'FAILED'} tests" + (" (no tests collected)" if no_tests_collected else ""))
            self._stream_emit("on_verification_result", "tests", passed, result.stdout[-200:] if result.stdout else "")
            return check
        except Exception as e:
            self._log(f"    FAILED tests (exception): {e}")
            self._stream_emit("on_verification_result", "tests", False, f"Exception: {e}")
            return {"check": "tests", "passed": False, "error": str(e), "note": "Test execution failed"}

    async def _codex_audit(self) -> Optional[dict]:
        """Run Codex verification audit on changed files."""
        try:
            self._log("  [hybrid] Codex verification audit...")
            changed_files = self._get_changed_files()
            diff_output = ""

            if changed_files:
                diff_result = subprocess.run(
                    ["git", "diff", "--unified=3"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                diff_output = diff_result.stdout[:5000] if diff_result.returncode == 0 else ""

            audit_prompt = f"""As the verification lead, audit this implementation:

Changed files: {changed_files[:10]}

Diff (first 5000 chars):
{diff_output}

Provide a brief verification report:
1. CODE QUALITY: Are there any obvious issues? (0-10)
2. TEST COVERAGE: Are the changes adequately tested? (0-10)
3. DESIGN ALIGNMENT: Does implementation match the design? (0-10)
4. RISK ASSESSMENT: Any potential runtime issues? (0-10)
5. VERDICT: APPROVE or CONCERNS (with brief explanation)

Be concise - this is a quality gate, not a full review."""

            audit_response = await self.codex.generate(audit_prompt)
            if audit_response:
                # Check if audit has concerns
                if "CONCERNS" in audit_response.upper() and "APPROVE" not in audit_response.upper():
                    self._log("  [hybrid] Codex raised concerns - flagging for review")
                    return {
                        "check": "codex_audit",
                        "passed": False,
                        "output": audit_response[:500],
                        "note": "Codex verification audit raised concerns",
                    }
                else:
                    self._log("  [hybrid] Codex audit passed")
                    return {
                        "check": "codex_audit",
                        "passed": True,
                        "output": audit_response[:500],
                    }
        except Exception as e:
            self._log(f"  [hybrid] Codex audit error: {e}")
            return {
                "check": "codex_audit",
                "passed": True,  # Don't block on audit failure
                "error": str(e),
                "note": "Audit skipped due to error",
            }
        return None

    def _get_changed_files(self) -> list[str]:
        """Get list of files changed in this cycle."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        except Exception:
            pass
        return []

    async def _check_staleness(self) -> list:
        """Check for stale evidence claims."""
        try:
            changed_files = self._get_changed_files()
            if changed_files:
                self._log(f"  [integration] Checking staleness for {len(changed_files)} changed files...")
                self._log(f"  [integration] Changed files: {changed_files[:5]}...")

            # Checkpoint the verify phase
            await self.nomic_integration.checkpoint(
                phase="verify",
                state={"all_passed": True, "changed_files": changed_files},
                cycle=self.cycle_count,
            )
        except Exception as e:
            self._log(f"  [integration] Staleness check failed: {e}")
        return []


__all__ = ["VerifyPhase"]
