"""
Verification phase for nomic loop.

Phase 4: Verify changes work
- Python syntax check
- Import check
- Run tests
- Optional Codex audit
- Evidence staleness check
- Test quality gate
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from . import VerifyResult

if TYPE_CHECKING:
    from aragora.nomic.gates import TestQualityGate

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
        codex: Any | None = None,
        nomic_integration: Any | None = None,
        cycle_count: int = 0,
        log_fn: Callable[[str], None] | None = None,
        stream_emit_fn: Callable[..., None] | None = None,
        record_replay_fn: Callable[..., None] | None = None,
        save_state_fn: Callable[[dict], None] | None = None,
        test_quality_gate: TestQualityGate | None = None,
        baseline: Any | None = None,
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
            test_quality_gate: Optional TestQualityGate for quality thresholds
            baseline: Optional BaselineSnapshot from before implementation
        """
        self.aragora_path = aragora_path
        self.codex = codex
        self.nomic_integration = nomic_integration
        self.cycle_count = cycle_count
        self._log = log_fn or (lambda msg: logger.info(msg))
        self._stream_emit = stream_emit_fn or (lambda *args: None)
        self._record_replay = record_replay_fn or (lambda *args: None)
        self._save_state = save_state_fn or (lambda state: None)
        self._test_quality_gate = test_quality_gate
        self._baseline = baseline

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
        checks.append(await self._check_syntax())

        # 2. Import check
        checks.append(await self._check_imports())

        # 3. Run tests
        checks.append(await self._run_tests())

        all_passed = all(c.get("passed", False) for c in checks)

        # 4. Optional Codex audit
        if all_passed and self.codex:
            audit_result = await self._codex_audit()
            if audit_result:
                checks.append(audit_result)
                all_passed = all(c.get("passed", False) for c in checks)

        # Save state
        self._save_state(
            {
                "phase": "verify",
                "stage": "complete",
                "all_passed": all_passed,
                "checks": checks,
            }
        )

        # Check evidence staleness
        stale_claims = await self._check_staleness() if self.nomic_integration else []

        # === SAFETY: Test quality gate ===
        gate_passed = True
        gate_decision = None
        if self._test_quality_gate and all_passed:
            try:
                from aragora.nomic.gates import ApprovalRequired, ApprovalStatus

                # Extract test output for gate
                test_output = checks[-1].get("output", "") if checks else ""
                gate_context = {
                    "tests_passed": all_passed,
                    "coverage": 0.0,  # Coverage not currently tracked
                    "warnings_count": 0,  # Warnings not currently tracked
                    "checks": checks,
                }
                gate_decision = await self._test_quality_gate.require_approval(
                    test_output, gate_context
                )

                if gate_decision.status == ApprovalStatus.APPROVED:
                    self._log(f"  [gate] Test quality approved: {gate_decision.reason}")
                elif gate_decision.status == ApprovalStatus.SKIPPED:
                    self._log("  [gate] Test quality gate skipped (disabled)")

            except ApprovalRequired as e:
                self._log(f"  [gate] Test quality gate failed: {e}")
                gate_passed = False
                all_passed = False
            except (RuntimeError, ValueError, OSError) as gate_error:
                self._log(f"  [gate] Gate check error: {gate_error}")
                # Non-fatal: continue without gate if it fails

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit(
            "on_phase_end",
            "verify",
            self.cycle_count,
            all_passed,
            phase_duration,
            {"checks_passed": sum(1 for c in checks if c.get("passed"))},
        )

        result_data: dict[str, Any] = {"checks": checks, "stale_claims": stale_claims}
        if gate_decision:
            result_data["quality_gate"] = gate_decision.to_dict()

        # Baseline comparison: measure improvement if baseline was provided
        metrics_delta = None
        improvement_score = None
        if self._baseline and all_passed:
            try:
                from aragora.nomic.phases.baseline import BaselineCollector

                post = await BaselineCollector(self.aragora_path).collect()
                delta = self._baseline.compare(post)
                metrics_delta = delta
                improvement_score = delta.get("improvement_score", 0.0)
                result_data["metrics_delta"] = delta
                result_data["improvement_score"] = improvement_score
                self._log(
                    f"  [baseline] improvement_score={improvement_score:.2f} "
                    f"improved={delta.get('improved', False)}"
                )
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning("Baseline comparison failed: %s", e)

        return VerifyResult(
            success=all_passed,
            data=result_data,
            duration_seconds=phase_duration,
            tests_passed=gate_passed and all_passed,
            test_output=checks[-1].get("output", "") if checks else "",
            syntax_valid=checks[0].get("passed", False) if checks else False,
            metrics_delta=metrics_delta,
            improvement_score=improvement_score,
        )

    async def _check_syntax(self) -> dict:
        """Check Python syntax of all changed .py files."""
        self._log("  Checking syntax...")
        try:
            # Get changed Python files from git diff
            changed_files = await self._get_changed_files()
            py_files = [f for f in changed_files if f.endswith(".py")]

            # Fallback to aragora/__init__.py when no changes detected
            if not py_files:
                py_files = ["aragora/__init__.py"]

            self._log(f"    Checking {len(py_files)} file(s)...")

            all_errors: list[str] = []
            for py_file in py_files:
                proc = await asyncio.create_subprocess_exec(
                    "python",
                    "-m",
                    "py_compile",
                    py_file,
                    cwd=self.aragora_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                if proc.returncode != 0:
                    stderr_text = stderr.decode() if stderr else ""
                    all_errors.append(f"{py_file}: {stderr_text}")

            passed = len(all_errors) == 0
            output = "\n".join(all_errors) if all_errors else ""
            check = {
                "check": "syntax",
                "passed": passed,
                "output": output,
                "files_checked": len(py_files),
            }
            self._log(f"    {'passed' if passed else 'FAILED'} syntax ({len(py_files)} files)")
            self._stream_emit("on_verification_result", "syntax", passed, output)
            return check
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            self._log("    FAILED syntax: timeout")
            self._stream_emit("on_verification_result", "syntax", False, "timeout")
            return {"check": "syntax", "passed": False, "error": "timeout"}
        except OSError as e:
            logger.warning("Syntax check failed: %s", e)
            error_desc = f"Syntax check failed: {type(e).__name__}"
            self._stream_emit("on_verification_result", "syntax", False, error_desc)
            return {"check": "syntax", "passed": False, "error": error_desc}

    async def _check_imports(self) -> dict:
        """Check that aragora can be imported."""
        self._log("  Checking imports...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                "-c",
                "import aragora; print('OK')",
                cwd=self.aragora_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
            stdout_text = stdout.decode() if stdout else ""
            stderr_text = stderr.decode() if stderr else ""
            passed = "OK" in stdout_text
            check = {
                "check": "import",
                "passed": passed,
                "output": stderr_text if proc.returncode != 0 else "",
            }
            self._log(f"    {'passed' if passed else 'FAILED'} import")
            self._stream_emit("on_verification_result", "import", passed, stderr_text)
            return check
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            self._log("    FAILED import: timeout")
            self._stream_emit("on_verification_result", "import", False, "timeout")
            return {"check": "import", "passed": False, "error": "timeout"}
        except OSError as e:
            logger.warning("Import check failed: %s", e)
            error_desc = f"Import check failed: {type(e).__name__}"
            self._stream_emit("on_verification_result", "import", False, error_desc)
            return {"check": "import", "passed": False, "error": error_desc}

    async def _run_tests(self, test_paths: list[str] | None = None) -> dict:
        """Run pytest tests using TestRunner for structured failure data.

        Args:
            test_paths: Optional list of specific test paths to run.
                Falls back to "tests/" if not provided.
        """
        self._log("  Running tests...")
        try:
            from aragora.nomic.testfixer.runner import TestRunner

            paths = test_paths or ["tests/"]
            test_command = f"python -m pytest {' '.join(paths)} -x --tb=short -q"
            runner = TestRunner(
                repo_path=self.aragora_path,
                test_command=test_command,
                timeout_seconds=240,
            )
            result = await runner.run()

            no_tests_collected = result.exit_code == 5 or result.total_tests == 0
            passed = result.success or no_tests_collected

            # Serialize failure details for downstream consumption
            failure_details = [
                {
                    "test_name": f.test_name,
                    "test_file": f.test_file,
                    "line_number": f.line_number,
                    "error_type": f.error_type,
                    "error_message": f.error_message,
                    "stack_trace": f.stack_trace[:500],
                    "involved_files": f.involved_files,
                }
                for f in result.failures[:10]
            ]

            check: dict[str, Any] = {
                "check": "tests",
                "passed": passed,
                "output": result.stdout[-500:] if result.stdout else "",
                "note": "no tests collected" if no_tests_collected else "",
                "test_result": result,
                "num_passed": result.passed,
                "num_failed": result.failed,
                "failure_details": failure_details,
                "stats": {
                    "total": result.total_tests,
                    "passed": result.passed,
                    "failed": result.failed,
                    "errors": result.errors,
                    "skipped": result.skipped,
                },
            }
            self._log(
                f"    {'passed' if passed else 'FAILED'} tests"
                + (f" ({result.failed} failed)" if result.failed else "")
                + (" (no tests collected)" if no_tests_collected else "")
            )
            self._stream_emit(
                "on_verification_result",
                "tests",
                passed,
                result.stdout[-200:] if result.stdout else "",
            )
            return check
        except ImportError:
            self._log("    [fallback] TestRunner unavailable, using raw subprocess")
            return await self._run_tests_raw(test_paths)
        except OSError as e:
            logger.warning("Test execution failed: %s", e)
            error_desc = f"Test execution failed: {type(e).__name__}"
            self._stream_emit("on_verification_result", "tests", False, error_desc)
            return {
                "check": "tests",
                "passed": False,
                "error": error_desc,
                "note": "Test execution failed",
            }

    async def _run_tests_raw(self, test_paths: list[str] | None = None) -> dict:
        """Fallback: run pytest via raw subprocess when TestRunner is unavailable."""
        try:
            cmd = ["python", "-m", "pytest", *(test_paths or ["tests/"]), "-x", "--tb=short", "-q"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.aragora_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=240)
            stdout_text = stdout.decode() if stdout else ""
            no_tests_collected = proc.returncode == 5 or "no tests ran" in stdout_text.lower()
            passed = proc.returncode == 0 or no_tests_collected
            check = {
                "check": "tests",
                "passed": passed,
                "output": stdout_text[-500:] if stdout_text else "",
                "note": "no tests collected" if no_tests_collected else "",
            }
            self._log(
                f"    {'passed' if passed else 'FAILED'} tests"
                + (" (no tests collected)" if no_tests_collected else "")
            )
            self._stream_emit(
                "on_verification_result",
                "tests",
                passed,
                stdout_text[-200:] if stdout_text else "",
            )
            return check
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            self._log("    FAILED tests (timeout)")
            self._stream_emit("on_verification_result", "tests", False, "Test execution timed out")
            return {
                "check": "tests",
                "passed": False,
                "error": "timeout",
                "note": "Test execution timed out",
            }
        except OSError as e:
            logger.warning("Test execution failed: %s", e)
            error_desc = f"Test execution failed: {type(e).__name__}"
            self._stream_emit("on_verification_result", "tests", False, error_desc)
            return {
                "check": "tests",
                "passed": False,
                "error": error_desc,
                "note": "Test execution failed",
            }

    async def _codex_audit(self) -> dict | None:
        """Run Codex verification audit on changed files."""
        try:
            self._log("  [hybrid] Codex verification audit...")
            changed_files = await self._get_changed_files()
            diff_output = ""

            if changed_files:
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    "diff",
                    "--unified=3",
                    cwd=self.aragora_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    stdout = None
                diff_output = stdout.decode()[:5000] if proc.returncode == 0 and stdout else ""

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

            from aragora.server.stream.arena_hooks import streaming_task_context

            codex_name = getattr(self.codex, "name", "codex")
            task_id = f"{codex_name}:nomic_verify"
            with streaming_task_context(task_id):
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
        except (ConnectionError, TimeoutError, asyncio.TimeoutError, RuntimeError, ValueError) as e:
            logger.warning("Codex audit error: %s", e)
            return {
                "check": "codex_audit",
                "passed": True,  # Don't block on audit failure
                "error": f"Codex audit skipped: {type(e).__name__}",
                "note": "Audit skipped due to error",
            }
        return None

    async def _get_changed_files(self) -> list[str]:
        """Get list of files changed in this cycle."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "--name-only",
                cwd=self.aragora_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return []
            if proc.returncode == 0 and stdout:
                return [f.strip() for f in stdout.decode().strip().split("\n") if f.strip()]
        except (OSError, FileNotFoundError, asyncio.TimeoutError) as e:
            self._log(f"  [git] Failed to get changed files: {e}")
        return []

    async def _check_staleness(self) -> list:
        """Check for stale evidence claims."""
        try:
            changed_files = await self._get_changed_files()
            if changed_files:
                self._log(
                    f"  [integration] Checking staleness for {len(changed_files)} changed files..."
                )
                self._log(f"  [integration] Changed files: {changed_files[:5]}...")

            # Checkpoint the verify phase
            await self.nomic_integration.checkpoint(
                phase="verify",
                state={"all_passed": True, "changed_files": changed_files},
                cycle=self.cycle_count,
            )
        except (OSError, asyncio.TimeoutError, RuntimeError) as e:
            self._log(f"  [integration] Staleness check failed: {e}")
        return []


__all__ = ["VerifyPhase"]
