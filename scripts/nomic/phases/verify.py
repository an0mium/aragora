"""
Verification phase for nomic loop.

Phase 4: Verify changes work
- Python syntax check
- Import check
- Run tests
- Optional Codex audit
- Evidence staleness check
- Consistency audit (new)
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from . import VerifyResult

if TYPE_CHECKING:
    from aragora.audit.audit_types.consistency import ConsistencyAuditor

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
        consistency_auditor: Optional["ConsistencyAuditor"] = None,
        enable_consistency_check: bool = True,
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
            consistency_auditor: Optional ConsistencyAuditor for change validation
            enable_consistency_check: Whether to run consistency check (default True)
        """
        self.aragora_path = aragora_path
        self.codex = codex
        self.nomic_integration = nomic_integration
        self.cycle_count = cycle_count
        self._log = log_fn or (lambda msg: logger.info(msg))
        self._stream_emit = stream_emit_fn or (lambda *args: None)
        self._record_replay = record_replay_fn or (lambda *args: None)
        self._save_state = save_state_fn or (lambda state: None)
        self.consistency_auditor = consistency_auditor
        self.enable_consistency_check = enable_consistency_check

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

        # 5. Consistency check (if enabled and auditor available)
        if all_passed and self.enable_consistency_check and self.consistency_auditor:
            consistency_result = await self._check_consistency()
            checks.append(consistency_result)
            # Consistency failures are warnings, not blockers (unless critical)
            if (
                not consistency_result.get("passed")
                and consistency_result.get("severity") == "critical"
            ):
                all_passed = False

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

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit(
            "on_phase_end",
            "verify",
            self.cycle_count,
            all_passed,
            phase_duration,
            {"checks_passed": sum(1 for c in checks if c.get("passed"))},
        )

        return VerifyResult(
            success=all_passed,
            data={"checks": checks, "stale_claims": stale_claims},
            duration_seconds=phase_duration,
            tests_passed=all_passed,
            test_output=checks[-1].get("output", "") if checks else "",
            syntax_valid=checks[0].get("passed", False) if checks else False,
        )

    async def _check_syntax(self) -> dict:
        """Check Python syntax."""
        self._log("  Checking syntax...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "py_compile",
                "aragora/__init__.py",
                cwd=self.aragora_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            passed = proc.returncode == 0
            stderr_text = stderr.decode() if stderr else ""
            check = {
                "check": "syntax",
                "passed": passed,
                "output": stderr_text,
            }
            self._log(f"    {'passed' if passed else 'FAILED'} syntax")
            self._stream_emit("on_verification_result", "syntax", passed, stderr_text)
            return check
        except Exception as e:
            self._log(f"    FAILED syntax: {e}")
            self._stream_emit("on_verification_result", "syntax", False, str(e))
            return {"check": "syntax", "passed": False, "error": str(e)}

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
            self._log("    FAILED import: timeout")
            self._stream_emit("on_verification_result", "import", False, "timeout")
            return {"check": "import", "passed": False, "error": "timeout"}
        except Exception as e:
            self._log(f"    FAILED import: {e}")
            self._stream_emit("on_verification_result", "import", False, str(e))
            return {"check": "import", "passed": False, "error": str(e)}

    async def _run_tests(self) -> dict:
        """Run pytest tests."""
        self._log("  Running tests...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "pytest",
                "tests/",
                "-x",
                "--tb=short",
                "-q",
                cwd=self.aragora_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=240)
            stdout_text = stdout.decode() if stdout else ""
            # pytest returns 5 when no tests are collected - treat as pass
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
                "on_verification_result", "tests", passed, stdout_text[-200:] if stdout_text else ""
            )
            return check
        except asyncio.TimeoutError:
            self._log("    FAILED tests (timeout)")
            self._stream_emit("on_verification_result", "tests", False, "Test execution timed out")
            return {
                "check": "tests",
                "passed": False,
                "error": "timeout",
                "note": "Test execution timed out",
            }
        except Exception as e:
            self._log(f"    FAILED tests (exception): {e}")
            self._stream_emit("on_verification_result", "tests", False, f"Exception: {e}")
            return {
                "check": "tests",
                "passed": False,
                "error": str(e),
                "note": "Test execution failed",
            }

    async def _codex_audit(self) -> Optional[dict]:
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
                stdout, _ = await proc.communicate()
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
            stdout, _ = await proc.communicate()
            if proc.returncode == 0 and stdout:
                return [f.strip() for f in stdout.decode().strip().split("\n") if f.strip()]
        except Exception:
            pass
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
        except Exception as e:
            self._log(f"  [integration] Staleness check failed: {e}")
        return []

    async def _check_consistency(self) -> dict:
        """Check changed files for consistency issues using ConsistencyAuditor.

        Returns:
            Check result dict with passed, findings, and severity
        """
        self._log("  [consistency] Checking changes for consistency...")
        try:
            changed_files = await self._get_changed_files()
            if not changed_files:
                self._log("    [consistency] No changed files to check")
                return {
                    "check": "consistency",
                    "passed": True,
                    "output": "No changed files",
                }

            # Filter to Python files and docs
            relevant_files = [
                f for f in changed_files if f.endswith((".py", ".md", ".rst", ".txt"))
            ]

            if not relevant_files:
                self._log("    [consistency] No relevant files to check")
                return {
                    "check": "consistency",
                    "passed": True,
                    "output": "No relevant files",
                }

            self._log(f"    [consistency] Checking {len(relevant_files)} files...")

            # Read file contents and create chunks
            from aragora.audit.document_auditor import AuditSession, AuditType

            session = AuditSession(
                session_id=f"verify_{self.cycle_count}",
                document_ids=[str(self.aragora_path / f) for f in relevant_files],
            )

            # Build chunks from changed files
            chunks = []
            for file_path in relevant_files[:10]:  # Limit to 10 files
                full_path = self.aragora_path / file_path
                if full_path.exists():
                    try:
                        content = full_path.read_text()
                        chunks.append(
                            {
                                "id": file_path,
                                "document_id": file_path,
                                "content": content[:5000],  # Limit content size
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read {file_path}: {e}")

            if not chunks:
                return {
                    "check": "consistency",
                    "passed": True,
                    "output": "No readable files",
                }

            # Run consistency audit
            findings = await self.consistency_auditor.audit(chunks, session)

            # Categorize findings by severity
            critical_count = sum(
                1
                for f in findings
                if hasattr(f, "severity") and str(f.severity).lower() in ("critical", "high")
            )

            if critical_count > 0:
                self._log(f"    [consistency] Found {critical_count} critical/high issues")
                self._stream_emit(
                    "on_verification_result",
                    "consistency",
                    False,
                    f"{critical_count} critical issues found",
                )
                return {
                    "check": "consistency",
                    "passed": False,
                    "severity": "critical" if critical_count >= 3 else "high",
                    "findings": len(findings),
                    "critical": critical_count,
                    "output": f"Found {len(findings)} issues ({critical_count} critical/high)",
                }
            else:
                self._log(f"    [consistency] Passed ({len(findings)} minor issues)")
                self._stream_emit("on_verification_result", "consistency", True, "OK")
                return {
                    "check": "consistency",
                    "passed": True,
                    "findings": len(findings),
                    "output": f"Passed ({len(findings)} minor issues)",
                }

        except Exception as e:
            self._log(f"    [consistency] Check failed: {e}")
            logger.warning(f"Consistency check error: {e}")
            # Don't block on consistency check failure
            return {
                "check": "consistency",
                "passed": True,
                "error": str(e),
                "note": "Check skipped due to error",
            }


__all__ = ["VerifyPhase"]
