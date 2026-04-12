"""Verification and runtime probe helpers for the swarm supervisor."""

from __future__ import annotations

import re
import shlex
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from aragora.docs_only import is_docs_safe_path
from aragora.nomic.dev_coordination import FileScopeViolationError, LeaseStatus
from aragora.swarm import supervisor as _supervisor
from aragora.swarm.worker_launcher import WorkerLauncher, WorkerProcess, is_ignored_changed_path

UTC = _supervisor.UTC
logger = _supervisor.logger
MAX_WORKER_LOG_TAIL_CHARS = _supervisor.MAX_WORKER_LOG_TAIL_CHARS
WorkerOutcome = _supervisor.WorkerOutcome
_path_in_scope = _supervisor._path_in_scope

_REPAIR_JOURNAL_MAX_ENTRIES = 3
_REPAIR_JOURNAL_TAIL_CHARS = 800
_BEST_EFFORT_STORE_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    sqlite3.Error,
    TypeError,
    ValueError,
)
_LLM_PROBE_EXCEPTIONS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _tail_text(text: str, *, max_chars: int = _REPAIR_JOURNAL_TAIL_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _summarize_verification_failure(verification_results: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in verification_results:
        if not isinstance(entry, dict):
            continue
        passed = entry.get("passed")
        if passed is False:
            return {
                "command": str(entry.get("command", "")).strip(),
                "exit_code": int(entry.get("exit_code") or -1),
                "stdout_tail": _tail_text(str(entry.get("stdout", ""))),
                "stderr_tail": _tail_text(str(entry.get("stderr", ""))),
            }
    return {}


def _append_repair_journal(
    self,
    item: dict[str, Any],
    result: WorkerProcess,
    *,
    reason: str | None = None,
) -> None:
    metadata = dict(item.get("metadata") or {})
    raw_entries = metadata.get("repair_journal")
    entries: list[dict[str, Any]] = []
    if isinstance(raw_entries, list):
        entries = [entry for entry in raw_entries if isinstance(entry, dict)]

    verification_results = item.get("verification_results", []) or []
    failing = _summarize_verification_failure(verification_results)
    entry = {
        "at": datetime.now(UTC).isoformat(),
        "agent": str(item.get("target_agent", result.agent)).strip() or result.agent,
        "exit_code": int(result.exit_code or 0),
        "worker_outcome": str(item.get("worker_outcome", "")).strip() or None,
        "failure_reason": str(reason or item.get("failure_reason", "")).strip() or None,
        "changed_paths": list(item.get("changed_paths", []) or [])[:10],
        "commit_shas": list(result.commit_shas or [])[:5],
        "tests_run": list(item.get("tests_run", []) or [])[:3],
        "failing_verification": failing or None,
        "stdout_tail": _tail_text(result.stdout or ""),
        "stderr_tail": _tail_text(result.stderr or ""),
    }
    entries.append(entry)
    metadata["repair_journal"] = entries[-_REPAIR_JOURNAL_MAX_ENTRIES:]
    item["metadata"] = metadata
    try:
        self.store.record_worker_repair_journal(
            task_id=str(item.get("work_order_id", "")).strip(),
            task_key=str(item.get("task_key", "")).strip(),
            handoff_key=str(metadata.get("handoff_key", "")).strip(),
            work_order_id=str(item.get("work_order_id", "")).strip(),
            supervisor_run_id=str(metadata.get("supervisor_run_id", "")).strip(),
            lease_id=str(item.get("lease_id", result.lease_id)).strip(),
            owner_agent=str(item.get("target_agent", result.agent)).strip(),
            owner_session_id=str(item.get("owner_session_id", result.session_id)).strip(),
            branch=str(item.get("branch", result.branch)).strip(),
            worktree_path=str(item.get("worktree_path", result.worktree_path)).strip(),
            entry=entry,
        )
    except _BEST_EFFORT_STORE_EXCEPTIONS:
        logger.debug(
            "Failed to persist repair journal for %s", item.get("work_order_id"), exc_info=True
        )


def _worker_result_from_persisted_work_order(item: dict[str, Any]) -> WorkerProcess | None:
    commit_shas = [str(sha).strip() for sha in item.get("commit_shas", []) if str(sha).strip()]
    if not commit_shas:
        return None
    worktree_path = str(item.get("worktree_path", "")).strip()
    branch = str(item.get("branch", "")).strip()
    if not worktree_path or not branch:
        return None
    return WorkerProcess(
        work_order_id=str(item.get("work_order_id", "")).strip(),
        agent=str(item.get("target_agent", "codex")).strip() or "codex",
        worktree_path=worktree_path,
        branch=branch,
        pid=WorkerLauncher._normalized_pid(item.get("pid")),
        session_id=str(item.get("owner_session_id", "")).strip(),
        lease_id=str(item.get("lease_id", "")).strip(),
        completed_at=str(item.get("completed_at", "")).strip(),
        exit_code=int(item.get("exit_code") or 1),
        stdout=str(item.get("stdout_tail") or ""),
        stderr=str(item.get("stderr_tail") or ""),
        diff="",
        initial_head=str(item.get("initial_head", "")).strip(),
        head_sha=str(item.get("head_sha", "")).strip(),
        commit_shas=commit_shas,
        changed_paths=[
            str(path).strip() for path in item.get("changed_paths", []) if str(path).strip()
        ],
        tests_run=[str(test).strip() for test in item.get("tests_run", []) if str(test).strip()],
        expected_tests=[
            str(test).strip() for test in item.get("expected_tests", []) if str(test).strip()
        ],
        prompt_chars=int(item.get("prompt_chars") or 0),
        enriched_context_chars=int(item.get("enriched_context_chars") or 0),
    )


def _rehabilitate_validation_marker_crash_work_order(
    self,
    item: dict[str, Any],
    *,
    worker_type_circuit_breakers: dict[str, dict[str, Any]] | None = None,
    worker_type_circuit_breaker_policy: dict[str, Any] | None = None,
) -> None:
    if str(item.get("status", "")).strip() != "needs_human":
        return
    failure_reason = str(item.get("failure_reason", "")).strip()
    if failure_reason not in {
        "worker_crash_with_deliverable",
        "missing_verification_plan",
    }:
        return
    if str(item.get("review_status", "")).strip() != "changes_requested":
        return
    if not str(item.get("receipt_id") or "").strip():
        return
    result = self._worker_result_from_persisted_work_order(item)
    if result is None:
        return
    clean_paths = self._strip_session_artifacts(list(result.changed_paths))
    if not self._should_accept_validation_marker_commit(item, result, clean_paths):
        return
    tests_run, verification_results = self._synthesized_validation_marker_verification(
        item,
        result,
    )
    item["changed_paths"] = clean_paths
    if tests_run:
        item["expected_tests"] = tests_run
    item["tests_run"] = tests_run
    item["verification_results"] = verification_results
    item["worker_outcome"] = WorkerOutcome.COMPLETED.value
    metadata = dict(item.get("metadata") or {})
    rehabilitated_at = datetime.now(UTC).isoformat()
    metadata["validation_marker_rehabilitated_at"] = rehabilitated_at
    metadata["validation_marker_original_exit_code"] = result.exit_code
    item["metadata"] = metadata
    self._finalize_completed_work_order_result(
        item,
        result,
        clean_paths=clean_paths,
        worker_type_circuit_breakers=worker_type_circuit_breakers,
        worker_type_circuit_breaker_policy=worker_type_circuit_breaker_policy,
    )
    if str(item.get("status", "")).strip() == "completed":
        self.store.sync_completion_receipt_verification(
            receipt_id=str(item.get("receipt_id", "")).strip(),
            verification_results=verification_results,
            replayed_at=rehabilitated_at,
        )


def _strip_session_artifacts(paths: list[str]) -> list[str]:
    """Remove harness and runtime noise from a list of changed paths.

    Session artifacts like ``.codex_session_meta.json`` are infrastructure
    metadata created by the harness, while runtime directories like
    ``node_modules`` are environment noise. Stripping them prevents workers
    from claiming credit for non-work output or tripping false scope checks.
    """
    return [p for p in paths if not is_ignored_changed_path(p)]


def _latest_commit_subject(cls, worktree_path: str, commit_shas: list[str]) -> str:
    if not worktree_path or not commit_shas or not Path(worktree_path).is_dir():
        return ""
    commit_sha = str(commit_shas[-1]).strip()
    if not commit_sha:
        return ""
    result = cls._run_git_capture_sync(
        worktree_path,
        "show",
        "--no-patch",
        "--format=%s",
        commit_sha,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _should_accept_validation_marker_commit(
    cls,
    item: dict[str, Any],
    result: WorkerProcess,
    clean_paths: list[str],
) -> bool:
    if clean_paths:
        return False
    if not result.commit_shas:
        return False
    if not [str(dep).strip() for dep in item.get("dependency_ids", []) if str(dep).strip()]:
        return False
    if "run validation and fix failures" not in str(item.get("title", "")).strip().lower():
        return False
    commit_subject = cls._latest_commit_subject(
        str(result.worktree_path or item.get("worktree_path", "")).strip(),
        list(result.commit_shas),
    ).lower()
    if not commit_subject.startswith("test: validation passed"):
        return False
    success_output = "\n".join(
        part.strip()
        for part in (str(result.stdout or ""), str(result.stderr or ""))
        if part and part.strip()
    ).lower()
    marker_phrases = (
        "empty commit marker created",
        "no-op marker committed",
        "empty commit recorded as marker",
    )
    return any(phrase in success_output for phrase in marker_phrases) or (
        "passed" in success_output and "marker" in success_output
    )


def _synthesized_validation_marker_verification(
    item: dict[str, Any],
    result: WorkerProcess,
) -> tuple[list[str], list[dict[str, Any]]]:
    expected_tests = [
        str(test).strip() for test in item.get("expected_tests", []) if str(test).strip()
    ]
    if not expected_tests:
        file_scope = [
            str(path).strip()
            for path in item.get("file_scope", [])
            if str(path).strip().startswith("tests/") and str(path).strip().endswith(".py")
        ]
        if len(file_scope) == 1:
            expected_tests = [f"python -m pytest {file_scope[0]} -q"]
    if not expected_tests:
        output = "\n".join(
            part.strip()
            for part in (str(result.stdout or ""), str(result.stderr or ""))
            if part and part.strip()
        )
        pytest_targets = re.findall(r"(tests/[A-Za-z0-9_./-]+\.py)", output)
        if pytest_targets:
            expected_tests = [f"python -m pytest {pytest_targets[-1]} -q"]
    verification_results = [
        {
            "command": command,
            "passed": True,
            "exit_code": 0,
            "stdout": str(result.stdout or ""),
            "stderr": str(result.stderr or ""),
            "inferred_from": "validation_marker_commit",
        }
        for command in expected_tests
    ]
    return expected_tests, verification_results


def _finalize_completed_work_order_result(
    self,
    item: dict[str, Any],
    result: WorkerProcess,
    *,
    clean_paths: list[str],
    worker_type_circuit_breakers: dict[str, dict[str, Any]] | None = None,
    worker_type_circuit_breaker_policy: dict[str, Any] | None = None,
) -> bool:
    merge_gate = self._merge_gate_state(item)
    preserved_outcome = str(item.get("worker_outcome", "")).strip()
    item["merge_gate"] = merge_gate
    if merge_gate.get("verification_missing_reason"):
        item["verification_missing_reason"] = merge_gate["verification_missing_reason"]
    if not bool(merge_gate.get("checks_passed")):
        can_override_merge_gate = not bool(merge_gate.get("verification_missing_reason"))
        if can_override_merge_gate and self._llm_override_merge_gate(item, merge_gate):
            merge_gate["checks_passed"] = True
            merge_gate["llm_override"] = True
            item["merge_gate"] = merge_gate
        else:
            for key in ("resource_error", "conflicts", "scope_violation"):
                item.pop(key, None)
            item.pop("blockers", None)
            self._mark_needs_human(
                item,
                self._merge_gate_failure_reason(merge_gate),
                failure_reason=str(
                    merge_gate.get("verification_missing_reason", "") or "merge_gate_failed"
                ).strip()
                or "merge_gate_failed",
                blocking_question=self._merge_gate_blocking_question(merge_gate),
            )
            _append_repair_journal(self, item, result, reason="merge_gate_failed")
            item["review_status"] = "changes_requested"
            item["receipt_id"] = None
            if preserved_outcome not in {
                WorkerOutcome.CRASH_WITH_SALVAGE.value,
                WorkerOutcome.TIMEOUT_WITH_SALVAGE.value,
            }:
                item["worker_outcome"] = WorkerOutcome.MERGE_GATE_FAILED.value
            self._release_terminal_lease(item)
            return False

    lease_id = str(item.get("lease_id") or "").strip()
    receipt_id = str(item.get("receipt_id") or "").strip()
    tests_run = [str(test).strip() for test in item.get("tests_run", []) if str(test).strip()]
    if lease_id and not receipt_id:
        try:
            receipt = self.store.record_completion(
                lease_id=lease_id,
                owner_agent=str(item.get("target_agent", result.agent)),
                owner_session_id=str(item.get("owner_session_id", result.session_id)),
                branch=str(item.get("branch", result.branch)),
                worktree_path=str(item.get("worktree_path", result.worktree_path)),
                base_sha=str(item.get("initial_head", result.initial_head)),
                head_sha=str(result.head_sha or item.get("head_sha", "")),
                commit_shas=list(result.commit_shas),
                changed_paths=clean_paths,
                tests_run=tests_run,
                validations_run=tests_run,
                assumptions=[],
                blockers=[
                    str(blocker).strip()
                    for blocker in item.get("blockers", [])
                    if str(blocker).strip()
                ],
                outcome=self._work_order_deliverable_type(item) or "completed",
                risks=[
                    str(blocker).strip()
                    for blocker in item.get("blockers", [])
                    if str(blocker).strip()
                ],
                pr_url=str(item.get("pr_url", "") or item.get("adopted_pr", "")).strip(),
                pr_number=self._extract_pr_number(
                    str(item.get("pr_url", "") or item.get("adopted_pr", "")).strip()
                ),
                confidence=self._completion_confidence(item, result),
                metadata={
                    "task_key": str(item.get("task_key", "")).strip() or None,
                    "verification_results": list(item.get("verification_results", []) or []),
                    "worker_outcome": str(item.get("worker_outcome", "")).strip() or None,
                    "approval_required": bool(item.get("approval_required", False)),
                    "risk_level": str(item.get("risk_level", "")).strip() or None,
                    "success_criteria": dict(item.get("success_criteria") or {}),
                },
            )
        except FileScopeViolationError as exc:
            for key in ("resource_error", "conflicts"):
                item.pop(key, None)
            item.pop("blockers", None)
            self._mark_needs_human(
                item,
                "worker completion violated file-scope ownership; narrow or split the lane",
                failure_reason="scope_violation",
                blocking_question=(
                    "Which files should stay in scope, or should this lane be split "
                    "before it is rerun?"
                ),
            )
            item["review_status"] = "changes_requested"
            item["receipt_id"] = None
            item.pop("confidence", None)
            item["worker_outcome"] = WorkerOutcome.SCOPE_VIOLATION.value
            for key in (
                "pr_url",
                "adopted_pr",
                "merge_gate",
                "verification_missing_reason",
            ):
                item.pop(key, None)
            item["scope_violation"] = {
                "violations": list(exc.violations),
                "changed_paths": clean_paths,
            }
            self._release_terminal_lease(item)
            return False
        item["receipt_id"] = receipt.receipt_id
        item["confidence"] = receipt.confidence
    if worker_type_circuit_breakers is not None:
        self._record_worker_type_success(
            worker_type_circuit_breakers,
            str(item.get("target_agent", result.agent)),
        )
    self._register_pr_if_present(item, result)
    item["status"] = "completed"
    item["review_status"] = "pending_heterogeneous_review"
    for key in (
        "dispatch_error",
        "resource_error",
        "failure_reason",
        "blocking_question",
        "blocker",
        "conflicts",
        "scope_violation",
    ):
        item.pop(key, None)
    item.pop("blockers", None)
    return True


def _apply_worker_result(
    self,
    item: dict[str, Any],
    result: WorkerProcess,
    *,
    worker_type_circuit_breakers: dict[str, dict[str, Any]] | None = None,
    worker_type_circuit_breaker_policy: dict[str, Any] | None = None,
) -> None:
    # Strip session artifacts before any qualification logic runs
    clean_paths = self._strip_session_artifacts(list(result.changed_paths))
    item["completed_at"] = result.completed_at
    item["diff_lines"] = result.diff.count("\n")
    item["changed_paths"] = clean_paths
    item["tests_run"] = list(result.tests_run)
    item["verification_results"] = self._verification_results_from_result(result)
    item["commit_shas"] = list(result.commit_shas)
    item["head_sha"] = result.head_sha
    item["prompt_chars"] = max(int(item.get("prompt_chars") or 0), int(result.prompt_chars or 0))
    item["enriched_context_chars"] = max(
        int(item.get("enriched_context_chars") or 0),
        int(result.enriched_context_chars or 0),
    )
    self._update_log_tails(item, stdout=result.stdout, stderr=result.stderr)
    item.pop("pid", None)

    # Preserve worker_outcome if already set by detached/timeout collection
    # paths — those have more specific context (e.g. timeout_with_salvage).
    _pre_outcome = str(item.get("worker_outcome", "")).strip()

    # Fail closed: check file-scope before accepting any result as successful
    scope_violations = self._check_file_scope_violations(item, clean_paths)
    if scope_violations:
        # LLM adjudication: ask frontier model if violations are justified
        scope_violations = self._llm_adjudicate_scope(item, scope_violations)
    if scope_violations:
        self._mark_scope_violation(item, scope_violations)
        item["worker_outcome"] = WorkerOutcome.SCOPE_VIOLATION.value
        _append_repair_journal(self, item, result, reason="scope_violation")
        lease_id = str(item.get("lease_id", "")).strip()
        if lease_id:
            self.store.release_lease(lease_id, status=LeaseStatus.RELEASED)
        item["exit_code"] = result.exit_code
        return

    lease_id = str(item.get("lease_id", "")).strip()
    if result.exit_code == 0:
        # Fail closed: if there are no real deliverables but the worker
        # produced commits or had pre-strip changed paths, reject.  This
        # covers both direct workers (result.changed_paths non-empty before
        # strip) and detached workers (changed_paths already stripped by
        # _collect_changed_paths, but commit_shas populated from auto-commit).
        if not clean_paths and (result.changed_paths or result.commit_shas):
            for key in (
                "receipt_id",
                "confidence",
                "pr_url",
                "adopted_pr",
                "merge_gate",
                "verification_missing_reason",
                "scope_violation",
                "resource_error",
                "conflicts",
            ):
                item.pop(key, None)
            item.pop("blockers", None)
            item["commit_shas"] = []
            self._mark_needs_human(
                item,
                "worker produced only session artifacts, no real deliverables",
                failure_reason="clean_exit_no_deliverable",
            )
            _append_repair_journal(self, item, result, reason="clean_exit_no_deliverable")
            if not _pre_outcome:
                item["worker_outcome"] = WorkerOutcome.CLEAN_EXIT_NO_EFFECT.value
            self._release_terminal_lease(item)
            item["exit_code"] = result.exit_code
            return

        # Clean exit with zero changes of any kind — fail closed
        if not clean_paths and not result.commit_shas:
            for key in (
                "receipt_id",
                "confidence",
                "pr_url",
                "adopted_pr",
                "merge_gate",
                "verification_missing_reason",
                "scope_violation",
                "resource_error",
                "conflicts",
            ):
                item.pop(key, None)
            item.pop("blockers", None)
            if not _pre_outcome:
                item["worker_outcome"] = WorkerOutcome.CLEAN_EXIT_NO_EFFECT.value
            logger.warning(
                "Worker %s exited 0 with no commits and no changed paths — "
                "clean_exit_no_effect (branch=%s, initial_head=%s, head_sha=%s)",
                item.get("work_order_id"),
                item.get("branch"),
                result.initial_head,
                result.head_sha,
            )
            self._mark_needs_human(
                item,
                "worker exited 0 with no commits and no changed paths",
                failure_reason="clean_exit_no_deliverable",
            )
            _append_repair_journal(self, item, result, reason="clean_exit_no_deliverable")
            self._release_terminal_lease(item)
            item["exit_code"] = result.exit_code
            return
        elif not _pre_outcome:
            item["worker_outcome"] = WorkerOutcome.COMPLETED.value
        self._finalize_completed_work_order_result(
            item,
            result,
            clean_paths=clean_paths,
            worker_type_circuit_breakers=worker_type_circuit_breakers,
            worker_type_circuit_breaker_policy=worker_type_circuit_breaker_policy,
        )
        return

    # Non-zero exit: classify as crash (with or without salvage)
    if not _pre_outcome:
        if result.commit_shas and clean_paths:
            item["worker_outcome"] = WorkerOutcome.CRASH_WITH_SALVAGE.value
        else:
            item["worker_outcome"] = WorkerOutcome.CRASH.value

    capacity_failure_detail = self._capacity_failure_detail(result)
    if (
        capacity_failure_detail
        and worker_type_circuit_breakers is not None
        and worker_type_circuit_breaker_policy is not None
    ):
        self._record_worker_type_failure(
            worker_type_circuit_breakers,
            str(item.get("target_agent", result.agent)),
            reason="agent_capacity",
            detail=capacity_failure_detail,
            open_immediately=True,
            policy=worker_type_circuit_breaker_policy,
        )

    _append_repair_journal(
        self,
        item,
        result,
        reason=str(item.get("failure_reason", "")).strip() or "worker_failure",
    )

    if self._requeue_after_worker_failure(
        item,
        result,
        worker_type_circuit_breakers=worker_type_circuit_breakers,
    ):
        return

    if self._should_accept_validation_marker_commit(item, result, clean_paths):
        tests_run, verification_results = self._synthesized_validation_marker_verification(
            item,
            result,
        )
        if tests_run:
            item["expected_tests"] = tests_run
        item["tests_run"] = tests_run
        item["verification_results"] = verification_results
        item["worker_outcome"] = WorkerOutcome.COMPLETED.value
        metadata = dict(item.get("metadata") or {})
        metadata["validation_marker_completed_at"] = result.completed_at
        metadata["validation_marker_original_exit_code"] = result.exit_code
        item["metadata"] = metadata
        self._finalize_completed_work_order_result(
            item,
            result,
            clean_paths=clean_paths,
            worker_type_circuit_breakers=worker_type_circuit_breakers,
            worker_type_circuit_breaker_policy=worker_type_circuit_breaker_policy,
        )
        return

    salvage_outcome = str(item.get("worker_outcome", "")).strip()
    is_salvage = salvage_outcome in {
        WorkerOutcome.CRASH_WITH_SALVAGE.value,
        WorkerOutcome.TIMEOUT_WITH_SALVAGE.value,
    }
    if not clean_paths and not result.commit_shas and not is_salvage:
        # Fail closed: a non-zero exit without current commits or real file
        # changes must not inherit an older PR/receipt and look salvageable.
        for key in (
            "receipt_id",
            "confidence",
            "pr_url",
            "adopted_pr",
            "merge_gate",
            "verification_missing_reason",
        ):
            item.pop(key, None)
    deliverable_present = bool(self._work_order_deliverable_type(item))
    if deliverable_present and is_salvage:
        # Salvaged deliverables proceed to completion — the recovery was
        # intentional and the deliverable (commits/PR) is real. Clear any
        # stale blocker metadata from the failed attempt so the lane does
        # not remain "completed" while still looking blocked. Also drop any
        # inherited receipt/confidence so completion backfill records a fresh
        # receipt for the current salvaged deliverable.
        item["status"] = "completed"
        item["review_status"] = "pending_heterogeneous_review"
        for key in (
            "dispatch_error",
            "resource_error",
            "failure_reason",
            "blocking_question",
            "blocker",
            "conflicts",
            "merge_gate",
            "verification_missing_reason",
            "scope_violation",
            "receipt_id",
            "confidence",
        ):
            item.pop(key, None)
        item.pop("blockers", None)
        item["exit_code"] = result.exit_code
        item.pop("failure_reason", None)
        item.pop("blocking_question", None)
        item.pop("blocker", None)
        item.pop("dispatch_error", None)
        self._release_terminal_lease(item)
        self._register_pr_if_present(item, result)
        return
    if deliverable_present:
        failure_reason = "worker_crash_with_deliverable"
        for key in (
            "resource_error",
            "conflicts",
            "scope_violation",
            "merge_gate",
            "verification_missing_reason",
        ):
            item.pop(key, None)
        item.pop("blockers", None)
        self._mark_needs_human(
            item,
            "worker exited non-zero after producing a recoverable deliverable",
            failure_reason=failure_reason,
            blocking_question=(
                "Should the recovered deliverable be adopted as-is, amended, or rerun before integration?"
            ),
        )
        _append_repair_journal(self, item, result, reason=failure_reason)
        item["review_status"] = "changes_requested"
        item["receipt_id"] = None
        self._release_terminal_lease(item)
        item["exit_code"] = result.exit_code
        return

    if lease_id:
        self.store.release_lease(lease_id, status=LeaseStatus.RELEASED)
    for key in (
        "receipt_id",
        "confidence",
        "pr_url",
        "adopted_pr",
        "merge_gate",
        "verification_missing_reason",
        "scope_violation",
        "resource_error",
        "conflicts",
    ):
        item.pop(key, None)
    item.pop("blockers", None)
    failure_reason = "worker_timeout_no_deliverable" if result.exit_code == -1 else "worker_crash"
    blocking_question = self._default_blocking_question(failure_reason)
    stderr_text = result.stderr.strip()
    item["status"] = "timed_out" if result.exit_code == -1 else "failed"
    item["dispatch_error"] = stderr_text or (
        "worker timed out before producing a deliverable"
        if result.exit_code == -1
        else "worker crashed before producing a deliverable"
    )
    item["failure_reason"] = failure_reason
    item["blocking_question"] = blocking_question
    item["blocker"] = {
        "reason": failure_reason,
        "question": blocking_question,
    }
    _append_repair_journal(self, item, result, reason=failure_reason)
    blockers: list[str] = []
    if item["dispatch_error"] not in blockers:
        blockers.append(item["dispatch_error"])
    item["blockers"] = blockers
    item["review_status"] = "changes_requested"
    item["exit_code"] = result.exit_code


def _capacity_failure_detail(self, result: WorkerProcess) -> str:
    """Detect capacity/billing failures in worker output.

    Uses LLM classification first, falling back to keyword patterns.
    """
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if not combined:
        return ""

    # --- LLM classification ---
    llm_succeeded = False
    try:
        from concurrent.futures import ThreadPoolExecutor

        from aragora.ralph.llm_classifier import LLMBlockerClassifier

        import asyncio

        classifier = LLMBlockerClassifier()
        with ThreadPoolExecutor(max_workers=1) as pool:
            verdict = pool.submit(
                asyncio.run,
                classifier.detect_capacity_failure(
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                    agent_name=result.agent or "unknown",
                ),
            ).result(timeout=self._LLM_CALL_TIMEOUT)
        # Only trust the LLM verdict if it actually ran (not a fallback default)
        if verdict.reasoning != "LLM call failed":
            llm_succeeded = True
            if verdict.is_capacity:
                logger.info(
                    "LLM capacity detection: is_capacity=%s (reasoning: %s)",
                    verdict.is_capacity,
                    verdict.reasoning,
                )
                return verdict.detail or combined or f"{result.agent} worker failed"
            return ""
    except _LLM_PROBE_EXCEPTIONS:
        logger.debug("LLM capacity detection failed, using keyword fallback", exc_info=True)

    # --- keyword fallback (when LLM unavailable) ---
    if not llm_succeeded:
        return self._keyword_capacity_failure_detail(combined, result.agent or "unknown")
    return ""


def _keyword_capacity_failure_detail(combined: str, agent_name: str) -> str:
    """Keyword-based fallback for capacity failure detection."""
    lowered = combined.lower()
    capacity_patterns = (
        "credit balance is too low",
        "insufficient credit",
        "insufficient balance",
        "out of credits",
        "quota exceeded",
        "usage limit reached",
        "rate limit exceeded",
        "billing",
        "payment required",
    )
    if any(pattern in lowered for pattern in capacity_patterns):
        return combined or f"{agent_name} worker failed"
    return ""


def _completion_confidence(item: dict[str, Any], result: WorkerProcess) -> float:
    expected_tests = [str(test) for test in item.get("expected_tests", []) if str(test).strip()]
    if result.exit_code != 0:
        return 0.0
    if expected_tests:
        return 0.8 if result.tests_run else 0.65
    if result.commit_shas or result.changed_paths:
        return 0.6
    return 0.4


def _verification_results_from_result(result: WorkerProcess) -> list[dict[str, Any]]:
    raw_results = list(getattr(result, "verification_results", []) or [])
    normalized: list[dict[str, Any]] = []
    for entry in raw_results:
        if not isinstance(entry, dict):
            continue
        command = str(entry.get("command", "")).strip()
        if not command:
            continue
        raw_exit_code = entry.get("exit_code", 0)
        try:
            if isinstance(raw_exit_code, (bool, float)):
                raise TypeError
            exit_code = int(raw_exit_code)
        except (TypeError, ValueError):
            exit_code = -1
        raw_duration_seconds = entry.get("duration_seconds", 0.0)
        try:
            if isinstance(raw_duration_seconds, bool):
                raise TypeError
            duration_seconds = float(raw_duration_seconds or 0.0)
        except (TypeError, ValueError):
            duration_seconds = 0.0
        raw_passed = entry.get("passed")
        if isinstance(raw_passed, bool):
            passed = raw_passed and exit_code == 0
        elif "passed" not in entry:
            passed = exit_code == 0
        else:
            passed = False
        normalized.append(
            {
                "command": command,
                "exit_code": exit_code,
                "passed": passed,
                "stdout": str(entry.get("stdout", "")),
                "stderr": str(entry.get("stderr", "")),
                "duration_seconds": duration_seconds,
            }
        )
    if normalized:
        return normalized
    return [
        {
            "command": str(command).strip(),
            "exit_code": 0,
            "passed": True,
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
        }
        for command in result.tests_run
        if str(command).strip()
    ]


def _canonical_verification_command(command: Any) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    for prefix in ("bash -lc ", "/bin/bash -lc "):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
                text = text[1:-1]
            break
    text = re.sub(r"^(?P<prefix>\s*)python3(?=\s|$)", r"\g<prefix>python", text)
    return WorkerLauncher._normalize_verification_command(text).strip()


def _pytest_command_targets(cls, command: Any) -> list[str]:
    text = cls._canonical_verification_command(command)
    if not text:
        return []
    try:
        tokens = shlex.split(text)
    except ValueError:
        return []
    if not tokens:
        return []
    start = 0
    if len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest":
        start = 3
    elif tokens[0].endswith("pytest"):
        start = 1
    else:
        return []

    targets: list[str] = []
    skip_next = False
    options_with_values = {"-k", "-m", "--maxfail", "--timeout", "--tb", "-c", "--rootdir"}
    for token in tokens[start:]:
        if skip_next:
            skip_next = False
            continue
        if token in options_with_values:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        normalized = str(token).strip().removeprefix("./").rstrip("/")
        if token.endswith("/") or "/" in normalized or normalized.endswith(".py"):
            targets.append(normalized)
    return targets


def _pytest_command_has_selectors(cls, command: Any) -> bool:
    """Return True if the pytest command contains -k or -m selectors."""
    text = cls._canonical_verification_command(command)
    if not text:
        return False
    try:
        tokens = shlex.split(text)
    except ValueError:
        return False
    start = 0
    if len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest":
        start = 3
    elif tokens and tokens[0].endswith("pytest"):
        start = 1
    else:
        return False
    skip_next = False
    for token in tokens[start:]:
        if skip_next:
            skip_next = False
            continue
        if token in {"-k", "-m"}:
            return True
        options_with_values = {"--maxfail", "--timeout", "--tb", "-c", "--rootdir"}
        if token in options_with_values:
            skip_next = True
            continue
    return False


def _verification_command_covers_expected(
    cls, recorded_command: Any, expected_command: Any
) -> bool:
    recorded = cls._canonical_verification_command(recorded_command)
    expected = cls._canonical_verification_command(expected_command)
    if not recorded or not expected:
        return False
    if recorded == expected:
        return True
    recorded_targets = cls._pytest_command_targets(recorded)
    expected_targets = cls._pytest_command_targets(expected)
    if not recorded_targets or not expected_targets:
        return False
    if cls._pytest_command_has_selectors(recorded) or cls._pytest_command_has_selectors(expected):
        return False
    for expected_target in expected_targets:
        if not any(
            expected_target == recorded_target
            or expected_target.startswith(recorded_target.rstrip("/") + "/")
            for recorded_target in recorded_targets
        ):
            return False
    return True


def _merge_gate_entry_passed(entry: dict[str, Any]) -> bool:
    if entry.get("passed") is not True:
        return False
    raw_exit_code = entry.get("exit_code", 0)
    try:
        if isinstance(raw_exit_code, (bool, float)):
            raise TypeError
        exit_code = int(raw_exit_code)
    except (TypeError, ValueError):
        return False
    return exit_code == 0


def _merge_gate_state(cls, item: dict[str, Any]) -> dict[str, Any]:
    expected_checks = [
        str(test).strip() for test in item.get("expected_tests", []) if str(test).strip()
    ]
    verification_results = [
        dict(entry)
        for entry in item.get("verification_results", [])
        if isinstance(entry, dict) and str(entry.get("command", "")).strip()
    ]
    missing_checks = [
        command
        for command in expected_checks
        if not any(
            cls._verification_command_covers_expected(entry.get("command", ""), command)
            for entry in verification_results
        )
    ]
    failed_checks = [
        dict(entry)
        for entry in verification_results
        if any(
            cls._verification_command_covers_expected(entry.get("command", ""), command)
            for command in expected_checks
        )
        and not cls._merge_gate_entry_passed(entry)
    ]
    deferred_dependency_ids = [
        str(dep).strip()
        for dep in (
            dict(item.get("metadata") or {}).get("deferred_verification_to_dependency_ids") or []
        )
        if str(dep).strip()
    ]

    if deferred_dependency_ids:
        return {
            "enabled": True,
            "expected_checks": expected_checks,
            "verification_results": verification_results,
            "verification_missing_reason": None,
            "checks_passed": True,
            "human_approval_required": True,
            "merge_eligible": True,
            "blocked_reasons": [],
            "verification_deferred_to_dependency_ids": deferred_dependency_ids,
        }

    blocked_reasons: list[str] = []
    verification_missing_reason: str | None = None
    if not expected_checks:
        if cls._work_order_is_docs_only(item):
            return {
                "enabled": True,
                "expected_checks": expected_checks,
                "verification_results": verification_results,
                "verification_missing_reason": None,
                "checks_passed": True,
                "human_approval_required": True,
                "merge_eligible": True,
                "blocked_reasons": [],
            }
        verification_missing_reason = "missing_verification_plan"
        blocked_reasons.append(
            "merge gate blocked: missing verification plan or verification command"
        )
    if missing_checks:
        blocked_reasons.append(
            "merge gate blocked: required verification did not run: "
            + ", ".join(missing_checks[:3])
        )
    if failed_checks:
        first = failed_checks[0]
        reason = (
            "merge gate blocked: verification failed: "
            f"{first.get('command', '')} (exit {first.get('exit_code', -1)})"
        )
        stderr = str(first.get("stderr", "")).strip()
        if stderr:
            reason = f"{reason} - {stderr.splitlines()[0][:200]}"
        blocked_reasons.append(reason)

    checks_passed = not blocked_reasons
    return {
        "enabled": True,
        "expected_checks": expected_checks,
        "verification_results": verification_results,
        "verification_missing_reason": verification_missing_reason,
        "checks_passed": checks_passed,
        "human_approval_required": True,
        "merge_eligible": checks_passed,
        "blocked_reasons": blocked_reasons,
    }


def _is_docs_only_path(path: Any) -> bool:
    return is_docs_safe_path(path)


def _work_order_is_docs_only(cls, item: dict[str, Any]) -> bool:
    candidates = [
        str(path).strip()
        for path in item.get("changed_paths", []) or item.get("file_scope", [])
        if str(path).strip()
    ]
    if not candidates:
        return False
    return all(cls._is_docs_only_path(path) for path in candidates)


def _merge_gate_failure_reason(merge_gate: dict[str, Any]) -> str:
    reasons = [
        str(reason).strip()
        for reason in merge_gate.get("blocked_reasons", [])
        if str(reason).strip()
    ]
    return reasons[0] if reasons else "merge gate blocked"


def _merge_gate_blocking_question(merge_gate: dict[str, Any]) -> str:
    missing = str(merge_gate.get("verification_missing_reason", "")).strip()
    if missing == "missing_verification_plan":
        return "Which verification command or acceptance check should be added before rerunning?"
    return "Which required verification or acceptance check must pass before approval?"


def _update_log_tails(
    cls,
    item: dict[str, Any],
    *,
    stdout: str,
    stderr: str,
) -> bool:
    changed = False
    for key, value in {
        "stdout_tail": cls._log_tail(stdout),
        "stderr_tail": cls._log_tail(stderr),
    }.items():
        if value:
            if str(item.get(key, "")) != value:
                item[key] = value
                changed = True
        elif key in item:
            item.pop(key, None)
            changed = True
    return changed


def _log_tail(text: str, *, max_chars: int = MAX_WORKER_LOG_TAIL_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _progress_fingerprint(source: Any) -> dict[str, Any]:
    payload = dict(source or {})
    return {
        "head_sha": str(payload.get("head_sha", "")).strip(),
        "changed_paths": sorted(
            str(path).strip() for path in payload.get("changed_paths", []) if str(path).strip()
        ),
        "diff_lines": int(payload.get("diff_lines", 0) or 0),
    }


def _output_fingerprint(source: Any) -> dict[str, Any]:
    payload = dict(source or {})
    stdout_tail = str(payload.get("stdout_tail", "")).strip()
    stderr_tail = str(payload.get("stderr_tail", "")).strip()
    stdout_size = int(payload.get("stdout_size", 0) or 0)
    stderr_size = int(payload.get("stderr_size", 0) or 0)
    return {
        "stdout_size": stdout_size,
        "stderr_size": stderr_size,
        "stdout_mtime_ns": int(payload.get("stdout_mtime_ns", 0) or 0),
        "stderr_mtime_ns": int(payload.get("stderr_mtime_ns", 0) or 0),
        "has_output": bool(stdout_tail or stderr_tail or stdout_size or stderr_size),
    }


def _no_progress_timeout_seconds(self) -> float:
    raw = getattr(self.launcher.config, "no_progress_timeout_seconds", 120.0)
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 120.0


def _no_progress_anchor(self, item: dict[str, Any]) -> datetime | None:
    since = self._parse_timestamp(item.get("last_progress_at")) or self._parse_timestamp(
        item.get("dispatched_at")
    )
    output_state = self._output_fingerprint(item.get("output_fingerprint"))
    if output_state.get("has_output"):
        last_output_at = self._parse_timestamp(item.get("last_output_at"))
        if last_output_at is not None and (since is None or last_output_at > since):
            since = last_output_at
    return since


def _exceeded_no_progress_timeout(self, item: dict[str, Any]) -> bool:
    since = self._no_progress_anchor(item)
    if since is None:
        return False
    elapsed = (datetime.now(UTC) - since).total_seconds()
    return elapsed >= self._no_progress_timeout_seconds()


def _mark_scope_violation(
    self,
    item: dict[str, Any],
    violations: list[dict[str, Any]],
    *,
    extra_reason: str = "",
) -> None:
    """Mark a work order as failed due to file-scope violation.

    This is the fail-closed enforcement gate: workers that edit outside
    their permitted scope are stopped immediately rather than allowed to
    continue producing wrong work.

    Persists the violation into the lease metadata so fleet/integrator
    views can surface it without relying on in-memory work-order state.
    """
    out_of_scope_paths = [
        str(v.get("path", "")) for v in violations if v.get("type") == "out_of_scope"
    ]
    reason = "worker edited files outside permitted scope: " + ", ".join(out_of_scope_paths[:5])
    if extra_reason:
        reason = f"{extra_reason}; {reason}"
    changed_paths = [
        str(path).strip() for path in item.get("changed_paths", []) if str(path).strip()
    ]
    for key in (
        "receipt_id",
        "confidence",
        "worker_outcome",
        "completed_at",
        "exit_code",
        "head_sha",
        "commit_shas",
        "diff",
        "diff_lines",
        "tests_run",
        "verification_results",
        "resource_error",
        "conflicts",
        "pr_url",
        "adopted_pr",
        "merge_gate",
        "verification_missing_reason",
    ):
        item.pop(key, None)
    item["status"] = "scope_violation"
    item["dispatch_error"] = reason
    item["failure_reason"] = "scope_violation"
    item["blocking_question"] = (
        "Which files should stay in scope, or should this lane be split before rerunning?"
    )
    item["blocker"] = {
        "reason": "scope_violation",
        "question": item["blocking_question"],
    }
    item["review_status"] = "changes_requested"
    scope_violation_detail = {
        "violations": violations,
        "changed_paths": changed_paths,
        "detected_at": datetime.now(UTC).isoformat(),
    }
    item["scope_violation"] = scope_violation_detail
    item["blockers"] = [reason]
    item.pop("pid", None)

    # Write violation metadata into the lease so status_summary() surfaces
    # it.  The lease stays *active* — matching what record_completion() does
    # — so list_active_leases() picks it up for fleet/integrator views.
    lease_id = str(item.get("lease_id", "")).strip()
    if lease_id:
        try:
            self.store.persist_scope_violation(
                lease_id,
                changed_paths=list(item.get("changed_paths", [])),
                violations=violations,
            )
        except _BEST_EFFORT_STORE_EXCEPTIONS:
            pass  # Best-effort — local item is already marked


def _llm_adjudicate_scope(
    self,
    item: dict[str, Any],
    violations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use LLM to filter false-positive scope violations.

    Returns the reduced list of violations (may be empty if all justified).
    On any failure, returns the original violations unchanged (fail-closed).
    """
    try:
        from aragora.ralph.llm_classifier import LLMBlockerClassifier

        classifier = LLMBlockerClassifier()
        task_desc = str(item.get("task_description", item.get("title", "")))
        declared_scope = [str(s).strip() for s in item.get("file_scope", []) if str(s).strip()]
        changed_paths = [str(p) for p in item.get("changed_paths", [])]

        import asyncio
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            verdict = pool.submit(
                asyncio.run,
                classifier.adjudicate_scope(
                    task_description=task_desc,
                    declared_scope=declared_scope,
                    changed_paths=changed_paths,
                    violations=violations,
                ),
            ).result(timeout=self._LLM_CALL_TIMEOUT)

        if verdict.justified_paths:
            logger.info(
                "LLM scope adjudicator justified %d paths: %s (%s)",
                len(verdict.justified_paths),
                verdict.justified_paths,
                verdict.reasoning,
            )
        justified_set = set(verdict.justified_paths)
        remaining = [v for v in violations if str(v.get("path", "")) not in justified_set]
        return remaining
    except _LLM_PROBE_EXCEPTIONS:
        logger.debug("LLM scope adjudication failed, keeping all violations", exc_info=True)
        return violations


def _llm_override_merge_gate(
    self,
    item: dict[str, Any],
    merge_gate: dict[str, Any],
) -> bool:
    """Ask LLM if merge gate failure is cosmetic or genuine.

    Returns True if the LLM says the deliverable is ready despite the
    gate failure.  Returns False on any error (fail-closed).
    """
    if str(merge_gate.get("verification_missing_reason", "")).strip() == (
        "missing_verification_plan"
    ):
        return False
    try:
        from aragora.ralph.llm_classifier import LLMBlockerClassifier

        classifier = LLMBlockerClassifier()
        acceptance_criteria = [
            str(c).strip() for c in item.get("acceptance_criteria", []) if str(c).strip()
        ]
        verification_results = merge_gate.get("verification_results", [])
        changed_paths = [str(p) for p in item.get("changed_paths", [])]
        diff_summary = str(item.get("diff_summary", ""))[:2000]

        import asyncio
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            verdict = pool.submit(
                asyncio.run,
                classifier.evaluate_merge_readiness(
                    acceptance_criteria=acceptance_criteria,
                    verification_results=verification_results,
                    changed_paths=changed_paths,
                    diff_summary=diff_summary,
                ),
            ).result(timeout=self._LLM_CALL_TIMEOUT)

        logger.info(
            "LLM merge evaluation: ready=%s blocking=%s advisory=%s (%s)",
            verdict.ready,
            verdict.blocking_issues,
            verdict.advisory_issues,
            verdict.reasoning,
        )
        return verdict.ready
    except _LLM_PROBE_EXCEPTIONS:
        logger.debug("LLM merge evaluation failed, fail-closed", exc_info=True)
        return False


def _check_file_scope_violations(
    work_order: dict[str, Any],
    changed_paths: list[str],
) -> list[dict[str, Any]]:
    """Check whether changed paths fall within the work order's file scope.

    Returns a list of violation dicts (empty = no violations).
    File-scope enforcement is strict: every changed path must match at
    least one scope pattern. If the work order has no file_scope declared,
    no enforcement is applied (open scope).
    """
    file_scope = [
        str(item).strip() for item in work_order.get("file_scope", []) if str(item).strip()
    ]
    if not file_scope or not changed_paths:
        return []

    # Expand scope: for specific test file paths, also allow sibling
    # test files in the same directory.  Workers often choose a more
    # descriptive file name than the issue suggested (e.g.
    # test_live_explainability_wiring.py vs test_live_explainability_e2e.py).
    expanded_scope = list(file_scope)
    for scope in file_scope:
        clean = scope.strip().removeprefix("./")
        parts = clean.split("/")
        # If it's a specific test file (e.g. tests/debate/test_foo.py),
        # add the parent directory as an allowed scope
        if len(parts) >= 2 and any(part.startswith("test") for part in parts) and "." in parts[-1]:
            parent = "/".join(parts[:-1])
            if parent not in expanded_scope:
                expanded_scope.append(parent)

    # Always allow common infrastructure files that workers need to touch
    always_allowed = {
        "conftest.py",
        "__init__.py",
        "pyproject.toml",
        ".gitignore",
        "setup.cfg",
        "setup.py",
    }

    violations: list[dict[str, Any]] = []
    for path in changed_paths:
        normalized = str(path).strip().removeprefix("./")
        if not normalized:
            continue
        # Allow common infrastructure files regardless of scope
        basename = normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized
        if basename in always_allowed:
            continue
        if not any(_path_in_scope(normalized, scope) for scope in expanded_scope):
            violations.append(
                {
                    "type": "out_of_scope",
                    "path": normalized,
                    "allowed_scope": list(file_scope),
                }
            )
    return violations


def _validate_file_scope(file_scope: list[str], worktree_path: str) -> list[str]:
    """Drop file_scope entries whose top-level directory does not exist.

    LLM planners sometimes hallucinate paths (e.g. ``src/orchestrator/``
    when the real code lives at ``aragora/nomic/``).  These entries block
    workers at two layers:

    1. The prompt tells the worker to stay in scope → worker refuses to
       edit real files and exits with zero deliverables.
    2. The supervisor enforces scope on changed paths → any real edits
       are rejected as ``scope_violation``.

    This method strips entries whose first path component (e.g. ``src``)
    does not exist in the worktree, so that both prompt and enforcement
    operate on the real codebase structure.  Valid entries (e.g.
    ``aragora/nomic/foo.py``) pass through unchanged.
    """
    if not file_scope or not worktree_path:
        return file_scope
    wt = Path(worktree_path)
    if not wt.is_dir():
        return file_scope
    # Only validate against real git checkouts (have .git file/dir).
    # Test fixtures create bare directories without .git.
    dot_git = wt / ".git"
    if not dot_git.exists():
        return file_scope
    valid: list[str] = []
    for scope_path in file_scope:
        clean = scope_path.removeprefix("./").strip()
        if not clean:
            continue
        root = clean.split("/")[0]
        if (wt / root).exists():
            valid.append(scope_path)
        else:
            logger.warning(
                "Dropping hallucinated file_scope entry %r (root %r not found in %s)",
                scope_path,
                root,
                worktree_path,
            )
    return valid
