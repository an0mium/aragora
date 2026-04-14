"""Helpers for boss-loop outcome and metrics emission."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from aragora.swarm.outcome_learner import load_category_success_rates
from aragora.swarm.terminal_truth import TerminalClass, classify_from_metrics

logger = logging.getLogger(__name__)


def _serialize_blocker_evidence(value: Any) -> str | None:
    """Normalize blocker evidence into stable text for JSONL metrics rows."""
    if value is None:
        return None
    if isinstance(value, str):
        compact = " ".join(value.split())
        return compact or None
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            pass
    compact = " ".join(str(value).split())
    return compact or None


def extract_iteration_metrics(worker_result: dict[str, Any]) -> tuple[int, int, int]:
    """Summarize changed files and test verification from a worker run.

    Returns (files_changed, tests_run, tests_passed).
    """
    run_dict = worker_result.get("run")
    if not isinstance(run_dict, dict):
        return 0, 0, 0

    changed_files: list[str] = []
    tests_run: list[str] = []
    tests_passed = 0
    saw_verification_results = False

    for work_order in run_dict.get("work_orders", []):
        if not isinstance(work_order, dict):
            continue
        changed_files.extend(
            str(path).strip() for path in work_order.get("changed_paths", []) if str(path).strip()
        )
        tests_run.extend(
            str(command).strip()
            for command in work_order.get("tests_run", [])
            if str(command).strip()
        )
        verification_results = work_order.get("verification_results", [])
        if not isinstance(verification_results, list):
            continue
        for verification in verification_results:
            if not isinstance(verification, dict):
                continue
            saw_verification_results = True
            if verification.get("passed") is True:
                tests_passed += 1

    unique_changed_files = list(dict.fromkeys(changed_files))
    unique_tests_run = list(dict.fromkeys(tests_run))
    if (
        not saw_verification_results
        and unique_tests_run
        and str(worker_result.get("status", "")).strip().lower() == "completed"
    ):
        tests_passed = len(unique_tests_run)

    return len(unique_changed_files), len(unique_tests_run), tests_passed


def append_iteration_metrics(
    *,
    metrics_jsonl_path: str | None,
    outcome_learner_window: int,
    deferred_queue_depth: int,
    iteration: int,
    issue_number: int | None,
    worker_result: dict[str, Any],
    elapsed_seconds: float,
    files_changed: int,
    tests_run: int,
    tests_passed: int,
) -> None:
    """Append one JSONL row for a finalized boss-loop iteration."""
    metrics_path_text = str(metrics_jsonl_path or "").strip()
    if not metrics_path_text:
        return

    try:
        metrics_path = Path(metrics_path_text)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        run_dict: Any = worker_result.get("run")
        receipt_metadata: Any = worker_result.get("receipt_metadata")
        prompt_chars: int = 0
        enriched_context_chars: int = 0
        prompt_version: str = "v2"
        issue_title: str = str(
            receipt_metadata.get("issue_title", "") if isinstance(receipt_metadata, dict) else ""
        ).strip()
        issue_title = issue_title or str(worker_result.get("issue_title", "")).strip()
        is_decomposed: bool = bool(re.search(r"\[from #\d+\]", issue_title))
        cohort_tag: str | None = "B0-cohort" if issue_title.startswith("[B0-cohort]") else None
        publish_action: str | None = (
            str((worker_result.get("publish_result") or {}).get("action", "")).strip() or None
        )
        category_success_rates: dict[str, float] = load_category_success_rates(
            window_size=outcome_learner_window
        )
        sanitizer_outcome: str | None = (
            str(worker_result.get("sanitizer_outcome", "")).strip() or None
        )
        checks_failed: list[Any] = (
            raw_checks if isinstance(raw_checks := worker_result.get("checks_failed"), list) else []
        )

        if isinstance(run_dict, dict):
            for wo in run_dict.get("work_orders", []):
                if isinstance(wo, dict):
                    prompt_chars += int(wo.get("prompt_chars", 0) or 0)
                    enriched_context_chars += int(wo.get("enriched_context_chars", 0) or 0)

        # Extract failure/blocker evidence for operator visibility (BC-03)
        failure_reason: str | None = str(worker_result.get("failure_reason", "")).strip() or None
        blocker_kind: str | None = str(worker_result.get("blocker_kind", "")).strip() or None
        blocker_evidence: str | None = None
        for candidate in (
            worker_result.get("blocker_evidence"),
            receipt_metadata.get("blocker_evidence")
            if isinstance(receipt_metadata, dict)
            else None,
            worker_result.get("dispatch_gate"),
            receipt_metadata.get("dispatch_gate") if isinstance(receipt_metadata, dict) else None,
        ):
            blocker_evidence = _serialize_blocker_evidence(candidate)
            if blocker_evidence:
                break
        needs_human_reasons: Any = worker_result.get("reasons")
        if isinstance(needs_human_reasons, list) and needs_human_reasons:
            if not failure_reason:
                failure_reason = str(needs_human_reasons[0]).strip()[:200]

        payload: dict[str, Any] = {
            "iteration": int(iteration),
            "issue_number": issue_number,
            "issue_title": issue_title[:120] if issue_title else None,
            "worker_status": str(worker_result.get("status", "")).strip() or "unknown",
            "worker_outcome": str(worker_result.get("outcome", "")).strip() or None,
            "elapsed_seconds": float(elapsed_seconds or 0.0),
            "files_changed": files_changed,
            "tests_run": tests_run,
            "tests_passed": tests_passed,
            "prompt_version": prompt_version,
            "prompt_chars": prompt_chars,
            "enriched_context_chars": enriched_context_chars,
            "is_decomposed_issue": is_decomposed,
            "deferred_queue_depth": deferred_queue_depth,
            "sanitizer_outcome": sanitizer_outcome,
            "sanitizer_checks_failed_count": sum(1 for item in checks_failed if str(item).strip()),
            "cohort_tag": cohort_tag,
            "has_deliverable": bool(worker_result.get("deliverable")),
            "publish_action": publish_action,
            "failure_reason": failure_reason,
            "blocker_kind": blocker_kind,
            "blocker_evidence": blocker_evidence,
            "category_success_rates": category_success_rates,
        }

        try:
            terminal_class = classify_from_metrics(payload)
            payload["terminal_class"] = terminal_class.value
        except Exception:
            payload["terminal_class"] = TerminalClass.RESCUE_NO_DELIVERABLE.value

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
    except Exception as exc:
        logger.debug("Boss metrics emission skipped: %s", exc)
