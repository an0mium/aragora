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

        run_dict = worker_result.get("run")
        receipt_metadata = worker_result.get("receipt_metadata")
        prompt_chars = 0
        enriched_context_chars = 0
        prompt_version = "v2"
        issue_title = str(
            receipt_metadata.get("issue_title", "") if isinstance(receipt_metadata, dict) else ""
        ).strip()
        issue_title = issue_title or str(worker_result.get("issue_title", "")).strip()
        is_decomposed = bool(re.search(r"\[from #\d+\]", issue_title))
        cohort_tag = "B0-cohort" if issue_title.startswith("[B0-cohort]") else None
        publish_action = (
            str((worker_result.get("publish_result") or {}).get("action", "")).strip() or None
        )
        category_success_rates = load_category_success_rates(window_size=outcome_learner_window)
        sanitizer_outcome = str(worker_result.get("sanitizer_outcome", "")).strip() or None
        checks_failed = (
            raw_checks if isinstance(raw_checks := worker_result.get("checks_failed"), list) else []
        )

        if isinstance(run_dict, dict):
            for wo in run_dict.get("work_orders", []):
                if isinstance(wo, dict):
                    prompt_chars += int(wo.get("prompt_chars", 0) or 0)
                    enriched_context_chars += int(wo.get("enriched_context_chars", 0) or 0)

        # Extract failure/blocker evidence for operator visibility (BC-03)
        failure_reason = str(worker_result.get("failure_reason", "")).strip() or None
        blocker_kind = str(worker_result.get("blocker_kind", "")).strip() or None
        needs_human_reasons = worker_result.get("reasons")
        if isinstance(needs_human_reasons, list) and needs_human_reasons:
            if not failure_reason:
                failure_reason = str(needs_human_reasons[0]).strip()[:200]

        payload = {
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
