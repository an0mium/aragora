"""Persistence helpers for TestFixer learning records."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.nomic.testfixer.orchestrator import FixAttempt, FixLoopResult

logger = logging.getLogger(__name__)


class TestFixerAttemptStore:
    """Append-only JSONL store for fix attempts and run summaries."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict[str, Any]) -> None:
        record["recorded_at"] = datetime.now().isoformat()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug("store.append path=%s type=%s", self.path, record.get("type"))

    def record_attempt(self, attempt: "FixAttempt") -> None:
        diagnostics: dict[str, Any] | None = None
        if attempt.test_result_after and attempt.test_result_after.diagnostics:
            diagnostics = attempt.test_result_after.diagnostics.to_dict()
        self._append(
            {
                "type": "attempt",
                "run_id": attempt.run_id,
                "iteration": attempt.iteration,
                "failure": {
                    "test_name": attempt.failure.test_name,
                    "test_file": attempt.failure.test_file,
                    "error_type": attempt.failure.error_type,
                    "error_message": attempt.failure.error_message,
                },
                "analysis": {
                    "category": attempt.analysis.category.value,
                    "fix_target": attempt.analysis.fix_target.value,
                    "confidence": attempt.analysis.confidence,
                    "root_cause": attempt.analysis.root_cause,
                    "root_cause_file": attempt.analysis.root_cause_file,
                },
                "proposal": {
                    "id": attempt.proposal.id,
                    "description": attempt.proposal.description,
                    "confidence": attempt.proposal.post_debate_confidence,
                    "diff": attempt.proposal.as_diff(),
                },
                "applied": attempt.applied,
                "success": attempt.success,
                "notes": attempt.notes,
                "diagnostics": diagnostics,
            }
        )

    def record_run(self, result: "FixLoopResult") -> None:
        final_diagnostics: dict[str, Any] | None = None
        if result.final_test_result and result.final_test_result.diagnostics:
            final_diagnostics = result.final_test_result.diagnostics.to_dict()
        self._append(
            {
                "type": "run",
                "run_id": result.run_id,
                "status": result.status.value,
                "started_at": result.started_at.isoformat(),
                "finished_at": result.finished_at.isoformat(),
                "total_iterations": result.total_iterations,
                "fixes_applied": result.fixes_applied,
                "fixes_successful": result.fixes_successful,
                "fixes_reverted": result.fixes_reverted,
                "final_diagnostics": final_diagnostics,
                "attempts": [
                    {
                        "iteration": attempt.iteration,
                        "failure": attempt.failure.test_name,
                        "success": attempt.success,
                    }
                    for attempt in result.attempts
                ],
            }
        )
