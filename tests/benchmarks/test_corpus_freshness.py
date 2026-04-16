from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "docs/benchmarks/corpus.json"
LATEST_TRUTH_PATH = (
    REPO_ROOT
    / "docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/latest.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus_issue_numbers(corpus: dict[str, Any]) -> list[int]:
    return sorted(int(issue["issue_id"]) for issue in corpus["issues"])


def test_current_benchmark_corpus_has_fresh_verifiable_truth() -> None:
    corpus = _read_json(CORPUS_PATH)
    truth = _read_json(LATEST_TRUTH_PATH)
    issue_numbers = _corpus_issue_numbers(corpus)

    truth_corpus = truth["corpus"]
    assert truth_corpus["path"] == "docs/benchmarks/corpus.json"
    assert truth_corpus["corpus_id"] == corpus["corpus_id"]
    assert truth_corpus["revision"] == corpus["revision"]
    assert truth_corpus["issue_count"] == len(issue_numbers)
    assert truth_corpus["membership_issue_numbers"] == issue_numbers

    truth_records = {
        int(record["issue_number"]): record
        for record in truth["issues"]
        if int(record["issue_number"]) in issue_numbers
    }
    assert sorted(truth_records) == issue_numbers
    assert truth["coverage"]["status"] == "complete"
    assert truth["coverage"]["missing_issue_numbers"] == []

    freshness = truth["corpus_freshness"]
    assert freshness["status"] == "fresh"
    assert freshness["stale_closed_issue_count"] == 0
    assert freshness["stale_closed_issue_numbers"] == []
    assert freshness["stale_closed_issues"] == []
    assert freshness["linkage_error_count"] == 0
    assert freshness["linkage_errors"] == []

    stale_closed_records = [
        record
        for record in truth_records.values()
        if str(record.get("issue_state", "")).upper() == "CLOSED"
        and (
            record.get("truth_state") == "no_linked_pr"
            or record.get("stale_corpus_issue")
            or record.get("stale_corpus_reason")
        )
    ]
    assert stale_closed_records == []

    unverifiable_records = [
        record
        for record in truth_records.values()
        if record.get("linkage_verification_incomplete")
        or record.get("linkage_status") != "verified"
        or not record.get("truth_success")
    ]
    assert unverifiable_records == []
