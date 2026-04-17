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

VALID_EXPECTED_STATUSES = {"verified", "in_progress"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus_issue_numbers(corpus: dict[str, Any]) -> list[int]:
    return sorted(int(issue["issue_id"]) for issue in corpus["issues"])


def _corpus_expected_status(corpus: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for item in corpus["issues"]:
        issue_id = int(item["issue_id"])
        status = str(item.get("expected_status") or "verified").strip().lower()
        if status not in VALID_EXPECTED_STATUSES:
            raise AssertionError(
                f"corpus issue #{issue_id} has unknown expected_status={status!r}; "
                f"must be one of {sorted(VALID_EXPECTED_STATUSES)}"
            )
        mapping[issue_id] = status
    return mapping


def test_current_benchmark_corpus_has_fresh_verifiable_truth() -> None:
    corpus = _read_json(CORPUS_PATH)
    truth = _read_json(LATEST_TRUTH_PATH)
    issue_numbers = _corpus_issue_numbers(corpus)
    expected_by_number = _corpus_expected_status(corpus)

    truth_corpus = truth["corpus"]
    assert truth_corpus["path"] == "docs/benchmarks/corpus.json"
    assert truth_corpus["corpus_id"] == corpus["corpus_id"]
    assert truth_corpus["revision"] == corpus["revision"]
    assert truth_corpus["issue_count"] == len(issue_numbers)
    assert truth_corpus["membership_issue_numbers"] == issue_numbers

    verified_numbers = sorted(n for n, s in expected_by_number.items() if s == "verified")
    in_progress_numbers = sorted(n for n, s in expected_by_number.items() if s == "in_progress")
    assert truth_corpus.get("verified_expected_count") == len(verified_numbers)
    assert truth_corpus.get("in_progress_expected_count") == len(in_progress_numbers)

    truth_records = {
        int(record["issue_number"]): record
        for record in truth["issues"]
        if int(record["issue_number"]) in issue_numbers
    }
    assert sorted(truth_records) == issue_numbers

    # Every corpus record must carry its expected_status back on the artifact
    # so downstream consumers (dashboards, the boss loop, humans) see it.
    for issue_number, record in truth_records.items():
        assert record.get("expected_status") == expected_by_number[issue_number], (
            f"issue #{issue_number}: expected_status mismatch "
            f"(corpus={expected_by_number[issue_number]!r}, "
            f"artifact={record.get('expected_status')!r})"
        )

    assert truth["coverage"]["status"] == "complete"
    assert truth["coverage"]["missing_issue_numbers"] == []

    freshness = truth["corpus_freshness"]
    # Verified-expected issues must have no stale-closed entries. In-progress
    # issues aren't in scope for stale-closed checks by construction: they're
    # expected to be open and not yet PR-linked.
    assert freshness["status"] == "fresh"
    assert freshness["stale_closed_issue_count"] == 0
    assert freshness["stale_closed_issue_numbers"] == []
    assert freshness["stale_closed_issues"] == []
    assert freshness["linkage_error_count"] == 0
    assert freshness["linkage_errors"] == []

    # Strict verification only applies to verified-expected records.
    verified_records = [truth_records[n] for n in verified_numbers]
    stale_closed_verified = [
        record
        for record in verified_records
        if str(record.get("issue_state", "")).upper() == "CLOSED"
        and (
            record.get("truth_state") == "no_linked_pr"
            or record.get("stale_corpus_issue")
            or record.get("stale_corpus_reason")
        )
    ]
    assert stale_closed_verified == []

    unverifiable_verified = [
        record
        for record in verified_records
        if record.get("linkage_verification_incomplete")
        or record.get("linkage_status") != "verified"
        or not record.get("truth_success")
    ]
    assert unverifiable_verified == []

    # In-progress records: they must still be open and must not be stale-closed.
    # Every other state is legitimate while autonomy is working the issue.
    in_progress_records = [truth_records[n] for n in in_progress_numbers]
    stuck_closed_in_progress = [
        record
        for record in in_progress_records
        if str(record.get("issue_state", "")).upper() == "CLOSED"
        and (record.get("truth_state") == "no_linked_pr" or record.get("stale_corpus_issue"))
    ]
    assert stuck_closed_in_progress == [], (
        "in_progress corpus issues closed without a verified PR link indicate "
        "silent regressions — they should be explicitly promoted to verified or "
        "removed from the corpus."
    )

    # The primary metric must be keyed to the verified subset only — this is
    # the PMF fix from the Turn 11 triage: the benchmark surface should report
    # autonomy-proven ratio, not a quotient over a pre-solved snapshot.
    primary = truth["primary_metrics"]
    assert "truth_success_rate_verified" in primary, (
        "primary_metrics must expose truth_success_rate_verified to distinguish "
        "autonomy-verified successes from in-flight corpus membership"
    )
    if verified_numbers:
        computed = sum(1 for record in verified_records if record.get("truth_success")) / len(
            verified_numbers
        )
        assert abs(primary["truth_success_rate_verified"] - round(computed, 4)) < 1e-6
