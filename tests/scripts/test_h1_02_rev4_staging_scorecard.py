"""Unit tests for ``scripts/h1_02_rev4_staging_scorecard.py``.

The H1-02 advisory scorecard reads two inputs:

1. The rev-4 staging manifest (33 entries by spec).
2. The dry-run dispatch ledger written by PR #6828's
   ``h1_01_dry_run_dispatch.py``.

Tests cover:

- The promotion-floor predicate (≥15 dispatched).
- Per-execution-class breakdown sums.
- Latest-outcome resolution when an issue has multiple ledger rows.
- Empty-ledger and missing-ledger graceful behavior.
- Markdown emission with the floor-status line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import h1_02_rev4_staging_scorecard as scorer  # noqa: E402


@pytest.fixture()
def fixture_corpus() -> dict:
    """A 4-entry staging corpus across three execution classes."""
    return {
        "revision": 4,
        "status": "staging",
        "issues": [
            {"issue_id": 5126, "execution_class": "missing_test_coverage"},
            {"issue_id": 5128, "execution_class": "missing_test_coverage"},
            {"issue_id": 5788, "execution_class": "exception_narrowing"},
            {"issue_id": 5808, "execution_class": "silent_exception_replacement"},
        ],
    }


def _row(issue_number: int, outcome: str) -> dict:
    return {
        "issue_number": issue_number,
        "worker_status": "dry_run",
        "worker_outcome": f"dry_run:{outcome}",
        "terminal_class": f"dry_run_{outcome}",
        "publish_action": "dry_run_no_publish",
        "cohort_tag": "h1-01-dry-run-evidence",
    }


class TestComputeScorecardAggregate:
    def test_zero_dispatched_when_ledger_empty(self, fixture_corpus: dict) -> None:
        card = scorer.compute_scorecard(fixture_corpus, [])
        assert card.dispatched_count == 0
        assert card.accepted_count == 0
        assert card.accepted_rate == 0.0
        assert card.promotion_floor_met is False

    def test_all_accepted_yields_full_accepted_rate(self, fixture_corpus: dict) -> None:
        rows = [_row(it["issue_id"], "accepted") for it in fixture_corpus["issues"]]
        card = scorer.compute_scorecard(fixture_corpus, rows)
        assert card.dispatched_count == 4
        assert card.accepted_count == 4
        assert card.accepted_rate == 1.0

    def test_promotion_floor_predicate_at_15(self) -> None:
        # Build a synthetic 16-entry corpus, dispatch 15 → floor met
        corpus = {
            "revision": 4,
            "status": "staging",
            "issues": [
                {"issue_id": 1000 + i, "execution_class": "missing_test_coverage"}
                for i in range(16)
            ],
        }
        rows = [_row(1000 + i, "accepted") for i in range(15)]
        card = scorer.compute_scorecard(corpus, rows)
        assert card.dispatched_count == 15
        assert card.promotion_floor == 15
        assert card.promotion_floor_met is True

    def test_promotion_floor_pending_below_15(self) -> None:
        corpus = {
            "revision": 4,
            "status": "staging",
            "issues": [
                {"issue_id": 2000 + i, "execution_class": "exception_narrowing"} for i in range(20)
            ],
        }
        rows = [_row(2000 + i, "accepted") for i in range(14)]
        card = scorer.compute_scorecard(corpus, rows)
        assert card.dispatched_count == 14
        assert card.promotion_floor_met is False


class TestPerClassBreakdown:
    def test_per_class_totals_match_corpus(self, fixture_corpus: dict) -> None:
        card = scorer.compute_scorecard(fixture_corpus, [])
        assert card.per_class["missing_test_coverage"].total == 2
        assert card.per_class["exception_narrowing"].total == 1
        assert card.per_class["silent_exception_replacement"].total == 1

    def test_outcome_buckets_count_correctly(self, fixture_corpus: dict) -> None:
        rows = [
            _row(5126, "accepted"),
            _row(5128, "rewritten"),
            _row(5788, "dropped"),
            _row(5808, "quarantined"),
        ]
        card = scorer.compute_scorecard(fixture_corpus, rows)
        cm = card.per_class["missing_test_coverage"]
        assert cm.dispatched == 2
        assert cm.accepted == 1
        assert cm.rewritten == 1
        en = card.per_class["exception_narrowing"]
        assert en.dropped == 1
        ser = card.per_class["silent_exception_replacement"]
        assert ser.quarantined == 1


class TestLatestOutcomeResolution:
    def test_latest_non_skipped_outcome_wins(self, fixture_corpus: dict) -> None:
        # Issue 5126 has rows in this order: skipped → accepted.
        # The non-skipped outcome must win.
        rows = [_row(5126, "skipped"), _row(5126, "accepted")]
        card = scorer.compute_scorecard(fixture_corpus, rows)
        assert card.per_class["missing_test_coverage"].accepted == 1
        assert card.per_class["missing_test_coverage"].skipped == 0

    def test_all_skipped_keeps_skipped(self, fixture_corpus: dict) -> None:
        rows = [_row(5126, "skipped"), _row(5126, "skipped")]
        card = scorer.compute_scorecard(fixture_corpus, rows)
        assert card.per_class["missing_test_coverage"].accepted == 0
        assert card.per_class["missing_test_coverage"].skipped == 1


class TestLoadLedger:
    def test_missing_ledger_returns_empty(self, tmp_path: Path) -> None:
        rows = scorer._load_ledger(tmp_path / "does-not-exist.jsonl")
        assert rows == []

    def test_malformed_lines_are_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "ledger.jsonl"
        p.write_text(
            "\n".join(
                [
                    json.dumps({"issue_number": 1, "worker_outcome": "dry_run:accepted"}),
                    "not-json-at-all",
                    "",
                    json.dumps({"issue_number": 2, "worker_outcome": "dry_run:dropped"}),
                ]
            )
        )
        rows = scorer._load_ledger(p)
        assert len(rows) == 2

    def test_non_dict_payloads_are_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "ledger.jsonl"
        p.write_text(
            "\n".join(
                [
                    json.dumps([1, 2, 3]),
                    json.dumps("a string"),
                    json.dumps({"issue_number": 1, "worker_outcome": "dry_run:accepted"}),
                ]
            )
        )
        rows = scorer._load_ledger(p)
        assert len(rows) == 1


class TestMarkdownRendering:
    def test_render_floor_met_includes_promotion_pointer(self, fixture_corpus: dict) -> None:
        # Bump corpus to ≥15 entries so we trip floor_met
        big_corpus = {
            "revision": 4,
            "status": "staging",
            "issues": [
                {"issue_id": 3000 + i, "execution_class": "missing_test_coverage"}
                for i in range(20)
            ],
        }
        rows = [_row(3000 + i, "accepted") for i in range(15)]
        card = scorer.compute_scorecard(big_corpus, rows)
        md = scorer.render_markdown(card)
        assert "Promotion floor is met" in md
        assert "OK:" in md
        assert "| missing_test_coverage |" in md

    def test_render_floor_pending_includes_remaining(self, fixture_corpus: dict) -> None:
        card = scorer.compute_scorecard(fixture_corpus, [])
        md = scorer.render_markdown(card)
        assert "Promotion floor not yet met" in md
        assert "Need 15 more dispatched entries" in md


class TestEndToEndWriteArtifacts:
    def test_main_writes_json_and_md(self, tmp_path: Path, capsys) -> None:
        corpus_path = tmp_path / "corpus.json"
        corpus_path.write_text(
            json.dumps(
                {
                    "revision": 4,
                    "status": "staging",
                    "issues": [
                        {"issue_id": 9001, "execution_class": "missing_test_coverage"},
                    ],
                }
            )
        )
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text(json.dumps(_row(9001, "accepted")) + "\n")
        json_out = tmp_path / "out.json"
        md_out = tmp_path / "out.md"
        rc = scorer.main(
            [
                "--corpus-path",
                str(corpus_path),
                "--ledger-path",
                str(ledger_path),
                "--scorecard-json-path",
                str(json_out),
                "--scorecard-md-path",
                str(md_out),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "h1-02 rev-4 staging scorecard:" in out
        assert json_out.exists()
        assert md_out.exists()
        payload = json.loads(json_out.read_text())
        assert payload["scorecard"]["dispatched_count"] == 1
        assert payload["scorecard"]["accepted_count"] == 1

    def test_main_with_json_flag_emits_to_stdout(self, tmp_path: Path, capsys) -> None:
        corpus_path = tmp_path / "corpus.json"
        corpus_path.write_text(
            json.dumps(
                {
                    "revision": 4,
                    "status": "staging",
                    "issues": [{"issue_id": 9002, "execution_class": "small_refactor"}],
                }
            )
        )
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text("")
        json_out = tmp_path / "out.json"
        md_out = tmp_path / "out.md"
        rc = scorer.main(
            [
                "--json",
                "--corpus-path",
                str(corpus_path),
                "--ledger-path",
                str(ledger_path),
                "--scorecard-json-path",
                str(json_out),
                "--scorecard-md-path",
                str(md_out),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert "scorecard" in payload
        assert payload["scorecard"]["total_staging"] == 1
