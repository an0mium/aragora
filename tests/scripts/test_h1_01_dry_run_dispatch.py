"""Unit tests for ``scripts/h1_01_dry_run_dispatch.py``.

The dispatcher is the H1-01 recorded-evidence producer. It must:

1. Tolerate gh-offline environments without crashing (so CI without
   GH_TOKEN, and contributors without authenticated gh, can still run it).
2. Always record a row per dispatched issue, even on fetch failure, so
   the dispatch ledger reflects every staging entry regardless of network.
3. Namespace every recorded row with ``terminal_class="dry_run_*"`` and
   ``worker_status="dry_run"`` so downstream consumers cannot confuse
   dry-run evidence with real boss-loop terminal classes.
4. Idempotently overwrite the per-issue summary on re-run while
   appending to the dispatch ledger.

These tests run in pure offline mode (``--offline``) so they never call
``gh``. The sanitizer is the real ``aragora.swarm.task_sanitizer``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import h1_01_dry_run_dispatch as dispatcher  # noqa: E402


@pytest.fixture()
def tmp_ledger(tmp_path: Path) -> Path:
    return tmp_path / "boss_metrics_h1_01_dry_run.jsonl"


@pytest.fixture()
def fake_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Replace the real rev-4 corpus path with a tiny fixture corpus."""
    corpus_path = tmp_path / "corpus_rev4.json"
    corpus_path.write_text(
        json.dumps(
            {
                "revision": 4,
                "status": "staging",
                "issues": [
                    {
                        "issue_id": 5126,
                        "title": "fixture: missing test coverage",
                        "execution_class": "missing_test_coverage",
                    },
                    {
                        "issue_id": 5788,
                        "title": "fixture: exception narrowing",
                        "execution_class": "exception_narrowing",
                    },
                    {
                        "issue_id": 5808,
                        "title": "fixture: silent exception replacement",
                        "execution_class": "silent_exception_replacement",
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(dispatcher, "CORPUS_REV4_PATH", corpus_path)
    return corpus_path


@pytest.fixture()
def tmp_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    summary_json = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"
    monkeypatch.setattr(dispatcher, "SUMMARY_JSON", summary_json)
    monkeypatch.setattr(dispatcher, "SUMMARY_MD", summary_md)
    return summary_json, summary_md


class TestDispatcherOfflineMode:
    def test_offline_records_one_row_per_issue(
        self,
        fake_corpus: Path,
        tmp_summary: tuple[Path, Path],
        tmp_ledger: Path,
    ) -> None:
        rc = dispatcher.main(["--offline", "--ledger-path", str(tmp_ledger)])
        assert rc == 0
        assert tmp_ledger.exists()
        rows = [json.loads(line) for line in tmp_ledger.read_text().splitlines() if line.strip()]
        assert len(rows) == 3
        # Every row carries the dry-run namespacing the test contract requires.
        for row in rows:
            assert row["worker_status"] == "dry_run"
            assert str(row["worker_outcome"]).startswith("dry_run:")
            assert str(row["terminal_class"]).startswith("dry_run_")
            assert row["publish_action"] == "dry_run_no_publish"
            assert row["cohort_tag"] == "h1-01-dry-run-evidence"

    def test_offline_skips_have_skipped_terminal_class(
        self,
        fake_corpus: Path,
        tmp_summary: tuple[Path, Path],
        tmp_ledger: Path,
    ) -> None:
        # In --offline mode every fetch is marked offline:requested so the
        # row terminal_class must be dry_run_skipped and worker_outcome
        # must be the matching dry_run:skipped string.
        dispatcher.main(["--offline", "--ledger-path", str(tmp_ledger)])
        rows = [json.loads(line) for line in tmp_ledger.read_text().splitlines() if line.strip()]
        for row in rows:
            assert row["terminal_class"] == "dry_run_skipped"
            assert row["worker_outcome"] == "dry_run:skipped"
            assert row["failure_reason"] == "offline:requested"

    def test_offline_with_limit_dispatches_subset(
        self,
        fake_corpus: Path,
        tmp_summary: tuple[Path, Path],
        tmp_ledger: Path,
    ) -> None:
        dispatcher.main(["--offline", "--limit", "2", "--ledger-path", str(tmp_ledger)])
        rows = [json.loads(line) for line in tmp_ledger.read_text().splitlines() if line.strip()]
        assert len(rows) == 2

    def test_summary_json_has_aggregate_keys(
        self,
        fake_corpus: Path,
        tmp_summary: tuple[Path, Path],
        tmp_ledger: Path,
    ) -> None:
        summary_json, _ = tmp_summary
        dispatcher.main(["--offline", "--ledger-path", str(tmp_ledger)])
        payload: dict[str, Any] = json.loads(summary_json.read_text())
        assert "summary" in payload
        assert payload["summary"]["total"] == 3
        assert payload["summary"]["fetch_ok"] == 0  # offline: no fetches succeed
        # Every row preserves its execution_class.
        rows = payload["rows"]
        classes = {r["execution_class"] for r in rows}
        assert classes == {
            "missing_test_coverage",
            "exception_narrowing",
            "silent_exception_replacement",
        }

    def test_summary_markdown_emits_per_class_breakdown(
        self,
        fake_corpus: Path,
        tmp_summary: tuple[Path, Path],
        tmp_ledger: Path,
    ) -> None:
        _, summary_md = tmp_summary
        dispatcher.main(["--offline", "--ledger-path", str(tmp_ledger)])
        text = summary_md.read_text()
        assert "# H1-01 dry-run dispatch summary" in text
        assert "missing_test_coverage" in text
        assert "exception_narrowing" in text
        assert "| #5126 |" in text


class TestClassifyTerminal:
    @pytest.mark.parametrize(
        "outcome,expected",
        [
            ("accepted", "dry_run_accepted"),
            ("rewritten", "dry_run_rewritten"),
            ("dropped", "dry_run_dropped"),
            ("quarantined", "dry_run_quarantined"),
            ("anything_else", "dry_run_unknown"),
        ],
    )
    def test_classify_maps_known_outcomes(self, outcome: str, expected: str) -> None:
        assert dispatcher._classify_terminal(outcome) == expected


class TestRowForIssueOffline:
    def test_offline_row_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # _row_for_issue with offline=True must never invoke the sanitizer
        # because there is nothing to sanitize. Asserting that by tripping
        # an explicit fail in the sanitizer constructor.
        original = dispatcher._row_for_issue
        from aragora.swarm import task_sanitizer as ts

        sentinel = {"called": False}

        class _BoomSanitizer(ts.TaskSanitizer):
            def __init__(self, *args, **kwargs) -> None:
                sentinel["called"] = True
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(ts, "TaskSanitizer", _BoomSanitizer)
        row = original(9999, "missing_test_coverage", offline=True)
        assert row.fetch_status == "offline:requested"
        assert row.terminal_class == "dry_run_skipped"
        assert sentinel["called"] is False


class TestDirectScriptInvocation:
    def test_online_mode_imports_repo_package_without_pythonpath(self, tmp_path: Path) -> None:
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_gh = fake_bin / "gh"
        fake_gh.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "import json",
                    "print(json.dumps({",
                    "    'title': 'fixture: direct invocation sanitizer import',",
                    "    'body': 'Please add a focused regression test for this bounded bug.',",
                    "    'state': 'OPEN',",
                    "}))",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
        env["ARAGORA_H1_01_SUMMARY_JSON"] = str(tmp_path / "summary.json")
        env["ARAGORA_H1_01_SUMMARY_MD"] = str(tmp_path / "summary.md")

        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "h1_01_dry_run_dispatch.py"),
                "--limit",
                "1",
                "--ledger-path",
                str(tmp_path / "ledger.jsonl"),
                "--json",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["summary"]["total"] == 1
        assert payload["summary"]["fetch_ok"] == 1
        assert (tmp_path / "summary.json").is_file()
        assert (tmp_path / "summary.md").is_file()


class TestPersistMetricsRowAppends:
    def test_two_invocations_append(self, tmp_ledger: Path) -> None:
        # Ledger is append-only by design; idempotency lives in the
        # per-issue summary, not the ledger.
        row = dispatcher.DispatchRow(
            issue_id=42,
            title="t",
            execution_class="missing_test_coverage",
            sanitizer_outcome="accepted",
            sanitizer_reason="",
            sanitizer_checks_failed=[],
            sanitizer_confidence=0.98,
            body_chars=10,
            sanitized_chars=10,
            fetch_status="ok",
            terminal_class="dry_run_accepted",
        )
        dispatcher._persist_metrics_row(row, ledger_path=tmp_ledger)
        dispatcher._persist_metrics_row(row, ledger_path=tmp_ledger)
        rows = [json.loads(line) for line in tmp_ledger.read_text().splitlines() if line.strip()]
        assert len(rows) == 2


class TestStagingFreshnessInvariant:
    """The parallel freshness invariant for the rev-4 staging corpus.

    The canonical rev-3 invariant in ``tests/benchmarks/test_corpus_freshness.py``
    measures ``docs/benchmarks/corpus.json``. This class measures
    ``tests/benchmarks/corpus_rev4.json`` against the dry-run dispatch
    ledger. Once promotion occurs, the canonical invariant fires; until
    then, this invariant gives an early signal.
    """

    def test_dry_run_evidence_aggregates_per_issue(self, tmp_ledger: Path) -> None:
        for issue_id in (5126, 5788, 5808):
            row = dispatcher.DispatchRow(
                issue_id=issue_id,
                title="t",
                execution_class="missing_test_coverage",
                sanitizer_outcome="accepted",
                sanitizer_reason="",
                sanitizer_checks_failed=[],
                sanitizer_confidence=0.9,
                body_chars=200,
                sanitized_chars=200,
                fetch_status="ok",
                terminal_class="dry_run_accepted",
            )
            dispatcher._persist_metrics_row(row, ledger_path=tmp_ledger)

        # Use the canonical aggregator from test_corpus_freshness so any
        # future change in that aggregator is automatically reflected in
        # this parallel invariant.
        sys.path.insert(0, str(REPO_ROOT / "tests" / "benchmarks"))
        from test_corpus_freshness import load_dispatch_outcomes, load_metrics_rows

        rows = load_metrics_rows(tmp_ledger)
        outcomes = load_dispatch_outcomes(rows)
        assert set(outcomes.keys()) == {5126, 5788, 5808}
        for outcome_list in outcomes.values():
            assert outcome_list
            assert all(o.startswith("dry_run:") for o in outcome_list)
