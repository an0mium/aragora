"""PR-keyed round budget wiring in the A1 reconciler (issue #9042, Tier 4).

Proves the acceptance criteria on a synthetic stuck cycle:
- budget is keyed by PR and survives head drift (repair commits consume it,
  never reset it);
- exhaustion flags ``needs_adjudication`` and stops re-running;
- budget 0 (the default) is byte-identical to the pre-#9042 reconciler and
  never touches the convergence ledger.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aragora.swarm.convergence_ledger import ConvergenceLedger
from aragora.swarm.merge_quorum_reconcile import EvidenceComment, QuorumRun

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reconcile_merge_quorum.py"

NOW = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def mod():
    spec = importlib.util.spec_from_file_location("reconcile_merge_quorum_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Fleet:
    """Fake GitHub surface for one churning PR."""

    def __init__(self) -> None:
        self.head = "h1"
        self.rerun_calls: list[int] = []

    def install(self, mod, monkeypatch) -> None:
        monkeypatch.setattr(
            mod,
            "fetch_pr_context",
            lambda repo, pr: {
                "head_sha": self.head,
                "head_committed_at": (NOW - timedelta(hours=2)).isoformat(),
                "has_real_required_failure": False,
            },
        )
        monkeypatch.setattr(
            mod,
            "fetch_latest_quorum_run",
            lambda repo, head_sha: QuorumRun(
                run_id=1000,
                created_at=(NOW - timedelta(hours=1)).isoformat(),
                conclusion="FAILURE",
                head_sha=head_sha,
            ),
        )
        monkeypatch.setattr(
            mod,
            "fetch_evidence_comments",
            lambda repo, pr, head_sha, committed_at: [
                EvidenceComment(
                    created_at=(NOW - timedelta(minutes=5)).isoformat(),
                    would_count=True,
                    reviewer_id="claude",
                )
            ],
        )
        # Divergence guard needs live packet fetches; identity keeps this a
        # pure plan_rerun wiring test.
        monkeypatch.setattr(
            mod,
            "guard_rerun_classification_divergence",
            lambda decision, **kwargs: decision,
        )
        monkeypatch.setattr(
            mod, "execute_rerun", lambda repo, run_id: self.rerun_calls.append(run_id) or True
        )


def _run(mod, tmp_path, *extra: str) -> list[dict]:
    out = tmp_path / "plan.json"
    argv = [
        "--repo",
        "synaptent/aragora",
        "--pr",
        "77",
        "--apply",
        "--cooldown-minutes",
        "0",
        "--max-reruns",
        "99",
        "--state-file",
        str(tmp_path / "state.json"),
        "--convergence-ledger",
        str(tmp_path / "ledger.json"),
        "--json",
        *extra,
    ]
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert mod.main(argv) == 0
    out.write_text(buf.getvalue())
    return json.loads(buf.getvalue())["plan"]


class TestSyntheticStuckCycle:
    def test_budget_survives_head_drift_and_routes_to_adjudication(
        self, mod, monkeypatch, tmp_path
    ):
        fleet = _Fleet()
        fleet.install(mod, monkeypatch)
        budget = ["--pr-round-budget", "3"]

        # Three repair rounds: each cycle the "repairer" pushes a new head.
        for round_number, head in enumerate(["h1", "h2", "h3"], start=1):
            fleet.head = head
            (record,) = _run(mod, tmp_path, *budget)
            assert record["applied"] is True, record
            assert record["pr_rounds_consumed"] == round_number - 1
            assert "needs_adjudication" not in record

        ledger = ConvergenceLedger(tmp_path / "ledger.json")
        assert ledger.rounds(77) == 3  # distinct heads counted, never reset

        # Fourth repair commit: budget exhausted -> adjudication, no rerun.
        fleet.head = "h4"
        rerun_count_before = len(fleet.rerun_calls)
        (record,) = _run(mod, tmp_path, *budget)
        assert record["needs_adjudication"] is True
        assert record["should_rerun"] is False
        assert record["applied"] is False
        assert len(fleet.rerun_calls) == rerun_count_before
        assert "net-value adjudication" in record["reason"]

    def test_same_head_rerun_does_not_consume_budget(self, mod, monkeypatch, tmp_path):
        fleet = _Fleet()
        fleet.install(mod, monkeypatch)
        budget = ["--pr-round-budget", "3", "--max-reruns", "99"]
        fleet.head = "h1"
        _run(mod, tmp_path, *budget)
        _run(mod, tmp_path, *budget)  # same head again
        assert ConvergenceLedger(tmp_path / "ledger.json").rounds(77) == 1

    def test_recorded_adjudication_surfaces_in_plan(self, mod, monkeypatch, tmp_path):
        fleet = _Fleet()
        fleet.install(mod, monkeypatch)
        ledger = ConvergenceLedger(tmp_path / "ledger.json")
        for head in ["h1", "h2", "h3"]:
            ledger.record_round(77, head)
        ledger.record_adjudication(77, verdict="close", rationale="net-negative churn")
        (record,) = _run(mod, tmp_path, "--pr-round-budget", "3")
        assert record["needs_adjudication"] is True
        assert record["adjudication"]["verdict"] == "close"


class TestBudgetDisabledByDefault:
    def test_default_off_is_behavior_identical_and_ledger_untouched(
        self, mod, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("ARAGORA_PR_ROUND_BUDGET", raising=False)
        fleet = _Fleet()
        fleet.install(mod, monkeypatch)
        for head in ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"]:
            fleet.head = head
            (record,) = _run(mod, tmp_path)
            assert record["applied"] is True
            assert "needs_adjudication" not in record
            assert "pr_rounds_consumed" not in record
        assert not (tmp_path / "ledger.json").exists()

    def test_env_default_feeds_parser(self, mod, monkeypatch):
        monkeypatch.setenv("ARAGORA_PR_ROUND_BUDGET", "6")
        assert mod._default_pr_round_budget() == 6
        monkeypatch.setenv("ARAGORA_PR_ROUND_BUDGET", "garbage")
        assert mod._default_pr_round_budget() == 0
        monkeypatch.setenv("ARAGORA_PR_ROUND_BUDGET", "-3")
        assert mod._default_pr_round_budget() == 0
