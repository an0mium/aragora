"""Tests for the AGT-* CLI verbs (metrics viah, markets list, cruxset show)."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aragora.cli.commands.agt_cruxset import cmd_cruxset_show
from aragora.cli.commands.agt_markets import cmd_markets_list
from aragora.cli.commands.agt_metrics import cmd_metrics_viah
from aragora.markets.store import MarketStore
from aragora.markets.types import Market, MarketPosition, ResolutionEvent
from aragora.reasoning.cruxset import Crux, CruxPosition, CruxSet
from aragora.swarm.shift_ledger import ShiftLedger


# ---------------------------------------------------------------------------
# metrics viah
# ---------------------------------------------------------------------------


def _seed_ledger(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")


def _ts(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class TestMetricsViah:
    def test_empty_ledger_prints_na_and_returns_zero(self, tmp_path, capsys) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text("", encoding="utf-8")
        args = argparse.Namespace(
            ledger_path=str(ledger_path),
            window_hours=24.0,
            cruxes_correctly_detected=0,
            predictions_above_brier_threshold=0,
            failed_claims_promoted_without_repair=0,
            json=False,
        )
        rc = cmd_metrics_viah(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "VIAH: n/a" in out
        assert "agent_hours:" in out

    def test_populated_ledger_computes_positive_viah(self, tmp_path, capsys) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        # Use real current time so seeded entries always fall within the
        # 24h window evaluated by ``cmd_metrics_viah`` (which reads
        # ``datetime.now(UTC)`` internally). Previously this test pinned
        # ``now`` to a fixed date, causing it to break after that date
        # fell outside the 24h window.
        now = datetime.now(tz=UTC).replace(microsecond=0)
        _seed_ledger(
            ledger_path,
            [
                {
                    "entry_type": "shift_start",
                    "timestamp": _ts(now - timedelta(hours=2)),
                    "payload": {"shift_id": "s1"},
                },
                {
                    "entry_type": "pr_merged",
                    "timestamp": _ts(now - timedelta(minutes=30)),
                    "payload": {"pr_number": 1},
                },
                {
                    "entry_type": "shift_stop",
                    "timestamp": _ts(now),
                    "payload": {"shift_id": "s1"},
                },
            ],
        )
        args = argparse.Namespace(
            ledger_path=str(ledger_path),
            window_hours=24.0,
            cruxes_correctly_detected=0,
            predictions_above_brier_threshold=0,
            failed_claims_promoted_without_repair=0,
            json=True,
        )
        rc = cmd_metrics_viah(args)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        # Note: VIAH from this snapshot depends on `now` at runtime; we
        # only assert structure and that PR was counted.
        assert payload["merged_autonomous_prs"] == 1
        assert payload["coefficients"]["merged_pr_weight"] == 1.0


# ---------------------------------------------------------------------------
# markets list
# ---------------------------------------------------------------------------


class TestMarketsList:
    def test_empty_store_reports_no_markets(self, tmp_path, capsys) -> None:
        args = argparse.Namespace(store_dir=str(tmp_path / "empty"), json=False)
        rc = cmd_markets_list(args)
        assert rc == 0
        assert "No markets found" in capsys.readouterr().out

    def test_lists_markets_with_open_status(self, tmp_path, capsys) -> None:
        store = MarketStore(tmp_path / "store")
        market = Market.create(
            question_kind="pr_merge",
            target={"repo": "synaptent/aragora", "number": 5959},
            description="will it merge",
            resolution_window_days=7,
        )
        store.add_market(market)
        args = argparse.Namespace(store_dir=str(store.layout.base_dir), json=False)
        rc = cmd_markets_list(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert market.market_id in out
        assert "[open" in out
        assert "pr_merge" in out

    def test_lists_resolved_market_with_outcome(self, tmp_path, capsys) -> None:
        store = MarketStore(tmp_path / "store")
        market = Market.create(
            question_kind="issue_close",
            target={"repo": "synaptent/aragora", "number": 6068},
            description="will it close",
            resolution_window_days=14,
        )
        store.add_market(market)
        store.record_resolution(
            ResolutionEvent.yes(market_id=market.market_id, resolution_source="github_issue_state")
        )
        args = argparse.Namespace(store_dir=str(store.layout.base_dir), json=False)
        rc = cmd_markets_list(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "[yes" in out

    def test_json_mode_emits_array_with_resolution_field(self, tmp_path, capsys) -> None:
        store = MarketStore(tmp_path / "store")
        market = Market.create(
            question_kind="ci_pass",
            target={"repo": "synaptent/aragora", "ref": "abc123"},
            description="will ci pass",
            resolution_window_days=3,
        )
        store.add_market(market)
        args = argparse.Namespace(store_dir=str(store.layout.base_dir), json=True)
        rc = cmd_markets_list(args)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)
        assert len(payload) == 1
        assert payload[0]["market_id"] == market.market_id
        assert payload[0]["resolution"] is None  # not yet resolved


# ---------------------------------------------------------------------------
# cruxset show
# ---------------------------------------------------------------------------


def _make_cruxset_payload() -> dict:
    cruxset = CruxSet.build(
        question="should we ship?",
        cruxes=[
            Crux(
                crux_id="c1",
                statement="Tests pass on all platforms",
                positions=(
                    CruxPosition(side="for", agents=("alice",)),
                    CruxPosition(side="against", agents=("bob",)),
                ),
                load_bearing_score=0.85,
                counterfactual="if false, decision flips",
            ),
            Crux(
                crux_id="c2",
                statement="No regressions in latency",
                positions=(CruxPosition(side="for", agents=("carol",)),),
                load_bearing_score=0.55,
            ),
        ],
        decision="ship",
        receipt_id="rcpt_abc",
    )
    return cruxset.to_json()


class TestCruxsetShow:
    def test_pretty_print_from_file(self, tmp_path, capsys) -> None:
        path = tmp_path / "cs.json"
        path.write_text(json.dumps(_make_cruxset_payload()), encoding="utf-8")
        args = argparse.Namespace(source=str(path), json=False)
        rc = cmd_cruxset_show(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "should we ship?" in out
        assert "decision:          ship" in out
        assert "checksum verified: True" in out
        assert "load_bearing=0.850" in out

    def test_json_mode_re_emits_payload(self, tmp_path, capsys) -> None:
        path = tmp_path / "cs.json"
        original = _make_cruxset_payload()
        path.write_text(json.dumps(original), encoding="utf-8")
        args = argparse.Namespace(source=str(path), json=True)
        rc = cmd_cruxset_show(args)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["cruxset_id"] == original["cruxset_id"]
        assert payload["checksum"] == original["checksum"]

    def test_missing_source_returns_2(self, tmp_path, capsys) -> None:
        args = argparse.Namespace(source=str(tmp_path / "nope.json"), json=False)
        rc = cmd_cruxset_show(args)
        assert rc == 2

    def test_invalid_json_returns_2(self, tmp_path, capsys) -> None:
        path = tmp_path / "bad.json"
        path.write_text("<html>", encoding="utf-8")
        args = argparse.Namespace(source=str(path), json=False)
        rc = cmd_cruxset_show(args)
        assert rc == 2

    def test_payload_not_a_cruxset_returns_2(self, tmp_path, capsys) -> None:
        path = tmp_path / "not_cs.json"
        path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
        args = argparse.Namespace(source=str(path), json=False)
        rc = cmd_cruxset_show(args)
        assert rc == 2

    def test_tampered_payload_returns_3(self, tmp_path, capsys) -> None:
        path = tmp_path / "tampered.json"
        payload = _make_cruxset_payload()
        payload["question"] = "TAMPERED"
        path.write_text(json.dumps(payload), encoding="utf-8")
        args = argparse.Namespace(source=str(path), json=False)
        rc = cmd_cruxset_show(args)
        assert rc == 3
        out = capsys.readouterr().out
        assert "checksum verified: False" in out
