"""Tests for the AGT-03.3 calibration consumer CLI."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aragora.cli.commands.agt_calibration import (
    cmd_calibration_leaderboard,
    cmd_calibration_report,
)
from aragora.markets.store import MarketStore
from aragora.markets.types import Market, MarketPosition, ResolutionEvent


def _make_args(**kwargs) -> argparse.Namespace:
    """Build an argparse.Namespace with sensible defaults for the calibration CLI."""
    defaults: dict = {
        "store_dir": kwargs["store_dir"],
        "agent": None,
        "window_days": 90.0,
        "half_life_days": 30.0,
        "min_scored": 5,
        "sort_by": "decayed",
        "json": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


_market_counter = 0


def _seed_position_pair(
    store: MarketStore,
    *,
    agent_id: str,
    probability: float,
    yes_outcome: bool,
    resolved_at: datetime,
    stake: int = 10,
) -> tuple[Market, MarketPosition, ResolutionEvent]:
    """Seed one market + one position + one resolution into the store.

    Uses a monotonic counter so repeated calls always produce a fresh
    market_id (Market.market_id is deterministically derived from the
    target dict).
    """
    global _market_counter
    _market_counter += 1
    market = Market.create(
        question_kind="ci_pass",
        target={
            "repo": "synaptent/aragora",
            "ref": f"sha_{agent_id}_{_market_counter}_{int(probability * 1000)}",
        },
        description="will ci pass",
        resolution_window_days=14,
    )
    store.add_market(market)
    position = MarketPosition.create(
        market_id=market.market_id,
        agent_id=agent_id,
        probability=probability,
        stake=stake,
    )
    store.add_position(position)
    if yes_outcome:
        event = ResolutionEvent.yes(
            market_id=market.market_id,
            resolution_source="github_ci",
            resolved_at=resolved_at,
        )
    else:
        event = ResolutionEvent.no(
            market_id=market.market_id,
            resolution_source="github_ci",
            resolved_at=resolved_at,
        )
    store.record_resolution(event)
    return market, position, event


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class TestCalibrationReport:
    def test_empty_store_prints_no_positions_and_returns_zero(self, tmp_path: Path, capsys) -> None:
        store = MarketStore(tmp_path / "store")
        store.list_markets()  # force layout init
        args = _make_args(store_dir=str(store.layout.base_dir))
        rc = cmd_calibration_report(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "No positions found" in out

    def test_single_well_calibrated_agent(self, tmp_path: Path, capsys) -> None:
        """A perfectly-confident correct prediction yields Brier 0."""
        store = MarketStore(tmp_path / "store")
        now = datetime.now(tz=UTC)
        _seed_position_pair(
            store,
            agent_id="oracle",
            probability=1.0,
            yes_outcome=True,
            resolved_at=now - timedelta(days=10),
        )
        args = _make_args(store_dir=str(store.layout.base_dir))
        rc = cmd_calibration_report(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "oracle" in out
        # Mean Brier should be 0.0000 for a P=1, outcome=YES position.
        assert "0.0000" in out

    def test_single_miscalibrated_agent_brier_1(self, tmp_path: Path, capsys) -> None:
        """A perfectly-wrong prediction yields Brier 1."""
        store = MarketStore(tmp_path / "store")
        now = datetime.now(tz=UTC)
        _seed_position_pair(
            store,
            agent_id="badcaller",
            probability=1.0,
            yes_outcome=False,  # predicted YES, got NO
            resolved_at=now - timedelta(days=5),
        )
        args = _make_args(store_dir=str(store.layout.base_dir))
        rc = cmd_calibration_report(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "badcaller" in out
        assert "1.0000" in out

    def test_window_filter_excludes_old_resolutions(self, tmp_path: Path, capsys) -> None:
        """A resolution older than the window should not appear in the report."""
        store = MarketStore(tmp_path / "store")
        now = datetime.now(tz=UTC)
        _seed_position_pair(
            store,
            agent_id="ancient",
            probability=0.5,
            yes_outcome=True,
            resolved_at=now - timedelta(days=180),  # well outside 90d window
        )
        args = _make_args(store_dir=str(store.layout.base_dir), window_days=90.0)
        rc = cmd_calibration_report(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "No positions found" in out

    def test_agent_filter_restricts_output(self, tmp_path: Path, capsys) -> None:
        store = MarketStore(tmp_path / "store")
        now = datetime.now(tz=UTC)
        _seed_position_pair(
            store,
            agent_id="alice",
            probability=0.7,
            yes_outcome=True,
            resolved_at=now - timedelta(days=2),
        )
        _seed_position_pair(
            store,
            agent_id="bob",
            probability=0.3,
            yes_outcome=False,
            resolved_at=now - timedelta(days=2),
        )
        args = _make_args(store_dir=str(store.layout.base_dir), agent="alice")
        rc = cmd_calibration_report(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "alice" in out
        assert "bob" not in out

    def test_json_mode_emits_full_breakdown(self, tmp_path: Path, capsys) -> None:
        store = MarketStore(tmp_path / "store")
        now = datetime.now(tz=UTC)
        _seed_position_pair(
            store,
            agent_id="alice",
            probability=0.6,
            yes_outcome=True,
            resolved_at=now - timedelta(days=1),
        )
        args = _make_args(store_dir=str(store.layout.base_dir), json=True)
        rc = cmd_calibration_report(args)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["window_days"] == 90.0
        assert payload["half_life_days"] == 30.0
        assert len(payload["agents"]) == 1
        agent_record = payload["agents"][0]
        assert agent_record["agent_id"] == "alice"
        assert agent_record["scored_positions"] == 1
        # P(YES)=0.6, outcome=YES → Brier = (0.6 - 1.0)^2 = 0.16
        assert agent_record["mean_brier"] == pytest.approx(0.16, abs=1e-6)


# ---------------------------------------------------------------------------
# leaderboard
# ---------------------------------------------------------------------------


class TestCalibrationLeaderboard:
    def test_floor_excludes_low_sample_agents(self, tmp_path: Path, capsys) -> None:
        """min_scored=5 should hide agents with fewer scored positions."""
        store = MarketStore(tmp_path / "store")
        now = datetime.now(tz=UTC)
        # Single position only — below the default floor of 5.
        _seed_position_pair(
            store,
            agent_id="rookie",
            probability=0.5,
            yes_outcome=True,
            resolved_at=now - timedelta(days=1),
        )
        args = _make_args(store_dir=str(store.layout.base_dir), min_scored=5)
        rc = cmd_calibration_leaderboard(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "No agents meet the minimum-scored floor" in out

    def test_ranks_eligible_agents_ascending_by_decayed_brier(self, tmp_path: Path, capsys) -> None:
        """Better-calibrated agents (lower Brier) appear first."""
        store = MarketStore(tmp_path / "store")
        now = datetime.now(tz=UTC)
        # Seed 5 perfect predictions for "oracle" → Brier 0.
        for i in range(5):
            _seed_position_pair(
                store,
                agent_id="oracle",
                probability=1.0,
                yes_outcome=True,
                resolved_at=now - timedelta(days=i + 1),
            )
        # Seed 5 maximally-wrong predictions for "wrongbot" → Brier 1.
        for i in range(5):
            _seed_position_pair(
                store,
                agent_id="wrongbot",
                probability=1.0,
                yes_outcome=False,
                resolved_at=now - timedelta(days=i + 1),
            )
        args = _make_args(store_dir=str(store.layout.base_dir), min_scored=5)
        rc = cmd_calibration_leaderboard(args)
        assert rc == 0
        out = capsys.readouterr().out
        oracle_pos = out.find("oracle")
        wrongbot_pos = out.find("wrongbot")
        assert oracle_pos != -1
        assert wrongbot_pos != -1
        assert oracle_pos < wrongbot_pos  # oracle ranks first

    def test_invalid_sort_by_returns_2(self, tmp_path: Path, capsys) -> None:
        store = MarketStore(tmp_path / "store")
        store.list_markets()
        args = _make_args(store_dir=str(store.layout.base_dir), sort_by="bogus")
        rc = cmd_calibration_leaderboard(args)
        assert rc == 2
        out = capsys.readouterr().out
        assert "Invalid --sort-by" in out

    def test_json_includes_excluded_below_floor(self, tmp_path: Path, capsys) -> None:
        """Agents below the floor must still be visible in the JSON output."""
        store = MarketStore(tmp_path / "store")
        now = datetime.now(tz=UTC)
        _seed_position_pair(
            store,
            agent_id="rookie",
            probability=0.5,
            yes_outcome=True,
            resolved_at=now - timedelta(days=1),
        )
        args = _make_args(store_dir=str(store.layout.base_dir), min_scored=5, json=True)
        rc = cmd_calibration_leaderboard(args)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["leaderboard"] == []
        assert len(payload["excluded_below_floor"]) == 1
        assert payload["excluded_below_floor"][0]["agent_id"] == "rookie"
