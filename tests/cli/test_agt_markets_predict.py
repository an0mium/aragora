"""Tests for ``aragora markets predict`` CLI verb (AGT-04, issue #6065).

All tests are fully hermetic: they use a ``tmp_path`` store and deterministic
market/position fixtures.  No network calls, no live dispatch, no queue
mutations.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import pytest

from aragora.cli.commands.agt_markets import cmd_markets_predict
from aragora.markets.store import MarketStore
from aragora.markets.types import Market, ResolutionEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_market(tmp_path, *, number: int = 42) -> Market:
    store = MarketStore(tmp_path)
    market = Market.create(
        question_kind="pr_merge",
        target={"repo": "synaptent/aragora", "number": number},
        description=f"will PR #{number} merge",
        resolution_window_days=30,
        created_at=datetime(2026, 4, 17, tzinfo=UTC),
    )
    store.add_market(market)
    return market


def _args(
    *,
    store_dir,
    market_id: str,
    agent: str = "claude-sonnet",
    probability: float = 0.75,
    stake: int = 10,
    rationale: str = "",
    emit_json: bool = False,
) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.store_dir = str(store_dir)
    ns.market_id = market_id
    ns.agent = agent
    ns.probability = probability
    ns.stake = stake
    ns.rationale = rationale
    ns.json = emit_json
    return ns


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestPredictHappyPath:
    def test_records_position_in_store(self, tmp_path) -> None:
        market = _make_market(tmp_path)
        args = _args(store_dir=tmp_path, market_id=market.market_id)
        rc = cmd_markets_predict(args)
        assert rc == 0
        store = MarketStore(tmp_path)
        positions = store.list_positions(market_id=market.market_id)
        assert len(positions) == 1
        pos = positions[0]
        assert pos.agent_id == "claude-sonnet"
        assert pos.probability == pytest.approx(0.75)
        assert pos.stake == 10

    def test_json_output_roundtrips(self, tmp_path, capsys) -> None:
        market = _make_market(tmp_path)
        args = _args(store_dir=tmp_path, market_id=market.market_id, emit_json=True)
        rc = cmd_markets_predict(args)
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["market_id"] == market.market_id
        assert payload["agent_id"] == "claude-sonnet"
        assert payload["probability"] == pytest.approx(0.75)
        assert payload["stake"] == 10

    def test_rationale_preserved(self, tmp_path) -> None:
        market = _make_market(tmp_path)
        args = _args(
            store_dir=tmp_path,
            market_id=market.market_id,
            rationale="CI is green and two approvals are in",
        )
        rc = cmd_markets_predict(args)
        assert rc == 0
        store = MarketStore(tmp_path)
        pos = store.list_positions(market_id=market.market_id)[0]
        assert pos.rationale == "CI is green and two approvals are in"

    def test_multiple_agents_can_predict_same_market(self, tmp_path) -> None:
        market = _make_market(tmp_path)
        for agent, prob in [("agent-a", 0.6), ("agent-b", 0.9)]:
            rc = cmd_markets_predict(
                _args(store_dir=tmp_path, market_id=market.market_id, agent=agent, probability=prob)
            )
            assert rc == 0
        store = MarketStore(tmp_path)
        positions = store.list_positions(market_id=market.market_id)
        assert len(positions) == 2
        agents = {p.agent_id for p in positions}
        assert agents == {"agent-a", "agent-b"}


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


class TestPredictErrorPath:
    def test_unknown_market_returns_nonzero(self, tmp_path) -> None:
        args = _args(store_dir=tmp_path, market_id="mkt_nonexistent_abc123")
        rc = cmd_markets_predict(args)
        assert rc != 0

    def test_resolved_market_returns_nonzero(self, tmp_path) -> None:
        market = _make_market(tmp_path)
        store = MarketStore(tmp_path)
        store.record_resolution(
            ResolutionEvent.yes(
                market_id=market.market_id,
                resolution_source="test",
                evidence={"merged": True},
            )
        )
        args = _args(store_dir=tmp_path, market_id=market.market_id)
        rc = cmd_markets_predict(args)
        assert rc != 0

    def test_probability_out_of_range_returns_nonzero(self, tmp_path) -> None:
        market = _make_market(tmp_path)
        args = _args(store_dir=tmp_path, market_id=market.market_id, probability=1.5)
        rc = cmd_markets_predict(args)
        assert rc != 0

    def test_stake_exceeds_cap_returns_nonzero(self, tmp_path) -> None:
        market = _make_market(tmp_path)
        args = _args(store_dir=tmp_path, market_id=market.market_id, stake=101)
        rc = cmd_markets_predict(args)
        assert rc != 0

    def test_zero_stake_returns_nonzero(self, tmp_path) -> None:
        market = _make_market(tmp_path)
        args = _args(store_dir=tmp_path, market_id=market.market_id, stake=0)
        rc = cmd_markets_predict(args)
        assert rc != 0

    def test_error_message_to_stderr(self, tmp_path, capsys) -> None:
        args = _args(store_dir=tmp_path, market_id="mkt_missing_xyz")
        cmd_markets_predict(args)
        err = capsys.readouterr().err
        assert "mkt_missing_xyz" in err
