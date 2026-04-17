"""Tests for aragora.markets.store — JSONL persistence and invariants."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aragora.markets.store import MarketStore, MarketStoreError
from aragora.markets.types import (
    MAX_POSITION_STAKE,
    Market,
    MarketPosition,
    ResolutionEvent,
)


def _make_pr_market(*, repo: str = "owner/repo", number: int = 1) -> Market:
    return Market.create(
        question_kind="pr_merge",
        target={"repo": repo, "number": number},
        description=f"will {repo}#{number} merge",
        resolution_window_days=7,
        created_at=datetime(2026, 4, 17, tzinfo=UTC),
    )


class TestStoreLifecycle:
    def test_fresh_store_is_empty(self, tmp_path) -> None:
        store = MarketStore(tmp_path / "markets")
        assert store.list_markets() == []
        assert store.list_positions() == []
        assert store.resolutions_by_market() == {}

    def test_add_market_and_reload_persists(self, tmp_path) -> None:
        path = tmp_path / "markets"
        store_a = MarketStore(path)
        market = _make_pr_market(number=42)
        store_a.add_market(market)
        store_b = MarketStore(path)
        assert store_b.get_market(market.market_id) == market
        assert len(store_b.list_markets()) == 1

    def test_add_market_idempotent_on_identical_payload(self, tmp_path) -> None:
        store = MarketStore(tmp_path)
        market = _make_pr_market(number=1)
        store.add_market(market)
        store.add_market(market)  # second add must not raise
        assert len(store.list_markets()) == 1

    def test_add_market_rejects_id_collision_with_different_fields(self, tmp_path) -> None:
        store = MarketStore(tmp_path)
        market = _make_pr_market(number=1)
        store.add_market(market)
        # Build a divergent market with the same id by hand
        divergent = Market(
            market_id=market.market_id,
            question_kind="pr_merge",
            target={"repo": "owner/repo", "number": 1},
            description="DIVERGENT DESCRIPTION",
            created_at=market.created_at,
            expires_at=market.expires_at,
        )
        with pytest.raises(MarketStoreError):
            store.add_market(divergent)

    def test_malformed_jsonl_lines_are_skipped(self, tmp_path) -> None:
        store = MarketStore(tmp_path)
        market = _make_pr_market(number=1)
        store.add_market(market)
        # Inject a bogus line; reload must skip it
        with store.layout.markets_path.open("a") as handle:
            handle.write("NOT_JSON\n")
            handle.write('{"missing": "fields"}\n')
        store.reload()
        assert len(store.list_markets()) == 1


class TestPositionInvariants:
    def test_add_position_requires_existing_market(self, tmp_path) -> None:
        store = MarketStore(tmp_path)
        position = MarketPosition.create(
            market_id="mkt_unknown",
            agent_id="agent",
            probability=0.5,
            stake=10,
        )
        with pytest.raises(MarketStoreError):
            store.add_position(position)

    def test_add_position_blocked_after_resolution(self, tmp_path) -> None:
        store = MarketStore(tmp_path)
        market = _make_pr_market(number=1)
        store.add_market(market)
        store.record_resolution(
            ResolutionEvent.yes(
                market_id=market.market_id,
                resolution_source="github_pr_state",
            )
        )
        position = MarketPosition.create(
            market_id=market.market_id,
            agent_id="agent",
            probability=0.5,
            stake=10,
        )
        with pytest.raises(MarketStoreError):
            store.add_position(position)

    def test_position_stake_capped(self, tmp_path) -> None:
        store = MarketStore(tmp_path)
        market = _make_pr_market(number=1)
        store.add_market(market)
        # MarketPosition.create already enforces this; the store guard is
        # belt-and-suspenders for hand-built positions:
        with pytest.raises(ValueError):
            MarketPosition.create(
                market_id=market.market_id,
                agent_id="agent",
                probability=0.5,
                stake=MAX_POSITION_STAKE + 1,
            )

    def test_list_agent_positions_filters(self, tmp_path) -> None:
        store = MarketStore(tmp_path)
        market = _make_pr_market(number=1)
        store.add_market(market)
        # Distinct submitted_at per call so position_id (content-addressed)
        # does not collapse the duplicate-agent entries into one.
        entries = [
            ("alice", datetime(2026, 4, 17, 12, 0, tzinfo=UTC)),
            ("bob", datetime(2026, 4, 17, 12, 1, tzinfo=UTC)),
            ("alice", datetime(2026, 4, 17, 12, 2, tzinfo=UTC)),
        ]
        for agent, ts in entries:
            store.add_position(
                MarketPosition.create(
                    market_id=market.market_id,
                    agent_id=agent,
                    probability=0.5,
                    stake=10,
                    submitted_at=ts,
                )
            )
        alice_positions = store.list_agent_positions("alice")
        assert len(alice_positions) == 2
        assert all(pos.agent_id == "alice" for pos in alice_positions)


class TestResolutionInvariants:
    def test_record_resolution_requires_existing_market(self, tmp_path) -> None:
        store = MarketStore(tmp_path)
        with pytest.raises(MarketStoreError):
            store.record_resolution(
                ResolutionEvent.yes(
                    market_id="mkt_unknown",
                    resolution_source="github_pr_state",
                )
            )

    def test_divergent_resolution_rejected(self, tmp_path) -> None:
        store = MarketStore(tmp_path)
        market = _make_pr_market(number=1)
        store.add_market(market)
        store.record_resolution(
            ResolutionEvent.yes(
                market_id=market.market_id,
                resolution_source="github_pr_state",
                resolved_at=datetime(2026, 4, 24, tzinfo=UTC),
            )
        )
        with pytest.raises(MarketStoreError):
            store.record_resolution(
                ResolutionEvent.no(
                    market_id=market.market_id,
                    resolution_source="github_pr_state",
                    resolved_at=datetime(2026, 4, 24, tzinfo=UTC),
                )
            )

    def test_iter_unresolved_markets_excludes_resolved(self, tmp_path) -> None:
        store = MarketStore(tmp_path)
        a = _make_pr_market(number=1)
        b = _make_pr_market(number=2)
        store.add_market(a)
        store.add_market(b)
        store.record_resolution(
            ResolutionEvent.yes(
                market_id=a.market_id,
                resolution_source="github_pr_state",
            )
        )
        unresolved = list(store.iter_unresolved_markets())
        assert [m.market_id for m in unresolved] == [b.market_id]
