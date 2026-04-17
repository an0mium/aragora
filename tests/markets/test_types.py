"""Tests for aragora.markets.types — schema validation and roundtrip."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aragora.markets.types import (
    MAX_POSITION_STAKE,
    MAX_RESOLUTION_WINDOW_DAYS,
    Market,
    MarketPosition,
    ResolutionEvent,
)


class TestMarketCreate:
    def test_pr_merge_market_id_is_content_addressed(self) -> None:
        a = Market.create(
            question_kind="pr_merge",
            target={"repo": "synaptent/aragora", "number": 5959},
            description="will #5959 merge",
            resolution_window_days=7,
            created_at=datetime(2026, 4, 17, tzinfo=UTC),
        )
        b = Market.create(
            question_kind="pr_merge",
            target={"repo": "synaptent/aragora", "number": 5959},
            description="duplicate",
            resolution_window_days=14,  # different window must not change id
            created_at=datetime(2026, 4, 18, tzinfo=UTC),
        )
        assert a.market_id == b.market_id

    def test_distinct_pr_targets_have_distinct_ids(self) -> None:
        a = Market.create(
            question_kind="pr_merge",
            target={"repo": "synaptent/aragora", "number": 1},
            description="a",
            resolution_window_days=7,
        )
        b = Market.create(
            question_kind="pr_merge",
            target={"repo": "synaptent/aragora", "number": 2},
            description="b",
            resolution_window_days=7,
        )
        assert a.market_id != b.market_id

    def test_invalid_repo_rejected(self) -> None:
        with pytest.raises(ValueError):
            Market.create(
                question_kind="pr_merge",
                target={"repo": "no-slash", "number": 1},
                description="x",
                resolution_window_days=7,
            )

    def test_pr_merge_requires_positive_int_number(self) -> None:
        with pytest.raises(ValueError):
            Market.create(
                question_kind="pr_merge",
                target={"repo": "owner/repo", "number": "five"},
                description="x",
                resolution_window_days=7,
            )

    def test_ci_pass_requires_ref(self) -> None:
        with pytest.raises(ValueError):
            Market.create(
                question_kind="ci_pass",
                target={"repo": "owner/repo"},
                description="x",
                resolution_window_days=3,
            )

    def test_window_bounds(self) -> None:
        with pytest.raises(ValueError):
            Market.create(
                question_kind="pr_merge",
                target={"repo": "owner/repo", "number": 1},
                description="x",
                resolution_window_days=0,
            )
        with pytest.raises(ValueError):
            Market.create(
                question_kind="pr_merge",
                target={"repo": "owner/repo", "number": 1},
                description="x",
                resolution_window_days=MAX_RESOLUTION_WINDOW_DAYS + 1,
            )

    def test_is_expired_uses_utc(self) -> None:
        created = datetime(2026, 4, 17, tzinfo=UTC)
        market = Market.create(
            question_kind="pr_merge",
            target={"repo": "owner/repo", "number": 1},
            description="x",
            resolution_window_days=7,
            created_at=created,
        )
        assert not market.is_expired(now=created + timedelta(days=6))
        assert market.is_expired(now=created + timedelta(days=7, seconds=1))

    def test_to_json_roundtrip(self) -> None:
        market = Market.create(
            question_kind="ci_pass",
            target={"repo": "owner/repo", "ref": "abc123"},
            description="will branch ci pass",
            resolution_window_days=3,
        )
        roundtrip = Market.from_json(market.to_json())
        assert roundtrip == market


class TestMarketPosition:
    def test_create_validates_probability_bounds(self) -> None:
        with pytest.raises(ValueError):
            MarketPosition.create(
                market_id="mkt_pr_merge_x", agent_id="agent", probability=-0.1, stake=10
            )
        with pytest.raises(ValueError):
            MarketPosition.create(
                market_id="mkt_pr_merge_x", agent_id="agent", probability=1.01, stake=10
            )

    def test_create_validates_stake_bounds(self) -> None:
        with pytest.raises(ValueError):
            MarketPosition.create(
                market_id="mkt_pr_merge_x", agent_id="agent", probability=0.5, stake=0
            )
        with pytest.raises(ValueError):
            MarketPosition.create(
                market_id="mkt_pr_merge_x",
                agent_id="agent",
                probability=0.5,
                stake=MAX_POSITION_STAKE + 1,
            )

    def test_create_requires_non_empty_ids(self) -> None:
        with pytest.raises(ValueError):
            MarketPosition.create(market_id="", agent_id="agent", probability=0.5, stake=10)
        with pytest.raises(ValueError):
            MarketPosition.create(market_id="mkt_x", agent_id=" ", probability=0.5, stake=10)

    def test_position_id_is_deterministic_for_same_inputs(self) -> None:
        ts = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        a = MarketPosition.create(
            market_id="mkt_x",
            agent_id="agent",
            probability=0.5,
            stake=10,
            submitted_at=ts,
        )
        b = MarketPosition.create(
            market_id="mkt_x",
            agent_id="agent",
            probability=0.5,
            stake=10,
            submitted_at=ts,
        )
        assert a.position_id == b.position_id

    def test_to_json_roundtrip(self) -> None:
        position = MarketPosition.create(
            market_id="mkt_x",
            agent_id="agent",
            probability=0.7,
            stake=25,
            rationale="thesis",
        )
        roundtrip = MarketPosition.from_json(position.to_json())
        assert roundtrip == position


class TestResolutionEvent:
    def test_yes_no_inconclusive_factories(self) -> None:
        ts = datetime(2026, 4, 17, tzinfo=UTC)
        yes = ResolutionEvent.yes(
            market_id="mkt_x",
            resolution_source="github_pr_state",
            resolved_at=ts,
        )
        no = ResolutionEvent.no(
            market_id="mkt_x",
            resolution_source="github_pr_state",
            resolved_at=ts,
        )
        inc = ResolutionEvent.inconclusive(
            market_id="mkt_x",
            resolution_source="github_pr_state",
            resolved_at=ts,
        )
        assert yes.outcome == "yes"
        assert no.outcome == "no"
        assert inc.outcome == "inconclusive"

    def test_to_json_roundtrip(self) -> None:
        ts = datetime(2026, 4, 17, tzinfo=UTC)
        event = ResolutionEvent.yes(
            market_id="mkt_x",
            resolution_source="github_pr_state",
            evidence={"repo": "owner/repo", "number": 1, "merged": True},
            resolved_at=ts,
        )
        roundtrip = ResolutionEvent.from_json(event.to_json())
        assert roundtrip == event
