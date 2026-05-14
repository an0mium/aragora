"""Tests for AGT-04 SD-3: aragora/markets/credit_settlement.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aragora.blockchain.compute_budget import (
    ACCURACY_REWARD_SCALE,
    INACCURACY_PENALTY_SCALE,
    ComputeBudgetManager,
)
from aragora.markets.credit_settlement import (
    CreditSettlementError,
    settle_batch_credits,
    settle_position_credit,
)
from aragora.markets.types import MAX_POSITION_STAKE, MarketPosition, ResolutionEvent

_FLAG = "ARAGORA_SYNTHETIC_MARKETS_ENABLED"
_NOW = datetime(2026, 5, 14, tzinfo=UTC)


def _pos(
    *, agent_id: str = "a1", market_id: str = "mkt_x", probability: float = 0.8, stake: int = 10
) -> MarketPosition:
    return MarketPosition.create(
        market_id=market_id,
        agent_id=agent_id,
        probability=probability,
        stake=stake,
        submitted_at=_NOW,
    )


def _yes(mid: str = "mkt_x") -> ResolutionEvent:
    return ResolutionEvent.yes(market_id=mid, resolution_source="t", resolved_at=_NOW)


def _no(mid: str = "mkt_x") -> ResolutionEvent:
    return ResolutionEvent.no(market_id=mid, resolution_source="t", resolved_at=_NOW)


def _inc(mid: str = "mkt_x") -> ResolutionEvent:
    return ResolutionEvent.inconclusive(market_id=mid, resolution_source="t", resolved_at=_NOW)


class TestFeatureGate:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_FLAG, raising=False)
        with pytest.raises(CreditSettlementError, match=_FLAG):
            settle_position_credit(_pos(), _yes(), ComputeBudgetManager())

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
    def test_truthy_values_enable(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv(_FLAG, val)
        delta = settle_position_credit(_pos(probability=1.0), _yes(), ComputeBudgetManager())
        assert isinstance(delta, int)

    @pytest.mark.parametrize("val", ["0", "false", "no", ""])
    def test_falsy_values_raise(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv(_FLAG, val)
        with pytest.raises(CreditSettlementError):
            settle_position_credit(_pos(), _yes(), ComputeBudgetManager())

    def test_require_enabled_false_bypasses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_FLAG, raising=False)
        delta = settle_position_credit(
            _pos(probability=1.0), _yes(), ComputeBudgetManager(), require_enabled=False
        )
        assert isinstance(delta, int)

    def test_batch_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_FLAG, raising=False)
        with pytest.raises(CreditSettlementError, match=_FLAG):
            settle_batch_credits([_pos()], {"mkt_x": _yes()}, ComputeBudgetManager())

    def test_batch_require_enabled_false_bypasses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_FLAG, raising=False)
        result = settle_batch_credits(
            [_pos(probability=1.0)],
            {"mkt_x": _yes()},
            ComputeBudgetManager(),
            require_enabled=False,
        )
        assert isinstance(result, list)


class TestSingleSettlement:
    def test_perfect_yes_rewards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        mgr = ComputeBudgetManager()
        delta = settle_position_credit(_pos(probability=1.0, stake=10), _yes(), mgr)
        assert delta == 10
        assert mgr.get_budget("a1").earned_tokens > 0
        assert mgr.get_budget("a1").penalty_tokens == 0

    def test_perfect_no_rewards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        mgr = ComputeBudgetManager()
        delta = settle_position_credit(_pos(probability=0.0, stake=10), _no(), mgr)
        assert delta == 10
        assert mgr.get_budget("a1").earned_tokens > 0

    def test_maximally_wrong_penalises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        mgr = ComputeBudgetManager()
        delta = settle_position_credit(_pos(probability=0.0, stake=10), _yes(), mgr)
        assert delta == -10
        assert mgr.get_budget("a1").penalty_tokens > 0
        assert mgr.get_budget("a1").earned_tokens == 0

    def test_inconclusive_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        mgr = ComputeBudgetManager()
        delta = settle_position_credit(_pos(), _inc(), mgr)
        assert delta == 0
        assert mgr.get_budget("a1").earned_tokens == 0
        assert mgr.get_budget("a1").penalty_tokens == 0

    def test_random_payout_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        delta = settle_position_credit(
            _pos(probability=0.5, stake=1), _yes(), ComputeBudgetManager()
        )
        assert delta == 0

    def test_positive_delta_means_reward(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        mgr = ComputeBudgetManager()
        settle_position_credit(_pos(probability=0.9, stake=10), _yes(), mgr)
        assert mgr.get_budget("a1").earned_tokens > 0

    def test_negative_delta_means_penalty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        mgr = ComputeBudgetManager()
        settle_position_credit(_pos(probability=0.1, stake=10), _yes(), mgr)
        assert mgr.get_budget("a1").penalty_tokens > 0

    def test_return_type_is_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        delta = settle_position_credit(_pos(probability=0.7), _yes(), ComputeBudgetManager())
        assert type(delta) is int


class TestBatchSettlement:
    def test_empty_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        assert settle_batch_credits([], {}, ComputeBudgetManager()) == []

    def test_unresolved_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        result = settle_batch_credits([_pos(market_id="open")], {}, ComputeBudgetManager())
        assert result == []

    def test_settled_position_id_in_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        pos = _pos(probability=1.0)
        result = settle_batch_credits([pos], {"mkt_x": _yes()}, ComputeBudgetManager())
        assert len(result) == 1
        pid, delta = result[0]
        assert pid == pos.position_id
        assert delta == 10

    def test_multiple_agents_settled_independently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        a = _pos(agent_id="A", probability=1.0)
        b = _pos(agent_id="B", probability=0.0)
        mgr = ComputeBudgetManager()
        result = settle_batch_credits([a, b], {"mkt_x": _yes()}, mgr)
        d = {pid: delta for pid, delta in result}
        assert d[a.position_id] == 10
        assert d[b.position_id] == -10

    def test_mix_resolved_and_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        resolved = _pos(agent_id="R", market_id="mkt_done", probability=1.0)
        open_ = _pos(agent_id="O", market_id="mkt_open", probability=0.5)
        result = settle_batch_credits(
            [resolved, open_], {"mkt_done": _yes("mkt_done")}, ComputeBudgetManager()
        )
        assert len(result) == 1
        assert result[0][0] == resolved.position_id


class TestCreditDeltaValues:
    def test_perfect_delta_equals_max_stake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        stake = MAX_POSITION_STAKE
        delta = settle_position_credit(
            _pos(probability=1.0, stake=stake),
            _yes(),
            ComputeBudgetManager(),
            require_enabled=False,
        )
        assert delta == stake

    def test_worst_delta_equals_neg_max_stake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        stake = MAX_POSITION_STAKE
        delta = settle_position_credit(
            _pos(probability=0.0, stake=stake),
            _yes(),
            ComputeBudgetManager(),
            require_enabled=False,
        )
        assert delta == -stake

    def test_max_reward_tokens_on_perfect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        mgr = ComputeBudgetManager()
        settle_position_credit(_pos(probability=1.0, stake=10), _yes(), mgr)
        assert mgr.get_budget("a1").earned_tokens == int(1.0 * ACCURACY_REWARD_SCALE)

    def test_max_penalty_tokens_on_worst(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_FLAG, "1")
        mgr = ComputeBudgetManager()
        settle_position_credit(_pos(probability=0.0, stake=10), _yes(), mgr)
        # accuracy=0.0 → penalty = int((1.0 - 0.0) * INACCURACY_PENALTY_SCALE)
        assert mgr.get_budget("a1").penalty_tokens == int(1.0 * INACCURACY_PENALTY_SCALE)

    def test_penalty_inversion_epistemic_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """penalize_inaccuracy receives accuracy (not wrongness)."""
        monkeypatch.setenv(_FLAG, "1")
        calls: list[float] = []

        class _Spy(ComputeBudgetManager):
            def penalize_inaccuracy(self, agent_id: str, epistemic_score: float) -> int:
                calls.append(epistemic_score)
                return super().penalize_inaccuracy(agent_id, epistemic_score=epistemic_score)

        settle_position_credit(_pos(probability=0.0, stake=10), _yes(), _Spy())
        assert len(calls) == 1
        assert abs(calls[0] - 0.0) < 1e-9
