"""Tests for the AGT-03 Phase 2 Manifold bet adapter (write path)."""

from __future__ import annotations

import json
from typing import Callable

import pytest

from aragora.connectors.prediction_markets.manifold import (
    MANIFOLD_WRITE_FLAG,
    ManifoldBetAdapter,
    ManifoldBetResult,
    ManifoldError,
    manifold_write_enabled,
)


def test_package_namespace_exports_write_adapter() -> None:
    from aragora.connectors.prediction_markets import (
        MANIFOLD_WRITE_FLAG as package_flag,
        ManifoldBetAdapter as PackageBetAdapter,
        ManifoldBetResult as PackageBetResult,
        manifold_write_enabled as package_write_enabled,
    )

    assert package_flag == MANIFOLD_WRITE_FLAG
    assert PackageBetAdapter is ManifoldBetAdapter
    assert PackageBetResult is ManifoldBetResult
    assert package_write_enabled is manifold_write_enabled


def _mkt(mid: str = "m1", *, liq: int | None = 2000) -> str:
    p: dict = {
        "id": mid,
        "slug": f"s-{mid}",
        "question": "Q?",
        "creatorUsername": "a",
        "createdTime": 1_700_000_000_000,
        "closeTime": 1_800_000_000_000,
        "resolution": None,
        "isResolved": False,
        "outcomeType": "BINARY",
    }
    if liq is not None:
        p["totalLiquidity"] = liq
    return json.dumps(p)


def _stub(routes: dict) -> Callable[..., tuple[int, str]]:
    def client(m: str, url: str, h: dict, b: str | None = None) -> tuple[int, str]:
        for (rm, suffix), resp in routes.items():
            if rm.upper() == m.upper() and url.endswith(suffix):
                return resp
        return (404, json.dumps({"error": f"no stub: {m} {url}"}))

    return client


def _adapter(routes: dict, mp: pytest.MonkeyPatch, **kw) -> ManifoldBetAdapter:
    mp.setenv(MANIFOLD_WRITE_FLAG, "1")
    return ManifoldBetAdapter(http_client=_stub(routes), api_key="k", **kw)


# flag -----------------------------------------------------------------


def test_write_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MANIFOLD_WRITE_FLAG, raising=False)
    assert manifold_write_enabled() is False


@pytest.mark.parametrize("v,expected", [("1", True), ("true", True), ("0", False), ("", False)])
def test_write_flag_values(monkeypatch: pytest.MonkeyPatch, v: str, expected: bool) -> None:
    monkeypatch.setenv(MANIFOLD_WRITE_FLAG, v)
    assert manifold_write_enabled() is expected


def test_place_bet_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MANIFOLD_WRITE_FLAG, raising=False)
    a = ManifoldBetAdapter(http_client=lambda *a, **kw: (200, "{}"), api_key="k")
    with pytest.raises(ManifoldError, match="disabled"):
        a.place_bet("m", probability=0.6, stake_mana=10)


# caps -----------------------------------------------------------------


def test_per_market_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _adapter(
        {
            ("GET", "market/m1"): (200, _mkt("m1", liq=9999)),
            ("POST", "bet"): (200, json.dumps({"id": "b"})),
        },
        monkeypatch,
        per_market_cap_mana=50,
        per_day_cap_mana=500,
    )
    a.place_bet("m1", probability=0.5, stake_mana=50)
    with pytest.raises(ManifoldError, match="per-market cap"):
        a.place_bet("m1", probability=0.5, stake_mana=1)


def test_per_day_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _adapter(
        {
            ("GET", "market/m1"): (200, _mkt("m1", liq=9999)),
            ("GET", "market/m2"): (200, _mkt("m2", liq=9999)),
            ("POST", "bet"): (200, json.dumps({"id": "b"})),
        },
        monkeypatch,
        per_market_cap_mana=60,
        per_day_cap_mana=100,
    )
    a.place_bet("m1", probability=0.5, stake_mana=60)
    with pytest.raises(ManifoldError, match="per-day cap"):
        a.place_bet("m2", probability=0.5, stake_mana=50)


def test_liquidity_fraction_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # 100 mana liquidity → 5% cap = 5 mana; stake of 10 exceeds it
    a = _adapter(
        {("GET", "market/s"): (200, _mkt("s", liq=100))},
        monkeypatch,
        per_market_cap_mana=1000,
        per_day_cap_mana=10_000,
    )
    with pytest.raises(ManifoldError, match="liquidity fraction"):
        a.place_bet("s", probability=0.5, stake_mana=10)


def test_no_liquidity_skips_fraction_check(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _adapter(
        {
            ("GET", "market/nl"): (200, _mkt("nl", liq=None)),
            ("POST", "bet"): (200, json.dumps({"id": "b_nl"})),
        },
        monkeypatch,
        per_market_cap_mana=100,
        per_day_cap_mana=1000,
    )
    assert a.place_bet("nl", probability=0.5, stake_mana=50).bet_id == "b_nl"


# happy path -----------------------------------------------------------


def test_place_bet_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _adapter(
        {
            ("GET", "market/mx"): (200, _mkt("mx")),
            ("POST", "bet"): (200, json.dumps({"id": "bet_abc"})),
        },
        monkeypatch,
    )
    r = a.place_bet("mx", probability=0.75, stake_mana=25)
    assert isinstance(r, ManifoldBetResult)
    assert r == ManifoldBetResult(
        bet_id="bet_abc", market_id="mx", stake_mana=25, probability=0.75, outcome="YES"
    )


def test_outcome_no_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _adapter(
        {("GET", "market/my"): (200, _mkt("my")), ("POST", "bet"): (200, json.dumps({"id": "b"}))},
        monkeypatch,
    )
    assert a.place_bet("my", probability=0.3, stake_mana=10, outcome="NO").outcome == "NO"


def test_betid_alias_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _adapter(
        {
            ("GET", "market/ma"): (200, _mkt("ma")),
            ("POST", "bet"): (200, json.dumps({"betId": "alias_xyz"})),
        },
        monkeypatch,
    )
    assert a.place_bet("ma", probability=0.5, stake_mana=10).bet_id == "alias_xyz"


def test_counters_accumulate(monkeypatch: pytest.MonkeyPatch) -> None:
    n = {"c": 0}

    def client(m: str, url: str, h: dict, b: str | None = None) -> tuple[int, str]:
        if m.upper() == "GET":
            return (200, _mkt("mz", liq=10_000))
        n["c"] += 1
        return (200, json.dumps({"id": f"b{n['c']}"}))

    monkeypatch.setenv(MANIFOLD_WRITE_FLAG, "1")
    a = ManifoldBetAdapter(
        http_client=client, api_key="k", per_market_cap_mana=200, per_day_cap_mana=1000
    )
    a.place_bet("mz", probability=0.5, stake_mana=30)
    a.place_bet("mz", probability=0.6, stake_mana=30)
    assert a._market_stakes["mz"] == 60


# validation / errors --------------------------------------------------


@pytest.mark.parametrize("prob", [0.0, 1.0, -0.1, 1.5])
def test_bad_probability(monkeypatch: pytest.MonkeyPatch, prob: float) -> None:
    monkeypatch.setenv(MANIFOLD_WRITE_FLAG, "1")
    a = ManifoldBetAdapter(http_client=lambda *a, **kw: (200, "{}"), api_key="k")
    with pytest.raises(ManifoldError, match="probability"):
        a.place_bet("m", probability=prob, stake_mana=10)


def test_zero_stake_and_bad_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MANIFOLD_WRITE_FLAG, "1")
    a = ManifoldBetAdapter(http_client=lambda *a, **kw: (200, "{}"), api_key="k")
    with pytest.raises(ManifoldError, match="stake_mana"):
        a.place_bet("m", probability=0.5, stake_mana=0)
    with pytest.raises(ManifoldError, match="outcome"):
        a.place_bet("m", probability=0.5, stake_mana=10, outcome="MAYBE")


def test_http_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _adapter(
        {
            ("GET", "market/me"): (200, _mkt("me")),
            ("POST", "bet"): (400, json.dumps({"error": "bad"})),
        },
        monkeypatch,
    )
    with pytest.raises(ManifoldError, match="HTTP 400"):
        a.place_bet("me", probability=0.5, stake_mana=10)


def test_missing_bet_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _adapter(
        {
            ("GET", "market/mn"): (200, _mkt("mn")),
            ("POST", "bet"): (200, json.dumps({"amount": 10})),
        },
        monkeypatch,
    )
    with pytest.raises(ManifoldError, match="missing"):
        a.place_bet("mn", probability=0.5, stake_mana=10)


def test_counters_unchanged_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _adapter(
        {("GET", "market/mf"): (200, _mkt("mf")), ("POST", "bet"): (500, "err")}, monkeypatch
    )
    with pytest.raises(ManifoldError):
        a.place_bet("mf", probability=0.5, stake_mana=10)
    assert a._market_stakes.get("mf", 0) == 0
