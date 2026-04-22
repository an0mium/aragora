"""Tests for the AGT-03 Metaculus read-only adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timezone
from typing import Any

import pytest

from aragora.connectors.prediction_markets.metaculus import (
    MetaculusAdapter,
    MetaculusError,
    MetaculusQuestion,
    MetaculusResolution,
    _normalize_outcome,
    metaculus_to_market_resolution,
)


def _p(**kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": 1234,
        "title": "Q?",
        "question_type": "binary",
        "active_state": "active",
        "resolution": None,
        "close_time": None,
        "resolve_time": None,
        "created_time": "2026-01-01T00:00:00Z",
    }
    d.update(kw)
    return d


def _stub(responses: dict[str, tuple[int, str]]):
    def c(m: str, url: str, h: dict) -> tuple[int, str]:
        for k, (s, b) in responses.items():
            if k in url:
                return s, b
        return 404, '{"error":"no stub"}'

    return c


# -- parsing --


def test_parse_basic() -> None:
    q = MetaculusQuestion.from_api_payload(_p())
    assert q.question_id == 1234 and not q.is_resolved and q.resolution is None


def test_parse_resolved() -> None:
    q = MetaculusQuestion.from_api_payload(_p(active_state="resolved", resolution=1.0))
    assert q.is_resolved and q.resolution == 1.0


def test_parse_missing_id_raises() -> None:
    with pytest.raises(MetaculusError, match="missing 'id'"):
        MetaculusQuestion.from_api_payload({"title": "x"})


def test_parse_non_int_id_raises() -> None:
    with pytest.raises(MetaculusError, match="not an integer"):
        MetaculusQuestion.from_api_payload({"id": "bad"})


def test_parse_community_q2() -> None:
    p = _p()
    p["community_prediction"] = {"full": {"q2": 0.73}}
    assert MetaculusQuestion.from_api_payload(p).community_q2 == pytest.approx(0.73)


def test_parse_malformed_cp_ignored() -> None:
    p = _p()
    p["community_prediction"] = "junk"
    assert MetaculusQuestion.from_api_payload(p).community_q2 is None


# -- normalize_outcome --


@pytest.mark.parametrize(
    "res,exp",
    [
        (1.0, "yes"),
        (0.0, "no"),
        (-1.0, "inconclusive"),
        (None, "inconclusive"),
        (0.5, "inconclusive"),
    ],
)
def test_normalize_outcome(res: float | None, exp: str) -> None:
    assert _normalize_outcome(res) == exp


# -- adapter --


def test_fetch_question() -> None:
    a = MetaculusAdapter(
        http_client=_stub({"questions/99/": (200, json.dumps(_p(id=99, title="T")))})
    )
    q = a.fetch_question(99)
    assert q.question_id == 99 and q.title == "T"


def test_fetch_http_error_raises() -> None:
    with pytest.raises(MetaculusError, match="HTTP 500"):
        MetaculusAdapter(http_client=_stub({"questions/5/": (500, "x")})).fetch_question(5)


def test_fetch_non_object_raises() -> None:
    with pytest.raises(MetaculusError, match="non-object payload"):
        MetaculusAdapter(http_client=_stub({"questions/3/": (200, "[]")})).fetch_question(3)


def test_transport_error_raises() -> None:
    def broken(m: str, u: str, h: dict) -> tuple[int, str]:
        raise OSError("refused")

    with pytest.raises(MetaculusError, match="transport error"):
        MetaculusAdapter(http_client=broken).fetch_question(1)


def test_list_paginated_response() -> None:
    items = [_p(id=i) for i in [10, 20, 30]]
    a = MetaculusAdapter(
        http_client=_stub({"questions/": (200, json.dumps({"count": 3, "results": items}))})
    )
    assert {q.question_id for q in a.list_questions(limit=3)} == {10, 20, 30}


def test_list_raw_list_response() -> None:
    a = MetaculusAdapter(http_client=_stub({"questions/": (200, json.dumps([_p(id=7)]))}))
    assert len(a.list_questions()) == 1


def test_list_skips_malformed() -> None:
    items = [{"bad": "entry"}, _p(id=55)]
    a = MetaculusAdapter(
        http_client=_stub({"questions/": (200, json.dumps({"count": 2, "results": items}))})
    )
    qs = a.list_questions()
    assert len(qs) == 1 and qs[0].question_id == 55


def test_list_limit_out_of_range() -> None:
    a = MetaculusAdapter(http_client=_stub({}))
    with pytest.raises(MetaculusError, match="limit must be"):
        a.list_questions(limit=0)
    with pytest.raises(MetaculusError, match="limit must be"):
        a.list_questions(limit=101)


def test_list_unexpected_shape_raises() -> None:
    a = MetaculusAdapter(http_client=_stub({"questions/": (200, json.dumps("bad"))}))
    with pytest.raises(MetaculusError, match="unexpected shape"):
        a.list_questions()


def test_discover_window_filter() -> None:
    now = datetime(2026, 4, 22, tzinfo=UTC)
    items = [
        _p(id=1, close_time="2026-04-27T00:00:00Z"),
        _p(id=2, close_time="2026-07-01T00:00:00Z"),
    ]
    a = MetaculusAdapter(
        http_client=_stub({"questions/": (200, json.dumps({"count": 2, "results": items}))}),
        min_window_days=30,
    )
    result = a.discover_open_binary_questions(now=now)
    assert len(result) == 1 and result[0].question_id == 2


def test_discover_excludes_no_close_time() -> None:
    now = datetime(2026, 4, 22, tzinfo=UTC)
    a = MetaculusAdapter(
        http_client=_stub({"questions/": (200, json.dumps({"count": 1, "results": [_p()]}))})
    )
    assert a.discover_open_binary_questions(now=now) == []


def test_fetch_resolution_yes() -> None:
    p = _p(id=20, active_state="resolved", resolution=1.0, resolve_time="2026-03-01T12:00:00Z")
    p["community_prediction"] = {"full": {"q2": 0.8}}
    a = MetaculusAdapter(http_client=_stub({"questions/20/": (200, json.dumps(p))}))
    r = a.fetch_resolution(20)
    assert r is not None and r.outcome == "yes" and r.community_q2 == pytest.approx(0.8)


def test_fetch_resolution_no() -> None:
    p = _p(id=30, active_state="resolved", resolution=0.0)
    r = MetaculusAdapter(
        http_client=_stub({"questions/30/": (200, json.dumps(p))})
    ).fetch_resolution(30)
    assert r is not None and r.outcome == "no"


def test_fetch_resolution_unresolved_is_none() -> None:
    a = MetaculusAdapter(http_client=_stub({"questions/10/": (200, json.dumps(_p(id=10)))}))
    assert a.fetch_resolution(10) is None


def test_fetch_resolution_ambiguous_is_inconclusive() -> None:
    p = _p(id=40, active_state="resolved", resolution=-1.0)
    r = MetaculusAdapter(
        http_client=_stub({"questions/40/": (200, json.dumps(p))})
    ).fetch_resolution(40)
    assert r is not None and r.outcome == "inconclusive"


# -- bridge --


def _res(outcome: str, qid: int = 100, q2: float | None = None) -> MetaculusResolution:
    return MetaculusResolution(qid, outcome, "2026-03-15T00:00:00Z", q2, {})


def test_bridge_yes() -> None:
    mr = metaculus_to_market_resolution(_res("yes", q2=0.75))
    assert mr.outcome == "yes" and mr.resolution_source == "metaculus"
    assert mr.market_id == "metaculus:100" and mr.evidence.get("community_q2") == pytest.approx(
        0.75
    )


def test_bridge_no() -> None:
    assert metaculus_to_market_resolution(_res("no", qid=200)).outcome == "no"


def test_bridge_inconclusive() -> None:
    assert metaculus_to_market_resolution(_res("inconclusive")).outcome == "inconclusive"


def test_bridge_explicit_resolved_at() -> None:
    override = datetime(2025, 6, 1, tzinfo=timezone.utc)
    mr = metaculus_to_market_resolution(_res("yes"), resolved_at=override)
    assert mr.resolved_at is not None and "2025-06-01" in mr.resolved_at


def test_bridge_parsed_resolved_at() -> None:
    mr = metaculus_to_market_resolution(_res("yes"))
    assert mr.resolved_at is not None and "2026-03-15" in mr.resolved_at
