"""Tests for ``aragora markets create`` and ``aragora markets resolve`` CLI verbs.

Hermetic: tmp_path store only.  No network calls, no queue mutations.

Advances: issue #6065 (AGT-04), sub-deliverable 4 — operator CLI create/resolve.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta

import pytest

import aragora.markets.types as market_types
from aragora.cli.commands.agt_markets import (
    cmd_markets_create,
    cmd_markets_list,
    cmd_markets_resolve,
)
from aragora.markets.store import MarketStore
from aragora.markets.types import Market


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_args(
    tmp_path,
    *,
    type="pr_merge",
    repo="synaptent/aragora",
    number=42,
    ref=None,
    window_days=None,
    description="",
    emit_json=False,
):
    return argparse.Namespace(
        store_dir=str(tmp_path),
        type=type,
        repo=repo,
        number=number,
        ref=ref,
        window_days=window_days,
        description=description,
        json=emit_json,
    )


def _resolve_args(tmp_path, market_id, *, outcome="yes", evidence="", emit_json=False):
    return argparse.Namespace(
        store_dir=str(tmp_path),
        market_id=market_id,
        outcome=outcome,
        evidence=evidence,
        json=emit_json,
    )


def _seed(tmp_path, *, kind="pr_merge", number=99):
    store = MarketStore(tmp_path)
    m = Market.create(
        question_kind=kind,
        target={"repo": "synaptent/aragora", "number": number},
        description="test",
        resolution_window_days=7,
    )
    store.add_market(m)
    return m


# ---------------------------------------------------------------------------
# cmd_markets_create
# ---------------------------------------------------------------------------


def test_create_pr_merge_returns_zero_and_persists(tmp_path, capsys):
    rc = cmd_markets_create(_create_args(tmp_path))
    assert rc == 0
    assert "market created" in capsys.readouterr().out
    assert MarketStore(tmp_path).list_markets()[0].question_kind == "pr_merge"


def test_create_issue_close_uses_30d_default(tmp_path):
    cmd_markets_create(_create_args(tmp_path, type="issue_close", number=5, window_days=None))
    m = MarketStore(tmp_path).list_markets()[0]
    created = datetime.fromisoformat(m.created_at.replace("Z", "+00:00"))
    expires = datetime.fromisoformat(m.expires_at.replace("Z", "+00:00"))
    assert (expires - created).days == 30


def test_create_ci_pass_from_ref(tmp_path):
    rc = cmd_markets_create(_create_args(tmp_path, type="ci_pass", number=None, ref="main"))
    assert rc == 0
    assert MarketStore(tmp_path).list_markets()[0].target["ref"] == "main"


def test_create_json_output(tmp_path, capsys):
    cmd_markets_create(_create_args(tmp_path, emit_json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["question_kind"] == "pr_merge"
    assert "market_id" in payload and "expires_at" in payload


def test_create_missing_number_nonzero(tmp_path, capsys):
    rc = cmd_markets_create(_create_args(tmp_path, number=None))
    assert rc != 0
    assert "number" in capsys.readouterr().err.lower()


def test_create_ci_pass_missing_ref_nonzero(tmp_path, capsys):
    rc = cmd_markets_create(_create_args(tmp_path, type="ci_pass", number=None, ref=None))
    assert rc != 0
    assert "ref" in capsys.readouterr().err.lower()


def test_create_invalid_repo_nonzero(tmp_path):
    rc = cmd_markets_create(_create_args(tmp_path, repo="not-valid"))
    assert rc != 0


def test_create_zero_window_nonzero(tmp_path, capsys):
    rc = cmd_markets_create(_create_args(tmp_path, window_days=0))
    assert rc == 1
    assert "resolution_window_days" in capsys.readouterr().err


def test_create_duplicate_market_nonzero(tmp_path, capsys, monkeypatch):
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    times = iter([created_at, created_at + timedelta(seconds=1)])
    monkeypatch.setattr(market_types, "_utc_now", lambda: next(times))

    assert cmd_markets_create(_create_args(tmp_path, number=77)) == 0
    capsys.readouterr()

    rc = cmd_markets_create(_create_args(tmp_path, number=77))
    assert rc == 1
    assert "already exists" in capsys.readouterr().err.lower()
    assert len(MarketStore(tmp_path).list_markets()) == 1


# ---------------------------------------------------------------------------
# cmd_markets_resolve
# ---------------------------------------------------------------------------


def test_resolve_yes_persisted(tmp_path, capsys):
    m = _seed(tmp_path)
    rc = cmd_markets_resolve(_resolve_args(tmp_path, m.market_id, outcome="yes"))
    assert rc == 0
    event = MarketStore(tmp_path).resolutions_by_market()[m.market_id]
    assert event.outcome == "yes"
    assert event.resolution_source == "operator_cli"


def test_resolve_inconclusive_persisted(tmp_path):
    m = _seed(tmp_path)
    cmd_markets_resolve(_resolve_args(tmp_path, m.market_id, outcome="inconclusive"))
    event = MarketStore(tmp_path).resolutions_by_market()[m.market_id]
    assert event.outcome == "inconclusive"


def test_resolve_evidence_stored(tmp_path):
    m = _seed(tmp_path)
    cmd_markets_resolve(_resolve_args(tmp_path, m.market_id, evidence="merged at abc"))
    event = MarketStore(tmp_path).resolutions_by_market()[m.market_id]
    assert event.evidence.get("note") == "merged at abc"


def test_resolve_json_output(tmp_path, capsys):
    m = _seed(tmp_path)
    cmd_markets_resolve(_resolve_args(tmp_path, m.market_id, outcome="no", emit_json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "no" and payload["market_id"] == m.market_id


def test_resolve_unknown_market_nonzero(tmp_path, capsys):
    rc = cmd_markets_resolve(_resolve_args(tmp_path, "mkt_does_not_exist"))
    assert rc == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_resolve_already_resolved_nonzero(tmp_path, capsys):
    m = _seed(tmp_path)
    cmd_markets_resolve(_resolve_args(tmp_path, m.market_id, outcome="yes"))
    rc = cmd_markets_resolve(_resolve_args(tmp_path, m.market_id, outcome="no"))
    assert rc == 1
    assert "already resolved" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# Round-trip: create → resolve → list
# ---------------------------------------------------------------------------


def test_create_resolve_round_trip_visible_in_list(tmp_path, capsys):
    cmd_markets_create(_create_args(tmp_path, number=55))
    m = MarketStore(tmp_path).list_markets()[0]
    cmd_markets_resolve(_resolve_args(tmp_path, m.market_id, outcome="yes"))
    capsys.readouterr()  # drain create+resolve output
    cmd_markets_list(argparse.Namespace(store_dir=str(tmp_path), json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["resolution"]["outcome"] == "yes"
