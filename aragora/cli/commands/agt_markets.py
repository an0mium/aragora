"""CLI commands: ``aragora markets list`` and ``aragora markets predict``.

Reads a synthetic-market store and prints a one-line summary per market,
or records a new agent position against an open market.

See aragora.markets for the AGT-04 substrate (issue #6065).

``predict`` is the first write-path verb: it records a :class:`MarketPosition`
in the local JSONL store.  It is flag-free and default-off by virtue of being
an explicit CLI invocation rather than an automated path.  Creating markets and
resolving them are deferred to follow-up CLI verbs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aragora.markets.store import MarketStore, MarketStoreError
from aragora.markets.types import MarketPosition


def cmd_markets_list(args: argparse.Namespace) -> int:
    """List markets in the given store directory."""
    base_dir = Path(getattr(args, "store_dir", ".aragora_markets")).expanduser()
    store = MarketStore(base_dir)
    markets = store.list_markets()
    resolutions = store.resolutions_by_market()

    if getattr(args, "json", False):
        out = []
        for market in markets:
            entry = market.to_json()
            entry["resolution"] = (
                resolutions[market.market_id].to_json() if market.market_id in resolutions else None
            )
            out.append(entry)
        print(json.dumps(out, sort_keys=True, indent=2))
        return 0

    if not markets:
        print(f"No markets found in {base_dir}")
        return 0

    print(f"{len(markets)} market(s) in {base_dir}:")
    for market in markets:
        outcome_str = "open"
        if market.market_id in resolutions:
            outcome_str = resolutions[market.market_id].outcome
        print(
            f"  [{outcome_str:12s}] {market.market_id}  "
            f"{market.question_kind:<14s}  expires={market.expires_at}"
        )
        if market.description:
            print(f"      {market.description}")
    return 0


def cmd_markets_predict(args: argparse.Namespace) -> int:
    """Record an agent position on an open market.

    Writes a :class:`~aragora.markets.types.MarketPosition` to the local
    JSONL store.  The market must exist in the store and must not yet be
    resolved.  Exits non-zero on any validation or store error.
    """
    base_dir = Path(getattr(args, "store_dir", ".aragora_markets")).expanduser()
    market_id: str = args.market_id
    agent_id: str = args.agent
    probability: float = args.probability
    stake: int = args.stake
    rationale: str = getattr(args, "rationale", "") or ""
    emit_json: bool = getattr(args, "json", False)

    store = MarketStore(base_dir)

    market = store.get_market(market_id)
    if market is None:
        print(f"error: market {market_id!r} not found in {base_dir}", file=sys.stderr)
        return 1

    resolutions = store.resolutions_by_market()
    if market_id in resolutions:
        resolution = resolutions[market_id]
        print(
            f"error: market {market_id!r} is already resolved "
            f"(outcome={resolution.outcome}); cannot add positions",
            file=sys.stderr,
        )
        return 1

    try:
        position = MarketPosition.create(
            market_id=market_id,
            agent_id=agent_id,
            probability=probability,
            stake=stake,
            rationale=rationale,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        saved = store.add_position(position)
    except MarketStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if emit_json:
        print(json.dumps(saved.to_json(), sort_keys=True, indent=2))
    else:
        print(
            f"position recorded: {saved.position_id}\n"
            f"  market  : {saved.market_id}\n"
            f"  agent   : {saved.agent_id}\n"
            f"  P(YES)  : {saved.probability:.4f}\n"
            f"  stake   : {saved.stake}\n"
            f"  at      : {saved.submitted_at}"
        )
        if saved.rationale:
            print(f"  rationale: {saved.rationale}")
    return 0


__all__ = ["cmd_markets_list", "cmd_markets_predict"]
