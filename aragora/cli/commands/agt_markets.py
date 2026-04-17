"""CLI command: ``aragora markets list``.

Reads a synthetic-market store and prints a one-line summary per market.
See aragora.markets for the AGT-04 substrate (issue #6065).

This is read-only. Creating markets, taking positions, and resolving
markets are deferred to follow-up CLI verbs once the substrate-first
gate opens for AGT-04 production use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aragora.markets.store import MarketStore


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


__all__ = ["cmd_markets_list"]
