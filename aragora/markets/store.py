"""JSONL-backed persistence for synthetic markets, positions, and resolutions.

The store is intentionally simple — append-only JSONL files plus an
in-memory index. The on-disk shape is the same shape the AGT-05 wiring
will consume, so this layer is the seam that lets the reputation flow
work against either a live store or a deterministic test fixture.

Files (under ``base_dir``):
- ``markets.jsonl``      — one Market per line
- ``positions.jsonl``    — one MarketPosition per line
- ``resolutions.jsonl``  — one ResolutionEvent per line
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from aragora.markets.types import (
    MAX_POSITION_STAKE,
    Market,
    MarketPosition,
    ResolutionEvent,
)

logger = logging.getLogger(__name__)


class MarketStoreError(RuntimeError):
    """Raised on store-layer invariant violations."""


@dataclass(frozen=True)
class StoreLayout:
    base_dir: Path
    markets_path: Path
    positions_path: Path
    resolutions_path: Path

    @classmethod
    def from_base(cls, base_dir: Path) -> "StoreLayout":
        return cls(
            base_dir=base_dir,
            markets_path=base_dir / "markets.jsonl",
            positions_path=base_dir / "positions.jsonl",
            resolutions_path=base_dir / "resolutions.jsonl",
        )


class MarketStore:
    """Append-only JSONL store for markets, positions, and resolutions.

    The store is created lazily — files are only written when objects
    are added. Reading from a fresh store returns empty iterables.

    Concurrency: the store assumes a single writer per ``base_dir``.
    Multi-writer coordination is out of scope for AGT-04; AGT-05's
    settlement flow will introduce the locking primitives.
    """

    def __init__(self, base_dir: Path | str) -> None:
        path = Path(base_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        self._layout = StoreLayout.from_base(path)
        self._markets: dict[str, Market] = {}
        self._positions: dict[str, MarketPosition] = {}
        self._resolutions: dict[str, ResolutionEvent] = {}
        self._loaded = False

    @property
    def layout(self) -> StoreLayout:
        return self._layout

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_if_needed(self) -> None:
        if self._loaded:
            return
        self._markets = _load_jsonl(self._layout.markets_path, Market.from_json, "market_id")
        self._positions = _load_jsonl(
            self._layout.positions_path, MarketPosition.from_json, "position_id"
        )
        self._resolutions = _load_jsonl(
            self._layout.resolutions_path, ResolutionEvent.from_json, "market_id"
        )
        self._loaded = True

    def reload(self) -> None:
        """Drop the in-memory cache and re-read from disk on next access."""
        self._loaded = False

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def add_market(self, market: Market) -> Market:
        self._load_if_needed()
        existing = self._markets.get(market.market_id)
        if existing is not None:
            if existing == market:
                return existing
            raise MarketStoreError(
                f"market {market.market_id} already exists with different fields"
            )
        _append_jsonl(self._layout.markets_path, market.to_json())
        self._markets[market.market_id] = market
        return market

    def get_market(self, market_id: str) -> Market | None:
        self._load_if_needed()
        return self._markets.get(market_id)

    def list_markets(self) -> list[Market]:
        self._load_if_needed()
        return list(self._markets.values())

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def add_position(self, position: MarketPosition) -> MarketPosition:
        self._load_if_needed()
        if position.position_id in self._positions:
            return self._positions[position.position_id]
        if position.market_id not in self._markets:
            raise MarketStoreError(f"cannot add position for unknown market {position.market_id}")
        if position.stake > MAX_POSITION_STAKE:
            raise MarketStoreError(
                f"stake {position.stake} exceeds MAX_POSITION_STAKE={MAX_POSITION_STAKE}"
            )
        if position.market_id in self._resolutions:
            raise MarketStoreError(
                f"market {position.market_id} is already resolved; cannot add positions"
            )
        _append_jsonl(self._layout.positions_path, position.to_json())
        self._positions[position.position_id] = position
        return position

    def list_positions(self, *, market_id: str | None = None) -> list[MarketPosition]:
        self._load_if_needed()
        if market_id is None:
            return list(self._positions.values())
        return [pos for pos in self._positions.values() if pos.market_id == market_id]

    def list_agent_positions(self, agent_id: str) -> list[MarketPosition]:
        self._load_if_needed()
        return [pos for pos in self._positions.values() if pos.agent_id == agent_id]

    # ------------------------------------------------------------------
    # Resolutions
    # ------------------------------------------------------------------

    def record_resolution(self, event: ResolutionEvent) -> ResolutionEvent:
        self._load_if_needed()
        if event.market_id not in self._markets:
            raise MarketStoreError(f"cannot resolve unknown market {event.market_id}")
        existing = self._resolutions.get(event.market_id)
        if existing is not None:
            if existing == event:
                return existing
            raise MarketStoreError(f"market {event.market_id} already has a divergent resolution")
        _append_jsonl(self._layout.resolutions_path, event.to_json())
        self._resolutions[event.market_id] = event
        return event

    def resolutions_by_market(self) -> dict[str, ResolutionEvent]:
        self._load_if_needed()
        return dict(self._resolutions)

    # ------------------------------------------------------------------
    # Convenience iteration
    # ------------------------------------------------------------------

    def iter_unresolved_markets(self) -> Iterable[Market]:
        self._load_if_needed()
        for market in self._markets.values():
            if market.market_id not in self._resolutions:
                yield market


def _append_jsonl(path: Path, payload: dict) -> None:
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _load_jsonl(path: Path, deserialize, key_field: str) -> dict:
    if not path.exists():
        return {}
    out: dict = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                obj = deserialize(payload)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                logger.warning("skipping malformed line in %s: %s", path, exc)
                continue
            key = getattr(obj, key_field)
            out[key] = obj
    return out


__all__ = ["MarketStore", "MarketStoreError", "StoreLayout"]
