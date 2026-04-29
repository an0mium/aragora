"""CLI commands: ``aragora calibration report`` and ``aragora calibration leaderboard``.

AGT-03.3 calibration consumer surface.

Reads the synthetic-market store (positions + resolutions) and produces
per-agent rolling-window Brier scores using
``aragora.markets.scoring.aggregate_brier``. This is the first operator
surface that closes the AGT-03.3 loop: agents predict via market
positions, markets resolve, and we report which agents are well- or
poorly-calibrated.

Read-only. Recording predictions and resolutions stays with the swarm
runtime; this command only consumes already-recorded events.

The report verb prints (or emits as JSON) the Brier breakdown for one
or all agents over a rolling window. The leaderboard verb sorts agents
by (decayed) Brier ascending — lower is better — with a minimum-sample
floor so single-position agents do not dominate.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aragora.markets.scoring import BrierBreakdown, aggregate_brier
from aragora.markets.store import MarketStore


def _filter_positions_within_window(positions, resolutions, *, now: datetime, window_days: float):
    """Return positions whose market resolution is within the rolling window.

    Unresolved positions are kept (aggregate_brier will skip them); any
    position whose resolution is older than ``window_days`` is dropped
    so the rolling-window semantic is honored even when the underlying
    store has a long history.
    """
    cutoff = now - timedelta(days=window_days)
    out = []
    for pos in positions:
        resolution = resolutions.get(pos.market_id)
        if resolution is None:
            out.append(pos)
            continue
        try:
            resolved_at = datetime.fromisoformat(
                resolution.resolved_at.replace("Z", "+00:00")
            ).astimezone(UTC)
        except (ValueError, AttributeError):
            out.append(pos)
            continue
        if resolved_at >= cutoff:
            out.append(pos)
    return out


def _breakdown_to_dict(breakdown: BrierBreakdown) -> dict:
    """Serialize a BrierBreakdown to a JSON-safe dict.

    None values are preserved (rather than coerced to 0) so callers can
    distinguish ``no scored positions`` from ``perfect-zero Brier``.
    """
    return {
        "agent_id": breakdown.agent_id,
        "scored_positions": breakdown.scored_positions,
        "inconclusive_positions": breakdown.inconclusive_positions,
        "mean_brier": breakdown.mean_brier,
        "stake_weighted_brier": breakdown.stake_weighted_brier,
        "decayed_brier": breakdown.decayed_brier,
        "total_stake": breakdown.total_stake,
    }


def _format_brier(value: float | None) -> str:
    """Format a Brier score for human-readable output."""
    return f"{value:.4f}" if value is not None else "n/a"


def cmd_calibration_report(args: argparse.Namespace) -> int:
    """Print a per-agent Brier breakdown over a rolling window.

    When ``--agent`` is omitted, prints a breakdown for every agent
    that has at least one position in the window.
    """
    base_dir = Path(getattr(args, "store_dir", ".aragora_markets")).expanduser()
    store = MarketStore(base_dir)

    window_days = float(getattr(args, "window_days", 90.0))
    half_life_days = getattr(args, "half_life_days", 30.0)
    if half_life_days is not None:
        half_life_days = float(half_life_days)

    now = datetime.now(tz=UTC)
    resolutions = store.resolutions_by_market()
    all_positions = store.list_positions()
    windowed = _filter_positions_within_window(
        all_positions, resolutions, now=now, window_days=window_days
    )

    agent_filter = getattr(args, "agent", None)
    if agent_filter:
        agent_ids = [agent_filter]
    else:
        agent_ids = sorted({pos.agent_id for pos in windowed})

    breakdowns: list[BrierBreakdown] = [
        aggregate_brier(
            agent_id=aid,
            positions=windowed,
            resolutions=resolutions,
            now=now,
            half_life_days=half_life_days,
        )
        for aid in agent_ids
    ]

    if getattr(args, "json", False):
        out = {
            "store_dir": str(base_dir),
            "window_days": window_days,
            "half_life_days": half_life_days,
            "now": now.isoformat(),
            "agents": [_breakdown_to_dict(b) for b in breakdowns],
        }
        print(json.dumps(out, sort_keys=True, indent=2))
        return 0

    if not breakdowns:
        print(f"No positions found in {base_dir} within the last {window_days:.0f} days.")
        return 0

    print(
        f"Calibration report (window={window_days:.0f}d, "
        f"half_life={half_life_days}d, store={base_dir}):"
    )
    print(
        f"  {'agent':<24s} {'scored':>7s} {'incon':>6s} "
        f"{'mean':>9s} {'stake_wt':>10s} {'decayed':>9s} {'stake':>8s}"
    )
    for breakdown in breakdowns:
        print(
            f"  {breakdown.agent_id:<24s} {breakdown.scored_positions:>7d} "
            f"{breakdown.inconclusive_positions:>6d} "
            f"{_format_brier(breakdown.mean_brier):>9s} "
            f"{_format_brier(breakdown.stake_weighted_brier):>10s} "
            f"{_format_brier(breakdown.decayed_brier):>9s} "
            f"{breakdown.total_stake:>8d}"
        )
    return 0


def cmd_calibration_leaderboard(args: argparse.Namespace) -> int:
    """Print agents sorted by Brier ascending (lower = better calibrated).

    Honors a minimum scored-position floor so that agents with only a
    handful of positions cannot dominate the leaderboard. The default
    floor (5) matches the AGT-03 plan's "weekly rolling 90d Brier" cadence
    of meaningful sample sizes.
    """
    base_dir = Path(getattr(args, "store_dir", ".aragora_markets")).expanduser()
    store = MarketStore(base_dir)

    window_days = float(getattr(args, "window_days", 90.0))
    half_life_days = getattr(args, "half_life_days", 30.0)
    if half_life_days is not None:
        half_life_days = float(half_life_days)
    min_scored = int(getattr(args, "min_scored", 5))
    sort_by = getattr(args, "sort_by", "decayed") or "decayed"

    if sort_by not in ("decayed", "mean", "stake_weighted"):
        print(
            f"Invalid --sort-by={sort_by!r}; expected one of decayed, mean, stake_weighted",
        )
        return 2

    now = datetime.now(tz=UTC)
    resolutions = store.resolutions_by_market()
    all_positions = store.list_positions()
    windowed = _filter_positions_within_window(
        all_positions, resolutions, now=now, window_days=window_days
    )

    agent_ids = sorted({pos.agent_id for pos in windowed})
    breakdowns: list[BrierBreakdown] = [
        aggregate_brier(
            agent_id=aid,
            positions=windowed,
            resolutions=resolutions,
            now=now,
            half_life_days=half_life_days,
        )
        for aid in agent_ids
    ]

    eligible = [b for b in breakdowns if b.scored_positions >= min_scored]
    excluded = [b for b in breakdowns if b.scored_positions < min_scored]

    sort_key_map = {
        "decayed": lambda b: b.decayed_brier,
        "mean": lambda b: b.mean_brier,
        "stake_weighted": lambda b: b.stake_weighted_brier,
    }
    sort_key = sort_key_map[sort_by]

    eligible_sorted = sorted(
        eligible,
        key=lambda b: (sort_key(b) is None, sort_key(b) if sort_key(b) is not None else 1.0),
    )

    if getattr(args, "json", False):
        out = {
            "store_dir": str(base_dir),
            "window_days": window_days,
            "half_life_days": half_life_days,
            "min_scored": min_scored,
            "sort_by": sort_by,
            "now": now.isoformat(),
            "leaderboard": [_breakdown_to_dict(b) for b in eligible_sorted],
            "excluded_below_floor": [_breakdown_to_dict(b) for b in excluded],
        }
        print(json.dumps(out, sort_keys=True, indent=2))
        return 0

    if not eligible_sorted:
        print(
            f"No agents meet the minimum-scored floor "
            f"(min_scored={min_scored}, window={window_days:.0f}d, "
            f"store={base_dir}).",
        )
        if excluded:
            print(f"  {len(excluded)} agent(s) below the floor — see --json for detail.")
        return 0

    print(
        f"Calibration leaderboard (sort={sort_by}, window={window_days:.0f}d, "
        f"half_life={half_life_days}d, min_scored={min_scored}, store={base_dir}):"
    )
    print(
        f"  {'rank':>4s}  {'agent':<24s} {'scored':>7s} {'mean':>9s} "
        f"{'stake_wt':>10s} {'decayed':>9s}"
    )
    for rank, breakdown in enumerate(eligible_sorted, start=1):
        print(
            f"  {rank:>4d}  {breakdown.agent_id:<24s} {breakdown.scored_positions:>7d} "
            f"{_format_brier(breakdown.mean_brier):>9s} "
            f"{_format_brier(breakdown.stake_weighted_brier):>10s} "
            f"{_format_brier(breakdown.decayed_brier):>9s}"
        )
    if excluded:
        print(
            f"  ({len(excluded)} agent(s) below min_scored={min_scored} "
            f"floor — see --json for detail.)"
        )
    return 0


__all__ = ["cmd_calibration_report", "cmd_calibration_leaderboard"]
