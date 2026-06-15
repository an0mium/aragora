#!/usr/bin/env python3
"""Probe freshness of the B0 / TW-03 proof-surface status docs.

The v13 prompt-proof-first hardening track requires a tiny, read-only
helper an operator can run locally (or wire into a cron / LaunchAgent)
to answer the question: "are the published proof surfaces still fresh
enough to be trusted as the canonical recurring receipts?".

Inputs
------
Two repo-tracked status surfaces are inspected:

* ``docs/status/B0_BENCHMARK_TRUTH_STATUS.md`` — the recurring TW-02
  benchmark truth publication surface (B0 in the v13 spec).
* ``docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md`` — the recurring
  TW-03 rescue productization status surface.

Each file is expected to contain a single canonical ``Last updated:``
line near the top, in either ISO 8601 form
(``2026-05-17T14:36:51Z``) or bare ``YYYY-MM-DD`` form.

Behaviour
---------
* ``--max-age-days N`` (default ``7``) caps how old a surface can be
  before it is considered stale.
* ``--surfaces b0,tw03`` (default both) scopes the probe.
* Emits a single JSON document on stdout with one record per surface
  shaped like
  ``{"surface": ..., "last_updated": ..., "age_days": ..., "fresh": ...}``.
* Exits ``0`` when every probed surface is fresh, non-zero otherwise.
* Treats a malformed or missing ``Last updated:`` line as a hard error
  and exits non-zero with a clear message on stderr.
* Treats a materially future-dated ``Last updated:`` line as a hard
  error, so bad clock data cannot make a stale proof surface appear
  fresh. A small tolerance absorbs benign runner clock skew.

Design constraints
~~~~~~~~~~~~~~~~~~
* Pure stdlib. No third-party imports. No ``aragora`` package import.
* Never mutates the inspected status files. Never writes to the
  filesystem. Never invokes git, gh, or any network call.
* Safe to run from any working directory: the script resolves the
  repo root from ``__file__``.

Usage
-----
::

    # Default: both surfaces, 7-day window.
    python3 scripts/probe_proof_surface_freshness.py

    # Scope to just the B0 surface and tighten the window to 3 days.
    python3 scripts/probe_proof_surface_freshness.py \\
        --surfaces b0 --max-age-days 3

    # Operator-facing pretty output:
    python3 scripts/probe_proof_surface_freshness.py --pretty

Suggested cron / manual cadence: every 24 hours from a LaunchAgent or
GitHub Actions cron job that is *not* on the protected main-required
list. This probe is observability-only; it must never block the main
branch directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Canonical surface registry.
#
# The mapping below is the single source of truth for which proof-surface
# status documents this probe knows about. Keys are the short surface
# identifiers accepted on the ``--surfaces`` CLI flag (case-insensitive);
# values are the repo-relative path to the corresponding status file.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

SURFACE_PATHS: dict[str, Path] = {
    "b0": Path("docs/status/B0_BENCHMARK_TRUTH_STATUS.md"),
    "tw03": Path("docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md"),
}

DEFAULT_SURFACES: tuple[str, ...] = ("b0", "tw03")
DEFAULT_MAX_AGE_DAYS: int = 7
FUTURE_TIMESTAMP_TOLERANCE_SECONDS: int = 5 * 60

_LAST_UPDATED_PATTERN = re.compile(
    r"^\s*Last\s+updated\s*:\s*(?P<value>\S+)\s*$",
    re.IGNORECASE,
)


class FreshnessProbeError(Exception):
    """Raised when a surface cannot be probed (missing file, bad header)."""


@dataclasses.dataclass(frozen=True)
class SurfaceProbeResult:
    """Outcome of probing a single surface."""

    surface: str
    path: str
    last_updated: str
    age_days: float
    fresh: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "path": self.path,
            "last_updated": self.last_updated,
            "age_days": round(self.age_days, 4),
            "fresh": self.fresh,
        }


# ---------------------------------------------------------------------------
# Parsing helpers.
# ---------------------------------------------------------------------------


def parse_last_updated(text: str) -> dt.datetime:
    """Parse the first ``Last updated:`` line from a markdown surface.

    Accepts either ISO 8601 (``2026-05-17T14:36:51Z`` / ``+00:00``) or
    bare ``YYYY-MM-DD``. The returned ``datetime`` is always tz-aware
    in UTC so callers can do safe deltas.

    Raises
    ------
    FreshnessProbeError
        If no ``Last updated:`` line is found, or the value cannot be
        parsed as one of the supported formats.
    """
    for line in text.splitlines():
        match = _LAST_UPDATED_PATTERN.match(line)
        if not match:
            continue
        raw = match.group("value").strip()
        return _coerce_timestamp(raw)
    raise FreshnessProbeError(
        "no 'Last updated:' line found in surface (first 5 lines should "
        "contain a canonical 'Last updated: <iso8601 or YYYY-MM-DD>' "
        "header)"
    )


def _coerce_timestamp(raw: str) -> dt.datetime:
    """Coerce a ``Last updated:`` value into a tz-aware UTC datetime."""
    # Normalise trailing ``Z`` (Zulu time) into the ``+00:00`` form that
    # ``datetime.fromisoformat`` understands across Python 3.10+.
    normalised = raw.strip()
    if normalised.endswith("Z"):
        normalised = normalised[:-1] + "+00:00"

    # Try the full ISO 8601 form first.
    try:
        parsed = dt.datetime.fromisoformat(normalised)
    except ValueError:
        parsed = None  # type: ignore[assignment]

    # Fall back to the bare ``YYYY-MM-DD`` calendar-date form.
    if parsed is None:
        try:
            parsed_date = dt.date.fromisoformat(normalised)
        except ValueError as exc:
            raise FreshnessProbeError(
                f"malformed 'Last updated:' value {raw!r}: expected "
                "ISO 8601 timestamp or YYYY-MM-DD"
            ) from exc
        parsed = dt.datetime.combine(parsed_date, dt.time(0, 0, 0))

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


# ---------------------------------------------------------------------------
# Probe core.
# ---------------------------------------------------------------------------


def probe_surface(
    surface: str,
    *,
    repo_root: Path,
    max_age_days: float,
    now: dt.datetime,
) -> SurfaceProbeResult:
    """Probe a single named surface against ``max_age_days``."""
    try:
        rel = SURFACE_PATHS[surface.lower()]
    except KeyError as exc:
        raise FreshnessProbeError(
            f"unknown surface {surface!r}: valid surfaces are {sorted(SURFACE_PATHS)}"
        ) from exc

    path = repo_root / rel
    if not path.is_file():
        raise FreshnessProbeError(
            f"surface {surface!r} not found at {path} (expected a tracked markdown status doc)"
        )

    text = path.read_text(encoding="utf-8")
    try:
        last_updated = parse_last_updated(text)
    except FreshnessProbeError as exc:
        raise FreshnessProbeError(f"surface {surface!r} at {path}: {exc}") from exc

    age = now - last_updated
    if age.total_seconds() < -FUTURE_TIMESTAMP_TOLERANCE_SECONDS:
        future_days = abs(age.total_seconds()) / 86400.0
        raise FreshnessProbeError(
            f"surface {surface!r} at {path}: 'Last updated:' value "
            f"{last_updated.strftime('%Y-%m-%dT%H:%M:%SZ')} is {future_days:.2f} "
            "day(s) in the future"
        )

    age_days = max(age.total_seconds(), 0.0) / 86400.0
    fresh = age_days <= max_age_days

    # Render the timestamp in canonical ``Z`` form for deterministic
    # JSON output regardless of the original spelling in the doc.
    rendered = last_updated.strftime("%Y-%m-%dT%H:%M:%SZ")

    return SurfaceProbeResult(
        surface=surface.lower(),
        path=str(rel),
        last_updated=rendered,
        age_days=age_days,
        fresh=fresh,
    )


def probe_surfaces(
    surfaces: Sequence[str],
    *,
    repo_root: Path,
    max_age_days: float,
    now: dt.datetime,
) -> list[SurfaceProbeResult]:
    """Probe ``surfaces`` and return one result per entry, in order."""
    results: list[SurfaceProbeResult] = []
    for surface in surfaces:
        results.append(
            probe_surface(
                surface,
                repo_root=repo_root,
                max_age_days=max_age_days,
                now=now,
            )
        )
    return results


# ---------------------------------------------------------------------------
# CLI plumbing.
# ---------------------------------------------------------------------------


def _split_surfaces(raw: str) -> list[str]:
    parts = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--surfaces must be a non-empty comma-separated list")
    unknown = [p for p in parts if p not in SURFACE_PATHS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown surface(s): {', '.join(unknown)} (valid: {', '.join(sorted(SURFACE_PATHS))})"
        )
    # Preserve order but drop duplicates so the JSON output never has
    # redundant entries when an operator types ``--surfaces b0,b0``.
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the freshness of the B0 / TW-03 proof-surface status "
            "docs and exit non-zero if any are stale."
        ),
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=DEFAULT_MAX_AGE_DAYS,
        help=(
            "Maximum permitted age (in days) of the 'Last updated:' "
            "header before a surface is reported stale. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--surfaces",
        type=_split_surfaces,
        default=list(DEFAULT_SURFACES),
        help=(
            "Comma-separated list of surfaces to probe. Default: "
            "b0,tw03. Valid values: " + ",".join(sorted(SURFACE_PATHS)) + "."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=(
            "Override the repo root used to resolve surface paths. "
            "Defaults to the repo containing this script."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output with 2-space indent.",
    )
    return parser


def _render_json(
    results: Iterable[SurfaceProbeResult],
    *,
    max_age_days: float,
    pretty: bool,
) -> str:
    payload = {
        "max_age_days": max_age_days,
        "surfaces": [r.to_dict() for r in results],
        "fresh": all(r.fresh for r in results),
    }
    if pretty:
        return json.dumps(payload, indent=2, sort_keys=True)
    return json.dumps(payload, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    surfaces = list(args.surfaces)
    if args.max_age_days < 0:
        print(
            "error: --max-age-days must be non-negative",
            file=sys.stderr,
        )
        return 2

    now = dt.datetime.now(tz=dt.timezone.utc)

    try:
        results = probe_surfaces(
            surfaces,
            repo_root=args.repo_root,
            max_age_days=args.max_age_days,
            now=now,
        )
    except FreshnessProbeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = _render_json(results, max_age_days=args.max_age_days, pretty=args.pretty)
    print(output)

    stale = [r for r in results if not r.fresh]
    if stale:
        offenders = ", ".join(f"{r.surface}(age_days={r.age_days:.2f})" for r in stale)
        print(
            f"error: stale proof surface(s) detected: {offenders}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI entrypoint
    raise SystemExit(main())
