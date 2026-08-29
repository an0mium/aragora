"""Live X API fetch support for the bookmark/like ingestors.

Gives :class:`TwitterBookmarksIngestor` and :class:`TwitterLikesIngestor` an
``api:`` source mode: ``ingest("api:")`` fetches from the X API v2 (OAuth2
user context) instead of a data-export file, mapping API tweets into the same
entry shape the export parsers already consume.

Incremental sync: bookmarks/likes endpoints have no ``since_id``, and
bookmark order is bookmark-time (not tweet-time), so snowflake comparison is
wrong. Instead the last run's newest entry ids are remembered in
``.aragora/x_intake/state.json`` and fetching stops at the first already-seen
id (stop-when-seen).

``api:`` alone fetches up to :data:`DEFAULT_MAX_ITEMS`; ``api:500`` overrides
the cap. Full-history backfill belongs to the data-export file path, which
has no 800-item API ceiling.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

API_SOURCE_PREFIX = "api:"
DEFAULT_MAX_ITEMS = 200
# CWD-relative by default (the CLI and digest job run from the repo root);
# override with ARAGORA_X_INTAKE_STATE for other working directories.
STATE_PATH = Path(os.environ.get("ARAGORA_X_INTAKE_STATE", ".aragora/x_intake/state.json"))
# Enough remembered ids to bridge the overlap window between runs
SEEN_IDS_KEPT = 300

__all__ = ["API_SOURCE_PREFIX", "is_api_source", "fetch_live_entries"]


def is_api_source(source: Any) -> bool:
    """True when an ingest source string requests live API mode."""
    return isinstance(source, str) and source.strip().lower().startswith(API_SOURCE_PREFIX)


def _parse_max_items(source: str) -> int:
    suffix = source.strip()[len(API_SOURCE_PREFIX) :].strip()
    if not suffix:
        return DEFAULT_MAX_ITEMS
    if suffix.isdigit():
        return max(1, int(suffix))
    # A typo like 'api:5OO' silently capping at the default would interact
    # badly with the continuity guard — fail loudly instead.
    raise ValueError(f"invalid api source {source!r}: expected 'api:' or 'api:<max_items>'")


def _load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read X intake state %s: %s", path, exc)
    return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


async def fetch_live_entries(
    source_type: str,
    source: str,
    *,
    connector: Any = None,
    state_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch new bookmark/like entries from the X API in export-entry shape.

    Args:
        source_type: ``"twitter_bookmark"`` or ``"twitter_like"``.
        source: The ``api:`` source string (optionally ``api:<max_items>``).
        connector: Injected TwitterConnector-compatible object (tests).
        state_path: Override for the incremental-sync state file.

    Returns:
        Entries newest-first, stopping at the first id seen in a prior run.
    """
    max_items = _parse_max_items(source)
    path = Path(state_path) if state_path else STATE_PATH

    if connector is None:
        from aragora.connectors.twitter import TwitterConnector

        connector = TwitterConnector()

    user_id = await connector.get_authenticated_user_id()
    if not user_id:
        logger.warning(
            "X live ingestion unavailable: no user-context token at %s "
            "(run scripts/x_oauth_setup.py) — data-export --file mode still works",
            Path(".aragora/x_intake/oauth.json").resolve(),
        )
        return []

    fetch_page = (
        connector.fetch_bookmarks_page
        if source_type == "twitter_bookmark"
        else connector.fetch_liked_page
    )

    # State is scoped per authenticated user so switching OAuth accounts
    # against the same checkout cannot cross-contaminate seen-id sets.
    # Legacy (pre-user-scoped) state is read as a fallback seed.
    state_key = f"{source_type}:{user_id}"
    state = _load_state(path)
    prior = state.get(state_key) or state.get(source_type) or {}
    seen_ids = set(prior.get("seen_ids", []))

    entries: list[dict[str, Any]] = []
    pagination_token: str | None = None
    hit_seen = False
    fetch_failed = False
    while len(entries) < max_items and not hit_seen:
        page, pagination_token = await fetch_page(user_id, pagination_token)
        if page is None:
            # Request failure is NOT feed exhaustion — never treat it as
            # continuity or the guard would advance state over a gap.
            fetch_failed = True
            break
        if not page:
            break
        for entry in page:
            tweet_id = str(entry.get("tweetId") or "")
            if tweet_id and tweet_id in seen_ids:
                hit_seen = True
                break
            entries.append(entry)
            if len(entries) >= max_items:
                break
        if not pagination_token:
            break

    # Continuity guard: advancing seen_ids is only safe when this run bridged
    # to the previous seen set (hit_seen), genuinely exhausted the feed, or
    # has no prior state to protect (first run for this user). A truncated or
    # failed fetch must not advance state, or the gap behind this run's
    # entries would be skipped forever.
    feed_exhausted = not pagination_token and not fetch_failed
    reached_continuity = hit_seen or feed_exhausted or not seen_ids
    if entries and reached_continuity and not (fetch_failed and seen_ids):
        newest_ids = [str(e.get("tweetId")) for e in entries if e.get("tweetId")]
        merged = newest_ids + list(prior.get("seen_ids", []))
        state[state_key] = {"seen_ids": merged[:SEEN_IDS_KEPT]}
        _save_state(path, state)
    elif entries:
        logger.warning(
            "%s fetch stopped early (%s) before reaching previously seen ids; "
            "state not advanced — re-run%s to close the gap",
            source_type,
            "request failure" if fetch_failed else f"max_items={max_items}",
            "" if fetch_failed else f" with a higher cap (e.g. 'api:{max_items * 2}')",
        )

    logger.info(
        "Fetched %d new %s entries from X API (stopped at seen id: %s)",
        len(entries),
        source_type,
        hit_seen,
    )
    return entries
