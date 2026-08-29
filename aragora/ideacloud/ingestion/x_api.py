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
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

API_SOURCE_PREFIX = "api:"
DEFAULT_MAX_ITEMS = 200
STATE_PATH = Path(".aragora/x_intake/state.json")
# Enough remembered ids to bridge the overlap window between runs
SEEN_IDS_KEPT = 300

__all__ = ["API_SOURCE_PREFIX", "is_api_source", "fetch_live_entries"]


def is_api_source(source: Any) -> bool:
    """True when an ingest source string requests live API mode."""
    return isinstance(source, str) and source.strip().lower().startswith(API_SOURCE_PREFIX)


def _parse_max_items(source: str) -> int:
    suffix = source.strip()[len(API_SOURCE_PREFIX) :].strip()
    if suffix.isdigit():
        return max(1, int(suffix))
    return DEFAULT_MAX_ITEMS


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
            "X live ingestion unavailable: no user-context token "
            "(run scripts/x_oauth_setup.py) — data-export --file mode still works"
        )
        return []

    fetch_page = (
        connector.fetch_bookmarks_page
        if source_type == "twitter_bookmark"
        else connector.fetch_liked_page
    )

    state = _load_state(path)
    seen_ids = set((state.get(source_type) or {}).get("seen_ids", []))

    entries: list[dict[str, Any]] = []
    pagination_token: str | None = None
    hit_seen = False
    while len(entries) < max_items and not hit_seen:
        page, pagination_token = await fetch_page(user_id, pagination_token)
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
    # to the previous seen set (hit_seen) or exhausted the feed. If max_items
    # truncated the fetch mid-gap, advancing state would permanently skip the
    # items between this run's oldest entry and the old seen set.
    reached_continuity = hit_seen or not pagination_token or not seen_ids
    if entries and reached_continuity:
        newest_ids = [str(e.get("tweetId")) for e in entries if e.get("tweetId")]
        merged = newest_ids + [i for i in (state.get(source_type) or {}).get("seen_ids", [])]
        state[source_type] = {"seen_ids": merged[:SEEN_IDS_KEPT]}
        _save_state(path, state)
    elif entries:
        logger.warning(
            "%s fetch hit max_items=%d before reaching previously seen ids; "
            "state not advanced — re-run with a higher cap (e.g. 'api:%d') to close the gap",
            source_type,
            max_items,
            max_items * 2,
        )

    logger.info(
        "Fetched %d new %s entries from X API (stopped at seen id: %s)",
        len(entries),
        source_type,
        hit_seen,
    )
    return entries
