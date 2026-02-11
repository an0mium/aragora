"""
Gastown Metrics — Real implementation querying nomic stores.

Provides the gastown extension entry point for observability metrics.
Dashboard handlers import from here rather than reaching into
aragora.nomic internals directly.

Metrics provided:
- get_beads_completed_count(hours): Count of beads completed in time window
- get_convoy_completion_rate(): Percentage of convoys completed successfully
- get_gupp_recovery_count(hours): Count of GUPP recovery events in time window
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _get_nomic_dir() -> Path:
    """Get the nomic data directory."""
    import os

    env_dir = (
        os.environ.get("ARAGORA_DATA_DIR")
        or os.environ.get("ARAGORA_NOMIC_DIR")
        or os.environ.get("NOMIC_DIR")
    )
    if env_dir:
        return Path(env_dir)

    # Default fallback
    return Path.home() / ".aragora" / "nomic"


def get_beads_completed_count(hours: int = 24) -> int:
    """
    Return count of beads completed within the specified time window.

    Args:
        hours: Number of hours to look back (default: 24)

    Returns:
        Count of completed beads in the time window
    """
    try:
        from aragora.nomic.beads import BeadStatus, BeadStore

        nomic_dir = _get_nomic_dir()
        bead_dir = nomic_dir / "beads"

        if not bead_dir.exists():
            return 0

        store = BeadStore(bead_dir)
        # BeadStore.list_beads is async - run in event loop
        import asyncio

        try:
            asyncio.get_running_loop()
            # If we're in an async context, we can't use asyncio.run()
            # Return 0 to avoid blocking the event loop
            return 0
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            beads = asyncio.run(store.list_beads(status=BeadStatus.COMPLETED))

        # Filter by time window
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        count = 0
        # BeadStore.list_beads returns an iterable of Bead objects
        for bead in beads:  # type: ignore[attr-defined]
            # Check if bead has completed_at timestamp
            completed_at = getattr(bead, "completed_at", None)
            if completed_at is None:
                # Fallback to updated_at if completed_at not available
                completed_at = getattr(bead, "updated_at", None)
            if completed_at is not None and completed_at >= cutoff:
                count += 1
            elif completed_at is None:
                # If no timestamp, count it (conservative approach)
                count += 1

        return count
    except ImportError:
        logger.debug("BeadStore not available, returning 0 for beads_completed_count")
        return 0
    except Exception as e:
        logger.warning(f"Error getting beads completed count: {e}")
        return 0


def get_convoy_completion_rate() -> float:
    """
    Return the convoy completion rate as a percentage (0.0 to 100.0).

    Returns:
        Percentage of convoys that completed successfully
    """
    try:
        from aragora.nomic.convoys import ConvoyStatus

        nomic_dir = _get_nomic_dir()
        convoy_file = nomic_dir / "beads" / "convoys.jsonl"

        if not convoy_file.exists():
            return 0.0

        import json

        total = 0
        completed = 0

        with open(convoy_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    total += 1
                    if status == ConvoyStatus.COMPLETED.value:
                        completed += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        if total == 0:
            return 0.0

        return round((completed / total) * 100, 1)
    except ImportError:
        logger.debug("Convoy module not available, returning 0.0 for convoy_completion_rate")
        return 0.0
    except Exception as e:
        logger.warning(f"Error getting convoy completion rate: {e}")
        return 0.0


def get_gupp_recovery_count(hours: int = 24) -> int:
    """
    Return count of GUPP recovery events within the specified time window.

    GUPP recovery events occur when an agent starts up and finds pending
    work on its hook that needs to be processed.

    Args:
        hours: Number of hours to look back (default: 24)

    Returns:
        Count of GUPP recovery events in the time window
    """
    try:
        nomic_dir = _get_nomic_dir()
        hooks_dir = nomic_dir / "hooks"

        if not hooks_dir.exists():
            return 0

        import json

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recovery_count = 0

        # Scan hook files for recovery events
        for hook_file in hooks_dir.glob("*.jsonl"):
            try:
                with open(hook_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            # Look for beads that were recovered (status change from pending)
                            # Recovery is indicated by a bead that was assigned but reprocessed
                            recovered_at = data.get("recovered_at")
                            if recovered_at:
                                try:
                                    recovered_dt = datetime.fromisoformat(recovered_at)
                                    if recovered_dt >= cutoff:
                                        recovery_count += 1
                                except (ValueError, TypeError) as e:
                                    logger.debug("Failed to parse datetime value: %s", e)
                        except (json.JSONDecodeError, KeyError):
                            continue
            except Exception as e:
                logger.debug(f"Error reading hook file {hook_file}: {e}")
                continue

        return recovery_count
    except Exception as e:
        logger.warning(f"Error getting GUPP recovery count: {e}")
        return 0


__all__ = [
    "get_beads_completed_count",
    "get_convoy_completion_rate",
    "get_gupp_recovery_count",
]
