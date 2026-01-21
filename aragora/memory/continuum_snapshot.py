"""
Continuum Memory Snapshot Operations.

Extracted from continuum.py for maintainability.
Provides checkpoint integration with export/restore capabilities.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.memory.tier_manager import MemoryTier

logger = logging.getLogger(__name__)


class ContinuumSnapshotMixin:
    """
    Mixin providing snapshot export/restore for ContinuumMemory.

    Enables complete debate state restoration with memory context
    via the checkpoint system.
    """

    # These must be provided by the main class
    hyperparams: Dict[str, Any]
    _tier_manager: Any

    def connection(self) -> Any:
        """Get database connection context manager."""
        raise NotImplementedError

    def export_snapshot(
        self,
        tiers: Optional[List["MemoryTier"]] = None,
        include_metadata: bool = True,
        max_entries_per_tier: int = 100,
    ) -> Dict[str, Any]:
        """
        Export current memory state as a serializable snapshot.

        Used by checkpoint system to capture memory context for debate restoration.
        The snapshot includes all memory entries needed to restore the cognitive
        context that informed debate responses.

        Args:
            tiers: Specific tiers to export (default: all tiers)
            include_metadata: Whether to include entry metadata
            max_entries_per_tier: Maximum entries per tier to export (for size control)

        Returns:
            Dict with:
                - entries: List of serialized memory entries
                - tier_counts: Count per tier
                - hyperparams: Current hyperparameter values
                - snapshot_time: ISO timestamp
                - total_entries: Total entry count

        Example:
            # In checkpoint creation:
            snapshot = continuum_memory.export_snapshot()
            checkpoint.continuum_memory_state = snapshot
        """
        from aragora.memory.tier_manager import MemoryTier
        from aragora.utils.json_helpers import safe_json_loads

        # Import schema version from main module
        try:
            from aragora.memory.continuum import CONTINUUM_SCHEMA_VERSION
        except ImportError:
            CONTINUUM_SCHEMA_VERSION = 3

        if tiers is None:
            tiers = list(MemoryTier)

        with self.connection() as conn:
            cursor = conn.cursor()

            entries = []
            tier_counts: Dict[str, int] = {}

            for tier in tiers:
                # Get entries for this tier, sorted by importance
                cursor.execute(
                    """
                    SELECT id, tier, content, importance, surprise_score,
                           consolidation_score, update_count, success_count,
                           failure_count, created_at, updated_at, metadata,
                           COALESCE(red_line, 0), COALESCE(red_line_reason, '')
                    FROM continuum_memory
                    WHERE tier = ?
                    ORDER BY importance DESC
                    LIMIT ?
                    """,
                    (tier.value, max_entries_per_tier),
                )

                rows = cursor.fetchall()
                tier_counts[tier.value] = len(rows)

                for row in rows:
                    entry_dict = {
                        "id": row[0],
                        "tier": row[1],
                        "content": row[2],
                        "importance": row[3],
                        "surprise_score": row[4],
                        "consolidation_score": row[5],
                        "update_count": row[6],
                        "success_count": row[7],
                        "failure_count": row[8],
                        "created_at": row[9],
                        "updated_at": row[10],
                        "red_line": bool(row[12]),
                        "red_line_reason": row[13],
                    }
                    if include_metadata:
                        entry_dict["metadata"] = safe_json_loads(row[11], {})
                    entries.append(entry_dict)

        return {
            "entries": entries,
            "tier_counts": tier_counts,
            "hyperparams": dict(self.hyperparams),
            "snapshot_time": datetime.now().isoformat(),
            "total_entries": sum(tier_counts.values()),
            "version": CONTINUUM_SCHEMA_VERSION,
        }

    def restore_snapshot(
        self,
        snapshot: Dict[str, Any],
        merge_mode: str = "replace",
        restore_hyperparams: bool = False,
    ) -> Dict[str, int]:
        """
        Restore memory state from a snapshot.

        Used by checkpoint system to restore memory context when resuming debates.
        Entries from the snapshot are inserted into the database, replacing or
        merging with existing entries.

        Args:
            snapshot: Snapshot dict from export_snapshot()
            merge_mode: How to handle existing entries:
                - "replace": Overwrite existing entries with snapshot data
                - "keep": Keep existing entries, only insert new ones
                - "merge": Update existing entries with higher importance wins
            restore_hyperparams: Whether to restore hyperparameters from snapshot

        Returns:
            Dict with:
                - restored: Count of entries restored
                - skipped: Count of entries skipped (in keep mode)
                - updated: Count of entries updated (in merge mode)

        Example:
            # In checkpoint restoration:
            if checkpoint.continuum_memory_state:
                result = continuum_memory.restore_snapshot(checkpoint.continuum_memory_state)
                logger.info(f"Restored {result['restored']} memory entries")
        """
        if "entries" not in snapshot:
            logger.warning("Invalid snapshot: missing 'entries' key")
            return {"restored": 0, "skipped": 0, "updated": 0}

        entries = snapshot["entries"]
        restored = 0
        skipped = 0
        updated = 0

        # Optionally restore hyperparams
        if restore_hyperparams and "hyperparams" in snapshot:
            self.hyperparams.update(snapshot["hyperparams"])
            self._tier_manager.promotion_cooldown_hours = self.hyperparams.get(
                "promotion_cooldown_hours", 24.0
            )

        with self.connection() as conn:
            cursor = conn.cursor()

            for entry in entries:
                entry_id = entry.get("id")
                if not entry_id:
                    continue

                # Check if entry exists
                cursor.execute(
                    "SELECT importance FROM continuum_memory WHERE id = ?",
                    (entry_id,),
                )
                existing = cursor.fetchone()

                if existing:
                    if merge_mode == "keep":
                        skipped += 1
                        continue
                    elif merge_mode == "merge":
                        # Keep the one with higher importance
                        if existing[0] >= entry.get("importance", 0):
                            skipped += 1
                            continue
                        # Update existing entry
                        cursor.execute(
                            """
                            UPDATE continuum_memory
                            SET importance = ?, surprise_score = ?, consolidation_score = ?,
                                updated_at = ?
                            WHERE id = ?
                            """,
                            (
                                entry.get("importance", 0.5),
                                entry.get("surprise_score", 0.0),
                                entry.get("consolidation_score", 0.0),
                                datetime.now().isoformat(),
                                entry_id,
                            ),
                        )
                        updated += 1
                        continue

                # Insert or replace entry
                metadata_str = json.dumps(entry.get("metadata", {}))
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO continuum_memory
                    (id, tier, content, importance, surprise_score, consolidation_score,
                     update_count, success_count, failure_count, created_at, updated_at,
                     metadata, red_line, red_line_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry_id,
                        entry.get("tier", "slow"),
                        entry.get("content", ""),
                        entry.get("importance", 0.5),
                        entry.get("surprise_score", 0.0),
                        entry.get("consolidation_score", 0.0),
                        entry.get("update_count", 1),
                        entry.get("success_count", 0),
                        entry.get("failure_count", 0),
                        entry.get("created_at", datetime.now().isoformat()),
                        entry.get("updated_at", datetime.now().isoformat()),
                        metadata_str,
                        1 if entry.get("red_line") else 0,
                        entry.get("red_line_reason", ""),
                    ),
                )
                restored += 1

            conn.commit()

        logger.info(
            "continuum_memory_snapshot_restored restored=%d skipped=%d updated=%d",
            restored,
            skipped,
            updated,
        )

        return {"restored": restored, "skipped": skipped, "updated": updated}


__all__ = ["ContinuumSnapshotMixin"]
