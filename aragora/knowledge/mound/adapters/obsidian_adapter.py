"""
ObsidianAdapter - Bridges Obsidian vaults to the Knowledge Mound.

Ingests notes from an Obsidian vault into the Knowledge Mound with
metadata, tags, and backlinks preserved.

Supports bidirectional sync:
- Forward: Obsidian vault notes -> Knowledge Mound
- Reverse: KM validation results -> Obsidian note frontmatter

Usage:
    from aragora.connectors.knowledge.obsidian import ObsidianConfig, ObsidianConnector
    from aragora.knowledge.mound.adapters import ObsidianAdapter

    config = ObsidianConfig(vault_path="~/Vault")
    connector = ObsidianConnector(config)
    adapter = ObsidianAdapter(connector=connector, workspace_id="team-1")

    # Forward sync
    result = await adapter.sync_to_km()

    # Reverse sync: KM validation -> Obsidian frontmatter
    result = await adapter.sync_from_km()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from aragora.connectors.knowledge.obsidian import ObsidianConfig, ObsidianConnector
from aragora.knowledge.mound.adapters._base import (
    KnowledgeMoundAdapter,
    ADAPTER_CIRCUIT_CONFIGS,
    AdapterCircuitBreakerConfig,
)
from aragora.knowledge.mound.adapters._types import SyncResult, ValidationSyncResult
from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

logger = logging.getLogger(__name__)

# Obsidian is local IO, so use tighter circuit breaker thresholds
ADAPTER_CIRCUIT_CONFIGS["obsidian"] = AdapterCircuitBreakerConfig(
    failure_threshold=3,
    success_threshold=2,
    timeout_seconds=20.0,
    half_open_max_calls=2,
)


@dataclass
class ObsidianSyncConfig:
    """Configuration for Obsidian → Knowledge Mound sync."""

    workspace_id: str = "default"
    watch_tags: list[str] | None = None
    include_untagged: bool = False
    max_notes: int | None = None


class ObsidianAdapter(KnowledgeMoundAdapter):
    """Adapter that ingests Obsidian notes into the Knowledge Mound."""

    adapter_name = "obsidian"
    source_type = "document"

    def __init__(
        self,
        connector: ObsidianConnector | None = None,
        config: ObsidianConfig | None = None,
        vault_path: str | Path | None = None,
        sync_config: ObsidianSyncConfig | None = None,
        workspace_id: str = "default",
        event_callback: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            connector: Pre-configured ObsidianConnector
            config: ObsidianConfig (used if connector not provided)
            vault_path: Vault path (used if config not provided)
            sync_config: Sync behavior configuration
            workspace_id: Knowledge Mound workspace ID
            event_callback: Optional event callback
        """
        super().__init__(**kwargs)

        if connector is None:
            if config is None:
                if vault_path is not None:
                    config = ObsidianConfig(vault_path=str(vault_path))
                else:
                    config = ObsidianConfig.from_env()
            if config is not None:
                connector = ObsidianConnector(config)

        self._connector = connector
        self._config = config or getattr(connector, "_config", None)
        self._sync_config = sync_config or ObsidianSyncConfig(workspace_id=workspace_id)
        self._event_callback = event_callback

        if self._sync_config.watch_tags is None and self._config is not None:
            self._sync_config.watch_tags = list(self._config.watch_tags)

    @property
    def connector(self) -> ObsidianConnector | None:
        """Return the underlying Obsidian connector."""
        return self._connector

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit event via callback if configured."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:  # noqa: BLE001 - adapter isolation
                logger.debug("ObsidianAdapter event callback failed: %s", e)

    def _get_mound(self) -> Any | None:
        """Get Knowledge Mound instance."""
        try:
            from aragora.knowledge.mound import get_knowledge_mound

            return get_knowledge_mound(workspace_id=self._sync_config.workspace_id)
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.debug("Could not get knowledge mound: %s", e)
            return None

    async def sync_to_km(
        self,
        knowledge_mound: Any | None = None,
        since: datetime | None = None,
        limit: int | None = None,
        tags: list[str] | None = None,
        include_untagged: bool | None = None,
    ) -> SyncResult:
        """Sync Obsidian notes into the Knowledge Mound.

        Args:
            knowledge_mound: Optional Knowledge Mound instance
            since: Only ingest notes modified after this time
            limit: Maximum notes to ingest
            tags: Optional tag filter (overrides config)
            include_untagged: Whether to ingest notes without matching tags
        """
        start_time = time.time()
        synced = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        connector = self._connector
        if connector is None or not connector.is_configured:
            return SyncResult(
                records_synced=0,
                records_skipped=0,
                records_failed=1,
                errors=["Obsidian connector not configured or vault unavailable"],
                duration_ms=(time.time() - start_time) * 1000,
            )

        mound = knowledge_mound or self._get_mound()
        if mound is None:
            return SyncResult(
                records_synced=0,
                records_skipped=0,
                records_failed=1,
                errors=["Knowledge Mound not available"],
                duration_ms=(time.time() - start_time) * 1000,
            )

        watch_tags = tags or self._sync_config.watch_tags or []
        include_untagged = (
            include_untagged if include_untagged is not None else self._sync_config.include_untagged
        )
        max_notes = limit if limit is not None else self._sync_config.max_notes

        try:
            async with self._resilient_call("sync_to_km"):
                from aragora.connectors.enterprise.base import SyncState

                sync_state = SyncState(connector_id=connector.name)
                if since is not None:
                    sync_state.last_sync_at = since

                async for item in connector.sync_items(sync_state, batch_size=max_notes or 1000):
                    item_tags: list[str] = []
                    if isinstance(item.metadata, dict):
                        item_tags = item.metadata.get("tags", []) or []

                    if watch_tags and not any(t in item_tags for t in watch_tags):
                        if not include_untagged:
                            skipped += 1
                            continue

                    try:
                        req = IngestionRequest(
                            content=item.content,
                            workspace_id=self._sync_config.workspace_id,
                            source_type=KnowledgeSource.DOCUMENT,
                            document_id=item.source_id,
                            node_type="document",
                            confidence=item.confidence,
                            topics=[t.lstrip("#") for t in item_tags if isinstance(t, str)],
                            metadata={
                                "source": "obsidian",
                                "title": item.title,
                                "url": item.url,
                                "tags": item_tags,
                                "note_type": item.metadata.get("note_type")
                                if item.metadata
                                else None,
                                "path": item.source_id,
                            },
                        )
                        await mound.ingest(req)
                        synced += 1
                    except (RuntimeError, ValueError, AttributeError, KeyError) as e:  # noqa: BLE001 - adapter isolation
                        failed += 1
                        logger.warning("Obsidian note ingestion failed: %s", e)
                        errors.append("Note ingestion failed")

                    if max_notes is not None and (synced + skipped + failed) >= max_notes:
                        break

        except (RuntimeError, ValueError, TypeError, AttributeError) as e:  # noqa: BLE001 - adapter isolation
            logger.warning("Obsidian sync failed: %s", e)
            errors.append("Obsidian sync failed")

        duration_ms = (time.time() - start_time) * 1000
        self._emit_event(
            "obsidian_sync_complete",
            {
                "synced": synced,
                "skipped": skipped,
                "failed": failed,
                "duration_ms": duration_ms,
            },
        )

        return SyncResult(
            records_synced=synced,
            records_skipped=skipped,
            records_failed=failed,
            errors=errors,
            duration_ms=duration_ms,
        )

    async def sync_from_km(
        self,
        knowledge_mound: Any | None = None,
        min_confidence: float = 0.0,
        limit: int | None = None,
    ) -> ValidationSyncResult:
        """Sync KM validation results back to Obsidian note frontmatter.

        For each note that has been ingested into the KM, this queries the
        KM for validation data (confidence, cross-debate utility, validation
        status) and writes it back to the note's frontmatter.

        Fields written:
            - km_confidence: KM's confidence score for this document
            - km_validated_at: ISO timestamp of last validation
            - km_validation_result: Validation outcome string
            - cross_debate_utility: How useful across debates (if available)

        Args:
            knowledge_mound: Optional Knowledge Mound instance.
            min_confidence: Skip notes below this KM confidence threshold.
            limit: Maximum notes to update in this sync pass.

        Returns:
            ValidationSyncResult with counts of analyzed/updated/skipped.
        """
        start_time = time.time()
        analyzed = 0
        updated = 0
        skipped = 0
        errors: list[str] = []

        connector = self._connector
        if connector is None or not connector.is_configured:
            return ValidationSyncResult(
                records_analyzed=0,
                records_updated=0,
                records_skipped=0,
                errors=["Obsidian connector not configured or vault unavailable"],
                duration_ms=(time.time() - start_time) * 1000,
            )

        mound = knowledge_mound or self._get_mound()
        if mound is None:
            return ValidationSyncResult(
                records_analyzed=0,
                records_updated=0,
                records_skipped=0,
                errors=["Knowledge Mound not available"],
                duration_ms=(time.time() - start_time) * 1000,
            )

        try:
            async with self._resilient_call("sync_from_km"):
                # Iterate vault notes and look up KM validation data
                for note in connector._iter_notes():
                    if limit is not None and (analyzed >= limit):
                        break

                    analyzed += 1
                    note_path = note.path

                    try:
                        # Query KM for validation data about this document
                        km_data = await self._query_km_validation(
                            mound, note_path, note.frontmatter.aragora_id
                        )

                        if km_data is None:
                            skipped += 1
                            continue

                        km_confidence = km_data.get("confidence", 0.0)
                        if km_confidence < min_confidence:
                            skipped += 1
                            self._record_validation_outcome(
                                note_path,
                                "skipped",
                                km_confidence,
                                details={"reason": "below_min_confidence"},
                            )
                            continue

                        # Write validation results back to note frontmatter
                        result_note = await connector.write_km_validation(
                            path=note_path,
                            km_confidence=km_confidence,
                            km_validation_result=km_data.get("validation_result", "validated"),
                            cross_debate_utility=km_data.get("cross_debate_utility"),
                            extra_fields=km_data.get("extra_fields"),
                        )

                        if result_note is not None:
                            updated += 1
                            self._record_validation_outcome(
                                note_path,
                                "applied",
                                km_confidence,
                            )
                        else:
                            skipped += 1
                            self._record_validation_outcome(
                                note_path,
                                "skipped",
                                km_confidence,
                                details={"reason": "write_returned_none"},
                            )

                    except (RuntimeError, ValueError, AttributeError, KeyError) as e:  # noqa: BLE001 - adapter isolation
                        errors.append(f"Reverse sync failed for {note_path}")
                        logger.warning("Obsidian reverse sync failed for %s: %s", note_path, e)

        except (RuntimeError, ValueError, TypeError, AttributeError) as e:  # noqa: BLE001 - adapter isolation
            logger.warning("Obsidian reverse sync failed: %s", e)
            errors.append("Obsidian reverse sync failed")

        duration_ms = (time.time() - start_time) * 1000
        self._emit_event(
            "obsidian_reverse_sync_complete",
            {
                "analyzed": analyzed,
                "updated": updated,
                "skipped": skipped,
                "duration_ms": duration_ms,
            },
        )

        return ValidationSyncResult(
            records_analyzed=analyzed,
            records_updated=updated,
            records_skipped=skipped,
            errors=errors,
            duration_ms=duration_ms,
        )

    async def _query_km_validation(
        self,
        mound: Any,
        note_path: str,
        aragora_id: str | None,
    ) -> dict[str, Any] | None:
        """Query the Knowledge Mound for validation data about a note.

        Tries to find the note's KM record by document_id (note path) or
        aragora_id and extracts validation metadata.

        Args:
            mound: Knowledge Mound instance.
            note_path: Relative path of the note within the vault.
            aragora_id: Optional Aragora ID from the note's frontmatter.

        Returns:
            Dict with km validation fields, or None if no KM record found.
        """
        try:
            # Try query by document_id (note path)
            query_terms = [note_path]
            if aragora_id:
                query_terms.append(aragora_id)

            for term in query_terms:
                try:
                    results = await mound.query(term, limit=1)
                    if results:
                        result = results[0] if isinstance(results, list) else results
                        result_dict = result.to_dict() if hasattr(result, "to_dict") else result
                        if isinstance(result_dict, dict):
                            metadata = result_dict.get("metadata", {}) or {}
                            return {
                                "confidence": result_dict.get(
                                    "confidence",
                                    metadata.get("confidence", 0.5),
                                ),
                                "validation_result": metadata.get("validation_result", "validated"),
                                "cross_debate_utility": metadata.get("cross_debate_utility"),
                                "extra_fields": None,
                            }
                except (RuntimeError, ValueError, AttributeError, KeyError, TypeError):
                    continue

            return None

        except (RuntimeError, ValueError, AttributeError, KeyError, TypeError) as e:
            logger.debug("KM query for %s failed: %s", note_path, e)
            return None


__all__ = ["ObsidianAdapter", "ObsidianSyncConfig"]
