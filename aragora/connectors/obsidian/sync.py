"""Obsidian bidirectional sync service.

Forward sync: Obsidian vault → KnowledgeMound (notes become knowledge)
Reverse sync: KnowledgeMound → Obsidian vault (results become notes)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Frontmatter delimiters
_FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class SyncConfig:
    """Configuration for Obsidian bidirectional sync."""

    vault_path: str
    workspace_id: str = "default"
    watch_tags: list[str] | None = None  # Only sync notes with these tags
    include_untagged: bool = False
    results_folder: str = "aragora-results"  # Folder for reverse-sync notes
    poll_interval: float = 5.0
    max_notes_per_sync: int = 100


@dataclass
class SyncResult:
    """Result of a sync operation."""

    direction: str  # "forward" | "reverse"
    synced: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


class ObsidianSyncService:
    """Bidirectional sync between an Obsidian vault and KnowledgeMound.

    Forward sync:
        Reads markdown notes from the vault, extracts content and frontmatter,
        and ingests into KnowledgeMound with ``aragora-id`` tags for tracking.

    Reverse sync:
        Queries KnowledgeMound for recent pipeline results and debate outcomes,
        writes them as formatted markdown notes with provenance in frontmatter.

    Usage::

        service = ObsidianSyncService(SyncConfig(vault_path="/path/to/vault"))
        await service.sync_forward()
        await service.sync_reverse()
    """

    def __init__(self, config: SyncConfig) -> None:
        self.config = config
        self.vault = Path(config.vault_path).resolve()
        self._results_dir = self.vault / config.results_folder

    async def sync_forward(self) -> SyncResult:
        """Sync Obsidian notes → KnowledgeMound."""
        start = time.monotonic()
        result = SyncResult(direction="forward")

        try:
            from aragora.connectors.obsidian.watcher import ObsidianVaultWatcher

            watcher = ObsidianVaultWatcher(
                vault_path=str(self.vault),
                poll_interval=self.config.poll_interval,
            )
            changes = watcher.scan_changes()

            # On first run, scan_changes returns all files as "created"
            notes_to_sync = [c for c in changes if c.change_type in ("created", "modified")][
                : self.config.max_notes_per_sync
            ]

            for change in notes_to_sync:
                try:
                    content, frontmatter = self._parse_note(change.abs_path)
                    if not content.strip():
                        result.skipped += 1
                        continue

                    # Check tag filter
                    if self.config.watch_tags:
                        note_tags = frontmatter.get("tags", [])
                        if not any(t in note_tags for t in self.config.watch_tags):
                            if not self.config.include_untagged:
                                result.skipped += 1
                                continue

                    await self._ingest_note(
                        path=change.path,
                        content=content,
                        frontmatter=frontmatter,
                    )
                    result.synced += 1
                except Exception as exc:
                    logger.warning("Failed to sync note %s: %s", change.path, exc)
                    result.failed += 1
                    result.errors.append(f"{change.path}: {type(exc).__name__}")

        except Exception:
            logger.exception("Forward sync failed")
            result.errors.append("Forward sync aborted")

        result.duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Forward sync complete: synced=%d, skipped=%d, failed=%d (%.0fms)",
            result.synced,
            result.skipped,
            result.failed,
            result.duration_ms,
        )
        return result

    async def sync_reverse(
        self,
        since: datetime | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> SyncResult:
        """Sync KnowledgeMound results → Obsidian notes.

        Parameters
        ----------
        since:
            Only export results newer than this timestamp.
        query:
            Optional semantic query to filter results.
        limit:
            Maximum number of results to export.
        """
        start = time.monotonic()
        result = SyncResult(direction="reverse")

        try:
            # Ensure results directory exists
            self._results_dir.mkdir(parents=True, exist_ok=True)

            items = await self._query_km(query=query or "", limit=limit)

            for item in items:
                try:
                    note_path = self._write_result_note(item)
                    if note_path:
                        result.synced += 1
                    else:
                        result.skipped += 1
                except Exception as exc:
                    logger.warning("Failed to write result note: %s", exc)
                    result.failed += 1
                    result.errors.append(str(type(exc).__name__))

        except Exception:
            logger.exception("Reverse sync failed")
            result.errors.append("Reverse sync aborted")

        result.duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Reverse sync complete: synced=%d, skipped=%d, failed=%d (%.0fms)",
            result.synced,
            result.skipped,
            result.failed,
            result.duration_ms,
        )
        return result

    # -- forward sync helpers -----------------------------------------------

    async def _ingest_note(
        self,
        path: str,
        content: str,
        frontmatter: dict[str, Any],
    ) -> None:
        """Ingest a single note into KnowledgeMound."""
        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound()
            await mound.ingest(
                {
                    "content": content,
                    "workspace_id": self.config.workspace_id,
                    "source_type": "document",
                    "node_type": "obsidian_note",
                    "document_id": f"obsidian:{path}",
                    "metadata": {
                        "source": "obsidian",
                        "vault_path": str(self.vault),
                        "note_path": path,
                        "frontmatter": frontmatter,
                        "tags": frontmatter.get("tags", []),
                    },
                }
            )
        except ImportError:
            logger.debug("KnowledgeMound not available for ingestion")
        except Exception:
            logger.exception("KM ingestion failed for %s", path)
            raise

    def _parse_note(self, abs_path: str) -> tuple[str, dict[str, Any]]:
        """Parse a markdown note into content and frontmatter."""
        text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        frontmatter: dict[str, Any] = {}

        match = _FM_PATTERN.match(text)
        if match:
            fm_text = match.group(1)
            # Simple YAML-like parsing (avoid heavy yaml dependency)
            for line in fm_text.split("\n"):
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    if value.startswith("[") and value.endswith("]"):
                        # Simple list parsing
                        items = value[1:-1].split(",")
                        frontmatter[key] = [i.strip().strip("\"'") for i in items if i.strip()]
                    elif value.lower() in ("true", "false"):
                        frontmatter[key] = value.lower() == "true"
                    else:
                        frontmatter[key] = value.strip("\"'")
            content = text[match.end() :]
        else:
            content = text

        return content, frontmatter

    # -- reverse sync helpers -----------------------------------------------

    async def _query_km(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Query KnowledgeMound for recent results."""
        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound()
            results = await mound.query(query or "pipeline result", limit=limit)
            return [
                {
                    "id": getattr(r, "id", str(i)),
                    "content": getattr(r, "content", str(r)),
                    "score": getattr(r, "relevance_score", 0.0),
                    "metadata": getattr(r, "metadata", {}),
                }
                for i, r in enumerate(results)
            ]
        except ImportError:
            logger.debug("KnowledgeMound not available for querying")
            return []
        except Exception:
            logger.exception("KM query failed")
            return []

    def _write_result_note(self, item: dict[str, Any]) -> Path | None:
        """Write a KM result as an Obsidian note."""
        item_id = item.get("id", "unknown")
        content = item.get("content", "")
        metadata = item.get("metadata", {})
        score = item.get("score", 0.0)

        if not content:
            return None

        # Build filename from ID (sanitize for filesystem)
        safe_id = re.sub(r"[^\w\-]", "_", str(item_id))[:50]
        note_path = self._results_dir / f"{safe_id}.md"

        # Skip if already written
        if note_path.exists():
            return None

        # Build frontmatter
        now = datetime.now(timezone.utc).isoformat()
        tags = metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        fm_lines = [
            "---",
            f"aragora-id: {item_id}",
            f"created: {now}",
            "source: knowledge-mound",
            f"confidence: {score:.2f}",
        ]
        if tags:
            fm_lines.append(f"tags: [{', '.join(tags)}]")
        fm_lines.append("---")
        fm_lines.append("")

        # Build note content
        title = metadata.get("title", f"Result: {safe_id}")
        note_content = "\n".join(fm_lines) + f"# {title}\n\n{content}\n"

        note_path.write_text(note_content, encoding="utf-8")
        logger.debug("Wrote result note: %s", note_path)
        return note_path
