"""
Cross-System Memory Deduplication Engine.

Detects and manages duplicate content across memory systems
(ContinuumMemory, KnowledgeMound, Supermemory, claude-mem).

Uses content hashing for exact matches and optional semantic
similarity for near-duplicates.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DedupMatch:
    """A detected duplicate pair."""

    item_id_a: str
    source_a: str
    item_id_b: str
    source_b: str
    match_type: str  # "exact" or "near"
    similarity: float  # 1.0 for exact, 0-1 for near
    content_hash: str


@dataclass
class DedupCheckResult:
    """Result of checking an item for duplicates before write."""

    is_duplicate: bool = False
    existing_id: str | None = None
    existing_source: str | None = None
    similarity: float = 0.0
    content_hash: str = ""


@dataclass
class CrossSystemDedupReport:
    """Report from a cross-system dedup scan."""

    total_items_scanned: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    matches: list[DedupMatch] = field(default_factory=list)
    duration_ms: float = 0.0


class CrossSystemDedupEngine:
    """Detects duplicates across memory systems.

    Uses SHA-256 content hashing for exact match detection and
    optional token-overlap similarity for near-duplicate detection.
    """

    def __init__(
        self,
        near_duplicate_threshold: float = 0.85,
    ):
        self._near_duplicate_threshold = near_duplicate_threshold
        # content_hash -> (item_id, source_system)
        self._hash_index: dict[str, tuple[str, str]] = {}

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA-256 hash of normalized text content."""
        normalized = _normalize_text(content)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def register_item(self, item_id: str, source: str, content: str) -> None:
        """Register an item in the hash index for future dedup checks."""
        content_hash = self.compute_content_hash(content)
        self._hash_index[content_hash] = (item_id, source)

    async def check_duplicate_before_write(
        self,
        content: str,
        targets: list[str] | None = None,
    ) -> DedupCheckResult:
        """Check if content already exists before writing.

        Args:
            content: Content to check
            targets: Optional list of source systems to check against

        Returns:
            DedupCheckResult indicating if content is a duplicate
        """
        content_hash = self.compute_content_hash(content)

        # Check exact match in hash index
        if content_hash in self._hash_index:
            existing_id, existing_source = self._hash_index[content_hash]
            if targets is None or existing_source in targets:
                return DedupCheckResult(
                    is_duplicate=True,
                    existing_id=existing_id,
                    existing_source=existing_source,
                    similarity=1.0,
                    content_hash=content_hash,
                )

        return DedupCheckResult(
            is_duplicate=False,
            content_hash=content_hash,
        )

    async def scan_cross_system_duplicates(
        self,
        items: list[dict[str, Any]],
    ) -> CrossSystemDedupReport:
        """Scan a collection of items for cross-system duplicates.

        Args:
            items: List of dicts with keys: id, source, content

        Returns:
            CrossSystemDedupReport with found duplicates
        """
        import time

        start = time.time()
        matches: list[DedupMatch] = []
        hash_map: dict[str, list[dict[str, Any]]] = {}

        # Group by content hash for exact matches
        for item in items:
            content = item.get("content", "")
            content_hash = self.compute_content_hash(content)
            if content_hash not in hash_map:
                hash_map[content_hash] = []
            hash_map[content_hash].append(item)

        exact_count = 0
        for content_hash, group in hash_map.items():
            if len(group) < 2:
                continue
            # Create match pairs for all cross-system duplicates
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    # Only report cross-system duplicates
                    if a.get("source") != b.get("source"):
                        matches.append(
                            DedupMatch(
                                item_id_a=a.get("id", ""),
                                source_a=a.get("source", ""),
                                item_id_b=b.get("id", ""),
                                source_b=b.get("source", ""),
                                match_type="exact",
                                similarity=1.0,
                                content_hash=content_hash,
                            )
                        )
                        exact_count += 1

        # Near-duplicate detection via token overlap
        near_count = 0
        if self._near_duplicate_threshold < 1.0:
            near_count = self._find_near_duplicates(items, matches)

        duration_ms = (time.time() - start) * 1000

        return CrossSystemDedupReport(
            total_items_scanned=len(items),
            exact_duplicates=exact_count,
            near_duplicates=near_count,
            matches=matches,
            duration_ms=duration_ms,
        )

    def _find_near_duplicates(
        self,
        items: list[dict[str, Any]],
        matches: list[DedupMatch],
    ) -> int:
        """Find near-duplicate items using token overlap similarity."""
        found = 0
        # Only check items from different sources, skip exact matches
        existing_pairs: set[tuple[str, str]] = set()
        for m in matches:
            existing_pairs.add((m.item_id_a, m.item_id_b))
            existing_pairs.add((m.item_id_b, m.item_id_a))

        tokenized = []
        for item in items:
            content = item.get("content", "")
            tokens = set(_normalize_text(content).split())
            tokenized.append((item, tokens))

        for i in range(len(tokenized)):
            item_a, tokens_a = tokenized[i]
            if not tokens_a:
                continue
            for j in range(i + 1, len(tokenized)):
                item_b, tokens_b = tokenized[j]
                if not tokens_b:
                    continue
                # Skip same-source
                if item_a.get("source") == item_b.get("source"):
                    continue
                # Skip already-matched exact pairs
                pair = (item_a.get("id", ""), item_b.get("id", ""))
                if pair in existing_pairs:
                    continue

                # Jaccard similarity
                intersection = len(tokens_a & tokens_b)
                union = len(tokens_a | tokens_b)
                similarity = intersection / union if union > 0 else 0.0

                if similarity >= self._near_duplicate_threshold:
                    matches.append(
                        DedupMatch(
                            item_id_a=item_a.get("id", ""),
                            source_a=item_a.get("source", ""),
                            item_id_b=item_b.get("id", ""),
                            source_b=item_b.get("source", ""),
                            match_type="near",
                            similarity=similarity,
                            content_hash="",
                        )
                    )
                    found += 1

        return found

    def get_hash_index_size(self) -> int:
        """Get the number of items in the hash index."""
        return len(self._hash_index)

    def clear_hash_index(self) -> None:
        """Clear the hash index."""
        self._hash_index.clear()


def _normalize_text(text: str) -> str:
    """Normalize text for hashing: lowercase, collapse whitespace, strip."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def dedup_results(
    results: list[dict[str, Any]],
    threshold: float = 0.95,
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate a list of result dicts by content hash.

    Simple helper for gateway query result dedup.

    Args:
        results: List of result dicts with "content" key
        threshold: Not used for exact dedup (kept for API compat)

    Returns:
        Tuple of (deduped_results, num_removed)
    """
    seen_hashes: set[str] = set()
    deduped: list[dict[str, Any]] = []
    removed = 0

    for result in results:
        content = result.get("content", "")
        content_hash = CrossSystemDedupEngine.compute_content_hash(content)
        if content_hash in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(content_hash)
        deduped.append(result)

    return deduped, removed
