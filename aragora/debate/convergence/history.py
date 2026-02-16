"""
Convergence history store for tracking debate convergence metrics.

Persists convergence speed data (similarity scores per round, rounds to
convergence, final similarity) so that future debates on similar topics
can estimate optimal round counts and detect patterns.

The store uses ContinuumMemory when available, falling back to an
in-memory dict for lightweight usage or testing.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Module-level singleton
_convergence_history_store: ConvergenceHistoryStore | None = None
_store_lock = threading.Lock()


class ConvergenceHistoryStore:
    """Stores and retrieves convergence metrics from past debates.

    Each record is keyed by a topic hash and contains:
    - ``convergence_round``: The round at which convergence was first detected
      (0 if the debate never converged).
    - ``total_rounds``: Total rounds executed.
    - ``final_similarity``: The final average similarity score.
    - ``per_round_similarity``: List of average similarity per round.
    - ``topic_hash``: MD5 of the debate task (not for security).
    - ``timestamp``: When the record was stored.

    The ``find_similar`` method performs a simple keyword overlap search
    against stored topic snippets to find relevant past convergence data.
    When a ``ContinuumMemory`` is provided, records are also persisted
    there for cross-session survival.
    """

    def __init__(
        self,
        continuum_memory: Any | None = None,
        max_records: int = 1000,
    ) -> None:
        """Initialize the convergence history store.

        Args:
            continuum_memory: Optional ContinuumMemory for persistent storage.
            max_records: Maximum in-memory records before LRU eviction.
        """
        self._continuum_memory = continuum_memory
        self._max_records = max_records
        # topic_hash -> record dict
        self._records: dict[str, dict[str, Any]] = {}
        # topic_hash -> topic_snippet for search
        self._topic_index: dict[str, str] = {}
        self._lock = threading.Lock()

    def store(
        self,
        topic: str,
        convergence_round: int,
        total_rounds: int,
        final_similarity: float,
        per_round_similarity: list[float] | None = None,
        debate_id: str = "",
    ) -> str:
        """Store convergence metrics for a completed debate.

        Args:
            topic: The debate task/topic string.
            convergence_round: Round at which convergence was detected (0 = never).
            total_rounds: Total rounds in the debate.
            final_similarity: Final average similarity between agents.
            per_round_similarity: Optional list of avg similarity per round.
            debate_id: Optional debate ID for cross-referencing.

        Returns:
            The topic hash key under which the record was stored.
        """
        topic_hash = hashlib.md5(topic.encode(), usedforsecurity=False).hexdigest()
        record = {
            "topic_hash": topic_hash,
            "convergence_round": convergence_round,
            "total_rounds": total_rounds,
            "final_similarity": final_similarity,
            "per_round_similarity": per_round_similarity or [],
            "debate_id": debate_id,
            "timestamp": time.time(),
        }

        with self._lock:
            # LRU eviction
            if len(self._records) >= self._max_records:
                # Remove oldest record
                oldest_key = min(
                    self._records, key=lambda k: self._records[k].get("timestamp", 0)
                )
                del self._records[oldest_key]
                self._topic_index.pop(oldest_key, None)

            self._records[topic_hash] = record
            # Store a snippet of the topic for keyword matching
            self._topic_index[topic_hash] = topic[:200].lower()

        # Optionally persist to ContinuumMemory for cross-session survival
        if self._continuum_memory:
            try:
                from aragora.memory.continuum import MemoryTier

                memory_id = f"convergence_history_{topic_hash}"
                content = (
                    f"[convergence] topic={topic[:100]} "
                    f"converged_at_round={convergence_round}/{total_rounds} "
                    f"final_similarity={final_similarity:.2f}"
                )
                self._continuum_memory.add(
                    id=memory_id,
                    content=content,
                    tier=MemoryTier.SLOW,  # Long-lived but not urgent
                    importance=0.4,
                    metadata={
                        "type": "convergence_history",
                        "topic_hash": topic_hash,
                        "convergence_round": convergence_round,
                        "total_rounds": total_rounds,
                        "final_similarity": final_similarity,
                        "debate_id": debate_id,
                    },
                )
            except (ImportError, AttributeError, TypeError, RuntimeError, ValueError) as e:
                logger.debug("[convergence_history] ContinuumMemory persist failed: %s", e)

        logger.debug(
            "[convergence_history] Stored: topic_hash=%s rounds=%d/%d sim=%.2f",
            topic_hash[:8],
            convergence_round,
            total_rounds,
            final_similarity,
        )

        return topic_hash

    def find_similar(self, topic: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find convergence records for similar topics.

        Uses simple keyword overlap to find relevant past records.
        This is intentionally lightweight -- semantic search is handled
        elsewhere in the system.

        Args:
            topic: The topic to find similar records for.
            limit: Maximum number of records to return.

        Returns:
            List of record dicts sorted by relevance (keyword overlap).
        """
        if not self._records:
            return []

        topic_lower = topic[:200].lower()
        topic_words = set(topic_lower.split())

        if not topic_words:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []

        with self._lock:
            for topic_hash, snippet in self._topic_index.items():
                snippet_words = set(snippet.split())
                if not snippet_words:
                    continue
                overlap = len(topic_words & snippet_words)
                if overlap > 0:
                    # Jaccard similarity
                    score = overlap / len(topic_words | snippet_words)
                    record = self._records.get(topic_hash)
                    if record:
                        scored.append((score, record))

        # Sort by relevance (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def get_record(self, topic_hash: str) -> dict[str, Any] | None:
        """Get a specific convergence record by topic hash.

        Args:
            topic_hash: The MD5 hash of the topic.

        Returns:
            The record dict, or None if not found.
        """
        with self._lock:
            return self._records.get(topic_hash)

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics.

        Returns:
            Dict with record_count, max_records, and avg metrics.
        """
        with self._lock:
            record_count = len(self._records)
            if record_count == 0:
                return {
                    "record_count": 0,
                    "max_records": self._max_records,
                }

            total_conv = sum(r["convergence_round"] for r in self._records.values())
            total_rounds = sum(r["total_rounds"] for r in self._records.values())
            total_sim = sum(r["final_similarity"] for r in self._records.values())

            return {
                "record_count": record_count,
                "max_records": self._max_records,
                "avg_convergence_round": total_conv / record_count,
                "avg_total_rounds": total_rounds / record_count,
                "avg_final_similarity": total_sim / record_count,
            }

    def clear(self) -> None:
        """Clear all stored records."""
        with self._lock:
            self._records.clear()
            self._topic_index.clear()


def get_convergence_history_store() -> ConvergenceHistoryStore | None:
    """Get the module-level convergence history store singleton.

    Returns:
        The singleton store, or None if not yet initialized.
    """
    return _convergence_history_store


def set_convergence_history_store(store: ConvergenceHistoryStore | None) -> None:
    """Set the module-level convergence history store singleton.

    Args:
        store: The store instance, or None to clear.
    """
    global _convergence_history_store
    with _store_lock:
        _convergence_history_store = store


def init_convergence_history_store(
    continuum_memory: Any | None = None,
    max_records: int = 1000,
) -> ConvergenceHistoryStore:
    """Initialize and return the convergence history store singleton.

    Creates the singleton if it doesn't exist, or returns the existing one.

    Args:
        continuum_memory: Optional ContinuumMemory for persistence.
        max_records: Maximum in-memory records.

    Returns:
        The initialized store.
    """
    global _convergence_history_store
    with _store_lock:
        if _convergence_history_store is None:
            _convergence_history_store = ConvergenceHistoryStore(
                continuum_memory=continuum_memory,
                max_records=max_records,
            )
        return _convergence_history_store


__all__ = [
    "ConvergenceHistoryStore",
    "get_convergence_history_store",
    "set_convergence_history_store",
    "init_convergence_history_store",
]
