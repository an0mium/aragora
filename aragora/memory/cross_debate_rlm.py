"""
Cross-debate RLM memory for institutional knowledge.

Uses hierarchical compression to maintain context from previous debates,
enabling agents to reference and build upon past discussions.

Usage:
    from aragora.memory.cross_debate_rlm import CrossDebateMemory

    memory = CrossDebateMemory()
    await memory.add_debate(debate_result)

    # Get relevant context for a new debate
    context = await memory.get_relevant_context(
        task="Design a new API",
        max_tokens=2000,
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.core import DebateResult
    from aragora.rlm.types import RLMContext

logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    """Memory tiers for cross-debate context."""

    HOT = "hot"  # Recent debates, full detail
    WARM = "warm"  # Older debates, summary level
    COLD = "cold"  # Old debates, abstract level
    ARCHIVE = "archive"  # Very old, minimal context


@dataclass
class DebateMemoryEntry:
    """Entry in the cross-debate memory."""

    debate_id: str
    task: str
    domain: str
    timestamp: datetime
    tier: MemoryTier
    participants: list[str]
    consensus_reached: bool
    final_answer: str
    key_insights: list[str]
    compressed_context: str
    rlm_context: Optional[Any] = None
    token_count: int = 0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "debate_id": self.debate_id,
            "task": self.task,
            "domain": self.domain,
            "timestamp": self.timestamp.isoformat(),
            "tier": self.tier.value,
            "participants": self.participants,
            "consensus_reached": self.consensus_reached,
            "final_answer": self.final_answer,
            "key_insights": self.key_insights,
            "compressed_context": self.compressed_context,
            "token_count": self.token_count,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebateMemoryEntry":
        """Create from dictionary."""
        return cls(
            debate_id=data["debate_id"],
            task=data["task"],
            domain=data.get("domain", "general"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tier=MemoryTier(data.get("tier", "warm")),
            participants=data.get("participants", []),
            consensus_reached=data.get("consensus_reached", False),
            final_answer=data.get("final_answer", ""),
            key_insights=data.get("key_insights", []),
            compressed_context=data.get("compressed_context", ""),
            token_count=data.get("token_count", 0),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"])
                if data.get("last_accessed")
                else None
            ),
        )


@dataclass
class CrossDebateConfig:
    """Configuration for cross-debate memory."""

    # Tier thresholds
    hot_duration: timedelta = field(default_factory=lambda: timedelta(hours=24))
    warm_duration: timedelta = field(default_factory=lambda: timedelta(days=7))
    cold_duration: timedelta = field(default_factory=lambda: timedelta(days=30))

    # Memory limits
    max_entries: int = 1000
    max_hot_entries: int = 50
    max_warm_entries: int = 200
    max_cold_entries: int = 500

    # Token budgets per tier
    hot_token_budget: int = 5000
    warm_token_budget: int = 2000
    cold_token_budget: int = 500

    # Compression settings
    enable_rlm: bool = True
    compression_levels: int = 3

    # Persistence
    persist_to_disk: bool = True
    storage_path: Optional[Path] = None


class CrossDebateMemory:
    """
    Cross-debate memory with RLM compression.

    Maintains institutional knowledge from past debates,
    automatically managing memory tiers and compression.
    """

    def __init__(self, config: Optional[CrossDebateConfig] = None):
        """Initialize cross-debate memory."""
        self.config = config or CrossDebateConfig()
        self._entries: dict[str, DebateMemoryEntry] = {}
        self._compressor = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the memory system."""
        if self._initialized:
            return

        if self.config.persist_to_disk and self.config.storage_path:
            await self._load_from_disk()

        self._initialized = True

    async def _get_compressor(self):
        """Lazy-load the RLM compressor."""
        if self._compressor is None and self.config.enable_rlm:
            try:
                from aragora.rlm import HierarchicalCompressor

                self._compressor = HierarchicalCompressor()
            except ImportError:
                logger.warning("RLM module not available for cross-debate memory")
        return self._compressor

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4

    def _generate_id(self, task: str, timestamp: datetime) -> str:
        """Generate a unique ID for a debate entry."""
        content = f"{task}{timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _determine_tier(self, timestamp: datetime) -> MemoryTier:
        """Determine the appropriate tier based on age."""
        age = datetime.now() - timestamp

        if age < self.config.hot_duration:
            return MemoryTier.HOT
        if age < self.config.warm_duration:
            return MemoryTier.WARM
        if age < self.config.cold_duration:
            return MemoryTier.COLD
        return MemoryTier.ARCHIVE

    async def add_debate(self, result: "DebateResult") -> str:
        """
        Add a debate result to memory.

        Args:
            result: The debate result to add

        Returns:
            The ID of the memory entry
        """
        await self.initialize()
        async with self._lock:
            timestamp = datetime.now()
            debate_id = getattr(result, "debate_id", None) or self._generate_id(
                getattr(result, "task", ""), timestamp
            )

            # Extract key information
            task = getattr(result, "task", "")
            domain = getattr(result, "domain", "general")
            participants = getattr(result, "participants", [])
            consensus_reached = getattr(result, "consensus_reached", False)
            final_answer = getattr(result, "final_answer", "")

            # Extract key insights from messages
            key_insights = await self._extract_insights(result)

            # Build context for compression
            full_context = self._build_debate_context(result)

            # Compress the context
            compressed_context = await self._compress_context(full_context)

            # Create entry
            entry = DebateMemoryEntry(
                debate_id=debate_id,
                task=task,
                domain=domain,
                timestamp=timestamp,
                tier=MemoryTier.HOT,
                participants=participants,
                consensus_reached=consensus_reached,
                final_answer=final_answer,
                key_insights=key_insights,
                compressed_context=compressed_context,
                token_count=self._estimate_tokens(compressed_context),
            )

            self._entries[debate_id] = entry

            # Manage memory limits
            await self._manage_memory_limits()

            # Persist if enabled
            if self.config.persist_to_disk:
                await self._save_to_disk()

            return debate_id

    async def _extract_insights(self, result: "DebateResult") -> list[str]:
        """Extract key insights from a debate result."""
        insights = []

        # Extract from final answer
        if hasattr(result, "final_answer") and result.final_answer:
            insights.append(f"Conclusion: {result.final_answer[:200]}")

        # Extract from critiques
        if hasattr(result, "critiques"):
            for critique in result.critiques[:3]:
                if hasattr(critique, "summary") and critique.summary:
                    insights.append(f"Insight: {critique.summary[:150]}")

        # Extract consensus points
        if hasattr(result, "consensus_reached") and result.consensus_reached:
            insights.append("Consensus was reached among participants.")

        return insights[:5]  # Limit to 5 insights

    def _build_debate_context(self, result: "DebateResult") -> str:
        """Build full context from debate result."""
        parts = []

        if hasattr(result, "task"):
            parts.append(f"Task: {result.task}")

        if hasattr(result, "messages"):
            for msg in result.messages[:20]:  # Limit messages
                agent = getattr(msg, "agent", "unknown")
                content = getattr(msg, "content", "")[:500]
                parts.append(f"{agent}: {content}")

        if hasattr(result, "final_answer"):
            parts.append(f"Final Answer: {result.final_answer}")

        return "\n\n".join(parts)

    async def _compress_context(self, context: str) -> str:
        """Compress context using RLM."""
        if not context:
            return ""

        compressor = await self._get_compressor()
        if not compressor:
            # Simple truncation fallback
            return context[:2000]

        try:
            result = await compressor.compress(
                context,
                source_type="debate_memory",
                max_levels=self.config.compression_levels,
            )

            if result and result.context:
                from aragora.rlm.types import AbstractionLevel

                return result.context.get_at_level(AbstractionLevel.SUMMARY) or context[:2000]
        except Exception as e:
            logger.warning(f"Failed to compress debate context: {e}")

        return context[:2000]

    async def get_relevant_context(
        self,
        task: str,
        domain: Optional[str] = None,
        max_tokens: int = 2000,
        include_tiers: Optional[list[MemoryTier]] = None,
    ) -> str:
        """
        Get relevant context from past debates.

        Args:
            task: The current task/question
            domain: Optional domain filter
            max_tokens: Maximum tokens to return
            include_tiers: Tiers to include (default: all)

        Returns:
            Formatted context from relevant past debates
        """
        await self.initialize()

        tiers = include_tiers or [MemoryTier.HOT, MemoryTier.WARM, MemoryTier.COLD]
        relevant_entries = []

        # Score entries by relevance
        task_words = set(task.lower().split())

        for entry in self._entries.values():
            if entry.tier not in tiers:
                continue

            if domain and entry.domain != domain:
                continue

            # Score by word overlap
            entry_words = set(entry.task.lower().split())
            overlap = len(task_words & entry_words)

            if overlap > 0:
                relevant_entries.append((overlap, entry))

        # Sort by relevance
        relevant_entries.sort(reverse=True, key=lambda x: x[0])

        # Build context within token budget
        parts = []
        tokens_used = 0

        for _, entry in relevant_entries[:10]:
            # Get token budget based on tier
            tier_budget = {
                MemoryTier.HOT: self.config.hot_token_budget,
                MemoryTier.WARM: self.config.warm_token_budget,
                MemoryTier.COLD: self.config.cold_token_budget,
                MemoryTier.ARCHIVE: 200,
            }.get(entry.tier, 500)

            available = min(tier_budget, max_tokens - tokens_used)
            if available <= 0:
                break

            # Format entry
            entry_text = self._format_entry(entry, available)
            entry_tokens = self._estimate_tokens(entry_text)

            if tokens_used + entry_tokens <= max_tokens:
                parts.append(entry_text)
                tokens_used += entry_tokens

                # Update access tracking
                entry.access_count += 1
                entry.last_accessed = datetime.now()

        return "\n\n---\n\n".join(parts)

    def _format_entry(self, entry: DebateMemoryEntry, max_tokens: int) -> str:
        """Format a memory entry for context."""
        max_chars = max_tokens * 4

        parts = [
            f"## Previous Debate: {entry.task[:100]}",
            f"Date: {entry.timestamp.strftime('%Y-%m-%d')}",
            f"Consensus: {'Yes' if entry.consensus_reached else 'No'}",
        ]

        if entry.key_insights:
            parts.append("Key Insights:")
            for insight in entry.key_insights[:3]:
                parts.append(f"- {insight}")

        if entry.final_answer:
            answer_preview = entry.final_answer[:500]
            parts.append(f"Conclusion: {answer_preview}")

        result = "\n".join(parts)
        return result[:max_chars]

    async def _manage_memory_limits(self) -> None:
        """Manage memory limits and tier transitions."""
        now = datetime.now()

        # Update tiers based on age
        for entry in self._entries.values():
            new_tier = self._determine_tier(entry.timestamp)
            if new_tier != entry.tier:
                entry.tier = new_tier

                # Recompress if needed
                if new_tier in [MemoryTier.COLD, MemoryTier.ARCHIVE]:
                    # Further compress for cold storage
                    entry.compressed_context = entry.compressed_context[:500]
                    entry.token_count = self._estimate_tokens(entry.compressed_context)

        # Enforce limits per tier
        tier_counts: dict[MemoryTier, list[DebateMemoryEntry]] = {
            tier: [] for tier in MemoryTier
        }

        for entry in self._entries.values():
            tier_counts[entry.tier].append(entry)

        # Remove excess entries (oldest first)
        limits = {
            MemoryTier.HOT: self.config.max_hot_entries,
            MemoryTier.WARM: self.config.max_warm_entries,
            MemoryTier.COLD: self.config.max_cold_entries,
            MemoryTier.ARCHIVE: self.config.max_entries - sum([
                self.config.max_hot_entries,
                self.config.max_warm_entries,
                self.config.max_cold_entries,
            ]),
        }

        for tier, entries in tier_counts.items():
            limit = limits.get(tier, 100)
            if len(entries) > limit:
                # Sort by last accessed, remove least accessed
                entries.sort(key=lambda e: e.last_accessed or e.timestamp)
                to_remove = entries[: len(entries) - limit]
                for entry in to_remove:
                    del self._entries[entry.debate_id]

    async def _save_to_disk(self) -> None:
        """Save memory to disk."""
        if not self.config.storage_path:
            return

        try:
            self.config.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "entries": [e.to_dict() for e in self._entries.values()],
            }
            self.config.storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save cross-debate memory: {e}")

    async def _load_from_disk(self) -> None:
        """Load memory from disk."""
        if not self.config.storage_path or not self.config.storage_path.exists():
            return

        try:
            data = json.loads(self.config.storage_path.read_text())
            for entry_data in data.get("entries", []):
                entry = DebateMemoryEntry.from_dict(entry_data)
                self._entries[entry.debate_id] = entry
        except Exception as e:
            logger.error(f"Failed to load cross-debate memory: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Get memory statistics."""
        tier_stats = {tier.value: 0 for tier in MemoryTier}
        total_tokens = 0

        for entry in self._entries.values():
            tier_stats[entry.tier.value] += 1
            total_tokens += entry.token_count

        return {
            "total_entries": len(self._entries),
            "tier_distribution": tier_stats,
            "total_tokens": total_tokens,
        }

    async def clear(self) -> None:
        """Clear all memory."""
        async with self._lock:
            self._entries.clear()
            if self.config.persist_to_disk and self.config.storage_path:
                self.config.storage_path.unlink(missing_ok=True)
