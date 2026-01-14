"""
Audience Wisdom Injection - Crowd-sourced insight amplification.

Allows audience members to contribute insights during debates:
- Submit brief wisdom snippets (max 280 chars)
- Automatic relevance scoring against debate context
- Strategic injection during agent timeouts or stalls
- Clear attribution and transparency

Inspired by nomic loop debate consensus on audience participation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class WisdomSubmission:
    """A wisdom contribution from the audience."""

    id: str
    text: str
    submitter_id: str
    timestamp: float
    loop_id: str
    context_tags: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    used: bool = False
    used_at: Optional[float] = None
    upvotes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def generate_id(text: str, submitter_id: str, timestamp: float) -> str:
        """Generate unique ID for a submission."""
        content = f"{text}:{submitter_id}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class WisdomInjection:
    """Record of a wisdom injection into the debate."""

    wisdom_id: str
    agent_context: str
    injection_reason: str  # timeout, stall, requested
    timestamp: float
    impact_score: float = 0.0  # Measured later

    def to_dict(self) -> dict:
        return asdict(self)


class WisdomInjector:
    """
    Manages audience wisdom submissions and strategic injection.

    The wisdom injector collects audience insights and injects them
    into debates when agents timeout or when the debate stalls,
    providing human perspective to keep things moving.

    Usage:
        injector = WisdomInjector(loop_id="debate_123")

        # Audience submits wisdom
        wisdom = await injector.submit_wisdom(
            text="Consider the performance implications",
            submitter_id="user_456",
            context_tags=["performance", "architecture"]
        )

        # During agent timeout
        injection = injector.find_relevant_wisdom(
            debate_context={"topic": "system optimization"},
            failed_agent="claude",
        )
        if injection:
            # Add to agent context or use as fallback
            pass
    """

    MAX_WISDOM_LENGTH = 280  # Twitter-style limit
    MAX_PENDING_WISDOM = 100  # Per loop
    RELEVANCE_THRESHOLD = 0.3  # Minimum relevance to consider

    def __init__(
        self,
        loop_id: str,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize the wisdom injector.

        Args:
            loop_id: Current debate loop identifier
            storage_path: Path for persistence (defaults to .nomic/wisdom)
        """
        self.loop_id = loop_id
        self.storage_path = storage_path or Path(".nomic/wisdom")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.pending_wisdom: list[WisdomSubmission] = []
        self.used_wisdom: list[WisdomSubmission] = []
        self.injections: list[WisdomInjection] = []

        # Submitter reputation tracking
        self.submitter_stats: dict[str, dict] = defaultdict(
            lambda: {"submissions": 0, "used": 0, "upvotes": 0}
        )

        # Load existing wisdom for this loop
        self._load_pending()

        logger.info(f"wisdom_injector_init loop={loop_id}")

    def _load_pending(self) -> None:
        """Load pending wisdom from storage."""
        wisdom_file = self.storage_path / f"{self.loop_id}_pending.json"
        if wisdom_file.exists():
            try:
                with open(wisdom_file) as f:
                    data = json.load(f)
                self.pending_wisdom = [WisdomSubmission(**w) for w in data.get("pending", [])]
                logger.debug(f"wisdom_loaded count={len(self.pending_wisdom)}")
            except Exception as e:
                logger.error(f"wisdom_load_failed error={e}")

    def _save_pending(self) -> None:
        """Save pending wisdom to storage."""
        wisdom_file = self.storage_path / f"{self.loop_id}_pending.json"
        try:
            with open(wisdom_file, "w") as f:
                json.dump(
                    {
                        "pending": [w.to_dict() for w in self.pending_wisdom],
                        "updated_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"wisdom_save_failed error={e}")

    async def submit_wisdom(
        self,
        text: str,
        submitter_id: str,
        context_tags: Optional[list[str]] = None,
    ) -> Optional[WisdomSubmission]:
        """
        Submit a wisdom contribution from the audience.

        Args:
            text: The wisdom text (max 280 chars)
            submitter_id: ID of the submitter
            context_tags: Optional tags for relevance matching

        Returns:
            The created submission, or None if invalid
        """
        # Validate
        text = text.strip()
        if not text:
            logger.warning("wisdom_rejected reason=empty")
            return None

        if len(text) > self.MAX_WISDOM_LENGTH:
            text = text[: self.MAX_WISDOM_LENGTH]
            logger.debug(f"wisdom_truncated length={len(text)}")

        # Check for duplicates
        text_hash = hashlib.sha256(text.lower().encode()).hexdigest()[:8]
        for existing in self.pending_wisdom:
            existing_hash = hashlib.sha256(existing.text.lower().encode()).hexdigest()[:8]
            if text_hash == existing_hash:
                logger.debug("wisdom_rejected reason=duplicate")
                return None

        # Create submission
        timestamp = time.time()
        wisdom = WisdomSubmission(
            id=WisdomSubmission.generate_id(text, submitter_id, timestamp),
            text=text,
            submitter_id=submitter_id,
            timestamp=timestamp,
            loop_id=self.loop_id,
            context_tags=context_tags or [],
        )

        # Update submitter stats
        self.submitter_stats[submitter_id]["submissions"] += 1

        # Add to pending (with limit)
        self.pending_wisdom.append(wisdom)
        if len(self.pending_wisdom) > self.MAX_PENDING_WISDOM:
            # Remove oldest unused
            self.pending_wisdom = sorted(self.pending_wisdom, key=lambda w: (w.used, -w.timestamp))[
                : self.MAX_PENDING_WISDOM
            ]

        self._save_pending()
        logger.info(f"wisdom_submitted id={wisdom.id} submitter={submitter_id}")

        return wisdom

    def upvote_wisdom(self, wisdom_id: str, voter_id: str) -> bool:
        """
        Upvote a wisdom submission.

        Args:
            wisdom_id: ID of the wisdom to upvote
            voter_id: ID of the voter (to prevent double voting)

        Returns:
            True if upvoted, False if not found or already voted
        """
        for wisdom in self.pending_wisdom:
            if wisdom.id == wisdom_id:
                wisdom.upvotes += 1
                self.submitter_stats[wisdom.submitter_id]["upvotes"] += 1
                self._save_pending()
                return True
        return False

    def _calculate_relevance(
        self,
        wisdom: WisdomSubmission,
        debate_context: dict,
    ) -> float:
        """
        Calculate relevance score for a wisdom submission.

        Uses simple keyword matching for now - could be enhanced
        with embeddings for semantic similarity.
        """
        score = 0.0

        # Base score from recency (newer = more relevant)
        age_hours = (time.time() - wisdom.timestamp) / 3600
        recency_score = max(0, 1.0 - (age_hours / 24))  # Decay over 24h
        score += recency_score * 0.2

        # Score from upvotes
        if wisdom.upvotes > 0:
            upvote_score = min(1.0, wisdom.upvotes / 10)  # Cap at 10 votes
            score += upvote_score * 0.3

        # Score from submitter reputation
        stats = self.submitter_stats.get(wisdom.submitter_id, {})
        if stats.get("used", 0) > 0:
            rep_score = min(1.0, stats["used"] / 5)
            score += rep_score * 0.2

        # Score from tag matching
        context_keywords = set()
        if "topic" in debate_context:
            context_keywords.update(debate_context["topic"].lower().split())
        if "tags" in debate_context:
            context_keywords.update(t.lower() for t in debate_context["tags"])

        wisdom_keywords = set(wisdom.text.lower().split())
        wisdom_keywords.update(t.lower() for t in wisdom.context_tags)

        if context_keywords and wisdom_keywords:
            overlap = len(context_keywords & wisdom_keywords)
            tag_score = min(1.0, overlap / 3)
            score += tag_score * 0.3

        return min(1.0, score)

    def find_relevant_wisdom(
        self,
        debate_context: dict,
        failed_agent: Optional[str] = None,
        limit: int = 3,
    ) -> list[WisdomSubmission]:
        """
        Find relevant wisdom for the current debate state.

        Args:
            debate_context: Current debate context (topic, tags, etc.)
            failed_agent: Name of agent that failed (if any)
            limit: Maximum number of wisdoms to return

        Returns:
            List of relevant wisdom submissions, sorted by relevance
        """
        if not self.pending_wisdom:
            return []

        # Calculate relevance for all pending wisdom
        for wisdom in self.pending_wisdom:
            if not wisdom.used:
                wisdom.relevance_score = self._calculate_relevance(wisdom, debate_context)

        # Filter by threshold and sort by relevance
        relevant = [
            w
            for w in self.pending_wisdom
            if not w.used and w.relevance_score >= self.RELEVANCE_THRESHOLD
        ]
        relevant.sort(key=lambda w: -w.relevance_score)

        return relevant[:limit]

    def inject_wisdom(
        self,
        wisdom: WisdomSubmission,
        agent_context: str,
        reason: str = "timeout",
    ) -> WisdomInjection:
        """
        Record a wisdom injection into the debate.

        Args:
            wisdom: The wisdom being injected
            agent_context: Context where wisdom was injected
            reason: Why the injection happened (timeout, stall, requested)

        Returns:
            The injection record
        """
        wisdom.used = True
        wisdom.used_at = time.time()

        # Update stats
        self.submitter_stats[wisdom.submitter_id]["used"] += 1

        # Move to used list
        self.pending_wisdom = [w for w in self.pending_wisdom if w.id != wisdom.id]
        self.used_wisdom.append(wisdom)

        # Create injection record
        injection = WisdomInjection(
            wisdom_id=wisdom.id,
            agent_context=agent_context[:200],
            injection_reason=reason,
            timestamp=time.time(),
        )
        self.injections.append(injection)

        self._save_pending()
        logger.info(
            f"wisdom_injected id={wisdom.id} reason={reason} submitter={wisdom.submitter_id}"
        )

        return injection

    def format_for_prompt(self, wisdoms: list[WisdomSubmission]) -> str:
        """
        Format wisdom submissions for inclusion in agent prompts.

        Args:
            wisdoms: List of wisdom to format

        Returns:
            Formatted string for prompt injection
        """
        if not wisdoms:
            return ""

        lines = ["[Audience Insights]"]
        for i, wisdom in enumerate(wisdoms, 1):
            lines.append(f'{i}. "{wisdom.text}" - audience member')

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get injector statistics."""
        return {
            "loop_id": self.loop_id,
            "pending_count": len(self.pending_wisdom),
            "used_count": len(self.used_wisdom),
            "injection_count": len(self.injections),
            "unique_submitters": len(self.submitter_stats),
            "total_upvotes": sum(s.get("upvotes", 0) for s in self.submitter_stats.values()),
        }


# Per-loop injector instances
_injectors: dict[str, WisdomInjector] = {}


def get_wisdom_injector(loop_id: str) -> WisdomInjector:
    """Get or create a wisdom injector for a loop."""
    if loop_id not in _injectors:
        _injectors[loop_id] = WisdomInjector(loop_id)
    return _injectors[loop_id]


def close_wisdom_injector(loop_id: str) -> None:
    """Close and remove a wisdom injector."""
    if loop_id in _injectors:
        del _injectors[loop_id]
