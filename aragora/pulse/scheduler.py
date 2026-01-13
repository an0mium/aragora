"""
Pulse Debate Scheduler.

Automates debate creation from trending topics with:
- Configurable polling intervals
- Duplicate detection via topic hashing
- Rate limiting to prevent debate flooding
- Analytics on topic → debate success correlation

Usage:
    scheduler = PulseDebateScheduler(pulse_manager, debate_store)
    await scheduler.start()

    # Later
    await scheduler.stop()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from aragora.exceptions import ConfigurationError
from aragora.pulse.ingestor import PulseManager, TrendingTopic
from aragora.pulse.store import ScheduledDebateRecord, ScheduledDebateStore

logger = logging.getLogger(__name__)


class SchedulerState(str, Enum):
    """State of the pulse debate scheduler."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class SchedulerConfig:
    """Configuration for the PulseDebateScheduler."""

    # Polling
    poll_interval_seconds: int = 300  # 5 minutes
    platforms: List[str] = field(default_factory=lambda: ["hackernews", "reddit"])

    # Rate limiting
    max_debates_per_hour: int = 6
    min_interval_between_debates: int = 600  # 10 minutes

    # Topic filtering
    min_volume_threshold: int = 100
    min_controversy_score: float = 0.3
    allowed_categories: List[str] = field(
        default_factory=lambda: ["tech", "ai", "science", "programming"]
    )
    blocked_categories: List[str] = field(default_factory=lambda: ["politics", "religion"])

    # Deduplication
    dedup_window_hours: int = 24

    # Debate settings
    debate_rounds: int = 3
    consensus_threshold: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "poll_interval_seconds": self.poll_interval_seconds,
            "platforms": self.platforms,
            "max_debates_per_hour": self.max_debates_per_hour,
            "min_interval_between_debates": self.min_interval_between_debates,
            "min_volume_threshold": self.min_volume_threshold,
            "min_controversy_score": self.min_controversy_score,
            "allowed_categories": self.allowed_categories,
            "blocked_categories": self.blocked_categories,
            "dedup_window_hours": self.dedup_window_hours,
            "debate_rounds": self.debate_rounds,
            "consensus_threshold": self.consensus_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchedulerConfig":
        """Create from dictionary."""
        return cls(
            poll_interval_seconds=data.get("poll_interval_seconds", 300),
            platforms=data.get("platforms", ["hackernews", "reddit"]),
            max_debates_per_hour=data.get("max_debates_per_hour", 6),
            min_interval_between_debates=data.get("min_interval_between_debates", 600),
            min_volume_threshold=data.get("min_volume_threshold", 100),
            min_controversy_score=data.get("min_controversy_score", 0.3),
            allowed_categories=data.get(
                "allowed_categories", ["tech", "ai", "science", "programming"]
            ),
            blocked_categories=data.get("blocked_categories", ["politics", "religion"]),
            dedup_window_hours=data.get("dedup_window_hours", 24),
            debate_rounds=data.get("debate_rounds", 3),
            consensus_threshold=data.get("consensus_threshold", 0.7),
        )


@dataclass
class TopicScore:
    """Scored topic for debate selection."""

    topic: TrendingTopic
    score: float
    reasons: List[str] = field(default_factory=list)

    @property
    def is_viable(self) -> bool:
        """Check if the topic is viable for debate."""
        return self.score > 0


@dataclass
class SchedulerMetrics:
    """Runtime metrics for the scheduler."""

    polls_completed: int = 0
    topics_evaluated: int = 0
    topics_filtered: int = 0
    debates_created: int = 0
    debates_failed: int = 0
    duplicates_skipped: int = 0
    last_poll_at: Optional[float] = None
    last_debate_at: Optional[float] = None
    start_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        uptime = None
        if self.start_time:
            uptime = time.time() - self.start_time

        return {
            "polls_completed": self.polls_completed,
            "topics_evaluated": self.topics_evaluated,
            "topics_filtered": self.topics_filtered,
            "debates_created": self.debates_created,
            "debates_failed": self.debates_failed,
            "duplicates_skipped": self.duplicates_skipped,
            "last_poll_at": self.last_poll_at,
            "last_debate_at": self.last_debate_at,
            "uptime_seconds": uptime,
        }


# Type alias for debate creator callback
DebateCreatorFn = Callable[
    [str, int, float],  # topic_text, rounds, consensus_threshold
    Coroutine[Any, Any, Optional[Dict[str, Any]]],  # Returns debate result or None
]


class TopicSelector:
    """Selects and scores topics for debate suitability."""

    # Keywords that indicate controversy/debate-worthiness
    CONTROVERSY_KEYWORDS = [
        "should",
        "vs",
        "versus",
        "debate",
        "controversy",
        "opinion",
        "disagree",
        "argument",
        "controversial",
        "battle",
        "fight",
        "war",
        "challenge",
        "question",
        "problem",
        "issue",
        "concern",
    ]

    # Keywords for boosting certain topics
    BOOST_KEYWORDS = [
        "ai",
        "artificial intelligence",
        "machine learning",
        "ethics",
        "future",
        "impact",
        "change",
        "new",
        "breakthrough",
        "revolutionary",
    ]

    def __init__(self, config: SchedulerConfig):
        self.config = config

    def score_topic(self, topic: TrendingTopic) -> TopicScore:
        """Score a topic for debate suitability.

        Returns:
            TopicScore with score and reasons
        """
        score = 0.0
        reasons: List[str] = []

        # Check category filters
        if self.config.allowed_categories:
            if topic.category in self.config.allowed_categories:
                score += 0.3
                reasons.append(f"category '{topic.category}' is allowed")
            elif topic.category not in self.config.blocked_categories:
                score += 0.1
                reasons.append(f"category '{topic.category}' is neutral")
            else:
                return TopicScore(topic, -1.0, [f"category '{topic.category}' is blocked"])

        if topic.category in self.config.blocked_categories:
            return TopicScore(topic, -1.0, [f"category '{topic.category}' is blocked"])

        # Check volume threshold
        if topic.volume >= self.config.min_volume_threshold:
            volume_score = min(0.3, topic.volume / 10000)  # Cap at 0.3
            score += volume_score
            reasons.append(f"volume {topic.volume} meets threshold")
        else:
            return TopicScore(
                topic,
                -1.0,
                [f"volume {topic.volume} below threshold {self.config.min_volume_threshold}"],
            )

        # Calculate controversy score
        controversy = self._calculate_controversy_score(topic.topic)
        if controversy >= self.config.min_controversy_score:
            score += controversy * 0.4
            reasons.append(f"controversy score {controversy:.2f}")
        else:
            # Low controversy but not disqualifying
            score += controversy * 0.2
            reasons.append(f"low controversy score {controversy:.2f}")

        # Boost for certain keywords
        boost = self._calculate_boost_score(topic.topic)
        if boost > 0:
            score += boost
            reasons.append(f"keyword boost +{boost:.2f}")

        return TopicScore(topic, score, reasons)

    def _calculate_controversy_score(self, text: str) -> float:
        """Calculate how controversial/debate-worthy a topic is."""
        text_lower = text.lower()
        matches = sum(1 for kw in self.CONTROVERSY_KEYWORDS if kw in text_lower)
        # Normalize to 0-1 range
        return min(1.0, matches / 3)

    def _calculate_boost_score(self, text: str) -> float:
        """Calculate boost score from topic keywords."""
        text_lower = text.lower()
        matches = sum(1 for kw in self.BOOST_KEYWORDS if kw in text_lower)
        return min(0.2, matches * 0.05)

    def select_best_topics(
        self,
        topics: List[TrendingTopic],
        limit: int = 5,
    ) -> List[TopicScore]:
        """Select the best topics for debate.

        Args:
            topics: List of trending topics to evaluate
            limit: Maximum number of topics to return

        Returns:
            List of TopicScore objects sorted by score descending
        """
        scored = [self.score_topic(t) for t in topics]

        # Filter to viable topics only
        viable = [s for s in scored if s.is_viable]

        # Sort by score descending
        viable.sort(key=lambda x: x.score, reverse=True)

        return viable[:limit]


class PulseDebateScheduler:
    """
    Background scheduler for trending topic debates.

    Polls PulseManager for trending topics, selects debate-worthy ones,
    and creates debates automatically with rate limiting and dedup.

    Usage:
        # Setup
        pulse_manager = PulseManager()
        pulse_manager.add_ingestor("hackernews", HackerNewsIngestor())

        store = ScheduledDebateStore("data/scheduled_debates.db")
        scheduler = PulseDebateScheduler(pulse_manager, store)

        # Define how debates are created
        async def create_debate(topic: str, rounds: int, threshold: float):
            arena = Arena(Environment(task=topic), agents, protocol)
            result = await arena.run()
            return {"debate_id": result.id, ...}

        scheduler.set_debate_creator(create_debate)

        # Start
        await scheduler.start()

        # Later
        await scheduler.stop()
    """

    def __init__(
        self,
        pulse_manager: PulseManager,
        store: ScheduledDebateStore,
        config: Optional[SchedulerConfig] = None,
    ):
        """Initialize the scheduler.

        Args:
            pulse_manager: PulseManager instance for fetching trending topics
            store: ScheduledDebateStore for persistence and deduplication
            config: Optional configuration (uses defaults if not provided)
        """
        self.pulse_manager = pulse_manager
        self.store = store
        self.config = config or SchedulerConfig()

        self._state = SchedulerState.STOPPED
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._metrics = SchedulerMetrics()
        self._topic_selector = TopicSelector(self.config)
        self._debate_creator: Optional[DebateCreatorFn] = None
        self._run_id = ""
        self._debates_this_hour: List[float] = []

        logger.info("PulseDebateScheduler initialized")

    @property
    def state(self) -> SchedulerState:
        """Current scheduler state."""
        return self._state

    @property
    def metrics(self) -> SchedulerMetrics:
        """Current runtime metrics."""
        return self._metrics

    def set_debate_creator(self, creator: DebateCreatorFn) -> None:
        """Set the callback function for creating debates.

        The creator function should accept:
            topic_text: str - The topic to debate
            rounds: int - Number of debate rounds
            consensus_threshold: float - Threshold for consensus

        And return a dict with at least:
            debate_id: str - Unique debate identifier
            consensus_reached: bool
            confidence: float
            rounds_used: int
        """
        self._debate_creator = creator
        logger.info("Debate creator callback set")

    async def start(self) -> None:
        """Start the scheduler loop.

        Raises:
            RuntimeError: If already running or no debate creator set
        """
        if self._state == SchedulerState.RUNNING:
            logger.warning("Scheduler already running")
            return

        if not self._debate_creator:
            raise ConfigurationError(
                component="PulseDebateScheduler",
                reason="No debate creator set. Call set_debate_creator() first",
            )

        self._state = SchedulerState.RUNNING
        self._stop_event.clear()
        self._run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        self._metrics = SchedulerMetrics(start_time=time.time())

        logger.info(f"Starting PulseDebateScheduler (run_id={self._run_id})")

        self._task = asyncio.create_task(self._scheduler_loop())

    async def stop(self, graceful: bool = True) -> None:
        """Stop the scheduler.

        Args:
            graceful: If True, wait for current operation to complete
        """
        if self._state == SchedulerState.STOPPED:
            logger.warning("Scheduler already stopped")
            return

        logger.info(f"Stopping PulseDebateScheduler (graceful={graceful})")
        self._state = SchedulerState.STOPPED
        self._stop_event.set()

        if self._task and not self._task.done():
            if graceful:
                # Wait for task to complete with timeout
                try:
                    await asyncio.wait_for(self._task, timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning("Graceful stop timed out, cancelling task")
                    self._task.cancel()
            else:
                self._task.cancel()

            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("PulseDebateScheduler stopped")

    async def pause(self) -> None:
        """Pause the scheduler (stops polling but keeps state)."""
        if self._state != SchedulerState.RUNNING:
            logger.warning(f"Cannot pause scheduler in state {self._state}")
            return

        self._state = SchedulerState.PAUSED
        logger.info("PulseDebateScheduler paused")

    async def resume(self) -> None:
        """Resume a paused scheduler."""
        if self._state != SchedulerState.PAUSED:
            logger.warning(f"Cannot resume scheduler in state {self._state}")
            return

        self._state = SchedulerState.RUNNING
        logger.info("PulseDebateScheduler resumed")

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update scheduler configuration.

        Args:
            updates: Dictionary of config keys to update
        """
        config_dict = self.config.to_dict()
        config_dict.update(updates)
        self.config = SchedulerConfig.from_dict(config_dict)
        self._topic_selector = TopicSelector(self.config)
        logger.info(f"Scheduler config updated: {list(updates.keys())}")

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status.

        Returns:
            Dict with state, config, and metrics
        """
        return {
            "state": self._state.value,
            "run_id": self._run_id,
            "config": self.config.to_dict(),
            "metrics": self._metrics.to_dict(),
        }

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while not self._stop_event.is_set():
            try:
                if self._state == SchedulerState.RUNNING:
                    await self._poll_and_create()

                # Wait for next poll interval or stop event
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.poll_interval_seconds,
                    )
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    pass  # Continue to next poll

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                # Wait before retrying to avoid tight error loop
                await asyncio.sleep(60)

        logger.info("Scheduler loop ended")

    async def _poll_and_create(self) -> None:
        """Poll for trending topics and create debates."""
        logger.debug("Polling for trending topics...")

        # Check hourly rate limit
        if not self._can_create_debate():
            logger.debug("Hourly debate limit reached, skipping poll")
            return

        try:
            # Fetch trending topics
            topics = await self.pulse_manager.get_trending_topics(
                platforms=self.config.platforms,
                limit_per_platform=10,
                filters={
                    "min_volume": self.config.min_volume_threshold,
                    "skip_toxic": True,
                },
            )

            self._metrics.polls_completed += 1
            self._metrics.last_poll_at = time.time()
            self._metrics.topics_evaluated += len(topics)

            if not topics:
                logger.debug("No trending topics found")
                return

            # Score and select best topics
            scored_topics = self._topic_selector.select_best_topics(topics, limit=5)
            self._metrics.topics_filtered += len(topics) - len(scored_topics)

            if not scored_topics:
                logger.debug("No suitable topics after scoring")
                return

            # Try each topic until we find one that's not a duplicate
            for scored in scored_topics:
                topic = scored.topic

                # Check deduplication
                if self.store.is_duplicate(topic.topic, self.config.dedup_window_hours):
                    logger.debug(f"Skipping duplicate topic: {topic.topic[:50]}...")
                    self._metrics.duplicates_skipped += 1
                    continue

                # Create debate
                await self._create_debate(topic, scored)
                break  # Only create one debate per poll

        except Exception as e:
            logger.error(f"Error in poll_and_create: {e}", exc_info=True)

    async def _create_debate(self, topic: TrendingTopic, scored: TopicScore) -> None:
        """Create a debate for the given topic."""
        logger.info(
            f"Creating debate for topic: {topic.topic[:50]}... "
            f"(platform={topic.platform}, score={scored.score:.2f})"
        )

        debate_id = f"pulse-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        record_id = f"sched-{int(time.time())}-{uuid.uuid4().hex[:6]}"

        try:
            # Call the debate creator
            result = await self._debate_creator(
                topic.to_debate_prompt(),
                self.config.debate_rounds,
                self.config.consensus_threshold,
            )

            if result:
                # Record successful debate
                debate_id = result.get("debate_id", debate_id)
                consensus_reached = result.get("consensus_reached", False)
                confidence = result.get("confidence", 0.0)
                rounds_used = result.get("rounds_used", 0)

                record = ScheduledDebateRecord(
                    id=record_id,
                    topic_hash=self.store.hash_topic(topic.topic),
                    topic_text=topic.topic,
                    platform=topic.platform,
                    category=topic.category,
                    volume=topic.volume,
                    debate_id=debate_id,
                    created_at=time.time(),
                    consensus_reached=consensus_reached,
                    confidence=confidence,
                    rounds_used=rounds_used,
                    scheduler_run_id=self._run_id,
                )
                self.store.record_scheduled_debate(record)

                self._metrics.debates_created += 1
                self._metrics.last_debate_at = time.time()
                self._debates_this_hour.append(time.time())

                logger.info(
                    f"Debate created: {debate_id} "
                    f"(consensus={consensus_reached}, confidence={confidence:.2f})"
                )
            else:
                # Debate creation returned None
                self._metrics.debates_failed += 1
                logger.warning("Debate creator returned None")

        except Exception as e:
            self._metrics.debates_failed += 1
            logger.error(f"Failed to create debate: {e}", exc_info=True)

            # Still record the attempt
            record = ScheduledDebateRecord(
                id=record_id,
                topic_hash=self.store.hash_topic(topic.topic),
                topic_text=topic.topic,
                platform=topic.platform,
                category=topic.category,
                volume=topic.volume,
                debate_id=None,
                created_at=time.time(),
                consensus_reached=None,
                confidence=None,
                rounds_used=0,
                scheduler_run_id=self._run_id,
            )
            self.store.record_scheduled_debate(record)

    def _can_create_debate(self) -> bool:
        """Check if we can create a debate based on rate limits."""
        now = time.time()
        hour_ago = now - 3600

        # Remove debates older than 1 hour
        self._debates_this_hour = [t for t in self._debates_this_hour if t > hour_ago]

        # Check hourly limit
        if len(self._debates_this_hour) >= self.config.max_debates_per_hour:
            return False

        # Check minimum interval
        if self._debates_this_hour:
            last_debate = max(self._debates_this_hour)
            if now - last_debate < self.config.min_interval_between_debates:
                return False

        return True


__all__ = [
    "PulseDebateScheduler",
    "SchedulerConfig",
    "SchedulerState",
    "SchedulerMetrics",
    "TopicSelector",
    "TopicScore",
]
