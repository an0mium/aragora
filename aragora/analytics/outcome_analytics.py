"""
Decision Outcome Analytics.

Tracks decision outcomes over time for the analytics dashboard:
- Consensus rates and trends
- Round efficiency (average rounds to conclusion)
- Agent contribution scoring
- Decision quality trends
- Topic distribution analysis
- Individual debate outcome summaries

Complements debate_analytics.py (raw debate metrics) by focusing on
decision quality and outcome patterns across the organization.

Usage:
    from aragora.analytics.outcome_analytics import (
        OutcomeAnalytics,
        get_outcome_analytics,
    )

    analytics = get_outcome_analytics()
    rate = await analytics.get_consensus_rate(period="30d")
    trend = await analytics.get_decision_quality_trend(period="90d")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Period string to timedelta mapping
_PERIOD_DELTAS: dict[str, timedelta] = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
    "365d": timedelta(days=365),
}


def _parse_period(period: str) -> timedelta:
    """Parse a period string into a timedelta.

    Args:
        period: Period string like "30d", "90d", "7d", "24h", "365d".

    Returns:
        Corresponding timedelta.

    Raises:
        ValueError: If period string is not recognized.
    """
    delta = _PERIOD_DELTAS.get(period)
    if delta is None:
        raise ValueError(
            f"Invalid period '{period}'. Valid periods: {', '.join(sorted(_PERIOD_DELTAS))}"
        )
    return delta


@dataclass
class QualityDataPoint:
    """A single data point in the decision quality trend."""

    timestamp: datetime
    consensus_rate: float
    avg_confidence: float
    avg_rounds: float
    debate_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "consensus_rate": round(self.consensus_rate, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_rounds": round(self.avg_rounds, 2),
            "debate_count": self.debate_count,
        }


@dataclass
class AgentContribution:
    """Contribution metrics for a single agent."""

    agent_id: str
    agent_name: str
    debates_participated: int = 0
    consensus_contributions: int = 0
    avg_confidence: float = 0.0
    contribution_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "debates_participated": self.debates_participated,
            "consensus_contributions": self.consensus_contributions,
            "avg_confidence": round(self.avg_confidence, 4),
            "contribution_score": round(self.contribution_score, 4),
        }


@dataclass
class OutcomeSummary:
    """Summary of a single debate's outcome."""

    debate_id: str
    task: str
    consensus_reached: bool
    confidence: float
    rounds: int
    agents: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    topic: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "debate_id": self.debate_id,
            "task": self.task,
            "consensus_reached": self.consensus_reached,
            "confidence": round(self.confidence, 4),
            "rounds": self.rounds,
            "agents": self.agents,
            "duration_seconds": round(self.duration_seconds, 2),
            "topic": self.topic,
            "created_at": self.created_at,
        }


class OutcomeAnalytics:
    """
    Decision outcome analytics service.

    Analyzes debate outcomes to surface patterns in consensus rates,
    agent contributions, decision quality trends, and topic distribution.

    Uses DebateAnalytics as the underlying data source and adds
    outcome-focused aggregation and scoring on top.

    Example:
        analytics = OutcomeAnalytics()
        rate = await analytics.get_consensus_rate(period="30d")
        scores = await analytics.get_agent_contribution_scores(period="30d")
        trend = await analytics.get_decision_quality_trend(period="90d")
    """

    def __init__(self, db_path: str | None = None):
        """Initialize outcome analytics.

        Args:
            db_path: Optional path to the SQLite database.
                     If None, uses the shared DebateAnalytics instance.
        """
        self._db_path = db_path
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    def _get_debate_analytics(self):
        """Lazy import and retrieve DebateAnalytics instance."""
        from aragora.analytics.debate_analytics import (
            DebateAnalytics,
            get_debate_analytics,
        )

        if self._db_path:
            return DebateAnalytics(db_path=self._db_path)
        return get_debate_analytics()

    def _get_cached(self, key: str) -> Any | None:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now(timezone.utc) - timestamp < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Cache a value."""
        self._cache[key] = (value, datetime.now(timezone.utc))

    async def get_consensus_rate(self, period: str = "30d") -> float:
        """Get the percentage of debates that reached consensus.

        Args:
            period: Time period to analyze (e.g., "24h", "7d", "30d", "90d").

        Returns:
            Consensus rate as a float between 0.0 and 1.0.
        """
        cache_key = f"consensus_rate:{period}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        delta = _parse_period(period)
        da = self._get_debate_analytics()
        stats = await da.get_debate_stats(days_back=delta.days or 1)

        rate = stats.consensus_rate
        self._set_cached(cache_key, rate)
        return rate

    async def get_average_rounds(self, period: str = "30d") -> float:
        """Get the mean number of rounds to conclusion.

        Args:
            period: Time period to analyze.

        Returns:
            Average number of rounds as a float.
        """
        cache_key = f"avg_rounds:{period}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        delta = _parse_period(period)
        da = self._get_debate_analytics()
        stats = await da.get_debate_stats(days_back=delta.days or 1)

        avg = stats.avg_rounds
        self._set_cached(cache_key, avg)
        return avg

    async def get_agent_contribution_scores(
        self, period: str = "30d"
    ) -> dict[str, AgentContribution]:
        """Get contribution scores for agents that participated in debates.

        Scores are based on: participation frequency, consensus contribution
        rate, and average confidence when the agent is involved.

        Args:
            period: Time period to analyze.

        Returns:
            Dict mapping agent_id to AgentContribution dataclass.
        """
        cache_key = f"contributions:{period}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        delta = _parse_period(period)
        da = self._get_debate_analytics()
        leaderboard = await da.get_agent_leaderboard(limit=50, days_back=delta.days or 1)

        contributions: dict[str, AgentContribution] = {}
        for agent in leaderboard:
            debates = agent.debates_participated
            consensus = agent.consensus_contributions
            participation_score = min(debates / 10.0, 1.0)
            consensus_score = consensus / debates if debates > 0 else 0.0
            vote_score = agent.vote_ratio

            # Weighted combination: 30% participation, 40% consensus, 30% votes
            score = 0.3 * participation_score + 0.4 * consensus_score + 0.3 * vote_score

            contributions[agent.agent_id] = AgentContribution(
                agent_id=agent.agent_id,
                agent_name=agent.agent_name,
                debates_participated=debates,
                consensus_contributions=consensus,
                avg_confidence=vote_score,
                contribution_score=score,
            )

        self._set_cached(cache_key, contributions)
        return contributions

    async def get_decision_quality_trend(self, period: str = "90d") -> list[QualityDataPoint]:
        """Get decision quality metrics over time.

        Divides the period into weekly buckets and computes quality
        metrics for each bucket.

        Args:
            period: Time period to analyze.

        Returns:
            List of QualityDataPoint, one per week in the period.
        """
        cache_key = f"quality_trend:{period}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        delta = _parse_period(period)
        da = self._get_debate_analytics()

        # Divide into weekly buckets
        total_days = delta.days or 1
        bucket_size = 7
        num_buckets = max(total_days // bucket_size, 1)

        points: list[QualityDataPoint] = []
        now = datetime.now(timezone.utc)

        for i in range(num_buckets):
            bucket_end_offset = i * bucket_size
            bucket_start_offset = bucket_end_offset + bucket_size

            # Use debate_analytics to get stats for this bucket window
            # We query the full window and subtract to approximate buckets
            stats_full = await da.get_debate_stats(days_back=bucket_start_offset or 1)
            stats_partial = await da.get_debate_stats(days_back=bucket_end_offset or 1)

            bucket_debates = stats_full.total_debates - stats_partial.total_debates
            bucket_consensus = stats_full.consensus_reached - stats_partial.consensus_reached

            if bucket_debates > 0:
                bucket_rate = bucket_consensus / bucket_debates
            else:
                bucket_rate = 0.0

            bucket_ts = now - timedelta(days=bucket_start_offset)

            points.append(
                QualityDataPoint(
                    timestamp=bucket_ts,
                    consensus_rate=bucket_rate,
                    avg_confidence=stats_full.consensus_rate,
                    avg_rounds=stats_full.avg_rounds,
                    debate_count=bucket_debates,
                )
            )

        # Oldest first
        points.reverse()
        self._set_cached(cache_key, points)
        return points

    async def get_topic_distribution(self, period: str = "30d") -> dict[str, int]:
        """Get distribution of debate topics in the period.

        Extracts topic keywords from debate tasks and counts occurrences.

        Args:
            period: Time period to analyze.

        Returns:
            Dict mapping topic string to count.
        """
        cache_key = f"topics:{period}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        delta = _parse_period(period)
        da = self._get_debate_analytics()

        import sqlite3

        topics: dict[str, int] = {}
        period_start = datetime.now(timezone.utc) - delta

        try:
            with sqlite3.connect(da.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT agents, protocol
                    FROM debate_records
                    WHERE created_at >= ?
                    """,
                    (period_start.isoformat(),),
                )
                for row in cursor.fetchall():
                    protocol = row["protocol"] or "general"
                    topics[protocol] = topics.get(protocol, 0) + 1
        except (sqlite3.Error, OSError) as e:
            logger.warning("Failed to get topic distribution: %s", e)

        self._set_cached(cache_key, topics)
        return topics

    async def get_outcome_summary(self, debate_id: str) -> OutcomeSummary | None:
        """Get a detailed outcome summary for a single debate.

        Args:
            debate_id: The ID of the debate.

        Returns:
            OutcomeSummary dataclass, or None if debate not found.
        """
        da = self._get_debate_analytics()

        import json
        import sqlite3

        try:
            with sqlite3.connect(da.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT debate_id, status, rounds, consensus_reached,
                           duration_seconds, agents, protocol, created_at
                    FROM debate_records
                    WHERE debate_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (debate_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                agents = json.loads(row["agents"]) if row["agents"] else []
                return OutcomeSummary(
                    debate_id=row["debate_id"],
                    task=row["protocol"] or "",
                    consensus_reached=bool(row["consensus_reached"]),
                    confidence=1.0 if row["consensus_reached"] else 0.0,
                    rounds=row["rounds"] or 0,
                    agents=agents,
                    duration_seconds=row["duration_seconds"] or 0.0,
                    topic=row["protocol"] or "",
                    created_at=row["created_at"] or "",
                )
        except (sqlite3.Error, OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to get outcome summary for %s: %s", debate_id, e)
            return None


# Global instance
_outcome_analytics: OutcomeAnalytics | None = None
_lock = threading.Lock()


def get_outcome_analytics(db_path: str | None = None) -> OutcomeAnalytics:
    """Get or create global outcome analytics instance."""
    global _outcome_analytics
    if _outcome_analytics is None:
        with _lock:
            if _outcome_analytics is None:
                _outcome_analytics = OutcomeAnalytics(db_path=db_path)
    return _outcome_analytics


__all__ = [
    "OutcomeAnalytics",
    "QualityDataPoint",
    "OutcomeSummary",
    "AgentContribution",
    "get_outcome_analytics",
]
