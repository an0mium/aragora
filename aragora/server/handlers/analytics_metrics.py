"""
Analytics Dashboard Metrics endpoint handlers.

Provides REST APIs for analytics dashboard showing debate metrics and agent performance:

Debate Analytics:
- GET /api/analytics/debates/overview - Total debates, consensus rate, avg rounds
- GET /api/analytics/debates/trends - Debates over time (daily/weekly/monthly)
- GET /api/analytics/debates/topics - Topic distribution
- GET /api/analytics/debates/outcomes - Win/loss/draw distribution

Agent Performance:
- GET /api/analytics/agents/leaderboard - ELO rankings with win rates
- GET /api/analytics/agents/{agent_id}/performance - Individual agent stats
- GET /api/analytics/agents/comparison - Compare multiple agents
- GET /api/analytics/agents/trends - Agent performance over time

Usage Analytics:
- GET /api/analytics/usage/tokens - Token consumption trends
- GET /api/analytics/usage/costs - Cost breakdown by provider/model
- GET /api/analytics/usage/active_users - Active user counts
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from aragora.config import CACHE_TTL_ANALYTICS

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    ttl_cache,
)
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Valid time granularities
VALID_GRANULARITIES = {"daily", "weekly", "monthly"}

# Valid time ranges for trend queries
VALID_TIME_RANGES = {"7d", "14d", "30d", "90d", "180d", "365d", "all"}


def _parse_time_range(time_range: str) -> Optional[datetime]:
    """Parse time range string into a start datetime.

    Args:
        time_range: Time range string like '7d', '30d', '365d', or 'all'

    Returns:
        datetime for start of range, or None for 'all'
    """
    if time_range == "all":
        return None

    match = re.match(r"^(\d+)d$", time_range)
    if not match:
        return datetime.now(timezone.utc) - timedelta(days=30)  # Default

    days = int(match.group(1))
    return datetime.now(timezone.utc) - timedelta(days=days)


def _group_by_time(
    items: List[Dict[str, Any]],
    timestamp_key: str,
    granularity: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group items by time bucket based on granularity.

    Args:
        items: List of items with timestamp field
        timestamp_key: Key name for timestamp in items
        granularity: 'daily', 'weekly', or 'monthly'

    Returns:
        Dict mapping bucket key to list of items
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in items:
        ts = item.get(timestamp_key)
        if not ts:
            continue

        # Parse timestamp if string
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
        elif isinstance(ts, datetime):
            dt = ts
        else:
            continue

        # Generate bucket key based on granularity
        if granularity == "daily":
            key = dt.strftime("%Y-%m-%d")
        elif granularity == "weekly":
            # ISO week number
            key = dt.strftime("%Y-W%W")
        else:  # monthly
            key = dt.strftime("%Y-%m")

        groups[key].append(item)

    return dict(groups)


class AnalyticsMetricsHandler(BaseHandler):
    """Handler for analytics metrics dashboard endpoints."""

    ROUTES = [
        # Debate Analytics
        "/api/v1/analytics/debates/overview",
        "/api/v1/analytics/debates/trends",
        "/api/v1/analytics/debates/topics",
        "/api/v1/analytics/debates/outcomes",
        # Agent Performance
        "/api/v1/analytics/agents/leaderboard",
        "/api/v1/analytics/agents/comparison",
        "/api/v1/analytics/agents/trends",
        # Usage Analytics
        "/api/v1/analytics/usage/tokens",
        "/api/v1/analytics/usage/costs",
        "/api/v1/analytics/usage/active_users",
    ]

    # Pattern for agent-specific performance endpoint
    AGENT_PERFORMANCE_PATTERN = re.compile(
        r"^/api/v1/analytics/agents/([a-zA-Z0-9_-]+)/performance$"
    )

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Check agent performance pattern
        return bool(self.AGENT_PERFORMANCE_PATTERN.match(path))

    @rate_limit(rpm=60)
    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route GET requests to appropriate methods."""
        # Debate Analytics
        if path == "/api/v1/analytics/debates/overview":
            return self._get_debates_overview(query_params)
        elif path == "/api/v1/analytics/debates/trends":
            return self._get_debates_trends(query_params)
        elif path == "/api/v1/analytics/debates/topics":
            return self._get_debates_topics(query_params)
        elif path == "/api/v1/analytics/debates/outcomes":
            return self._get_debates_outcomes(query_params)

        # Agent Performance
        elif path == "/api/v1/analytics/agents/leaderboard":
            return self._get_agents_leaderboard(query_params)
        elif path == "/api/v1/analytics/agents/comparison":
            return self._get_agents_comparison(query_params)
        elif path == "/api/v1/analytics/agents/trends":
            return self._get_agents_trends(query_params)

        # Agent-specific performance
        match = self.AGENT_PERFORMANCE_PATTERN.match(path)
        if match:
            agent_id = match.group(1)
            return self._get_agent_performance(agent_id, query_params)

        # Usage Analytics
        if path == "/api/v1/analytics/usage/tokens":
            return self._get_usage_tokens(query_params)
        elif path == "/api/v1/analytics/usage/costs":
            return self._get_usage_costs(query_params)
        elif path == "/api/v1/analytics/usage/active_users":
            return self._get_active_users(query_params)

        return None

    # =========================================================================
    # Debate Analytics Endpoints
    # =========================================================================

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_debates_overview")
    @handle_errors("get debates overview")
    def _get_debates_overview(self, query_params: dict) -> HandlerResult:
        """
        Get debate overview statistics.

        GET /api/v1/analytics/debates/overview

        Query params:
        - time_range: Time range filter (7d, 30d, 90d, 365d, all) - default 30d
        - org_id: Filter by organization (optional)

        Response:
        {
            "time_range": "30d",
            "total_debates": 1250,
            "debates_this_period": 150,
            "debates_previous_period": 130,
            "growth_rate": 15.4,
            "consensus_reached": 1100,
            "consensus_rate": 88.0,
            "avg_rounds": 3.2,
            "avg_agents_per_debate": 3.5,
            "avg_confidence": 0.85,
            "generated_at": "2026-01-23T12:00:00Z"
        }
        """
        time_range = query_params.get("time_range", "30d")
        if time_range not in VALID_TIME_RANGES:
            time_range = "30d"

        org_id = query_params.get("org_id")

        storage = self.get_storage()
        if not storage:
            return json_response(
                {
                    "time_range": time_range,
                    "total_debates": 0,
                    "debates_this_period": 0,
                    "consensus_reached": 0,
                    "consensus_rate": 0.0,
                    "avg_rounds": 0.0,
                    "avg_agents_per_debate": 0.0,
                    "avg_confidence": 0.0,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Get debates from storage
        debates = storage.list_debates(limit=10000, org_id=org_id)

        # Parse time range
        start_time = _parse_time_range(time_range)
        now = datetime.now(timezone.utc)

        # Filter debates by time range
        period_debates = []
        previous_period_debates = []
        all_debates = []

        for debate in debates:
            debate_dict = debate if isinstance(debate, dict) else vars(debate)
            created_at_str = debate_dict.get("created_at", "")

            if created_at_str:
                try:
                    if isinstance(created_at_str, datetime):
                        created_at = created_at_str
                    else:
                        created_at = datetime.fromisoformat(
                            str(created_at_str).replace("Z", "+00:00")
                        )
                except (ValueError, TypeError):
                    created_at = None
            else:
                created_at = None

            all_debates.append(debate_dict)

            if start_time and created_at:
                if created_at >= start_time:
                    period_debates.append(debate_dict)

                # Calculate previous period for comparison
                period_days = (now - start_time).days
                previous_start = start_time - timedelta(days=period_days)
                if previous_start <= created_at < start_time:
                    previous_period_debates.append(debate_dict)
            elif not start_time:
                period_debates.append(debate_dict)

        # Calculate metrics
        total_debates = len(all_debates)
        debates_this_period = len(period_debates)
        debates_previous_period = len(previous_period_debates)

        # Growth rate
        growth_rate = 0.0
        if debates_previous_period > 0:
            growth_rate = (
                (debates_this_period - debates_previous_period) / debates_previous_period
            ) * 100

        # Consensus metrics
        consensus_count = 0
        total_rounds = 0
        total_agents = 0
        total_confidence = 0.0
        confidence_count = 0

        for debate in period_debates:
            if debate.get("consensus_reached"):
                consensus_count += 1

            # Get rounds from result
            result = debate.get("result", {})
            if isinstance(result, dict):
                rounds = result.get("rounds_used", result.get("rounds", 0))
                total_rounds += rounds
                confidence = result.get("confidence", 0.0)
                if confidence > 0:
                    total_confidence += confidence
                    confidence_count += 1

            # Count agents
            agents = debate.get("agents", [])
            if isinstance(agents, list):
                total_agents += len(agents)

        consensus_rate = (
            (consensus_count / debates_this_period * 100) if debates_this_period > 0 else 0.0
        )
        avg_rounds = total_rounds / debates_this_period if debates_this_period > 0 else 0.0
        avg_agents = total_agents / debates_this_period if debates_this_period > 0 else 0.0
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0

        return json_response(
            {
                "time_range": time_range,
                "total_debates": total_debates,
                "debates_this_period": debates_this_period,
                "debates_previous_period": debates_previous_period,
                "growth_rate": round(growth_rate, 1),
                "consensus_reached": consensus_count,
                "consensus_rate": round(consensus_rate, 1),
                "avg_rounds": round(avg_rounds, 1),
                "avg_agents_per_debate": round(avg_agents, 1),
                "avg_confidence": round(avg_confidence, 2),
                "generated_at": now.isoformat(),
            }
        )

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_debates_trends")
    @handle_errors("get debates trends")
    def _get_debates_trends(self, query_params: dict) -> HandlerResult:
        """
        Get debate trends over time.

        GET /api/v1/analytics/debates/trends

        Query params:
        - time_range: Time range filter (7d, 30d, 90d, 365d, all) - default 30d
        - granularity: Aggregation granularity (daily, weekly, monthly) - default daily
        - org_id: Filter by organization (optional)

        Response:
        {
            "time_range": "30d",
            "granularity": "daily",
            "data_points": [
                {
                    "period": "2026-01-01",
                    "total": 12,
                    "consensus_reached": 10,
                    "consensus_rate": 83.3,
                    "avg_rounds": 3.1
                },
                ...
            ],
            "generated_at": "2026-01-23T12:00:00Z"
        }
        """
        time_range = query_params.get("time_range", "30d")
        if time_range not in VALID_TIME_RANGES:
            time_range = "30d"

        granularity = query_params.get("granularity", "daily")
        if granularity not in VALID_GRANULARITIES:
            granularity = "daily"

        org_id = query_params.get("org_id")

        storage = self.get_storage()
        if not storage:
            return json_response(
                {
                    "time_range": time_range,
                    "granularity": granularity,
                    "data_points": [],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Get debates from storage
        debates = storage.list_debates(limit=10000, org_id=org_id)

        # Parse time range
        start_time = _parse_time_range(time_range)

        # Filter debates by time range and convert to dicts
        period_debates = []
        for debate in debates:
            debate_dict = debate if isinstance(debate, dict) else vars(debate)
            created_at_str = debate_dict.get("created_at", "")

            if created_at_str:
                try:
                    if isinstance(created_at_str, datetime):
                        created_at = created_at_str
                    else:
                        created_at = datetime.fromisoformat(
                            str(created_at_str).replace("Z", "+00:00")
                        )

                    if start_time is None or created_at >= start_time:
                        debate_dict["_parsed_time"] = created_at
                        period_debates.append(debate_dict)
                except (ValueError, TypeError):
                    continue

        # Group by time period
        groups = _group_by_time(period_debates, "_parsed_time", granularity)

        # Calculate metrics for each period
        data_points = []
        for period, debates_in_period in sorted(groups.items()):
            total = len(debates_in_period)
            consensus_count = sum(1 for d in debates_in_period if d.get("consensus_reached"))
            consensus_rate = (consensus_count / total * 100) if total > 0 else 0.0

            total_rounds = 0
            for d in debates_in_period:
                result = d.get("result", {})
                if isinstance(result, dict):
                    total_rounds += result.get("rounds_used", result.get("rounds", 0))
            avg_rounds = total_rounds / total if total > 0 else 0.0

            data_points.append(
                {
                    "period": period,
                    "total": total,
                    "consensus_reached": consensus_count,
                    "consensus_rate": round(consensus_rate, 1),
                    "avg_rounds": round(avg_rounds, 1),
                }
            )

        return json_response(
            {
                "time_range": time_range,
                "granularity": granularity,
                "data_points": data_points,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_debates_topics")
    @handle_errors("get debates topics")
    def _get_debates_topics(self, query_params: dict) -> HandlerResult:
        """
        Get topic distribution for debates.

        GET /api/v1/analytics/debates/topics

        Query params:
        - time_range: Time range filter (7d, 30d, 90d, 365d, all) - default 30d
        - limit: Maximum topics to return (default 20)
        - org_id: Filter by organization (optional)

        Response:
        {
            "time_range": "30d",
            "topics": [
                {
                    "topic": "security",
                    "count": 45,
                    "percentage": 18.5,
                    "consensus_rate": 92.0
                },
                ...
            ],
            "total_debates": 250,
            "generated_at": "2026-01-23T12:00:00Z"
        }
        """
        time_range = query_params.get("time_range", "30d")
        if time_range not in VALID_TIME_RANGES:
            time_range = "30d"

        try:
            limit = int(query_params.get("limit", "20"))
            limit = max(1, min(limit, 100))
        except (ValueError, TypeError):
            limit = 20

        org_id = query_params.get("org_id")

        storage = self.get_storage()
        if not storage:
            return json_response(
                {
                    "time_range": time_range,
                    "topics": [],
                    "total_debates": 0,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Get debates from storage
        debates = storage.list_debates(limit=10000, org_id=org_id)

        # Parse time range
        start_time = _parse_time_range(time_range)

        # Extract topics and count
        topic_counts: Counter = Counter()
        topic_consensus: Dict[str, List[bool]] = defaultdict(list)
        total_debates = 0

        for debate in debates:
            debate_dict = debate if isinstance(debate, dict) else vars(debate)
            created_at_str = debate_dict.get("created_at", "")

            # Check time range
            if start_time:
                try:
                    if isinstance(created_at_str, datetime):
                        created_at = created_at_str
                    else:
                        created_at = datetime.fromisoformat(
                            str(created_at_str).replace("Z", "+00:00")
                        )

                    if created_at < start_time:
                        continue
                except (ValueError, TypeError):
                    continue

            total_debates += 1

            # Extract topic from task or domain
            task = debate_dict.get("task", "")
            domain = debate_dict.get("domain", "")
            result = debate_dict.get("result", {})

            # Try to get domain from result metadata
            if isinstance(result, dict):
                domain = result.get("domain", domain)

            # Use domain if available, otherwise extract from task
            if domain:
                topic = domain.lower()
            elif task:
                # Simple topic extraction from task (first significant word)
                words = task.lower().split()
                topic = words[0] if words else "general"
            else:
                topic = "general"

            topic_counts[topic] += 1
            topic_consensus[topic].append(bool(debate_dict.get("consensus_reached")))

        # Build topic list with metrics
        topics = []
        for topic, count in topic_counts.most_common(limit):
            consensus_list = topic_consensus[topic]
            consensus_count = sum(1 for c in consensus_list if c)
            consensus_rate = (
                (consensus_count / len(consensus_list) * 100) if consensus_list else 0.0
            )
            percentage = (count / total_debates * 100) if total_debates > 0 else 0.0

            topics.append(
                {
                    "topic": topic,
                    "count": count,
                    "percentage": round(percentage, 1),
                    "consensus_rate": round(consensus_rate, 1),
                }
            )

        return json_response(
            {
                "time_range": time_range,
                "topics": topics,
                "total_debates": total_debates,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_debates_outcomes")
    @handle_errors("get debates outcomes")
    def _get_debates_outcomes(self, query_params: dict) -> HandlerResult:
        """
        Get debate outcome distribution (win/loss/draw).

        GET /api/v1/analytics/debates/outcomes

        Query params:
        - time_range: Time range filter (7d, 30d, 90d, 365d, all) - default 30d
        - org_id: Filter by organization (optional)

        Response:
        {
            "time_range": "30d",
            "outcomes": {
                "consensus": 120,
                "majority": 45,
                "dissent": 15,
                "no_resolution": 10
            },
            "total_debates": 190,
            "by_confidence": {
                "high": {"count": 100, "consensus_rate": 95.0},
                "medium": {"count": 60, "consensus_rate": 80.0},
                "low": {"count": 30, "consensus_rate": 50.0}
            },
            "generated_at": "2026-01-23T12:00:00Z"
        }
        """
        time_range = query_params.get("time_range", "30d")
        if time_range not in VALID_TIME_RANGES:
            time_range = "30d"

        org_id = query_params.get("org_id")

        storage = self.get_storage()
        if not storage:
            return json_response(
                {
                    "time_range": time_range,
                    "outcomes": {"consensus": 0, "majority": 0, "dissent": 0, "no_resolution": 0},
                    "total_debates": 0,
                    "by_confidence": {},
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Get debates from storage
        debates = storage.list_debates(limit=10000, org_id=org_id)

        # Parse time range
        start_time = _parse_time_range(time_range)

        # Count outcomes
        outcomes = {"consensus": 0, "majority": 0, "dissent": 0, "no_resolution": 0}
        confidence_buckets: Dict[str, List[bool]] = {
            "high": [],
            "medium": [],
            "low": [],
        }
        total_debates = 0

        for debate in debates:
            debate_dict = debate if isinstance(debate, dict) else vars(debate)
            created_at_str = debate_dict.get("created_at", "")

            # Check time range
            if start_time:
                try:
                    if isinstance(created_at_str, datetime):
                        created_at = created_at_str
                    else:
                        created_at = datetime.fromisoformat(
                            str(created_at_str).replace("Z", "+00:00")
                        )

                    if created_at < start_time:
                        continue
                except (ValueError, TypeError):
                    continue

            total_debates += 1

            # Determine outcome
            consensus_reached = debate_dict.get("consensus_reached", False)
            result = debate_dict.get("result", {})
            confidence = 0.0

            if isinstance(result, dict):
                confidence = result.get("confidence", 0.0)
                outcome_type = result.get("outcome_type", "")

                if outcome_type == "consensus" or (consensus_reached and confidence >= 0.8):
                    outcomes["consensus"] += 1
                elif outcome_type == "majority" or (consensus_reached and confidence >= 0.5):
                    outcomes["majority"] += 1
                elif outcome_type == "dissent" or not consensus_reached:
                    if confidence >= 0.3:
                        outcomes["dissent"] += 1
                    else:
                        outcomes["no_resolution"] += 1
                else:
                    outcomes["no_resolution"] += 1
            else:
                if consensus_reached:
                    outcomes["consensus"] += 1
                else:
                    outcomes["no_resolution"] += 1

            # Bucket by confidence
            if confidence >= 0.8:
                confidence_buckets["high"].append(consensus_reached)
            elif confidence >= 0.5:
                confidence_buckets["medium"].append(consensus_reached)
            else:
                confidence_buckets["low"].append(consensus_reached)

        # Calculate confidence bucket metrics
        by_confidence = {}
        for bucket, consensus_list in confidence_buckets.items():
            count = len(consensus_list)
            if count > 0:
                consensus_count = sum(1 for c in consensus_list if c)
                consensus_rate = (consensus_count / count) * 100
                by_confidence[bucket] = {
                    "count": count,
                    "consensus_rate": round(consensus_rate, 1),
                }

        return json_response(
            {
                "time_range": time_range,
                "outcomes": outcomes,
                "total_debates": total_debates,
                "by_confidence": by_confidence,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    # =========================================================================
    # Agent Performance Endpoints
    # =========================================================================

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_agents_leaderboard")
    @handle_errors("get agents leaderboard")
    def _get_agents_leaderboard(self, query_params: dict) -> HandlerResult:
        """
        Get agent leaderboard with ELO rankings and win rates.

        GET /api/v1/analytics/agents/leaderboard

        Query params:
        - limit: Maximum agents to return (default 20)
        - domain: Filter by domain (optional)

        Response:
        {
            "leaderboard": [
                {
                    "rank": 1,
                    "agent_name": "claude",
                    "elo": 1650,
                    "wins": 120,
                    "losses": 30,
                    "draws": 10,
                    "win_rate": 75.0,
                    "games_played": 160,
                    "calibration_score": 0.85
                },
                ...
            ],
            "total_agents": 15,
            "generated_at": "2026-01-23T12:00:00Z"
        }
        """
        try:
            limit = int(query_params.get("limit", "20"))
            limit = max(1, min(limit, 100))
        except (ValueError, TypeError):
            limit = 20

        domain = query_params.get("domain")

        elo_system = self.get_elo_system()
        if not elo_system:
            return json_response(
                {
                    "leaderboard": [],
                    "total_agents": 0,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Get leaderboard from ELO system
        agents = elo_system.get_leaderboard(limit=limit, domain=domain)

        leaderboard = []
        for rank, agent in enumerate(agents, 1):
            agent_data = {
                "rank": rank,
                "agent_name": agent.agent_name,
                "elo": round(agent.elo, 0),
                "wins": agent.wins,
                "losses": agent.losses,
                "draws": agent.draws,
                "win_rate": round(agent.win_rate * 100, 1),
                "games_played": agent.games_played,
            }

            # Add calibration score if available
            if hasattr(agent, "calibration_score"):
                agent_data["calibration_score"] = round(agent.calibration_score, 2)

            leaderboard.append(agent_data)

        # Get total agent count
        total_agents = len(elo_system.list_agents())

        return json_response(
            {
                "leaderboard": leaderboard,
                "total_agents": total_agents,
                "domain": domain,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    @handle_errors("get agent performance")
    def _get_agent_performance(self, agent_id: str, query_params: dict) -> HandlerResult:
        """
        Get individual agent performance statistics.

        GET /api/v1/analytics/agents/{agent_id}/performance

        Query params:
        - time_range: Time range filter (7d, 30d, 90d, 365d, all) - default 30d

        Response:
        {
            "agent_id": "claude",
            "agent_name": "Claude",
            "time_range": "30d",
            "elo": 1650,
            "elo_change": +25,
            "rank": 1,
            "wins": 120,
            "losses": 30,
            "draws": 10,
            "win_rate": 75.0,
            "games_played": 160,
            "consensus_contribution_rate": 85.0,
            "domain_performance": {
                "security": {"elo": 1700, "wins": 45, "losses": 8},
                "performance": {"elo": 1620, "wins": 30, "losses": 12}
            },
            "recent_matches": [...],
            "elo_history": [...],
            "generated_at": "2026-01-23T12:00:00Z"
        }
        """
        time_range = query_params.get("time_range", "30d")
        if time_range not in VALID_TIME_RANGES:
            time_range = "30d"

        elo_system = self.get_elo_system()
        if not elo_system:
            return error_response("ELO system not available", 503)

        # Get agent rating
        try:
            agent = elo_system.get_rating(agent_id)
        except (ValueError, KeyError):
            return error_response(f"Agent not found: {agent_id}", 404)

        # Get ELO history
        elo_history = elo_system.get_elo_history(agent_id, limit=50)

        # Calculate ELO change
        elo_change = 0
        if len(elo_history) >= 2:
            elo_change = agent.elo - elo_history[-1][1]

        # Get recent matches
        recent_matches = elo_system.get_recent_matches(limit=10)
        agent_matches = [m for m in recent_matches if agent_id in m.get("participants", [])]

        # Get rank
        leaderboard = elo_system.get_leaderboard(limit=100)
        rank = None
        for idx, a in enumerate(leaderboard, 1):
            if a.agent_name == agent_id:
                rank = idx
                break

        # Build response
        response = {
            "agent_id": agent_id,
            "agent_name": agent.agent_name,
            "time_range": time_range,
            "elo": round(agent.elo, 0),
            "elo_change": round(elo_change, 0),
            "rank": rank,
            "wins": agent.wins,
            "losses": agent.losses,
            "draws": agent.draws,
            "win_rate": round(agent.win_rate * 100, 1),
            "games_played": agent.games_played,
            "debates_count": agent.debates_count,
        }

        # Add domain performance if available
        if agent.domain_elos:
            response["domain_performance"] = {
                domain: {"elo": round(elo, 0)} for domain, elo in agent.domain_elos.items()
            }

        # Add calibration metrics if available
        if hasattr(agent, "calibration_score"):
            response["calibration_score"] = round(agent.calibration_score, 2)
        if hasattr(agent, "calibration_accuracy"):
            response["calibration_accuracy"] = round(agent.calibration_accuracy, 2)

        # Add recent matches
        response["recent_matches"] = agent_matches[:5]

        # Add ELO history for charting
        response["elo_history"] = [
            {"timestamp": ts, "elo": round(elo, 0)} for ts, elo in elo_history
        ]

        response["generated_at"] = datetime.now(timezone.utc).isoformat()

        return json_response(response)

    @handle_errors("get agents comparison")
    def _get_agents_comparison(self, query_params: dict) -> HandlerResult:
        """
        Compare multiple agents.

        GET /api/v1/analytics/agents/comparison

        Query params:
        - agents: Comma-separated list of agent names (required)

        Response:
        {
            "agents": ["claude", "gpt-4", "gemini"],
            "comparison": [
                {
                    "agent_name": "claude",
                    "elo": 1650,
                    "wins": 120,
                    "losses": 30,
                    "win_rate": 75.0,
                    "calibration_score": 0.85
                },
                ...
            ],
            "head_to_head": {
                "claude_vs_gpt-4": {"claude_wins": 15, "gpt-4_wins": 10, "draws": 5},
                ...
            },
            "generated_at": "2026-01-23T12:00:00Z"
        }
        """
        agents_param = query_params.get("agents", "")
        if not agents_param:
            return error_response("agents parameter is required (comma-separated list)", 400)

        agent_names = [a.strip() for a in agents_param.split(",") if a.strip()]
        if len(agent_names) < 2:
            return error_response("At least 2 agents required for comparison", 400)
        if len(agent_names) > 10:
            return error_response("Maximum 10 agents allowed for comparison", 400)

        elo_system = self.get_elo_system()
        if not elo_system:
            return error_response("ELO system not available", 503)

        # Get ratings for all agents
        comparison = []
        for agent_name in agent_names:
            try:
                agent = elo_system.get_rating(agent_name)
                agent_data = {
                    "agent_name": agent.agent_name,
                    "elo": round(agent.elo, 0),
                    "wins": agent.wins,
                    "losses": agent.losses,
                    "draws": agent.draws,
                    "win_rate": round(agent.win_rate * 100, 1),
                    "games_played": agent.games_played,
                }
                if hasattr(agent, "calibration_score"):
                    agent_data["calibration_score"] = round(agent.calibration_score, 2)
                comparison.append(agent_data)
            except (ValueError, KeyError):
                comparison.append(
                    {
                        "agent_name": agent_name,
                        "error": "Agent not found",
                    }
                )

        # Get head-to-head stats
        head_to_head = {}
        for i, agent_a in enumerate(agent_names):
            for agent_b in agent_names[i + 1 :]:
                try:
                    h2h = elo_system.get_head_to_head(agent_a, agent_b)
                    key = f"{agent_a}_vs_{agent_b}"
                    head_to_head[key] = {
                        f"{agent_a}_wins": h2h.get("a_wins", 0),
                        f"{agent_b}_wins": h2h.get("b_wins", 0),
                        "draws": h2h.get("draws", 0),
                        "total_matches": h2h.get("total", 0),
                    }
                except Exception:
                    pass

        return json_response(
            {
                "agents": agent_names,
                "comparison": comparison,
                "head_to_head": head_to_head,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_agents_trends")
    @handle_errors("get agents trends")
    def _get_agents_trends(self, query_params: dict) -> HandlerResult:
        """
        Get agent performance trends over time.

        GET /api/v1/analytics/agents/trends

        Query params:
        - agents: Comma-separated list of agent names (optional, defaults to top 5)
        - time_range: Time range filter (7d, 30d, 90d, 365d, all) - default 30d
        - granularity: Aggregation granularity (daily, weekly, monthly) - default daily

        Response:
        {
            "agents": ["claude", "gpt-4"],
            "time_range": "30d",
            "granularity": "daily",
            "trends": {
                "claude": [
                    {"period": "2026-01-01", "elo": 1640, "games": 5},
                    ...
                ],
                "gpt-4": [...]
            },
            "generated_at": "2026-01-23T12:00:00Z"
        }
        """
        time_range = query_params.get("time_range", "30d")
        if time_range not in VALID_TIME_RANGES:
            time_range = "30d"

        granularity = query_params.get("granularity", "daily")
        if granularity not in VALID_GRANULARITIES:
            granularity = "daily"

        elo_system = self.get_elo_system()
        if not elo_system:
            return error_response("ELO system not available", 503)

        # Get agents to track
        agents_param = query_params.get("agents", "")
        if agents_param:
            agent_names = [a.strip() for a in agents_param.split(",") if a.strip()]
        else:
            # Default to top 5 agents
            leaderboard = elo_system.get_leaderboard(limit=5)
            agent_names = [a.agent_name for a in leaderboard]

        # Get ELO history for each agent
        trends: Dict[str, List[Dict[str, Any]]] = {}

        for agent_name in agent_names[:10]:  # Limit to 10 agents
            try:
                history = elo_system.get_elo_history(agent_name, limit=100)

                # Convert to time series
                data_points = []
                for timestamp, elo in history:
                    try:
                        if isinstance(timestamp, str):
                            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        else:
                            dt = timestamp

                        # Generate period key
                        if granularity == "daily":
                            period = dt.strftime("%Y-%m-%d")
                        elif granularity == "weekly":
                            period = dt.strftime("%Y-W%W")
                        else:
                            period = dt.strftime("%Y-%m")

                        data_points.append(
                            {
                                "period": period,
                                "elo": round(elo, 0),
                                "timestamp": dt.isoformat(),
                            }
                        )
                    except (ValueError, TypeError):
                        continue

                # Group by period (take latest ELO for each period)
                period_data: Dict[str, Dict[str, Any]] = {}
                for dp in data_points:
                    period = dp["period"]
                    if (
                        period not in period_data
                        or dp["timestamp"] > period_data[period]["timestamp"]
                    ):
                        period_data[period] = dp

                trends[agent_name] = sorted(
                    [{"period": k, "elo": v["elo"]} for k, v in period_data.items()],
                    key=lambda x: x["period"],
                )
            except Exception as e:
                logger.warning(f"Failed to get trends for agent {agent_name}: {e}")
                trends[agent_name] = []

        return json_response(
            {
                "agents": agent_names,
                "time_range": time_range,
                "granularity": granularity,
                "trends": trends,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    # =========================================================================
    # Usage Analytics Endpoints
    # =========================================================================

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_usage_tokens")
    @handle_errors("get usage tokens")
    def _get_usage_tokens(self, query_params: dict) -> HandlerResult:
        """
        Get token consumption trends.

        GET /api/v1/analytics/usage/tokens

        Query params:
        - org_id: Organization ID (required)
        - time_range: Time range filter (7d, 30d, 90d, 365d, all) - default 30d
        - granularity: Aggregation granularity (daily, weekly, monthly) - default daily

        Response:
        {
            "org_id": "...",
            "time_range": "30d",
            "granularity": "daily",
            "summary": {
                "total_tokens_in": 5000000,
                "total_tokens_out": 1000000,
                "total_tokens": 6000000,
                "avg_tokens_per_day": 200000
            },
            "trends": [
                {"period": "2026-01-01", "tokens_in": 180000, "tokens_out": 35000},
                ...
            ],
            "generated_at": "2026-01-23T12:00:00Z"
        }
        """
        org_id = query_params.get("org_id")
        if not org_id:
            return error_response("org_id is required", 400)

        time_range = query_params.get("time_range", "30d")
        if time_range not in VALID_TIME_RANGES:
            time_range = "30d"

        granularity = query_params.get("granularity", "daily")
        if granularity not in VALID_GRANULARITIES:
            granularity = "daily"

        try:
            from aragora.billing.cost_tracker import get_cost_tracker

            tracker = get_cost_tracker()
            stats = tracker.get_workspace_stats(org_id)

            # Get summary data
            total_in = stats.get("total_tokens_in", 0)
            total_out = stats.get("total_tokens_out", 0)

            # Parse time range for day calculation
            start_time = _parse_time_range(time_range)
            days = 30
            if start_time:
                days = (datetime.now(timezone.utc) - start_time).days

            avg_per_day = (total_in + total_out) / days if days > 0 else 0

            return json_response(
                {
                    "org_id": org_id,
                    "time_range": time_range,
                    "granularity": granularity,
                    "summary": {
                        "total_tokens_in": total_in,
                        "total_tokens_out": total_out,
                        "total_tokens": total_in + total_out,
                        "avg_tokens_per_day": round(avg_per_day, 0),
                    },
                    "by_agent": stats.get("cost_by_agent", {}),
                    "by_model": stats.get("cost_by_model", {}),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        except ImportError:
            return json_response(
                {
                    "org_id": org_id,
                    "time_range": time_range,
                    "granularity": granularity,
                    "summary": {
                        "total_tokens_in": 0,
                        "total_tokens_out": 0,
                        "total_tokens": 0,
                        "avg_tokens_per_day": 0,
                    },
                    "message": "Cost tracker not available",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_usage_costs")
    @handle_errors("get usage costs")
    def _get_usage_costs(self, query_params: dict) -> HandlerResult:
        """
        Get cost breakdown by provider and model.

        GET /api/v1/analytics/usage/costs

        Query params:
        - org_id: Organization ID (required)
        - time_range: Time range filter (7d, 30d, 90d, 365d, all) - default 30d

        Response:
        {
            "org_id": "...",
            "time_range": "30d",
            "summary": {
                "total_cost_usd": "125.50",
                "avg_cost_per_day": "4.18",
                "avg_cost_per_debate": "0.84"
            },
            "by_provider": {
                "anthropic": {"cost": "80.00", "percentage": 63.7},
                "openai": {"cost": "45.50", "percentage": 36.3}
            },
            "by_model": {
                "claude-opus-4": {"cost": "60.00", "tokens": 400000},
                "gpt-4": {"cost": "45.50", "tokens": 200000}
            },
            "generated_at": "2026-01-23T12:00:00Z"
        }
        """
        org_id = query_params.get("org_id")
        if not org_id:
            return error_response("org_id is required", 400)

        time_range = query_params.get("time_range", "30d")
        if time_range not in VALID_TIME_RANGES:
            time_range = "30d"

        try:
            from aragora.billing.cost_tracker import get_cost_tracker

            tracker = get_cost_tracker()
            stats = tracker.get_workspace_stats(org_id)

            total_cost = float(stats.get("total_cost_usd", "0"))

            # Parse time range for day calculation
            start_time = _parse_time_range(time_range)
            days = 30
            if start_time:
                days = (datetime.now(timezone.utc) - start_time).days

            avg_per_day = total_cost / days if days > 0 else 0

            # Get API call count for per-debate cost
            api_calls = stats.get("total_api_calls", 0)
            avg_per_debate = total_cost / api_calls if api_calls > 0 else 0

            # Build provider breakdown
            by_agent = stats.get("cost_by_agent", {})
            by_model = stats.get("cost_by_model", {})

            # Calculate provider percentages
            by_provider = {}
            for agent, cost_str in by_agent.items():
                cost = float(cost_str)
                percentage = (cost / total_cost * 100) if total_cost > 0 else 0
                by_provider[agent] = {
                    "cost": f"{cost:.2f}",
                    "percentage": round(percentage, 1),
                }

            return json_response(
                {
                    "org_id": org_id,
                    "time_range": time_range,
                    "summary": {
                        "total_cost_usd": f"{total_cost:.2f}",
                        "avg_cost_per_day": f"{avg_per_day:.2f}",
                        "avg_cost_per_debate": f"{avg_per_debate:.2f}",
                        "total_api_calls": api_calls,
                    },
                    "by_provider": by_provider,
                    "by_model": by_model,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        except ImportError:
            return json_response(
                {
                    "org_id": org_id,
                    "time_range": time_range,
                    "summary": {
                        "total_cost_usd": "0.00",
                        "avg_cost_per_day": "0.00",
                        "avg_cost_per_debate": "0.00",
                    },
                    "by_provider": {},
                    "by_model": {},
                    "message": "Cost tracker not available",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_active_users")
    @handle_errors("get active users")
    def _get_active_users(self, query_params: dict) -> HandlerResult:
        """
        Get active user counts.

        GET /api/v1/analytics/usage/active_users

        Query params:
        - org_id: Organization ID (optional, returns global stats if not provided)
        - time_range: Time range filter (7d, 30d, 90d) - default 30d

        Response:
        {
            "org_id": "...",
            "time_range": "30d",
            "active_users": {
                "daily": 25,
                "weekly": 85,
                "monthly": 150
            },
            "user_growth": {
                "new_users": 15,
                "churned_users": 5,
                "net_growth": 10
            },
            "activity_distribution": {
                "power_users": 10,
                "regular_users": 50,
                "occasional_users": 90
            },
            "generated_at": "2026-01-23T12:00:00Z"
        }
        """
        org_id = query_params.get("org_id")
        time_range = query_params.get("time_range", "30d")
        if time_range not in {"7d", "30d", "90d"}:
            time_range = "30d"

        # Try to get user store from context
        user_store = self.ctx.get("user_store")

        if not user_store:
            # Return mock data when user store not available
            return json_response(
                {
                    "org_id": org_id,
                    "time_range": time_range,
                    "active_users": {
                        "daily": 0,
                        "weekly": 0,
                        "monthly": 0,
                    },
                    "user_growth": {
                        "new_users": 0,
                        "churned_users": 0,
                        "net_growth": 0,
                    },
                    "activity_distribution": {
                        "power_users": 0,
                        "regular_users": 0,
                        "occasional_users": 0,
                    },
                    "message": "User store not available",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        try:
            # Get active user counts if method exists
            if hasattr(user_store, "get_active_user_counts"):
                counts = user_store.get_active_user_counts(org_id=org_id)
            else:
                counts = {"daily": 0, "weekly": 0, "monthly": 0}

            # Get user growth if method exists
            if hasattr(user_store, "get_user_growth"):
                growth = user_store.get_user_growth(org_id=org_id, days=30)
            else:
                growth = {"new_users": 0, "churned_users": 0, "net_growth": 0}

            return json_response(
                {
                    "org_id": org_id,
                    "time_range": time_range,
                    "active_users": counts,
                    "user_growth": growth,
                    "activity_distribution": {
                        "power_users": 0,
                        "regular_users": 0,
                        "occasional_users": 0,
                    },
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to get active users: {e}")
            return json_response(
                {
                    "org_id": org_id,
                    "time_range": time_range,
                    "active_users": {"daily": 0, "weekly": 0, "monthly": 0},
                    "error": str(e),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )


__all__ = ["AnalyticsMetricsHandler"]
