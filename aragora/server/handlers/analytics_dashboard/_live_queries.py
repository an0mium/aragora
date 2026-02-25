"""Live database queries for analytics dashboard.

Replaces hardcoded stub responses with real queries against:
- DebateAnalytics (debate_analytics.db) for debate metrics, agent performance
- EloSystem for agent ELO rankings and win rates
- DebateStorage for recent debate history

Each function returns data in the exact JSON shape the frontend expects,
or None if the query fails or returns empty data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _query_debate_analytics() -> dict[str, Any] | None:
    """Query DebateAnalytics for debate stats and agent performance.

    Returns a dict with keys usable by multiple stub endpoints, or None on failure.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics

        analytics = get_debate_analytics()

        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, can't use run_until_complete
                return None
        except RuntimeError:
            pass

        stats = asyncio.run(analytics.get_debate_stats(days_back=30))
        if stats.total_debates == 0:
            return None

        leaderboard = asyncio.run(analytics.get_agent_leaderboard(limit=10, days_back=30))
        cost_breakdown = asyncio.run(analytics.get_cost_breakdown(days_back=30))
        try:
            from aragora.analytics.debate_analytics import DebateMetricType
            debate_metric: Any = DebateMetricType.DEBATE_COUNT
        except ImportError:
            debate_metric = "debate_count"
        debate_trends = asyncio.run(
            analytics.get_usage_trends(
                metric=debate_metric,
                days_back=7,
            )
        )

        return {
            "stats": stats,
            "leaderboard": leaderboard,
            "cost_breakdown": cost_breakdown,
            "debate_trends": debate_trends,
        }
    except Exception:  # noqa: BLE001 -- graceful fallback
        logger.debug("DebateAnalytics query failed, will use fallback", exc_info=True)
        return None


def query_summary() -> dict[str, Any] | None:
    """Query real debate summary metrics.

    Returns the exact shape expected by /api/analytics/summary, or None.
    """
    try:
        from aragora.analytics.debate_analytics import (
            get_debate_analytics,
        )

        analytics = get_debate_analytics()

        import asyncio

        stats = asyncio.run(analytics.get_debate_stats(days_back=30))
        if stats.total_debates == 0:
            return None

        # Get top topics from debate records
        top_topics = _query_top_topics()

        consensus_rate = (
            round(stats.consensus_rate * 100, 1)
            if stats.consensus_rate <= 1.0
            else round(stats.consensus_rate, 1)
        )

        return {
            "summary": {
                "total_debates": stats.total_debates,
                "total_messages": stats.total_messages,
                "consensus_rate": consensus_rate,
                "avg_debate_duration_ms": round(stats.avg_duration_seconds * 1000, 0),
                "active_users_24h": _query_active_users_24h(),
                "top_topics": top_topics,
            }
        }
    except Exception:  # noqa: BLE001
        logger.debug("Summary query failed", exc_info=True)
        return None


def _query_top_topics() -> list[dict[str, Any]]:
    """Get top debate topics from DebateStorage."""
    try:
        from aragora.server.storage import DebateStorage

        st = DebateStorage()
        with st.connection() as conn:
            rows = conn.execute(
                """
                SELECT task, COUNT(*) as cnt
                FROM debates
                WHERE created_at >= ?
                GROUP BY task
                ORDER BY cnt DESC
                LIMIT 4
                """,
                ((datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),),
            ).fetchall()

            if not rows:
                return []

            return [
                {
                    "topic": row[0][:60] if isinstance(row[0], str) else str(row[0])[:60],
                    "count": row[1],
                }
                for row in rows
            ]
    except Exception:  # noqa: BLE001
        return []


def _query_active_users_24h() -> int:
    """Count active users in last 24 hours from debate records."""
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import sqlite3

        analytics = get_debate_analytics()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        with sqlite3.connect(analytics.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM debate_records WHERE created_at >= ? AND user_id IS NOT NULL",
                (cutoff,),
            ).fetchone()
            return row[0] if row else 0
    except Exception:  # noqa: BLE001
        return 0


def query_finding_trends() -> dict[str, Any] | None:
    """Query real finding trends over the last 5 days.

    Returns the exact shape expected by /api/analytics/trends/findings, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import sqlite3

        analytics = get_debate_analytics()
        with sqlite3.connect(analytics.db_path) as conn:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            rows = conn.execute(
                """
                SELECT DATE(created_at) as day, COUNT(*) as total,
                       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as resolved
                FROM debate_records
                WHERE created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY day DESC
                LIMIT 5
                """,
                (cutoff,),
            ).fetchall()

            if not rows:
                return None

            trends = [
                {"date": row[0], "findings": row[1], "resolved": row[2] or 0}
                for row in reversed(rows)
            ]
            return {"trends": trends}
    except Exception:  # noqa: BLE001
        logger.debug("Finding trends query failed", exc_info=True)
        return None


def query_remediation() -> dict[str, Any] | None:
    """Query real remediation metrics.

    Returns the exact shape expected by /api/analytics/remediation, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import sqlite3

        analytics = get_debate_analytics()
        with sqlite3.connect(analytics.db_path) as conn:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as remediated,
                    SUM(CASE WHEN status != 'completed' THEN 1 ELSE 0 END) as pending,
                    AVG(CASE WHEN status = 'completed' THEN duration_seconds / 3600.0 END) as avg_hours
                FROM debate_records
                WHERE created_at >= ?
                """,
                (cutoff,),
            ).fetchone()

            if not row or (row[0] or 0) == 0:
                return None

            total = row[0] or 0
            remediated = row[1] or 0
            pending = row[2] or 0
            avg_hours = round(row[3] or 0, 1)
            rate = round(remediated / total * 100, 1) if total > 0 else 0

            return {
                "metrics": {
                    "total_findings": total,
                    "remediated": remediated,
                    "pending": pending,
                    "avg_remediation_time_hours": avg_hours,
                    "remediation_rate": rate,
                }
            }
    except Exception:  # noqa: BLE001
        logger.debug("Remediation query failed", exc_info=True)
        return None


def query_agents() -> dict[str, Any] | None:
    """Query real agent performance data from DebateAnalytics + ELO.

    Returns the exact shape expected by /api/analytics/agents, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import asyncio

        analytics = get_debate_analytics()
        leaderboard = asyncio.run(analytics.get_agent_leaderboard(limit=5, days_back=30))

        if not leaderboard:
            return None

        # Try to enrich with real ELO data
        elo_ratings = _query_elo_ratings()

        agents = []
        for agent in leaderboard:
            elo = elo_ratings.get(agent.agent_id, agent.current_elo)
            win_rate = agent.vote_ratio if agent.vote_ratio > 0 else 0.5
            agents.append(
                {
                    "agent_id": agent.agent_id,
                    "name": agent.agent_name or agent.agent_id,
                    "debates": agent.debates_participated,
                    "win_rate": round(win_rate, 2),
                    "elo": round(elo, 0),
                }
            )

        return {"agents": agents}
    except Exception:  # noqa: BLE001
        logger.debug("Agent metrics query failed", exc_info=True)
        return None


def _query_elo_ratings() -> dict[str, float]:
    """Get ELO ratings from the ranking system."""
    try:
        from aragora.ranking.elo import EloSystem

        elo = EloSystem()
        lb = elo.get_leaderboard(limit=20)
        return {r.agent_name: r.elo for r in lb}
    except Exception:  # noqa: BLE001
        return {}


def query_cost() -> dict[str, Any] | None:
    """Query real cost analytics.

    Returns the exact shape expected by /api/analytics/cost, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import asyncio
        import sqlite3

        analytics = get_debate_analytics()
        breakdown = asyncio.run(analytics.get_cost_breakdown(days_back=30))

        if float(breakdown.total_cost) == 0:
            return None

        # Cost by model
        cost_by_model = {k: round(float(v), 2) for k, v in breakdown.by_model.items()}

        # Cost by provider mapped to debate type (using protocol field)
        cost_by_debate_type: dict[str, float] = {}
        with sqlite3.connect(analytics.db_path) as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(protocol, 'general') as dtype,
                       SUM(CAST(total_cost AS REAL)) as cost
                FROM debate_records
                WHERE created_at >= ?
                GROUP BY protocol
                ORDER BY cost DESC
                """,
                ((datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),),
            ).fetchall()
            for row in rows:
                cost_by_debate_type[row[0]] = round(row[1] or 0, 2)

        # Daily cost trend (last 5 days)
        cost_trend = []
        with sqlite3.connect(analytics.db_path) as conn:
            rows = conn.execute(
                """
                SELECT DATE(created_at) as day, SUM(CAST(total_cost AS REAL)) as cost
                FROM debate_records
                WHERE created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY day DESC
                LIMIT 5
                """,
                ((datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),),
            ).fetchall()
            for row in reversed(rows):
                cost_trend.append({"date": row[0], "cost_usd": round(row[1] or 0, 2)})

        total = round(float(breakdown.total_cost), 2)
        # Project monthly from daily average
        days_elapsed = max(1, (datetime.now(timezone.utc) - breakdown.period_start).days)
        projected = round(total / days_elapsed * 30, 2)

        return {
            "analysis": {
                "total_cost_usd": total,
                "cost_by_model": cost_by_model,
                "cost_by_debate_type": cost_by_debate_type,
                "projected_monthly_cost": projected,
                "cost_trend": cost_trend,
            }
        }
    except Exception:  # noqa: BLE001
        logger.debug("Cost query failed", exc_info=True)
        return None


def query_cost_breakdown() -> dict[str, Any] | None:
    """Query real per-agent cost breakdown.

    Returns the exact shape expected by /api/analytics/cost/breakdown, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import sqlite3

        analytics = get_debate_analytics()

        with sqlite3.connect(analytics.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT agent_id, SUM(CAST(cost AS REAL)) as spend,
                       COUNT(DISTINCT debate_id) as debates
                FROM agent_records
                WHERE created_at >= ?
                GROUP BY agent_id
                ORDER BY spend DESC
                LIMIT 10
                """,
                ((datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),),
            ).fetchall()

            if not rows:
                return None

            total_spend = sum(row["spend"] or 0 for row in rows)
            if total_spend == 0:
                return None

            agents = [
                {
                    "agent": row["agent_id"],
                    "spend_usd": round(row["spend"] or 0, 2),
                    "debates": row["debates"] or 0,
                }
                for row in rows
            ]

            # Try to get budget utilization
            budget_pct = 0.0
            try:
                from aragora.billing.cost_tracker import get_cost_tracker

                tracker = get_cost_tracker()
                budget = tracker.get_budget()
                if budget and budget.monthly_limit_usd:
                    budget_pct = round(total_spend / float(budget.monthly_limit_usd) * 100, 1)
            except Exception:  # noqa: BLE001
                pass

            return {
                "breakdown": {
                    "total_spend_usd": round(total_spend, 2),
                    "agents": agents,
                    "budget_utilization_pct": budget_pct,
                }
            }
    except Exception:  # noqa: BLE001
        logger.debug("Cost breakdown query failed", exc_info=True)
        return None


def query_compliance() -> dict[str, Any] | None:
    """Query real compliance scorecard.

    Returns the exact shape expected by /api/analytics/compliance, or None.
    """
    try:
        from aragora.compliance.monitor import get_compliance_monitor

        monitor = get_compliance_monitor()
        if monitor is None:
            return None
        scores = getattr(monitor, "get_scorecard", lambda: None)()

        if not scores:
            return None

        categories = []
        total_score = 0
        for name, data in scores.items():
            score = data.get("score", 0)
            total_score += score
            status = "pass" if score >= 80 else ("warning" if score >= 60 else "fail")
            categories.append({"name": name, "score": score, "status": status})

        if not categories:
            return None

        overall = round(total_score / len(categories))

        return {
            "compliance": {
                "overall_score": overall,
                "categories": categories,
                "last_audit": datetime.now(timezone.utc).isoformat(),
            }
        }
    except Exception:  # noqa: BLE001
        logger.debug("Compliance query failed", exc_info=True)
        return None


def query_heatmap() -> dict[str, Any] | None:
    """Query real activity heatmap data.

    Returns the exact shape expected by /api/analytics/heatmap, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import sqlite3

        analytics = get_debate_analytics()

        with sqlite3.connect(analytics.db_path) as conn:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            rows = conn.execute(
                """
                SELECT
                    CAST(strftime('%w', created_at) AS INTEGER) as dow,
                    CAST(strftime('%H', created_at) AS INTEGER) / 3 as time_slot,
                    COUNT(*) as cnt
                FROM debate_records
                WHERE created_at >= ?
                GROUP BY dow, time_slot
                """,
                (cutoff,),
            ).fetchall()

            if not rows:
                return None

            x_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            y_labels = ["9AM", "12PM", "3PM", "6PM"]
            # Initialize 4x5 grid
            values = [[0] * 5 for _ in range(4)]
            max_val = 0

            for row in rows:
                dow = row[0]  # 0=Sun, 1=Mon, ...
                time_slot = row[1]  # 0-7 (3-hour blocks)
                cnt = row[2]
                # Map to our grid: Mon-Fri (dow 1-5), time slots 3-6 (9AM-6PM)
                col = dow - 1  # Mon=0, Tue=1, ...
                slot_row = time_slot - 3  # 9AM=0, 12PM=1, 3PM=2, 6PM=3
                if 0 <= col < 5 and 0 <= slot_row < 4:
                    values[slot_row][col] += cnt
                    max_val = max(max_val, values[slot_row][col])

            if max_val == 0:
                return None

            return {
                "heatmap": {
                    "x_labels": x_labels,
                    "y_labels": y_labels,
                    "values": values,
                    "max_value": max_val,
                }
            }
    except Exception:  # noqa: BLE001
        logger.debug("Heatmap query failed", exc_info=True)
        return None


def query_tokens() -> dict[str, Any] | None:
    """Query real token usage summary.

    Returns the exact shape expected by /api/analytics/tokens, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import sqlite3

        analytics = get_debate_analytics()

        with sqlite3.connect(analytics.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

            # Total tokens
            row = conn.execute(
                """
                SELECT SUM(tokens_in) as tin, SUM(tokens_out) as tout
                FROM agent_records WHERE created_at >= ?
                """,
                (cutoff,),
            ).fetchone()

            total_in = row["tin"] or 0
            total_out = row["tout"] or 0
            if total_in + total_out == 0:
                return None

            # By agent
            by_agent = {}
            rows = conn.execute(
                """
                SELECT agent_id, SUM(tokens_in) + SUM(tokens_out) as total
                FROM agent_records WHERE created_at >= ?
                GROUP BY agent_id ORDER BY total DESC
                """,
                (cutoff,),
            ).fetchall()
            for r in rows:
                by_agent[r["agent_id"]] = r["total"] or 0

            # By model
            by_model = {}
            rows = conn.execute(
                """
                SELECT model, SUM(tokens_in) + SUM(tokens_out) as total
                FROM agent_records WHERE created_at >= ? AND model IS NOT NULL
                GROUP BY model ORDER BY total DESC
                """,
                (cutoff,),
            ).fetchall()
            for r in rows:
                by_model[r["model"]] = r["total"] or 0

            days_elapsed = max(1, 30)
            avg_per_day = round((total_in + total_out) / days_elapsed)

            return {
                "summary": {
                    "total_tokens_in": total_in,
                    "total_tokens_out": total_out,
                    "total_tokens": total_in + total_out,
                    "avg_tokens_per_day": avg_per_day,
                },
                "by_agent": by_agent,
                "by_model": by_model,
            }
    except Exception:  # noqa: BLE001
        logger.debug("Tokens query failed", exc_info=True)
        return None


def query_token_trends() -> dict[str, Any] | None:
    """Query real token usage trends.

    Returns the exact shape expected by /api/analytics/tokens/trends, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import sqlite3

        analytics = get_debate_analytics()

        with sqlite3.connect(analytics.db_path) as conn:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            rows = conn.execute(
                """
                SELECT DATE(created_at) as day,
                       SUM(tokens_in) as tin, SUM(tokens_out) as tout
                FROM agent_records
                WHERE created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY day DESC
                LIMIT 5
                """,
                (cutoff,),
            ).fetchall()

            if not rows:
                return None

            trends = [
                {
                    "date": row[0],
                    "tokens_in": row[1] or 0,
                    "tokens_out": row[2] or 0,
                }
                for row in reversed(rows)
            ]
            return {"trends": trends}
    except Exception:  # noqa: BLE001
        logger.debug("Token trends query failed", exc_info=True)
        return None


def query_token_providers() -> dict[str, Any] | None:
    """Query real token usage by provider.

    Returns the exact shape expected by /api/analytics/tokens/providers, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import sqlite3

        analytics = get_debate_analytics()

        with sqlite3.connect(analytics.db_path) as conn:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            rows = conn.execute(
                """
                SELECT provider, SUM(tokens_in) + SUM(tokens_out) as total
                FROM agent_records
                WHERE created_at >= ? AND provider IS NOT NULL AND provider != ''
                GROUP BY provider
                ORDER BY total DESC
                """,
                (cutoff,),
            ).fetchall()

            if not rows:
                return None

            grand_total = sum(row[1] or 0 for row in rows)
            if grand_total == 0:
                return None

            # Map provider names to display names
            provider_names = {
                "anthropic": "Anthropic",
                "openai": "OpenAI",
                "google": "Google",
                "mistral": "Mistral",
                "xai": "xAI",
                "deepseek": "DeepSeek",
                "openrouter": "OpenRouter",
            }

            providers = []
            for row in rows:
                tokens = row[1] or 0
                display = provider_names.get(row[0], row[0].title())
                pct = round(tokens / grand_total * 100, 1)
                providers.append({"provider": display, "tokens": tokens, "pct": pct})

            return {"providers": providers}
    except Exception:  # noqa: BLE001
        logger.debug("Token providers query failed", exc_info=True)
        return None


def query_flips_summary() -> dict[str, Any] | None:
    """Query real flip detection summary.

    Returns the exact shape expected by /api/analytics/flips/summary, or None.
    """
    try:
        from aragora.insights.flip_detector import FlipDetector

        detector = FlipDetector()
        summary = detector.get_flip_summary()

        total = summary.get("total", 0)
        if total == 0:
            return None

        consistent = total - summary.get("by_type", {}).get("contradiction", 0)
        inconsistent = summary.get("by_type", {}).get("contradiction", 0)

        return {
            "summary": {
                "total": total,
                "consistent": max(0, consistent),
                "inconsistent": inconsistent,
            }
        }
    except Exception:  # noqa: BLE001
        logger.debug("Flips summary query failed", exc_info=True)
        return None


def query_flips_recent() -> dict[str, Any] | None:
    """Query real recent flip events.

    Returns the exact shape expected by /api/analytics/flips/recent, or None.
    """
    try:
        from aragora.insights.flip_detector import FlipDetector

        detector = FlipDetector()
        flips = detector.get_recent_flips(limit=5)

        if not flips:
            return None

        formatted = []
        for f in flips[:3]:
            formatted.append(
                {
                    "agent": getattr(f, "agent_name", "unknown"),
                    "topic": getattr(f, "topic", "")[:30],
                    "from": getattr(f, "old_position", "unknown"),
                    "to": getattr(f, "new_position", "unknown"),
                    "date": getattr(f, "detected_at", datetime.now(timezone.utc)).isoformat()[:10]
                    if hasattr(f, "detected_at")
                    else "",
                }
            )

        return {"flips": formatted}
    except Exception:  # noqa: BLE001
        logger.debug("Recent flips query failed", exc_info=True)
        return None


def query_flips_consistency() -> dict[str, Any] | None:
    """Query real agent consistency scores.

    Returns the exact shape expected by /api/analytics/flips/consistency, or None.
    """
    try:
        from aragora.insights.flip_detector import FlipDetector

        detector = FlipDetector()
        summary = detector.get_flip_summary()
        agent_names = list(summary.get("by_agent", {}).keys())

        if not agent_names:
            return None

        scores = detector.get_agents_consistency_batch(agent_names)
        if not scores:
            return None

        consistency = []
        for agent_name, score_data in scores.items():
            # score_data may be a float or object with .consistency attribute
            if isinstance(score_data, (int, float)):
                score = float(score_data)
            else:
                score = getattr(score_data, "consistency", 0.0)
                if isinstance(score, str) and score.endswith("%"):
                    score = float(score.rstrip("%")) / 100.0
            consistency.append(
                {
                    "agent": agent_name,
                    "consistency_score": round(score, 2),
                }
            )

        consistency.sort(key=lambda x: x["consistency_score"], reverse=True)
        return {"consistency": consistency}
    except Exception:  # noqa: BLE001
        logger.debug("Flips consistency query failed", exc_info=True)
        return None


def query_flips_trends() -> dict[str, Any] | None:
    """Query real flip trends over time.

    Returns the exact shape expected by /api/analytics/flips/trends, or None.
    """
    try:
        from aragora.insights.flip_detector import FlipDetector

        detector = FlipDetector()
        with detector.db.connection() as conn:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            rows = conn.execute(
                """
                SELECT DATE(detected_at) as day, COUNT(*) as cnt
                FROM detected_flips
                WHERE detected_at >= ?
                GROUP BY DATE(detected_at)
                ORDER BY day DESC
                LIMIT 5
                """,
                (cutoff,),
            ).fetchall()

            if not rows:
                return None

            trends = [{"date": row[0], "flips": row[1]} for row in reversed(rows)]
            return {"trends": trends}
    except Exception:  # noqa: BLE001
        logger.debug("Flips trends query failed", exc_info=True)
        return None


def query_deliberations() -> dict[str, Any] | None:
    """Query real deliberation summary.

    Returns the exact shape expected by /api/analytics/deliberations, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import asyncio

        analytics = get_debate_analytics()
        stats = asyncio.run(analytics.get_debate_stats(days_back=30))

        if stats.total_debates == 0:
            return None

        consensus_rate = (
            round(stats.consensus_rate * 100, 1)
            if stats.consensus_rate <= 1.0
            else round(stats.consensus_rate, 1)
        )

        return {
            "summary": {
                "total": stats.total_debates,
                "consensus_rate": consensus_rate,
            }
        }
    except Exception:  # noqa: BLE001
        logger.debug("Deliberations query failed", exc_info=True)
        return None


def query_deliberations_channels() -> dict[str, Any] | None:
    """Query deliberation stats by channel.

    Returns the exact shape expected by /api/analytics/deliberations/channels, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import sqlite3

        analytics = get_debate_analytics()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        with sqlite3.connect(analytics.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT COALESCE(protocol, 'api') as channel,
                       COUNT(*) as cnt,
                       CASE WHEN COUNT(*) > 0
                            THEN ROUND(SUM(consensus_reached) * 100.0 / COUNT(*), 1)
                            ELSE 0 END as rate
                FROM debate_records
                WHERE created_at >= ?
                GROUP BY channel
                ORDER BY cnt DESC
                LIMIT 5
                """,
                (cutoff,),
            ).fetchall()

            if not rows:
                return None

            channels = [
                {
                    "channel": row["channel"],
                    "count": row["cnt"],
                    "consensus_rate": row["rate"],
                }
                for row in rows
            ]
            return {"channels": channels}
    except Exception:  # noqa: BLE001
        logger.debug("Deliberations channels query failed", exc_info=True)
        return None


def query_deliberations_consensus() -> dict[str, Any] | None:
    """Query consensus rates by method.

    Returns the exact shape expected by /api/analytics/deliberations/consensus, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import sqlite3

        analytics = get_debate_analytics()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        with sqlite3.connect(analytics.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    CASE
                        WHEN rounds <= 2 THEN 'majority'
                        WHEN rounds <= 4 THEN 'supermajority'
                        ELSE 'unanimous'
                    END as method,
                    COUNT(*) as cnt,
                    ROUND(AVG(rounds), 1) as avg_rounds
                FROM debate_records
                WHERE created_at >= ? AND consensus_reached = 1
                GROUP BY method
                ORDER BY cnt DESC
                """,
                (cutoff,),
            ).fetchall()

            if not rows:
                return None

            consensus = [
                {
                    "method": row[0],
                    "count": row[1],
                    "avg_rounds": row[2] or 0,
                }
                for row in rows
            ]
            return {"consensus": consensus}
    except Exception:  # noqa: BLE001
        logger.debug("Deliberations consensus query failed", exc_info=True)
        return None


def query_deliberations_performance() -> dict[str, Any] | None:
    """Query deliberation performance metrics.

    Returns the exact shape expected by /api/analytics/deliberations/performance, or None.
    """
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics
        import asyncio

        analytics = get_debate_analytics()
        stats = asyncio.run(analytics.get_debate_stats(days_back=30))

        if stats.total_debates == 0:
            return None

        convergence_rate = (
            stats.consensus_rate if stats.consensus_rate <= 1.0 else stats.consensus_rate / 100.0
        )

        return {
            "performance": [
                {"metric": "avg_duration_s", "value": round(stats.avg_duration_seconds, 1)},
                {"metric": "avg_rounds", "value": round(stats.avg_rounds, 1)},
                {"metric": "avg_agents", "value": round(stats.avg_agents_per_debate, 1)},
                {"metric": "convergence_rate", "value": round(convergence_rate, 2)},
            ]
        }
    except Exception:  # noqa: BLE001
        logger.debug("Deliberations performance query failed", exc_info=True)
        return None


# Registry mapping endpoint paths to their query functions
LIVE_QUERY_REGISTRY: dict[str, Any] = {
    "/api/analytics/summary": query_summary,
    "/api/analytics/trends/findings": query_finding_trends,
    "/api/analytics/remediation": query_remediation,
    "/api/analytics/agents": query_agents,
    "/api/analytics/cost": query_cost,
    "/api/analytics/cost/breakdown": query_cost_breakdown,
    "/api/analytics/compliance": query_compliance,
    "/api/analytics/heatmap": query_heatmap,
    "/api/analytics/tokens": query_tokens,
    "/api/analytics/tokens/trends": query_token_trends,
    "/api/analytics/tokens/providers": query_token_providers,
    "/api/analytics/flips/summary": query_flips_summary,
    "/api/analytics/flips/recent": query_flips_recent,
    "/api/analytics/flips/consistency": query_flips_consistency,
    "/api/analytics/flips/trends": query_flips_trends,
    "/api/analytics/deliberations": query_deliberations,
    "/api/analytics/deliberations/channels": query_deliberations_channels,
    "/api/analytics/deliberations/consensus": query_deliberations_consensus,
    "/api/analytics/deliberations/performance": query_deliberations_performance,
}
