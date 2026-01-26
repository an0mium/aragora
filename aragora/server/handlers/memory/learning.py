"""
Cross-cycle learning analytics endpoint handlers.

Endpoints:
- GET /api/learning/cycles - Get all cycle summaries
- GET /api/learning/patterns - Get learned patterns across cycles
- GET /api/learning/agent-evolution - Get agent performance evolution
- GET /api/learning/insights - Get aggregated insights from cycles
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from aragora.rbac.decorators import require_permission  # noqa: F401

from ..base import (
    HandlerResult,
    error_response,
    get_clamped_int_param,
    get_db_connection,
    handle_errors,
    json_response,
)
from ..secure import ForbiddenError, SecureHandler, UnauthorizedError
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# RBAC permission for learning endpoints
LEARNING_PERMISSION = "memory:write"

# Rate limiter for learning endpoints (30 requests per minute - ML operations)
_learning_limiter = RateLimiter(requests_per_minute=30)


class LearningHandler(SecureHandler):
    """Handler for cross-cycle learning analytics endpoints.

    Requires authentication and memory:read permission (RBAC).
    """

    ROUTES = [
        "/api/v1/learning/cycles",
        "/api/v1/learning/patterns",
        "/api/v1/learning/agent-evolution",
        "/api/v1/learning/insights",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    async def handle(  # type: ignore[override]
        self, path: str, query_params: dict, handler=None
    ) -> Optional[HandlerResult]:
        """Route GET requests with RBAC."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _learning_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for learning endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # RBAC: Require authentication and memory:read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, LEARNING_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required to access learning data", 401)
        except ForbiddenError as e:
            logger.warning(f"Learning endpoint access denied: {e}")
            return error_response(str(e), 403)

        if path == "/api/v1/learning/cycles":
            limit = get_clamped_int_param(query_params, "limit", 20, min_val=1, max_val=100)
            return self._get_cycle_summaries(limit)
        if path == "/api/v1/learning/patterns":
            return self._get_learned_patterns()
        if path == "/api/v1/learning/agent-evolution":
            return self._get_agent_evolution()
        if path == "/api/v1/learning/insights":
            limit = get_clamped_int_param(query_params, "limit", 50, min_val=1, max_val=200)
            return self._get_aggregated_insights(limit)
        return None

    def _get_nomic_dir(self) -> Optional[Path]:
        """Get the nomic directory path."""
        nomic_dir = self.ctx.get("nomic_dir")
        if nomic_dir:
            return Path(nomic_dir)
        return None

    @handle_errors("cycle summaries")
    def _get_cycle_summaries(self, limit: int) -> HandlerResult:
        """Get summaries of all nomic loop cycles."""
        nomic_dir = self._get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        replays_dir = nomic_dir / "replays"
        if not replays_dir.exists():
            return json_response({"cycles": [], "count": 0})

        cycles = []
        for cycle_dir in sorted(replays_dir.iterdir(), reverse=True):
            if not cycle_dir.is_dir() or not cycle_dir.name.startswith("nomic-cycle-"):
                continue

            meta_file = cycle_dir / "meta.json"
            if not meta_file.exists():
                continue

            try:
                with open(meta_file) as f:
                    meta = json.load(f)

                # Extract cycle number
                cycle_num = int(cycle_dir.name.replace("nomic-cycle-", ""))

                cycles.append(
                    {
                        "cycle": cycle_num,
                        "debate_id": meta.get("debate_id", ""),
                        "topic": meta.get("topic", ""),
                        "agents": [a.get("name", "") for a in meta.get("agents", [])],
                        "started_at": meta.get("started_at"),
                        "ended_at": meta.get("ended_at"),
                        "duration_ms": meta.get("duration_ms"),
                        "status": meta.get("status", "unknown"),
                        "final_verdict": meta.get("final_verdict"),
                        "event_count": meta.get("event_count", 0),
                        "success": meta.get("status") == "completed"
                        and meta.get("final_verdict") is not None,
                    }
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse {meta_file}: {e}")
                continue

            if len(cycles) >= limit:
                break

        return json_response(
            {
                "cycles": cycles,
                "count": len(cycles),
                "total_cycles": len(cycles),  # Use bounded count instead of loading all entries
                "has_more": len(cycles) >= limit,
            }
        )

    @handle_errors("learned patterns")
    def _get_learned_patterns(self) -> HandlerResult:
        """Get patterns learned across cycles from consensus memory."""
        nomic_dir = self._get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        # Try to load from consensus memory
        patterns: dict[str, list | dict] = {
            "successful_patterns": [],
            "failed_patterns": [],
            "recurring_themes": [],
            "agent_specializations": {},
        }

        # Check risk register for patterns
        risk_file = nomic_dir / "risk_register.jsonl"
        if risk_file.exists():
            failed_cycles = []
            successful_cycles = []
            try:
                with open(risk_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get("confidence", 1.0) < 0.3:
                                failed_cycles.append(
                                    {
                                        "cycle": entry.get("cycle", 0),
                                        "phase": entry.get("phase", ""),
                                        "task": entry.get("task", "")[:100],
                                        "error": entry.get("error", "")[:200],
                                    }
                                )
                            else:
                                successful_cycles.append(
                                    {
                                        "cycle": entry.get("cycle", 0),
                                        "phase": entry.get("phase", ""),
                                        "confidence": entry.get("confidence", 0),
                                    }
                                )
                        except json.JSONDecodeError:
                            continue
                patterns["failed_patterns"] = failed_cycles[-10:]
                patterns["successful_patterns"] = successful_cycles[-10:]
            except Exception as e:
                logger.warning(f"Failed to read risk register: {e}")

        # Analyze replays for recurring themes (bounded iteration)
        replays_dir = nomic_dir / "replays"
        if replays_dir.exists():
            theme_counts: dict[str, int] = defaultdict(int)
            agent_wins: dict[str, int] = defaultdict(int)

            max_to_scan = 500  # Reasonable upper bound to prevent memory exhaustion
            scanned = 0
            for cycle_dir in replays_dir.iterdir():
                if scanned >= max_to_scan:
                    break
                if not cycle_dir.is_dir():
                    continue
                scanned += 1
                meta_file = cycle_dir / "meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            meta = json.load(f)
                        topic = meta.get("topic", "").lower()
                        for keyword in [
                            "security",
                            "performance",
                            "testing",
                            "refactor",
                            "api",
                            "fix",
                            "feature",
                        ]:
                            if keyword in topic:
                                theme_counts[keyword] += 1
                        # Track winning agents
                        winner = meta.get("winner")
                        if winner:
                            agent_wins[winner] += 1
                    except (json.JSONDecodeError, ValueError):
                        continue

            patterns["recurring_themes"] = [
                {"theme": k, "count": v}
                for k, v in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            ]
            patterns["agent_specializations"] = dict(agent_wins)

        return json_response(patterns)

    @handle_errors("agent evolution")
    def _get_agent_evolution(self) -> HandlerResult:
        """Get agent performance evolution over cycles."""
        nomic_dir = self._get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        # Track agent performance over time
        evolution: dict[str, list[dict]] = defaultdict(list)

        replays_dir = nomic_dir / "replays"
        cycles_analyzed = 0
        if replays_dir.exists():
            # Collect cycle directories with bounds (prevent memory exhaustion)
            max_to_scan = 500
            cycle_dirs: list[Path] = []
            for cycle_dir in replays_dir.iterdir():
                if len(cycle_dirs) >= max_to_scan:
                    break
                if cycle_dir.is_dir() and cycle_dir.name.startswith("nomic-cycle-"):
                    cycle_dirs.append(cycle_dir)

            # Sort only the bounded subset
            for cycle_dir in sorted(cycle_dirs, key=lambda d: d.name):
                try:
                    cycle_num = int(cycle_dir.name.replace("nomic-cycle-", ""))
                except ValueError:
                    continue

                meta_file = cycle_dir / "meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            meta = json.load(f)

                        cycles_analyzed += 1
                        agents = meta.get("agents", [])
                        vote_tally = meta.get("vote_tally", {})
                        winner = meta.get("winner")
                        status = meta.get("status", "unknown")

                        for agent_info in agents:
                            agent_name = agent_info.get("name", "unknown")
                            votes = vote_tally.get(agent_name, 0)
                            is_winner = agent_name == winner

                            evolution[agent_name].append(
                                {
                                    "cycle": cycle_num,
                                    "votes": votes,
                                    "is_winner": is_winner,
                                    "participated": True,
                                    "status": status,
                                }
                            )
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.debug(f"Failed to parse {meta_file}: {e}")
                        continue

        # Calculate trends
        agent_trends = {}
        for agent, data_points in evolution.items():
            if len(data_points) < 2:
                trend = "stable"
            else:
                recent = data_points[-3:] if len(data_points) >= 3 else data_points
                win_rate = sum(1 for d in recent if d.get("is_winner")) / len(recent)
                if win_rate > 0.5:
                    trend = "improving"
                elif win_rate < 0.2:
                    trend = "declining"
                else:
                    trend = "stable"

            agent_trends[agent] = {
                "data_points": data_points[-20:],  # Last 20 cycles
                "total_cycles": len(data_points),
                "total_wins": sum(1 for d in data_points if d.get("is_winner")),
                "trend": trend,
            }

        return json_response(
            {
                "agents": agent_trends,
                "total_cycles_analyzed": cycles_analyzed,  # Use bounded count
            }
        )

    @handle_errors("aggregated insights")
    def _get_aggregated_insights(self, limit: int) -> HandlerResult:
        """Get insights aggregated from all cycles."""
        nomic_dir = self._get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        insights = []

        # Check for insight store database
        insight_db = nomic_dir / "insights.db"
        if insight_db.exists():
            try:
                with get_db_connection(str(insight_db)) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT insight_id, debate_id, category, content, confidence, created_at
                        FROM insights
                        ORDER BY created_at DESC
                        LIMIT ?
                    """,
                        (limit,),
                    )
                    for row in cursor.fetchall():
                        insights.append(
                            {
                                "insight_id": row[0],
                                "debate_id": row[1],
                                "category": row[2],
                                "content": row[3],
                                "confidence": row[4],
                                "created_at": row[5],
                            }
                        )
            except Exception as e:
                logger.warning(f"Failed to read insights DB: {e}")

        # Aggregate by category
        category_counts: dict[str, int] = defaultdict(int)
        for insight in insights:
            category_counts[insight.get("category", "general")] += 1

        return json_response(
            {
                "insights": insights,
                "count": len(insights),
                "by_category": dict(category_counts),
            }
        )
