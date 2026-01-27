"""
Gas Town Dashboard API Handlers.

Provides REST APIs for the Gas Town multi-agent orchestration dashboard:
- Overview stats and metrics
- Convoy progress tracking
- Agent workload distribution
- Bead queue depth
- GUPP recovery events

Endpoints:
- GET /api/v1/dashboard/gastown/overview - Get Gas Town overview
- GET /api/v1/dashboard/gastown/convoys - Get convoy list with progress
- GET /api/v1/dashboard/gastown/agents - Get agent workload distribution
- GET /api/v1/dashboard/gastown/beads - Get bead queue stats
- GET /api/v1/dashboard/gastown/metrics - Get throughput metrics
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.secure import SecureHandler, ForbiddenError, UnauthorizedError
from aragora.server.versioning.compat import strip_version_prefix

logger = logging.getLogger(__name__)

# In-memory cache for dashboard data
_gt_dashboard_cache: Dict[str, Dict[str, Any]] = {}
_gt_dashboard_cache_lock = threading.Lock()

# Cache TTL (15 seconds for near-real-time)
CACHE_TTL = 15


def _get_cached_data(key: str) -> Optional[Dict[str, Any]]:
    """Get cached dashboard data if not expired."""
    with _gt_dashboard_cache_lock:
        cached = _gt_dashboard_cache.get(key)
        if cached:
            if datetime.now(timezone.utc).timestamp() - cached.get("cached_at", 0) < CACHE_TTL:
                return cached.get("data")
    return None


def _set_cached_data(key: str, data: Dict[str, Any]) -> None:
    """Cache dashboard data."""
    with _gt_dashboard_cache_lock:
        _gt_dashboard_cache[key] = {
            "data": data,
            "cached_at": datetime.now(timezone.utc).timestamp(),
        }


class GasTownDashboardHandler(SecureHandler):
    """Handler for Gas Town dashboard endpoints."""

    RESOURCE_TYPE = "gastown"

    ROUTES = [
        "/api/v1/dashboard/gastown/overview",
        "/api/v1/dashboard/gastown/convoys",
        "/api/v1/dashboard/gastown/agents",
        "/api/v1/dashboard/gastown/beads",
        "/api/v1/dashboard/gastown/metrics",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        path = strip_version_prefix(path)
        return path.startswith("/api/v1/dashboard/gastown/") or path in self.ROUTES

    async def handle(
        self,
        path: str,
        query_params: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Route Gas Town dashboard requests."""
        # Require authentication
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

        # Check read permission
        try:
            self.check_permission(auth_context, "gastown:read")
        except ForbiddenError:
            return error_response("Permission denied: gastown:read", 403)

        path = strip_version_prefix(path)

        # Route to appropriate handler
        if path == "/api/v1/dashboard/gastown/overview":
            return await self._get_overview(query_params)
        elif path == "/api/v1/dashboard/gastown/convoys":
            return await self._get_convoys(query_params)
        elif path == "/api/v1/dashboard/gastown/agents":
            return await self._get_agents(query_params)
        elif path == "/api/v1/dashboard/gastown/beads":
            return await self._get_beads(query_params)
        elif path == "/api/v1/dashboard/gastown/metrics":
            return await self._get_metrics(query_params)

        return None

    async def _get_overview(self, query_params: Dict[str, Any]) -> HandlerResult:
        """Get Gas Town overview dashboard data.

        Returns:
            - Total convoys (active, completed, failed)
            - Total beads (by status)
            - Agent counts (by role)
            - Witness patrol status
            - Mayor status
            - Recent activity summary
        """
        force_refresh = query_params.get("refresh", "").lower() == "true"

        if not force_refresh:
            cached = _get_cached_data("overview")
            if cached:
                return json_response(cached)

        now = datetime.now(timezone.utc)
        overview: Dict[str, Any] = {
            "generated_at": now.isoformat(),
            "convoys": {"active": 0, "completed": 0, "failed": 0, "total": 0},
            "beads": {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0, "total": 0},
            "agents": {"mayor": 0, "witness": 0, "polecat": 0, "crew": 0, "total": 0},
            "witness_patrol": {"active": False, "last_check": None},
            "mayor": {"active": False, "node_id": None},
        }

        # Get convoy stats
        try:
            from aragora.nomic.convoys import ConvoyManager, ConvoyStatus

            manager = ConvoyManager()
            convoys = await manager.list_convoys()
            overview["convoys"]["total"] = len(convoys)
            for c in convoys:
                if c.status == ConvoyStatus.IN_PROGRESS:
                    overview["convoys"]["active"] += 1
                elif c.status == ConvoyStatus.COMPLETED:
                    overview["convoys"]["completed"] += 1
                elif c.status == ConvoyStatus.FAILED:
                    overview["convoys"]["failed"] += 1
        except ImportError:
            logger.debug("Convoy module not available")
        except Exception as e:
            logger.debug(f"Could not get convoy stats: {e}")

        # Get bead stats
        try:
            from aragora.nomic.beads import BeadManager, BeadStatus

            manager = BeadManager()
            for status in BeadStatus:
                beads = await manager.list_beads(status=status, limit=1000)
                count = len(beads)
                overview["beads"][status.value] = count
                overview["beads"]["total"] += count
        except ImportError:
            logger.debug("Bead module not available")
        except Exception as e:
            logger.debug(f"Could not get bead stats: {e}")

        # Get agent stats
        try:
            from aragora.nomic.agent_roles import AgentHierarchy, AgentRole

            hierarchy = AgentHierarchy()
            for role in AgentRole:
                agents = await hierarchy.list_agents(role=role)
                count = len(agents)
                overview["agents"][role.value] = count
                overview["agents"]["total"] += count
        except ImportError:
            logger.debug("Agent roles module not available")
        except Exception as e:
            logger.debug(f"Could not get agent stats: {e}")

        # Get witness patrol status
        try:
            from aragora.server.startup import get_witness_behavior

            witness = get_witness_behavior()
            if witness:
                overview["witness_patrol"]["active"] = witness._running
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get witness status: {e}")

        # Get mayor status
        try:
            from aragora.server.startup import get_mayor_coordinator

            coordinator = get_mayor_coordinator()
            if coordinator:
                overview["mayor"]["active"] = coordinator.is_mayor
                overview["mayor"]["node_id"] = coordinator.get_current_mayor_node()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get mayor status: {e}")

        _set_cached_data("overview", overview)
        return json_response(overview)

    async def _get_convoys(self, query_params: Dict[str, Any]) -> HandlerResult:
        """Get convoy list with progress.

        Returns list of convoys with:
        - ID, title, status
        - Progress (completed/total beads)
        - Created/updated timestamps
        - Assigned agents
        """
        limit = int(query_params.get("limit", 20))
        status_filter = query_params.get("status")

        try:
            from aragora.nomic.convoys import ConvoyManager, ConvoyStatus

            manager = ConvoyManager()

            filter_status = None
            if status_filter:
                try:
                    filter_status = ConvoyStatus(status_filter)
                except ValueError:
                    return error_response(f"Invalid status: {status_filter}", 400)

            convoys = await manager.list_convoys(status=filter_status)

            result = []
            for c in convoys[:limit]:
                progress_pct = 0.0
                if c.total_beads > 0:
                    progress_pct = (c.completed_beads / c.total_beads) * 100

                result.append(
                    {
                        "id": c.id,
                        "title": c.title,
                        "description": getattr(c, "description", ""),
                        "status": c.status.value,
                        "total_beads": c.total_beads,
                        "completed_beads": c.completed_beads,
                        "failed_beads": getattr(c, "failed_beads", 0),
                        "progress_percentage": round(progress_pct, 1),
                        "created_at": c.created_at.isoformat() if c.created_at else None,
                        "updated_at": getattr(c, "updated_at", c.created_at).isoformat()
                        if getattr(c, "updated_at", c.created_at)
                        else None,
                    }
                )

            return json_response(
                {
                    "convoys": result,
                    "total": len(convoys),
                    "showing": len(result),
                }
            )

        except ImportError as e:
            logger.debug(f"Convoy module not available: {e}")
            return json_response({"convoys": [], "total": 0, "showing": 0})
        except Exception as e:
            logger.error(f"Error getting convoys: {e}")
            return error_response(f"Error getting convoys: {e}", 500)

    async def _get_agents(self, query_params: Dict[str, Any]) -> HandlerResult:
        """Get agent workload distribution.

        Returns:
        - Agents grouped by role
        - Active bead count per agent
        - Completion rate per agent
        """
        try:
            from aragora.nomic.agent_roles import AgentHierarchy, AgentRole

            hierarchy = AgentHierarchy()

            agents_by_role: Dict[str, List[Dict[str, Any]]] = {role.value: [] for role in AgentRole}

            for role in AgentRole:
                agents = await hierarchy.list_agents(role=role)
                for a in agents:
                    agents_by_role[role.value].append(
                        {
                            "agent_id": a.agent_id,
                            "role": a.role.value,
                            "supervised_by": a.supervised_by,
                            "is_ephemeral": a.is_ephemeral,
                            "assigned_at": a.assigned_at.isoformat() if a.assigned_at else None,
                            "capabilities": [c.value for c in a.capabilities]
                            if a.capabilities
                            else [],
                        }
                    )

            # Calculate totals
            totals = {role: len(agents) for role, agents in agents_by_role.items()}
            totals["total"] = sum(totals.values())

            return json_response(
                {
                    "agents_by_role": agents_by_role,
                    "totals": totals,
                }
            )

        except ImportError as e:
            logger.debug(f"Agent roles module not available: {e}")
            return json_response(
                {
                    "agents_by_role": {},
                    "totals": {"total": 0},
                }
            )
        except Exception as e:
            logger.error(f"Error getting agents: {e}")
            return error_response(f"Error getting agents: {e}", 500)

    async def _get_beads(self, query_params: Dict[str, Any]) -> HandlerResult:
        """Get bead queue stats.

        Returns:
        - Bead counts by status
        - Queue depth over time
        - Average completion time
        """
        try:
            from aragora.nomic.beads import BeadManager, BeadStatus

            manager = BeadManager()

            stats: Dict[str, Any] = {
                "by_status": {},
                "by_priority": {},
                "queue_depth": 0,
                "processing": 0,
            }

            total = 0
            for status in BeadStatus:
                beads = await manager.list_beads(status=status, limit=1000)
                count = len(beads)
                stats["by_status"][status.value] = count
                total += count

                if status.value == "pending":
                    stats["queue_depth"] = count
                elif status.value == "in_progress":
                    stats["processing"] = count

            stats["total"] = total

            # Get priority distribution
            try:
                from aragora.nomic.beads import BeadPriority

                for priority in BeadPriority:
                    beads = await manager.list_beads(priority=priority, limit=1000)
                    stats["by_priority"][priority.value] = len(beads)
            except (ImportError, AttributeError):
                pass

            return json_response(stats)

        except ImportError as e:
            logger.debug(f"Bead module not available: {e}")
            return json_response(
                {
                    "by_status": {},
                    "queue_depth": 0,
                    "processing": 0,
                    "total": 0,
                }
            )
        except Exception as e:
            logger.error(f"Error getting beads: {e}")
            return error_response(f"Error getting beads: {e}", 500)

    async def _get_metrics(self, query_params: Dict[str, Any]) -> HandlerResult:
        """Get throughput metrics.

        Returns:
        - Beads completed per hour
        - Average bead duration
        - Convoy completion rate
        - GUPP recovery events (if available)
        """
        hours = int(query_params.get("hours", 24))

        metrics: Dict[str, Any] = {
            "period_hours": hours,
            "beads_per_hour": 0.0,
            "avg_bead_duration_minutes": None,
            "convoy_completion_rate": 0.0,
            "gupp_recovery_events": 0,
        }

        # Try to get metrics from Prometheus or internal counters
        try:
            from aragora.nomic.metrics import (
                get_beads_completed_count,
                get_convoy_completion_rate,
                get_gupp_recovery_count,
            )

            beads_completed = get_beads_completed_count(hours=hours)
            metrics["beads_per_hour"] = round(beads_completed / max(hours, 1), 2)
            metrics["convoy_completion_rate"] = get_convoy_completion_rate()
            metrics["gupp_recovery_events"] = get_gupp_recovery_count(hours=hours)

        except ImportError:
            # Fallback: estimate from current data
            try:
                from aragora.nomic.convoys import ConvoyManager, ConvoyStatus

                manager = ConvoyManager()
                convoys = await manager.list_convoys()
                completed = sum(1 for c in convoys if c.status == ConvoyStatus.COMPLETED)
                total = len(convoys)
                if total > 0:
                    metrics["convoy_completion_rate"] = round((completed / total) * 100, 1)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Could not get detailed metrics: {e}")

        return json_response(metrics)
