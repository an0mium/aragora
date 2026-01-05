"""
Relationship endpoint handlers.

Endpoints:
- GET /api/relationships/summary - Global relationship overview
- GET /api/relationships/graph - Full relationship graph for visualizations
- GET /api/relationships/stats - Relationship system statistics
- GET /api/relationship/{agent_a}/{agent_b} - Detailed relationship between two agents
"""

import logging
import re
from pathlib import Path
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    get_float_param,
    validate_agent_name,
)

logger = logging.getLogger(__name__)

# Safe ID pattern for path segments
SAFE_ID_PATTERN = r'^[a-zA-Z0-9_-]+$'

# Lazy imports for optional dependencies
RELATIONSHIP_TRACKER_AVAILABLE = False
RelationshipTracker = None
AgentRelationship = None

try:
    from aragora.agents.grounded import RelationshipTracker as _RT, AgentRelationship as _AR
    RelationshipTracker = _RT
    AgentRelationship = _AR
    RELATIONSHIP_TRACKER_AVAILABLE = True
except ImportError:
    pass


def _safe_error_message(e: Exception, context: str = "") -> str:
    """Return a sanitized error message for client responses."""
    logger.error(f"Error in {context}: {type(e).__name__}: {e}", exc_info=True)
    error_type = type(e).__name__
    if error_type in ("FileNotFoundError", "OSError"):
        return "Resource not found"
    elif error_type in ("json.JSONDecodeError", "ValueError"):
        return "Invalid data format"
    elif error_type in ("TimeoutError", "asyncio.TimeoutError"):
        return "Operation timed out"
    return "An error occurred"


class RelationshipHandler(BaseHandler):
    """Handler for relationship endpoints."""

    ROUTES = [
        "/api/relationships/summary",
        "/api/relationships/graph",
        "/api/relationships/stats",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle dynamic route: /api/relationship/{agent_a}/{agent_b}
        if path.startswith("/api/relationship/") and path.count("/") >= 4:
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route relationship requests to appropriate methods."""
        nomic_dir = self.ctx.get("nomic_dir")

        if path == "/api/relationships/summary":
            return self._get_summary(nomic_dir)

        if path == "/api/relationships/graph":
            min_debates = get_int_param(query_params, 'min_debates', 3)
            min_score = get_float_param(query_params, 'min_score', 0.0)
            return self._get_graph(nomic_dir, min_debates, min_score)

        if path == "/api/relationships/stats":
            return self._get_stats(nomic_dir)

        # Handle /api/relationship/{agent_a}/{agent_b}
        if path.startswith("/api/relationship/"):
            parts = path.split("/")
            if len(parts) >= 5:
                agent_a = parts[3]
                agent_b = parts[4]
                # Validate agent names
                is_valid_a, err_a = validate_agent_name(agent_a)
                if not is_valid_a:
                    return error_response(f"Invalid agent_a: {err_a}", 400)
                is_valid_b, err_b = validate_agent_name(agent_b)
                if not is_valid_b:
                    return error_response(f"Invalid agent_b: {err_b}", 400)
                return self._get_pair_detail(nomic_dir, agent_a, agent_b)
            return error_response("Invalid path format", 400)

        return None

    def _get_tracker(self, nomic_dir: Optional[Path]) -> Optional["RelationshipTracker"]:
        """Get or create a RelationshipTracker instance."""
        if not RELATIONSHIP_TRACKER_AVAILABLE:
            return None
        try:
            # Use the grounded positions DB if nomic_dir is set
            if nomic_dir:
                db_path = nomic_dir / "grounded_positions.db"
                if db_path.exists():
                    return RelationshipTracker(elo_db_path=str(db_path))
            # Fall back to default
            return RelationshipTracker()
        except Exception as e:
            logger.warning(f"Failed to create RelationshipTracker: {e}")
            return None

    def _get_summary(self, nomic_dir: Optional[Path]) -> HandlerResult:
        """Get global relationship overview."""
        if not RELATIONSHIP_TRACKER_AVAILABLE:
            return error_response("Relationship tracker not available", 503)

        try:
            tracker = self._get_tracker(nomic_dir)
            if not tracker:
                return error_response("Failed to initialize relationship tracker", 503)

            # Collect all unique agents and their relationships
            all_agents = set()
            all_relationships = []
            agent_relationship_counts = {}

            # We need to query the DB directly to get all relationships
            # Use a helper to get all pairs from the database
            import sqlite3
            conn = sqlite3.connect(tracker.elo_db_path)
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_relationships'"
            )
            if not cursor.fetchone():
                conn.close()
                return json_response({
                    "total_relationships": 0,
                    "strongest_rivalry": None,
                    "strongest_alliance": None,
                    "most_connected_agent": None,
                    "avg_rivalry_score": 0.0,
                    "avg_alliance_score": 0.0,
                })

            cursor.execute("""
                SELECT agent_a, agent_b, debate_count, agreement_count,
                       a_wins_over_b, b_wins_over_a
                FROM agent_relationships
                WHERE debate_count >= 3
            """)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return json_response({
                    "total_relationships": 0,
                    "strongest_rivalry": None,
                    "strongest_alliance": None,
                    "most_connected_agent": None,
                    "avg_rivalry_score": 0.0,
                    "avg_alliance_score": 0.0,
                })

            # Process relationships
            strongest_rivalry = None
            strongest_rivalry_score = 0.0
            strongest_alliance = None
            strongest_alliance_score = 0.0
            rivalry_scores = []
            alliance_scores = []

            for row in rows:
                agent_a, agent_b, debate_count, agreement_count, a_wins, b_wins = row
                all_agents.add(agent_a)
                all_agents.add(agent_b)

                # Count connections per agent
                agent_relationship_counts[agent_a] = agent_relationship_counts.get(agent_a, 0) + 1
                agent_relationship_counts[agent_b] = agent_relationship_counts.get(agent_b, 0) + 1

                # Get relationship object to compute scores
                rel = tracker.get_relationship(agent_a, agent_b)
                rivalry_score = rel.rivalry_score
                alliance_score = rel.alliance_score

                if rivalry_score > 0:
                    rivalry_scores.append(rivalry_score)
                    if rivalry_score > strongest_rivalry_score:
                        strongest_rivalry_score = rivalry_score
                        strongest_rivalry = {"agents": [agent_a, agent_b], "score": rivalry_score}

                if alliance_score > 0:
                    alliance_scores.append(alliance_score)
                    if alliance_score > strongest_alliance_score:
                        strongest_alliance_score = alliance_score
                        strongest_alliance = {"agents": [agent_a, agent_b], "score": alliance_score}

            # Find most connected agent
            most_connected = None
            if agent_relationship_counts:
                most_connected_name = max(agent_relationship_counts, key=agent_relationship_counts.get)
                most_connected = {
                    "name": most_connected_name,
                    "relationship_count": agent_relationship_counts[most_connected_name]
                }

            return json_response({
                "total_relationships": len(rows),
                "strongest_rivalry": strongest_rivalry,
                "strongest_alliance": strongest_alliance,
                "most_connected_agent": most_connected,
                "avg_rivalry_score": sum(rivalry_scores) / len(rivalry_scores) if rivalry_scores else 0.0,
                "avg_alliance_score": sum(alliance_scores) / len(alliance_scores) if alliance_scores else 0.0,
            })

        except Exception as e:
            return error_response(_safe_error_message(e, "relationships_summary"), 500)

    def _get_graph(
        self, nomic_dir: Optional[Path], min_debates: int, min_score: float
    ) -> HandlerResult:
        """Get full relationship graph for visualizations."""
        if not RELATIONSHIP_TRACKER_AVAILABLE:
            return error_response("Relationship tracker not available", 503)

        try:
            tracker = self._get_tracker(nomic_dir)
            if not tracker:
                return error_response("Failed to initialize relationship tracker", 503)

            import sqlite3
            conn = sqlite3.connect(tracker.elo_db_path)
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_relationships'"
            )
            if not cursor.fetchone():
                conn.close()
                return json_response({
                    "nodes": [],
                    "edges": [],
                    "stats": {"node_count": 0, "edge_count": 0}
                })

            cursor.execute("""
                SELECT agent_a, agent_b, debate_count
                FROM agent_relationships
                WHERE debate_count >= ?
            """, (min_debates,))
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return json_response({
                    "nodes": [],
                    "edges": [],
                    "stats": {"node_count": 0, "edge_count": 0}
                })

            # Build nodes and edges
            nodes_data = {}  # agent -> {debate_count, rivals, allies}
            edges = []

            for row in rows:
                agent_a, agent_b, debate_count = row

                # Initialize node data
                for agent in [agent_a, agent_b]:
                    if agent not in nodes_data:
                        nodes_data[agent] = {"debate_count": 0, "rivals": 0, "allies": 0}

                # Get relationship scores
                rel = tracker.get_relationship(agent_a, agent_b)
                rivalry_score = rel.rivalry_score
                alliance_score = rel.alliance_score

                # Apply score filter
                max_score = max(rivalry_score, alliance_score)
                if max_score < min_score:
                    continue

                # Determine relationship type
                if rivalry_score > alliance_score and rivalry_score > 0.3:
                    rel_type = "rivalry"
                    nodes_data[agent_a]["rivals"] += 1
                    nodes_data[agent_b]["rivals"] += 1
                elif alliance_score > rivalry_score and alliance_score > 0.3:
                    rel_type = "alliance"
                    nodes_data[agent_a]["allies"] += 1
                    nodes_data[agent_b]["allies"] += 1
                else:
                    rel_type = "neutral"

                # Update debate counts
                nodes_data[agent_a]["debate_count"] += debate_count
                nodes_data[agent_b]["debate_count"] += debate_count

                edges.append({
                    "source": agent_a,
                    "target": agent_b,
                    "rivalry_score": round(rivalry_score, 3),
                    "alliance_score": round(alliance_score, 3),
                    "debate_count": debate_count,
                    "type": rel_type,
                })

            # Build nodes list
            nodes = [
                {
                    "id": agent,
                    "debate_count": data["debate_count"],
                    "rivals": data["rivals"],
                    "allies": data["allies"],
                }
                for agent, data in nodes_data.items()
            ]

            return json_response({
                "nodes": nodes,
                "edges": edges,
                "stats": {"node_count": len(nodes), "edge_count": len(edges)}
            })

        except Exception as e:
            return error_response(_safe_error_message(e, "relationships_graph"), 500)

    def _get_pair_detail(
        self, nomic_dir: Optional[Path], agent_a: str, agent_b: str
    ) -> HandlerResult:
        """Get detailed relationship between two specific agents."""
        if not RELATIONSHIP_TRACKER_AVAILABLE:
            return error_response("Relationship tracker not available", 503)

        try:
            tracker = self._get_tracker(nomic_dir)
            if not tracker:
                return error_response("Failed to initialize relationship tracker", 503)

            rel = tracker.get_relationship(agent_a, agent_b)

            if rel.debate_count == 0:
                return json_response({
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "relationship_exists": False,
                    "message": "No recorded interactions between these agents"
                })

            # Compute derived metrics
            agreement_rate = rel.agreement_count / rel.debate_count if rel.debate_count > 0 else 0
            rivalry_score = rel.rivalry_score
            alliance_score = rel.alliance_score

            # Determine relationship type
            if rivalry_score > alliance_score and rivalry_score > 0.3:
                rel_type = "rivalry"
            elif alliance_score > rivalry_score and alliance_score > 0.3:
                rel_type = "alliance"
            else:
                rel_type = "neutral"

            return json_response({
                "agent_a": rel.agent_a,
                "agent_b": rel.agent_b,
                "relationship_exists": True,
                "debate_count": rel.debate_count,
                "agreement_count": rel.agreement_count,
                "agreement_rate": round(agreement_rate, 3),
                "rivalry_score": round(rivalry_score, 3),
                "alliance_score": round(alliance_score, 3),
                "relationship_type": rel_type,
                "head_to_head": {
                    f"{rel.agent_a}_wins": rel.a_wins_over_b,
                    f"{rel.agent_b}_wins": rel.b_wins_over_a,
                },
                "critique_balance": {
                    f"{rel.agent_a}_to_{rel.agent_b}": rel.critique_count_a_to_b,
                    f"{rel.agent_b}_to_{rel.agent_a}": rel.critique_count_b_to_a,
                },
                "influence": {
                    f"{rel.agent_a}_on_{rel.agent_b}": round(rel.influence_a_on_b, 3),
                    f"{rel.agent_b}_on_{rel.agent_a}": round(rel.influence_b_on_a, 3),
                },
            })

        except Exception as e:
            return error_response(_safe_error_message(e, "relationship_pair_detail"), 500)

    def _get_stats(self, nomic_dir: Optional[Path]) -> HandlerResult:
        """Get relationship system statistics."""
        if not RELATIONSHIP_TRACKER_AVAILABLE:
            return error_response("Relationship tracker not available", 503)

        try:
            tracker = self._get_tracker(nomic_dir)
            if not tracker:
                return error_response("Failed to initialize relationship tracker", 503)

            import sqlite3
            conn = sqlite3.connect(tracker.elo_db_path)
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_relationships'"
            )
            if not cursor.fetchone():
                conn.close()
                return json_response({
                    "total_tracked_pairs": 0,
                    "total_debates_tracked": 0,
                    "rivalries": {"count": 0, "avg_score": 0.0},
                    "alliances": {"count": 0, "avg_score": 0.0},
                    "neutral": {"count": 0},
                    "most_debated_pair": None,
                    "highest_agreement_pair": None,
                })

            # Get all relationships
            cursor.execute("""
                SELECT agent_a, agent_b, debate_count, agreement_count
                FROM agent_relationships
            """)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return json_response({
                    "total_tracked_pairs": 0,
                    "total_debates_tracked": 0,
                    "rivalries": {"count": 0, "avg_score": 0.0},
                    "alliances": {"count": 0, "avg_score": 0.0},
                    "neutral": {"count": 0},
                    "most_debated_pair": None,
                    "highest_agreement_pair": None,
                })

            # Process statistics
            total_debates = 0
            rivalries = []
            alliances = []
            neutral_count = 0
            most_debated = None
            most_debated_count = 0
            highest_agreement = None
            highest_agreement_rate = 0.0

            for row in rows:
                agent_a, agent_b, debate_count, agreement_count = row
                total_debates += debate_count

                # Track most debated pair
                if debate_count > most_debated_count:
                    most_debated_count = debate_count
                    most_debated = {"agents": [agent_a, agent_b], "debates": debate_count}

                # Track highest agreement (min 3 debates)
                if debate_count >= 3:
                    agreement_rate = agreement_count / debate_count
                    if agreement_rate > highest_agreement_rate:
                        highest_agreement_rate = agreement_rate
                        highest_agreement = {
                            "agents": [agent_a, agent_b],
                            "rate": round(agreement_rate, 3)
                        }

                    # Categorize relationship
                    rel = tracker.get_relationship(agent_a, agent_b)
                    rivalry_score = rel.rivalry_score
                    alliance_score = rel.alliance_score

                    if rivalry_score > alliance_score and rivalry_score > 0.3:
                        rivalries.append(rivalry_score)
                    elif alliance_score > rivalry_score and alliance_score > 0.3:
                        alliances.append(alliance_score)
                    else:
                        neutral_count += 1

            return json_response({
                "total_tracked_pairs": len(rows),
                "total_debates_tracked": total_debates,
                "rivalries": {
                    "count": len(rivalries),
                    "avg_score": round(sum(rivalries) / len(rivalries), 3) if rivalries else 0.0
                },
                "alliances": {
                    "count": len(alliances),
                    "avg_score": round(sum(alliances) / len(alliances), 3) if alliances else 0.0
                },
                "neutral": {"count": neutral_count},
                "most_debated_pair": most_debated,
                "highest_agreement_pair": highest_agreement,
            })

        except Exception as e:
            return error_response(_safe_error_message(e, "relationships_stats"), 500)
