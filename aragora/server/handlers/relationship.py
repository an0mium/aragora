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
from functools import wraps
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.agents.grounded import RelationshipTracker as _RelationshipTrackerType

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    get_float_param,
    get_db_connection,
    table_exists,
    SAFE_AGENT_PATTERN,
)
from aragora.utils.optional_imports import try_import
from aragora.persistence.db_config import DatabaseType, get_db_path

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies using centralized utility
_relationship_imports, RELATIONSHIP_TRACKER_AVAILABLE = try_import(
    "aragora.agents.grounded", "RelationshipTracker", "AgentRelationship"
)
RelationshipTracker = _relationship_imports["RelationshipTracker"]
AgentRelationship = _relationship_imports["AgentRelationship"]

from aragora.server.error_utils import safe_error_message as _safe_error_message


# =============================================================================
# Score Computation Utilities
# =============================================================================


def compute_rivalry_score(
    debate_count: int, agreement_count: int, a_wins: int, b_wins: int
) -> float:
    """Compute rivalry score between two agents.

    Rivalry is high when agents frequently disagree and have competitive win rates.

    Args:
        debate_count: Number of debates between the agents
        agreement_count: Number of debates where they agreed
        a_wins: Wins by agent A over agent B
        b_wins: Wins by agent B over agent A

    Returns:
        Rivalry score from 0.0 to 1.0
    """
    if debate_count < 3:
        return 0.0
    disagreement_rate = 1 - (agreement_count / debate_count)
    total_wins = a_wins + b_wins
    competitiveness = 1 - abs(a_wins - b_wins) / max(total_wins, 1)
    frequency_factor = min(1.0, debate_count / 20)
    return disagreement_rate * competitiveness * frequency_factor


def compute_alliance_score(debate_count: int, agreement_count: int) -> float:
    """Compute alliance score between two agents.

    Alliance is high when agents frequently agree.

    Note: Full alliance_score also uses critique acceptance rates, but those
    aren't always available. This simplified version uses just agreement rate.

    Args:
        debate_count: Number of debates between the agents
        agreement_count: Number of debates where they agreed

    Returns:
        Alliance score from 0.0 to 1.0
    """
    if debate_count < 3:
        return 0.0
    agreement_rate = agreement_count / debate_count
    # Simplified: alliance_score = agreement_rate * 0.6 + acceptance_rate * 0.4
    # Since we don't have critique data, use agreement_rate * 0.6 as baseline
    return agreement_rate * 0.6


def determine_relationship_type(
    rivalry_score: float, alliance_score: float, threshold: float = 0.3
) -> str:
    """Determine relationship type based on rivalry and alliance scores.

    Args:
        rivalry_score: Rivalry score (0.0 to 1.0)
        alliance_score: Alliance score (0.0 to 1.0)
        threshold: Minimum score to classify as rivalry/alliance (default: 0.3)

    Returns:
        Relationship type: "rivalry", "alliance", or "neutral"
    """
    if rivalry_score > alliance_score and rivalry_score > threshold:
        return "rivalry"
    elif alliance_score > rivalry_score and alliance_score > threshold:
        return "alliance"
    return "neutral"


# =============================================================================
# Handler Decorators
# =============================================================================


def require_tracker(func: Callable) -> Callable:
    """Decorator that handles tracker availability and initialization.

    Converts nomic_dir parameter to an initialized tracker. Methods decorated
    with this receive a guaranteed non-None RelationshipTracker.

    Usage:
        # Call site: self._get_summary(nomic_dir)
        @require_tracker
        def _get_summary(self, tracker: "RelationshipTracker") -> HandlerResult:
            # tracker is guaranteed non-None
            ...
    """
    @wraps(func)
    def wrapper(self, nomic_dir: Optional[Path], *args, **kwargs) -> HandlerResult:
        if not RELATIONSHIP_TRACKER_AVAILABLE:
            return error_response("Relationship tracker not available", 503)
        tracker = self._get_tracker(nomic_dir)
        if not tracker:
            return error_response("Failed to initialize relationship tracker", 503)
        return func(self, tracker, *args, **kwargs)
    return wrapper


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
            params, err = self.extract_path_params(path, [
                (2, "agent_a", SAFE_AGENT_PATTERN),
                (3, "agent_b", SAFE_AGENT_PATTERN),
            ])
            if err:
                return err
            return self._get_pair_detail(nomic_dir, params["agent_a"], params["agent_b"])

        return None

    def _get_tracker(self, nomic_dir: Optional[Path]) -> Optional["RelationshipTracker"]:
        """Get or create a RelationshipTracker instance."""
        if not RELATIONSHIP_TRACKER_AVAILABLE:
            return None
        try:
            # Use the positions DB if nomic_dir is set
            if nomic_dir:
                db_path = get_db_path(DatabaseType.POSITIONS, nomic_dir)
                if db_path.exists():
                    return RelationshipTracker(elo_db_path=str(db_path))
            # Fall back to default
            return RelationshipTracker()
        except Exception as e:
            logger.warning(f"Failed to create RelationshipTracker: {e}")
            return None

    @require_tracker
    def _get_summary(self, tracker: "RelationshipTracker") -> HandlerResult:
        """Get global relationship overview."""
        try:
            # Collect all unique agents and their relationships
            all_agents = set()
            all_relationships = []
            agent_relationship_counts = {}

            # We need to query the DB directly to get all relationships
            # Use a helper to get all pairs from the database
            with get_db_connection(str(tracker.elo_db_path)) as conn:
                cursor = conn.cursor()

                if not table_exists(cursor, "agent_relationships"):
                    return json_response({
                        "total_relationships": 0,
                        "strongest_rivalry": None,
                        "strongest_alliance": None,
                        "most_connected_agent": None,
                        "avg_rivalry_score": 0.0,
                        "avg_alliance_score": 0.0,
                    })

                # Safety limit to prevent memory explosion - cap at 10,000 relationships
                MAX_RELATIONSHIPS = 10000
                cursor.execute("""
                    SELECT agent_a, agent_b, debate_count, agreement_count,
                           a_wins_over_b, b_wins_over_a
                    FROM agent_relationships
                    WHERE debate_count >= 3
                    ORDER BY debate_count DESC
                    LIMIT ?
                """, (MAX_RELATIONSHIPS,))
                rows = cursor.fetchall()

                if len(rows) == MAX_RELATIONSHIPS:
                    logger.warning(f"Relationship summary hit limit of {MAX_RELATIONSHIPS} - results may be incomplete")

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

                # Compute scores inline (avoids N+1 query)
                rivalry_score = compute_rivalry_score(
                    debate_count, agreement_count, a_wins, b_wins
                )
                alliance_score = compute_alliance_score(
                    debate_count, agreement_count
                )

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

    @require_tracker
    def _get_graph(
        self, tracker: "RelationshipTracker", min_debates: int, min_score: float
    ) -> HandlerResult:
        """Get full relationship graph for visualizations."""
        try:
            with get_db_connection(str(tracker.elo_db_path)) as conn:
                cursor = conn.cursor()

                if not table_exists(cursor, "agent_relationships"):
                    return json_response({
                        "nodes": [],
                        "edges": [],
                        "stats": {"node_count": 0, "edge_count": 0}
                    })

                cursor.execute("""
                    SELECT agent_a, agent_b, debate_count, agreement_count,
                           a_wins_over_b, b_wins_over_a
                    FROM agent_relationships
                    WHERE debate_count >= ?
                """, (min_debates,))
                rows = cursor.fetchall()

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
                agent_a, agent_b, debate_count, agreement_count, a_wins, b_wins = row

                # Initialize node data
                for agent in [agent_a, agent_b]:
                    if agent not in nodes_data:
                        nodes_data[agent] = {"debate_count": 0, "rivals": 0, "allies": 0}

                # Compute scores inline (avoids N+1 query)
                rivalry_score = compute_rivalry_score(
                    debate_count, agreement_count, a_wins, b_wins
                )
                alliance_score = compute_alliance_score(
                    debate_count, agreement_count
                )

                # Apply score filter
                max_score = max(rivalry_score, alliance_score)
                if max_score < min_score:
                    continue

                # Determine relationship type and update node counters
                rel_type = determine_relationship_type(rivalry_score, alliance_score)
                if rel_type == "rivalry":
                    nodes_data[agent_a]["rivals"] += 1
                    nodes_data[agent_b]["rivals"] += 1
                elif rel_type == "alliance":
                    nodes_data[agent_a]["allies"] += 1
                    nodes_data[agent_b]["allies"] += 1

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

    @require_tracker
    def _get_pair_detail(
        self, tracker: "RelationshipTracker", agent_a: str, agent_b: str
    ) -> HandlerResult:
        """Get detailed relationship between two specific agents."""
        try:
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
            rel_type = determine_relationship_type(rivalry_score, alliance_score)

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

    @require_tracker
    def _get_stats(self, tracker: "RelationshipTracker") -> HandlerResult:
        """Get relationship system statistics."""
        try:
            with get_db_connection(str(tracker.elo_db_path)) as conn:
                cursor = conn.cursor()

                if not table_exists(cursor, "agent_relationships"):
                    return json_response({
                        "total_tracked_pairs": 0,
                        "total_debates_tracked": 0,
                        "rivalries": {"count": 0, "avg_score": 0.0},
                        "alliances": {"count": 0, "avg_score": 0.0},
                        "neutral": {"count": 0},
                        "most_debated_pair": None,
                        "highest_agreement_pair": None,
                    })

                # Get all relationships with full data for score computation
                cursor.execute("""
                    SELECT agent_a, agent_b, debate_count, agreement_count,
                           a_wins_over_b, b_wins_over_a
                    FROM agent_relationships
                """)
                rows = cursor.fetchall()

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
                agent_a, agent_b, debate_count, agreement_count, a_wins, b_wins = row
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

                    # Compute scores inline (avoids N+1 query)
                    rivalry_score = compute_rivalry_score(
                        debate_count, agreement_count, a_wins, b_wins
                    )
                    alliance_score = compute_alliance_score(
                        debate_count, agreement_count
                    )

                    rel_type = determine_relationship_type(rivalry_score, alliance_score)
                    if rel_type == "rivalry":
                        rivalries.append(rivalry_score)
                    elif rel_type == "alliance":
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
