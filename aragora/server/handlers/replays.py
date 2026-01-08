"""
Replays and learning evolution endpoint handlers.

Endpoints:
- GET /api/replays - List available replays
- GET /api/replays/:replay_id - Get specific replay with events
- GET /api/learning/evolution - Get meta-learning patterns
- GET /api/meta-learning/stats - Get meta-learning hyperparameters and efficiency stats
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    ttl_cache,
    safe_json_parse,
    handle_errors,
)
from aragora.config import (
    DB_TIMEOUT_SECONDS,
    CACHE_TTL_REPLAYS_LIST,
    CACHE_TTL_LEARNING_EVOLUTION,
    CACHE_TTL_META_LEARNING,
)
from aragora.memory.database import MemoryDatabase

logger = logging.getLogger(__name__)


class ReplaysHandler(BaseHandler):
    """Handler for replays and learning evolution endpoints."""

    ROUTES = [
        "/api/replays",
        "/api/learning/evolution",
        "/api/meta-learning/stats",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Dynamic route for specific replay
        if path.startswith("/api/replays/") and len(path.split('/')) == 4:
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route replay requests to appropriate methods."""
        nomic_dir = self.ctx.get("nomic_dir")

        if path == "/api/replays":
            return self._list_replays(nomic_dir)

        if path == "/api/learning/evolution":
            limit = get_int_param(query_params, 'limit', 20)
            return self._get_learning_evolution(nomic_dir, min(limit, 100))

        if path == "/api/meta-learning/stats":
            limit = get_int_param(query_params, 'limit', 20)
            return self._get_meta_learning_stats(nomic_dir, min(limit, 50))

        if path.startswith("/api/replays/"):
            replay_id, err = self.extract_path_param(path, 2, "replay_id")
            if err:
                return err
            # Support pagination for large replay files
            offset = get_int_param(query_params, 'offset', 0)
            limit = get_int_param(query_params, 'limit', 1000)
            return self._get_replay(nomic_dir, replay_id, offset, min(limit, 5000))

        return None

    @ttl_cache(ttl_seconds=CACHE_TTL_REPLAYS_LIST, key_prefix="replays_list", skip_first=True)
    @handle_errors("replays list retrieval")
    def _list_replays(self, nomic_dir: Optional[Path]) -> HandlerResult:
        """List available replay directories."""
        if not nomic_dir:
            return json_response([])

        replays_dir = nomic_dir / "replays"
        if not replays_dir.exists():
            return json_response([])

        replays = []
        for replay_path in replays_dir.iterdir():
            if replay_path.is_dir():
                meta_file = replay_path / "meta.json"
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text())
                        replays.append({
                            "id": replay_path.name,
                            "topic": meta.get("topic", replay_path.name),
                            "agents": [a.get("name") for a in meta.get("agents", [])],
                            "schema_version": meta.get("schema_version", "1.0"),
                        })
                    except json.JSONDecodeError:
                        # Skip malformed meta files
                        continue

        return json_response(sorted(replays, key=lambda x: x["id"], reverse=True))

    @handle_errors("replay retrieval")
    def _get_replay(
        self, nomic_dir: Optional[Path], replay_id: str, offset: int = 0, limit: int = 1000
    ) -> HandlerResult:
        """Get a specific replay with events (streaming with pagination).

        Args:
            nomic_dir: Base directory for nomic data
            replay_id: ID of the replay to fetch
            offset: Number of events to skip (for pagination)
            limit: Maximum number of events to return

        Returns:
            Replay metadata and paginated events
        """
        if not nomic_dir:
            return error_response("Replays not configured", 503)

        replay_dir = nomic_dir / "replays" / replay_id
        if not replay_dir.exists():
            return error_response(f"Replay not found: {replay_id}", 404)

        # Load meta
        meta_file = replay_dir / "meta.json"
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except json.JSONDecodeError:
                meta = {"error": "Failed to parse meta.json"}

        # Stream events with pagination (bounded memory usage)
        events_file = replay_dir / "events.jsonl"
        events: list[dict[str, Any]] = []
        total_events = 0
        if events_file.exists():
            with open(events_file, 'r') as f:
                for i, line in enumerate(f):
                    total_events += 1
                    # Skip until we reach offset
                    if i < offset:
                        continue
                    # Stop after limit
                    if len(events) >= limit:
                        continue  # Keep counting total
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        return json_response({
            "id": replay_id,
            "meta": meta,
            "events": events,
            "event_count": len(events),
            "total_events": total_events,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(events) < total_events,
        })

    @ttl_cache(ttl_seconds=CACHE_TTL_LEARNING_EVOLUTION, key_prefix="learning_evolution", skip_first=True)
    @handle_errors("learning evolution retrieval")
    def _get_learning_evolution(
        self, nomic_dir: Optional[Path], limit: int
    ) -> HandlerResult:
        """Get learning/evolution data from meta_learning.db."""
        if not nomic_dir:
            return json_response({"patterns": [], "count": 0})

        db_path = nomic_dir / "meta_learning.db"
        if not db_path.exists():
            return json_response({"patterns": [], "count": 0})

        try:
            db = MemoryDatabase(str(db_path))
            with db.connection() as conn:
                conn.row_factory = sqlite3.Row

                # Get recent patterns
                cursor = conn.execute("""
                    SELECT * FROM meta_patterns
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                patterns = [dict(row) for row in cursor.fetchall()]

            return json_response({
                "patterns": patterns,
                "count": len(patterns),
            })
        except sqlite3.OperationalError as e:
            # Table may not exist yet
            if "no such table" in str(e):
                return json_response({"patterns": [], "count": 0})
            raise

    @ttl_cache(ttl_seconds=CACHE_TTL_META_LEARNING, key_prefix="meta_learning_stats", skip_first=True)
    @handle_errors("meta learning stats retrieval")
    def _get_meta_learning_stats(
        self, nomic_dir: Optional[Path], limit: int
    ) -> HandlerResult:
        """Get meta-learning hyperparameters and efficiency stats.

        Returns current hyperparameters, adjustment history, and efficiency metrics.
        """
        if not nomic_dir:
            return json_response({
                "status": "no_data",
                "current_hyperparams": {},
                "adjustment_history": [],
                "efficiency_log": [],
            })

        db_path = nomic_dir / "meta_learning.db"
        if not db_path.exists():
            return json_response({
                "status": "no_database",
                "current_hyperparams": {},
                "adjustment_history": [],
                "efficiency_log": [],
            })

        try:
            db = MemoryDatabase(str(db_path))
            with db.connection() as conn:
                conn.row_factory = sqlite3.Row

                # Get current hyperparameters (most recent)
                cursor = conn.execute("""
                    SELECT hyperparams, metrics, adjustment_reason, created_at
                    FROM meta_hyperparams
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                current_hyperparams = safe_json_parse(row["hyperparams"], {}) if row else {}

                # Get adjustment history
                cursor = conn.execute("""
                    SELECT hyperparams, metrics, adjustment_reason, created_at
                    FROM meta_hyperparams
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                adjustment_history = [
                    {
                        "hyperparams": safe_json_parse(row["hyperparams"], {}),
                        "metrics": safe_json_parse(row["metrics"]),
                        "reason": row["adjustment_reason"],
                        "timestamp": row["created_at"],
                    }
                    for row in cursor.fetchall()
                ]

                # Get efficiency log
                cursor = conn.execute("""
                    SELECT cycle_number, metrics, created_at
                    FROM meta_efficiency_log
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                efficiency_log = [
                    {
                        "cycle": row["cycle_number"],
                        "metrics": safe_json_parse(row["metrics"], {}),
                        "timestamp": row["created_at"],
                    }
                    for row in cursor.fetchall()
                ]

                # Compute trend from efficiency log
                trend = "insufficient_data"
                if len(efficiency_log) >= 4:
                    mid = len(efficiency_log) // 2
                    first_half = efficiency_log[mid:]  # Older entries (reversed order)
                    second_half = efficiency_log[:mid]  # Newer entries
                    if first_half and second_half:
                        first_retention = sum(
                            e.get("metrics", {}).get("pattern_retention_rate", 0.5)
                            for e in first_half
                        ) / len(first_half)
                        second_retention = sum(
                            e.get("metrics", {}).get("pattern_retention_rate", 0.5)
                            for e in second_half
                        ) / len(second_half)
                        if second_retention > first_retention + 0.05:
                            trend = "improving"
                        elif second_retention < first_retention - 0.05:
                            trend = "declining"
                        else:
                            trend = "stable"

            return json_response({
                "status": "ok",
                "current_hyperparams": current_hyperparams,
                "adjustment_history": adjustment_history,
                "efficiency_log": efficiency_log,
                "trend": trend,
                "evaluations": len(efficiency_log),
            })
        except sqlite3.OperationalError as e:
            # Table may not exist yet
            if "no such table" in str(e):
                return json_response({
                    "status": "tables_not_initialized",
                    "current_hyperparams": {},
                    "adjustment_history": [],
                    "efficiency_log": [],
                })
            raise
