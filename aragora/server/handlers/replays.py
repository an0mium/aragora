"""
Replays and learning evolution endpoint handlers.

Endpoints:
- GET /api/replays - List available replays
- GET /api/replays/:replay_id - Get specific replay with events
- GET /api/learning/evolution - Get meta-learning patterns
"""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    ttl_cache,
)
from aragora.config import DB_TIMEOUT_SECONDS
from aragora.server.validation import SAFE_ID_PATTERN

logger = logging.getLogger(__name__)

from aragora.server.error_utils import safe_error_message as _safe_error_message


class ReplaysHandler(BaseHandler):
    """Handler for replays and learning evolution endpoints."""

    ROUTES = [
        "/api/replays",
        "/api/learning/evolution",
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

        if path.startswith("/api/replays/"):
            # Block path traversal
            if '..' in path:
                return error_response("Invalid replay ID", 400)
            replay_id = path.split('/')[-1]
            if not replay_id or not re.match(SAFE_ID_PATTERN, replay_id):
                return error_response("Invalid replay ID format", 400)
            return self._get_replay(nomic_dir, replay_id)

        return None

    @ttl_cache(ttl_seconds=120, key_prefix="replays_list", skip_first=True)
    def _list_replays(self, nomic_dir: Optional[Path]) -> HandlerResult:
        """List available replay directories."""
        if not nomic_dir:
            return json_response([])

        try:
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
        except Exception as e:
            return error_response(_safe_error_message(e, "list_replays"), 500)

    @ttl_cache(ttl_seconds=300, key_prefix="replay_detail", skip_first=True)
    def _get_replay(self, nomic_dir: Optional[Path], replay_id: str) -> HandlerResult:
        """Get a specific replay with events."""
        if not nomic_dir:
            return error_response("Replays not configured", 503)

        try:
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

            # Load events
            events_file = replay_dir / "events.jsonl"
            events = []
            if events_file.exists():
                for line in events_file.read_text().strip().split("\n"):
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
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "get_replay"), 500)

    @ttl_cache(ttl_seconds=600, key_prefix="learning_evolution", skip_first=True)
    def _get_learning_evolution(
        self, nomic_dir: Optional[Path], limit: int
    ) -> HandlerResult:
        """Get learning/evolution data from meta_learning.db."""
        if not nomic_dir:
            return json_response({"patterns": [], "count": 0})

        try:
            db_path = nomic_dir / "meta_learning.db"
            if not db_path.exists():
                return json_response({"patterns": [], "count": 0})

            with sqlite3.connect(str(db_path), timeout=DB_TIMEOUT_SECONDS) as conn:
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
            return error_response(_safe_error_message(e, "learning_evolution"), 500)
        except Exception as e:
            return error_response(_safe_error_message(e, "learning_evolution"), 500)
