"""
HTTP API for debate retrieval and static file serving.

Provides REST endpoints for fetching debates and serves the viewer HTML.
"""

import json
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

from aragora.server.storage import DebateStorage
from aragora.replay.storage import ReplayStorage
from aragora.replay.reader import ReplayReader


class DebateAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for debate API."""

    storage: Optional[DebateStorage] = None
    replay_storage: Optional[ReplayStorage] = None
    static_dir: Optional[Path] = None

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # API routes
        if path.startswith('/api/debates/'):
            slug = path.split('/')[-1]
            self._get_debate(slug)
        elif path == '/api/debates':
            limit = int(query.get('limit', [20])[0])
            self._list_debates(limit)
        elif path.startswith('/api/replays/'):
            debate_id = path.split('/')[-1]
            self._get_replay(debate_id)
        elif path == '/api/replays':
            limit = int(query.get('limit', [20])[0])
            self._list_replays(limit)
        elif path == '/api/health':
            self._health_check()

        # Static file serving
        elif path in ('/', '/index.html'):
            self._serve_file('index.html')
        elif path == '/viewer.html' or path.startswith('/viewer'):
            self._serve_file('viewer.html')
        elif path.endswith(('.html', '.css', '.js', '.json')):
            self._serve_file(path.lstrip('/'))
        else:
            self.send_error(404, f"Not found: {path}")

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self._add_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        # Fork replay endpoint
        if path.startswith('/api/replays/') and path.endswith('/fork'):
            debate_id = path.split('/')[-3]  # /api/replays/{id}/fork
            self._fork_replay(debate_id)
        else:
            self.send_error(404, f"Not found: {path}")

    def _get_debate(self, slug: str) -> None:
        """Get a single debate by slug."""
        if not self.storage:
            self.send_error(500, "Storage not configured")
            return

        debate = self.storage.get_by_slug(slug)
        if debate:
            self._send_json(debate)
        else:
            self.send_error(404, f"Debate not found: {slug}")

    def _list_debates(self, limit: int = 20) -> None:
        """List recent debates."""
        if not self.storage:
            self._send_json([])
            return

        debates = self.storage.list_recent(limit)
        self._send_json([{
            "slug": d.slug,
            "task": d.task[:100] + "..." if len(d.task) > 100 else d.task,
            "agents": d.agents,
            "consensus": d.consensus_reached,
            "confidence": d.confidence,
            "views": d.view_count,
            "created": d.created_at.isoformat(),
        } for d in debates])

    def _list_replays(self, limit: int = 20) -> None:
        """List recent replays."""
        if not self.replay_storage:
            self._send_json([])
            return

        replays = self.replay_storage.list_recordings(limit)
        self._send_json(replays)

    def _get_replay(self, debate_id: str) -> None:
        """Get a replay bundle by debate_id."""
        if not self.replay_storage:
            self.send_error(500, "Replay storage not configured")
            return

        session_dir = self.replay_storage.storage_dir / debate_id
        meta_path = session_dir / "meta.json"
        events_path = session_dir / "events.jsonl"

        if not meta_path.exists() or not events_path.exists():
            self.send_error(404, f"Replay not found: {debate_id}")
            return

        try:
            # Read metadata
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            # Read events
            events = []
            with open(events_path, 'r', encoding='utf-8') as f:
                for line in f:
                    events.append(json.loads(line.strip()))

            self._send_json({
                "meta": meta,
                "events": events
            })
        except (OSError, json.JSONDecodeError) as e:
            self.send_error(500, f"Error reading replay: {e}")

    def _fork_replay(self, debate_id: str) -> None:
        """Fork a replay at a specific event into a new live debate."""
        if not self.replay_storage:
            self.send_error(500, "Replay storage not configured")
            return

        # Read POST data
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_error(400, "Missing request body")
            return

        try:
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(400, "Invalid JSON")
            return

        event_id = data.get("event_id")
        config_overrides = data.get("config", {})

        if not event_id:
            self.send_error(400, "Missing event_id")
            return

        # Load replay
        session_dir = self.replay_storage.storage_dir / debate_id
        meta_path = session_dir / "meta.json"
        events_path = session_dir / "events.jsonl"

        if not meta_path.exists() or not events_path.exists():
            self.send_error(404, f"Replay not found: {debate_id}")
            return

        try:
            # Read metadata
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            # Read events up to the fork point
            events = []
            with open(events_path, 'r', encoding='utf-8') as f:
                for line in f:
                    event = json.loads(line.strip())
                    events.append(event)
                    if event.get("event_id") == event_id:
                        break

            # Generate new debate ID
            fork_id = f"{debate_id}-fork-{uuid.uuid4().hex[:8]}"

            # Return fork information
            fork_data = {
                "fork_id": fork_id,
                "parent_id": debate_id,
                "fork_point": event_id,
                "status": "ready",
                "meta": meta,
                "events": events,
                "config_overrides": config_overrides,
                "message": "Fork created. Use this fork_id to start a new debate via WebSocket."
            }

            self._send_json(fork_data)

        except Exception as e:
            self.send_error(500, f"Fork failed: {str(e)}")
        except Exception as e:
            self.send_error(500, f"Error reading replay: {str(e)}")

    def _health_check(self) -> None:
        """Health check endpoint."""
        self._send_json({
            "status": "ok",
            "storage": self.storage is not None,
            "replay_storage": self.replay_storage is not None,
            "static_dir": str(self.static_dir) if self.static_dir else None,
        })

    def _serve_file(self, filename: str) -> None:
        """Serve a static file."""
        if not self.static_dir:
            self.send_error(404, "Static directory not configured")
            return

        filepath = self.static_dir / filename
        if not filepath.exists():
            self.send_error(404, f"File not found: {filename}")
            return

        # Determine content type
        content_type = 'text/html'
        if filename.endswith('.css'):
            content_type = 'text/css'
        elif filename.endswith('.js'):
            content_type = 'application/javascript'
        elif filename.endswith('.json'):
            content_type = 'application/json'

        try:
            content = filepath.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self._add_cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def _send_json(self, data) -> None:
        """Send JSON response."""
        content = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self._add_cors_headers()
        self.end_headers()
        self.wfile.write(content)

    def _add_cors_headers(self) -> None:
        """Add CORS headers for cross-origin requests."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, format: str, *args) -> None:
        """Suppress default logging (too verbose)."""
        pass


def run_api_server(
    storage: DebateStorage,
    replay_storage: Optional[ReplayStorage] = None,
    port: int = 8080,
    static_dir: Optional[Path] = None,
    host: str = "",
) -> None:
    """
    Run the HTTP API server.

    Args:
        storage: DebateStorage instance for debate retrieval
        replay_storage: ReplayStorage instance for replay retrieval
        port: Port to listen on (default 8080)
        static_dir: Directory containing static files (viewer.html, etc.)
        host: Host to bind to (default "" = all interfaces)
    """
    DebateAPIHandler.storage = storage
    DebateAPIHandler.replay_storage = replay_storage
    DebateAPIHandler.static_dir = static_dir

    server = HTTPServer((host, port), DebateAPIHandler)
    print(f"API server: http://localhost:{port}")
    print(f"  - GET /api/debates - List recent debates")
    print(f"  - GET /api/debates/<slug> - Get debate by slug")
    print(f"  - GET /api/replays - List recent replays")
    print(f"  - GET /api/replays/<debate_id> - Get replay bundle")
    print(f"  - GET /viewer.html?id=<slug> - View debate")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAPI server stopped")
        server.shutdown()
