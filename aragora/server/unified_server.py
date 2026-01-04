"""
Unified server combining HTTP API and WebSocket streaming.

Provides a single entry point for:
- HTTP API at /api/* endpoints
- WebSocket streaming at ws://host:port/ws
- Static file serving for the live dashboard
"""

import asyncio
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Optional
from urllib.parse import urlparse, parse_qs

from .stream import DebateStreamServer, SyncEventEmitter
from .storage import DebateStorage

# Optional Supabase persistence
try:
    from aragora.persistence import SupabaseClient
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    SupabaseClient = None

# Optional InsightStore for debate insights
try:
    from aragora.insights.store import InsightStore
    INSIGHTS_AVAILABLE = True
except ImportError:
    INSIGHTS_AVAILABLE = False
    InsightStore = None

# Optional EloSystem for agent rankings
try:
    from aragora.ranking.elo import EloSystem
    RANKING_AVAILABLE = True
except ImportError:
    RANKING_AVAILABLE = False
    EloSystem = None


class UnifiedHandler(BaseHTTPRequestHandler):
    """HTTP handler with API endpoints and static file serving."""

    storage: Optional[DebateStorage] = None
    static_dir: Optional[Path] = None
    stream_emitter: Optional[SyncEventEmitter] = None
    nomic_state_file: Optional[Path] = None
    persistence: Optional["SupabaseClient"] = None  # Supabase client for history
    insight_store: Optional["InsightStore"] = None  # InsightStore for debate insights
    elo_system: Optional["EloSystem"] = None  # EloSystem for agent rankings

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
        elif path == '/api/health':
            self._health_check()
        elif path == '/api/nomic/state':
            self._get_nomic_state()
        elif path == '/api/nomic/log':
            lines = int(query.get('lines', [100])[0])
            self._get_nomic_log(lines)

        # History API (Supabase)
        elif path == '/api/history/cycles':
            loop_id = query.get('loop_id', [None])[0]
            limit = int(query.get('limit', [50])[0])
            self._get_history_cycles(loop_id, limit)
        elif path == '/api/history/events':
            loop_id = query.get('loop_id', [None])[0]
            limit = int(query.get('limit', [100])[0])
            self._get_history_events(loop_id, limit)
        elif path == '/api/history/debates':
            loop_id = query.get('loop_id', [None])[0]
            limit = int(query.get('limit', [50])[0])
            self._get_history_debates(loop_id, limit)
        elif path == '/api/history/summary':
            loop_id = query.get('loop_id', [None])[0]
            self._get_history_summary(loop_id)

        # Insights API (debate consensus feature)
        elif path == '/api/insights/recent':
            limit = int(query.get('limit', [20])[0])
            self._get_recent_insights(limit)

        # Leaderboard API (debate consensus feature)
        elif path == '/api/leaderboard':
            limit = int(query.get('limit', [20])[0])
            domain = query.get('domain', [None])[0]
            self._get_leaderboard(limit, domain)
        elif path == '/api/matches/recent':
            limit = int(query.get('limit', [10])[0])
            self._get_recent_matches(limit)
        elif path.startswith('/api/agent/') and path.endswith('/history'):
            agent = path.split('/')[3]
            limit = int(query.get('limit', [30])[0])
            self._get_agent_history(agent, limit)

        # Static file serving
        elif path in ('/', '/index.html'):
            self._serve_file('index.html')
        elif path.endswith(('.html', '.css', '.js', '.json', '.ico', '.svg', '.png')):
            self._serve_file(path.lstrip('/'))
        else:
            # Try serving as a static file
            self._serve_file(path.lstrip('/'))

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self._add_cors_headers()
        self.end_headers()

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

    def _health_check(self) -> None:
        """Health check endpoint."""
        self._send_json({
            "status": "ok",
            "storage": self.storage is not None,
            "streaming": self.stream_emitter is not None,
            "static_dir": str(self.static_dir) if self.static_dir else None,
        })

    def _get_nomic_state(self) -> None:
        """Get current nomic loop state."""
        if not self.nomic_state_file or not self.nomic_state_file.exists():
            self._send_json({"status": "idle", "message": "No active nomic loop"})
            return

        try:
            with open(self.nomic_state_file) as f:
                state = json.load(f)
            self._send_json(state)
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)})

    def _get_nomic_log(self, lines: int = 100) -> None:
        """Get last N lines of nomic loop log."""
        if not self.nomic_state_file:
            self._send_json({"lines": []})
            return

        log_file = self.nomic_state_file.parent / "nomic_loop.log"
        if not log_file.exists():
            self._send_json({"lines": []})
            return

        try:
            with open(log_file) as f:
                all_lines = f.readlines()
            self._send_json({"lines": all_lines[-lines:]})
        except Exception as e:
            self._send_json({"lines": [], "error": str(e)})

    def _get_history_cycles(self, loop_id: Optional[str], limit: int) -> None:
        """Get nomic cycles from Supabase."""
        if not self.persistence:
            self._send_json({"error": "Persistence not configured", "cycles": []})
            return

        try:
            import asyncio
            cycles = asyncio.get_event_loop().run_until_complete(
                self.persistence.list_cycles(loop_id=loop_id, limit=limit)
            )
            self._send_json({
                "cycles": [c.to_dict() for c in cycles],
                "count": len(cycles),
            })
        except Exception as e:
            self._send_json({"error": str(e), "cycles": []})

    def _get_history_events(self, loop_id: Optional[str], limit: int) -> None:
        """Get stream events from Supabase."""
        if not self.persistence:
            self._send_json({"error": "Persistence not configured", "events": []})
            return

        if not loop_id:
            self._send_json({"error": "loop_id required", "events": []})
            return

        try:
            import asyncio
            events = asyncio.get_event_loop().run_until_complete(
                self.persistence.get_events(loop_id=loop_id, limit=limit)
            )
            self._send_json({
                "events": [e.to_dict() for e in events],
                "count": len(events),
            })
        except Exception as e:
            self._send_json({"error": str(e), "events": []})

    def _get_history_debates(self, loop_id: Optional[str], limit: int) -> None:
        """Get debate artifacts from Supabase."""
        if not self.persistence:
            self._send_json({"error": "Persistence not configured", "debates": []})
            return

        try:
            import asyncio
            debates = asyncio.get_event_loop().run_until_complete(
                self.persistence.list_debates(loop_id=loop_id, limit=limit)
            )
            self._send_json({
                "debates": [d.to_dict() for d in debates],
                "count": len(debates),
            })
        except Exception as e:
            self._send_json({"error": str(e), "debates": []})

    def _get_history_summary(self, loop_id: Optional[str]) -> None:
        """Get summary statistics for a loop."""
        if not self.persistence:
            self._send_json({"error": "Persistence not configured"})
            return

        if not loop_id:
            self._send_json({"error": "loop_id required"})
            return

        try:
            import asyncio
            summary = asyncio.get_event_loop().run_until_complete(
                self.persistence.get_loop_summary(loop_id)
            )
            self._send_json(summary)
        except Exception as e:
            self._send_json({"error": str(e)})

    def _get_recent_insights(self, limit: int) -> None:
        """Get recent insights from InsightStore (debate consensus feature)."""
        if not self.insight_store:
            self._send_json({"error": "Insights not configured", "insights": []})
            return

        try:
            import asyncio
            insights = asyncio.get_event_loop().run_until_complete(
                self.insight_store.get_recent_insights(limit=limit)
            )
            self._send_json({
                "insights": [
                    {
                        "id": i.id,
                        "type": i.type.value,
                        "title": i.title,
                        "description": i.description,
                        "confidence": i.confidence,
                        "agents_involved": i.agents_involved,
                        "evidence": i.evidence[:3] if i.evidence else [],
                    }
                    for i in insights
                ],
                "count": len(insights),
            })
        except Exception as e:
            self._send_json({"error": str(e), "insights": []})

    def _get_leaderboard(self, limit: int, domain: Optional[str]) -> None:
        """Get agent leaderboard by ELO ranking (debate consensus feature)."""
        if not self.elo_system:
            self._send_json({"error": "Rankings not configured", "agents": []})
            return

        try:
            agents = self.elo_system.get_leaderboard(limit=limit, domain=domain)
            self._send_json({
                "agents": [
                    {
                        "name": a.agent_name,
                        "elo": round(a.elo),
                        "wins": a.wins,
                        "losses": a.losses,
                        "draws": a.draws,
                        "win_rate": round(a.win_rate * 100, 1),
                        "games": a.games_played,
                    }
                    for a in agents
                ],
                "count": len(agents),
            })
        except Exception as e:
            self._send_json({"error": str(e), "agents": []})

    def _get_recent_matches(self, limit: int) -> None:
        """Get recent match results (debate consensus feature)."""
        if not self.elo_system:
            self._send_json({"error": "Rankings not configured", "matches": []})
            return

        try:
            # Use EloSystem's encapsulated method instead of raw SQL
            matches = self.elo_system.get_recent_matches(limit=limit)
            self._send_json({"matches": matches, "count": len(matches)})
        except Exception as e:
            self._send_json({"error": str(e), "matches": []})

    def _get_agent_history(self, agent: str, limit: int) -> None:
        """Get ELO history for an agent (debate consensus feature)."""
        if not self.elo_system:
            self._send_json({"error": "Rankings not configured", "history": []})
            return

        try:
            history = self.elo_system.get_elo_history(agent, limit=limit)
            self._send_json({
                "agent": agent,
                "history": [{"debate_id": h[0], "elo": h[1]} for h in history],
                "count": len(history),
            })
        except Exception as e:
            self._send_json({"error": str(e), "history": []})

    def _serve_file(self, filename: str) -> None:
        """Serve a static file."""
        if not self.static_dir:
            self.send_error(404, "Static directory not configured")
            return

        filepath = self.static_dir / filename
        if not filepath.exists():
            # Try index.html for SPA routing
            filepath = self.static_dir / "index.html"
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
        elif filename.endswith('.ico'):
            content_type = 'image/x-icon'
        elif filename.endswith('.svg'):
            content_type = 'image/svg+xml'
        elif filename.endswith('.png'):
            content_type = 'image/png'

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
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, format: str, *args) -> None:
        """Suppress default logging."""
        pass


class UnifiedServer:
    """
    Combined HTTP + WebSocket server for the nomic loop dashboard.

    Usage:
        server = UnifiedServer(
            http_port=8080,
            ws_port=8765,
            static_dir=Path("aragora/live/out"),
            nomic_dir=Path("/path/to/aragora/.nomic"),
        )
        await server.start()  # Starts both servers
    """

    def __init__(
        self,
        http_port: int = 8080,
        ws_port: int = 8765,
        ws_host: str = "0.0.0.0",
        http_host: str = "",
        static_dir: Optional[Path] = None,
        nomic_dir: Optional[Path] = None,
        storage: Optional[DebateStorage] = None,
        enable_persistence: bool = True,
    ):
        self.http_port = http_port
        self.ws_port = ws_port
        self.ws_host = ws_host
        self.http_host = http_host
        self.static_dir = static_dir
        self.nomic_dir = nomic_dir
        self.storage = storage

        # Create WebSocket server
        self.stream_server = DebateStreamServer(host=ws_host, port=ws_port)

        # Initialize Supabase persistence if available
        self.persistence = None
        if enable_persistence and PERSISTENCE_AVAILABLE:
            self.persistence = SupabaseClient()
            if self.persistence.is_configured:
                print("[server] Supabase persistence enabled")
            else:
                self.persistence = None

        # Setup HTTP handler
        UnifiedHandler.storage = storage
        UnifiedHandler.static_dir = static_dir
        UnifiedHandler.stream_emitter = self.stream_server.emitter
        UnifiedHandler.persistence = self.persistence
        if nomic_dir:
            UnifiedHandler.nomic_state_file = nomic_dir / "nomic_state.json"
            # Initialize InsightStore from nomic directory
            if INSIGHTS_AVAILABLE:
                insights_path = nomic_dir / "aragora_insights.db"
                if insights_path.exists():
                    UnifiedHandler.insight_store = InsightStore(str(insights_path))
                    print("[server] InsightStore loaded for API access")
            # Initialize EloSystem from nomic directory
            if RANKING_AVAILABLE:
                elo_path = nomic_dir / "agent_elo.db"
                if elo_path.exists():
                    UnifiedHandler.elo_system = EloSystem(str(elo_path))
                    print("[server] EloSystem loaded for leaderboard API")

    @property
    def emitter(self) -> SyncEventEmitter:
        """Get the event emitter for nomic loop integration."""
        return self.stream_server.emitter

    def _run_http_server(self) -> None:
        """Run HTTP server in a thread."""
        server = HTTPServer((self.http_host, self.http_port), UnifiedHandler)
        server.serve_forever()

    async def start(self) -> None:
        """Start both HTTP and WebSocket servers."""
        print(f"Starting unified server...")
        print(f"  HTTP API:   http://localhost:{self.http_port}")
        print(f"  WebSocket:  ws://localhost:{self.ws_port}")
        if self.static_dir:
            print(f"  Static dir: {self.static_dir}")
        if self.nomic_dir:
            print(f"  Nomic dir:  {self.nomic_dir}")

        # Start HTTP server in background thread
        http_thread = Thread(target=self._run_http_server, daemon=True)
        http_thread.start()

        # Start WebSocket server in foreground
        await self.stream_server.start()


async def run_unified_server(
    http_port: int = 8080,
    ws_port: int = 8765,
    static_dir: Optional[Path] = None,
    nomic_dir: Optional[Path] = None,
) -> None:
    """
    Convenience function to run the unified server.

    Args:
        http_port: Port for HTTP API (default 8080)
        ws_port: Port for WebSocket streaming (default 8765)
        static_dir: Directory containing static files (dashboard build)
        nomic_dir: Path to .nomic directory for state access
    """
    server = UnifiedServer(
        http_port=http_port,
        ws_port=ws_port,
        static_dir=static_dir,
        nomic_dir=nomic_dir,
    )
    await server.start()
