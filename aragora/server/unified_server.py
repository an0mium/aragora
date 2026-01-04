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

from .stream import DebateStreamServer, SyncEventEmitter, StreamEvent, StreamEventType, create_arena_hooks
from .storage import DebateStorage
from .documents import DocumentStore, parse_document, get_supported_formats, SUPPORTED_EXTENSIONS

# For ad-hoc debates
import threading
import uuid

# Valid agent types (allowlist for security)
ALLOWED_AGENT_TYPES = frozenset({
    "codex", "claude", "openai", "gemini-cli", "grok-cli",
    "qwen-cli", "deepseek-cli", "gemini", "ollama",
    "anthropic-api", "openai-api"
})

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

# Optional FlipDetector for position reversal detection
try:
    from aragora.insights.flip_detector import (
        FlipDetector,
        format_flip_for_ui,
        format_consistency_for_ui,
    )
    FLIP_DETECTOR_AVAILABLE = True
except ImportError:
    FLIP_DETECTOR_AVAILABLE = False
    FlipDetector = None

# Optional debate orchestrator for ad-hoc debates
try:
    from aragora.debate.orchestrator import Arena, DebateProtocol
    from aragora.agents.base import create_agent
    from aragora.core import Environment
    DEBATE_AVAILABLE = True
except ImportError:
    DEBATE_AVAILABLE = False
    Arena = None
    DebateProtocol = None
    create_agent = None
    Environment = None

# Track active ad-hoc debates
_active_debates: dict[str, dict] = {}


def _run_async(coro):
    """Run async coroutine in HTTP handler thread (which may not have an event loop)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't use run_until_complete if loop is running
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop in this thread, create one
        return asyncio.run(coro)


class UnifiedHandler(BaseHTTPRequestHandler):
    """HTTP handler with API endpoints and static file serving."""

    storage: Optional[DebateStorage] = None
    static_dir: Optional[Path] = None
    stream_emitter: Optional[SyncEventEmitter] = None
    nomic_state_file: Optional[Path] = None
    persistence: Optional["SupabaseClient"] = None  # Supabase client for history
    insight_store: Optional["InsightStore"] = None  # InsightStore for debate insights
    elo_system: Optional["EloSystem"] = None  # EloSystem for agent rankings
    document_store: Optional[DocumentStore] = None  # Document store for uploads
    flip_detector: Optional["FlipDetector"] = None  # FlipDetector for position reversals

    def _safe_int(self, query: dict, key: str, default: int, max_val: int = 100) -> int:
        """Safely parse integer query param with bounds checking."""
        try:
            val = int(query.get(key, [default])[0])
            return min(max(val, 1), max_val)
        except (ValueError, IndexError, TypeError):
            return default

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
            limit = self._safe_int(query, 'limit', 20, 100)
            self._list_debates(limit)
        elif path == '/api/health':
            self._health_check()
        elif path == '/api/nomic/state':
            self._get_nomic_state()
        elif path == '/api/nomic/log':
            lines = self._safe_int(query, 'lines', 100, 1000)
            self._get_nomic_log(lines)

        # History API (Supabase)
        elif path == '/api/history/cycles':
            loop_id = query.get('loop_id', [None])[0]
            limit = self._safe_int(query, 'limit', 50, 200)
            self._get_history_cycles(loop_id, limit)
        elif path == '/api/history/events':
            loop_id = query.get('loop_id', [None])[0]
            limit = self._safe_int(query, 'limit', 100, 500)
            self._get_history_events(loop_id, limit)
        elif path == '/api/history/debates':
            loop_id = query.get('loop_id', [None])[0]
            limit = self._safe_int(query, 'limit', 50, 200)
            self._get_history_debates(loop_id, limit)
        elif path == '/api/history/summary':
            loop_id = query.get('loop_id', [None])[0]
            self._get_history_summary(loop_id)

        # Insights API (debate consensus feature)
        elif path == '/api/insights/recent':
            limit = self._safe_int(query, 'limit', 20, 100)
            self._get_recent_insights(limit)

        # Leaderboard API (debate consensus feature)
        elif path == '/api/leaderboard':
            limit = self._safe_int(query, 'limit', 20, 50)
            domain = query.get('domain', [None])[0]
            self._get_leaderboard(limit, domain)
        elif path == '/api/matches/recent':
            limit = self._safe_int(query, 'limit', 10, 50)
            loop_id = query.get('loop_id', [None])[0]
            self._get_recent_matches(limit, loop_id)
        elif path.startswith('/api/agent/') and path.endswith('/history'):
            agent = path.split('/')[3]
            limit = self._safe_int(query, 'limit', 30, 100)
            self._get_agent_history(agent, limit)

        # Document API
        elif path == '/api/documents':
            self._list_documents()
        elif path == '/api/documents/formats':
            self._send_json(get_supported_formats())
        elif path.startswith('/api/documents/'):
            doc_id = path.split('/')[-1]
            self._get_document(doc_id)

        # Replay API
        elif path == '/api/replays':
            self._list_replays()
        elif path.startswith('/api/replays/') and not path.endswith('/fork'):
            replay_id = path.split('/')[3]
            self._get_replay(replay_id)

        # Learning Evolution API
        elif path == '/api/learning/evolution':
            self._get_learning_evolution()

        # Flip Detection API
        elif path == '/api/flips/recent':
            limit = self._safe_int(query, 'limit', 20, 100)
            self._get_recent_flips(limit)
        elif path == '/api/flips/summary':
            self._get_flip_summary()
        elif path.startswith('/api/agent/') and path.endswith('/consistency'):
            agent = path.split('/')[3]
            self._get_agent_consistency(agent)
        elif path.startswith('/api/agent/') and path.endswith('/flips'):
            agent = path.split('/')[3]
            limit = self._safe_int(query, 'limit', 20, 100)
            self._get_agent_flips(agent, limit)

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

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/documents/upload':
            self._upload_document()
        elif path == '/api/debate':
            self._start_debate()
        else:
            self.send_error(404, f"Unknown POST endpoint: {path}")

    def _upload_document(self) -> None:
        """Handle document upload."""
        if not self.document_store:
            self._send_json({"error": "Document storage not configured"}, status=500)
            return

        # Get content length
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self._send_json({"error": "No content provided"}, status=400)
            return

        # Check max size (10MB)
        max_size = 10 * 1024 * 1024
        if content_length > max_size:
            self._send_json({"error": f"File too large. Max size: 10MB"}, status=400)
            return

        content_type = self.headers.get('Content-Type', '')

        # Handle multipart form data
        if 'multipart/form-data' in content_type:
            # Parse boundary
            boundary = None
            for part in content_type.split(';'):
                if 'boundary=' in part:
                    boundary = part.split('=')[1].strip()
                    break

            if not boundary:
                self._send_json({"error": "No boundary in multipart data"}, status=400)
                return

            # Read body
            body = self.rfile.read(content_length)

            # Parse multipart - simple implementation
            boundary_bytes = f'--{boundary}'.encode()
            parts = body.split(boundary_bytes)

            file_content = None
            filename = None

            for part in parts:
                if b'Content-Disposition' not in part:
                    continue

                # Extract filename
                try:
                    header_end = part.index(b'\r\n\r\n')
                    headers_raw = part[:header_end].decode('utf-8', errors='ignore')
                    file_data = part[header_end + 4:]

                    # Remove trailing boundary markers
                    if file_data.endswith(b'--\r\n'):
                        file_data = file_data[:-4]
                    elif file_data.endswith(b'\r\n'):
                        file_data = file_data[:-2]

                    # Extract filename from headers
                    if 'filename="' in headers_raw:
                        start = headers_raw.index('filename="') + 10
                        end = headers_raw.index('"', start)
                        filename = headers_raw[start:end]
                        file_content = file_data
                        break
                except (ValueError, IndexError):
                    continue

            if not file_content or not filename:
                self._send_json({"error": "No file found in upload"}, status=400)
                return

        else:
            # Raw file upload - get filename from header
            filename = self.headers.get('X-Filename', 'document.txt')
            file_content = self.rfile.read(content_length)

        # Validate file extension
        ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if ext not in SUPPORTED_EXTENSIONS:
            self._send_json({
                "error": f"Unsupported file type: {ext}",
                "supported": list(SUPPORTED_EXTENSIONS)
            }, status=400)
            return

        # Parse document
        try:
            doc = parse_document(file_content, filename)
            doc_id = self.document_store.add(doc)

            self._send_json({
                "success": True,
                "document": {
                    "id": doc_id,
                    "filename": doc.filename,
                    "word_count": doc.word_count,
                    "page_count": doc.page_count,
                    "preview": doc.preview,
                }
            })
        except ImportError as e:
            self._send_json({"error": str(e)}, status=400)
        except Exception as e:
            self._send_json({"error": f"Failed to parse document: {str(e)}"}, status=500)

    def _start_debate(self) -> None:
        """Start an ad-hoc debate with specified question.

        Accepts JSON body with:
            question: The topic/question to debate (required)
            agents: Comma-separated agent list (optional, default: "claude,openai")
            rounds: Number of debate rounds (optional, default: 3)
            consensus: Consensus method (optional, default: "majority")
        """
        global _active_debates

        if not DEBATE_AVAILABLE:
            self._send_json({"error": "Debate orchestrator not available"}, status=500)
            return

        if not self.stream_emitter:
            self._send_json({"error": "Event streaming not configured"}, status=500)
            return

        # Parse JSON body
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self._send_json({"error": "No content provided"}, status=400)
            return

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        # Validate required fields
        question = data.get('question', '').strip()
        if not question:
            self._send_json({"error": "question field is required"}, status=400)
            return

        # Parse optional fields
        agents_str = data.get('agents', 'claude,openai')
        rounds = min(max(int(data.get('rounds', 3)), 1), 10)  # Clamp to 1-10
        consensus = data.get('consensus', 'majority')

        # Generate debate ID
        debate_id = f"adhoc_{uuid.uuid4().hex[:8]}"

        # Track this debate
        _active_debates[debate_id] = {
            "id": debate_id,
            "question": question,
            "status": "starting",
            "agents": agents_str,
            "rounds": rounds,
        }

        # Set loop_id on emitter so events are tagged
        self.stream_emitter.set_loop_id(debate_id)

        # Start debate in background thread
        def run_debate():
            import asyncio as _asyncio

            try:
                # Parse agents
                agent_specs = []
                for spec in agents_str.split(","):
                    spec = spec.strip()
                    if ":" in spec:
                        agent_type, role = spec.split(":", 1)
                    else:
                        agent_type = spec
                        role = None
                    # Validate agent type against allowlist
                    if agent_type.lower() not in ALLOWED_AGENT_TYPES:
                        raise ValueError(f"Invalid agent type: {agent_type}. Allowed: {', '.join(sorted(ALLOWED_AGENT_TYPES))}")
                    agent_specs.append((agent_type, role))

                # Create agents
                agents = []
                for i, (agent_type, role) in enumerate(agent_specs):
                    if role is None:
                        if i == 0:
                            role = "proposer"
                        elif i == len(agent_specs) - 1:
                            role = "synthesizer"
                        else:
                            role = "critic"
                    agent = create_agent(
                        model_type=agent_type,
                        name=f"{agent_type}_{role}",
                        role=role,
                    )
                    agents.append(agent)

                # Create environment and protocol
                env = Environment(task=question, context="", max_rounds=rounds)
                protocol = DebateProtocol(rounds=rounds, consensus=consensus)

                # Create arena with hooks
                hooks = create_arena_hooks(self.stream_emitter)
                arena = Arena(env, agents, protocol, event_hooks=hooks, event_emitter=self.stream_emitter, loop_id=debate_id)

                # Run debate
                _active_debates[debate_id]["status"] = "running"
                result = _asyncio.run(arena.run())
                _active_debates[debate_id]["status"] = "completed"
                _active_debates[debate_id]["result"] = {
                    "final_answer": result.final_answer,
                    "consensus_reached": result.consensus_reached,
                    "confidence": result.confidence,
                }
            except Exception as e:
                _active_debates[debate_id]["status"] = "error"
                _active_debates[debate_id]["error"] = str(e)
                # Emit error event
                self.stream_emitter.emit(StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"error": str(e), "debate_id": debate_id},
                ))

        debate_thread = threading.Thread(target=run_debate, daemon=True)
        debate_thread.start()

        # Return immediately with debate ID
        self._send_json({
            "success": True,
            "debate_id": debate_id,
            "question": question,
            "agents": agents_str.split(","),
            "rounds": rounds,
            "status": "starting",
            "message": "Debate started. Connect to WebSocket to receive events.",
        })

    def _list_documents(self) -> None:
        """List all uploaded documents."""
        if not self.document_store:
            self._send_json({"documents": [], "error": "Document storage not configured"})
            return

        docs = self.document_store.list_all()
        self._send_json({"documents": docs, "count": len(docs)})

    def _get_document(self, doc_id: str) -> None:
        """Get a document by ID."""
        if not self.document_store:
            self.send_error(500, "Document storage not configured")
            return

        doc = self.document_store.get(doc_id)
        if doc:
            self._send_json(doc.to_dict())
        else:
            self.send_error(404, f"Document not found: {doc_id}")

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
            cycles = _run_async(
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
            events = _run_async(
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
            debates = _run_async(
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
            summary = _run_async(
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
            insights = _run_async(
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

    def _get_recent_matches(self, limit: int, loop_id: Optional[str] = None) -> None:
        """Get recent match results (debate consensus feature).

        Args:
            limit: Maximum number of matches to return
            loop_id: Optional loop ID to filter matches by (multi-loop support)
        """
        if not self.elo_system:
            self._send_json({"error": "Rankings not configured", "matches": []})
            return

        try:
            # Use EloSystem's encapsulated method instead of raw SQL
            matches = self.elo_system.get_recent_matches(limit=limit)
            # Filter by loop_id if provided (multi-loop support)
            if loop_id:
                matches = [m for m in matches if m.get('loop_id') == loop_id]
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

    def _list_replays(self) -> None:
        """List available replay directories."""
        if not self.nomic_state_file:
            self._send_json([])
            return

        try:
            replays_dir = self.nomic_state_file.parent / "replays"
            if not replays_dir.exists():
                self._send_json([])
                return

            replays = []
            for replay_path in replays_dir.iterdir():
                if replay_path.is_dir():
                    meta_file = replay_path / "meta.json"
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text())
                        replays.append({
                            "id": replay_path.name,
                            "topic": meta.get("topic", replay_path.name),
                            "agents": [a.get("name") for a in meta.get("agents", [])],
                            "schema_version": meta.get("schema_version", "1.0"),
                        })
            self._send_json(sorted(replays, key=lambda x: x["id"], reverse=True))
        except Exception as e:
            self._send_json({"error": str(e)})

    def _get_replay(self, replay_id: str) -> None:
        """Get a specific replay with events."""
        if not self.nomic_state_file:
            self._send_json({"error": "Replays not configured"})
            return

        try:
            replay_dir = self.nomic_state_file.parent / "replays" / replay_id
            if not replay_dir.exists():
                self._send_json({"error": f"Replay not found: {replay_id}"})
                return

            # Load meta
            meta_file = replay_dir / "meta.json"
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

            # Load events
            events_file = replay_dir / "events.jsonl"
            events = []
            if events_file.exists():
                for line in events_file.read_text().strip().split("\n"):
                    if line:
                        events.append(json.loads(line))

            self._send_json({
                "id": replay_id,
                "meta": meta,
                "events": events,
            })
        except Exception as e:
            self._send_json({"error": str(e)})

    def _get_learning_evolution(self) -> None:
        """Get learning/evolution data from meta_learning.db."""
        if not self.nomic_state_file:
            self._send_json({"error": "Learning data not configured", "patterns": []})
            return

        try:
            import sqlite3
            db_path = self.nomic_state_file.parent / "meta_learning.db"
            if not db_path.exists():
                self._send_json({"patterns": [], "count": 0})
                return

            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # Get recent patterns
            cursor = conn.execute("""
                SELECT * FROM meta_patterns
                ORDER BY created_at DESC
                LIMIT 20
            """)
            patterns = [dict(row) for row in cursor.fetchall()]
            conn.close()

            self._send_json({
                "patterns": patterns,
                "count": len(patterns),
            })
        except Exception as e:
            self._send_json({"error": str(e), "patterns": []})

    def _get_recent_flips(self, limit: int) -> None:
        """Get recent position flips across all agents."""
        if not self.flip_detector:
            self._send_json({"error": "Flip detection not configured", "flips": []})
            return

        try:
            flips = self.flip_detector.get_recent_flips(limit=limit)
            self._send_json({
                "flips": [format_flip_for_ui(f) for f in flips],
                "count": len(flips),
            })
        except Exception as e:
            self._send_json({"error": str(e), "flips": []})

    def _get_flip_summary(self) -> None:
        """Get summary of all flips for dashboard display."""
        if not self.flip_detector:
            self._send_json({"error": "Flip detection not configured", "summary": {}})
            return

        try:
            summary = self.flip_detector.get_flip_summary()
            self._send_json({"summary": summary})
        except Exception as e:
            self._send_json({"error": str(e), "summary": {}})

    def _get_agent_consistency(self, agent: str) -> None:
        """Get consistency score for an agent."""
        if not self.flip_detector:
            self._send_json({"error": "Flip detection not configured", "consistency": {}})
            return

        try:
            score = self.flip_detector.get_agent_consistency(agent)
            self._send_json({"consistency": format_consistency_for_ui(score)})
        except Exception as e:
            self._send_json({"error": str(e), "consistency": {}})

    def _get_agent_flips(self, agent: str, limit: int) -> None:
        """Get flips for a specific agent."""
        if not self.flip_detector:
            self._send_json({"error": "Flip detection not configured", "flips": []})
            return

        try:
            flips = self.flip_detector.detect_flips_for_agent(agent, lookback_positions=limit)
            self._send_json({
                "agent": agent,
                "flips": [format_flip_for_ui(f) for f in flips],
                "count": len(flips),
            })
        except Exception as e:
            self._send_json({"error": str(e), "flips": []})

    def _serve_file(self, filename: str) -> None:
        """Serve a static file with path traversal protection."""
        if not self.static_dir:
            self.send_error(404, "Static directory not configured")
            return

        # Security: Resolve paths and prevent directory traversal
        try:
            filepath = (self.static_dir / filename).resolve()
            static_dir_resolved = self.static_dir.resolve()

            # Ensure resolved path is within static directory
            if not str(filepath).startswith(str(static_dir_resolved)):
                self.send_error(403, "Access denied")
                return
        except (ValueError, OSError):
            self.send_error(400, "Invalid path")
            return

        if not filepath.exists():
            # Try index.html for SPA routing
            filepath = self.static_dir / "index.html"
            if not filepath.exists():
                self.send_error(404, "File not found")
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
        except Exception:
            self.send_error(500, "Failed to read file")

    def _send_json(self, data, status: int = 200) -> None:
        """Send JSON response."""
        content = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self._add_cors_headers()
        self.end_headers()
        self.wfile.write(content)

    def _add_cors_headers(self) -> None:
        """Add CORS headers with origin validation."""
        # Security: Validate origin against allowlist instead of wildcard
        allowed_origins = {
            'http://localhost:3000',
            'http://localhost:8080',
            'http://127.0.0.1:3000',
            'http://127.0.0.1:8080',
            'https://aragora.ai',
            'https://www.aragora.ai',
        }
        request_origin = self.headers.get('Origin', '')

        if request_origin in allowed_origins:
            self.send_header('Access-Control-Allow-Origin', request_origin)
        elif not request_origin:
            # Same-origin requests don't have Origin header
            pass
        # else: no CORS header = browser blocks cross-origin request

        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Filename, Authorization')

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

            # Initialize FlipDetector from nomic directory
            if FLIP_DETECTOR_AVAILABLE:
                positions_path = nomic_dir / "aragora_positions.db"
                if positions_path.exists():
                    UnifiedHandler.flip_detector = FlipDetector(str(positions_path))
                    print("[server] FlipDetector loaded for position reversal API")

            # Initialize DocumentStore for file uploads
            doc_dir = nomic_dir / "documents"
            UnifiedHandler.document_store = DocumentStore(doc_dir)
            print(f"[server] DocumentStore initialized at {doc_dir}")

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
