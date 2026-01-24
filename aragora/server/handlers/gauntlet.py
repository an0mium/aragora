"""
Gauntlet endpoint handlers for adversarial stress-testing.

Endpoints:
- POST /api/gauntlet/run - Start a gauntlet stress-test
- GET /api/gauntlet/{id} - Get gauntlet status/results
- GET /api/gauntlet/{id}/receipt - Get decision receipt
- GET /api/gauntlet/{id}/heatmap - Get risk heatmap
- GET /api/gauntlet/personas - List available personas
- GET /api/gauntlet/results - List recent results with pagination
- GET /api/gauntlet/{id}/compare/{id2} - Compare two gauntlet runs
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import logging
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

if TYPE_CHECKING:
    from aragora.gauntlet.storage import GauntletStorage

from aragora.server.validation.entities import validate_gauntlet_id
from aragora.server.validation.schema import GAUNTLET_RUN_SCHEMA, validate_against_schema

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_int_param,
    get_string_param,
    json_response,
    safe_error_message,
)
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


# In-memory storage for in-flight gauntlet runs (pending/running)
# Completed runs are persisted to GauntletStorage
# Using OrderedDict for FIFO eviction when memory limit reached
import os
import threading
from collections import OrderedDict

_gauntlet_runs: OrderedDict[str, dict[str, Any]] = OrderedDict()

# Memory management for gauntlet runs
MAX_GAUNTLET_RUNS_IN_MEMORY = 500
_GAUNTLET_COMPLETED_TTL = 3600  # Keep completed runs for 1 hour
_GAUNTLET_MAX_AGE_SECONDS = 7200  # Max 2 hours for any entry regardless of status

# Lock for atomic quota check-and-increment (prevents TOCTOU race)
_quota_lock = threading.Lock()

# Enable durable job queue for gauntlet execution (survives restarts)
# Set ARAGORA_DURABLE_GAUNTLET=0 to disable (enabled by default)
_USE_DURABLE_QUEUE = os.environ.get("ARAGORA_DURABLE_GAUNTLET", "1").lower() not in (
    "0",
    "false",
    "no",
)


def _handle_task_exception(task: asyncio.Task[Any], task_name: str) -> None:
    """Handle exceptions from fire-and-forget async tasks."""
    if task.cancelled():
        logger.debug(f"Task {task_name} was cancelled")
    elif task.exception():
        exc = task.exception()
        logger.error(f"Task {task_name} failed with exception: {exc}", exc_info=exc)


def create_tracked_task(coro: Any, name: str) -> asyncio.Task[Any]:
    """Create an async task with exception logging.

    Use this instead of raw asyncio.create_task() for fire-and-forget tasks
    to ensure exceptions are logged rather than silently ignored.
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(lambda t: _handle_task_exception(t, name))
    return task


# Persistent storage singleton
_storage: Optional["GauntletStorage"] = None

# WebSocket broadcast function (set by unified server when streaming is enabled)
_gauntlet_broadcast_fn: Optional[Callable[..., Any]] = None


def set_gauntlet_broadcast_fn(broadcast_fn: Callable[..., Any]) -> None:
    """Set the broadcast function for WebSocket streaming."""
    global _gauntlet_broadcast_fn
    _gauntlet_broadcast_fn = broadcast_fn


def _get_storage() -> "GauntletStorage":
    """Get or create the persistent storage instance."""
    global _storage
    if _storage is None:
        from aragora.gauntlet.storage import GauntletStorage

        _storage = GauntletStorage()
    return _storage


def _cleanup_gauntlet_runs() -> None:
    """Remove old runs from memory to prevent unbounded growth.

    Cleanup strategy:
    1. Remove any entry older than MAX_AGE (regardless of status)
    2. Remove completed entries older than COMPLETED_TTL
    3. If still over limit, evict oldest entries (FIFO)
    """
    global _gauntlet_runs
    now = time.time()
    to_remove = []

    for run_id, run in _gauntlet_runs.items():
        created_at = run.get("created_at")

        # Try to get creation time from various fields
        entry_time = None
        if isinstance(created_at, (int, float)):
            entry_time = created_at
        elif isinstance(created_at, str):
            try:
                entry_time = datetime.fromisoformat(created_at).timestamp()
            except (ValueError, TypeError):
                pass

        # If no valid timestamp, check completed_at
        if entry_time is None:
            completed_at = run.get("completed_at")
            if completed_at:
                try:
                    entry_time = datetime.fromisoformat(completed_at).timestamp()
                except (ValueError, TypeError):
                    pass

        # Remove entries older than MAX_AGE regardless of status
        if entry_time and (now - entry_time) > _GAUNTLET_MAX_AGE_SECONDS:
            to_remove.append(run_id)
            continue

        # Remove completed entries older than TTL
        if run.get("status") == "completed":
            completed_at = run.get("completed_at")
            if completed_at:
                try:
                    completed_time = datetime.fromisoformat(completed_at).timestamp()
                    if now - completed_time > _GAUNTLET_COMPLETED_TTL:
                        to_remove.append(run_id)
                except (ValueError, TypeError):
                    pass

    for run_id in to_remove:
        _gauntlet_runs.pop(run_id, None)

    # If still over limit, evict oldest entries (FIFO via OrderedDict)
    while len(_gauntlet_runs) > MAX_GAUNTLET_RUNS_IN_MEMORY:
        _gauntlet_runs.popitem(last=False)  # Remove oldest


def recover_stale_gauntlet_runs(max_age_seconds: int = 7200) -> int:
    """
    Recover stale inflight gauntlet runs after server restart.

    Finds runs that were pending/running when the server stopped and marks
    them as interrupted. This should be called during server startup.

    Args:
        max_age_seconds: Maximum age in seconds for a run to be considered stale

    Returns:
        Number of stale runs recovered/marked as interrupted
    """
    try:
        storage = _get_storage()
        stale_runs = storage.list_stale_inflight(max_age_seconds=max_age_seconds)

        if not stale_runs:
            logger.debug("No stale gauntlet runs found to recover")
            return 0

        recovered = 0
        for run in stale_runs:
            try:
                # Mark as interrupted with error message
                storage.update_inflight_status(
                    gauntlet_id=run.gauntlet_id,
                    status="interrupted",
                    error=f"Server restarted while run was {run.status}. "
                    f"Progress was {run.progress_percent:.0f}% in phase '{run.current_phase or 'unknown'}'.",
                )

                # Also add to in-memory dict for immediate access
                _gauntlet_runs[run.gauntlet_id] = {
                    "gauntlet_id": run.gauntlet_id,
                    "status": "interrupted",
                    "input_type": run.input_type,
                    "input_summary": run.input_summary,
                    "persona": run.persona,
                    "agents": run.agents,
                    "profile": run.profile,
                    "created_at": run.created_at.isoformat(),
                    "error": f"Server restarted while run was {run.status}",
                    "progress_percent": run.progress_percent,
                    "current_phase": run.current_phase,
                }

                logger.info(
                    f"Marked stale gauntlet run {run.gauntlet_id} as interrupted "
                    f"(was {run.status}, {run.progress_percent:.0f}% complete)"
                )
                recovered += 1

            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(f"Failed to recover stale run {run.gauntlet_id}: {e}")

        if recovered:
            logger.info(f"Recovered {recovered} stale gauntlet runs after server restart")

        return recovered

    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.warning(f"Failed to recover stale gauntlet runs: {e}")
        return 0


class GauntletHandler(BaseHandler):
    """Handler for gauntlet stress-testing endpoints.

    Supports both versioned (/api/v1/gauntlet/*) and legacy (/api/gauntlet/*) routes.
    Legacy routes return a Deprecation header and should be migrated to v1.
    """

    # API version for this handler
    API_VERSION = "v1"

    # Gauntlet API routes
    ROUTES = [
        "/api/v1/gauntlet/run",
        "/api/v1/gauntlet/personas",
        "/api/v1/gauntlet/results",
        "/api/v1/gauntlet/*/receipt/verify",
        "/api/v1/gauntlet/*/receipt",
        "/api/v1/gauntlet/*/heatmap",
        "/api/v1/gauntlet/*/export",
        "/api/v1/gauntlet/*/compare/*",
        "/api/v1/gauntlet/*",
    ]

    # All gauntlet endpoints require authentication
    AUTH_REQUIRED_ENDPOINTS = [
        "/api/v1/gauntlet/run",
        "/api/v1/gauntlet/",
    ]

    def __init__(self, server_context: dict):
        super().__init__(server_context)  # type: ignore[arg-type]
        emitter = server_context.get("stream_emitter")
        if emitter and hasattr(emitter, "emit"):
            set_gauntlet_broadcast_fn(emitter.emit)

    def _is_legacy_route(self, path: str) -> bool:
        """Check if this is a legacy (non-versioned) route."""
        return path.startswith("/api/v1/gauntlet/") and not path.startswith("/api/v1/")

    def _normalize_path(self, path: str) -> str:
        """Normalize path by removing version prefix for routing logic."""
        if path.startswith("/api/v1/gauntlet/"):
            return path.replace("/api/v1/gauntlet/", "/api/v1/gauntlet/")
        return path

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the request.

        Supports both versioned (/api/v1/gauntlet/*) and legacy (/api/gauntlet/*) routes.
        """
        # Normalize path for matching
        normalized = self._normalize_path(path)

        # When called without method (e.g., from route index), just check path prefix
        if method == "GET" and normalized.startswith("/api/v1/gauntlet/"):
            return True
        if normalized == "/api/v1/gauntlet/run" and method == "POST":
            return True
        if normalized == "/api/v1/gauntlet/personas" and method == "GET":
            return True
        if normalized == "/api/v1/gauntlet/results" and method == "GET":
            return True
        if normalized.startswith("/api/v1/gauntlet/") and method == "GET":
            return True
        if normalized.startswith("/api/v1/gauntlet/") and method == "DELETE":
            return True
        return False

    def _add_version_headers(self, result: HandlerResult, original_path: str) -> HandlerResult:
        """Add API version headers and deprecation warning for legacy routes."""
        if result is None:
            return result

        # Initialize headers if needed
        if result.headers is None:
            result.headers = {}

        # Add API version header
        result.headers["X-API-Version"] = self.API_VERSION

        # Add deprecation header for legacy routes
        if self._is_legacy_route(original_path):
            result.headers["Deprecation"] = "true"
            result.headers["Sunset"] = "2026-06-01"  # 6 months notice
            result.headers["Link"] = f'</api/v1{original_path[4:]}>; rel="successor-version"'
            logger.debug(f"Legacy route accessed: {original_path}")

        return result

    @rate_limit(rpm=10)
    async def handle(  # type: ignore[override]
        self, path: str, method: str, handler: Any = None
    ) -> Optional[HandlerResult]:
        """Route request to appropriate handler.

        Note: This handler uses a different signature than BaseHandler.handle()
        because it needs the HTTP method to route requests appropriately.
        The unified server calls this with (path, method, handler) for
        handlers that implement can_handle with method support.

        Supports both versioned (/api/v1/gauntlet/*) and legacy (/api/gauntlet/*) routes.
        """
        original_path = path
        query_params: dict[str, Any] = {}
        if handler:
            query_str = handler.path.split("?", 1)[1] if "?" in handler.path else ""
            from urllib.parse import parse_qs

            query_params = parse_qs(query_str)

        # Normalize path for routing (remove version prefix)
        path = self._normalize_path(path)

        result: Optional[HandlerResult] = None

        # POST /api/gauntlet/run
        if path == "/api/v1/gauntlet/run" and method == "POST":
            result = await self._start_gauntlet(handler)

        # GET /api/gauntlet/personas
        elif path == "/api/v1/gauntlet/personas":
            result = self._list_personas()

        # GET /api/gauntlet/results - List with pagination
        elif path == "/api/v1/gauntlet/results":
            result = self._list_results(query_params)

        # POST /api/gauntlet/{id}/receipt/verify
        elif path.endswith("/receipt/verify") and method == "POST":
            gauntlet_id = path.split("/")[-3]
            is_valid, err = validate_gauntlet_id(gauntlet_id)
            if not is_valid:
                return error_response(err, 400)
            result = await self._verify_receipt(gauntlet_id, handler)

        # GET /api/gauntlet/{id}/receipt
        elif path.endswith("/receipt"):
            gauntlet_id = path.split("/")[-2]
            is_valid, err = validate_gauntlet_id(gauntlet_id)
            if not is_valid:
                return error_response(err, 400)
            result = await self._get_receipt(gauntlet_id, query_params)

        # GET /api/gauntlet/{id}/heatmap
        elif path.endswith("/heatmap"):
            gauntlet_id = path.split("/")[-2]
            is_valid, err = validate_gauntlet_id(gauntlet_id)
            if not is_valid:
                return error_response(err, 400)
            result = await self._get_heatmap(gauntlet_id, query_params)

        # GET /api/gauntlet/{id}/export
        elif path.endswith("/export"):
            gauntlet_id = path.split("/")[-2]
            is_valid, err = validate_gauntlet_id(gauntlet_id)
            if not is_valid:
                return error_response(err, 400)
            result = await self._export_report(gauntlet_id, query_params, handler)

        # GET /api/gauntlet/{id}/compare/{id2}
        elif "/compare/" in path:
            parts = path.split("/")
            if len(parts) >= 5:
                gauntlet_id = parts[-3]
                compare_id = parts[-1]
                # Validate both IDs
                is_valid, err = validate_gauntlet_id(gauntlet_id)
                if not is_valid:
                    return error_response(err, 400)
                is_valid, err = validate_gauntlet_id(compare_id)
                if not is_valid:
                    return error_response(f"Invalid compare ID: {err}", 400)
                result = self._compare_results(gauntlet_id, compare_id, query_params)

        # DELETE /api/gauntlet/{id}
        elif method == "DELETE" and path.startswith("/api/v1/gauntlet/"):
            gauntlet_id = path.split("/")[-1]
            if gauntlet_id and gauntlet_id not in ("run", "personas", "results"):
                is_valid, err = validate_gauntlet_id(gauntlet_id)
                if not is_valid:
                    return error_response(err, 400)
                result = self._delete_result(gauntlet_id, query_params)

        # GET /api/gauntlet/{id}
        elif path.startswith("/api/v1/gauntlet/"):
            gauntlet_id = path.split("/")[-1]
            if gauntlet_id and gauntlet_id not in ("run", "personas", "results"):
                is_valid, err = validate_gauntlet_id(gauntlet_id)
                if not is_valid:
                    return error_response(err, 400)
                result = await self._get_status(gauntlet_id)

        # Add version headers to result
        if result is not None:
            result = self._add_version_headers(result, original_path)

        return result

    def _list_personas(self) -> HandlerResult:
        """List available regulatory personas."""
        try:
            from aragora.gauntlet.personas import get_persona, list_personas

            personas_list = []
            for name in list_personas():
                persona = get_persona(name)
                personas_list.append(
                    {
                        "id": name,
                        "name": persona.name,
                        "description": persona.description,
                        "regulation": persona.regulation,
                        "attack_count": len(persona.attack_prompts),
                        "categories": list(set(a.category for a in persona.attack_prompts)),
                    }
                )

            return json_response(
                {
                    "personas": personas_list,
                    "count": len(personas_list),
                }
            )
        except ImportError:
            return json_response(
                {
                    "personas": [],
                    "count": 0,
                    "error": "Personas module not available",
                }
            )

    async def _start_gauntlet(self, handler: Any) -> HandlerResult:
        """Start a new gauntlet stress-test."""
        # Check quota before proceeding
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = None
        if hasattr(handler, "user_store"):
            user_store = handler.user_store
        elif hasattr(handler.__class__, "user_store"):
            user_store = handler.__class__.user_store

        user_ctx = extract_user_from_request(handler, user_store) if user_store else None

        # Atomic quota check-and-increment to prevent TOCTOU race condition
        # Lock ensures no concurrent requests can pass quota check simultaneously
        if user_ctx and user_ctx.is_authenticated and user_ctx.org_id:
            if user_store and hasattr(user_store, "get_organization_by_id"):
                with _quota_lock:
                    org = user_store.get_organization_by_id(user_ctx.org_id)
                    if org:
                        if org.is_at_limit:
                            return json_response(
                                {
                                    "error": "Monthly debate quota exceeded",
                                    "code": "quota_exceeded",
                                    "limit": org.limits.debates_per_month,
                                    "used": org.debates_used_this_month,
                                    "remaining": 0,
                                    "tier": org.tier.value,
                                    "upgrade_url": "/pricing",
                                    "message": f"Your {org.tier.value} plan allows {org.limits.debates_per_month} debates per month. Gauntlet runs count as debates. Upgrade to increase your limit.",
                                },
                                status=429,
                            )
                        # Increment immediately while holding lock to prevent race
                        if hasattr(user_store, "increment_usage"):
                            try:
                                user_store.increment_usage(user_ctx.org_id, 1)
                                logger.info(f"Incremented gauntlet usage for org {user_ctx.org_id}")
                            except Exception as ue:
                                logger.warning(
                                    f"Usage increment failed for org {user_ctx.org_id}: {ue}"
                                )

        # Parse request body (with Content-Length validation)
        data = self.read_json_body(handler)
        if data is None:
            return error_response("Invalid or too large request body", 400)

        # Validate request body against schema
        validation_result = validate_against_schema(data, GAUNTLET_RUN_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

        # Extract parameters (already validated)
        input_content = data.get("input_content", "")
        input_type = data.get("input_type", "spec")
        persona = data.get("persona")
        agents = data.get("agents", ["anthropic-api"])
        profile = data.get("profile", "default")

        # Generate gauntlet ID
        gauntlet_id = f"gauntlet-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        input_hash = hashlib.sha256(input_content.encode()).hexdigest()

        # Cleanup old completed runs before storing new one
        _cleanup_gauntlet_runs()

        # Calculate input summary
        input_summary = input_content[:200] + "..." if len(input_content) > 200 else input_content

        # Store initial state in-memory (for immediate access)
        _gauntlet_runs[gauntlet_id] = {
            "gauntlet_id": gauntlet_id,
            "status": "pending",
            "input_type": input_type,
            "input_summary": input_summary,
            "input_hash": input_hash,
            "persona": persona,
            "profile": profile,
            "created_at": datetime.now().isoformat(),
            "result": None,
        }

        # Persist to database for durability across server restarts
        # Include config_json so jobs can be recovered if server restarts
        import json as _json

        config_json = _json.dumps(
            {
                "input_content": input_content,
                "input_type": input_type,
                "persona": persona,
                "agents": agents,
                "profile": profile,
            }
        )

        try:
            storage = _get_storage()
            storage.save_inflight(
                gauntlet_id=gauntlet_id,
                status="pending",
                input_type=input_type,
                input_summary=input_summary,
                input_hash=input_hash,
                persona=persona,
                profile=profile,
                agents=agents,
                config_json=config_json,
            )
            logger.debug(f"Persisted inflight gauntlet run: {gauntlet_id}")
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"Failed to persist inflight gauntlet {gauntlet_id}: {e}")

        # Use durable job queue if enabled, otherwise fire-and-forget
        if _USE_DURABLE_QUEUE:
            # Durable queue - survives server restarts, supports retry
            try:
                from aragora.queue.workers.gauntlet_worker import enqueue_gauntlet_job

                create_tracked_task(
                    enqueue_gauntlet_job(
                        gauntlet_id=gauntlet_id,
                        input_content=input_content,
                        input_type=input_type,
                        persona=persona,
                        agents=agents,
                        profile=profile,
                    ),
                    name=f"enqueue-gauntlet-{gauntlet_id}",
                )
                logger.info(f"Enqueued gauntlet {gauntlet_id} to durable job queue")
            except ImportError as ie:
                logger.warning(f"Durable queue unavailable, falling back: {ie}")
                create_tracked_task(
                    self._run_gauntlet_async(
                        gauntlet_id, input_content, input_type, persona, agents, profile
                    ),
                    name=f"gauntlet-{gauntlet_id}",
                )
        else:
            # Fire-and-forget - simpler but doesn't survive restarts
            create_tracked_task(
                self._run_gauntlet_async(
                    gauntlet_id, input_content, input_type, persona, agents, profile
                ),
                name=f"gauntlet-{gauntlet_id}",
            )

        # Note: Usage increment moved to atomic check-and-increment section above

        return json_response(
            {
                "gauntlet_id": gauntlet_id,
                "status": "pending",
                "message": "Gauntlet stress-test started",
                "durable_queue": _USE_DURABLE_QUEUE,
            },
            status=202,
        )

    async def _run_gauntlet_async(
        self,
        gauntlet_id: str,
        input_content: str,
        input_type: str,
        persona: Optional[str],
        agents: list[str],
        profile: str,
    ) -> None:
        """Run gauntlet asynchronously."""
        try:
            from aragora.agents.base import AgentType, create_agent
            from aragora.gauntlet import (
                GauntletOrchestrator,
                GauntletProgress,
                InputType,
                OrchestratorConfig,
            )
            from aragora.server.stream.gauntlet_emitter import GauntletStreamEmitter

            # Create stream emitter if broadcasting is available
            emitter: Optional[GauntletStreamEmitter] = None
            if _gauntlet_broadcast_fn:
                emitter = GauntletStreamEmitter(
                    broadcast_fn=_gauntlet_broadcast_fn,
                    gauntlet_id=gauntlet_id,
                )

            # Update status (both in-memory and persistent)
            _gauntlet_runs[gauntlet_id]["status"] = "running"
            try:
                storage = _get_storage()
                storage.update_inflight_status(gauntlet_id, "running")
            except (OSError, RuntimeError, ValueError) as e:
                logger.debug(f"Failed to update inflight status: {e}")

            # Create agents
            agent_instances = []
            for agent_type in agents:
                try:
                    agent = create_agent(
                        model_type=cast(AgentType, agent_type),
                        name=f"{agent_type}_gauntlet",
                        role="auditor",
                    )
                    agent_instances.append(agent)
                except (ImportError, ValueError, RuntimeError) as e:
                    logger.warning(f"Could not create agent {agent_type}: {e}")

            if not agent_instances:
                _gauntlet_runs[gauntlet_id]["status"] = "failed"
                _gauntlet_runs[gauntlet_id]["error"] = "No agents could be created"
                return

            # Map input type
            input_type_map = {
                "spec": InputType.SPEC,
                "architecture": InputType.ARCHITECTURE,
                "policy": InputType.POLICY,
                "code": InputType.CODE,
                "strategy": InputType.STRATEGY,
                "contract": InputType.CONTRACT,
            }
            input_type_enum = input_type_map.get(input_type, InputType.SPEC)

            # Create config
            config = OrchestratorConfig(
                input_type=input_type_enum,
                input_content=input_content,
                persona=persona,
                max_duration_seconds=300,  # 5 minute max for API
            )

            # Emit start event
            if emitter:
                emitter.emit_start(
                    gauntlet_id=gauntlet_id,
                    input_type=input_type,
                    input_summary=input_content[:500],
                    agents=[a.name for a in agent_instances],
                    config_summary={"profile": profile, "persona": persona},
                )

            # Create progress callback that also emits streaming events and persists state
            def on_progress(progress: GauntletProgress) -> None:
                """Handle progress updates with streaming and persistence."""
                if emitter:
                    emitter.emit_progress(
                        progress=progress.percent / 100.0,
                        phase=progress.phase,
                        message=progress.message,
                    )
                    if progress.current_task:
                        emitter.emit_phase(progress.current_task, progress.message)

                # Update persistent status (throttled to avoid too many DB writes)
                # Only update on significant progress changes (every 10%)
                if int(progress.percent) % 10 == 0:
                    try:
                        storage = _get_storage()
                        storage.update_inflight_status(
                            gauntlet_id,
                            "running",
                            current_phase=progress.phase,
                            progress_percent=progress.percent,
                        )
                    except (OSError, RuntimeError, ValueError):
                        pass  # Non-critical, continue execution

            # Run gauntlet with progress callback
            orchestrator = GauntletOrchestrator(agent_instances, on_progress=on_progress)
            result = await orchestrator.run(config)

            # Emit verdict and complete events
            if emitter:
                emitter.emit_verdict(
                    verdict=result.verdict.value,
                    confidence=result.confidence,
                    risk_score=result.risk_score,
                    robustness_score=result.robustness_score,
                    critical_count=len(result.critical_findings),
                    high_count=len(result.high_findings),
                    medium_count=len(result.medium_findings),
                    low_count=len(result.low_findings),
                )
                emitter.emit_complete(
                    gauntlet_id=gauntlet_id,
                    verdict=result.verdict.value,
                    confidence=result.confidence,
                    findings_count=result.total_findings,
                    duration_seconds=result.duration_seconds,
                )

            # Store result
            completed_at = datetime.now().isoformat()
            result_dict = {
                "gauntlet_id": gauntlet_id,
                "verdict": result.verdict.value,
                "confidence": result.confidence,
                "risk_score": result.risk_score,
                "robustness_score": result.robustness_score,
                "coverage_score": result.coverage_score,
                "total_findings": result.total_findings,
                "critical_count": len(result.critical_findings),
                "high_count": len(result.high_findings),
                "medium_count": len(result.medium_findings),
                "low_count": len(result.low_findings),
                "findings": [
                    {
                        "id": f.finding_id,
                        "category": f.category,
                        "severity": f.severity,
                        "severity_level": f.severity_level,
                        "title": f.title,
                        "description": f.description[:500],
                    }
                    for f in result.all_findings[:20]  # Limit to 20 findings
                ],
            }

            # Update in-memory state
            _gauntlet_runs[gauntlet_id]["status"] = "completed"
            _gauntlet_runs[gauntlet_id]["completed_at"] = completed_at
            _gauntlet_runs[gauntlet_id]["result_obj"] = result
            _gauntlet_runs[gauntlet_id]["result"] = result_dict

            # Persist to storage
            try:
                storage = _get_storage()
                storage.save(result)
                logger.info(f"Gauntlet {gauntlet_id} persisted to storage")

                # Clean up inflight record after successful completion
                storage.delete_inflight(gauntlet_id)
            except (OSError, RuntimeError, ValueError) as storage_err:
                logger.warning(f"Failed to persist gauntlet {gauntlet_id}: {storage_err}")

            # Auto-persist decision receipt
            await self._auto_persist_receipt(result, gauntlet_id)

            # Clean up in-memory storage after persisting (keep result_obj for receipt generation)
            # In-memory entry can be removed after a timeout in production

        except (OSError, RuntimeError, ValueError, ImportError, asyncio.CancelledError) as e:
            logger.error(f"Gauntlet {gauntlet_id} failed: {e}")
            _gauntlet_runs[gauntlet_id]["status"] = "failed"
            _gauntlet_runs[gauntlet_id]["error"] = str(e)

            # Update persistent status to failed
            try:
                storage = _get_storage()
                storage.update_inflight_status(
                    gauntlet_id,
                    "failed",
                    error=str(e),
                )
            except (OSError, RuntimeError, ValueError):
                pass

    async def _get_status(self, gauntlet_id: str) -> HandlerResult:
        """Get gauntlet run status."""
        # Check in-memory first (for pending/running)
        if gauntlet_id in _gauntlet_runs:
            run = _gauntlet_runs[gauntlet_id]
            safe_run = {k: v for k, v in run.items() if k != "result_obj"}
            return json_response(safe_run)

        # Check persistent storage
        try:
            storage = _get_storage()

            # Check inflight table first (for in-progress runs after restart)
            inflight = storage.get_inflight(gauntlet_id)
            if inflight:
                return json_response(inflight.to_dict())

            # Check completed results table
            stored = storage.get(gauntlet_id)
            if stored:
                return json_response(
                    {
                        "gauntlet_id": gauntlet_id,
                        "status": "completed",
                        "result": stored,
                    }
                )
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"Storage lookup failed for {gauntlet_id}: {e}")

        return error_response(f"Gauntlet run not found: {gauntlet_id}", 404)

    async def _auto_persist_receipt(self, result: Any, gauntlet_id: str) -> None:
        """Auto-persist decision receipt after gauntlet completion.

        Generates and stores a decision receipt for compliance and audit trail.
        Optionally signs the receipt if ARAGORA_AUTO_SIGN_RECEIPTS=true.
        """
        try:
            from aragora.gauntlet.receipt import DecisionReceipt
            from aragora.storage.receipt_store import StoredReceipt, get_receipt_store

            # Get run data for input hash
            run = _gauntlet_runs.get(gauntlet_id, {})

            # Generate receipt from result
            receipt = DecisionReceipt.from_mode_result(
                result,
                input_hash=run.get("input_hash"),
            )

            # Create stored receipt
            stored = StoredReceipt(
                receipt_id=receipt.receipt_id,
                gauntlet_id=gauntlet_id,
                debate_id=getattr(result, "debate_id", None),
                created_at=time.time(),
                expires_at=None,  # Receipts don't expire by default
                verdict=receipt.verdict,
                confidence=receipt.confidence,
                risk_level=self._risk_level_from_score(receipt.robustness_score),
                risk_score=1.0 - receipt.robustness_score,  # Invert: higher score = lower risk
                checksum=hashlib.sha256(str(receipt.to_dict()).encode()).hexdigest(),
                data=receipt.to_dict(),
            )

            # Save to receipt store
            store = get_receipt_store()
            store.save(receipt.to_dict())
            logger.info(f"Decision receipt auto-persisted: {receipt.receipt_id}")

            # Emit receipt generated webhook
            try:
                from aragora.integrations.receipt_webhooks import get_receipt_notifier

                notifier = get_receipt_notifier()
                debate_id = getattr(result, "debate_id", None) or gauntlet_id
                agents = getattr(result, "agents_involved", None) or getattr(result, "agents", None)
                rounds = getattr(result, "rounds_completed", None) or getattr(
                    result, "rounds_used", None
                )
                findings_count = getattr(result, "total_findings", None)
                if findings_count is None:
                    findings_count = len(getattr(receipt, "vulnerability_details", []) or [])
                notifier.notify_receipt_generated(
                    receipt_id=receipt.receipt_id,
                    debate_id=debate_id,
                    verdict=receipt.verdict,
                    confidence=receipt.confidence,
                    hash=stored.checksum,
                    agents=agents,
                    rounds=rounds,
                    findings_count=findings_count,
                )
            except Exception as e:
                logger.debug(f"Receipt webhook notification skipped: {e}")

            # Optional auto-signing
            if os.environ.get("ARAGORA_AUTO_SIGN_RECEIPTS", "").lower() in ("true", "1", "yes"):
                try:
                    from aragora.gauntlet.signing import sign_receipt

                    signed = sign_receipt(receipt.to_dict())
                    store.update_signature(
                        receipt.receipt_id,
                        signature=signed.signature,
                        algorithm=signed.signature_metadata.algorithm,
                        key_id=signed.signature_metadata.key_id,
                    )
                    logger.info(f"Receipt auto-signed: {receipt.receipt_id}")
                except (ImportError, ValueError) as sign_err:
                    logger.warning(f"Auto-signing failed for {receipt.receipt_id}: {sign_err}")

        except ImportError as e:
            logger.debug(f"Receipt persistence skipped (module not available): {e}")
        except Exception as e:
            logger.warning(f"Failed to auto-persist receipt for {gauntlet_id}: {e}")

    def _risk_level_from_score(self, robustness_score: float) -> str:
        """Determine risk level from robustness score."""
        if robustness_score >= 0.8:
            return "LOW"
        elif robustness_score >= 0.6:
            return "MEDIUM"
        elif robustness_score >= 0.4:
            return "HIGH"
        else:
            return "CRITICAL"

    async def _get_receipt(self, gauntlet_id: str, query_params: dict) -> HandlerResult:
        """Get decision receipt for gauntlet run."""
        from aragora.gauntlet.errors import gauntlet_error_response
        from aragora.gauntlet.receipt import DecisionReceipt

        run = None
        result = None
        result_obj = None

        # Check in-memory first
        if gauntlet_id in _gauntlet_runs:
            run = _gauntlet_runs[gauntlet_id]
            if run["status"] != "completed":
                body, status = gauntlet_error_response(
                    "not_completed", {"gauntlet_id": gauntlet_id}
                )
                return json_response(body, status=status)
            result = run["result"]
            result_obj = run.get("result_obj")
        else:
            # Check persistent storage
            try:
                storage = _get_storage()
                stored = storage.get(gauntlet_id)
                if stored:
                    result = stored
                else:
                    body, status = gauntlet_error_response(
                        "gauntlet_not_found", {"gauntlet_id": gauntlet_id}
                    )
                    return json_response(body, status=status)
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(f"Storage lookup failed for {gauntlet_id}: {e}")
                body, status = gauntlet_error_response("storage_error", {"reason": str(e)})
                return json_response(body, status=status)

        # Generate receipt
        if result_obj:
            receipt = DecisionReceipt.from_mode_result(
                result_obj,
                input_hash=run.get("input_hash") if run else None,
            )
        else:
            receipt = DecisionReceipt(
                receipt_id=f"receipt-{gauntlet_id[-12:]}",
                gauntlet_id=gauntlet_id,
                timestamp=run.get("completed_at", "") if run else datetime.now().isoformat(),
                input_summary=run["input_summary"] if run else result.get("input_summary", ""),
                input_hash=(
                    run.get("input_hash", gauntlet_id)
                    if run
                    else result.get("input_hash", gauntlet_id)
                ),
                risk_summary={
                    "critical": result.get("critical_count", 0),
                    "high": result.get("high_count", 0),
                    "medium": result.get("medium_count", 0),
                    "low": result.get("low_count", 0),
                    "total": result.get("total_findings", 0),
                },
                attacks_attempted=0,
                attacks_successful=0,
                probes_run=0,
                vulnerabilities_found=result.get("total_findings", 0),
                verdict=result.get("verdict", "UNKNOWN").upper(),
                confidence=result.get("confidence", 0),
                robustness_score=result.get("robustness_score", 0),
            )

        # Return format based on query param
        # Supported formats: json (default), html, md, sarif, pdf, csv
        format_type = get_string_param(query_params, "format", "json")
        # Check if cryptographic signing is requested
        signed = get_string_param(query_params, "signed", "false") == "true"

        # Prepare receipt data (potentially signed)
        receipt_data = receipt.to_dict()
        if signed:
            try:
                from aragora.gauntlet.signing import sign_receipt

                signed_receipt = sign_receipt(receipt_data)
                receipt_data = signed_receipt.to_dict()
            except (ImportError, ValueError) as e:
                logger.warning(f"Receipt signing failed: {e}")
                # Continue with unsigned receipt

        def _notify_export(export_format: str, size_bytes: Optional[int] = None) -> None:
            try:
                from aragora.integrations.receipt_webhooks import get_receipt_notifier

                notifier = get_receipt_notifier()
                notifier.notify_receipt_exported(
                    receipt_id=receipt.receipt_id,
                    debate_id=gauntlet_id,
                    export_format=export_format,
                    file_size=size_bytes,
                )
            except Exception as e:
                logger.debug(f"Receipt export webhook skipped: {e}")

        if format_type == "html":
            html_bytes = receipt.to_html().encode("utf-8")
            _notify_export("html", len(html_bytes))
            return HandlerResult(
                status_code=200,
                content_type="text/html",
                body=html_bytes,
            )
        elif format_type == "md":
            md_bytes = receipt.to_markdown().encode("utf-8")
            _notify_export("markdown", len(md_bytes))
            return HandlerResult(
                status_code=200,
                content_type="text/markdown",
                body=md_bytes,
            )
        elif format_type == "sarif":
            # SARIF 2.1.0 format for security tool integration
            sarif_bytes = receipt.to_sarif_json().encode("utf-8")
            _notify_export("sarif", len(sarif_bytes))
            return HandlerResult(
                status_code=200,
                content_type="application/sarif+json",
                body=sarif_bytes,
                headers={"Content-Disposition": f'attachment; filename="{gauntlet_id}.sarif"'},
            )
        elif format_type == "pdf":
            # PDF format (requires weasyprint)
            try:
                pdf_bytes = receipt.to_pdf()
                _notify_export("pdf", len(pdf_bytes))
                return HandlerResult(
                    status_code=200,
                    content_type="application/pdf",
                    body=pdf_bytes,
                    headers={
                        "Content-Disposition": f'attachment; filename="{gauntlet_id}-receipt.pdf"'
                    },
                )
            except ImportError:
                return error_response(
                    "PDF export requires weasyprint. Install with: pip install weasyprint",
                    501,
                )
        elif format_type == "csv":
            # CSV format for spreadsheet import
            csv_bytes = receipt.to_csv().encode("utf-8")
            _notify_export("csv", len(csv_bytes))
            return HandlerResult(
                status_code=200,
                content_type="text/csv",
                body=csv_bytes,
                headers={
                    "Content-Disposition": f'attachment; filename="{gauntlet_id}-findings.csv"'
                },
            )
        else:
            _notify_export("json", len(json.dumps(receipt_data)))
            return json_response(receipt_data)

    async def _verify_receipt(self, gauntlet_id: str, handler: Any) -> HandlerResult:
        """Verify a signed decision receipt.

        Validates:
        1. Cryptographic signature authenticity
        2. Artifact hash integrity (content not tampered)
        3. Receipt ID matches gauntlet ID

        Request body should be a SignedReceipt dict with:
        - receipt: The receipt data
        - signature: Base64-encoded signature
        - signature_metadata: Algorithm, timestamp, key_id

        Returns verification result with detailed status.
        """
        from aragora.gauntlet.receipt import DecisionReceipt
        from aragora.gauntlet.signing import SignedReceipt, verify_receipt

        # Parse request body
        data = self.read_json_body(handler)
        if data is None:
            return error_response("Invalid or missing request body", 400)

        # Validate required fields
        if "receipt" not in data or "signature" not in data:
            return error_response("Missing required fields: 'receipt' and 'signature'", 400)

        if "signature_metadata" not in data:
            return error_response("Missing required field: 'signature_metadata'", 400)

        try:
            # Parse signed receipt
            signed_receipt = SignedReceipt.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            return error_response(f"Invalid signed receipt format: {e}", 400)

        # Initialize verification result
        verification_result = {
            "gauntlet_id": gauntlet_id,
            "receipt_id": signed_receipt.receipt_data.get("receipt_id"),
            "verified": False,
            "signature_valid": False,
            "integrity_valid": False,
            "id_match": False,
            "errors": [],
            "warnings": [],
            "verified_at": datetime.now().isoformat(),
        }

        # Check receipt ID matches gauntlet ID
        receipt_gauntlet_id = signed_receipt.receipt_data.get("gauntlet_id")
        if receipt_gauntlet_id == gauntlet_id:
            verification_result["id_match"] = True
        else:
            verification_result["errors"].append(
                f"Receipt gauntlet_id '{receipt_gauntlet_id}' does not match "
                f"requested gauntlet_id '{gauntlet_id}'"
            )

        # Verify cryptographic signature
        try:
            signature_valid = verify_receipt(signed_receipt)
            verification_result["signature_valid"] = signature_valid
            if not signature_valid:
                verification_result["errors"].append("Cryptographic signature is invalid")
        except (ImportError, ValueError, RuntimeError) as e:
            verification_result["errors"].append(f"Signature verification failed: {e}")

        # Verify artifact hash integrity
        try:
            receipt_dict = signed_receipt.receipt_data
            # Reconstruct DecisionReceipt to check integrity
            receipt = DecisionReceipt(
                receipt_id=receipt_dict.get("receipt_id", ""),
                gauntlet_id=receipt_dict.get("gauntlet_id", ""),
                timestamp=receipt_dict.get("timestamp", ""),
                input_summary=receipt_dict.get("input_summary", ""),
                input_hash=receipt_dict.get("input_hash", ""),
                risk_summary=receipt_dict.get("risk_summary", {}),
                attacks_attempted=receipt_dict.get("attacks_attempted", 0),
                attacks_successful=receipt_dict.get("attacks_successful", 0),
                probes_run=receipt_dict.get("probes_run", 0),
                vulnerabilities_found=receipt_dict.get("vulnerabilities_found", 0),
                verdict=receipt_dict.get("verdict", ""),
                confidence=receipt_dict.get("confidence", 0.0),
                robustness_score=receipt_dict.get("robustness_score", 0.0),
                artifact_hash=receipt_dict.get("artifact_hash", ""),
            )

            integrity_valid = receipt.verify_integrity()
            verification_result["integrity_valid"] = integrity_valid
            if not integrity_valid:
                verification_result["errors"].append(
                    "Artifact hash mismatch - receipt content may have been tampered"
                )
        except (KeyError, TypeError, ValueError) as e:
            verification_result["errors"].append(f"Integrity verification failed: {e}")

        # Set overall verification status
        verification_result["verified"] = (
            verification_result["signature_valid"]
            and verification_result["integrity_valid"]
            and verification_result["id_match"]
        )

        # Add metadata about the verification
        verification_result["signature_metadata"] = {
            "algorithm": signed_receipt.signature_metadata.algorithm,
            "key_id": signed_receipt.signature_metadata.key_id,
            "signed_at": signed_receipt.signature_metadata.timestamp,
        }

        # Emit webhook based on verification result
        try:
            from aragora.integrations.receipt_webhooks import get_receipt_notifier

            notifier = get_receipt_notifier()
            receipt_id = signed_receipt.receipt_data.get("receipt_id", "")
            receipt_hash = signed_receipt.receipt_data.get(
                "artifact_hash", ""
            ) or signed_receipt.receipt_data.get("checksum", "")
            computed_hash = ""
            try:
                computed_hash = receipt._calculate_hash()
            except Exception:
                computed_hash = ""

            if verification_result["verified"]:
                notifier.notify_receipt_verified(
                    receipt_id=receipt_id,
                    debate_id=gauntlet_id,
                    hash=receipt_hash,
                    computed_hash=computed_hash,
                    valid=True,
                )
            else:
                notifier.notify_receipt_integrity_failed(
                    receipt_id=receipt_id,
                    debate_id=gauntlet_id,
                    expected_hash=receipt_hash,
                    computed_hash=computed_hash,
                    error_message="; ".join(verification_result.get("errors", []))
                    or "verification failed",
                )
        except Exception as e:
            logger.debug(f"Receipt verification webhook skipped: {e}")

        # Return appropriate status code
        if verification_result["verified"]:
            return json_response(verification_result)
        else:
            # Return 200 with verification failure details (not a client error)
            return json_response(verification_result)

    async def _get_heatmap(self, gauntlet_id: str, query_params: dict) -> HandlerResult:
        """Get risk heatmap for gauntlet run."""
        from aragora.gauntlet.heatmap import HeatmapCell, RiskHeatmap

        run = None
        result = None
        result_obj = None

        # Check in-memory first
        if gauntlet_id in _gauntlet_runs:
            run = _gauntlet_runs[gauntlet_id]
            if run["status"] != "completed":
                return error_response("Gauntlet run not completed", 400)
            result = run["result"]
            result_obj = run.get("result_obj")
        else:
            # Check persistent storage
            try:
                storage = _get_storage()
                stored = storage.get(gauntlet_id)
                if stored:
                    result = stored
                else:
                    return error_response(f"Gauntlet run not found: {gauntlet_id}", 404)
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(f"Storage lookup failed for {gauntlet_id}: {e}")
                return error_response(f"Gauntlet run not found: {gauntlet_id}", 404)

        # Generate heatmap
        if result_obj:
            heatmap = RiskHeatmap.from_mode_result(result_obj)
        else:
            # COMPATIBILITY: Use vulnerabilities if findings not present (post-restart scenario)
            findings_data = result.get("findings") or result.get("vulnerabilities", [])

            cells = []
            categories = set()
            severities = ["critical", "high", "medium", "low"]

            for finding in findings_data:
                category = finding.get("category", "unknown")
                categories.add(category)

            category_severity_counts: dict[tuple[str, str], int] = {}
            for finding in findings_data:
                category = finding.get("category", "unknown")
                severity = finding.get("severity_level", "medium").lower()
                key = (category, severity)
                category_severity_counts[key] = category_severity_counts.get(key, 0) + 1

            for category in sorted(categories):
                for severity in severities:
                    count = category_severity_counts.get((category, severity), 0)
                    cells.append(
                        HeatmapCell(
                            category=category,
                            severity=severity,
                            count=count,
                        )
                    )

            heatmap = RiskHeatmap(
                cells=cells,
                categories=sorted(list(categories)),
                severities=severities,
                total_findings=result.get("total_findings", 0),
            )

        # Return format based on query param
        format_type = get_string_param(query_params, "format", "json")

        if format_type == "svg":
            return HandlerResult(
                status_code=200,
                content_type="image/svg+xml",
                body=heatmap.to_svg().encode("utf-8"),
            )
        elif format_type == "ascii":
            return HandlerResult(
                status_code=200,
                content_type="text/plain",
                body=heatmap.to_ascii().encode("utf-8"),
            )
        else:
            return json_response(heatmap.to_dict())

    def _list_results(self, query_params: dict) -> HandlerResult:
        """List recent gauntlet results with pagination."""
        try:
            storage = _get_storage()

            limit = get_int_param(query_params, "limit", 20)
            offset = get_int_param(query_params, "offset", 0)
            verdict = get_string_param(query_params, "verdict", None)
            min_severity = get_string_param(query_params, "min_severity", None)

            # Clamp values
            limit = min(max(limit, 1), 100)
            offset = max(offset, 0)

            results = storage.list_recent(
                limit=limit,
                offset=offset,
                verdict=verdict,
                min_severity=min_severity,
            )

            total = storage.count(verdict=verdict)

            return json_response(
                {
                    "results": [
                        {
                            "gauntlet_id": r.gauntlet_id,
                            "input_hash": r.input_hash,
                            "input_summary": (
                                r.input_summary[:100] + "..."
                                if len(r.input_summary) > 100
                                else r.input_summary
                            ),
                            "verdict": r.verdict,
                            "confidence": r.confidence,
                            "robustness_score": r.robustness_score,
                            "critical_count": r.critical_count,
                            "high_count": r.high_count,
                            "total_findings": r.total_findings,
                            "created_at": r.created_at.isoformat(),
                            "duration_seconds": r.duration_seconds,
                        }
                        for r in results
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            )
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Failed to list results: {e}")
            return error_response(safe_error_message(e, "list results"), 500)

    def _compare_results(self, id1: str, id2: str, query_params: dict) -> HandlerResult:
        """Compare two gauntlet results."""
        try:
            storage = _get_storage()
            comparison = storage.compare(id1, id2)

            if comparison is None:
                return error_response("One or both gauntlet runs not found", 404)

            return json_response(comparison)
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Failed to compare results: {e}")
            return error_response(safe_error_message(e, "compare results"), 500)

    def _delete_result(self, gauntlet_id: str, query_params: dict) -> HandlerResult:
        """Delete a gauntlet result."""
        try:
            # Remove from in-memory if present
            if gauntlet_id in _gauntlet_runs:
                del _gauntlet_runs[gauntlet_id]

            # Remove from persistent storage
            storage = _get_storage()
            deleted = storage.delete(gauntlet_id)

            if deleted:
                return json_response({"deleted": True, "gauntlet_id": gauntlet_id})
            else:
                return error_response(f"Gauntlet run not found: {gauntlet_id}", 404)
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            logger.error(f"Failed to delete result: {e}")
            return error_response(safe_error_message(e, "delete result"), 500)

    async def _export_report(
        self, gauntlet_id: str, query_params: dict, handler: Any = None
    ) -> HandlerResult:
        """Export a comprehensive gauntlet report.

        Query params:
        - format: json (default), html, full_html (includes CSS)
        - include_heatmap: true/false (default true)
        - include_findings: true/false (default true)
        """

        # Get result
        run = None
        result = None
        _result_obj = None  # GauntletResult object for enhanced report (in-memory only)

        if gauntlet_id in _gauntlet_runs:
            run = _gauntlet_runs[gauntlet_id]
            if run["status"] != "completed":
                return error_response("Gauntlet run not completed", 400)
            result = run["result"]
            _result_obj = run.get("result_obj")
        else:
            try:
                storage = _get_storage()
                stored = storage.get(gauntlet_id)
                if stored:
                    result = stored
                else:
                    return error_response(f"Gauntlet run not found: {gauntlet_id}", 404)
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(f"Storage lookup failed for {gauntlet_id}: {e}")
                return error_response(f"Gauntlet run not found: {gauntlet_id}", 404)

        # Parse options
        format_type = get_string_param(query_params, "format", "json")
        include_heatmap = get_string_param(query_params, "include_heatmap", "true") == "true"
        include_findings = get_string_param(query_params, "include_findings", "true") == "true"

        # Build comprehensive report
        report: dict[str, Any] = {
            "gauntlet_id": gauntlet_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "verdict": result.get("verdict", "UNKNOWN"),
                "confidence": result.get("confidence", 0),
                "robustness_score": result.get("robustness_score", 0),
                "risk_score": result.get("risk_score", 0),
                "coverage_score": result.get("coverage_score", 0),
            },
            "findings_summary": {
                "total": result.get("total_findings", 0),
                "critical": result.get("critical_count", 0),
                "high": result.get("high_count", 0),
                "medium": result.get("medium_count", 0),
                "low": result.get("low_count", 0),
            },
            "input": {
                "summary": run.get("input_summary", "") if run else result.get("input_summary", ""),
                "type": run.get("input_type", "") if run else result.get("input_type", ""),
                "hash": run.get("input_hash", "") if run else result.get("input_hash", ""),
            },
            "timing": {
                "created_at": run.get("created_at", "") if run else "",
                "completed_at": run.get("completed_at", "") if run else "",
            },
        }

        if include_findings:
            # COMPATIBILITY: GauntletResult stores "vulnerabilities", but API expects "findings"
            # Map vulnerabilities to findings for backwards compatibility
            findings = result.get("findings") or result.get("vulnerabilities", [])
            report["findings"] = findings

        if include_heatmap:
            # Generate heatmap data
            # COMPATIBILITY: Use vulnerabilities if findings not present (post-restart scenario)
            findings_for_heatmap = result.get("findings") or result.get("vulnerabilities", [])

            cells = []
            categories = set()
            severities = ["critical", "high", "medium", "low"]

            for finding in findings_for_heatmap:
                category = finding.get("category", "unknown")
                categories.add(category)

            category_severity_counts: dict[tuple[str, str], int] = {}
            # COMPATIBILITY: Use findings_for_heatmap (includes vulnerabilities fallback)
            for finding in findings_for_heatmap:
                category = finding.get("category", "unknown")
                severity = finding.get("severity_level", finding.get("severity", "medium")).lower()
                key = (category, severity)
                category_severity_counts[key] = category_severity_counts.get(key, 0) + 1

            for category in sorted(categories):
                for severity in severities:
                    count = category_severity_counts.get((category, severity), 0)
                    cells.append({"category": category, "severity": severity, "count": count})

            report["heatmap"] = {
                "cells": cells,
                "categories": sorted(list(categories)),
                "severities": severities,
            }

        # Enhanced report data from GauntletResult object (in-memory runs only)
        if _result_obj is not None:
            enhanced: dict[str, Any] = {}
            if hasattr(_result_obj, "verdict_reasoning") and _result_obj.verdict_reasoning:
                enhanced["verdict_reasoning"] = _result_obj.verdict_reasoning
            if hasattr(_result_obj, "attack_summary"):
                enhanced["attack_summary"] = (
                    _result_obj.attack_summary.__dict__
                    if hasattr(_result_obj.attack_summary, "__dict__")
                    else str(_result_obj.attack_summary)
                )
            if hasattr(_result_obj, "probe_summary"):
                enhanced["probe_summary"] = (
                    _result_obj.probe_summary.__dict__
                    if hasattr(_result_obj.probe_summary, "__dict__")
                    else str(_result_obj.probe_summary)
                )
            if hasattr(_result_obj, "scenario_summary"):
                enhanced["scenario_summary"] = (
                    _result_obj.scenario_summary.__dict__
                    if hasattr(_result_obj.scenario_summary, "__dict__")
                    else str(_result_obj.scenario_summary)
                )
            if enhanced:
                report["enhanced"] = enhanced

        if format_type == "json":
            return json_response(report)

        elif format_type == "html" or format_type == "full_html":
            # Generate HTML report
            verdict = report["summary"]["verdict"]
            verdict_color = (
                "#22c55e"
                if verdict in ["APPROVED", "PASS"]
                else "#ef4444"
                if verdict in ["REJECTED", "FAIL"]
                else "#eab308"
            )

            html_parts = [
                f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gauntlet Report - {gauntlet_id[:12]}</title>
    <style>
        :root {{ --bg: #0a0a0a; --surface: #1a1a1a; --border: #333; --text: #e0e0e0; --muted: #888; --green: #22c55e; --red: #ef4444; --yellow: #eab308; --cyan: #06b6d4; }}
        body {{ font-family: ui-monospace, monospace; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; line-height: 1.6; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .header {{ border-bottom: 2px solid {verdict_color}; padding-bottom: 1rem; margin-bottom: 2rem; }}
        .verdict {{ font-size: 2rem; color: {verdict_color}; margin: 0; }}
        .id {{ color: var(--muted); font-size: 0.75rem; }}
        .card {{ background: var(--surface); border: 1px solid var(--border); padding: 1rem; margin-bottom: 1rem; border-radius: 4px; }}
        .card-title {{ color: var(--cyan); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem; }}
        .stat {{ display: inline-block; margin-right: 2rem; }}
        .stat-value {{ font-size: 1.5rem; }}
        .stat-label {{ color: var(--muted); font-size: 0.75rem; }}
        .finding {{ border-left: 3px solid; padding: 0.5rem 1rem; margin: 0.5rem 0; }}
        .finding.critical {{ border-color: var(--red); background: rgba(239,68,68,0.1); }}
        .finding.high {{ border-color: #f97316; background: rgba(249,115,22,0.1); }}
        .finding.medium {{ border-color: var(--yellow); background: rgba(234,179,8,0.1); }}
        .finding.low {{ border-color: var(--cyan); background: rgba(6,182,212,0.1); }}
        .badge {{ display: inline-block; padding: 0.25rem 0.5rem; font-size: 0.7rem; border-radius: 2px; }}
        .badge.critical {{ background: var(--red); color: white; }}
        .badge.high {{ background: #f97316; color: white; }}
        .badge.medium {{ background: var(--yellow); color: black; }}
        .badge.low {{ background: var(--cyan); color: black; }}
        .heatmap {{ display: grid; gap: 2px; margin-top: 1rem; }}
        .heatmap-cell {{ padding: 0.5rem; text-align: center; font-size: 0.75rem; }}
        .footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.75rem; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="verdict">{verdict}</h1>
            <div class="id">Gauntlet ID: {gauntlet_id}</div>
            <div class="id">Generated: {report['generated_at']}</div>
        </div>

        <div class="card">
            <div class="card-title">Summary</div>
            <div class="stat">
                <div class="stat-value" style="color: {verdict_color}">{(report['summary']['confidence'] * 100):.0f}%</div>
                <div class="stat-label">Confidence</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color: var(--cyan)">{(report['summary']['robustness_score'] * 100):.0f}%</div>
                <div class="stat-label">Robustness</div>
            </div>
            <div class="stat">
                <div class="stat-value">{report['findings_summary']['total']}</div>
                <div class="stat-label">Total Findings</div>
            </div>
        </div>

        <div class="card">
            <div class="card-title">Findings Breakdown</div>
            <div class="stat">
                <div class="stat-value" style="color: var(--red)">{report['findings_summary']['critical']}</div>
                <div class="stat-label">Critical</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color: #f97316">{report['findings_summary']['high']}</div>
                <div class="stat-label">High</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color: var(--yellow)">{report['findings_summary']['medium']}</div>
                <div class="stat-label">Medium</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color: var(--cyan)">{report['findings_summary']['low']}</div>
                <div class="stat-label">Low</div>
            </div>
        </div>
""",
            ]

            if include_findings and report.get("findings"):
                html_parts.append(
                    """
        <div class="card">
            <div class="card-title">Findings Detail</div>
"""
                )
                for finding in report["findings"][:20]:  # Limit to 20 for HTML
                    severity = finding.get("severity_level", "medium").lower()
                    html_parts.append(
                        f"""
            <div class="finding {severity}">
                <span class="badge {severity}">{severity.upper()}</span>
                <strong>{finding.get('title', 'Unknown')}</strong>
                <div style="color: var(--muted); font-size: 0.85rem;">{finding.get('description', '')[:200]}</div>
            </div>
"""
                    )
                html_parts.append("        </div>")

            html_parts.append(
                f"""
        <div class="footer">
            Report generated by Aragora Gauntlet | {report['generated_at']}
        </div>
    </div>
</body>
</html>
"""
            )
            html_content = "".join(html_parts)

            return HandlerResult(
                status_code=200,
                content_type="text/html",
                body=html_content.encode("utf-8"),
            )

        else:
            return error_response(f"Unsupported format: {format_type}", 400)
