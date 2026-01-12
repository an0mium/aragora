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
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    handle_errors,
    get_string_param,
    get_bool_param,
    get_int_param,
    safe_json_parse,
    require_quota,
)
from .utils.rate_limit import rate_limit
from aragora.server.validation.entities import validate_gauntlet_id

logger = logging.getLogger(__name__)


# In-memory storage for in-flight gauntlet runs (pending/running)
# Completed runs are persisted to GauntletStorage
_gauntlet_runs: dict[str, dict[str, Any]] = {}

# Memory management for gauntlet runs
MAX_GAUNTLET_RUNS_IN_MEMORY = 500
_GAUNTLET_COMPLETED_TTL = 3600  # Keep completed runs for 1 hour

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
    """Remove old completed runs from memory (persisted ones are in storage)."""
    global _gauntlet_runs
    if len(_gauntlet_runs) <= MAX_GAUNTLET_RUNS_IN_MEMORY:
        return

    now = time.time()
    # Find completed runs older than TTL
    to_remove = []
    for run_id, run in _gauntlet_runs.items():
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


class GauntletHandler(BaseHandler):
    """Handler for gauntlet stress-testing endpoints."""

    ROUTES = [
        "/api/gauntlet/run",
        "/api/gauntlet/personas",
        "/api/gauntlet/results",
        "/api/gauntlet/*/receipt",
        "/api/gauntlet/*/heatmap",
        "/api/gauntlet/*/compare/*",
        "/api/gauntlet/*",
    ]

    # All gauntlet endpoints require authentication
    AUTH_REQUIRED_ENDPOINTS = [
        "/api/gauntlet/run",
        "/api/gauntlet/",
    ]

    def __init__(self, server_context: dict):
        super().__init__(server_context)
        emitter = server_context.get("stream_emitter")
        if emitter and hasattr(emitter, "emit"):
            set_gauntlet_broadcast_fn(emitter.emit)

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the request."""
        # When called without method (e.g., from route index), just check path prefix
        if method == "GET" and path.startswith("/api/gauntlet/"):
            return True
        if path == "/api/gauntlet/run" and method == "POST":
            return True
        if path == "/api/gauntlet/personas" and method == "GET":
            return True
        if path == "/api/gauntlet/results" and method == "GET":
            return True
        if path.startswith("/api/gauntlet/") and method == "GET":
            return True
        if path.startswith("/api/gauntlet/") and method == "DELETE":
            return True
        return False

    @rate_limit(rpm=10)
    async def handle(self, path: str, method: str, handler: Any = None) -> Optional[HandlerResult]:
        """Route request to appropriate handler."""
        query_params = {}
        if handler:
            query_str = handler.path.split("?", 1)[1] if "?" in handler.path else ""
            from urllib.parse import parse_qs
            query_params = parse_qs(query_str)

        # POST /api/gauntlet/run
        if path == "/api/gauntlet/run" and method == "POST":
            return await self._start_gauntlet(handler)

        # GET /api/gauntlet/personas
        if path == "/api/gauntlet/personas":
            return self._list_personas()

        # GET /api/gauntlet/results - List with pagination
        if path == "/api/gauntlet/results":
            return self._list_results(query_params)

        # GET /api/gauntlet/{id}/receipt
        if path.endswith("/receipt"):
            gauntlet_id = path.split("/")[-2]
            is_valid, err = validate_gauntlet_id(gauntlet_id)
            if not is_valid:
                return error_response(err, 400)
            return await self._get_receipt(gauntlet_id, query_params)

        # GET /api/gauntlet/{id}/heatmap
        if path.endswith("/heatmap"):
            gauntlet_id = path.split("/")[-2]
            is_valid, err = validate_gauntlet_id(gauntlet_id)
            if not is_valid:
                return error_response(err, 400)
            return await self._get_heatmap(gauntlet_id, query_params)

        # GET /api/gauntlet/{id}/compare/{id2}
        if "/compare/" in path:
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
                return self._compare_results(gauntlet_id, compare_id, query_params)

        # DELETE /api/gauntlet/{id}
        if method == "DELETE" and path.startswith("/api/gauntlet/"):
            gauntlet_id = path.split("/")[-1]
            if gauntlet_id and gauntlet_id not in ("run", "personas", "results"):
                is_valid, err = validate_gauntlet_id(gauntlet_id)
                if not is_valid:
                    return error_response(err, 400)
                return self._delete_result(gauntlet_id, query_params)

        # GET /api/gauntlet/{id}
        if path.startswith("/api/gauntlet/"):
            gauntlet_id = path.split("/")[-1]
            if gauntlet_id and gauntlet_id not in ("run", "personas", "results"):
                is_valid, err = validate_gauntlet_id(gauntlet_id)
                if not is_valid:
                    return error_response(err, 400)
                return await self._get_status(gauntlet_id)

        return None

    def _list_personas(self) -> HandlerResult:
        """List available regulatory personas."""
        try:
            from aragora.gauntlet.personas import list_personas, get_persona

            personas_list = []
            for name in list_personas():
                persona = get_persona(name)
                personas_list.append({
                    "id": name,
                    "name": persona.name,
                    "description": persona.description,
                    "regulation": persona.regulation,
                    "attack_count": len(persona.attack_prompts),
                    "categories": list(set(a.category for a in persona.attack_prompts)),
                })

            return json_response({
                "personas": personas_list,
                "count": len(personas_list),
            })
        except ImportError:
            return json_response({
                "personas": [],
                "count": 0,
                "error": "Personas module not available",
            })

    async def _start_gauntlet(self, handler: Any) -> HandlerResult:
        """Start a new gauntlet stress-test."""
        # Check quota before proceeding
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = None
        if hasattr(handler, 'user_store'):
            user_store = handler.user_store
        elif hasattr(handler.__class__, 'user_store'):
            user_store = handler.__class__.user_store

        user_ctx = extract_user_from_request(handler, user_store) if user_store else None

        if user_ctx and user_ctx.is_authenticated and user_ctx.org_id:
            if user_store and hasattr(user_store, 'get_organization_by_id'):
                org = user_store.get_organization_by_id(user_ctx.org_id)
                if org:
                    if org.is_at_limit:
                        return json_response({
                            "error": "Monthly debate quota exceeded",
                            "code": "quota_exceeded",
                            "limit": org.limits.debates_per_month,
                            "used": org.debates_used_this_month,
                            "remaining": 0,
                            "tier": org.tier.value,
                            "upgrade_url": "/pricing",
                            "message": f"Your {org.tier.value} plan allows {org.limits.debates_per_month} debates per month. Gauntlet runs count as debates. Upgrade to increase your limit.",
                        }, status=429)

        # Parse request body (with Content-Length validation)
        data = self.read_json_body(handler)
        if data is None:
            return error_response("Invalid or too large request body", 400)

        # Extract parameters
        input_content = data.get("input_content", "")
        input_type = data.get("input_type", "spec")
        persona = data.get("persona")
        agents = data.get("agents", ["anthropic-api"])
        profile = data.get("profile", "default")

        if not input_content:
            return error_response("input_content is required", 400)

        # Generate gauntlet ID
        gauntlet_id = f"gauntlet-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        input_hash = hashlib.sha256(input_content.encode()).hexdigest()

        # Cleanup old completed runs before storing new one
        _cleanup_gauntlet_runs()

        # Store initial state
        _gauntlet_runs[gauntlet_id] = {
            "gauntlet_id": gauntlet_id,
            "status": "pending",
            "input_type": input_type,
            "input_summary": input_content[:200] + "..." if len(input_content) > 200 else input_content,
            "input_hash": input_hash,
            "persona": persona,
            "profile": profile,
            "created_at": datetime.now().isoformat(),
            "result": None,
        }

        # In a production system, this would be queued for async processing
        # For now, we'll run it synchronously in a background task
        asyncio.create_task(self._run_gauntlet_async(
            gauntlet_id, input_content, input_type, persona, agents, profile
        ))

        # Increment usage on successful gauntlet start
        if user_ctx and user_ctx.is_authenticated and user_ctx.org_id:
            if user_store and hasattr(user_store, 'increment_usage'):
                try:
                    user_store.increment_usage(user_ctx.org_id, 1)
                    logger.info(
                        f"Incremented gauntlet usage for org {user_ctx.org_id}"
                    )
                except Exception as ue:
                    logger.warning(f"Usage increment failed for org {user_ctx.org_id}: {ue}")

        return json_response({
            "gauntlet_id": gauntlet_id,
            "status": "pending",
            "message": "Gauntlet stress-test started",
        }, status=202)

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
            from aragora.gauntlet import (
                GauntletOrchestrator,
                OrchestratorConfig,
                GauntletProgress,
                InputType,
            )
            from aragora.agents.base import create_agent
            from aragora.server.stream.gauntlet_emitter import GauntletStreamEmitter

            # Create stream emitter if broadcasting is available
            emitter: Optional[GauntletStreamEmitter] = None
            if _gauntlet_broadcast_fn:
                emitter = GauntletStreamEmitter(
                    broadcast_fn=_gauntlet_broadcast_fn,
                    gauntlet_id=gauntlet_id,
                )

            # Update status
            _gauntlet_runs[gauntlet_id]["status"] = "running"

            # Create agents
            agent_instances = []
            for agent_type in agents:
                try:
                    agent = create_agent(
                        model_type=agent_type,
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

            # Create progress callback that also emits streaming events
            def on_progress(progress: GauntletProgress) -> None:
                """Handle progress updates with streaming."""
                if emitter:
                    emitter.emit_progress(
                        progress=progress.percent / 100.0,
                        phase=progress.phase,
                        message=progress.message,
                    )
                    if progress.current_task:
                        emitter.emit_phase(progress.current_task, progress.message)

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
            except (OSError, RuntimeError, ValueError) as storage_err:
                logger.warning(f"Failed to persist gauntlet {gauntlet_id}: {storage_err}")

            # Clean up in-memory storage after persisting (keep result_obj for receipt generation)
            # In-memory entry can be removed after a timeout in production

        except (OSError, RuntimeError, ValueError, ImportError, asyncio.CancelledError) as e:
            logger.error(f"Gauntlet {gauntlet_id} failed: {e}")
            _gauntlet_runs[gauntlet_id]["status"] = "failed"
            _gauntlet_runs[gauntlet_id]["error"] = str(e)

    async def _get_status(self, gauntlet_id: str) -> HandlerResult:
        """Get gauntlet run status."""
        # Check in-memory first (for pending/running)
        if gauntlet_id in _gauntlet_runs:
            run = _gauntlet_runs[gauntlet_id]
            safe_run = {k: v for k, v in run.items() if k != "result_obj"}
            return json_response(safe_run)

        # Check persistent storage (for completed runs)
        try:
            storage = _get_storage()
            stored = storage.get(gauntlet_id)
            if stored:
                return json_response({
                    "gauntlet_id": gauntlet_id,
                    "status": "completed",
                    "result": stored,
                })
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"Storage lookup failed for {gauntlet_id}: {e}")

        return error_response(f"Gauntlet run not found: {gauntlet_id}", 404)

    async def _get_receipt(self, gauntlet_id: str, query_params: dict) -> HandlerResult:
        """Get decision receipt for gauntlet run."""
        from aragora.gauntlet.receipt import DecisionReceipt

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
                input_hash=run.get("input_hash", gauntlet_id) if run else result.get("input_hash", gauntlet_id),
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
        format_type = get_string_param(query_params, "format", "json")

        if format_type == "html":
            return (receipt.to_html(), 200, {"Content-Type": "text/html"})
        elif format_type == "md":
            return (receipt.to_markdown(), 200, {"Content-Type": "text/markdown"})
        else:
            return json_response(receipt.to_dict())

    async def _get_heatmap(self, gauntlet_id: str, query_params: dict) -> HandlerResult:
        """Get risk heatmap for gauntlet run."""
        from aragora.gauntlet.heatmap import RiskHeatmap, HeatmapCell

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
            cells = []
            categories = set()
            severities = ["critical", "high", "medium", "low"]

            for finding in result.get("findings", []):
                category = finding.get("category", "unknown")
                categories.add(category)

            category_severity_counts: dict[tuple[str, str], int] = {}
            for finding in result.get("findings", []):
                category = finding.get("category", "unknown")
                severity = finding.get("severity_level", "medium").lower()
                key = (category, severity)
                category_severity_counts[key] = category_severity_counts.get(key, 0) + 1

            for category in sorted(categories):
                for severity in severities:
                    count = category_severity_counts.get((category, severity), 0)
                    cells.append(HeatmapCell(
                        category=category,
                        severity=severity,
                        count=count,
                    ))

            heatmap = RiskHeatmap(
                cells=cells,
                categories=sorted(list(categories)),
                severities=severities,
                total_findings=result.get("total_findings", 0),
            )

        # Return format based on query param
        format_type = get_string_param(query_params, "format", "json")

        if format_type == "svg":
            return (heatmap.to_svg(), 200, {"Content-Type": "image/svg+xml"})
        elif format_type == "ascii":
            return (heatmap.to_ascii(), 200, {"Content-Type": "text/plain"})
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

            return json_response({
                "results": [
                    {
                        "gauntlet_id": r.gauntlet_id,
                        "input_hash": r.input_hash,
                        "input_summary": r.input_summary[:100] + "..." if len(r.input_summary) > 100 else r.input_summary,
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
            })
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Failed to list results: {e}")
            return error_response(f"Failed to list results: {e}", 500)

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
            return error_response(f"Failed to compare results: {e}", 500)

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
            return error_response(f"Failed to delete result: {e}", 500)
