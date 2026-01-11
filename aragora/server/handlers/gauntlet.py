"""
Gauntlet endpoint handlers for adversarial stress-testing.

Endpoints:
- POST /api/gauntlet/run - Start a gauntlet stress-test
- GET /api/gauntlet/{id} - Get gauntlet status/results
- GET /api/gauntlet/{id}/receipt - Get decision receipt
- GET /api/gauntlet/{id}/heatmap - Get risk heatmap
- GET /api/gauntlet/personas - List available personas
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    handle_errors,
    get_string_param,
    get_bool_param,
    safe_json_parse,
)
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


# In-memory storage for gauntlet runs (in production, use database)
_gauntlet_runs: dict[str, dict[str, Any]] = {}


class GauntletHandler(BaseHandler):
    """Handler for gauntlet stress-testing endpoints."""

    ROUTES = [
        "/api/gauntlet/run",
        "/api/gauntlet/personas",
        "/api/gauntlet/*/receipt",
        "/api/gauntlet/*/heatmap",
        "/api/gauntlet/*",
    ]

    # All gauntlet endpoints require authentication
    AUTH_REQUIRED_ENDPOINTS = [
        "/api/gauntlet/run",
        "/api/gauntlet/",
    ]

    def can_handle(self, path: str, method: str) -> bool:
        """Check if this handler can handle the request."""
        if path == "/api/gauntlet/run" and method == "POST":
            return True
        if path == "/api/gauntlet/personas" and method == "GET":
            return True
        if path.startswith("/api/gauntlet/") and method == "GET":
            return True
        return False

    @handle_errors
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

        # GET /api/gauntlet/{id}/receipt
        if path.endswith("/receipt"):
            gauntlet_id = path.split("/")[-2]
            return await self._get_receipt(gauntlet_id, query_params)

        # GET /api/gauntlet/{id}/heatmap
        if path.endswith("/heatmap"):
            gauntlet_id = path.split("/")[-2]
            return await self._get_heatmap(gauntlet_id, query_params)

        # GET /api/gauntlet/{id}
        if path.startswith("/api/gauntlet/"):
            gauntlet_id = path.split("/")[-1]
            if gauntlet_id and gauntlet_id not in ("run", "personas"):
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
        # Parse request body
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length) if content_length > 0 else b"{}"
            data = json.loads(body.decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

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

        # Store initial state
        _gauntlet_runs[gauntlet_id] = {
            "gauntlet_id": gauntlet_id,
            "status": "pending",
            "input_type": input_type,
            "input_summary": input_content[:200] + "..." if len(input_content) > 200 else input_content,
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
            from aragora.modes.gauntlet import (
                GauntletOrchestrator,
                GauntletConfig,
                InputType,
            )
            from aragora.agents.base import create_agent

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
                except Exception as e:
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
            config = GauntletConfig(
                input_type=input_type_enum,
                input_content=input_content,
                persona=persona,
                max_duration_seconds=300,  # 5 minute max for API
            )

            # Run gauntlet
            orchestrator = GauntletOrchestrator(agent_instances)
            result = await orchestrator.run(config)

            # Store result
            _gauntlet_runs[gauntlet_id]["status"] = "completed"
            _gauntlet_runs[gauntlet_id]["completed_at"] = datetime.now().isoformat()
            _gauntlet_runs[gauntlet_id]["result"] = {
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

        except Exception as e:
            logger.error(f"Gauntlet {gauntlet_id} failed: {e}")
            _gauntlet_runs[gauntlet_id]["status"] = "failed"
            _gauntlet_runs[gauntlet_id]["error"] = str(e)

    async def _get_status(self, gauntlet_id: str) -> HandlerResult:
        """Get gauntlet run status."""
        if gauntlet_id not in _gauntlet_runs:
            return error_response(f"Gauntlet run not found: {gauntlet_id}", 404)

        run = _gauntlet_runs[gauntlet_id]
        return json_response(run)

    async def _get_receipt(self, gauntlet_id: str, query_params: dict) -> HandlerResult:
        """Get decision receipt for gauntlet run."""
        if gauntlet_id not in _gauntlet_runs:
            return error_response(f"Gauntlet run not found: {gauntlet_id}", 404)

        run = _gauntlet_runs[gauntlet_id]
        if run["status"] != "completed":
            return error_response("Gauntlet run not completed", 400)

        result = run["result"]

        from aragora.gauntlet.receipt import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id=f"receipt-{gauntlet_id[-12:]}",
            gauntlet_id=gauntlet_id,
            timestamp=run.get("completed_at", ""),
            input_summary=run["input_summary"],
            input_hash=gauntlet_id,  # Simplified
            risk_summary={
                "critical": result["critical_count"],
                "high": result["high_count"],
                "medium": result["medium_count"],
                "low": result["low_count"],
                "total": result["total_findings"],
            },
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=result["total_findings"],
            verdict=result["verdict"].upper(),
            confidence=result["confidence"],
            robustness_score=result["robustness_score"],
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
        if gauntlet_id not in _gauntlet_runs:
            return error_response(f"Gauntlet run not found: {gauntlet_id}", 404)

        run = _gauntlet_runs[gauntlet_id]
        if run["status"] != "completed":
            return error_response("Gauntlet run not completed", 400)

        result = run["result"]

        from aragora.gauntlet.heatmap import RiskHeatmap, HeatmapCell

        # Build heatmap from findings
        cells = []
        categories = set()
        severities = ["critical", "high", "medium", "low"]

        for finding in result.get("findings", []):
            category = finding.get("category", "unknown")
            categories.add(category)

        # Count by category and severity
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
            total_findings=result["total_findings"],
        )

        # Return format based on query param
        format_type = get_string_param(query_params, "format", "json")

        if format_type == "svg":
            return (heatmap.to_svg(), 200, {"Content-Type": "image/svg+xml"})
        elif format_type == "ascii":
            return (heatmap.to_ascii(), 200, {"Content-Type": "text/plain"})
        else:
            return json_response(heatmap.to_dict())
