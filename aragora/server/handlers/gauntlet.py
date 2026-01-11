"""
Gauntlet API endpoint handlers.

Endpoints:
- GET /api/gauntlet/templates - List available Gauntlet templates
- GET /api/gauntlet/templates/{id} - Get template details
- POST /api/gauntlet/run - Run a Gauntlet validation
- GET /api/gauntlet/results/{id} - Get Gauntlet result
- GET /api/gauntlet/results/{id}/receipt - Get Decision Receipt
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    handle_errors,
    validate_path_segment,
    get_clamped_int_param,
    get_bounded_string_param,
)

logger = logging.getLogger(__name__)

# Gauntlet module imports are deferred to avoid circular imports
GAUNTLET_AVAILABLE = False
_gauntlet_initialized = False


def _init_gauntlet():
    """Initialize gauntlet imports (deferred to avoid circular imports)."""
    global GAUNTLET_AVAILABLE, _gauntlet_initialized
    global GauntletOrchestrator, GauntletConfig, GauntletResult
    global InputType, Verdict, DecisionReceipt
    global list_templates, get_template, ReceiptFormat

    if _gauntlet_initialized:
        return GAUNTLET_AVAILABLE

    _gauntlet_initialized = True

    try:
        from aragora.modes.gauntlet import (
            GauntletOrchestrator as _GauntletOrchestrator,
            GauntletConfig as _GauntletConfig,
            GauntletResult as _GauntletResult,
            InputType as _InputType,
            Verdict as _Verdict,
            QUICK_GAUNTLET,
            THOROUGH_GAUNTLET,
        )
        from aragora.export.decision_receipt import DecisionReceipt as _DecisionReceipt, generate_decision_receipt
        from enum import Enum

        # Assign to module-level names
        GauntletOrchestrator = _GauntletOrchestrator
        GauntletConfig = _GauntletConfig
        GauntletResult = _GauntletResult
        InputType = _InputType
        Verdict = _Verdict
        DecisionReceipt = _DecisionReceipt

        class ReceiptFormat(Enum):
            JSON = "json"
            MARKDOWN = "markdown"
            HTML = "html"

        # Simple template system using pre-configured profiles
        _TEMPLATES = {
            "quick": {"name": "Quick Validation", "config": QUICK_GAUNTLET, "description": "Fast stress-test for basic issues"},
            "thorough": {"name": "Thorough Validation", "config": THOROUGH_GAUNTLET, "description": "Comprehensive adversarial analysis"},
            "security": {"name": "Security Assessment", "config": _GauntletConfig(input_type=_InputType.CODE, enable_verification=True), "description": "Security-focused validation"},
        }

        def list_templates():
            return [{"id": k, "name": v["name"], "description": v["description"]} for k, v in _TEMPLATES.items()]

        def get_template(template_id: str):
            if template_id not in _TEMPLATES:
                raise ValueError(f"Unknown template: {template_id}")
            return _TEMPLATES[template_id]["config"]

        # Make available at module level
        globals()["list_templates"] = list_templates
        globals()["get_template"] = get_template
        globals()["ReceiptFormat"] = ReceiptFormat

        GAUNTLET_AVAILABLE = True
        logger.info("Gauntlet module initialized successfully")

    except ImportError as e:
        GAUNTLET_AVAILABLE = False
        logger.debug(f"Gauntlet module not available: {e}")

    return GAUNTLET_AVAILABLE


class GauntletHandler(BaseHandler):
    """Handler for Gauntlet adversarial validation endpoints."""

    ROUTES = [
        "/api/gauntlet/templates",
        "/api/gauntlet/run",
        "/api/gauntlet/results",
    ]

    def __init__(self, ctx: dict = None):
        """Initialize with context."""
        super().__init__(ctx)
        self._orchestrator = None  # GauntletOrchestrator, lazily loaded
        self._results_cache: dict = {}  # str -> GauntletResult

    @property
    def orchestrator(self) -> Optional["GauntletOrchestrator"]:
        """Lazy-load Gauntlet orchestrator."""
        if self._orchestrator is None and _init_gauntlet():
            nomic_dir = self.ctx.get("nomic_dir")
            self._orchestrator = GauntletOrchestrator(nomic_dir=nomic_dir)
        return self._orchestrator

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/gauntlet")

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Route GET requests."""
        if not _init_gauntlet():
            return error_response("Gauntlet module not available", 503)

        # GET /api/gauntlet/templates
        if path == "/api/gauntlet/templates" or path == "/api/gauntlet/templates/":
            return self._list_templates(query_params)

        # Parse path segments
        parts = path.rstrip("/").split("/")

        # GET /api/gauntlet/templates/{id}
        if len(parts) == 5 and parts[3] == "templates":
            template_id = parts[4]
            valid, err = validate_path_segment(template_id, "template_id")
            if not valid:
                return error_response(err or "Invalid template ID", 400)
            return self._get_template(template_id)

        # GET /api/gauntlet/results/{id}
        if len(parts) == 5 and parts[3] == "results":
            result_id = parts[4]
            valid, err = validate_path_segment(result_id, "result_id")
            if not valid:
                return error_response(err or "Invalid result ID", 400)
            return self._get_result(result_id)

        # GET /api/gauntlet/results/{id}/receipt
        if len(parts) == 6 and parts[3] == "results" and parts[5] == "receipt":
            result_id = parts[4]
            valid, err = validate_path_segment(result_id, "result_id")
            if not valid:
                return error_response(err or "Invalid result ID", 400)
            format_param = get_bounded_string_param(query_params, "format", "json", max_length=20)
            return self._get_receipt(result_id, format_param)

        return None

    def handle_post(self, path: str, body: dict, handler=None) -> Optional[HandlerResult]:
        """Route POST requests."""
        if not _init_gauntlet():
            return error_response("Gauntlet module not available", 503)

        # POST /api/gauntlet/run
        if path == "/api/gauntlet/run" or path == "/api/gauntlet/run/":
            return self._run_gauntlet(body)

        return None

    @handle_errors("list Gauntlet templates")
    def _list_templates(self, query_params: dict) -> HandlerResult:
        """List all available Gauntlet templates."""
        templates = list_templates()

        # Optional filtering
        domain = get_bounded_string_param(query_params, "domain", "", max_length=50)
        if domain:
            templates = [t for t in templates if t["domain"] == domain]

        tag = get_bounded_string_param(query_params, "tag", "", max_length=50)
        if tag:
            templates = [t for t in templates if tag in t.get("tags", [])]

        return json_response({
            "templates": templates,
            "count": len(templates),
        })

    @handle_errors("get Gauntlet template")
    def _get_template(self, template_id: str) -> HandlerResult:
        """Get details for a specific template."""
        import dataclasses
        try:
            config = get_template(template_id)
            # Convert dataclass to dict, handling enums
            config_dict = dataclasses.asdict(config)
            # Convert enum values to strings
            for key, value in config_dict.items():
                if hasattr(value, 'value'):
                    config_dict[key] = value.value
                elif isinstance(value, list):
                    config_dict[key] = [
                        v.value if hasattr(v, 'value') else v
                        for v in value
                    ]
            return json_response({
                "id": template_id,
                "config": config_dict,
            })
        except ValueError as e:
            return error_response(str(e), 404)

    @handle_errors("run Gauntlet validation")
    def _run_gauntlet(self, body: dict) -> HandlerResult:
        """Run a Gauntlet validation."""
        if not self.orchestrator:
            return error_response("Gauntlet not configured", 503)

        # Validate input
        input_text = body.get("input_text", "")
        if not input_text:
            return error_response("input_text is required", 400)

        if len(input_text) > 50000:  # 50KB limit
            return error_response("input_text too large (max 50KB)", 400)

        # Get template or custom config
        template_id = body.get("template")
        config = None

        if template_id:
            try:
                config = get_template(template_id)
            except ValueError as e:
                return error_response(str(e), 400)

        # Override with custom config if provided
        if body.get("config"):
            try:
                if config:
                    # Merge custom config with template
                    config_dict = config.to_dict()
                    config_dict.update(body["config"])
                    config = GauntletConfig.from_dict(config_dict)
                else:
                    config = GauntletConfig.from_dict(body["config"])
            except (KeyError, ValueError) as e:
                return error_response(f"Invalid config: {e}", 400)

        # Run Gauntlet (async)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                self.orchestrator.run(
                    input_text=input_text,
                    config=config,
                )
            )

            # Cache result for retrieval
            self._results_cache[result.id] = result

            # Return summary (full result available via GET /results/{id})
            return json_response({
                "id": result.id,
                "passed": result.passed,
                "confidence": result.confidence,
                "verdict_summary": result.verdict_summary,
                "severity_counts": result.severity_counts,
                "robustness_score": result.robustness_score,
                "risk_score": result.risk_score,
                "duration_ms": result.total_duration_ms,
                "findings_count": len(result.findings),
                "critical_findings": [
                    {
                        "id": f.id,
                        "title": f.title,
                        "description": f.description[:200],
                    }
                    for f in result.critical_findings
                ],
            }, status=201)

        except asyncio.TimeoutError:
            return error_response("Gauntlet validation timed out", 504)

    @handle_errors("get Gauntlet result")
    def _get_result(self, result_id: str) -> HandlerResult:
        """Get a Gauntlet result by ID."""
        result = self._results_cache.get(result_id)
        if not result:
            return error_response("Result not found", 404)

        return json_response(result.to_dict())

    @handle_errors("get Decision Receipt")
    def _get_receipt(self, result_id: str, format_str: str) -> HandlerResult:
        """Get Decision Receipt for a Gauntlet result."""
        result = self._results_cache.get(result_id)
        if not result:
            return error_response("Result not found", 404)

        receipt = DecisionReceipt.from_gauntlet_result(result)

        # Return based on format
        if format_str == "markdown" or format_str == "md":
            return HandlerResult(
                body=receipt.to_markdown().encode(),
                status=200,
                content_type="text/markdown",
            )
        elif format_str == "html":
            return HandlerResult(
                body=receipt.to_html().encode(),
                status=200,
                content_type="text/html",
            )
        else:  # json
            return json_response(receipt.to_dict())
