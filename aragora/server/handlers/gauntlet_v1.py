"""
Gauntlet API v1 - Versioned, OpenAPI-compliant endpoints.

Provides stable, documented API endpoints for Gauntlet functionality:
- GET /api/v1/gauntlet/schema/{type} - Get JSON schemas
- GET /api/v1/gauntlet/templates - List audit templates
- GET /api/v1/gauntlet/templates/{id} - Get specific template
- POST /api/v1/gauntlet/{id}/export - Export receipt in various formats
- GET /api/v1/gauntlet/{id}/heatmap/export - Export heatmap

All endpoints follow RFC 7807 for error responses.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    get_string_param,
    json_response,
)

logger = logging.getLogger(__name__)


# RFC 7807 Problem Types
PROBLEM_TYPE_BASE = "https://aragora.ai/problems"
PROBLEM_NOT_FOUND = f"{PROBLEM_TYPE_BASE}/not-found"
PROBLEM_VALIDATION = f"{PROBLEM_TYPE_BASE}/validation-error"
PROBLEM_INTERNAL = f"{PROBLEM_TYPE_BASE}/internal-error"


def rfc7807_error(
    status: int,
    title: str,
    detail: str,
    problem_type: str = PROBLEM_INTERNAL,
    instance: Optional[str] = None,
    **extra: Any,
) -> HandlerResult:
    """Create an RFC 7807 Problem Details response."""
    problem = {
        "type": problem_type,
        "title": title,
        "status": status,
        "detail": detail,
    }
    if instance:
        problem["instance"] = instance
    problem.update(extra)
    return HandlerResult(  # type: ignore[call-arg]
        status=status,
        body=json.dumps(problem).encode("utf-8"),
        content_type="application/problem+json",
    )


class GauntletSchemaHandler(BaseHandler):
    """
    GET /api/v1/gauntlet/schema/{type}

    Returns JSON Schema for the specified type.

    Path Parameters:
        type: Schema type (decision-receipt, risk-heatmap, problem-detail)

    Returns:
        JSON Schema document
    """

    def get_path_pattern(self) -> str:
        return r"/api/v1/gauntlet/schema/(?P<schema_type>[a-z-]+)"

    async def handle(  # type: ignore[override]
        self,
        body: Optional[Dict[str, Any]],
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> HandlerResult:
        try:
            schema_type = path_params.get("schema_type") if path_params else None

            if not schema_type:
                return rfc7807_error(
                    status=400,
                    title="Missing Schema Type",
                    detail="Schema type is required in the path",
                    problem_type=PROBLEM_VALIDATION,
                )

            from aragora.gauntlet.api import get_all_schemas

            schemas = get_all_schemas()

            if schema_type not in schemas:
                available = list(schemas.keys())
                return rfc7807_error(
                    status=404,
                    title="Schema Not Found",
                    detail=f"Schema type '{schema_type}' not found. Available: {available}",
                    problem_type=PROBLEM_NOT_FOUND,
                    available_schemas=available,
                )

            schema = schemas[schema_type]
            return json_response(schema)

        except Exception as e:
            logger.exception(f"Error getting schema: {e}")
            return rfc7807_error(
                status=500,
                title="Internal Server Error",
                detail=str(e),
            )


class GauntletAllSchemasHandler(BaseHandler):
    """
    GET /api/v1/gauntlet/schemas

    Returns all available JSON Schemas.

    Returns:
        Object mapping schema names to their definitions
    """

    def get_path_pattern(self) -> str:
        return r"/api/v1/gauntlet/schemas"

    async def handle(  # type: ignore[override]
        self,
        body: Optional[Dict[str, Any]],
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> HandlerResult:
        try:
            from aragora.gauntlet.api import get_all_schemas, SCHEMA_VERSION

            schemas = get_all_schemas()

            response = {
                "version": SCHEMA_VERSION,
                "schemas": schemas,
                "count": len(schemas),
            }

            return json_response(response)

        except Exception as e:
            logger.exception(f"Error getting schemas: {e}")
            return rfc7807_error(
                status=500,
                title="Internal Server Error",
                detail=str(e),
            )


class GauntletTemplatesListHandler(BaseHandler):
    """
    GET /api/v1/gauntlet/templates

    List all available audit templates.

    Query Parameters:
        category: Filter by category (compliance, security, legal, financial, operational)

    Returns:
        List of available templates with metadata
    """

    def get_path_pattern(self) -> str:
        return r"/api/v1/gauntlet/templates"

    async def handle(  # type: ignore[override]
        self,
        body: Optional[Dict[str, Any]],
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> HandlerResult:
        try:
            from aragora.gauntlet.api import list_templates, TemplateCategory

            category_str = get_string_param(query_params, "category")
            category = None

            if category_str:
                try:
                    category = TemplateCategory(category_str.lower())
                except ValueError:
                    valid_categories = [c.value for c in TemplateCategory]
                    return rfc7807_error(
                        status=400,
                        title="Invalid Category",
                        detail=f"Category must be one of: {valid_categories}",
                        problem_type=PROBLEM_VALIDATION,
                        valid_categories=valid_categories,
                    )

            templates = list_templates(category)

            response = {
                "templates": [
                    {
                        "id": t.id,
                        "name": t.name,
                        "category": t.category.value,
                        "description": t.description,
                        "version": t.version,
                        "regulations": t.regulations,
                        "supported_formats": [f.value for f in t.supported_formats],
                    }
                    for t in templates
                ],
                "count": len(templates),
            }

            return json_response(response)

        except Exception as e:
            logger.exception(f"Error listing templates: {e}")
            return rfc7807_error(
                status=500,
                title="Internal Server Error",
                detail=str(e),
            )


class GauntletTemplateHandler(BaseHandler):
    """
    GET /api/v1/gauntlet/templates/{id}

    Get a specific audit template by ID.

    Path Parameters:
        id: Template identifier

    Returns:
        Full template definition
    """

    def get_path_pattern(self) -> str:
        return r"/api/v1/gauntlet/templates/(?P<template_id>[a-z0-9-]+)"

    async def handle(  # type: ignore[override]
        self,
        body: Optional[Dict[str, Any]],
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> HandlerResult:
        try:
            template_id = path_params.get("template_id") if path_params else None

            if not template_id:
                return rfc7807_error(
                    status=400,
                    title="Missing Template ID",
                    detail="Template ID is required in the path",
                    problem_type=PROBLEM_VALIDATION,
                )

            from aragora.gauntlet.api import get_template, list_templates

            template = get_template(template_id)

            if not template:
                available = [t.id for t in list_templates()]
                return rfc7807_error(
                    status=404,
                    title="Template Not Found",
                    detail=f"Template '{template_id}' not found. Available: {available}",
                    problem_type=PROBLEM_NOT_FOUND,
                    available_templates=available,
                )

            return json_response(template.to_dict())

        except Exception as e:
            logger.exception(f"Error getting template: {e}")
            return rfc7807_error(
                status=500,
                title="Internal Server Error",
                detail=str(e),
            )


class GauntletReceiptExportHandler(BaseHandler):
    """
    POST /api/v1/gauntlet/{id}/export

    Export a decision receipt in the specified format.

    Path Parameters:
        id: Gauntlet run ID

    Body:
        format: Export format (json, markdown, html, csv, sarif)
        template_id: Optional audit template to apply
        options: Export options (include_provenance, include_config, etc.)

    Returns:
        Exported content with appropriate Content-Type
    """

    def get_path_pattern(self) -> str:
        return r"/api/v1/gauntlet/(?P<gauntlet_id>[a-zA-Z0-9-]+)/export"

    def get_methods(self) -> list[str]:
        return ["POST"]

    async def handle(  # type: ignore[override]
        self,
        body: Optional[Dict[str, Any]],
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> HandlerResult:
        try:
            gauntlet_id = path_params.get("gauntlet_id") if path_params else None

            if not gauntlet_id:
                return rfc7807_error(
                    status=400,
                    title="Missing Gauntlet ID",
                    detail="Gauntlet ID is required in the path",
                    problem_type=PROBLEM_VALIDATION,
                )

            # Parse request body
            body = body or {}
            format_str = body.get("format", "json").lower()
            template_id = body.get("template_id")
            options_dict = body.get("options", {})

            # Get the receipt
            from aragora.server.handlers.gauntlet import _gauntlet_runs, _get_storage

            # Try in-memory first
            run = _gauntlet_runs.get(gauntlet_id)

            # Try persistent storage
            if not run:
                storage = _get_storage()
                run = storage.get_result(gauntlet_id)  # type: ignore[attr-defined]

            if not run:
                return rfc7807_error(
                    status=404,
                    title="Gauntlet Not Found",
                    detail=f"Gauntlet run '{gauntlet_id}' not found",
                    problem_type=PROBLEM_NOT_FOUND,
                    instance=f"/api/v1/gauntlet/{gauntlet_id}",
                )

            # Check status
            status = run.get("status")
            if status != "completed":
                return rfc7807_error(
                    status=400,
                    title="Gauntlet Not Complete",
                    detail=f"Gauntlet run is '{status}', export requires 'completed' status",
                    problem_type=PROBLEM_VALIDATION,
                    current_status=status,
                )

            # Get or create receipt
            receipt = run.get("receipt")
            if not receipt:
                # Try to create receipt from result
                result = run.get("result")
                if result:
                    if hasattr(result, "to_receipt"):
                        receipt = result.to_receipt()
                    else:
                        # Result is a dict, need to reconstruct
                        from aragora.gauntlet.receipt import DecisionReceipt as DR

                        receipt = DR(
                            receipt_id=f"receipt-{datetime.now().strftime('%Y%m%d%H%M%S')}-{gauntlet_id[-8:]}",
                            gauntlet_id=gauntlet_id,
                            timestamp=result.get("completed_at", datetime.now().isoformat()),
                            input_summary=result.get("input_summary", ""),
                            input_hash=result.get("input_hash", ""),
                            risk_summary=result.get("risk_summary", {}),
                            attacks_attempted=result.get("attack_summary", {}).get(
                                "total_attacks", 0
                            ),
                            attacks_successful=result.get("attack_summary", {}).get(
                                "successful_attacks", 0
                            ),
                            probes_run=result.get("probe_summary", {}).get("probes_run", 0),
                            vulnerabilities_found=result.get("risk_summary", {}).get("total", 0),
                            verdict=result.get("verdict", "FAIL"),
                            confidence=result.get("confidence", 0.0),
                            robustness_score=result.get("robustness_score", 0.0),
                        )

            if not receipt:
                return rfc7807_error(
                    status=500,
                    title="Receipt Generation Failed",
                    detail="Could not generate receipt from gauntlet result",
                )

            # Apply template if specified
            if template_id:
                from aragora.gauntlet.api import get_template, TemplateFormat

                template = get_template(template_id)
                if not template:
                    return rfc7807_error(
                        status=404,
                        title="Template Not Found",
                        detail=f"Template '{template_id}' not found",
                        problem_type=PROBLEM_NOT_FOUND,
                    )

                # Map format to template format
                format_map = {
                    "markdown": TemplateFormat.MARKDOWN,
                    "html": TemplateFormat.HTML,
                    "json": TemplateFormat.JSON,
                    "text": TemplateFormat.TEXT,
                }
                template_format = format_map.get(format_str, TemplateFormat.MARKDOWN)

                # Render with template
                content = template.render(receipt, template_format)
                content_type = {
                    "markdown": "text/markdown",
                    "html": "text/html",
                    "json": "application/json",
                    "text": "text/plain",
                }.get(format_str, "text/plain")

                return HandlerResult(  # type: ignore[call-arg,arg-type]
                    status=200,
                    body=content,  # type: ignore[arg-type]
                    headers={"Content-Type": content_type},
                )

            # Export without template
            from aragora.gauntlet.api import (
                ReceiptExportFormat,
                ExportOptions,
                export_receipt,
            )

            # Map format string to enum
            format_map = {
                "json": ReceiptExportFormat.JSON,  # type: ignore[dict-item]
                "markdown": ReceiptExportFormat.MARKDOWN,  # type: ignore[dict-item]
                "html": ReceiptExportFormat.HTML,  # type: ignore[dict-item]
                "csv": ReceiptExportFormat.CSV,  # type: ignore[dict-item]
                "sarif": ReceiptExportFormat.SARIF,  # type: ignore[dict-item]
            }

            if format_str not in format_map:
                valid_formats = list(format_map.keys())
                return rfc7807_error(
                    status=400,
                    title="Invalid Format",
                    detail=f"Format must be one of: {valid_formats}",
                    problem_type=PROBLEM_VALIDATION,
                    valid_formats=valid_formats,
                )

            export_format = format_map[format_str]

            # Create export options
            options = ExportOptions(
                include_provenance=options_dict.get("include_provenance", True),
                include_config=options_dict.get("include_config", False),
                max_vulnerabilities=options_dict.get("max_vulnerabilities", 100),
                validate_schema=options_dict.get("validate_schema", False),
            )

            content = export_receipt(receipt, export_format, options)  # type: ignore[arg-type]

            # Set appropriate content type
            content_type = {
                "json": "application/json",
                "markdown": "text/markdown",
                "html": "text/html",
                "csv": "text/csv",
                "sarif": "application/sarif+json",
            }.get(format_str, "application/json")

            return HandlerResult(  # type: ignore[call-arg,arg-type]
                status=200,
                body=content,  # type: ignore[arg-type]
                headers={"Content-Type": content_type},
            )

        except Exception as e:
            logger.exception(f"Error exporting receipt: {e}")
            return rfc7807_error(
                status=500,
                title="Export Failed",
                detail=str(e),
            )


class GauntletHeatmapExportHandler(BaseHandler):
    """
    GET /api/v1/gauntlet/{id}/heatmap/export

    Export a risk heatmap in the specified format.

    Path Parameters:
        id: Gauntlet run ID

    Query Parameters:
        format: Export format (json, csv, svg, ascii, html)

    Returns:
        Exported heatmap content
    """

    def get_path_pattern(self) -> str:
        return r"/api/v1/gauntlet/(?P<gauntlet_id>[a-zA-Z0-9-]+)/heatmap/export"

    async def handle(  # type: ignore[override]
        self,
        body: Optional[Dict[str, Any]],
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> HandlerResult:
        try:
            gauntlet_id = path_params.get("gauntlet_id") if path_params else None

            if not gauntlet_id:
                return rfc7807_error(
                    status=400,
                    title="Missing Gauntlet ID",
                    detail="Gauntlet ID is required in the path",
                    problem_type=PROBLEM_VALIDATION,
                )

            format_str = get_string_param(query_params, "format", "json").lower()

            # Get the gauntlet run
            from aragora.server.handlers.gauntlet import _gauntlet_runs, _get_storage

            run = _gauntlet_runs.get(gauntlet_id)
            if not run:
                storage = _get_storage()
                run = storage.get_result(gauntlet_id)  # type: ignore[attr-defined]

            if not run:
                return rfc7807_error(
                    status=404,
                    title="Gauntlet Not Found",
                    detail=f"Gauntlet run '{gauntlet_id}' not found",
                    problem_type=PROBLEM_NOT_FOUND,
                )

            # Get heatmap
            heatmap = run.get("heatmap")
            if not heatmap:
                result = run.get("result")
                if result and hasattr(result, "to_heatmap"):
                    heatmap = result.to_heatmap()

            if not heatmap:
                return rfc7807_error(
                    status=404,
                    title="Heatmap Not Available",
                    detail="No heatmap data available for this gauntlet run",
                    problem_type=PROBLEM_NOT_FOUND,
                )

            # Export
            from aragora.gauntlet.api import HeatmapExportFormat, export_heatmap

            format_map = {
                "json": HeatmapExportFormat.JSON,
                "csv": HeatmapExportFormat.CSV,
                "svg": HeatmapExportFormat.SVG,
                "ascii": HeatmapExportFormat.ASCII,
                "html": HeatmapExportFormat.HTML,
            }

            if format_str not in format_map:
                valid_formats = list(format_map.keys())
                return rfc7807_error(
                    status=400,
                    title="Invalid Format",
                    detail=f"Format must be one of: {valid_formats}",
                    problem_type=PROBLEM_VALIDATION,
                    valid_formats=valid_formats,
                )

            export_format = format_map[format_str]
            content = export_heatmap(heatmap, export_format)

            content_type = {
                "json": "application/json",
                "csv": "text/csv",
                "svg": "image/svg+xml",
                "ascii": "text/plain",
                "html": "text/html",
            }.get(format_str, "application/json")

            return HandlerResult(  # type: ignore[call-arg,arg-type]
                status=200,
                body=content,  # type: ignore[arg-type]
                headers={"Content-Type": content_type},
            )

        except Exception as e:
            logger.exception(f"Error exporting heatmap: {e}")
            return rfc7807_error(
                status=500,
                title="Export Failed",
                detail=str(e),
            )


class GauntletValidateReceiptHandler(BaseHandler):
    """
    POST /api/v1/gauntlet/validate/receipt

    Validate a decision receipt against the JSON schema.

    Body:
        Receipt JSON to validate

    Returns:
        Validation result with any errors
    """

    def get_path_pattern(self) -> str:
        return r"/api/v1/gauntlet/validate/receipt"

    def get_methods(self) -> list[str]:
        return ["POST"]

    async def handle(  # type: ignore[override]
        self,
        body: Optional[Dict[str, Any]],
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> HandlerResult:
        try:
            if not body:
                return rfc7807_error(
                    status=400,
                    title="Missing Body",
                    detail="Request body with receipt data is required",
                    problem_type=PROBLEM_VALIDATION,
                )

            from aragora.gauntlet.api import validate_receipt

            is_valid, errors = validate_receipt(body)

            response = {
                "valid": is_valid,
                "errors": errors if errors else [],
                "error_count": len(errors),
            }

            return json_response(response)

        except Exception as e:
            logger.exception(f"Error validating receipt: {e}")
            return rfc7807_error(
                status=500,
                title="Validation Failed",
                detail=str(e),
            )


# Handler classes for registration
GAUNTLET_V1_HANDLERS = [
    GauntletSchemaHandler,
    GauntletAllSchemasHandler,
    GauntletTemplatesListHandler,
    GauntletTemplateHandler,
    GauntletReceiptExportHandler,
    GauntletHeatmapExportHandler,
    GauntletValidateReceiptHandler,
]


def register_gauntlet_v1_handlers(router: Any, server_context: Any = None) -> None:
    """Register all v1 Gauntlet handlers with a router."""
    ctx = server_context or {}
    for handler_cls in GAUNTLET_V1_HANDLERS:
        router.add_handler(handler_cls(ctx))  # type: ignore[arg-type]
