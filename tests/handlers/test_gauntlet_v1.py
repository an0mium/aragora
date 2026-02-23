"""Tests for Gauntlet v1 API handler (aragora/server/handlers/gauntlet_v1.py).

Covers all VersionedAPIHandler endpoints:
- GET  /api/v1/gauntlet/schema/{type}     - Get JSON schema by type
- GET  /api/v1/gauntlet/schemas            - Get all schemas
- GET  /api/v1/gauntlet/templates          - List audit templates
- GET  /api/v1/gauntlet/templates/{id}     - Get specific template
- POST /api/v1/gauntlet/{id}/export        - Export receipt
- GET  /api/v1/gauntlet/{id}/heatmap/export - Export heatmap
- POST /api/v1/gauntlet/validate/receipt   - Validate receipt
- RFC 7807 error responses
- RBAC permission checks
- Registration utility
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.gauntlet_v1 import (
    GAUNTLET_V1_HANDLERS,
    PROBLEM_INTERNAL,
    PROBLEM_NOT_FOUND,
    PROBLEM_TYPE_BASE,
    PROBLEM_VALIDATION,
    GauntletAllSchemasHandler,
    GauntletHeatmapExportHandler,
    GauntletReceiptExportHandler,
    GauntletSchemaHandler,
    GauntletSecureHandler,
    GauntletTemplateHandler,
    GauntletTemplatesListHandler,
    GauntletValidateReceiptHandler,
    register_gauntlet_v1_handlers,
    rfc7807_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract the JSON body from a HandlerResult."""
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return result.body
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


def _content_type(result: HandlerResult) -> str:
    """Extract content type from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.content_type
    return ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _bypass_rbac(request, monkeypatch):
    """Bypass RBAC for all tests except those marked no_auto_auth.

    Patches check_gauntlet_permission on the base class to always grant access.
    This is necessary because these handlers use a VersionedAPIHandler dispatch
    pattern (not BaseHandler) and need their own RBAC bypass.
    """
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    async def _grant(*args, **kwargs):
        return None  # None means permission granted

    monkeypatch.setattr(GauntletSecureHandler, "check_gauntlet_permission", _grant)
    yield


@pytest.fixture(autouse=True)
def _reset_gauntlet_runs():
    """Reset in-memory gauntlet runs between tests."""
    from aragora.server.handlers.gauntlet.storage import _gauntlet_runs

    original = dict(_gauntlet_runs)
    _gauntlet_runs.clear()
    yield
    _gauntlet_runs.clear()
    _gauntlet_runs.update(original)


@pytest.fixture(autouse=True)
def _ensure_templates_registered():
    """Ensure the gauntlet template registry is populated.

    Other tests in the suite may clear or corrupt the module-level
    ``_TEMPLATE_REGISTRY``.  This fixture snapshots it before the test
    and restores it afterwards so that ``list_templates()`` never
    returns an empty list unexpectedly.
    """
    from aragora.gauntlet.api.templates import (
        _TEMPLATE_REGISTRY,
        COMPLIANCE_TEMPLATE,
        SECURITY_TEMPLATE,
        LEGAL_TEMPLATE,
        FINANCIAL_TEMPLATE,
        OPERATIONAL_TEMPLATE,
    )

    # Ensure the built-in templates are present before the test runs
    defaults = {
        "compliance-standard": COMPLIANCE_TEMPLATE,
        "security-assessment": SECURITY_TEMPLATE,
        "legal-review": LEGAL_TEMPLATE,
        "financial-controls": FINANCIAL_TEMPLATE,
        "operational-review": OPERATIONAL_TEMPLATE,
    }
    saved = dict(_TEMPLATE_REGISTRY)
    for tid, tmpl in defaults.items():
        if tid not in _TEMPLATE_REGISTRY:
            _TEMPLATE_REGISTRY[tid] = tmpl
    yield
    # Restore original state
    _TEMPLATE_REGISTRY.clear()
    _TEMPLATE_REGISTRY.update(saved)


@pytest.fixture
def schema_handler():
    """Create a GauntletSchemaHandler."""
    return GauntletSchemaHandler(ctx={})


@pytest.fixture
def all_schemas_handler():
    """Create a GauntletAllSchemasHandler."""
    return GauntletAllSchemasHandler({})


@pytest.fixture
def templates_list_handler():
    """Create a GauntletTemplatesListHandler."""
    return GauntletTemplatesListHandler({})


@pytest.fixture
def template_handler():
    """Create a GauntletTemplateHandler."""
    return GauntletTemplateHandler(ctx={})


@pytest.fixture
def export_handler():
    """Create a GauntletReceiptExportHandler."""
    return GauntletReceiptExportHandler({})


@pytest.fixture
def heatmap_handler():
    """Create a GauntletHeatmapExportHandler."""
    return GauntletHeatmapExportHandler({})


@pytest.fixture
def validate_handler():
    """Create a GauntletValidateReceiptHandler."""
    return GauntletValidateReceiptHandler(ctx={})


def _mock_receipt(**overrides):
    """Create a mock DecisionReceipt-like object."""
    mock = MagicMock()
    mock.to_dict.return_value = {
        "receipt_id": "r-001",
        "gauntlet_id": "g-001",
        "timestamp": "2026-01-01T00:00:00",
        "verdict": "PASS",
        "confidence": 0.95,
        "robustness_score": 0.88,
    }
    mock.to_markdown.return_value = "# Receipt\nVERDICT: PASS"
    mock.to_html.return_value = "<h1>Receipt</h1>"
    for k, v in overrides.items():
        setattr(mock, k, v)
    return mock


def _add_gauntlet_run(run_id, status="completed", receipt=None, result=None, heatmap=None):
    """Add a gauntlet run to in-memory storage."""
    from aragora.server.handlers.gauntlet.storage import _gauntlet_runs

    run = {
        "id": run_id,
        "status": status,
    }
    if receipt is not None:
        run["receipt"] = receipt
    if result is not None:
        run["result"] = result
    if heatmap is not None:
        run["heatmap"] = heatmap
    _gauntlet_runs[run_id] = run
    return run


# ============================================================================
# RFC 7807 Error Helper
# ============================================================================


class TestRfc7807Error:
    """Test the rfc7807_error helper function."""

    def test_basic_error(self):
        result = rfc7807_error(400, "Bad Request", "Something went wrong")
        assert _status(result) == 400
        body = _body(result)
        assert body["type"] == PROBLEM_INTERNAL
        assert body["title"] == "Bad Request"
        assert body["detail"] == "Something went wrong"
        assert body["status"] == 400

    def test_custom_problem_type(self):
        result = rfc7807_error(404, "Not Found", "Gone", problem_type=PROBLEM_NOT_FOUND)
        body = _body(result)
        assert body["type"] == PROBLEM_NOT_FOUND

    def test_with_instance(self):
        result = rfc7807_error(404, "Not Found", "Missing", instance="/api/v1/gauntlet/123")
        body = _body(result)
        assert body["instance"] == "/api/v1/gauntlet/123"

    def test_without_instance(self):
        result = rfc7807_error(400, "Bad", "Nope")
        body = _body(result)
        assert "instance" not in body

    def test_extra_fields(self):
        result = rfc7807_error(400, "Validation", "Invalid", valid_formats=["json", "csv"])
        body = _body(result)
        assert body["valid_formats"] == ["json", "csv"]

    def test_content_type_is_problem_json(self):
        result = rfc7807_error(500, "Error", "Fail")
        assert _content_type(result) == "application/problem+json"

    def test_body_is_bytes(self):
        result = rfc7807_error(400, "Bad", "Nope")
        assert isinstance(result.body, bytes)

    def test_body_is_valid_json(self):
        result = rfc7807_error(500, "Error", "Fail")
        parsed = json.loads(result.body.decode("utf-8"))
        assert "type" in parsed
        assert "title" in parsed

    def test_multiple_extra_fields(self):
        result = rfc7807_error(
            400,
            "Err",
            "Err",
            available_schemas=["a", "b"],
            current_status="pending",
        )
        body = _body(result)
        assert body["available_schemas"] == ["a", "b"]
        assert body["current_status"] == "pending"


# ============================================================================
# Constants and Registration
# ============================================================================


class TestConstants:
    """Test module-level constants."""

    def test_problem_type_base(self):
        assert "aragora.ai" in PROBLEM_TYPE_BASE

    def test_problem_not_found(self):
        assert "not-found" in PROBLEM_NOT_FOUND

    def test_problem_validation(self):
        assert "validation" in PROBLEM_VALIDATION

    def test_problem_internal(self):
        assert "internal" in PROBLEM_INTERNAL

    def test_handler_list_contains_all_handlers(self):
        assert GauntletSchemaHandler in GAUNTLET_V1_HANDLERS
        assert GauntletAllSchemasHandler in GAUNTLET_V1_HANDLERS
        assert GauntletTemplatesListHandler in GAUNTLET_V1_HANDLERS
        assert GauntletTemplateHandler in GAUNTLET_V1_HANDLERS
        assert GauntletReceiptExportHandler in GAUNTLET_V1_HANDLERS
        assert GauntletHeatmapExportHandler in GAUNTLET_V1_HANDLERS
        assert GauntletValidateReceiptHandler in GAUNTLET_V1_HANDLERS

    def test_handler_list_length(self):
        assert len(GAUNTLET_V1_HANDLERS) == 7


class TestRegistration:
    """Test handler registration function."""

    def test_register_calls_add_handler(self):
        mock_router = MagicMock()
        register_gauntlet_v1_handlers(mock_router)
        assert mock_router.add_handler.call_count == 7

    def test_register_with_server_context(self):
        mock_router = MagicMock()
        ctx = {"key": "value"}
        register_gauntlet_v1_handlers(mock_router, server_context=ctx)
        assert mock_router.add_handler.call_count == 7

    def test_register_with_none_context(self):
        mock_router = MagicMock()
        register_gauntlet_v1_handlers(mock_router, server_context=None)
        assert mock_router.add_handler.call_count == 7


# ============================================================================
# GauntletSecureHandler Base
# ============================================================================


class TestGauntletSecureHandlerBase:
    """Test the base GauntletSecureHandler ABC."""

    def test_resource_type(self):
        handler = GauntletSchemaHandler(ctx={})
        assert handler.RESOURCE_TYPE == "gauntlet"

    def test_ctx_stored(self):
        ctx = {"key": "value"}
        handler = GauntletSchemaHandler(ctx=ctx)
        assert handler.ctx == ctx

    def test_default_methods_returns_get(self):
        handler = GauntletSchemaHandler(ctx={})
        assert handler.get_methods() == ["GET"]

    def test_export_handler_returns_post(self):
        handler = GauntletReceiptExportHandler({})
        assert handler.get_methods() == ["POST"]

    def test_validate_handler_returns_post(self):
        handler = GauntletValidateReceiptHandler(ctx={})
        assert handler.get_methods() == ["POST"]

    def test_schema_handler_init_with_none(self):
        handler = GauntletSchemaHandler(ctx=None)
        assert handler.ctx == {}

    def test_template_handler_init_with_none(self):
        handler = GauntletTemplateHandler(ctx=None)
        assert handler.ctx == {}

    def test_validate_handler_init_with_none(self):
        handler = GauntletValidateReceiptHandler(ctx=None)
        assert handler.ctx == {}


# ============================================================================
# Path Patterns
# ============================================================================


class TestPathPatterns:
    """Test URL path patterns for each handler."""

    def test_schema_path_pattern(self, schema_handler):
        assert "schema" in schema_handler.get_path_pattern()
        assert "schema_type" in schema_handler.get_path_pattern()

    def test_all_schemas_path_pattern(self, all_schemas_handler):
        assert "schemas" in all_schemas_handler.get_path_pattern()

    def test_templates_list_path_pattern(self, templates_list_handler):
        assert "templates" in templates_list_handler.get_path_pattern()

    def test_template_path_pattern(self, template_handler):
        assert "template_id" in template_handler.get_path_pattern()

    def test_export_path_pattern(self, export_handler):
        assert "gauntlet_id" in export_handler.get_path_pattern()
        assert "export" in export_handler.get_path_pattern()

    def test_heatmap_path_pattern(self, heatmap_handler):
        assert "heatmap" in heatmap_handler.get_path_pattern()
        assert "export" in heatmap_handler.get_path_pattern()

    def test_validate_path_pattern(self, validate_handler):
        assert "validate" in validate_handler.get_path_pattern()
        assert "receipt" in validate_handler.get_path_pattern()


# ============================================================================
# RBAC Permission Checks
# ============================================================================


class TestRBACPermissions:
    """Test RBAC permission enforcement (opt-out of auto-auth)."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_schema_handler_unauthorized(self):
        handler = GauntletSchemaHandler(ctx={})
        # Patch get_auth_context to raise UnauthorizedError
        with patch.object(handler, "get_auth_context", side_effect=Exception("no auth")):
            # The handler catches UnauthorizedError, not generic Exception,
            # so let's use the correct type
            from aragora.server.handlers.utils.auth import UnauthorizedError

            with patch.object(handler, "get_auth_context", side_effect=UnauthorizedError()):
                result = await handler.handle(
                    body=None,
                    path_params={"schema_type": "decision-receipt"},
                    handler=MagicMock(),
                )
                assert _status(result) == 401
                body = _body(result)
                assert "unauthorized" in body["type"].lower()

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_schema_handler_forbidden(self):
        handler = GauntletSchemaHandler(ctx={})
        from aragora.server.handlers.utils.auth import ForbiddenError

        mock_auth = AsyncMock(return_value=MagicMock())
        with patch.object(handler, "get_auth_context", mock_auth):
            with patch.object(handler, "check_permission", side_effect=ForbiddenError("no perm")):
                result = await handler.handle(
                    body=None,
                    path_params={"schema_type": "decision-receipt"},
                    handler=MagicMock(),
                )
                assert _status(result) == 403
                body = _body(result)
                assert "forbidden" in body["type"].lower()

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_templates_handler_unauthorized(self):
        handler = GauntletTemplatesListHandler({})
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch.object(handler, "get_auth_context", side_effect=UnauthorizedError()):
            result = await handler.handle(
                body=None,
                handler=MagicMock(),
            )
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_export_handler_unauthorized(self):
        handler = GauntletReceiptExportHandler({})
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch.object(handler, "get_auth_context", side_effect=UnauthorizedError()):
            result = await handler.handle(
                body={"format": "json"},
                path_params={"gauntlet_id": "g-001"},
                handler=MagicMock(),
            )
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_validate_handler_unauthorized(self):
        handler = GauntletValidateReceiptHandler(ctx={})
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch.object(handler, "get_auth_context", side_effect=UnauthorizedError()):
            result = await handler.handle(
                body={"receipt_id": "r-001"},
                handler=MagicMock(),
            )
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_heatmap_handler_forbidden(self):
        handler = GauntletHeatmapExportHandler({})
        from aragora.server.handlers.utils.auth import ForbiddenError

        mock_auth = AsyncMock(return_value=MagicMock())
        with patch.object(handler, "get_auth_context", mock_auth):
            with patch.object(handler, "check_permission", side_effect=ForbiddenError("nope")):
                result = await handler.handle(
                    body=None,
                    path_params={"gauntlet_id": "g-001"},
                    handler=MagicMock(),
                )
                assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_all_schemas_handler_unauthorized(self):
        handler = GauntletAllSchemasHandler({})
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch.object(handler, "get_auth_context", side_effect=UnauthorizedError()):
            result = await handler.handle(
                body=None,
                handler=MagicMock(),
            )
            assert _status(result) == 401


# ============================================================================
# GET /api/v1/gauntlet/schema/{type} - Schema Handler
# ============================================================================


class TestSchemaHandler:
    """Test GauntletSchemaHandler."""

    @pytest.mark.asyncio
    async def test_get_valid_schema(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": "decision-receipt"},
        )
        assert _status(result) == 200
        body = _body(result)
        assert "type" in body or "properties" in body or "$id" in body

    @pytest.mark.asyncio
    async def test_get_risk_heatmap_schema(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": "risk-heatmap"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_problem_detail_schema(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": "problem-detail"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_provenance_record_schema(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": "provenance-record"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_consensus_proof_schema(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": "consensus-proof"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_risk_summary_schema(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": "risk-summary"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_vulnerability_detail_schema(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": "vulnerability-detail"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_heatmap_cell_schema(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": "heatmap-cell"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unknown_schema_returns_404(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": "nonexistent-schema"},
        )
        assert _status(result) == 404
        body = _body(result)
        assert body["type"] == PROBLEM_NOT_FOUND
        assert "available_schemas" in body
        assert isinstance(body["available_schemas"], list)

    @pytest.mark.asyncio
    async def test_missing_schema_type_returns_400(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={},
        )
        assert _status(result) == 400
        body = _body(result)
        assert body["type"] == PROBLEM_VALIDATION

    @pytest.mark.asyncio
    async def test_none_path_params_returns_400(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params=None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_schema_type_returns_400(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": ""},
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_schema_import_error(self, schema_handler):
        with patch(
            "aragora.server.handlers.gauntlet_v1.GauntletSchemaHandler.handle",
            wraps=schema_handler.handle,
        ):
            with patch.dict("sys.modules", {"aragora.gauntlet.api": None}):
                result = await schema_handler.handle(
                    body=None,
                    path_params={"schema_type": "decision-receipt"},
                )
                # Import error triggers 500
                assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_schema_response_is_json(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": "decision-receipt"},
        )
        assert _content_type(result) == "application/json"


# ============================================================================
# GET /api/v1/gauntlet/schemas - All Schemas Handler
# ============================================================================


class TestAllSchemasHandler:
    """Test GauntletAllSchemasHandler."""

    @pytest.mark.asyncio
    async def test_get_all_schemas(self, all_schemas_handler):
        result = await all_schemas_handler.handle(body=None)
        assert _status(result) == 200
        body = _body(result)
        assert "schemas" in body
        assert "version" in body
        assert "count" in body
        assert body["count"] > 0
        assert isinstance(body["schemas"], dict)

    @pytest.mark.asyncio
    async def test_all_schemas_has_expected_keys(self, all_schemas_handler):
        result = await all_schemas_handler.handle(body=None)
        body = _body(result)
        schemas = body["schemas"]
        assert "decision-receipt" in schemas
        assert "risk-heatmap" in schemas
        assert "problem-detail" in schemas

    @pytest.mark.asyncio
    async def test_all_schemas_version(self, all_schemas_handler):
        result = await all_schemas_handler.handle(body=None)
        body = _body(result)
        assert body["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_all_schemas_count_matches(self, all_schemas_handler):
        result = await all_schemas_handler.handle(body=None)
        body = _body(result)
        assert body["count"] == len(body["schemas"])

    @pytest.mark.asyncio
    async def test_all_schemas_import_error(self, all_schemas_handler):
        with patch.dict("sys.modules", {"aragora.gauntlet.api": None}):
            result = await all_schemas_handler.handle(body=None)
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_all_schemas_response_is_json(self, all_schemas_handler):
        result = await all_schemas_handler.handle(body=None)
        assert _content_type(result) == "application/json"


# ============================================================================
# GET /api/v1/gauntlet/templates - Templates List Handler
# ============================================================================


class TestTemplatesListHandler:
    """Test GauntletTemplatesListHandler."""

    @pytest.mark.asyncio
    async def test_list_all_templates(self, templates_list_handler):
        result = await templates_list_handler.handle(body=None)
        assert _status(result) == 200
        body = _body(result)
        assert "templates" in body
        assert "count" in body
        assert isinstance(body["templates"], list)
        assert body["count"] == len(body["templates"])

    @pytest.mark.asyncio
    async def test_template_has_expected_fields(self, templates_list_handler):
        result = await templates_list_handler.handle(body=None)
        body = _body(result)
        if body["count"] > 0:
            t = body["templates"][0]
            assert "id" in t
            assert "name" in t
            assert "category" in t
            assert "description" in t
            assert "version" in t
            assert "regulations" in t
            assert "supported_formats" in t

    @pytest.mark.asyncio
    async def test_filter_by_compliance_category(self, templates_list_handler):
        result = await templates_list_handler.handle(
            body=None,
            query_params={"category": "compliance"},
        )
        assert _status(result) == 200
        body = _body(result)
        for t in body["templates"]:
            assert t["category"] == "compliance"

    @pytest.mark.asyncio
    async def test_filter_by_security_category(self, templates_list_handler):
        result = await templates_list_handler.handle(
            body=None,
            query_params={"category": "security"},
        )
        assert _status(result) == 200
        body = _body(result)
        for t in body["templates"]:
            assert t["category"] == "security"

    @pytest.mark.asyncio
    async def test_filter_by_legal_category(self, templates_list_handler):
        result = await templates_list_handler.handle(
            body=None,
            query_params={"category": "legal"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_filter_by_financial_category(self, templates_list_handler):
        result = await templates_list_handler.handle(
            body=None,
            query_params={"category": "financial"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_filter_by_operational_category(self, templates_list_handler):
        result = await templates_list_handler.handle(
            body=None,
            query_params={"category": "operational"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_invalid_category_returns_400(self, templates_list_handler):
        result = await templates_list_handler.handle(
            body=None,
            query_params={"category": "nonexistent"},
        )
        assert _status(result) == 400
        body = _body(result)
        assert body["type"] == PROBLEM_VALIDATION
        assert "valid_categories" in body

    @pytest.mark.asyncio
    async def test_category_case_insensitive(self, templates_list_handler):
        result = await templates_list_handler.handle(
            body=None,
            query_params={"category": "COMPLIANCE"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_no_category_filter(self, templates_list_handler):
        result = await templates_list_handler.handle(
            body=None,
            query_params={},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_none_query_params(self, templates_list_handler):
        result = await templates_list_handler.handle(
            body=None,
            query_params=None,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_templates_import_error(self, templates_list_handler):
        with patch.dict("sys.modules", {"aragora.gauntlet.api": None}):
            result = await templates_list_handler.handle(body=None)
            assert _status(result) == 500


# ============================================================================
# GET /api/v1/gauntlet/templates/{id} - Template Handler
# ============================================================================


class TestTemplateHandler:
    """Test GauntletTemplateHandler."""

    @pytest.mark.asyncio
    async def test_get_existing_template(self, template_handler):
        # Get the list of templates to find a valid ID
        from aragora.gauntlet.api import list_templates

        templates = list_templates()
        assert templates, "Template registry should be populated by fixture"

        template_id = templates[0].id
        result = await template_handler.handle(
            body=None,
            path_params={"template_id": template_id},
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == template_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_template_returns_404(self, template_handler):
        result = await template_handler.handle(
            body=None,
            path_params={"template_id": "does-not-exist-12345"},
        )
        assert _status(result) == 404
        body = _body(result)
        assert body["type"] == PROBLEM_NOT_FOUND
        assert "available_templates" in body

    @pytest.mark.asyncio
    async def test_missing_template_id_returns_400(self, template_handler):
        result = await template_handler.handle(
            body=None,
            path_params={},
        )
        assert _status(result) == 400
        body = _body(result)
        assert body["type"] == PROBLEM_VALIDATION

    @pytest.mark.asyncio
    async def test_none_path_params_returns_400(self, template_handler):
        result = await template_handler.handle(
            body=None,
            path_params=None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_template_id_returns_400(self, template_handler):
        result = await template_handler.handle(
            body=None,
            path_params={"template_id": ""},
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_template_response_has_sections(self, template_handler):
        from aragora.gauntlet.api import list_templates

        templates = list_templates()
        assert templates, "Template registry should be populated by fixture"

        result = await template_handler.handle(
            body=None,
            path_params={"template_id": templates[0].id},
        )
        body = _body(result)
        assert "sections" in body

    @pytest.mark.asyncio
    async def test_template_import_error(self, template_handler):
        with patch.dict("sys.modules", {"aragora.gauntlet.api": None}):
            result = await template_handler.handle(
                body=None,
                path_params={"template_id": "compliance"},
            )
            assert _status(result) == 500


# ============================================================================
# POST /api/v1/gauntlet/{id}/export - Receipt Export Handler
# ============================================================================


class TestReceiptExportHandler:
    """Test GauntletReceiptExportHandler."""

    @pytest.mark.asyncio
    async def test_export_json_with_receipt(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-001", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.export_receipt", return_value='{"test": true}'):
            result = await export_handler.handle(
                body={"format": "json"},
                path_params={"gauntlet_id": "g-001"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "application/json"

    @pytest.mark.asyncio
    async def test_export_markdown_with_receipt(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-002", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.export_receipt", return_value="# Receipt"):
            result = await export_handler.handle(
                body={"format": "markdown"},
                path_params={"gauntlet_id": "g-002"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "text/markdown"

    @pytest.mark.asyncio
    async def test_export_html_with_receipt(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-003", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.export_receipt", return_value="<html></html>"):
            result = await export_handler.handle(
                body={"format": "html"},
                path_params={"gauntlet_id": "g-003"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "text/html"

    @pytest.mark.asyncio
    async def test_export_csv_with_receipt(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-004", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.export_receipt", return_value="col1,col2\nval1,val2"):
            result = await export_handler.handle(
                body={"format": "csv"},
                path_params={"gauntlet_id": "g-004"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "text/csv"

    @pytest.mark.asyncio
    async def test_export_sarif_with_receipt(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-005", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.export_receipt", return_value='{"runs": []}'):
            result = await export_handler.handle(
                body={"format": "sarif"},
                path_params={"gauntlet_id": "g-005"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "application/sarif+json"

    @pytest.mark.asyncio
    async def test_export_default_format_is_json(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-006", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.export_receipt", return_value="{}"):
            result = await export_handler.handle(
                body={},
                path_params={"gauntlet_id": "g-006"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "application/json"

    @pytest.mark.asyncio
    async def test_export_invalid_format(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-007", status="completed", receipt=receipt)
        result = await export_handler.handle(
            body={"format": "pdf_invalid"},
            path_params={"gauntlet_id": "g-007"},
        )
        assert _status(result) == 400
        body = _body(result)
        assert "valid_formats" in body

    @pytest.mark.asyncio
    async def test_export_missing_gauntlet_id(self, export_handler):
        result = await export_handler.handle(
            body={"format": "json"},
            path_params={},
        )
        assert _status(result) == 400
        body = _body(result)
        assert body["type"] == PROBLEM_VALIDATION

    @pytest.mark.asyncio
    async def test_export_none_path_params(self, export_handler):
        result = await export_handler.handle(
            body={"format": "json"},
            path_params=None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_export_nonexistent_gauntlet(self, export_handler):
        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=MagicMock(get_result=MagicMock(return_value=None)),
        ):
            result = await export_handler.handle(
                body={"format": "json"},
                path_params={"gauntlet_id": "nonexistent-123"},
            )
        assert _status(result) == 404
        body = _body(result)
        assert body["type"] == PROBLEM_NOT_FOUND

    @pytest.mark.asyncio
    async def test_export_pending_gauntlet(self, export_handler):
        _add_gauntlet_run("g-pending", status="pending")
        result = await export_handler.handle(
            body={"format": "json"},
            path_params={"gauntlet_id": "g-pending"},
        )
        assert _status(result) == 400
        body = _body(result)
        assert "current_status" in body
        assert body["current_status"] == "pending"

    @pytest.mark.asyncio
    async def test_export_running_gauntlet(self, export_handler):
        _add_gauntlet_run("g-running", status="running")
        result = await export_handler.handle(
            body={"format": "json"},
            path_params={"gauntlet_id": "g-running"},
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_export_none_body_defaults_to_json(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-none-body", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.export_receipt", return_value="{}"):
            result = await export_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-none-body"},
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_export_with_options(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-opts", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.export_receipt", return_value="{}") as mock_export:
            result = await export_handler.handle(
                body={
                    "format": "json",
                    "options": {
                        "include_provenance": False,
                        "include_config": True,
                        "max_vulnerabilities": 50,
                        "validate_schema": True,
                    },
                },
                path_params={"gauntlet_id": "g-opts"},
            )
        assert _status(result) == 200
        # Verify the options were passed through
        call_args = mock_export.call_args
        options_arg = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("options")
        if options_arg is not None:
            assert options_arg.include_provenance is False
            assert options_arg.include_config is True

    @pytest.mark.asyncio
    async def test_export_with_template(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-tmpl", status="completed", receipt=receipt)

        mock_template = MagicMock()
        mock_template.render.return_value = "# Rendered with template"

        with patch("aragora.gauntlet.api.get_template", return_value=mock_template):
            result = await export_handler.handle(
                body={
                    "format": "markdown",
                    "template_id": "compliance",
                },
                path_params={"gauntlet_id": "g-tmpl"},
            )
        assert _status(result) == 200
        assert result.body == b"# Rendered with template"

    @pytest.mark.asyncio
    async def test_export_with_nonexistent_template(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-badtmpl", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.get_template", return_value=None):
            result = await export_handler.handle(
                body={
                    "format": "json",
                    "template_id": "nonexistent-template",
                },
                path_params={"gauntlet_id": "g-badtmpl"},
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_export_receipt_from_result_dict(self, export_handler):
        """Test creating receipt from result dict when no receipt exists."""
        result_data = {
            "completed_at": "2026-01-01T00:00:00",
            "input_summary": "test input",
            "input_hash": "abc123",
            "risk_summary": {"total": 2},
            "attack_summary": {"total_attacks": 5, "successful_attacks": 1},
            "probe_summary": {"probes_run": 10},
            "verdict": "PASS",
            "confidence": 0.9,
            "robustness_score": 0.85,
        }
        _add_gauntlet_run("g-from-result", status="completed", result=result_data)
        with patch("aragora.gauntlet.api.export_receipt", return_value='{"ok": true}'):
            result = await export_handler.handle(
                body={"format": "json"},
                path_params={"gauntlet_id": "g-from-result"},
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_export_receipt_from_result_with_to_receipt(self, export_handler):
        """Test creating receipt from result object that has to_receipt()."""
        mock_result = MagicMock()
        mock_result.to_receipt.return_value = _mock_receipt()
        _add_gauntlet_run("g-to-receipt", status="completed", result=mock_result)
        with patch("aragora.gauntlet.api.export_receipt", return_value="{}"):
            result = await export_handler.handle(
                body={"format": "json"},
                path_params={"gauntlet_id": "g-to-receipt"},
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_export_no_receipt_no_result(self, export_handler):
        """Run with no receipt and no result should return 500."""
        _add_gauntlet_run("g-no-receipt", status="completed")
        result = await export_handler.handle(
            body={"format": "json"},
            path_params={"gauntlet_id": "g-no-receipt"},
        )
        assert _status(result) == 500
        body = _body(result)
        assert "receipt" in body["detail"].lower() or "Receipt" in body["detail"]

    @pytest.mark.asyncio
    async def test_export_from_persistent_storage(self, export_handler):
        """Test finding a run in persistent storage when not in memory."""
        mock_storage = MagicMock()
        mock_storage.get_result.return_value = {
            "id": "g-stored",
            "status": "completed",
            "receipt": _mock_receipt(),
        }
        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            with patch("aragora.gauntlet.api.export_receipt", return_value="{}"):
                result = await export_handler.handle(
                    body={"format": "json"},
                    path_params={"gauntlet_id": "g-stored"},
                )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_export_format_case_insensitive(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-case", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.export_receipt", return_value="{}"):
            result = await export_handler.handle(
                body={"format": "JSON"},
                path_params={"gauntlet_id": "g-case"},
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_export_template_html_format(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-html-tmpl", status="completed", receipt=receipt)
        mock_template = MagicMock()
        mock_template.render.return_value = "<html>Report</html>"
        with patch("aragora.gauntlet.api.get_template", return_value=mock_template):
            result = await export_handler.handle(
                body={"format": "html", "template_id": "security"},
                path_params={"gauntlet_id": "g-html-tmpl"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "text/html"

    @pytest.mark.asyncio
    async def test_export_template_json_format(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-json-tmpl", status="completed", receipt=receipt)
        mock_template = MagicMock()
        mock_template.render.return_value = '{"report": true}'
        with patch("aragora.gauntlet.api.get_template", return_value=mock_template):
            result = await export_handler.handle(
                body={"format": "json", "template_id": "compliance"},
                path_params={"gauntlet_id": "g-json-tmpl"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "application/json"

    @pytest.mark.asyncio
    async def test_export_template_text_format(self, export_handler):
        receipt = _mock_receipt()
        _add_gauntlet_run("g-txt-tmpl", status="completed", receipt=receipt)
        mock_template = MagicMock()
        mock_template.render.return_value = "Plain text report"
        with patch("aragora.gauntlet.api.get_template", return_value=mock_template):
            result = await export_handler.handle(
                body={"format": "text", "template_id": "operational"},
                path_params={"gauntlet_id": "g-txt-tmpl"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "text/plain"


# ============================================================================
# GET /api/v1/gauntlet/{id}/heatmap/export - Heatmap Export Handler
# ============================================================================


class TestHeatmapExportHandler:
    """Test GauntletHeatmapExportHandler."""

    @pytest.mark.asyncio
    async def test_export_heatmap_json(self, heatmap_handler):
        mock_heatmap = MagicMock()
        _add_gauntlet_run("g-hm1", status="completed", heatmap=mock_heatmap)
        with patch("aragora.gauntlet.api.export_heatmap", return_value='{"cells": []}'):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-hm1"},
                query_params={"format": "json"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "application/json"

    @pytest.mark.asyncio
    async def test_export_heatmap_csv(self, heatmap_handler):
        mock_heatmap = MagicMock()
        _add_gauntlet_run("g-hm-csv", status="completed", heatmap=mock_heatmap)
        with patch("aragora.gauntlet.api.export_heatmap", return_value="col1,col2"):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-hm-csv"},
                query_params={"format": "csv"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "text/csv"

    @pytest.mark.asyncio
    async def test_export_heatmap_svg(self, heatmap_handler):
        mock_heatmap = MagicMock()
        _add_gauntlet_run("g-hm-svg", status="completed", heatmap=mock_heatmap)
        with patch("aragora.gauntlet.api.export_heatmap", return_value="<svg></svg>"):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-hm-svg"},
                query_params={"format": "svg"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "image/svg+xml"

    @pytest.mark.asyncio
    async def test_export_heatmap_ascii(self, heatmap_handler):
        mock_heatmap = MagicMock()
        _add_gauntlet_run("g-hm-ascii", status="completed", heatmap=mock_heatmap)
        with patch("aragora.gauntlet.api.export_heatmap", return_value="[H][M][L]"):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-hm-ascii"},
                query_params={"format": "ascii"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "text/plain"

    @pytest.mark.asyncio
    async def test_export_heatmap_html(self, heatmap_handler):
        mock_heatmap = MagicMock()
        _add_gauntlet_run("g-hm-html", status="completed", heatmap=mock_heatmap)
        with patch("aragora.gauntlet.api.export_heatmap", return_value="<table></table>"):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-hm-html"},
                query_params={"format": "html"},
            )
        assert _status(result) == 200
        assert _content_type(result) == "text/html"

    @pytest.mark.asyncio
    async def test_export_heatmap_default_format_json(self, heatmap_handler):
        mock_heatmap = MagicMock()
        _add_gauntlet_run("g-hm-default", status="completed", heatmap=mock_heatmap)
        with patch("aragora.gauntlet.api.export_heatmap", return_value="{}"):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-hm-default"},
                query_params={},
            )
        assert _status(result) == 200
        assert _content_type(result) == "application/json"

    @pytest.mark.asyncio
    async def test_export_heatmap_invalid_format(self, heatmap_handler):
        mock_heatmap = MagicMock()
        _add_gauntlet_run("g-hm-bad", status="completed", heatmap=mock_heatmap)
        result = await heatmap_handler.handle(
            body=None,
            path_params={"gauntlet_id": "g-hm-bad"},
            query_params={"format": "pdf"},
        )
        assert _status(result) == 400
        body = _body(result)
        assert "valid_formats" in body

    @pytest.mark.asyncio
    async def test_export_heatmap_missing_gauntlet_id(self, heatmap_handler):
        result = await heatmap_handler.handle(
            body=None,
            path_params={},
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_export_heatmap_none_path_params(self, heatmap_handler):
        result = await heatmap_handler.handle(
            body=None,
            path_params=None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_export_heatmap_nonexistent_gauntlet(self, heatmap_handler):
        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=MagicMock(get_result=MagicMock(return_value=None)),
        ):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "nonexistent"},
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_export_heatmap_no_heatmap_data(self, heatmap_handler):
        _add_gauntlet_run("g-no-hm", status="completed")
        result = await heatmap_handler.handle(
            body=None,
            path_params={"gauntlet_id": "g-no-hm"},
            query_params={"format": "json"},
        )
        assert _status(result) == 404
        body = _body(result)
        assert "heatmap" in body["detail"].lower()

    @pytest.mark.asyncio
    async def test_export_heatmap_from_result_to_heatmap(self, heatmap_handler):
        """Test creating heatmap from result with to_heatmap method."""
        mock_result = MagicMock()
        mock_heatmap = MagicMock()
        mock_result.to_heatmap.return_value = mock_heatmap
        _add_gauntlet_run("g-hm-from-result", status="completed", result=mock_result)
        with patch("aragora.gauntlet.api.export_heatmap", return_value="{}"):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-hm-from-result"},
                query_params={"format": "json"},
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_export_heatmap_from_persistent_storage(self, heatmap_handler):
        mock_heatmap = MagicMock()
        mock_storage = MagicMock()
        mock_storage.get_result.return_value = {
            "id": "g-stored-hm",
            "status": "completed",
            "heatmap": mock_heatmap,
        }
        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            with patch("aragora.gauntlet.api.export_heatmap", return_value="{}"):
                result = await heatmap_handler.handle(
                    body=None,
                    path_params={"gauntlet_id": "g-stored-hm"},
                    query_params={"format": "json"},
                )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_export_heatmap_format_case_insensitive(self, heatmap_handler):
        mock_heatmap = MagicMock()
        _add_gauntlet_run("g-hm-case", status="completed", heatmap=mock_heatmap)
        with patch("aragora.gauntlet.api.export_heatmap", return_value="{}"):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-hm-case"},
                query_params={"format": "JSON"},
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_export_heatmap_none_query_params(self, heatmap_handler):
        mock_heatmap = MagicMock()
        _add_gauntlet_run("g-hm-nq", status="completed", heatmap=mock_heatmap)
        with patch("aragora.gauntlet.api.export_heatmap", return_value="{}"):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-hm-nq"},
                query_params=None,
            )
        assert _status(result) == 200


# ============================================================================
# POST /api/v1/gauntlet/validate/receipt - Validate Receipt Handler
# ============================================================================


class TestValidateReceiptHandler:
    """Test GauntletValidateReceiptHandler."""

    @pytest.mark.asyncio
    async def test_validate_valid_receipt(self, validate_handler):
        mock_body = {"receipt_id": "r-001", "verdict": "PASS", "confidence": 0.9}
        with patch(
            "aragora.gauntlet.api.validate_receipt",
            return_value=(True, []),
        ):
            result = await validate_handler.handle(
                body=mock_body,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is True
        assert body["errors"] == []
        assert body["error_count"] == 0

    @pytest.mark.asyncio
    async def test_validate_invalid_receipt(self, validate_handler):
        mock_body = {"receipt_id": "r-001"}
        errors = ["Missing field: verdict", "Missing field: confidence"]
        with patch(
            "aragora.gauntlet.api.validate_receipt",
            return_value=(False, errors),
        ):
            result = await validate_handler.handle(
                body=mock_body,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is False
        assert body["error_count"] == 2
        assert len(body["errors"]) == 2

    @pytest.mark.asyncio
    async def test_validate_empty_body_returns_400(self, validate_handler):
        result = await validate_handler.handle(
            body=None,
        )
        assert _status(result) == 400
        body = _body(result)
        assert body["type"] == PROBLEM_VALIDATION

    @pytest.mark.asyncio
    async def test_validate_empty_dict_body_returns_400(self, validate_handler):
        result = await validate_handler.handle(
            body={},
        )
        # Empty dict is falsy in "if not body:" check
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_validate_import_error(self, validate_handler):
        with patch.dict("sys.modules", {"aragora.gauntlet.api": None}):
            result = await validate_handler.handle(
                body={"receipt_id": "r-001"},
            )
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_validate_response_is_json(self, validate_handler):
        with patch(
            "aragora.gauntlet.api.validate_receipt",
            return_value=(True, []),
        ):
            result = await validate_handler.handle(
                body={"receipt_id": "r-001"},
            )
        assert _content_type(result) == "application/json"

    @pytest.mark.asyncio
    async def test_validate_with_many_errors(self, validate_handler):
        errors = [f"Error {i}" for i in range(20)]
        with patch(
            "aragora.gauntlet.api.validate_receipt",
            return_value=(False, errors),
        ):
            result = await validate_handler.handle(
                body={"some": "data"},
            )
        body = _body(result)
        assert body["error_count"] == 20
        assert len(body["errors"]) == 20


# ============================================================================
# Protocol Compliance
# ============================================================================


class TestProtocolCompliance:
    """Test that handlers comply with the VersionedAPIHandler protocol."""

    def test_schema_handler_is_versioned(self, schema_handler):
        assert hasattr(schema_handler, "get_path_pattern")
        assert hasattr(schema_handler, "get_methods")
        assert hasattr(schema_handler, "handle")

    def test_all_schemas_handler_is_versioned(self, all_schemas_handler):
        assert callable(all_schemas_handler.get_path_pattern)
        assert callable(all_schemas_handler.get_methods)
        assert callable(all_schemas_handler.handle)

    def test_templates_list_handler_is_versioned(self, templates_list_handler):
        assert callable(templates_list_handler.get_path_pattern)
        assert callable(templates_list_handler.get_methods)

    def test_template_handler_is_versioned(self, template_handler):
        assert callable(template_handler.get_path_pattern)
        assert callable(template_handler.get_methods)

    def test_export_handler_is_versioned(self, export_handler):
        assert callable(export_handler.get_path_pattern)
        assert callable(export_handler.get_methods)

    def test_heatmap_handler_is_versioned(self, heatmap_handler):
        assert callable(heatmap_handler.get_path_pattern)
        assert callable(heatmap_handler.get_methods)

    def test_validate_handler_is_versioned(self, validate_handler):
        assert callable(validate_handler.get_path_pattern)
        assert callable(validate_handler.get_methods)

    def test_all_handlers_have_check_gauntlet_permission(self):
        for cls in GAUNTLET_V1_HANDLERS:
            handler = cls({})
            assert hasattr(handler, "check_gauntlet_permission")

    def test_all_handlers_have_check_permission(self):
        for cls in GAUNTLET_V1_HANDLERS:
            handler = cls({})
            assert hasattr(handler, "check_permission")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    @pytest.mark.asyncio
    async def test_schema_handler_with_extra_kwargs(self, schema_handler):
        result = await schema_handler.handle(
            body=None,
            path_params={"schema_type": "decision-receipt"},
            query_params=None,
            extra_param="should_be_ignored",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_templates_list_with_extra_kwargs(self, templates_list_handler):
        result = await templates_list_handler.handle(
            body=None,
            query_params={},
            extra="value",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_export_handler_storage_no_get_result(self, export_handler):
        """Storage object without get_result attribute."""
        mock_storage = MagicMock(spec=[])  # No attributes
        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            result = await export_handler.handle(
                body={"format": "json"},
                path_params={"gauntlet_id": "no-get-result"},
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_heatmap_result_no_to_heatmap(self, heatmap_handler):
        """Result object without to_heatmap method."""
        result_obj = {"some": "data"}  # Dict, no to_heatmap attribute
        _add_gauntlet_run("g-no-toheatmap", status="completed", result=result_obj)
        result = await heatmap_handler.handle(
            body=None,
            path_params={"gauntlet_id": "g-no-toheatmap"},
            query_params={"format": "json"},
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_export_receipt_bytes_content(self, export_handler):
        """Export returning bytes instead of string."""
        receipt = _mock_receipt()
        _add_gauntlet_run("g-bytes", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.export_receipt", return_value=b"binary content"):
            result = await export_handler.handle(
                body={"format": "json"},
                path_params={"gauntlet_id": "g-bytes"},
            )
        assert _status(result) == 200
        assert result.body == b"binary content"

    @pytest.mark.asyncio
    async def test_heatmap_export_bytes_content(self, heatmap_handler):
        """Heatmap export returning bytes instead of string."""
        mock_heatmap = MagicMock()
        _add_gauntlet_run("g-hm-bytes", status="completed", heatmap=mock_heatmap)
        with patch("aragora.gauntlet.api.export_heatmap", return_value=b"binary"):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-hm-bytes"},
                query_params={"format": "json"},
            )
        assert _status(result) == 200
        assert result.body == b"binary"

    @pytest.mark.asyncio
    async def test_export_template_render_bytes(self, export_handler):
        """Template render returning bytes instead of string."""
        receipt = _mock_receipt()
        _add_gauntlet_run("g-tmpl-bytes", status="completed", receipt=receipt)
        mock_template = MagicMock()
        mock_template.render.return_value = b"<html>bytes</html>"
        with patch("aragora.gauntlet.api.get_template", return_value=mock_template):
            result = await export_handler.handle(
                body={"format": "html", "template_id": "test"},
                path_params={"gauntlet_id": "g-tmpl-bytes"},
            )
        assert _status(result) == 200
        assert result.body == b"<html>bytes</html>"

    @pytest.mark.asyncio
    async def test_validate_receipt_value_error(self, validate_handler):
        """validate_receipt raising ValueError."""
        with patch(
            "aragora.gauntlet.api.validate_receipt",
            side_effect=ValueError("bad data"),
        ):
            result = await validate_handler.handle(
                body={"data": "test"},
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_export_handler_general_exception(self, export_handler):
        """General exception during export."""
        receipt = _mock_receipt()
        _add_gauntlet_run("g-exc", status="completed", receipt=receipt)
        with patch("aragora.gauntlet.api.export_receipt", side_effect=ValueError("fail")):
            result = await export_handler.handle(
                body={"format": "json"},
                path_params={"gauntlet_id": "g-exc"},
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_heatmap_export_exception(self, heatmap_handler):
        """Exception during heatmap export."""
        mock_heatmap = MagicMock()
        _add_gauntlet_run("g-hm-exc", status="completed", heatmap=mock_heatmap)
        with patch("aragora.gauntlet.api.export_heatmap", side_effect=TypeError("fail")):
            result = await heatmap_handler.handle(
                body=None,
                path_params={"gauntlet_id": "g-hm-exc"},
                query_params={"format": "json"},
            )
        assert _status(result) == 500
