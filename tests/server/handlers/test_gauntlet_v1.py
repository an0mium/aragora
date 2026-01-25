"""
Tests for aragora.server.handlers.gauntlet_v1 - Versioned Gauntlet API handlers.

Tests cover:
- Schema retrieval (single and all schemas)
- Template listing and retrieval
- Receipt export in various formats
- Heatmap export
- Receipt validation
- RFC 7807 error responses
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.gauntlet_v1 import (
    GauntletSchemaHandler,
    GauntletAllSchemasHandler,
    GauntletTemplatesListHandler,
    GauntletTemplateHandler,
    GauntletReceiptExportHandler,
    GauntletHeatmapExportHandler,
    GauntletValidateReceiptHandler,
    GAUNTLET_V1_HANDLERS,
    rfc7807_error,
    PROBLEM_NOT_FOUND,
    PROBLEM_VALIDATION,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def server_context() -> Dict[str, Any]:
    """Create empty server context for handlers."""
    return {}


@pytest.fixture
def schema_handler(server_context: Dict[str, Any]) -> GauntletSchemaHandler:
    """Create schema handler instance."""
    return GauntletSchemaHandler(server_context)


@pytest.fixture
def all_schemas_handler(server_context: Dict[str, Any]) -> GauntletAllSchemasHandler:
    """Create all schemas handler instance."""
    return GauntletAllSchemasHandler(server_context)


@pytest.fixture
def templates_list_handler(server_context: Dict[str, Any]) -> GauntletTemplatesListHandler:
    """Create templates list handler instance."""
    return GauntletTemplatesListHandler(server_context)


@pytest.fixture
def template_handler(server_context: Dict[str, Any]) -> GauntletTemplateHandler:
    """Create single template handler instance."""
    return GauntletTemplateHandler(server_context)


@pytest.fixture
def receipt_export_handler(server_context: Dict[str, Any]) -> GauntletReceiptExportHandler:
    """Create receipt export handler instance."""
    return GauntletReceiptExportHandler(server_context)


@pytest.fixture
def heatmap_export_handler(server_context: Dict[str, Any]) -> GauntletHeatmapExportHandler:
    """Create heatmap export handler instance."""
    return GauntletHeatmapExportHandler(server_context)


@pytest.fixture
def validate_receipt_handler(server_context: Dict[str, Any]) -> GauntletValidateReceiptHandler:
    """Create validate receipt handler instance."""
    return GauntletValidateReceiptHandler(server_context)


# ===========================================================================
# Helper Functions
# ===========================================================================


def parse_response_body(result) -> Dict[str, Any]:
    """Parse JSON response body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


# ===========================================================================
# rfc7807_error Tests
# ===========================================================================


class TestRfc7807Error:
    """Tests for RFC 7807 error response generation."""

    def test_basic_error_response(self):
        """Test creating a basic RFC 7807 error."""
        result = rfc7807_error(
            status=404,
            title="Not Found",
            detail="Resource not found",
        )

        assert result.status_code == 404
        assert result.content_type == "application/problem+json"

        body = parse_response_body(result)
        assert body["status"] == 404
        assert body["title"] == "Not Found"
        assert body["detail"] == "Resource not found"
        assert "type" in body

    def test_error_with_instance(self):
        """Test RFC 7807 error with instance URI."""
        result = rfc7807_error(
            status=404,
            title="Not Found",
            detail="Item not found",
            instance="/api/v1/gauntlet/123",
        )

        body = parse_response_body(result)
        assert body["instance"] == "/api/v1/gauntlet/123"

    def test_error_with_extra_fields(self):
        """Test RFC 7807 error with custom extension fields."""
        result = rfc7807_error(
            status=400,
            title="Validation Error",
            detail="Invalid input",
            problem_type=PROBLEM_VALIDATION,
            valid_values=["a", "b", "c"],
        )

        body = parse_response_body(result)
        assert body["valid_values"] == ["a", "b", "c"]


# ===========================================================================
# GauntletSchemaHandler Tests
# ===========================================================================


class TestGauntletSchemaHandler:
    """Tests for GET /api/v1/gauntlet/schema/{type}."""

    def test_path_pattern(self, schema_handler: GauntletSchemaHandler):
        """Test handler path pattern."""
        pattern = schema_handler.get_path_pattern()
        assert "schema" in pattern
        assert "schema_type" in pattern

    @pytest.mark.asyncio
    async def test_get_schema_success(self, schema_handler: GauntletSchemaHandler):
        """Test successfully retrieving a schema."""
        mock_schemas = {
            "decision-receipt": {"type": "object", "properties": {}},
            "risk-heatmap": {"type": "object", "properties": {}},
        }

        mock_api = MagicMock()
        mock_api.get_all_schemas.return_value = mock_schemas

        with patch.dict("sys.modules", {"aragora.gauntlet.api": mock_api}):
            result = await schema_handler.handle(
                body=None,
                path_params={"schema_type": "decision-receipt"},
            )

        assert result.status_code == 200
        body = parse_response_body(result)
        assert body["type"] == "object"

    @pytest.mark.asyncio
    async def test_get_schema_missing_type(self, schema_handler: GauntletSchemaHandler):
        """Test error when schema type is missing."""
        result = await schema_handler.handle(
            body=None,
            path_params={},
        )

        assert result.status_code == 400
        body = parse_response_body(result)
        assert "Missing Schema Type" in body["title"]

    @pytest.mark.asyncio
    async def test_get_schema_not_found(self, schema_handler: GauntletSchemaHandler):
        """Test error when schema type doesn't exist."""
        mock_schemas = {"decision-receipt": {}}

        mock_api = MagicMock()
        mock_api.get_all_schemas.return_value = mock_schemas

        with patch.dict("sys.modules", {"aragora.gauntlet.api": mock_api}):
            result = await schema_handler.handle(
                body=None,
                path_params={"schema_type": "nonexistent"},
            )

        assert result.status_code == 404
        body = parse_response_body(result)
        assert "Schema Not Found" in body["title"]
        assert "available_schemas" in body


# ===========================================================================
# GauntletAllSchemasHandler Tests
# ===========================================================================


class TestGauntletAllSchemasHandler:
    """Tests for GET /api/v1/gauntlet/schemas."""

    def test_path_pattern(self, all_schemas_handler: GauntletAllSchemasHandler):
        """Test handler path pattern."""
        pattern = all_schemas_handler.get_path_pattern()
        assert "/api/v1/gauntlet/schemas" in pattern

    @pytest.mark.asyncio
    async def test_get_all_schemas_success(self, all_schemas_handler: GauntletAllSchemasHandler):
        """Test successfully retrieving all schemas."""
        mock_schemas = {
            "decision-receipt": {"type": "object"},
            "risk-heatmap": {"type": "object"},
        }

        mock_api = MagicMock()
        mock_api.get_all_schemas.return_value = mock_schemas
        mock_api.SCHEMA_VERSION = "1.0.0"

        with patch.dict("sys.modules", {"aragora.gauntlet.api": mock_api}):
            result = await all_schemas_handler.handle(body=None)

        assert result.status_code == 200
        body = parse_response_body(result)
        assert "schemas" in body
        assert body["count"] == 2


# ===========================================================================
# GauntletTemplatesListHandler Tests
# ===========================================================================


class TestGauntletTemplatesListHandler:
    """Tests for GET /api/v1/gauntlet/templates."""

    def test_path_pattern(self, templates_list_handler: GauntletTemplatesListHandler):
        """Test handler path pattern."""
        pattern = templates_list_handler.get_path_pattern()
        assert "/api/v1/gauntlet/templates" in pattern

    @pytest.mark.asyncio
    async def test_list_templates_success(
        self, templates_list_handler: GauntletTemplatesListHandler
    ):
        """Test successfully listing templates."""
        mock_template = MagicMock()
        mock_template.id = "template-1"
        mock_template.name = "SOC 2 Template"
        mock_template.category.value = "compliance"
        mock_template.description = "SOC 2 compliance audit"
        mock_template.version = "1.0"
        mock_template.regulations = ["SOC 2"]
        mock_template.supported_formats = [MagicMock(value="json"), MagicMock(value="markdown")]

        mock_api = MagicMock()
        mock_api.list_templates.return_value = [mock_template]
        mock_api.TemplateCategory = MagicMock()

        with patch.dict("sys.modules", {"aragora.gauntlet.api": mock_api}):
            result = await templates_list_handler.handle(
                body=None,
                query_params={},
            )

        assert result.status_code == 200
        body = parse_response_body(result)
        assert body["count"] == 1
        assert body["templates"][0]["id"] == "template-1"

    @pytest.mark.asyncio
    async def test_list_templates_with_invalid_category(
        self, templates_list_handler: GauntletTemplatesListHandler
    ):
        """Test error with invalid category filter."""
        mock_api = MagicMock()
        mock_api.TemplateCategory.side_effect = ValueError("Invalid category")

        with patch.dict("sys.modules", {"aragora.gauntlet.api": mock_api}):
            result = await templates_list_handler.handle(
                body=None,
                query_params={"category": "invalid"},
            )

        assert result.status_code == 400
        body = parse_response_body(result)
        assert "Invalid Category" in body["title"]


# ===========================================================================
# GauntletTemplateHandler Tests
# ===========================================================================


class TestGauntletTemplateHandler:
    """Tests for GET /api/v1/gauntlet/templates/{id}."""

    def test_path_pattern(self, template_handler: GauntletTemplateHandler):
        """Test handler path pattern."""
        pattern = template_handler.get_path_pattern()
        assert "template_id" in pattern

    @pytest.mark.asyncio
    async def test_get_template_success(self, template_handler: GauntletTemplateHandler):
        """Test successfully retrieving a template."""
        mock_template = MagicMock()
        mock_template.to_dict.return_value = {
            "id": "template-1",
            "name": "SOC 2 Template",
        }

        mock_api = MagicMock()
        mock_api.get_template.return_value = mock_template

        with patch.dict("sys.modules", {"aragora.gauntlet.api": mock_api}):
            result = await template_handler.handle(
                body=None,
                path_params={"template_id": "template-1"},
            )

        assert result.status_code == 200
        body = parse_response_body(result)
        assert body["id"] == "template-1"

    @pytest.mark.asyncio
    async def test_get_template_not_found(self, template_handler: GauntletTemplateHandler):
        """Test error when template doesn't exist."""
        mock_api = MagicMock()
        mock_api.get_template.return_value = None
        mock_api.list_templates.return_value = []

        with patch.dict("sys.modules", {"aragora.gauntlet.api": mock_api}):
            result = await template_handler.handle(
                body=None,
                path_params={"template_id": "nonexistent"},
            )

        assert result.status_code == 404
        body = parse_response_body(result)
        assert "Template Not Found" in body["title"]

    @pytest.mark.asyncio
    async def test_get_template_missing_id(self, template_handler: GauntletTemplateHandler):
        """Test error when template ID is missing."""
        result = await template_handler.handle(
            body=None,
            path_params={},
        )

        assert result.status_code == 400
        body = parse_response_body(result)
        assert "Missing Template ID" in body["title"]


# ===========================================================================
# GauntletValidateReceiptHandler Tests
# ===========================================================================


class TestGauntletValidateReceiptHandler:
    """Tests for POST /api/v1/gauntlet/validate/receipt."""

    def test_path_pattern(self, validate_receipt_handler: GauntletValidateReceiptHandler):
        """Test handler path pattern."""
        pattern = validate_receipt_handler.get_path_pattern()
        assert "validate/receipt" in pattern

    def test_methods(self, validate_receipt_handler: GauntletValidateReceiptHandler):
        """Test handler accepts POST method."""
        methods = validate_receipt_handler.get_methods()
        assert "POST" in methods

    @pytest.mark.asyncio
    async def test_validate_receipt_success(
        self, validate_receipt_handler: GauntletValidateReceiptHandler
    ):
        """Test validating a valid receipt."""
        mock_api = MagicMock()
        mock_api.validate_receipt.return_value = (True, [])

        with patch.dict("sys.modules", {"aragora.gauntlet.api": mock_api}):
            result = await validate_receipt_handler.handle(
                body={"receipt_id": "test-123"},
            )

        assert result.status_code == 200
        body = parse_response_body(result)
        assert body["valid"] is True
        assert body["error_count"] == 0

    @pytest.mark.asyncio
    async def test_validate_receipt_invalid(
        self, validate_receipt_handler: GauntletValidateReceiptHandler
    ):
        """Test validating an invalid receipt."""
        mock_api = MagicMock()
        mock_api.validate_receipt.return_value = (False, ["Missing required field: gauntlet_id"])

        with patch.dict("sys.modules", {"aragora.gauntlet.api": mock_api}):
            result = await validate_receipt_handler.handle(
                body={"incomplete": "data"},
            )

        assert result.status_code == 200
        body = parse_response_body(result)
        assert body["valid"] is False
        assert body["error_count"] == 1

    @pytest.mark.asyncio
    async def test_validate_receipt_missing_body(
        self, validate_receipt_handler: GauntletValidateReceiptHandler
    ):
        """Test error when request body is missing."""
        result = await validate_receipt_handler.handle(body=None)

        assert result.status_code == 400
        body = parse_response_body(result)
        assert "Missing Body" in body["title"]


# ===========================================================================
# GauntletReceiptExportHandler Tests
# ===========================================================================


class TestGauntletReceiptExportHandler:
    """Tests for POST /api/v1/gauntlet/{id}/export."""

    def test_path_pattern(self, receipt_export_handler: GauntletReceiptExportHandler):
        """Test handler path pattern."""
        pattern = receipt_export_handler.get_path_pattern()
        assert "gauntlet_id" in pattern
        assert "export" in pattern

    def test_methods(self, receipt_export_handler: GauntletReceiptExportHandler):
        """Test handler accepts POST method."""
        methods = receipt_export_handler.get_methods()
        assert "POST" in methods

    @pytest.mark.asyncio
    async def test_export_missing_gauntlet_id(
        self, receipt_export_handler: GauntletReceiptExportHandler
    ):
        """Test error when gauntlet ID is missing."""
        result = await receipt_export_handler.handle(
            body={"format": "json"},
            path_params={},
        )

        assert result.status_code == 400
        body = parse_response_body(result)
        assert "Missing Gauntlet ID" in body["title"]

    @pytest.mark.asyncio
    async def test_export_gauntlet_not_found(
        self, receipt_export_handler: GauntletReceiptExportHandler
    ):
        """Test error when gauntlet run doesn't exist."""
        mock_gauntlet_module = MagicMock()
        mock_gauntlet_module._gauntlet_runs = {}
        mock_gauntlet_module._get_storage.return_value = MagicMock(
            get_result=MagicMock(return_value=None)
        )

        with patch.dict("sys.modules", {"aragora.server.handlers.gauntlet": mock_gauntlet_module}):
            result = await receipt_export_handler.handle(
                body={"format": "json"},
                path_params={"gauntlet_id": "nonexistent"},
            )

        assert result.status_code == 404
        body = parse_response_body(result)
        assert "Gauntlet Not Found" in body["title"]


# ===========================================================================
# GauntletHeatmapExportHandler Tests
# ===========================================================================


class TestGauntletHeatmapExportHandler:
    """Tests for GET /api/v1/gauntlet/{id}/heatmap/export."""

    def test_path_pattern(self, heatmap_export_handler: GauntletHeatmapExportHandler):
        """Test handler path pattern."""
        pattern = heatmap_export_handler.get_path_pattern()
        assert "gauntlet_id" in pattern
        assert "heatmap" in pattern

    @pytest.mark.asyncio
    async def test_export_missing_gauntlet_id(
        self, heatmap_export_handler: GauntletHeatmapExportHandler
    ):
        """Test error when gauntlet ID is missing."""
        result = await heatmap_export_handler.handle(
            body=None,
            path_params={},
        )

        assert result.status_code == 400
        body = parse_response_body(result)
        assert "Missing Gauntlet ID" in body["title"]

    @pytest.mark.asyncio
    async def test_export_gauntlet_not_found(
        self, heatmap_export_handler: GauntletHeatmapExportHandler
    ):
        """Test error when gauntlet run doesn't exist."""
        mock_gauntlet_module = MagicMock()
        mock_gauntlet_module._gauntlet_runs = {}
        mock_storage = MagicMock()
        # Storage without get_result method
        del mock_storage.get_result
        mock_gauntlet_module._get_storage.return_value = mock_storage

        with patch.dict("sys.modules", {"aragora.server.handlers.gauntlet": mock_gauntlet_module}):
            result = await heatmap_export_handler.handle(
                body=None,
                path_params={"gauntlet_id": "nonexistent"},
                query_params={},
            )

        assert result.status_code == 404
        body = parse_response_body(result)
        assert "Gauntlet Not Found" in body["title"]


# ===========================================================================
# Handler Registration Tests
# ===========================================================================


class TestHandlerRegistration:
    """Tests for handler registration utilities."""

    def test_all_handlers_registered(self):
        """Test that all handler classes are in GAUNTLET_V1_HANDLERS."""
        assert len(GAUNTLET_V1_HANDLERS) == 7

        handler_names = [h.__name__ for h in GAUNTLET_V1_HANDLERS]
        assert "GauntletSchemaHandler" in handler_names
        assert "GauntletAllSchemasHandler" in handler_names
        assert "GauntletTemplatesListHandler" in handler_names
        assert "GauntletTemplateHandler" in handler_names
        assert "GauntletReceiptExportHandler" in handler_names
        assert "GauntletHeatmapExportHandler" in handler_names
        assert "GauntletValidateReceiptHandler" in handler_names

    def test_handlers_can_be_instantiated(self, server_context: Dict[str, Any]):
        """Test that all handlers can be instantiated."""
        for handler_cls in GAUNTLET_V1_HANDLERS:
            handler = handler_cls(server_context)
            assert handler is not None
            assert hasattr(handler, "get_path_pattern")
            assert hasattr(handler, "handle")
