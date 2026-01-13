"""
Tests for OpenAPI and Postman collection generation.

Tests cover:
- OpenAPI schema generation
- Postman collection conversion
- Endpoint to Postman request mapping
- File export functionality
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from aragora.server.openapi import (
    generate_openapi_schema,
    generate_postman_collection,
    save_postman_collection,
    get_postman_json,
    handle_postman_request,
    _openapi_to_postman_request,
    get_endpoint_count,
)


# ============================================================================
# OpenAPI Schema Tests
# ============================================================================


class TestOpenAPISchema:
    """Tests for OpenAPI schema generation."""

    def test_generate_schema_structure(self):
        """Test schema has required OpenAPI 3.0 structure."""
        schema = generate_openapi_schema()

        assert schema["openapi"] == "3.0.3"
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema

    def test_schema_info(self):
        """Test schema info section."""
        schema = generate_openapi_schema()

        assert schema["info"]["title"] == "Aragora API"
        assert "version" in schema["info"]
        assert "description" in schema["info"]

    def test_schema_has_tags(self):
        """Test schema has tag definitions."""
        schema = generate_openapi_schema()

        assert "tags" in schema
        assert len(schema["tags"]) > 0

        # Check tag structure
        for tag in schema["tags"]:
            assert "name" in tag
            assert "description" in tag

    def test_schema_has_paths(self):
        """Test schema has endpoint paths."""
        schema = generate_openapi_schema()

        assert len(schema["paths"]) > 0

        # Check path structure
        for path, methods in schema["paths"].items():
            assert path.startswith("/")
            for method in methods:
                if method not in ("parameters", "servers"):
                    assert method in ("get", "post", "put", "delete", "patch")

    def test_schema_components(self):
        """Test schema has component schemas."""
        schema = generate_openapi_schema()

        assert "schemas" in schema["components"]
        assert "Error" in schema["components"]["schemas"]

    def test_get_endpoint_count(self):
        """Test endpoint counting."""
        count = get_endpoint_count()

        # Should have many endpoints
        assert count > 50


# ============================================================================
# Postman Collection Tests
# ============================================================================


class TestPostmanCollection:
    """Tests for Postman collection generation."""

    def test_collection_structure(self):
        """Test collection has valid Postman v2.1 structure."""
        collection = generate_postman_collection()

        assert "info" in collection
        assert "item" in collection
        assert (
            collection["info"]["schema"]
            == "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        )

    def test_collection_info(self):
        """Test collection info section."""
        collection = generate_postman_collection()

        assert collection["info"]["name"] == "Aragora API"
        assert "_postman_id" in collection["info"]
        assert "version" in collection["info"]

    def test_collection_variables(self):
        """Test collection has environment variables."""
        collection = generate_postman_collection()

        assert "variable" in collection
        var_keys = [v["key"] for v in collection["variable"]]
        assert "base_url" in var_keys
        assert "api_token" in var_keys

    def test_collection_auth(self):
        """Test collection has auth configuration."""
        collection = generate_postman_collection()

        assert "auth" in collection
        assert collection["auth"]["type"] == "bearer"

    def test_collection_folders(self):
        """Test collection has organized folders."""
        collection = generate_postman_collection()

        # Should have multiple folders (tags)
        assert len(collection["item"]) > 5

        # Each folder should have items
        for folder in collection["item"]:
            assert "name" in folder
            assert "item" in folder
            assert isinstance(folder["item"], list)

    def test_collection_request_count(self):
        """Test collection has correct request count."""
        collection = generate_postman_collection()

        request_count = sum(len(folder.get("item", [])) for folder in collection.get("item", []))

        # Should match OpenAPI endpoint count
        assert request_count > 50

    def test_custom_base_url(self):
        """Test collection accepts custom base URL."""
        collection = generate_postman_collection(base_url="https://api.example.com")

        # Check a request URL uses the custom base
        first_folder = collection["item"][0]
        if first_folder["item"]:
            first_request = first_folder["item"][0]
            assert "https://api.example.com" in first_request["request"]["url"]["raw"]


# ============================================================================
# Request Conversion Tests
# ============================================================================


class TestRequestConversion:
    """Tests for OpenAPI to Postman request conversion."""

    def test_basic_get_request(self):
        """Test conversion of simple GET endpoint."""
        details = {
            "summary": "Get health status",
            "description": "Check system health",
            "tags": ["System"],
            "responses": {"200": {"description": "OK"}},
        }

        request = _openapi_to_postman_request(
            path="/api/health",
            method="GET",
            details=details,
            base_url="{{base_url}}",
        )

        assert request["name"] == "Get health status"
        assert request["request"]["method"] == "GET"
        assert "{{base_url}}" in request["request"]["url"]["raw"]
        assert request["request"]["description"] == "Check system health"

    def test_path_parameters(self):
        """Test path parameter conversion."""
        details = {
            "summary": "Get debate by ID",
            "parameters": [{"name": "id", "in": "path", "required": True}],
        }

        request = _openapi_to_postman_request(
            path="/api/debates/{id}",
            method="GET",
            details=details,
            base_url="{{base_url}}",
        )

        # Should convert {id} to :id
        assert ":id" in request["request"]["url"]["raw"]
        assert "variable" in request["request"]["url"]
        assert any(v["key"] == "id" for v in request["request"]["url"]["variable"])

    def test_query_parameters(self):
        """Test query parameter conversion."""
        details = {
            "summary": "List debates",
            "parameters": [
                {"name": "limit", "in": "query", "required": False, "description": "Max results"},
                {"name": "offset", "in": "query", "required": False},
            ],
        }

        request = _openapi_to_postman_request(
            path="/api/debates",
            method="GET",
            details=details,
            base_url="{{base_url}}",
        )

        assert "query" in request["request"]["url"]
        query_keys = [q["key"] for q in request["request"]["url"]["query"]]
        assert "limit" in query_keys
        assert "offset" in query_keys

    def test_post_request_with_body(self):
        """Test POST request with request body."""
        details = {
            "summary": "Create debate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
        }

        request = _openapi_to_postman_request(
            path="/api/debates",
            method="POST",
            details=details,
            base_url="{{base_url}}",
        )

        assert request["request"]["method"] == "POST"
        assert "body" in request["request"]
        assert request["request"]["body"]["mode"] == "raw"

    def test_request_headers(self):
        """Test request includes standard headers."""
        details = {"summary": "Test endpoint"}

        request = _openapi_to_postman_request(
            path="/api/test",
            method="GET",
            details=details,
            base_url="{{base_url}}",
        )

        headers = {h["key"]: h["value"] for h in request["request"]["header"]}
        assert headers.get("Content-Type") == "application/json"
        assert headers.get("Accept") == "application/json"


# ============================================================================
# Export Tests
# ============================================================================


class TestExport:
    """Tests for file export functionality."""

    def test_get_postman_json(self):
        """Test JSON string generation."""
        json_str = get_postman_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert "info" in data
        assert "item" in data

    def test_handle_postman_request(self):
        """Test HTTP handler function."""
        content, content_type = handle_postman_request()

        assert content_type == "application/json"
        assert len(content) > 1000

        # Should be valid JSON
        data = json.loads(content)
        assert "info" in data

    def test_save_postman_collection(self, tmp_path):
        """Test saving collection to file."""
        output_path = tmp_path / "test.postman_collection.json"

        # Save collection using the function
        path_str, count = save_postman_collection(str(output_path))

        # Verify file was created
        assert output_path.exists()
        assert count > 50

        # Verify content is valid
        with open(output_path) as f:
            data = json.load(f)
        assert "info" in data
        assert "item" in data


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_openapi_to_postman_roundtrip(self):
        """Test that all OpenAPI endpoints are in Postman collection."""
        schema = generate_openapi_schema()
        collection = generate_postman_collection()

        # Count endpoints in each
        openapi_count = sum(
            1
            for methods in schema["paths"].values()
            for method in methods
            if method not in ("parameters", "servers")
        )

        postman_count = sum(len(folder.get("item", [])) for folder in collection.get("item", []))

        # Should match
        assert postman_count == openapi_count

    def test_all_tags_represented(self):
        """Test all OpenAPI tags have Postman folders."""
        schema = generate_openapi_schema()
        collection = generate_postman_collection()

        openapi_tags = {t["name"] for t in schema.get("tags", [])}
        postman_folders = {f["name"] for f in collection.get("item", [])}

        # Each OpenAPI tag with endpoints should have a folder
        for path, methods in schema["paths"].items():
            for method, details in methods.items():
                if method in ("parameters", "servers"):
                    continue
                tags = details.get("tags", [])
                if tags:
                    # At least one tag should have a folder
                    assert any(t in postman_folders for t in tags)

    def test_collection_is_importable(self):
        """Test collection can be imported into Postman (valid schema)."""
        collection = generate_postman_collection()

        # Required Postman v2.1 fields
        assert collection["info"]["schema"].startswith("https://schema.getpostman.com")
        assert "item" in collection

        # Validate each request
        for folder in collection["item"]:
            for request in folder.get("item", []):
                assert "name" in request
                assert "request" in request
                assert "method" in request["request"]
                assert "url" in request["request"]
