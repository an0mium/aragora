"""Tests for Workflow Templates namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# List Templates Operations
# =========================================================================

class TestWorkflowTemplatesList:
    """Tests for list workflow templates operations."""

    def test_list_templates_default(self) -> None:
        """List templates with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "templates": [
                    {
                        "id": "tpl_001",
                        "name": "Document Analysis",
                        "description": "Analyze documents with multiple agents",
                        "category": "analysis",
                    },
                    {
                        "id": "tpl_002",
                        "name": "Code Review",
                        "description": "Multi-agent code review workflow",
                        "category": "development",
                    },
                ],
                "total": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.workflow_templates.list()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/workflow/templates", params={"limit": 50, "offset": 0}
            )
            assert result["total"] == 2
            assert result["templates"][0]["name"] == "Document Analysis"
            client.close()

    def test_list_templates_with_category(self) -> None:
        """List templates filtered by category."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "templates": [{"id": "tpl_003", "category": "analysis"}],
                "total": 1,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.workflow_templates.list(category="analysis")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["category"] == "analysis"
            client.close()

    def test_list_templates_with_all_filters(self) -> None:
        """List templates with all filters applied."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"templates": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.workflow_templates.list(
                category="review",
                pattern="sequential",
                search="code",
                tags=["security", "automated"],
                limit=25,
                offset=10,
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["category"] == "review"
            assert params["pattern"] == "sequential"
            assert params["search"] == "code"
            assert params["tags"] == ["security", "automated"]
            assert params["limit"] == 25
            assert params["offset"] == 10
            client.close()

# =========================================================================
# Get Template Operations
# =========================================================================

class TestWorkflowTemplatesGet:
    """Tests for get workflow template operations."""

    def test_get_template(self) -> None:
        """Get a specific template by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "tpl_123",
                "name": "Security Audit",
                "description": "Comprehensive security audit workflow",
                "category": "security",
                "pattern": "parallel",
                "tags": ["security", "audit", "compliance"],
                "version": "1.2.0",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-15T12:00:00Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.workflow_templates.get("tpl_123")

            mock_request.assert_called_once_with("GET", "/api/v1/workflow/templates/tpl_123")
            assert result["id"] == "tpl_123"
            assert result["name"] == "Security Audit"
            assert result["pattern"] == "parallel"
            client.close()

# =========================================================================
# Get Template Package Operations
# =========================================================================

class TestWorkflowTemplatesPackage:
    """Tests for get template package operations."""

    def test_get_package(self) -> None:
        """Get the full template package."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "template": {
                    "id": "tpl_456",
                    "name": "Data Pipeline",
                    "category": "data",
                },
                "definition": {
                    "nodes": [
                        {"id": "input", "type": "input"},
                        {"id": "process", "type": "debate"},
                        {"id": "output", "type": "output"},
                    ],
                    "edges": [
                        {"from": "input", "to": "process"},
                        {"from": "process", "to": "output"},
                    ],
                },
                "dependencies": ["pandas", "numpy"],
                "examples": [{"name": "Basic usage", "inputs": {"data": "sample.csv"}}],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.workflow_templates.get_package("tpl_456")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/workflow/templates/tpl_456/package"
            )
            assert result["template"]["name"] == "Data Pipeline"
            assert len(result["definition"]["nodes"]) == 3
            assert result["dependencies"] == ["pandas", "numpy"]
            client.close()

# =========================================================================
# Run Template Operations
# =========================================================================

class TestAsyncWorkflowTemplates:
    """Tests for async Workflow Templates API."""

    @pytest.mark.asyncio
    async def test_async_list_templates(self) -> None:
        """List templates asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"templates": [], "total": 0}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.workflow_templates.list()

                mock_request.assert_called_once_with(
                    "GET",
                    "/api/v1/workflow/templates",
                    params={"limit": 50, "offset": 0},
                )
                assert "templates" in result

    @pytest.mark.asyncio
    async def test_async_list_templates_with_filters(self) -> None:
        """List templates with filters asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"templates": [], "total": 0}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.workflow_templates.list(category="automation", search="email")

                call_args = mock_request.call_args
                params = call_args[1]["params"]
                assert params["category"] == "automation"
                assert params["search"] == "email"

    @pytest.mark.asyncio
    async def test_async_get_template(self) -> None:
        """Get a template asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "tpl_async", "name": "Async Template"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.workflow_templates.get("tpl_async")

                mock_request.assert_called_once_with("GET", "/api/v1/workflow/templates/tpl_async")
                assert result["name"] == "Async Template"

    @pytest.mark.asyncio
    async def test_async_get_package(self) -> None:
        """Get template package asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "template": {"id": "tpl_pkg"},
                "definition": {},
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.workflow_templates.get_package("tpl_pkg")

                mock_request.assert_called_once_with(
                    "GET", "/api/v1/workflow/templates/tpl_pkg/package"
                )
                assert "definition" in result

