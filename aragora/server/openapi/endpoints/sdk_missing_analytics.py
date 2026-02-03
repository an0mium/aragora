"""SDK missing endpoints for Analytics.

This module contains endpoint definitions for analytics, reporting,
and performance monitoring features.
"""

from aragora.server.openapi.endpoints.sdk_missing_core import (
    _ok_response,
    STANDARD_ERRORS,
)

SDK_MISSING_ANALYTICS_ENDPOINTS: dict = {
    "/api/analytics/agents/{id}/performance": {
        "get": {
            "tags": ["Analytics"],
            "summary": "GET performance",
            "operationId": "getAgentsPerformance",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/analytics/connect": {
        "post": {
            "tags": ["Analytics"],
            "summary": "POST connect",
            "operationId": "postAnalyticsConnect",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/analytics/query": {
        "post": {
            "tags": ["Analytics"],
            "summary": "POST query",
            "operationId": "postAnalyticsQuery",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/analytics/reports/generate": {
        "post": {
            "tags": ["Analytics"],
            "summary": "POST generate",
            "operationId": "postReportsGenerate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/analytics/{id}": {
        "delete": {
            "tags": ["Analytics"],
            "summary": "DELETE {id}",
            "operationId": "deleteAnalytics",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
}

__all__ = ["SDK_MISSING_ANALYTICS_ENDPOINTS"]
