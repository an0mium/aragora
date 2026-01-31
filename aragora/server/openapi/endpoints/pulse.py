"""Pulse and trending endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

PULSE_ENDPOINTS = {
    "/api/pulse/trending": {
        "get": {
            "tags": ["Pulse"],
            "summary": "Trending topics",
            "operationId": "listPulseTrending",
            "description": "Get current trending debate topics",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 10, "maximum": 50},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Trending topics",
                    {
                        "topics": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "score": {"type": "number"},
                                },
                            },
                        }
                    },
                )
            },
        },
    },
    "/api/pulse/suggest": {
        "get": {
            "tags": ["Pulse"],
            "summary": "Suggest debate topic",
            "operationId": "listPulseSuggest",
            "description": "Get AI-suggested debate topic based on trends",
            "parameters": [{"name": "category", "in": "query", "schema": {"type": "string"}}],
            "responses": {
                "200": _ok_response(
                    "Suggested topic",
                    {
                        "topic": {"type": "string"},
                        "category": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                )
            },
        },
    },
}
