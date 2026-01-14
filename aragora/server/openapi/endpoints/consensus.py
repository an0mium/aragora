"""Consensus endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

CONSENSUS_ENDPOINTS = {
    "/api/consensus/similar": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Similar debates",
            "description": "Find debates similar to a given topic",
            "parameters": [
                {"name": "topic", "in": "query", "required": True, "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
            ],
            "responses": {"200": _ok_response("Similar debates")},
        },
    },
    "/api/consensus/settled": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Settled questions",
            "parameters": [
                {"name": "threshold", "in": "query", "schema": {"type": "number", "default": 0.8}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            ],
            "responses": {"200": _ok_response("Settled questions")},
        },
    },
    "/api/consensus/stats": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Consensus statistics",
            "responses": {"200": _ok_response("Consensus stats")},
        },
    },
    "/api/consensus/dissents": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Dissenting views",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}}
            ],
            "responses": {"200": _ok_response("Dissenting views")},
        },
    },
    "/api/consensus/contrarian-views": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Contrarian views",
            "responses": {"200": _ok_response("Contrarian views")},
        },
    },
    "/api/consensus/risk-warnings": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Risk warnings",
            "responses": {"200": _ok_response("Risk warnings")},
        },
    },
    "/api/consensus/domain/{domain}": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Domain consensus",
            "parameters": [
                {"name": "domain", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Domain consensus data")},
        },
    },
}
