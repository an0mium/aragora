"""Document and media endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

DOCUMENT_ENDPOINTS = {
    "/api/documents": {
        "get": {
            "tags": ["Documents"],
            "summary": "List documents",
            "responses": {"200": _ok_response("Document list")},
        },
    },
    "/api/documents/formats": {
        "get": {
            "tags": ["Documents"],
            "summary": "Supported formats",
            "responses": {"200": _ok_response("Supported formats")},
        },
    },
    "/api/documents/upload": {
        "post": {
            "tags": ["Documents"],
            "summary": "Upload document",
            "requestBody": {"content": {"multipart/form-data": {"schema": {"type": "object"}}}},
            "responses": {"201": _ok_response("Document uploaded")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/podcast/feed.xml": {
        "get": {
            "tags": ["Media"],
            "summary": "Podcast RSS feed",
            "responses": {"200": {"description": "RSS feed", "content": {"application/xml": {}}}},
        },
    },
    "/api/podcast/episodes": {
        "get": {
            "tags": ["Media"],
            "summary": "Podcast episodes",
            "responses": {"200": _ok_response("Episode list")},
        },
    },
    "/api/youtube/auth": {
        "get": {
            "tags": ["Social"],
            "summary": "YouTube auth URL",
            "responses": {"200": _ok_response("Auth URL")},
        },
    },
    "/api/youtube/callback": {
        "get": {
            "tags": ["Social"],
            "summary": "YouTube OAuth callback",
            "responses": {"200": _ok_response("Auth complete")},
        },
    },
    "/api/youtube/status": {
        "get": {
            "tags": ["Social"],
            "summary": "YouTube auth status",
            "responses": {"200": _ok_response("Auth status")},
        },
    },
}
