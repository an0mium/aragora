"""Document and media endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

DOCUMENT_ENDPOINTS = {
    "/api/documents": {
        "get": {
            "tags": ["Documents"],
            "summary": "List documents",
            "description": "Get list of uploaded documents available for debate context.",
            "operationId": "listDocuments",
            "responses": {"200": _ok_response("Document list")},
        },
    },
    "/api/documents/formats": {
        "get": {
            "tags": ["Documents"],
            "summary": "Supported formats",
            "description": "Get list of supported document formats for upload.",
            "operationId": "listDocumentsFormats",
            "responses": {"200": _ok_response("Supported formats")},
        },
    },
    "/api/documents/upload": {
        "post": {
            "tags": ["Documents"],
            "summary": "Upload document",
            "description": "Upload a document to be used as context in debates.",
            "operationId": "createDocumentsUpload",
            "requestBody": {"content": {"multipart/form-data": {"schema": {"type": "object"}}}},
            "responses": {"201": _ok_response("Document uploaded")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/podcast/feed.xml": {
        "get": {
            "tags": ["Media"],
            "summary": "Podcast RSS feed",
            "description": "Get RSS feed for debate podcast episodes.",
            "operationId": "listPodcastFeed.Xml",
            "responses": {"200": {"description": "RSS feed", "content": {"application/xml": {}}}},
        },
    },
    "/api/podcast/episodes": {
        "get": {
            "tags": ["Media"],
            "summary": "Podcast episodes",
            "description": "Get list of available podcast episodes from debates.",
            "operationId": "listPodcastEpisodes",
            "responses": {"200": _ok_response("Episode list")},
        },
    },
    "/api/youtube/auth": {
        "get": {
            "tags": ["Social"],
            "summary": "YouTube auth URL",
            "description": "Get OAuth authorization URL for YouTube integration.",
            "operationId": "listYoutubeAuth",
            "responses": {"200": _ok_response("Auth URL")},
        },
    },
    "/api/youtube/callback": {
        "get": {
            "tags": ["Social"],
            "summary": "YouTube OAuth callback",
            "description": "Handle OAuth callback from YouTube authorization.",
            "operationId": "listYoutubeCallback",
            "responses": {"200": _ok_response("Auth complete")},
        },
    },
    "/api/youtube/status": {
        "get": {
            "tags": ["Social"],
            "summary": "YouTube auth status",
            "description": "Check current YouTube authorization status.",
            "operationId": "listYoutubeStatus",
            "responses": {"200": _ok_response("Auth status")},
        },
    },
}
