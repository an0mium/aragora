"""Monitoring and metrics endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

METRICS_ENDPOINTS = {
    "/api/metrics": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "System metrics",
            "responses": {"200": _ok_response("Metrics data")},
        },
    },
    "/api/metrics/health": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "Metrics health",
            "responses": {"200": _ok_response("Metrics health")},
        },
    },
    "/api/metrics/cache": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "Cache metrics",
            "responses": {"200": _ok_response("Cache metrics")},
        },
    },
    "/api/metrics/system": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "System metrics",
            "responses": {"200": _ok_response("System metrics")},
        },
    },
    "/metrics": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "Prometheus metrics",
            "description": "Metrics in Prometheus format",
            "responses": {
                "200": {
                    "description": "Prometheus-formatted metrics",
                    "content": {"text/plain": {}},
                }
            },
        },
    },
}
