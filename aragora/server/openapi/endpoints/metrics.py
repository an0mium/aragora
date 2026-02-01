"""Monitoring and metrics endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

METRICS_ENDPOINTS = {
    "/api/metrics": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "System metrics",
            "description": "Get comprehensive system metrics for monitoring.",
            "operationId": "listMetrics",
            "responses": {
                "200": _ok_response(
                    "Metrics data",
                    {
                        "metrics": {"type": "object", "additionalProperties": {"type": "number"}},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "collection_duration_ms": {"type": "number"},
                    },
                )
            },
        },
    },
    "/api/metrics/health": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "Metrics health",
            "description": "Get health status of the metrics collection system.",
            "operationId": "listMetricsHealth",
            "responses": {
                "200": _ok_response(
                    "Metrics health",
                    {
                        "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                        "collectors": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "last_collection": {"type": "string", "format": "date-time"},
                                },
                            },
                        },
                    },
                )
            },
        },
    },
    "/api/metrics/cache": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "Cache metrics",
            "description": "Get cache hit rates and performance metrics.",
            "operationId": "listMetricsCache",
            "responses": {
                "200": _ok_response(
                    "Cache metrics",
                    {
                        "hit_rate": {"type": "number"},
                        "miss_rate": {"type": "number"},
                        "total_requests": {"type": "integer"},
                        "total_hits": {"type": "integer"},
                        "total_misses": {"type": "integer"},
                        "size_bytes": {"type": "integer"},
                        "max_size_bytes": {"type": "integer"},
                    },
                )
            },
        },
    },
    "/api/metrics/system": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "System metrics",
            "description": "Get system-level metrics including CPU, memory, and disk usage.",
            "operationId": "listMetricsSystem",
            "responses": {
                "200": _ok_response(
                    "System metrics",
                    {
                        "cpu_percent": {"type": "number"},
                        "memory_mb": {"type": "number"},
                        "memory_percent": {"type": "number"},
                        "disk_percent": {"type": "number"},
                        "open_file_descriptors": {"type": "integer"},
                        "thread_count": {"type": "integer"},
                        "uptime_seconds": {"type": "number"},
                    },
                )
            },
        },
    },
    "/metrics": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "Prometheus metrics",
            "operationId": "getPrometheusMetrics",
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
