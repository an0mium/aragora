"""Cross-pollination endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

CROSS_POLLINATION_ENDPOINTS = {
    "/api/cross-pollination/stats": {
        "get": {
            "tags": ["Cross-Pollination"],
            "summary": "Get subscriber statistics",
            "operationId": "listCrossPollinationStats",
            "description": (
                "Get statistics for all cross-subsystem event subscribers including "
                "event counts, success/failure rates, circuit breaker status, and latency metrics."
            ),
            "responses": {
                "200": _ok_response("Subscriber statistics with totals and per-handler metrics"),
            },
        },
    },
    "/api/cross-pollination/subscribers": {
        "get": {
            "tags": ["Cross-Pollination"],
            "summary": "List all subscribers",
            "operationId": "listCrossPollinationSubscribers",
            "description": (
                "Get a list of all registered cross-subsystem event subscribers "
                "with their subscribed event types and handler metadata."
            ),
            "responses": {
                "200": _ok_response("List of subscribers with event types and descriptions"),
            },
        },
    },
    "/api/cross-pollination/bridge": {
        "get": {
            "tags": ["Cross-Pollination"],
            "summary": "Get arena bridge status",
            "operationId": "listCrossPollinationBridge",
            "description": (
                "Get status of the Arena-to-CrossSubscriber event bridge including "
                "connection state, event mappings, and throughput statistics."
            ),
            "responses": {
                "200": _ok_response("Bridge status with connection state and event mappings"),
            },
        },
    },
    "/api/cross-pollination/metrics": {
        "get": {
            "tags": ["Cross-Pollination"],
            "summary": "Get Prometheus metrics",
            "operationId": "listCrossPollinationMetrics",
            "description": (
                "Get cross-pollination metrics in Prometheus/OpenMetrics text format. "
                "Includes event counts, handler durations, and circuit breaker states."
            ),
            "responses": {
                "200": {
                    "description": "Prometheus-format metrics text",
                    "content": {"text/plain": {}},
                },
            },
        },
    },
    "/api/cross-pollination/reset": {
        "post": {
            "tags": ["Cross-Pollination"],
            "summary": "Reset statistics",
            "operationId": "createCrossPollinationReset",
            "description": (
                "Reset all cross-pollination statistics counters and circuit breaker states. "
                "Useful for testing and debugging."
            ),
            "responses": {
                "200": _ok_response("Statistics reset confirmation with handler count"),
            },
        },
    },
    "/api/cross-pollination/handlers/{handler_name}/circuit-breaker": {
        "get": {
            "tags": ["Cross-Pollination"],
            "summary": "Get handler circuit breaker status",
            "operationId": "getCrossPollinationHandlersCircuitBreaker",
            "description": "Get circuit breaker state for a specific handler including failure count and cooldown.",
            "parameters": [
                {
                    "name": "handler_name",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Name of the handler (e.g., 'memory_to_rlm')",
                },
            ],
            "responses": {
                "200": _ok_response("Circuit breaker status with state and failure count"),
                "404": {"description": "Handler not found"},
            },
        },
        "post": {
            "tags": ["Cross-Pollination"],
            "summary": "Reset handler circuit breaker",
            "operationId": "createCrossPollinationHandlersCircuitBreaker",
            "description": "Reset the circuit breaker for a specific handler, clearing failure count and closing circuit.",
            "parameters": [
                {
                    "name": "handler_name",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Name of the handler to reset",
                },
            ],
            "responses": {
                "200": _ok_response("Circuit breaker reset confirmation"),
                "404": {"description": "Handler not found"},
            },
        },
    },
    "/api/cross-pollination/km": {
        "get": {
            "tags": ["Cross-Pollination"],
            "summary": "Get Knowledge Mound integration status",
            "operationId": "listCrossPollinationKm",
            "description": "Get status of cross-pollination integration with the Knowledge Mound including sync state and adapter health.",
            "responses": {
                "200": _ok_response("KM integration status with sync state and adapter health"),
            },
        },
    },
    "/api/cross-pollination/km/sync": {
        "post": {
            "tags": ["Cross-Pollination"],
            "summary": "Trigger Knowledge Mound sync",
            "operationId": "createCrossPollinationKmSync",
            "description": "Manually trigger synchronization of cross-pollination data to the Knowledge Mound.",
            "responses": {
                "200": _ok_response("Sync triggered with job ID"),
                "503": {"description": "Knowledge Mound unavailable"},
            },
        },
    },
    "/api/cross-pollination/km/staleness-check": {
        "get": {
            "tags": ["Cross-Pollination"],
            "summary": "Check data staleness",
            "operationId": "listCrossPollinationKmStalenessCheck",
            "description": "Check for stale cross-pollination data in the Knowledge Mound that needs revalidation.",
            "responses": {
                "200": _ok_response("Staleness report with entries needing refresh"),
            },
        },
    },
    "/api/cross-pollination/km/culture": {
        "get": {
            "tags": ["Cross-Pollination"],
            "summary": "Get debate culture patterns",
            "operationId": "listCrossPollinationKmCulture",
            "description": "Get learned debate culture patterns from cross-pollination analysis across debates.",
            "responses": {
                "200": _ok_response("Culture patterns with collaboration and dissent metrics"),
            },
        },
    },
}
