"""Deliberations endpoint definitions for OpenAPI documentation.

Deliberations are real-time vetted decisionmaking sessions where multiple agents
collaborate to reach consensus. These endpoints provide access to active sessions,
statistics, and real-time streaming of deliberation events.
"""

from typing import Any


def _deliberation_schema() -> dict[str, Any]:
    """Deliberation object schema."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Unique deliberation identifier"},
            "task": {"type": "string", "description": "The task being deliberated"},
            "status": {
                "type": "string",
                "enum": ["initializing", "active", "consensus_forming", "complete"],
                "description": "Current deliberation status",
            },
            "agents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Agents participating in the deliberation",
            },
            "current_round": {
                "type": "integer",
                "description": "Current round number",
            },
            "created_at": {
                "type": "string",
                "format": "date-time",
                "description": "When the deliberation started",
            },
            "updated_at": {
                "type": "string",
                "format": "date-time",
                "description": "Last activity timestamp",
            },
        },
    }


def _stats_schema() -> dict[str, Any]:
    """Deliberation statistics schema."""
    return {
        "type": "object",
        "properties": {
            "active_count": {
                "type": "integer",
                "description": "Number of currently active deliberations",
            },
            "completed_today": {
                "type": "integer",
                "description": "Deliberations completed in last 24 hours",
            },
            "average_consensus_time": {
                "type": "number",
                "description": "Average time to reach consensus (seconds)",
            },
            "average_rounds": {
                "type": "number",
                "description": "Average rounds per deliberation",
            },
            "top_agents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string"},
                        "contributions": {"type": "integer"},
                    },
                },
                "description": "Most active agents in recent deliberations",
            },
            "timestamp": {
                "type": "string",
                "format": "date-time",
                "description": "When stats were computed",
            },
        },
    }


DELIBERATIONS_ENDPOINTS = {
    # =========================================================================
    # GET Endpoints - Query and Monitoring
    # =========================================================================
    "/api/v1/deliberations/active": {
        "get": {
            "tags": ["Deliberations"],
            "summary": "List active deliberations",
            "operationId": "listActiveDeliberations",
            "description": "Returns all currently active deliberation sessions with their status and participating agents.",
            "responses": {
                "200": {
                    "description": "List of active deliberations",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "count": {
                                        "type": "integer",
                                        "description": "Number of active deliberations",
                                    },
                                    "deliberations": {
                                        "type": "array",
                                        "items": _deliberation_schema(),
                                    },
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Insufficient permissions"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/deliberations/stats": {
        "get": {
            "tags": ["Deliberations"],
            "summary": "Get deliberation statistics",
            "operationId": "getDeliberationStats",
            "description": "Returns aggregate statistics about deliberations including completion rates, timing metrics, and top contributors.",
            "responses": {
                "200": {
                    "description": "Deliberation statistics",
                    "content": {"application/json": {"schema": _stats_schema()}},
                },
                "401": {"description": "Authentication required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/deliberations/{deliberation_id}": {
        "get": {
            "tags": ["Deliberations"],
            "summary": "Get deliberation details",
            "operationId": "getDeliberation",
            "description": "Returns detailed information about a specific deliberation including messages, votes, and current state.",
            "parameters": [
                {
                    "name": "deliberation_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Unique identifier of the deliberation",
                }
            ],
            "responses": {
                "200": {
                    "description": "Deliberation details",
                    "content": {
                        "application/json": {
                            "schema": {
                                "allOf": [
                                    _deliberation_schema(),
                                    {
                                        "type": "object",
                                        "properties": {
                                            "messages": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "agent": {"type": "string"},
                                                        "content": {"type": "string"},
                                                        "round": {"type": "integer"},
                                                        "timestamp": {
                                                            "type": "string",
                                                            "format": "date-time",
                                                        },
                                                    },
                                                },
                                            },
                                            "votes": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "agent": {"type": "string"},
                                                        "position": {"type": "string"},
                                                        "confidence": {"type": "number"},
                                                    },
                                                },
                                            },
                                        },
                                    },
                                ]
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "404": {"description": "Deliberation not found"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/deliberations/stream": {
        "get": {
            "tags": ["Deliberations"],
            "summary": "Stream deliberation events",
            "operationId": "streamDeliberationEvents",
            "description": "Returns WebSocket connection information for real-time deliberation event streaming. Events include agent messages, votes, consensus detection, and status changes.",
            "responses": {
                "200": {
                    "description": "WebSocket stream configuration",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["websocket"],
                                        "description": "Connection type",
                                    },
                                    "url": {
                                        "type": "string",
                                        "description": "WebSocket URL to connect to",
                                    },
                                    "events": {
                                        "type": "object",
                                        "description": "Available event types",
                                        "properties": {
                                            "agent_message": {
                                                "type": "string",
                                                "description": "Agent sends a message",
                                            },
                                            "vote_cast": {
                                                "type": "string",
                                                "description": "Agent casts a vote",
                                            },
                                            "consensus_reached": {
                                                "type": "string",
                                                "description": "Consensus has been achieved",
                                            },
                                            "status_change": {
                                                "type": "string",
                                                "description": "Deliberation status changed",
                                            },
                                            "round_complete": {
                                                "type": "string",
                                                "description": "A round has completed",
                                            },
                                        },
                                    },
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
}
