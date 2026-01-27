"""Nomic loop endpoint definitions for OpenAPI documentation.

The Nomic loop is the autonomous self-improvement cycle for the Aragora system.
These endpoints provide control and monitoring of the loop's phases:
Context gathering, Debate, Design, Implementation, and Verification.
"""

from typing import Any


def _nomic_state_schema() -> dict[str, Any]:
    """Nomic loop state object schema."""
    return {
        "type": "object",
        "properties": {
            "running": {"type": "boolean", "description": "Whether the loop is running"},
            "phase": {
                "type": "string",
                "enum": ["context", "debate", "design", "implement", "verify", "idle"],
                "description": "Current phase of the nomic loop",
            },
            "cycle": {"type": "integer", "description": "Current cycle number"},
            "paused": {"type": "boolean", "description": "Whether the loop is paused"},
            "last_activity": {
                "type": "string",
                "format": "date-time",
                "description": "Timestamp of last activity",
            },
            "active_task": {
                "type": "string",
                "nullable": True,
                "description": "Current task being executed",
            },
        },
    }


def _nomic_health_schema() -> dict[str, Any]:
    """Nomic loop health object schema."""
    return {
        "type": "object",
        "properties": {
            "healthy": {"type": "boolean", "description": "Overall health status"},
            "stall_detected": {
                "type": "boolean",
                "description": "Whether a stall has been detected",
            },
            "time_in_phase_seconds": {
                "type": "number",
                "description": "Time spent in current phase",
            },
            "max_phase_duration_seconds": {
                "type": "number",
                "description": "Maximum allowed phase duration",
            },
            "memory_usage_mb": {"type": "number", "description": "Memory usage in MB"},
            "cpu_percent": {"type": "number", "description": "CPU utilization percentage"},
        },
    }


def _proposal_schema() -> dict[str, Any]:
    """Nomic proposal object schema."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Unique proposal identifier"},
            "title": {"type": "string", "description": "Proposal title"},
            "description": {"type": "string", "description": "Detailed proposal description"},
            "status": {
                "type": "string",
                "enum": ["pending", "approved", "rejected", "implemented"],
                "description": "Current proposal status",
            },
            "author": {"type": "string", "description": "Agent that created the proposal"},
            "created_at": {"type": "string", "format": "date-time"},
            "votes": {
                "type": "object",
                "properties": {
                    "approve": {"type": "integer"},
                    "reject": {"type": "integer"},
                },
            },
        },
    }


NOMIC_ENDPOINTS = {
    # =========================================================================
    # GET Endpoints - State and Monitoring
    # =========================================================================
    "/api/v1/nomic/state": {
        "get": {
            "tags": ["Nomic"],
            "summary": "Get nomic loop state",
            "operationId": "getNomicState",
            "description": "Returns the current state of the nomic self-improvement loop, including phase, cycle count, and activity status.",
            "responses": {
                "200": {
                    "description": "Current nomic loop state",
                    "content": {"application/json": {"schema": _nomic_state_schema()}},
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Insufficient permissions"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/health": {
        "get": {
            "tags": ["Nomic"],
            "summary": "Get nomic loop health",
            "operationId": "getNomicHealth",
            "description": "Returns health metrics for the nomic loop including stall detection, resource usage, and phase timing.",
            "responses": {
                "200": {
                    "description": "Nomic loop health metrics",
                    "content": {"application/json": {"schema": _nomic_health_schema()}},
                },
                "401": {"description": "Authentication required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/metrics": {
        "get": {
            "tags": ["Nomic"],
            "summary": "Get nomic loop metrics",
            "operationId": "getNomicMetrics",
            "description": "Returns Prometheus-style metrics summary for the nomic loop including cycle counts, phase durations, and success rates.",
            "responses": {
                "200": {
                    "description": "Prometheus metrics summary",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "cycles_completed": {"type": "integer"},
                                    "cycles_failed": {"type": "integer"},
                                    "avg_cycle_duration_seconds": {"type": "number"},
                                    "proposals_generated": {"type": "integer"},
                                    "proposals_approved": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/log": {
        "get": {
            "tags": ["Nomic"],
            "summary": "Get nomic loop logs",
            "operationId": "getNomicLogs",
            "description": "Returns recent log entries from the nomic loop with optional filtering by level and phase.",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 100},
                    "description": "Maximum number of log entries to return",
                },
                {
                    "name": "level",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["debug", "info", "warning", "error"]},
                    "description": "Filter by log level",
                },
                {
                    "name": "phase",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by nomic phase",
                },
            ],
            "responses": {
                "200": {
                    "description": "Log entries",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "entries": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "timestamp": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                                "level": {"type": "string"},
                                                "message": {"type": "string"},
                                                "phase": {"type": "string"},
                                            },
                                        },
                                    },
                                    "total": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/risk-register": {
        "get": {
            "tags": ["Nomic"],
            "summary": "Get risk register",
            "operationId": "getNomicRiskRegister",
            "description": "Returns entries from the nomic loop risk register, tracking potential issues and mitigations.",
            "responses": {
                "200": {
                    "description": "Risk register entries",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "risks": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "category": {"type": "string"},
                                                "severity": {
                                                    "type": "string",
                                                    "enum": ["low", "medium", "high", "critical"],
                                                },
                                                "description": {"type": "string"},
                                                "mitigation": {"type": "string"},
                                                "status": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            }
                        }
                    },
                },
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/proposals": {
        "get": {
            "tags": ["Nomic"],
            "summary": "List nomic proposals",
            "operationId": "listNomicProposals",
            "description": "Returns all proposals generated by the nomic loop, with optional status filtering.",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "schema": {
                        "type": "string",
                        "enum": ["pending", "approved", "rejected", "implemented"],
                    },
                    "description": "Filter by proposal status",
                },
            ],
            "responses": {
                "200": {
                    "description": "List of proposals",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "proposals": {
                                        "type": "array",
                                        "items": _proposal_schema(),
                                    },
                                    "total": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/witness/status": {
        "get": {
            "tags": ["Nomic", "Gas Town"],
            "summary": "Get witness patrol status",
            "operationId": "getWitnessStatus",
            "description": "Returns the status of the Gas Town witness patrol, which monitors agent behavior and system integrity.",
            "responses": {
                "200": {
                    "description": "Witness patrol status",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "active": {"type": "boolean"},
                                    "last_patrol": {"type": "string", "format": "date-time"},
                                    "anomalies_detected": {"type": "integer"},
                                    "agents_monitored": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/mayor/current": {
        "get": {
            "tags": ["Nomic", "Gas Town"],
            "summary": "Get current mayor",
            "operationId": "getCurrentMayor",
            "description": "Returns information about the current Gas Town mayor, the elected leader coordinating multi-agent operations.",
            "responses": {
                "200": {
                    "description": "Current mayor information",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "node_id": {"type": "string"},
                                    "elected_at": {"type": "string", "format": "date-time"},
                                    "term_expires": {"type": "string", "format": "date-time"},
                                    "region": {"type": "string"},
                                },
                            }
                        }
                    },
                },
            },
            "security": [{"bearerAuth": []}],
        }
    },
    # =========================================================================
    # POST Endpoints - Control Operations
    # =========================================================================
    "/api/v1/nomic/control/start": {
        "post": {
            "tags": ["Nomic"],
            "summary": "Start nomic loop",
            "operationId": "startNomicLoop",
            "description": "Starts the nomic self-improvement loop. Requires nomic:control permission.",
            "responses": {
                "200": {
                    "description": "Loop started successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "message": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Insufficient permissions (nomic:control required)"},
                "409": {"description": "Loop already running"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/control/stop": {
        "post": {
            "tags": ["Nomic"],
            "summary": "Stop nomic loop",
            "operationId": "stopNomicLoop",
            "description": "Gracefully stops the nomic loop after completing the current phase.",
            "responses": {
                "200": {
                    "description": "Loop stopped successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "message": {"type": "string"},
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
    "/api/v1/nomic/control/pause": {
        "post": {
            "tags": ["Nomic"],
            "summary": "Pause nomic loop",
            "operationId": "pauseNomicLoop",
            "description": "Pauses the nomic loop at the current position, allowing for inspection or manual intervention.",
            "responses": {
                "200": {
                    "description": "Loop paused successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "paused_at_phase": {"type": "string"},
                                },
                            }
                        }
                    },
                },
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/control/resume": {
        "post": {
            "tags": ["Nomic"],
            "summary": "Resume nomic loop",
            "operationId": "resumeNomicLoop",
            "description": "Resumes a paused nomic loop from where it was paused.",
            "responses": {
                "200": {
                    "description": "Loop resumed successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "resumed_at_phase": {"type": "string"},
                                },
                            }
                        }
                    },
                },
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/control/skip-phase": {
        "post": {
            "tags": ["Nomic"],
            "summary": "Skip current phase",
            "operationId": "skipNomicPhase",
            "description": "Skips the current phase and advances to the next phase in the nomic loop.",
            "responses": {
                "200": {
                    "description": "Phase skipped successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "skipped_phase": {"type": "string"},
                                    "new_phase": {"type": "string"},
                                },
                            }
                        }
                    },
                },
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/proposals/approve": {
        "post": {
            "tags": ["Nomic"],
            "summary": "Approve proposal",
            "operationId": "approveNomicProposal",
            "description": "Approves a pending nomic proposal for implementation.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "proposal_id": {
                                    "type": "string",
                                    "description": "ID of the proposal to approve",
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Optional approval reason",
                                },
                            },
                            "required": ["proposal_id"],
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Proposal approved",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "proposal": _proposal_schema(),
                                },
                            }
                        }
                    },
                },
                "404": {"description": "Proposal not found"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/nomic/proposals/reject": {
        "post": {
            "tags": ["Nomic"],
            "summary": "Reject proposal",
            "operationId": "rejectNomicProposal",
            "description": "Rejects a pending nomic proposal.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "proposal_id": {
                                    "type": "string",
                                    "description": "ID of the proposal to reject",
                                },
                                "reason": {"type": "string", "description": "Rejection reason"},
                            },
                            "required": ["proposal_id", "reason"],
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Proposal rejected",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "proposal": _proposal_schema(),
                                },
                            }
                        }
                    },
                },
                "404": {"description": "Proposal not found"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
}
