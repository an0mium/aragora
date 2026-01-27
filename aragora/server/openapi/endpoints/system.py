"""System endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

SYSTEM_ENDPOINTS = {
    "/api/health": {
        "get": {
            "tags": ["System"],
            "summary": "Health check",
            "description": """Get system health status for load balancers and monitoring.

**Returns:** 200 when healthy, 503 when degraded.

**Use cases:**
- Load balancer health probes
- Kubernetes liveness/readiness checks
- Uptime monitoring services""",
            "operationId": "getHealth",
            "responses": {
                "200": _ok_response("System healthy", "HealthCheck"),
                "503": {"description": "System degraded or unhealthy"},
            },
        },
    },
    "/api/health/detailed": {
        "get": {
            "tags": ["System"],
            "summary": "Detailed health check",
            "description": """Get detailed health status with component checks.

**Response includes:**
- Database connectivity status
- Redis/cache availability
- Memory usage statistics
- Active connection counts
- Observer metrics
- Recent error counts""",
            "operationId": "getDetailedHealth",
            "responses": {"200": _ok_response("Detailed health information")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/nomic/state": {
        "get": {
            "tags": ["System"],
            "summary": "Get nomic loop state",
            "description": """Get current state of the nomic self-improvement loop.

**Response includes:**
- Current phase (debate, design, implement, verify)
- Active cycle ID
- Last successful cycle timestamp
- Pending improvements queue""",
            "operationId": "getNomicState",
            "responses": {"200": _ok_response("Nomic state")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/nomic/health": {
        "get": {
            "tags": ["System"],
            "summary": "Nomic loop health",
            "description": """Get nomic loop health with stall detection.

**Stall detection:** Alerts if the nomic loop hasn't progressed within expected timeframes.

**Response includes:**
- Health status (healthy, stalled, degraded)
- Time since last activity
- Phase duration statistics""",
            "operationId": "getNomicHealth",
            "responses": {"200": _ok_response("Nomic health status")},
        },
    },
    "/api/nomic/log": {
        "get": {
            "tags": ["System"],
            "summary": "Get nomic logs",
            "description": """Get recent nomic loop log lines for debugging and monitoring.

**Authentication:** Required. Admin access recommended.

**Log levels:** DEBUG, INFO, WARNING, ERROR included.""",
            "operationId": "getNomicLogs",
            "parameters": [
                {
                    "name": "lines",
                    "in": "query",
                    "description": "Number of log lines to return",
                    "schema": {"type": "integer", "default": 100, "minimum": 1, "maximum": 1000},
                },
            ],
            "responses": {"200": _ok_response("Log lines")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/nomic/risk-register": {
        "get": {
            "tags": ["System"],
            "summary": "Risk register",
            "description": """Get risk register entries from nomic loop execution.

**Risk categories:**
- Code changes that could break existing functionality
- Security-sensitive modifications
- Performance-impacting changes
- External dependency updates""",
            "operationId": "getRiskRegister",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of risk entries to return",
                    "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
                },
            ],
            "responses": {"200": _ok_response("Risk entries")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/nomic/metrics": {
        "get": {
            "tags": ["System"],
            "summary": "Get nomic metrics",
            "description": """Get detailed metrics about nomic loop performance.

**Response includes:**
- Cycle completion times
- Phase durations
- Success/failure rates
- Resource usage per cycle""",
            "operationId": "getNomicMetrics",
            "responses": {
                "200": {
                    "description": "Nomic metrics",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "total_cycles": {"type": "integer"},
                                    "successful_cycles": {"type": "integer"},
                                    "failed_cycles": {"type": "integer"},
                                    "avg_cycle_duration_s": {"type": "number"},
                                    "phase_durations": {
                                        "type": "object",
                                        "properties": {
                                            "debate": {"type": "number"},
                                            "design": {"type": "number"},
                                            "implement": {"type": "number"},
                                            "verify": {"type": "number"},
                                        },
                                    },
                                    "resource_usage": {
                                        "type": "object",
                                        "properties": {
                                            "tokens_used": {"type": "integer"},
                                            "api_calls": {"type": "integer"},
                                        },
                                    },
                                },
                            }
                        }
                    },
                }
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/nomic/proposals": {
        "get": {
            "tags": ["System"],
            "summary": "List nomic proposals",
            "description": """Get pending and recent nomic improvement proposals.

**Response includes:**
- Proposal ID and description
- Proposed changes
- Debate outcome
- Approval status""",
            "operationId": "listNomicProposals",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by proposal status",
                    "schema": {
                        "type": "string",
                        "enum": ["pending", "approved", "rejected", "implemented"],
                    },
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of proposals to return",
                    "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
                },
            ],
            "responses": {
                "200": {
                    "description": "Nomic proposals",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "proposals": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "title": {"type": "string"},
                                                "description": {"type": "string"},
                                                "status": {
                                                    "type": "string",
                                                    "enum": [
                                                        "pending",
                                                        "approved",
                                                        "rejected",
                                                        "implemented",
                                                    ],
                                                },
                                                "debate_id": {"type": "string"},
                                                "created_at": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                                "updated_at": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                            },
                                        },
                                    },
                                    "total": {"type": "integer"},
                                },
                            }
                        }
                    },
                }
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/nomic/proposals/{id}": {
        "post": {
            "tags": ["System"],
            "summary": "Update proposal status",
            "description": "Approve or reject a nomic improvement proposal.",
            "operationId": "updateNomicProposal",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Proposal ID",
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["approve", "reject"],
                                    "description": "Action to take on the proposal",
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Reason for approval/rejection",
                                },
                            },
                            "required": ["action"],
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Proposal updated"),
                "404": {"description": "Proposal not found"},
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/nomic/control/start": {
        "post": {
            "tags": ["System"],
            "summary": "Start nomic loop",
            "description": "Start or resume the nomic self-improvement loop.",
            "operationId": "startNomicLoop",
            "responses": {
                "200": _ok_response("Loop started"),
                "409": {"description": "Loop already running"},
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/nomic/control/stop": {
        "post": {
            "tags": ["System"],
            "summary": "Stop nomic loop",
            "description": "Gracefully stop the nomic self-improvement loop.",
            "operationId": "stopNomicLoop",
            "responses": {
                "200": _ok_response("Loop stopped"),
                "409": {"description": "Loop not running"},
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/nomic/control/pause": {
        "post": {
            "tags": ["System"],
            "summary": "Pause nomic loop",
            "description": "Pause the nomic loop at the end of the current phase.",
            "operationId": "pauseNomicLoop",
            "responses": {
                "200": _ok_response("Loop paused"),
                "409": {"description": "Loop not running or already paused"},
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/nomic/control/resume": {
        "post": {
            "tags": ["System"],
            "summary": "Resume nomic loop",
            "description": "Resume a paused nomic loop.",
            "operationId": "resumeNomicLoop",
            "responses": {
                "200": _ok_response("Loop resumed"),
                "409": {"description": "Loop not paused"},
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/nomic/control/skip-phase": {
        "post": {
            "tags": ["System"],
            "summary": "Skip current phase",
            "description": "Skip the current nomic phase and move to the next one.",
            "operationId": "skipNomicPhase",
            "responses": {
                "200": _ok_response("Phase skipped"),
                "409": {"description": "Loop not running"},
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/modes": {
        "get": {
            "tags": ["System"],
            "summary": "List operational modes",
            "description": """Get available operational modes including builtin and custom modes.

**Builtin modes:** debate, gauntlet, research, code-review, etc.

**Custom modes:** User-defined workflow configurations.""",
            "operationId": "listModes",
            "responses": {"200": _ok_response("Available modes")},
        },
    },
    "/api/history/cycles": {
        "get": {
            "tags": ["System"],
            "summary": "Cycle history",
            "description": """Get history of completed nomic cycles.

**Response includes:** cycle ID, duration, outcome, changes made.""",
            "operationId": "getCycleHistory",
            "parameters": [
                {
                    "name": "loop_id",
                    "in": "query",
                    "description": "Filter by specific nomic loop ID",
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of cycles to return",
                    "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
                },
            ],
            "responses": {"200": _ok_response("Cycle history")},
        },
    },
    "/api/history/events": {
        "get": {
            "tags": ["System"],
            "summary": "Event history",
            "description": """Get history of system events.

**Event types:** debate_started, debate_completed, agent_registered, consensus_reached, etc.""",
            "operationId": "getEventHistory",
            "parameters": [
                {
                    "name": "loop_id",
                    "in": "query",
                    "description": "Filter by specific nomic loop ID",
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of events to return",
                    "schema": {"type": "integer", "default": 100, "minimum": 1, "maximum": 500},
                },
            ],
            "responses": {"200": _ok_response("Event history")},
        },
    },
    "/api/history/debates": {
        "get": {
            "tags": ["System"],
            "summary": "Debate history",
            "description": """Get history of completed debates.

**Response includes:** topic, participants, outcome, duration, consensus status.""",
            "operationId": "getDebateHistory",
            "parameters": [
                {
                    "name": "loop_id",
                    "in": "query",
                    "description": "Filter by specific nomic loop ID",
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of debates to return",
                    "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
                },
            ],
            "responses": {"200": _ok_response("Debate history")},
        },
    },
    "/api/history/summary": {
        "get": {
            "tags": ["System"],
            "summary": "History summary",
            "description": """Get aggregated summary statistics for system history.

**Includes:** total debates, consensus rate, average duration, top performing agents.""",
            "operationId": "getHistorySummary",
            "parameters": [
                {
                    "name": "loop_id",
                    "in": "query",
                    "description": "Filter by specific nomic loop ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {"200": _ok_response("Summary statistics")},
        },
    },
    "/api/system/maintenance": {
        "get": {
            "tags": ["System"],
            "summary": "Run database maintenance",
            "description": """Run database maintenance tasks.

**Tasks:**
- `status`: Get current database statistics
- `vacuum`: Reclaim unused space
- `analyze`: Update query planner statistics
- `checkpoint`: Force WAL checkpoint (SQLite)
- `full`: Run all maintenance tasks

**Authentication:** Admin access required.""",
            "operationId": "runMaintenance",
            "parameters": [
                {
                    "name": "task",
                    "in": "query",
                    "description": "Maintenance task to run",
                    "schema": {
                        "type": "string",
                        "enum": ["status", "vacuum", "analyze", "checkpoint", "full"],
                        "default": "status",
                    },
                },
            ],
            "responses": {"200": _ok_response("Maintenance results")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/openapi": {
        "get": {
            "tags": ["System"],
            "summary": "OpenAPI specification",
            "description": """Get the OpenAPI 3.1 specification for this API.

**Formats:** JSON (default), YAML available at /api/openapi.yaml

**Use cases:**
- Generate client SDKs
- Import into API documentation tools
- Validate API requests""",
            "operationId": "getOpenAPISpec",
            "responses": {
                "200": {"description": "OpenAPI schema", "content": {"application/json": {}}}
            },
        },
    },
}
