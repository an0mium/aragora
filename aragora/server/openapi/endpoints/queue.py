"""
OpenAPI endpoint definitions for Queue Management.

Endpoints for managing background job queues, workers, and dead letter queues.
"""

from aragora.server.openapi.helpers import (
    _ok_response,
    _array_response,
    STANDARD_ERRORS,
)

QUEUE_ENDPOINTS = {
    "/api/queue/jobs": {
        "get": {
            "tags": ["Queue"],
            "summary": "List queued jobs",
            "description": """List all jobs in the queue.

**Filtering options:**
- By status (pending, running, completed, failed)
- By job type
- By date range

**Pagination:** Uses cursor-based pagination for large result sets.""",
            "operationId": "listQueuedJobs",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "schema": {
                        "type": "string",
                        "enum": ["pending", "running", "completed", "failed"],
                    },
                },
                {"name": "type", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}},
                {"name": "cursor", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _array_response(
                    "List of jobs",
                    {
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                        "status": {"type": "string"},
                        "created_at": {"type": "string", "format": "date-time"},
                        "started_at": {"type": "string", "format": "date-time"},
                        "completed_at": {"type": "string", "format": "date-time"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Queue"],
            "summary": "Enqueue a new job",
            "description": """Add a new job to the processing queue.

**Job types:**
- debate: Run a debate
- analysis: Analyze content
- export: Export data
- cleanup: Clean up old data""",
            "operationId": "enqueueJob",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["type", "payload"],
                            "properties": {
                                "type": {"type": "string"},
                                "payload": {"type": "object"},
                                "priority": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "default": 5,
                                },
                                "delay_seconds": {"type": "integer", "minimum": 0},
                            },
                        }
                    }
                },
            },
            "responses": {
                "201": _ok_response(
                    "Job created",
                    {
                        "id": {"type": "string"},
                        "status": {"type": "string"},
                        "position": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/queue/jobs/{job_id}": {
        "get": {
            "tags": ["Queue"],
            "summary": "Get job details",
            "description": "Retrieve details of a specific job including its current status and results.",
            "operationId": "getQueueJob",
            "parameters": [
                {
                    "name": "job_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Job details",
                    {
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                        "status": {"type": "string"},
                        "payload": {"type": "object"},
                        "result": {"type": "object"},
                        "error": {"type": "string"},
                        "attempts": {"type": "integer"},
                        "created_at": {"type": "string", "format": "date-time"},
                    },
                ),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Queue"],
            "summary": "Cancel a job",
            "description": "Cancel a pending job. Running jobs cannot be cancelled.",
            "operationId": "cancelQueueJob",
            "parameters": [
                {
                    "name": "job_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Job cancelled", {"cancelled": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
                "409": {
                    "description": "Job cannot be cancelled (already running or completed)",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "error": {"type": "string"},
                                    "status": {"type": "string"},
                                },
                            }
                        }
                    },
                },
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/queue/jobs/{job_id}/retry": {
        "post": {
            "tags": ["Queue"],
            "summary": "Retry a failed job",
            "description": "Retry a failed job. Resets attempt count and re-queues.",
            "operationId": "retryQueueJob",
            "parameters": [
                {
                    "name": "job_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Job re-queued",
                    {"id": {"type": "string"}, "position": {"type": "integer"}},
                ),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/queue/stats": {
        "get": {
            "tags": ["Queue"],
            "summary": "Get queue statistics",
            "description": """Returns statistics about the job queue.

**Metrics included:**
- Jobs by status (pending, running, completed, failed)
- Average processing time
- Throughput (jobs/minute)
- Queue depth""",
            "operationId": "getQueueStats",
            "responses": {
                "200": _ok_response(
                    "Queue statistics",
                    {
                        "pending": {"type": "integer"},
                        "running": {"type": "integer"},
                        "completed": {"type": "integer"},
                        "failed": {"type": "integer"},
                        "avg_processing_ms": {"type": "number"},
                        "throughput_per_minute": {"type": "number"},
                    },
                ),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/queue/workers": {
        "get": {
            "tags": ["Queue"],
            "summary": "List active workers",
            "description": "Returns information about active queue workers.",
            "operationId": "listQueueWorkers",
            "responses": {
                "200": _array_response(
                    "Active workers",
                    {
                        "id": {"type": "string"},
                        "hostname": {"type": "string"},
                        "status": {"type": "string"},
                        "current_job": {"type": "string"},
                        "jobs_processed": {"type": "integer"},
                        "started_at": {"type": "string", "format": "date-time"},
                    },
                ),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/queue/dlq": {
        "get": {
            "tags": ["Queue"],
            "summary": "List dead letter queue",
            "description": "Returns jobs that failed permanently and were moved to the DLQ.",
            "operationId": "listDeadLetterQueue",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}},
            ],
            "responses": {
                "200": _array_response(
                    "DLQ jobs",
                    {
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                        "error": {"type": "string"},
                        "attempts": {"type": "integer"},
                        "failed_at": {"type": "string", "format": "date-time"},
                    },
                ),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/queue/dlq/requeue": {
        "post": {
            "tags": ["Queue"],
            "summary": "Requeue DLQ jobs",
            "description": "Move jobs from the dead letter queue back to the main queue for retry.",
            "operationId": "requeueDLQJobs",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "job_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific job IDs to requeue. If empty, requeues all.",
                                },
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response(
                    "Jobs requeued",
                    {"requeued": {"type": "integer"}, "failed": {"type": "integer"}},
                ),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/queue/cleanup": {
        "post": {
            "tags": ["Queue"],
            "summary": "Clean up old jobs",
            "description": "Remove completed and failed jobs older than the specified age.",
            "operationId": "cleanupQueue",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "older_than_days": {
                                    "type": "integer",
                                    "default": 7,
                                    "description": "Remove jobs older than this many days",
                                },
                                "status": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Only clean jobs with these statuses",
                                },
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response(
                    "Cleanup complete",
                    {"removed": {"type": "integer"}},
                ),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/queue/stale": {
        "get": {
            "tags": ["Queue"],
            "summary": "List stale jobs",
            "description": "Returns jobs that have been running longer than expected.",
            "operationId": "listStaleJobs",
            "parameters": [
                {
                    "name": "threshold_minutes",
                    "in": "query",
                    "schema": {"type": "integer", "default": 30},
                },
            ],
            "responses": {
                "200": _array_response(
                    "Stale jobs",
                    {
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                        "running_minutes": {"type": "number"},
                        "worker_id": {"type": "string"},
                    },
                ),
            },
            "security": [{"bearerAuth": []}],
        },
    },
}


def _task_queue_path_param(name: str, description: str) -> dict:
    return {
        "name": name,
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
        "description": description,
    }


def _task_queue_body(properties: dict, *, required: list[str] | None = None) -> dict:
    schema: dict = {
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }
    if required:
        schema["required"] = required
    return {
        "content": {
            "application/json": {
                "schema": schema,
            }
        }
    }


_TASK_QUEUE_COLLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "data": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        },
        "count": {"type": "integer"},
    },
    "additionalProperties": True,
}

_TASK_QUEUE_DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "data": {"type": "object", "additionalProperties": True},
    },
    "additionalProperties": True,
}

_TASK_QUEUE_CONFLICT = {
    "description": "Lease or file-scope conflict",
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "conflicts": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                    "violations": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                },
                "additionalProperties": True,
            }
        }
    },
}

QUEUE_ENDPOINTS.update(
    {
        "/api/v1/tasks/queue": {
            "get": {
                "tags": ["Tasks"],
                "summary": "List task queue items",
                "description": "List pending work queue items with optional status, work type, and limit filters.",
                "operationId": "listTaskQueueItems",
                "parameters": [
                    {
                        "name": "status",
                        "in": "query",
                        "schema": {"type": "string"},
                        "description": "Optional work status filter.",
                    },
                    {
                        "name": "work_type",
                        "in": "query",
                        "schema": {"type": "string"},
                        "description": "Optional work type filter.",
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                        "description": "Maximum queue items to return.",
                    },
                ],
                "responses": {
                    "200": _ok_response("Queue items", _TASK_QUEUE_COLLECTION_SCHEMA),
                    "400": STANDARD_ERRORS["400"],
                    "401": STANDARD_ERRORS["401"],
                    "403": STANDARD_ERRORS["403"],
                },
                "security": [{"bearerAuth": []}],
            },
        },
        "/api/v1/tasks/queue/stats": {
            "get": {
                "tags": ["Tasks"],
                "summary": "Get task queue statistics",
                "description": "Return aggregate statistics for the global developer work queue.",
                "operationId": "getTaskQueueStats",
                "responses": {
                    "200": _ok_response("Queue statistics", _TASK_QUEUE_DATA_SCHEMA),
                    "401": STANDARD_ERRORS["401"],
                    "403": STANDARD_ERRORS["403"],
                },
                "security": [{"bearerAuth": []}],
            },
        },
        "/api/v1/tasks/queue/{task_id}": {
            "get": {
                "tags": ["Tasks"],
                "summary": "Get task queue item",
                "description": "Retrieve a single task queue item by id.",
                "operationId": "getTaskQueueItem",
                "parameters": [_task_queue_path_param("task_id", "Task queue item id.")],
                "responses": {
                    "200": _ok_response("Queue item", _TASK_QUEUE_DATA_SCHEMA),
                    "401": STANDARD_ERRORS["401"],
                    "403": STANDARD_ERRORS["403"],
                    "404": STANDARD_ERRORS["404"],
                },
                "security": [{"bearerAuth": []}],
            },
        },
        "/api/v1/tasks/queue/sync": {
            "post": {
                "tags": ["Tasks"],
                "summary": "Synchronize task queue",
                "description": "Project pending and developer coordination work into the global task queue.",
                "operationId": "syncTaskQueue",
                "requestBody": _task_queue_body(
                    {
                        "include_pending": {"type": "boolean", "default": True},
                        "include_developer_tasks": {"type": "boolean", "default": True},
                        "complete_missing": {"type": "boolean", "default": True},
                    }
                ),
                "responses": {
                    "200": _ok_response("Queue synchronization result", _TASK_QUEUE_DATA_SCHEMA),
                    "401": STANDARD_ERRORS["401"],
                    "403": STANDARD_ERRORS["403"],
                    "500": STANDARD_ERRORS["500"],
                },
                "security": [{"bearerAuth": []}],
            },
        },
        "/api/v1/tasks/queue/{task_id}/claim": {
            "post": {
                "tags": ["Tasks"],
                "summary": "Claim task queue item",
                "description": "Claim a task queue item and create a lease for bounded implementation work.",
                "operationId": "claimTaskQueueItem",
                "parameters": [_task_queue_path_param("task_id", "Task queue item id.")],
                "requestBody": _task_queue_body(
                    {
                        "title": {"type": "string"},
                        "owner_agent": {"type": "string", "default": "unknown"},
                        "owner_session_id": {"type": "string", "default": "public-api"},
                        "branch": {"type": "string"},
                        "worktree_path": {"type": "string"},
                        "ttl_hours": {"type": "number", "default": 8.0},
                        "expected_tests": {"type": "array", "items": {"type": "string"}},
                        "allowed_globs": {"type": "array", "items": {"type": "string"}},
                        "claimed_paths": {"type": "array", "items": {"type": "string"}},
                        "metadata": {"type": "object", "additionalProperties": True},
                        "allow_overlap": {"type": "boolean", "default": False},
                    }
                ),
                "responses": {
                    "201": _ok_response("Created lease", _TASK_QUEUE_DATA_SCHEMA),
                    "401": STANDARD_ERRORS["401"],
                    "403": STANDARD_ERRORS["403"],
                    "404": STANDARD_ERRORS["404"],
                    "409": _TASK_QUEUE_CONFLICT,
                },
                "security": [{"bearerAuth": []}],
            },
        },
        "/api/v1/tasks/leases": {
            "get": {
                "tags": ["Tasks"],
                "summary": "List active task leases",
                "description": "List active task leases from the developer coordination store.",
                "operationId": "listTaskQueueLeases",
                "responses": {
                    "200": _ok_response("Active leases", _TASK_QUEUE_COLLECTION_SCHEMA),
                    "401": STANDARD_ERRORS["401"],
                    "403": STANDARD_ERRORS["403"],
                },
                "security": [{"bearerAuth": []}],
            },
        },
        "/api/v1/tasks/leases/{lease_id}/heartbeat": {
            "post": {
                "tags": ["Tasks"],
                "summary": "Heartbeat task lease",
                "description": "Refresh an active task lease and optionally adjust its TTL.",
                "operationId": "heartbeatTaskQueueLease",
                "parameters": [_task_queue_path_param("lease_id", "Task lease id.")],
                "requestBody": _task_queue_body(
                    {
                        "ttl_hours": {"type": "number"},
                    }
                ),
                "responses": {
                    "200": _ok_response("Updated lease", _TASK_QUEUE_DATA_SCHEMA),
                    "401": STANDARD_ERRORS["401"],
                    "403": STANDARD_ERRORS["403"],
                    "404": STANDARD_ERRORS["404"],
                },
                "security": [{"bearerAuth": []}],
            },
        },
        "/api/v1/tasks/leases/{lease_id}/release": {
            "post": {
                "tags": ["Tasks"],
                "summary": "Release task lease",
                "description": "Release a task lease without recording completion.",
                "operationId": "releaseTaskQueueLease",
                "parameters": [_task_queue_path_param("lease_id", "Task lease id.")],
                "responses": {
                    "200": _ok_response("Released lease", _TASK_QUEUE_DATA_SCHEMA),
                    "401": STANDARD_ERRORS["401"],
                    "403": STANDARD_ERRORS["403"],
                    "404": STANDARD_ERRORS["404"],
                },
                "security": [{"bearerAuth": []}],
            },
        },
        "/api/v1/tasks/leases/{lease_id}/complete": {
            "post": {
                "tags": ["Tasks"],
                "summary": "Complete task lease",
                "description": "Record completion metadata and receipts for an active task lease.",
                "operationId": "completeTaskQueueLease",
                "parameters": [_task_queue_path_param("lease_id", "Task lease id.")],
                "requestBody": _task_queue_body(
                    {
                        "owner_agent": {"type": "string"},
                        "owner_session_id": {"type": "string"},
                        "branch": {"type": "string"},
                        "worktree_path": {"type": "string"},
                        "base_sha": {"type": "string"},
                        "head_sha": {"type": "string"},
                        "commit_shas": {"type": "array", "items": {"type": "string"}},
                        "changed_paths": {"type": "array", "items": {"type": "string"}},
                        "tests_run": {"type": "array", "items": {"type": "string"}},
                        "validations_run": {"type": "array", "items": {"type": "string"}},
                        "assumptions": {"type": "array", "items": {"type": "string"}},
                        "blockers": {"type": "array", "items": {"type": "string"}},
                        "outcome": {"type": "string", "default": "completed"},
                        "risks": {"type": "array", "items": {"type": "string"}},
                        "pr_url": {"type": "string"},
                        "pr_number": {"type": "integer"},
                        "confidence": {"type": "number", "default": 0.0},
                        "metadata": {"type": "object", "additionalProperties": True},
                    }
                ),
                "responses": {
                    "200": _ok_response("Completion receipt", _TASK_QUEUE_DATA_SCHEMA),
                    "401": STANDARD_ERRORS["401"],
                    "403": STANDARD_ERRORS["403"],
                    "404": STANDARD_ERRORS["404"],
                    "409": _TASK_QUEUE_CONFLICT,
                },
                "security": [{"bearerAuth": []}],
            },
        },
        "/api/v1/tasks/salvage": {
            "get": {
                "tags": ["Tasks"],
                "summary": "List task salvage candidates",
                "description": "List stale or recoverable developer coordination work that may need salvage.",
                "operationId": "listTaskQueueSalvageCandidates",
                "responses": {
                    "200": _ok_response("Salvage candidates", _TASK_QUEUE_COLLECTION_SCHEMA),
                    "401": STANDARD_ERRORS["401"],
                    "403": STANDARD_ERRORS["403"],
                },
                "security": [{"bearerAuth": []}],
            },
        },
    }
)
