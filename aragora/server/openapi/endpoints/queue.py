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
