"""SDK missing endpoints: Notifications and Pulse (trending topics).

Contains OpenAPI schema definitions for:
- Email notification configuration and recipients
- Telegram notification configuration
- Notification channels (send, status, history, test)
- Pulse analytics and trending topics
- Pulse scheduler management (start, stop, pause, resume, config, history)
"""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS
from aragora.server.openapi.endpoints.sdk_missing_core import _method_stub


# =============================================================================
# Response Schemas
# =============================================================================

# Notifications schemas
_EMAIL_CONFIG_SCHEMA = {
    "enabled": {"type": "boolean"},
    "smtp_host": {"type": "string"},
    "smtp_port": {"type": "integer"},
    "from_address": {"type": "string", "format": "email"},
    "use_tls": {"type": "boolean"},
    "events": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Event types to notify on",
    },
}

_EMAIL_RECIPIENT_SCHEMA = {
    "id": {"type": "string"},
    "email": {"type": "string", "format": "email"},
    "name": {"type": "string"},
    "verified": {"type": "boolean"},
    "subscribed_events": {"type": "array", "items": {"type": "string"}},
    "created_at": {"type": "string", "format": "date-time"},
}

_EMAIL_RECIPIENTS_LIST_SCHEMA = {
    "recipients": {
        "type": "array",
        "items": {"type": "object", "properties": _EMAIL_RECIPIENT_SCHEMA},
    },
    "total": {"type": "integer"},
}

_NOTIFICATION_HISTORY_SCHEMA = {
    "notifications": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "channel": {"type": "string", "enum": ["email", "telegram", "slack", "webhook"]},
                "event_type": {"type": "string"},
                "recipient": {"type": "string"},
                "status": {"type": "string", "enum": ["sent", "delivered", "failed", "pending"]},
                "sent_at": {"type": "string", "format": "date-time"},
                "delivered_at": {"type": "string", "format": "date-time"},
                "error": {"type": "string"},
            },
        },
    },
    "total": {"type": "integer"},
    "page": {"type": "integer"},
    "page_size": {"type": "integer"},
}

_SEND_NOTIFICATION_RESPONSE = {
    "notification_id": {"type": "string"},
    "status": {"type": "string", "enum": ["queued", "sent", "failed"]},
    "recipients_count": {"type": "integer"},
}

_NOTIFICATION_STATUS_SCHEMA = {
    "channels": {
        "type": "object",
        "properties": {
            "email": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}, "configured": {"type": "boolean"}},
            },
            "telegram": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}, "configured": {"type": "boolean"}},
            },
            "slack": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}, "configured": {"type": "boolean"}},
            },
        },
    },
    "queue_depth": {"type": "integer"},
    "last_sent_at": {"type": "string", "format": "date-time"},
}

_TELEGRAM_CONFIG_SCHEMA = {
    "enabled": {"type": "boolean"},
    "bot_token_set": {"type": "boolean"},
    "chat_ids": {"type": "array", "items": {"type": "string"}},
    "events": {"type": "array", "items": {"type": "string"}},
}

_TEST_NOTIFICATION_RESPONSE = {
    "success": {"type": "boolean"},
    "channel": {"type": "string"},
    "message": {"type": "string"},
    "delivered_at": {"type": "string", "format": "date-time"},
}

# Pulse schemas
_PULSE_ANALYTICS_SCHEMA = {
    "period": {"type": "string"},
    "topics_processed": {"type": "integer"},
    "debates_triggered": {"type": "integer"},
    "sources": {
        "type": "object",
        "properties": {
            "hackernews": {"type": "integer"},
            "reddit": {"type": "integer"},
            "twitter": {"type": "integer"},
        },
    },
    "top_topics": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "score": {"type": "number"},
                "source": {"type": "string"},
                "debate_id": {"type": "string"},
            },
        },
    },
    "quality_scores": {
        "type": "object",
        "properties": {
            "average": {"type": "number"},
            "min": {"type": "number"},
            "max": {"type": "number"},
        },
    },
}

_DEBATE_TOPIC_RESPONSE = {
    "topic_id": {"type": "string"},
    "topic": {"type": "string"},
    "source": {"type": "string"},
    "quality_score": {"type": "number"},
    "debate_id": {"type": "string"},
    "status": {"type": "string", "enum": ["queued", "debating", "completed"]},
}

_SCHEDULER_CONFIG_SCHEMA = {
    "enabled": {"type": "boolean"},
    "interval_minutes": {"type": "integer"},
    "sources": {
        "type": "array",
        "items": {"type": "string", "enum": ["hackernews", "reddit", "twitter"]},
    },
    "quality_threshold": {"type": "number", "minimum": 0, "maximum": 1},
    "max_topics_per_run": {"type": "integer"},
    "auto_debate": {"type": "boolean"},
}

_SCHEDULER_HISTORY_SCHEMA = {
    "runs": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "started_at": {"type": "string", "format": "date-time"},
                "completed_at": {"type": "string", "format": "date-time"},
                "topics_found": {"type": "integer"},
                "topics_processed": {"type": "integer"},
                "debates_triggered": {"type": "integer"},
                "status": {"type": "string", "enum": ["completed", "failed", "partial"]},
                "error": {"type": "string"},
            },
        },
    },
    "total_runs": {"type": "integer"},
}

_SCHEDULER_STATUS_SCHEMA = {
    "state": {"type": "string", "enum": ["running", "paused", "stopped"]},
    "last_run_at": {"type": "string", "format": "date-time"},
    "next_run_at": {"type": "string", "format": "date-time"},
    "current_run_id": {"type": "string"},
    "topics_in_queue": {"type": "integer"},
}

_SCHEDULER_ACTION_RESPONSE = {
    "success": {"type": "boolean"},
    "state": {"type": "string", "enum": ["running", "paused", "stopped"]},
    "message": {"type": "string"},
}


# =============================================================================
# Request Body Schemas
# =============================================================================

_EMAIL_CONFIG_REQUEST = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "smtp_host": {"type": "string"},
        "smtp_port": {"type": "integer"},
        "smtp_user": {"type": "string"},
        "smtp_password": {"type": "string"},
        "from_address": {"type": "string", "format": "email"},
        "use_tls": {"type": "boolean"},
        "events": {"type": "array", "items": {"type": "string"}},
    },
}

_EMAIL_RECIPIENT_REQUEST = {
    "type": "object",
    "required": ["email"],
    "properties": {
        "email": {"type": "string", "format": "email"},
        "name": {"type": "string"},
        "subscribed_events": {"type": "array", "items": {"type": "string"}},
    },
}

_SEND_NOTIFICATION_REQUEST = {
    "type": "object",
    "required": ["channel", "message"],
    "properties": {
        "channel": {"type": "string", "enum": ["email", "telegram", "slack", "webhook"]},
        "recipients": {"type": "array", "items": {"type": "string"}},
        "subject": {"type": "string"},
        "message": {"type": "string"},
        "template_id": {"type": "string"},
        "data": {"type": "object"},
    },
}

_TELEGRAM_CONFIG_REQUEST = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "bot_token": {"type": "string"},
        "chat_ids": {"type": "array", "items": {"type": "string"}},
        "events": {"type": "array", "items": {"type": "string"}},
    },
}

_TEST_NOTIFICATION_REQUEST = {
    "type": "object",
    "required": ["channel"],
    "properties": {
        "channel": {"type": "string", "enum": ["email", "telegram", "slack"]},
        "recipient": {"type": "string"},
    },
}

_DEBATE_TOPIC_REQUEST = {
    "type": "object",
    "required": ["topic"],
    "properties": {
        "topic": {"type": "string"},
        "source": {"type": "string"},
        "auto_debate": {"type": "boolean"},
        "priority": {"type": "integer", "minimum": 1, "maximum": 10},
    },
}

_SCHEDULER_CONFIG_REQUEST = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "interval_minutes": {"type": "integer", "minimum": 1},
        "sources": {"type": "array", "items": {"type": "string"}},
        "quality_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "max_topics_per_run": {"type": "integer", "minimum": 1},
        "auto_debate": {"type": "boolean"},
    },
}


# =============================================================================
# Endpoint Definitions
# =============================================================================

SDK_MISSING_MEDIA_ENDPOINTS: dict = {
    "/api/notifications/email/config": {
        "post": {
            "tags": ["Notifications"],
            "summary": "Configure email notifications",
            "description": "Set up email notification settings including SMTP configuration",
            "operationId": "postEmailConfig",
            "requestBody": {
                "content": {"application/json": {"schema": _EMAIL_CONFIG_REQUEST}},
                "required": True,
            },
            "responses": {
                "200": _ok_response("Email configuration saved", _EMAIL_CONFIG_SCHEMA),
            },
        },
    },
    "/api/notifications/email/recipient": {
        "delete": {
            "tags": ["Notifications"],
            "summary": "Remove email recipient",
            "description": "Remove an email recipient from notifications",
            "operationId": "deleteEmailRecipient",
            "parameters": [
                {
                    "name": "email",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "string", "format": "email"},
                    "description": "Email to remove",
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Recipient removed",
                    {"deleted": {"type": "boolean"}, "email": {"type": "string"}},
                ),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Notifications"],
            "summary": "Add email recipient",
            "description": "Add a new email recipient for notifications",
            "operationId": "postEmailRecipient",
            "requestBody": {
                "content": {"application/json": {"schema": _EMAIL_RECIPIENT_REQUEST}},
                "required": True,
            },
            "responses": {
                "200": _ok_response("Recipient added", _EMAIL_RECIPIENT_SCHEMA),
            },
        },
    },
    "/api/notifications/email/recipients": {
        "get": {
            "tags": ["Notifications"],
            "summary": "List email recipients",
            "description": "List all configured email notification recipients",
            "operationId": "getEmailRecipients",
            "responses": {
                "200": _ok_response("Email recipients list", _EMAIL_RECIPIENTS_LIST_SCHEMA),
            },
        },
    },
    "/api/notifications/history": {
        "get": {
            "tags": ["Notifications"],
            "summary": "Get notification history",
            "description": "Retrieve history of sent notifications across all channels",
            "operationId": "getNotificationsHistory",
            "parameters": [
                {
                    "name": "channel",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["email", "telegram", "slack", "webhook"]},
                    "description": "Filter by channel",
                },
                {
                    "name": "page",
                    "in": "query",
                    "schema": {"type": "integer"},
                    "description": "Page number",
                },
                {
                    "name": "page_size",
                    "in": "query",
                    "schema": {"type": "integer"},
                    "description": "Items per page",
                },
            ],
            "responses": {
                "200": _ok_response("Notification history", _NOTIFICATION_HISTORY_SCHEMA),
            },
        },
    },
    "/api/notifications/send": {
        "post": {
            "tags": ["Notifications"],
            "summary": "Send notification",
            "description": "Send a notification through a specified channel",
            "operationId": "postNotificationsSend",
            "requestBody": {
                "content": {"application/json": {"schema": _SEND_NOTIFICATION_REQUEST}},
                "required": True,
            },
            "responses": {
                "200": _ok_response("Notification sent", _SEND_NOTIFICATION_RESPONSE),
            },
        },
    },
    "/api/notifications/status": {
        "get": {
            "tags": ["Notifications"],
            "summary": "Get notification status",
            "description": "Get the current status of all notification channels",
            "operationId": "getNotificationsStatus",
            "responses": {
                "200": _ok_response("Notification channels status", _NOTIFICATION_STATUS_SCHEMA),
            },
        },
    },
    "/api/notifications/telegram/config": {
        "post": {
            "tags": ["Notifications"],
            "summary": "Configure Telegram notifications",
            "description": "Set up Telegram bot configuration for notifications",
            "operationId": "postTelegramConfig",
            "requestBody": {
                "content": {"application/json": {"schema": _TELEGRAM_CONFIG_REQUEST}},
                "required": True,
            },
            "responses": {
                "200": _ok_response("Telegram configuration saved", _TELEGRAM_CONFIG_SCHEMA),
            },
        },
    },
    "/api/notifications/test": {
        "post": {
            "tags": ["Notifications"],
            "summary": "Test notification",
            "description": "Send a test notification to verify channel configuration",
            "operationId": "postNotificationsTest",
            "requestBody": {
                "content": {"application/json": {"schema": _TEST_NOTIFICATION_REQUEST}},
                "required": True,
            },
            "responses": {
                "200": _ok_response("Test notification result", _TEST_NOTIFICATION_RESPONSE),
            },
        },
    },
    "/api/pulse/analytics": {
        "get": {
            "tags": ["Pulse"],
            "summary": "Get Pulse analytics",
            "description": "Get analytics for trending topics processing and debates triggered",
            "operationId": "getPulseAnalytics",
            "parameters": [
                {
                    "name": "period",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["hour", "day", "week", "month"]},
                    "description": "Analytics period",
                }
            ],
            "responses": {
                "200": _ok_response("Pulse analytics", _PULSE_ANALYTICS_SCHEMA),
            },
        },
    },
    "/api/pulse/debate-topic": {
        "post": {
            "tags": ["Pulse"],
            "summary": "Submit debate topic",
            "description": "Manually submit a topic for debate consideration",
            "operationId": "postPulseDebateTopic",
            "requestBody": {
                "content": {"application/json": {"schema": _DEBATE_TOPIC_REQUEST}},
                "required": True,
            },
            "responses": {
                "200": _ok_response("Topic submitted", _DEBATE_TOPIC_RESPONSE),
            },
        },
    },
    "/api/pulse/scheduler/config": {
        "patch": {
            "tags": ["Pulse"],
            "summary": "Update scheduler config",
            "description": "Update Pulse scheduler configuration settings",
            "operationId": "patchSchedulerConfig",
            "requestBody": {
                "content": {"application/json": {"schema": _SCHEDULER_CONFIG_REQUEST}},
                "required": True,
            },
            "responses": {
                "200": _ok_response("Scheduler configuration updated", _SCHEDULER_CONFIG_SCHEMA),
            },
        },
    },
    "/api/pulse/scheduler/history": {
        "get": {
            "tags": ["Pulse"],
            "summary": "Get scheduler history",
            "description": "Retrieve history of scheduler runs and their results",
            "operationId": "getSchedulerHistory",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer"},
                    "description": "Number of runs to return",
                }
            ],
            "responses": {
                "200": _ok_response("Scheduler run history", _SCHEDULER_HISTORY_SCHEMA),
            },
        },
    },
    "/api/pulse/scheduler/pause": {
        "post": {
            "tags": ["Pulse"],
            "summary": "Pause scheduler",
            "description": "Pause the Pulse topic scheduler",
            "operationId": "postSchedulerPause",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Scheduler paused", _SCHEDULER_ACTION_RESPONSE),
            },
        },
    },
    "/api/pulse/scheduler/resume": {
        "post": {
            "tags": ["Pulse"],
            "summary": "Resume scheduler",
            "description": "Resume a paused Pulse topic scheduler",
            "operationId": "postSchedulerResume",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Scheduler resumed", _SCHEDULER_ACTION_RESPONSE),
            },
        },
    },
    "/api/pulse/scheduler/start": {
        "post": {
            "tags": ["Pulse"],
            "summary": "Start scheduler",
            "description": "Start the Pulse topic scheduler",
            "operationId": "postSchedulerStart",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Scheduler started", _SCHEDULER_ACTION_RESPONSE),
            },
        },
    },
    "/api/pulse/scheduler/status": {
        "get": {
            "tags": ["Pulse"],
            "summary": "Get scheduler status",
            "description": "Get current status of the Pulse scheduler",
            "operationId": "getSchedulerStatus",
            "responses": {
                "200": _ok_response("Scheduler status", _SCHEDULER_STATUS_SCHEMA),
            },
        },
    },
    "/api/pulse/scheduler/stop": {
        "post": {
            "tags": ["Pulse"],
            "summary": "Stop scheduler",
            "description": "Stop the Pulse topic scheduler",
            "operationId": "postSchedulerStop",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Scheduler stopped", _SCHEDULER_ACTION_RESPONSE),
            },
        },
    },
}


# =============================================================================
# Additional Method Stubs (pulse, notifications)
# =============================================================================

SDK_MISSING_MEDIA_ADDITIONAL: dict = {
    "/api/v1/pulse/analytics": {
        "get": _method_stub("Pulse", "GET", "Get pulse analytics", op_id="getPulseAnalyticsV1"),
        "patch": _method_stub(
            "Pulse", "PATCH", "Update pulse analytics", op_id="patchPulseAnalyticsV1", has_body=True
        ),
        "post": _method_stub(
            "Pulse", "POST", "Submit pulse analytics", op_id="postPulseAnalyticsV1", has_body=True
        ),
    },
    "/api/v1/pulse/debate-topic": {
        "patch": _method_stub(
            "Pulse", "PATCH", "Update debate topic", op_id="patchPulseDebateTopicV1", has_body=True
        ),
        "post": _method_stub(
            "Pulse", "POST", "Set debate topic", op_id="postPulseDebateTopicV1", has_body=True
        ),
    },
    "/api/v1/pulse/scheduler/history": {
        "patch": _method_stub(
            "Pulse",
            "PATCH",
            "Update scheduler history",
            op_id="patchPulseSchedulerHistoryV1",
            has_body=True,
        ),
        "post": _method_stub(
            "Pulse",
            "POST",
            "Record scheduler history",
            op_id="postPulseSchedulerHistoryV1",
            has_body=True,
        ),
    },
    "/api/v1/pulse/scheduler/pause": {
        "patch": _method_stub(
            "Pulse", "PATCH", "Pause scheduler", op_id="patchPulseSchedulerPauseV1", has_body=True
        ),
        "post": _method_stub(
            "Pulse", "POST", "Pause scheduler", op_id="postPulseSchedulerPauseV1", has_body=True
        ),
    },
    "/api/v1/pulse/scheduler/resume": {
        "patch": _method_stub(
            "Pulse", "PATCH", "Resume scheduler", op_id="patchPulseSchedulerResumeV1", has_body=True
        ),
        "post": _method_stub(
            "Pulse", "POST", "Resume scheduler", op_id="postPulseSchedulerResumeV1", has_body=True
        ),
    },
    "/api/v1/pulse/scheduler/start": {
        "patch": _method_stub(
            "Pulse", "PATCH", "Start scheduler", op_id="patchPulseSchedulerStartV1", has_body=True
        ),
        "post": _method_stub(
            "Pulse", "POST", "Start scheduler", op_id="postPulseSchedulerStartV1", has_body=True
        ),
    },
    "/api/v1/pulse/scheduler/status": {
        "patch": _method_stub(
            "Pulse",
            "PATCH",
            "Update scheduler status",
            op_id="patchPulseSchedulerStatusV1",
            has_body=True,
        ),
        "post": _method_stub(
            "Pulse",
            "POST",
            "Set scheduler status",
            op_id="postPulseSchedulerStatusV1",
            has_body=True,
        ),
    },
    "/api/v1/pulse/scheduler/stop": {
        "patch": _method_stub(
            "Pulse", "PATCH", "Stop scheduler", op_id="patchPulseSchedulerStopV1", has_body=True
        ),
        "post": _method_stub(
            "Pulse", "POST", "Stop scheduler", op_id="postPulseSchedulerStopV1", has_body=True
        ),
    },
    "/api/v1/pulse/scheduler/config": {
        "post": _method_stub(
            "Pulse",
            "POST",
            "Configure scheduler",
            op_id="postPulseSchedulerConfigV1",
            has_body=True,
        ),
        "patch": _method_stub(
            "Pulse",
            "PATCH",
            "Update scheduler config",
            op_id="patchPulseSchedulerConfigV1",
            has_body=True,
        ),
    },
    "/api/v1/transcription/config": {
        "post": _method_stub(
            "Transcription",
            "POST",
            "Configure transcription",
            op_id="postTranscriptionConfigV1",
            has_body=True,
        ),
    },
    # Non-versioned pulse endpoints
    "/api/pulse/analytics": {
        "get": _method_stub("Pulse", "GET", "Get pulse analytics", op_id="getPulseAnalytics"),
    },
    "/api/pulse/scheduler/history": {
        "get": _method_stub(
            "Pulse", "GET", "Get scheduler history", op_id="getPulseSchedulerHistory"
        ),
    },
    "/api/pulse/scheduler/status": {
        "get": _method_stub(
            "Pulse", "GET", "Get scheduler status", op_id="getPulseSchedulerStatus"
        ),
    },
    "/api/pulse/scheduler/config": {
        "patch": _method_stub(
            "Pulse",
            "PATCH",
            "Update scheduler config",
            op_id="patchPulseSchedulerConfig",
            has_body=True,
        ),
    },
    "/api/pulse/debate-topic": {
        "post": _method_stub(
            "Pulse", "POST", "Set debate topic", op_id="postPulseDebateTopic", has_body=True
        ),
    },
    "/api/pulse/scheduler/pause": {
        "post": _method_stub(
            "Pulse", "POST", "Pause scheduler", op_id="postPulseSchedulerPause", has_body=True
        ),
    },
    "/api/pulse/scheduler/resume": {
        "post": _method_stub(
            "Pulse", "POST", "Resume scheduler", op_id="postPulseSchedulerResume", has_body=True
        ),
    },
    "/api/pulse/scheduler/start": {
        "post": _method_stub(
            "Pulse", "POST", "Start scheduler", op_id="postPulseSchedulerStart", has_body=True
        ),
    },
    "/api/pulse/scheduler/stop": {
        "post": _method_stub(
            "Pulse", "POST", "Stop scheduler", op_id="postPulseSchedulerStop", has_body=True
        ),
    },
}
