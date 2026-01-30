"""Onboarding endpoint definitions."""

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS


def _response(description: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a response with optional inline schema."""
    resp: dict[str, Any] = {"description": description}
    if schema:
        resp["content"] = {"application/json": {"schema": schema}}
    return resp


ONBOARDING_ENDPOINTS: dict[str, Any] = {
    # -------------------------------------------------------------------------
    # Flow management
    # -------------------------------------------------------------------------
    "/api/v1/onboarding/flow": {
        "get": {
            "tags": ["Onboarding"],
            "summary": "Get current onboarding state",
            "description": (
                "Retrieve the current onboarding flow state for the authenticated user. "
                "Returns whether onboarding is needed, current step, completed steps, "
                "progress percentage, and recommended starter templates."
            ),
            "operationId": "getOnboardingFlow",
            "parameters": [
                {
                    "name": "user_id",
                    "in": "query",
                    "description": "User ID (defaults to authenticated user)",
                    "schema": {"type": "string"},
                },
                {
                    "name": "organization_id",
                    "in": "query",
                    "description": "Organization ID for org-scoped onboarding",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _response(
                    "Onboarding flow state",
                    {
                        "type": "object",
                        "properties": {
                            "exists": {"type": "boolean"},
                            "needs_onboarding": {"type": "boolean"},
                            "flow": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "current_step": {
                                        "type": "string",
                                        "enum": [
                                            "welcome",
                                            "use_case",
                                            "organization",
                                            "team_invite",
                                            "template_select",
                                            "first_debate",
                                            "receipt_review",
                                            "completion",
                                        ],
                                    },
                                    "completed_steps": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "use_case": {
                                        "type": ["string", "null"],
                                    },
                                    "selected_template_id": {
                                        "type": ["string", "null"],
                                    },
                                    "first_debate_id": {
                                        "type": ["string", "null"],
                                    },
                                    "quick_start_profile": {
                                        "type": ["string", "null"],
                                    },
                                    "team_invites_count": {"type": "integer"},
                                    "started_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                    "updated_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                    "completed_at": {
                                        "type": ["string", "null"],
                                        "format": "date-time",
                                    },
                                    "skipped": {"type": "boolean"},
                                    "progress_percentage": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                },
                            },
                            "recommended_templates": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/StarterTemplate"},
                            },
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Onboarding"],
            "summary": "Initialize onboarding flow",
            "description": (
                "Start a new onboarding flow for the authenticated user. "
                "Optionally specify a use case, quick-start profile, or a step to skip to."
            ),
            "operationId": "initOnboardingFlow",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "use_case": {
                                    "type": "string",
                                    "description": "Pre-defined use case for personalized onboarding",
                                    "enum": [
                                        "team_decisions",
                                        "architecture_review",
                                        "security_audit",
                                        "policy_review",
                                        "vendor_selection",
                                        "technical_planning",
                                        "compliance",
                                        "general",
                                    ],
                                },
                                "quick_start_profile": {
                                    "type": "string",
                                    "description": "Quick-start profile for immediate value",
                                    "enum": [
                                        "developer",
                                        "security",
                                        "executive",
                                        "product",
                                        "compliance",
                                        "sme",
                                    ],
                                },
                                "skip_to_step": {
                                    "type": "string",
                                    "description": "Step to skip forward to",
                                    "enum": [
                                        "welcome",
                                        "use_case",
                                        "organization",
                                        "team_invite",
                                        "template_select",
                                        "first_debate",
                                        "receipt_review",
                                        "completion",
                                    ],
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Onboarding flow initialized",
                    {
                        "type": "object",
                        "properties": {
                            "flow_id": {"type": "string"},
                            "current_step": {"type": "string"},
                            "use_case": {"type": ["string", "null"]},
                            "quick_start_profile": {"type": ["string", "null"]},
                            "recommended_templates": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/StarterTemplate"},
                            },
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # -------------------------------------------------------------------------
    # Quick-start
    # -------------------------------------------------------------------------
    "/api/v1/onboarding/quick-start": {
        "post": {
            "tags": ["Onboarding"],
            "summary": "Apply quick-start configuration",
            "description": (
                "Apply a quick-start profile for immediate value. "
                "This creates or updates the onboarding flow with pre-configured "
                "settings based on the selected profile (developer, security, executive, etc.)."
            ),
            "operationId": "applyOnboardingQuickStart",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["profile"],
                            "properties": {
                                "profile": {
                                    "type": "string",
                                    "description": "Quick-start profile to apply",
                                    "enum": [
                                        "developer",
                                        "security",
                                        "executive",
                                        "product",
                                        "compliance",
                                        "sme",
                                    ],
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Quick-start profile applied",
                    {
                        "type": "object",
                        "properties": {
                            "profile": {"type": "string"},
                            "config": {
                                "type": "object",
                                "properties": {
                                    "default_template": {"type": "string"},
                                    "suggested_templates": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "default_agents": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "default_rounds": {"type": "integer"},
                                    "focus_areas": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                            "default_template": {
                                "$ref": "#/components/schemas/StarterTemplate",
                            },
                            "suggested_templates": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/StarterTemplate"},
                            },
                            "message": {"type": "string"},
                            "next_action": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "template_id": {"type": "string"},
                                    "example_prompt": {"type": "string"},
                                },
                            },
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # -------------------------------------------------------------------------
    # First debate (guided)
    # -------------------------------------------------------------------------
    "/api/v1/onboarding/first-debate": {
        "post": {
            "tags": ["Onboarding"],
            "summary": "Start guided first debate",
            "description": (
                "Create a guided first debate as part of the onboarding experience. "
                "Uses a starter template or custom topic to demonstrate multi-agent debate. "
                "Enables receipt generation automatically for the first debate."
            ),
            "operationId": "startOnboardingFirstDebate",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "template_id": {
                                    "type": "string",
                                    "description": "Starter template ID to use",
                                },
                                "topic": {
                                    "type": "string",
                                    "description": "Custom topic (used if no template selected)",
                                },
                                "use_example": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Use the template's example prompt as the topic",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "First debate created",
                    {
                        "type": "object",
                        "properties": {
                            "debate_id": {"type": "string"},
                            "topic": {"type": "string"},
                            "config": {
                                "type": "object",
                                "properties": {
                                    "topic": {"type": "string"},
                                    "rounds": {"type": "integer"},
                                    "agents_count": {"type": "integer"},
                                    "is_onboarding": {"type": "boolean"},
                                    "enable_receipt_generation": {"type": "boolean"},
                                    "receipt_min_confidence": {"type": "number"},
                                },
                            },
                            "template": {
                                "oneOf": [
                                    {"$ref": "#/components/schemas/StarterTemplate"},
                                    {"type": "null"},
                                ],
                            },
                            "message": {"type": "string"},
                            "next_steps": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # -------------------------------------------------------------------------
    # Quick debate (one-click)
    # -------------------------------------------------------------------------
    "/api/v1/onboarding/quick-debate": {
        "post": {
            "tags": ["Onboarding"],
            "summary": "Start a one-click quick debate",
            "description": (
                "Create and immediately start a quick debate for onboarding. "
                "Uses minimal configuration for speed, defaulting to the express "
                "onboarding template with light debate format. Returns a WebSocket "
                "URL for real-time streaming of the debate."
            ),
            "operationId": "startOnboardingQuickDebate",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "template_id": {
                                    "type": "string",
                                    "default": "express_onboarding",
                                    "description": "Starter template ID (defaults to express onboarding)",
                                },
                                "topic": {
                                    "type": "string",
                                    "description": "Custom topic (uses template example if not provided)",
                                },
                                "profile": {
                                    "type": "string",
                                    "description": "Quick-start profile for agent selection",
                                    "enum": [
                                        "developer",
                                        "security",
                                        "executive",
                                        "product",
                                        "compliance",
                                        "sme",
                                    ],
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Quick debate started",
                    {
                        "type": "object",
                        "properties": {
                            "debate_id": {"type": "string"},
                            "websocket_url": {
                                "type": "string",
                                "description": "WebSocket URL for real-time debate streaming",
                            },
                            "topic": {"type": "string"},
                            "agents": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "rounds": {"type": "integer"},
                            "estimated_duration_seconds": {"type": "integer"},
                            "template": {
                                "$ref": "#/components/schemas/StarterTemplate",
                            },
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # -------------------------------------------------------------------------
    # Templates
    # -------------------------------------------------------------------------
    "/api/v1/onboarding/templates": {
        "get": {
            "tags": ["Onboarding"],
            "summary": "Get recommended starter templates",
            "description": (
                "Retrieve starter templates recommended for onboarding. "
                "Templates can be filtered by use case or quick-start profile. "
                "Results are ordered with the most relevant templates first."
            ),
            "operationId": "getOnboardingTemplates",
            "parameters": [
                {
                    "name": "use_case",
                    "in": "query",
                    "description": "Filter templates by use case",
                    "schema": {
                        "type": "string",
                        "enum": [
                            "team_decisions",
                            "architecture_review",
                            "security_audit",
                            "policy_review",
                            "vendor_selection",
                            "technical_planning",
                            "compliance",
                            "general",
                        ],
                    },
                },
                {
                    "name": "profile",
                    "in": "query",
                    "description": "Quick-start profile to prioritize templates for",
                    "schema": {
                        "type": "string",
                        "enum": [
                            "developer",
                            "security",
                            "executive",
                            "product",
                            "compliance",
                            "sme",
                        ],
                    },
                },
            ],
            "responses": {
                "200": _response(
                    "Starter templates",
                    {
                        "type": "object",
                        "properties": {
                            "templates": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/StarterTemplate"},
                            },
                            "total": {"type": "integer"},
                            "use_case": {"type": ["string", "null"]},
                            "profile": {"type": ["string", "null"]},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # -------------------------------------------------------------------------
    # Analytics
    # -------------------------------------------------------------------------
    "/api/v1/onboarding/analytics": {
        "get": {
            "tags": ["Onboarding"],
            "summary": "Get onboarding funnel analytics",
            "description": (
                "Retrieve onboarding funnel analytics including start, first-debate, "
                "and completion rates. Includes per-step completion counts and time range."
            ),
            "operationId": "getOnboardingAnalytics",
            "parameters": [
                {
                    "name": "organization_id",
                    "in": "query",
                    "description": "Filter analytics by organization",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _response(
                    "Onboarding analytics",
                    {
                        "type": "object",
                        "properties": {
                            "funnel": {
                                "type": "object",
                                "properties": {
                                    "started": {"type": "integer"},
                                    "first_debate": {"type": "integer"},
                                    "completed": {"type": "integer"},
                                    "completion_rate": {
                                        "type": "number",
                                        "description": "Percentage of users completing onboarding",
                                    },
                                },
                            },
                            "step_completion": {
                                "type": "object",
                                "additionalProperties": {"type": "integer"},
                                "description": "Completion count per onboarding step",
                            },
                            "total_events": {"type": "integer"},
                            "time_range": {
                                "type": "object",
                                "properties": {
                                    "earliest": {
                                        "type": ["string", "null"],
                                        "format": "date-time",
                                    },
                                    "latest": {
                                        "type": ["string", "null"],
                                        "format": "date-time",
                                    },
                                },
                            },
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
}
