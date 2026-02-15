#!/usr/bin/env python3
"""
Add request body schemas to OpenAPI operations that lack them.

Introspects handler source files to infer request body fields from
validate_body decorators, request_body dicts, and json body reads.
For endpoints where no schema can be inferred, generates a minimal
placeholder schema with "additionalProperties: true".

Usage:
    python scripts/add_openapi_request_schemas.py
    python scripts/add_openapi_request_schemas.py --spec docs/api/openapi.json --dry-run
    python scripts/add_openapi_request_schemas.py --verbose
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path


# Common schema fragments for reuse
_PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "description": "Agent persona description"},
        "traits": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Personality traits",
        },
        "expertise": {
            "type": "object",
            "additionalProperties": {"type": "number"},
            "description": "Domain expertise scores (0-1)",
        },
    },
}

_DEBATE_BASE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The question or topic to debate"},
        "agents": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agent identifiers to include",
        },
        "rounds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "default": 3,
            "description": "Number of debate rounds",
        },
        "consensus_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "default": 0.7,
            "description": "Threshold for consensus detection",
        },
        "protocol": {
            "type": "string",
            "enum": ["structured", "free", "adversarial", "collaborative"],
            "default": "structured",
            "description": "Debate protocol to use",
        },
        "context": {"type": "string", "description": "Additional context"},
        "workspace_id": {"type": "string", "description": "Workspace ID"},
    },
}

_DECISION_INTEGRITY_SCHEMA = {
    "type": "object",
    "properties": {
        "include_receipt": {
            "type": "boolean",
            "default": True,
            "description": "Include decision receipt in response",
        },
        "include_plan": {
            "type": "boolean",
            "default": True,
            "description": "Include implementation plan in response",
        },
        "include_context": {
            "type": "boolean",
            "default": False,
            "description": "Capture and include context snapshot (memory + knowledge)",
        },
        "plan_strategy": {
            "type": "string",
            "description": "Implementation plan strategy",
            "default": "single_task",
        },
        "execution_mode": {
            "type": "string",
            "description": "Execution mode (plan_only, request_approval, execute, workflow)",
        },
        "execution_engine": {
            "type": "string",
            "description": "Execution engine override (hybrid, fabric, computer_use, workflow)",
        },
        "parallel_execution": {
            "type": "boolean",
            "description": "Execute independent tasks in parallel",
        },
        "max_parallel": {
            "type": "integer",
            "description": "Max parallel tasks for execution",
        },
        "notify_origin": {
            "type": "boolean",
            "description": "Send progress and completion to originating channel",
        },
        "risk_level": {
            "type": "string",
            "description": "Risk level for approval requests",
            "default": "medium",
        },
        "approval_timeout_seconds": {
            "type": "integer",
            "description": "Approval timeout in seconds",
        },
        "approval_mode": {
            "type": "string",
            "description": "Approval mode (risk_based, always, never)",
        },
        "max_auto_risk": {
            "type": "string",
            "description": "Max risk level allowed for auto-approval",
        },
        "budget_limit_usd": {
            "type": "number",
            "description": "Budget cap for implementation execution",
        },
        "openclaw_actions": {
            "type": "array",
            "items": {"type": "object"},
            "description": "OpenClaw action overrides",
        },
        "computer_use_actions": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Computer-use action overrides",
        },
        "openclaw_session": {
            "type": "object",
            "description": "OpenClaw session configuration",
        },
        "implementation_profile": {
            "type": "object",
            "description": "Implementation profile overrides",
        },
        "implementers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Preferred implementer agents",
        },
        "critic": {"type": "string", "description": "Critic agent override"},
        "reviser": {"type": "string", "description": "Reviser agent override"},
        "strategy": {"type": "string", "description": "Implementation strategy"},
        "max_revisions": {"type": "integer", "description": "Max revision passes"},
        "complexity_router": {
            "type": "object",
            "description": "Route by task complexity",
        },
        "task_type_router": {
            "type": "object",
            "description": "Route by task type",
        },
        "capability_router": {
            "type": "object",
            "description": "Route by capability",
        },
        "fabric_models": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Fabric model pool",
        },
        "fabric_pool_id": {"type": "string", "description": "Fabric pool ID"},
        "fabric_min_agents": {
            "type": "integer",
            "description": "Fabric min agents",
        },
        "fabric_max_agents": {
            "type": "integer",
            "description": "Fabric max agents",
        },
        "fabric_timeout_seconds": {
            "type": "number",
            "description": "Fabric run timeout",
        },
        "channel_targets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Explicit channel targets",
        },
        "thread_id": {"type": "string", "description": "Thread ID override"},
        "thread_id_by_platform": {
            "type": "object",
            "description": "Thread IDs keyed by platform",
        },
    },
}

# Known request body schemas for common endpoint patterns
KNOWN_SCHEMAS: dict[str, dict] = {
    # ==========================================================================
    # Debate endpoints
    # ==========================================================================
    "POST /api/debates": {**_DEBATE_BASE_SCHEMA, "required": ["question"]},
    "POST /api/v1/debates": {**_DEBATE_BASE_SCHEMA, "required": ["question"]},
    "POST /api/v1/debates/{id}/followup": {
        "type": "object",
        "required": ["question"],
        "properties": {
            "question": {"type": "string", "description": "Follow-up question"},
            "inherit_context": {
                "type": "boolean",
                "default": True,
                "description": "Inherit context from parent debate",
            },
            "agents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Override agent list (defaults to parent agents)",
            },
        },
    },
    "POST /api/v1/debates/{id}/explainability/counterfactual": {
        "type": "object",
        "required": ["hypothesis"],
        "properties": {
            "hypothesis": {
                "type": "string",
                "description": "Alternative scenario to explore",
            },
            "factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Factors to vary in the counterfactual",
            },
        },
    },
    "POST /api/v1/debates/{id}/decision-integrity": _DECISION_INTEGRITY_SCHEMA,
    "POST /api/v1/debates/{param}/decision-integrity": _DECISION_INTEGRITY_SCHEMA,
    "PATCH /api/v1/debates/{id}/decision-integrity": _DECISION_INTEGRITY_SCHEMA,
    "PATCH /api/v1/debates/{param}/decision-integrity": _DECISION_INTEGRITY_SCHEMA,
    # ==========================================================================
    # Persona endpoints
    # ==========================================================================
    "POST /api/personas": {
        "type": "object",
        "required": ["agent_name"],
        "properties": {
            "agent_name": {"type": "string", "description": "Unique agent identifier"},
            **_PERSONA_SCHEMA["properties"],
        },
    },
    "POST /api/v1/personas": {
        "type": "object",
        "required": ["agent_name"],
        "properties": {
            "agent_name": {"type": "string", "description": "Unique agent identifier"},
            **_PERSONA_SCHEMA["properties"],
        },
    },
    "PUT /api/agent/{name}/persona": _PERSONA_SCHEMA,
    "PUT /api/v1/agent/{name}/persona": _PERSONA_SCHEMA,
    # ==========================================================================
    # Teams Bot endpoints
    # ==========================================================================
    "POST /api/v1/bots/teams/messages": {
        "type": "object",
        "required": ["conversation_id", "message"],
        "properties": {
            "conversation_id": {"type": "string", "description": "Teams conversation ID"},
            "message": {"type": "string", "description": "Message content"},
            "card": {"type": "object", "description": "Optional Adaptive Card payload"},
        },
    },
    # ==========================================================================
    # Policy endpoints
    # ==========================================================================
    "POST /api/policies/{id}/toggle": {
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean", "description": "Enable/disable policy"},
            "reason": {"type": "string", "description": "Reason for toggle"},
        },
    },
    "POST /api/v1/policies/{id}/toggle": {
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean", "description": "Enable/disable policy"},
            "reason": {"type": "string", "description": "Reason for toggle"},
        },
    },
    # ==========================================================================
    # Knowledge endpoints
    # ==========================================================================
    "POST /api/knowledge/entries": {
        "type": "object",
        "required": ["content"],
        "properties": {
            "content": {"type": "string", "description": "Knowledge entry content"},
            "title": {"type": "string", "description": "Entry title"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorization",
            },
            "visibility": {
                "type": "string",
                "enum": ["private", "workspace", "organization", "public"],
                "default": "workspace",
            },
        },
    },
    "POST /api/v1/knowledge/entries": {
        "type": "object",
        "required": ["content"],
        "properties": {
            "content": {"type": "string", "description": "Knowledge entry content"},
            "title": {"type": "string", "description": "Entry title"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorization",
            },
            "visibility": {
                "type": "string",
                "enum": ["private", "workspace", "organization", "public"],
                "default": "workspace",
            },
        },
    },
    # ==========================================================================
    # Workflow endpoints
    # ==========================================================================
    "POST /api/workflows": {
        "type": "object",
        "required": ["name", "steps"],
        "properties": {
            "name": {"type": "string", "description": "Workflow name"},
            "description": {"type": "string", "description": "Workflow description"},
            "steps": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Workflow step definitions",
            },
            "trigger": {"type": "object", "description": "Trigger configuration"},
        },
    },
    "POST /api/v1/workflows": {
        "type": "object",
        "required": ["name", "steps"],
        "properties": {
            "name": {"type": "string", "description": "Workflow name"},
            "description": {"type": "string", "description": "Workflow description"},
            "steps": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Workflow step definitions",
            },
            "trigger": {"type": "object", "description": "Trigger configuration"},
        },
    },
    # ==========================================================================
    # Agent endpoints
    # ==========================================================================
    "POST /api/agents/train": {
        "type": "object",
        "required": ["agent_name"],
        "properties": {
            "agent_name": {"type": "string", "description": "Agent to train"},
            "training_data": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Training examples",
            },
            "epochs": {"type": "integer", "minimum": 1, "default": 1},
        },
    },
    "POST /api/v1/agents/{name}/feedback": {
        "type": "object",
        "required": ["rating"],
        "properties": {
            "rating": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Rating 1-5",
            },
            "comment": {"type": "string", "description": "Optional feedback text"},
            "debate_id": {"type": "string", "description": "Associated debate ID"},
        },
    },
    # ==========================================================================
    # OAuth endpoints
    # ==========================================================================
    "POST /api/v1/oauth/token": {
        "type": "object",
        "required": ["grant_type"],
        "properties": {
            "grant_type": {
                "type": "string",
                "enum": ["authorization_code", "refresh_token", "client_credentials"],
            },
            "code": {"type": "string", "description": "Authorization code"},
            "redirect_uri": {"type": "string", "description": "Redirect URI"},
            "refresh_token": {"type": "string", "description": "Refresh token"},
            "client_id": {"type": "string"},
            "client_secret": {"type": "string"},
        },
    },
    "POST /api/v1/oauth/revoke": {
        "type": "object",
        "required": ["token"],
        "properties": {
            "token": {"type": "string", "description": "Token to revoke"},
            "token_type_hint": {
                "type": "string",
                "enum": ["access_token", "refresh_token"],
            },
        },
    },
    # ==========================================================================
    # Webhook endpoints
    # ==========================================================================
    "POST /api/webhooks": {
        "type": "object",
        "required": ["url", "events"],
        "properties": {
            "url": {"type": "string", "format": "uri", "description": "Webhook URL"},
            "events": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Events to subscribe to",
            },
            "secret": {"type": "string", "description": "Signing secret"},
            "active": {"type": "boolean", "default": True},
        },
    },
    "POST /api/v1/webhooks": {
        "type": "object",
        "required": ["url", "events"],
        "properties": {
            "url": {"type": "string", "format": "uri", "description": "Webhook URL"},
            "events": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Events to subscribe to",
            },
            "secret": {"type": "string", "description": "Signing secret"},
            "active": {"type": "boolean", "default": True},
        },
    },
    # ==========================================================================
    # Integration endpoints
    # ==========================================================================
    "POST /api/v1/integrations/slack/install": {
        "type": "object",
        "properties": {
            "workspace_id": {"type": "string"},
            "channel_id": {"type": "string"},
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Features to enable",
            },
        },
    },
    "POST /api/v1/integrations/teams/install": {
        "type": "object",
        "properties": {
            "tenant_id": {"type": "string"},
            "team_id": {"type": "string"},
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Features to enable",
            },
        },
    },
    # ==========================================================================
    # Gauntlet endpoints
    # ==========================================================================
    "POST /api/v1/gauntlet/run": {
        "type": "object",
        "required": ["question"],
        "properties": {
            "question": {"type": "string", "description": "Decision to evaluate"},
            "challenges": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific challenges to run",
            },
            "depth": {
                "type": "string",
                "enum": ["quick", "standard", "deep"],
                "default": "standard",
            },
        },
    },
    "POST /api/v1/gauntlet/receipts": {
        "type": "object",
        "required": ["debate_id"],
        "properties": {
            "debate_id": {"type": "string"},
            "include_evidence": {"type": "boolean", "default": True},
            "format": {
                "type": "string",
                "enum": ["json", "pdf", "markdown"],
                "default": "json",
            },
        },
    },
    # ==========================================================================
    # Notification endpoints
    # ==========================================================================
    "POST /api/v1/notifications/send": {
        "type": "object",
        "required": ["recipient", "message"],
        "properties": {
            "recipient": {"type": "string", "description": "User or channel ID"},
            "message": {"type": "string"},
            "channel": {
                "type": "string",
                "enum": ["email", "slack", "teams", "webhook"],
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high", "urgent"],
                "default": "normal",
            },
        },
    },
    # ==========================================================================
    # Budget endpoints
    # ==========================================================================
    "POST /api/v1/budgets": {
        "type": "object",
        "required": ["name", "limit"],
        "properties": {
            "name": {"type": "string"},
            "limit": {"type": "number", "minimum": 0},
            "period": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly", "quarterly", "annual"],
                "default": "monthly",
            },
            "alerts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "threshold": {"type": "number"},
                        "action": {"type": "string"},
                    },
                },
            },
        },
    },
    "PUT /api/v1/budgets/{id}": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "limit": {"type": "number", "minimum": 0},
            "period": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly", "quarterly", "annual"],
            },
            "enabled": {"type": "boolean"},
        },
    },
    # ==========================================================================
    # Control endpoints (Nomic Loop)
    # ==========================================================================
    "POST /api/nomic/control/start": {
        "type": "object",
        "properties": {
            "cycles": {"type": "integer", "minimum": 1, "description": "Number of cycles"},
            "goals": {"type": "array", "items": {"type": "string"}},
        },
    },
    "POST /api/v1/nomic/control/start": {
        "type": "object",
        "properties": {
            "cycles": {"type": "integer", "minimum": 1},
            "goals": {"type": "array", "items": {"type": "string"}},
        },
    },
    "POST /api/nomic/control/stop": {
        "type": "object",
        "properties": {"force": {"type": "boolean", "default": False}},
    },
    "POST /api/v1/nomic/control/stop": {
        "type": "object",
        "properties": {"force": {"type": "boolean", "default": False}},
    },
    "POST /api/nomic/control/pause": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/v1/nomic/control/pause": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/nomic/control/resume": {
        "type": "object",
        "properties": {"skip_current": {"type": "boolean"}},
    },
    "POST /api/v1/nomic/control/resume": {
        "type": "object",
        "properties": {"skip_current": {"type": "boolean"}},
    },
    "POST /api/nomic/control/skip-phase": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/v1/nomic/control/skip-phase": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    # ==========================================================================
    # Archive/lifecycle endpoints
    # ==========================================================================
    "POST /api/v1/debates/{id}/archive": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/v1/debates/{id}/pause": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/v1/debates/{id}/resume": {"type": "object", "properties": {}},
    "POST /api/v1/debates/{id}/start": {"type": "object", "properties": {}},
    "POST /api/v1/debates/{id}/stop": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/v1/debates/{id}/cancel": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/debates/{id}/messages": {
        "type": "object",
        "required": ["content"],
        "properties": {
            "content": {"type": "string", "description": "Message content"},
            "role": {"type": "string", "enum": ["user", "system"]},
        },
    },
    "POST /api/debates/{id}/publish/twitter": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "include_verdict": {"type": "boolean", "default": True},
        },
    },
    "POST /api/v1/debates/{id}/publish/twitter": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "include_verdict": {"type": "boolean", "default": True},
        },
    },
    "POST /api/debates/{id}/publish/youtube": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "visibility": {"type": "string", "enum": ["public", "unlisted", "private"]},
        },
    },
    "POST /api/v1/debates/{id}/publish/youtube": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "visibility": {"type": "string", "enum": ["public", "unlisted", "private"]},
        },
    },
    # ==========================================================================
    # Document endpoints
    # ==========================================================================
    "POST /api/documents/upload": {
        "type": "object",
        "required": ["file"],
        "properties": {
            "file": {"type": "string", "format": "binary"},
            "folder": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    },
    "POST /api/v1/documents/upload": {
        "type": "object",
        "required": ["file"],
        "properties": {
            "file": {"type": "string", "format": "binary"},
            "folder": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    },
    # ==========================================================================
    # Auth endpoints
    # ==========================================================================
    "POST /api/auth/logout": {"type": "object", "properties": {}},
    "POST /api/v1/auth/logout": {"type": "object", "properties": {}},
    "POST /api/auth/logout-all": {"type": "object", "properties": {}},
    "POST /api/v1/auth/logout-all": {"type": "object", "properties": {}},
    "POST /api/auth/mfa/setup": {
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["totp", "sms", "email"]},
            "phone": {"type": "string"},
        },
    },
    "POST /api/v1/auth/mfa/setup": {
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["totp", "sms", "email"]},
            "phone": {"type": "string"},
        },
    },
    # ==========================================================================
    # Memory endpoints
    # ==========================================================================
    "POST /api/v1/memory/continuum/consolidate": {
        "type": "object",
        "properties": {"tier": {"type": "string"}},
    },
    "POST /api/v1/memory/continuum/cleanup": {
        "type": "object",
        "properties": {"max_age_days": {"type": "integer"}},
    },
    # ==========================================================================
    # Retention endpoints
    # ==========================================================================
    "POST /api/retention/policies/{policy_id}/execute": {
        "type": "object",
        "properties": {"dry_run": {"type": "boolean"}},
    },
    "POST /api/v1/retention/policies/{policy_id}/execute": {
        "type": "object",
        "properties": {"dry_run": {"type": "boolean"}},
    },
    # ==========================================================================
    # Cross-pollination endpoints
    # ==========================================================================
    "POST /api/cross-pollination/reset": {
        "type": "object",
        "properties": {"workspace_id": {"type": "string"}},
    },
    "POST /api/v1/cross-pollination/reset": {
        "type": "object",
        "properties": {"workspace_id": {"type": "string"}},
    },
    "POST /api/cross-pollination/km/sync": {
        "type": "object",
        "properties": {"adapters": {"type": "array", "items": {"type": "string"}}},
    },
    "POST /api/v1/cross-pollination/km/sync": {
        "type": "object",
        "properties": {"adapters": {"type": "array", "items": {"type": "string"}}},
    },
    # ==========================================================================
    # Codebase endpoints
    # ==========================================================================
    "POST /api/v1/codebase/clear-cache": {
        "type": "object",
        "properties": {"scope": {"type": "string"}},
    },
    # ==========================================================================
    # Gmail endpoints
    # ==========================================================================
    "POST /api/v1/gmail/messages/{message_id}/archive": {"type": "object", "properties": {}},
    "POST /api/v1/gmail/threads/{thread_id}/archive": {"type": "object", "properties": {}},
    "POST /api/v1/gmail/drafts/{draft_id}/send": {"type": "object", "properties": {}},
    # ==========================================================================
    # Knowledge endpoints
    # ==========================================================================
    "POST /api/v1/knowledge/facts/{fact_id}/verify": {
        "type": "object",
        "properties": {"verdict": {"type": "boolean"}, "evidence": {"type": "string"}},
    },
    "POST /api/v1/knowledge/mound/revalidate/{node_id}": {"type": "object", "properties": {}},
    "POST /api/v1/knowledge/mound/sync/{adapter_name}": {
        "type": "object",
        "properties": {"force": {"type": "boolean"}},
    },
    # ==========================================================================
    # Audit session endpoints
    # ==========================================================================
    "POST /api/v1/audit/sessions/{session_id}/start": {"type": "object", "properties": {}},
    "POST /api/v1/audit/sessions/{session_id}/pause": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/v1/audit/sessions/{session_id}/resume": {"type": "object", "properties": {}},
    "POST /api/v1/audit/sessions/{session_id}/cancel": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    # ==========================================================================
    # Budget alert endpoints
    # ==========================================================================
    "POST /api/v1/budgets/{budget_id}/alerts/{alert_id}/acknowledge": {
        "type": "object",
        "properties": {"note": {"type": "string"}},
    },
    "POST /api/v1/budgets/{budget_id}/reset": {
        "type": "object",
        "properties": {"period": {"type": "string"}},
    },
    # ==========================================================================
    # Webhook endpoints
    # ==========================================================================
    "POST /api/v1/webhooks/slo/test": {"type": "object", "properties": {"url": {"type": "string"}}},
    "POST /api/v1/webhooks/dead-letter/{id}/retry": {"type": "object", "properties": {}},
    # ==========================================================================
    # Integration endpoints
    # ==========================================================================
    "POST /api/v1/integrations/{type}/test": {
        "type": "object",
        "properties": {"config": {"type": "object"}},
    },
    "POST /api/integrations/slack/uninstall": {
        "type": "object",
        "properties": {"workspace_id": {"type": "string"}},
    },
    "POST /api/v1/integrations/slack/uninstall": {
        "type": "object",
        "properties": {"workspace_id": {"type": "string"}},
    },
    "POST /api/integrations/discord/uninstall": {
        "type": "object",
        "properties": {"guild_id": {"type": "string"}},
    },
    "POST /api/v1/integrations/discord/uninstall": {
        "type": "object",
        "properties": {"guild_id": {"type": "string"}},
    },
    "POST /api/integrations/teams/refresh": {
        "type": "object",
        "properties": {"tenant_id": {"type": "string"}},
    },
    "POST /api/v1/integrations/teams/refresh": {
        "type": "object",
        "properties": {"tenant_id": {"type": "string"}},
    },
    # ==========================================================================
    # Nomic witness/mayor endpoints
    # ==========================================================================
    "POST /api/v1/nomic/witness/status": {"type": "object", "properties": {}},
    "POST /api/nomic/witness/status": {"type": "object", "properties": {}},
    "POST /api/v1/nomic/mayor/current": {"type": "object", "properties": {}},
    "POST /api/nomic/mayor/current": {"type": "object", "properties": {}},
    # ==========================================================================
    # Admin endpoints
    # ==========================================================================
    "POST /api/v1/admin/impersonate/{user_id}": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/v1/admin/nomic/reset": {
        "type": "object",
        "properties": {"confirm": {"type": "boolean"}},
    },
    "POST /api/v1/admin/nomic/pause": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/v1/admin/nomic/resume": {"type": "object", "properties": {}},
    "POST /api/v1/admin/security/status": {"type": "object", "properties": {}},
    "POST /api/admin/security/status": {"type": "object", "properties": {}},
    "POST /api/v1/admin/security/health": {"type": "object", "properties": {}},
    "POST /api/admin/security/health": {"type": "object", "properties": {}},
    # ==========================================================================
    # A2A endpoints
    # ==========================================================================
    "POST /api/v1/a2a/tasks/{task_id}/stream": {"type": "object", "properties": {}},
    # ==========================================================================
    # Queue endpoints
    # ==========================================================================
    "POST /api/queue/jobs/{job_id}/retry": {"type": "object", "properties": {}},
    "POST /api/v1/queue/jobs/{job_id}/retry": {"type": "object", "properties": {}},
    # ==========================================================================
    # SCIM endpoints
    # ==========================================================================
    "PUT /scim/v2/Groups/{group_id}": {
        "type": "object",
        "required": ["displayName"],
        "properties": {
            "displayName": {"type": "string"},
            "members": {"type": "array", "items": {"type": "object"}},
        },
    },
    "PATCH /scim/v2/Groups/{group_id}": {
        "type": "object",
        "required": ["Operations"],
        "properties": {
            "Operations": {"type": "array", "items": {"type": "object"}},
        },
    },
    "PUT /scim/v2/Users/{user_id}": {
        "type": "object",
        "required": ["userName"],
        "properties": {
            "userName": {"type": "string"},
            "name": {"type": "object"},
            "emails": {"type": "array", "items": {"type": "object"}},
            "active": {"type": "boolean"},
        },
    },
    "PATCH /scim/v2/Users/{user_id}": {
        "type": "object",
        "required": ["Operations"],
        "properties": {
            "Operations": {"type": "array", "items": {"type": "object"}},
        },
    },
    # ==========================================================================
    # Policy management endpoints
    # ==========================================================================
    "POST /api/policies/{id}/disable": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/v1/policies/{id}/disable": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/policies/{id}/enable": {"type": "object", "properties": {}},
    "POST /api/v1/policies/{id}/enable": {"type": "object", "properties": {}},
    # ==========================================================================
    # Pulse endpoints
    # ==========================================================================
    "PATCH /api/pulse/analytics": {
        "type": "object",
        "properties": {"enabled": {"type": "boolean"}, "retention_days": {"type": "integer"}},
    },
    "POST /api/pulse/analytics": {
        "type": "object",
        "properties": {"source": {"type": "string"}, "data": {"type": "object"}},
    },
    "PATCH /api/v1/pulse/analytics": {
        "type": "object",
        "properties": {"enabled": {"type": "boolean"}, "retention_days": {"type": "integer"}},
    },
    "POST /api/v1/pulse/analytics": {
        "type": "object",
        "properties": {"source": {"type": "string"}, "data": {"type": "object"}},
    },
    "PATCH /api/pulse/debate-topic": {
        "type": "object",
        "properties": {"active": {"type": "boolean"}},
    },
    "POST /api/pulse/debate-topic": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
    },
    "PATCH /api/v1/pulse/debate-topic": {
        "type": "object",
        "properties": {"active": {"type": "boolean"}},
    },
    "POST /api/v1/pulse/debate-topic": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
    },
    "PATCH /api/pulse/scheduler/config": {
        "type": "object",
        "properties": {"interval_minutes": {"type": "integer"}, "enabled": {"type": "boolean"}},
    },
    "POST /api/pulse/scheduler/config": {
        "type": "object",
        "properties": {"interval_minutes": {"type": "integer"}, "enabled": {"type": "boolean"}},
    },
    "PATCH /api/v1/pulse/scheduler/config": {
        "type": "object",
        "properties": {"interval_minutes": {"type": "integer"}, "enabled": {"type": "boolean"}},
    },
    "POST /api/v1/pulse/scheduler/config": {
        "type": "object",
        "properties": {"interval_minutes": {"type": "integer"}, "enabled": {"type": "boolean"}},
    },
    # ==========================================================================
    # Admin security endpoints
    # ==========================================================================
    "POST /api/v1/admin/security/keys": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": ["api_key", "signing_key", "encryption_key"]},
            "expires_at": {"type": "string", "format": "date-time"},
        },
    },
    "POST /api/admin/security/keys": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": ["api_key", "signing_key", "encryption_key"]},
            "expires_at": {"type": "string", "format": "date-time"},
        },
    },
    # ==========================================================================
    # OAuth auth flow endpoints
    # ==========================================================================
    "POST /api/integrations/slack/auth/url": {
        "type": "object",
        "properties": {
            "redirect_uri": {"type": "string"},
            "scopes": {"type": "array", "items": {"type": "string"}},
        },
    },
    "POST /api/v1/integrations/slack/auth/url": {
        "type": "object",
        "properties": {
            "redirect_uri": {"type": "string"},
            "scopes": {"type": "array", "items": {"type": "string"}},
        },
    },
    "POST /api/integrations/slack/auth/callback": {
        "type": "object",
        "properties": {"code": {"type": "string"}, "state": {"type": "string"}},
    },
    "POST /api/v1/integrations/slack/auth/callback": {
        "type": "object",
        "properties": {"code": {"type": "string"}, "state": {"type": "string"}},
    },
    "POST /api/integrations/teams/auth/url": {
        "type": "object",
        "properties": {
            "redirect_uri": {"type": "string"},
            "scopes": {"type": "array", "items": {"type": "string"}},
        },
    },
    "POST /api/v1/integrations/teams/auth/url": {
        "type": "object",
        "properties": {
            "redirect_uri": {"type": "string"},
            "scopes": {"type": "array", "items": {"type": "string"}},
        },
    },
    "POST /api/integrations/teams/auth/callback": {
        "type": "object",
        "properties": {"code": {"type": "string"}, "state": {"type": "string"}},
    },
    "POST /api/v1/integrations/teams/auth/callback": {
        "type": "object",
        "properties": {"code": {"type": "string"}, "state": {"type": "string"}},
    },
    # ==========================================================================
    # Personas options (read-only endpoints that shouldn't have POST body)
    # ==========================================================================
    "POST /api/personas/options": {"type": "object", "properties": {}},
    "PUT /api/personas/options": {"type": "object", "properties": {}},
    "POST /api/v1/personas/options": {"type": "object", "properties": {}},
    "PUT /api/v1/personas/options": {"type": "object", "properties": {}},
    # ==========================================================================
    # Onboarding endpoints
    # ==========================================================================
    "POST /api/v1/onboarding/progress": {
        "type": "object",
        "properties": {"step": {"type": "string"}, "completed": {"type": "boolean"}},
    },
    "POST /api/onboarding/progress": {
        "type": "object",
        "properties": {"step": {"type": "string"}, "completed": {"type": "boolean"}},
    },
    "PUT /api/v1/onboarding/preferences": {
        "type": "object",
        "properties": {"notifications": {"type": "boolean"}, "theme": {"type": "string"}},
    },
    "PUT /api/onboarding/preferences": {
        "type": "object",
        "properties": {"notifications": {"type": "boolean"}, "theme": {"type": "string"}},
    },
    # ==========================================================================
    # Tournament endpoints
    # ==========================================================================
    "POST /api/v1/tournaments": {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "agents": {"type": "array", "items": {"type": "string"}},
            "format": {"type": "string", "enum": ["round_robin", "single_elimination", "swiss"]},
        },
    },
    "POST /api/tournaments": {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "agents": {"type": "array", "items": {"type": "string"}},
            "format": {"type": "string", "enum": ["round_robin", "single_elimination", "swiss"]},
        },
    },
    # ==========================================================================
    # Training endpoints
    # ==========================================================================
    "POST /api/v1/training/examples": {
        "type": "object",
        "required": ["input", "output"],
        "properties": {
            "input": {"type": "string"},
            "output": {"type": "string"},
            "agent": {"type": "string"},
        },
    },
    "POST /api/training/examples": {
        "type": "object",
        "required": ["input", "output"],
        "properties": {
            "input": {"type": "string"},
            "output": {"type": "string"},
            "agent": {"type": "string"},
        },
    },
    # ==========================================================================
    # Media endpoints
    # ==========================================================================
    "POST /api/v1/media/upload": {
        "type": "object",
        "required": ["file"],
        "properties": {
            "file": {"type": "string", "format": "binary"},
            "type": {"type": "string", "enum": ["image", "audio", "video", "document"]},
        },
    },
    "POST /api/media/upload": {
        "type": "object",
        "required": ["file"],
        "properties": {
            "file": {"type": "string", "format": "binary"},
            "type": {"type": "string", "enum": ["image", "audio", "video", "document"]},
        },
    },
    # ==========================================================================
    # Deliberation endpoints
    # ==========================================================================
    "POST /api/v1/deliberations": {
        "type": "object",
        "required": ["topic"],
        "properties": {
            "topic": {"type": "string"},
            "participants": {"type": "array", "items": {"type": "string"}},
            "format": {"type": "string"},
        },
    },
    "POST /api/deliberations": {
        "type": "object",
        "required": ["topic"],
        "properties": {
            "topic": {"type": "string"},
            "participants": {"type": "array", "items": {"type": "string"}},
            "format": {"type": "string"},
        },
    },
    # ==========================================================================
    # Bot configuration endpoints
    # ==========================================================================
    "POST /api/v1/bots/telegram/config": {
        "type": "object",
        "properties": {"token": {"type": "string"}, "webhook_url": {"type": "string"}},
    },
    "POST /api/bots/telegram/config": {
        "type": "object",
        "properties": {"token": {"type": "string"}, "webhook_url": {"type": "string"}},
    },
    "POST /api/v1/bots/whatsapp/config": {
        "type": "object",
        "properties": {"phone_number_id": {"type": "string"}, "access_token": {"type": "string"}},
    },
    "POST /api/bots/whatsapp/config": {
        "type": "object",
        "properties": {"phone_number_id": {"type": "string"}, "access_token": {"type": "string"}},
    },
    "POST /api/v1/bots/zoom/config": {
        "type": "object",
        "properties": {"client_id": {"type": "string"}, "client_secret": {"type": "string"}},
    },
    "POST /api/bots/zoom/config": {
        "type": "object",
        "properties": {"client_id": {"type": "string"}, "client_secret": {"type": "string"}},
    },
    # ==========================================================================
    # Email endpoints
    # ==========================================================================
    "POST /api/v1/email/send": {
        "type": "object",
        "required": ["to", "subject"],
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "html": {"type": "boolean", "default": False},
        },
    },
    "POST /api/email/send": {
        "type": "object",
        "required": ["to", "subject"],
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "html": {"type": "boolean", "default": False},
        },
    },
    "POST /api/v1/email/followup": {
        "type": "object",
        "required": ["thread_id", "message"],
        "properties": {
            "thread_id": {"type": "string"},
            "message": {"type": "string"},
        },
    },
    "POST /api/email/followup": {
        "type": "object",
        "required": ["thread_id", "message"],
        "properties": {
            "thread_id": {"type": "string"},
            "message": {"type": "string"},
        },
    },
    # ==========================================================================
    # RLM endpoints
    # ==========================================================================
    "POST /api/v1/rlm/contexts": {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "content": {"type": "string"},
            "max_tokens": {"type": "integer"},
        },
    },
    "POST /api/rlm/contexts": {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "content": {"type": "string"},
            "max_tokens": {"type": "integer"},
        },
    },
    "POST /api/v1/rlm/strategies": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string"},
            "config": {"type": "object"},
        },
    },
    "POST /api/rlm/strategies": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string"},
            "config": {"type": "object"},
        },
    },
    # ==========================================================================
    # Dashboard endpoints
    # ==========================================================================
    "POST /api/v1/dashboard/widgets": {
        "type": "object",
        "required": ["type"],
        "properties": {
            "type": {"type": "string"},
            "title": {"type": "string"},
            "config": {"type": "object"},
            "position": {"type": "object"},
        },
    },
    "POST /api/dashboard/widgets": {
        "type": "object",
        "required": ["type"],
        "properties": {
            "type": {"type": "string"},
            "title": {"type": "string"},
            "config": {"type": "object"},
            "position": {"type": "object"},
        },
    },
    "PATCH /api/v1/dashboard/layout": {
        "type": "object",
        "properties": {
            "widgets": {"type": "array", "items": {"type": "object"}},
            "columns": {"type": "integer"},
        },
    },
    "PATCH /api/dashboard/layout": {
        "type": "object",
        "properties": {
            "widgets": {"type": "array", "items": {"type": "object"}},
            "columns": {"type": "integer"},
        },
    },
    # ==========================================================================
    # Repository endpoints
    # ==========================================================================
    "POST /api/v1/repository/scan": {
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {"type": "string"},
            "branch": {"type": "string", "default": "main"},
            "depth": {"type": "integer"},
        },
    },
    "POST /api/repository/scan": {
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {"type": "string"},
            "branch": {"type": "string", "default": "main"},
            "depth": {"type": "integer"},
        },
    },
    # ==========================================================================
    # Document query endpoints
    # ==========================================================================
    "POST /api/v1/documents/query": {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "filters": {"type": "object"},
            "limit": {"type": "integer", "default": 10},
        },
    },
    "POST /api/documents/query": {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "filters": {"type": "object"},
            "limit": {"type": "integer", "default": 10},
        },
    },
    "POST /api/v1/documents/summarize": {
        "type": "object",
        "required": ["document_id"],
        "properties": {
            "document_id": {"type": "string"},
            "max_length": {"type": "integer"},
        },
    },
    "POST /api/documents/summarize": {
        "type": "object",
        "required": ["document_id"],
        "properties": {
            "document_id": {"type": "string"},
            "max_length": {"type": "integer"},
        },
    },
    "POST /api/v1/documents/compare": {
        "type": "object",
        "required": ["document_ids"],
        "properties": {
            "document_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        },
    },
    "POST /api/documents/compare": {
        "type": "object",
        "required": ["document_ids"],
        "properties": {
            "document_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        },
    },
    # ==========================================================================
    # Cloud integration endpoints
    # ==========================================================================
    "POST /api/v1/cloud/storage/upload": {
        "type": "object",
        "required": ["file"],
        "properties": {
            "file": {"type": "string", "format": "binary"},
            "bucket": {"type": "string"},
            "path": {"type": "string"},
        },
    },
    "POST /api/cloud/storage/upload": {
        "type": "object",
        "required": ["file"],
        "properties": {
            "file": {"type": "string", "format": "binary"},
            "bucket": {"type": "string"},
            "path": {"type": "string"},
        },
    },
    # ==========================================================================
    # ML pipeline endpoints
    # ==========================================================================
    "POST /api/v1/ml/pipelines": {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "object"}},
            "config": {"type": "object"},
        },
    },
    "POST /api/ml/pipelines": {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "object"}},
            "config": {"type": "object"},
        },
    },
    # ==========================================================================
    # Evidence endpoints
    # ==========================================================================
    "POST /api/v1/evidence/collect": {
        "type": "object",
        "required": ["debate_id"],
        "properties": {
            "debate_id": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
            "depth": {"type": "string", "enum": ["shallow", "standard", "deep"]},
        },
    },
    "POST /api/evidence/collect": {
        "type": "object",
        "required": ["debate_id"],
        "properties": {
            "debate_id": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
            "depth": {"type": "string", "enum": ["shallow", "standard", "deep"]},
        },
    },
    # ==========================================================================
    # Learning endpoints
    # ==========================================================================
    "POST /api/v2/learning/sessions": {
        "type": "object",
        "required": ["topic"],
        "properties": {
            "topic": {"type": "string"},
            "mode": {"type": "string"},
            "config": {"type": "object"},
        },
    },
    "POST /api/v1/learning/sessions": {
        "type": "object",
        "required": ["topic"],
        "properties": {
            "topic": {"type": "string"},
            "mode": {"type": "string"},
            "config": {"type": "object"},
        },
    },
    # ==========================================================================
    # Agent selection endpoints
    # ==========================================================================
    "POST /api/v1/agent-selection/evaluate": {
        "type": "object",
        "required": ["task"],
        "properties": {
            "task": {"type": "string"},
            "candidates": {"type": "array", "items": {"type": "string"}},
            "criteria": {"type": "object"},
        },
    },
    "POST /api/agent-selection/evaluate": {
        "type": "object",
        "required": ["task"],
        "properties": {
            "task": {"type": "string"},
            "candidates": {"type": "array", "items": {"type": "string"}},
            "criteria": {"type": "object"},
        },
    },
    # ==========================================================================
    # Export endpoints (common pattern)
    # ==========================================================================
    "POST /api/v1/debates/{id}/export": {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["json", "csv", "pdf", "markdown"],
                "default": "json",
            },
            "include_metadata": {"type": "boolean", "default": True},
        },
    },
    "POST /api/debates/{id}/export": {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["json", "csv", "pdf", "markdown"],
                "default": "json",
            },
            "include_metadata": {"type": "boolean", "default": True},
        },
    },
    # ==========================================================================
    # Accounting endpoints
    # ==========================================================================
    "POST /api/v1/accounting/expenses/upload": {
        "type": "object",
        "required": ["file"],
        "properties": {
            "file": {"type": "string", "format": "binary"},
            "category": {"type": "string"},
        },
    },
    "POST /api/accounting/expenses/upload": {
        "type": "object",
        "required": ["file"],
        "properties": {
            "file": {"type": "string", "format": "binary"},
            "category": {"type": "string"},
        },
    },
    "POST /api/v1/accounting/invoices": {
        "type": "object",
        "required": ["customer_id", "amount"],
        "properties": {
            "customer_id": {"type": "string"},
            "amount": {"type": "number"},
            "description": {"type": "string"},
            "due_date": {"type": "string", "format": "date"},
        },
    },
    "POST /api/accounting/invoices": {
        "type": "object",
        "required": ["customer_id", "amount"],
        "properties": {
            "customer_id": {"type": "string"},
            "amount": {"type": "number"},
            "description": {"type": "string"},
            "due_date": {"type": "string", "format": "date"},
        },
    },
    # ==========================================================================
    # Gmail message operations
    # ==========================================================================
    "POST /api/v1/gmail/messages/{message_id}/mark": {
        "type": "object",
        "properties": {
            "read": {"type": "boolean"},
            "starred": {"type": "boolean"},
            "labels": {"type": "array", "items": {"type": "string"}},
        },
    },
    "POST /api/v1/gmail/messages/send": {
        "type": "object",
        "required": ["to", "subject", "body"],
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "cc": {"type": "string"},
            "bcc": {"type": "string"},
        },
    },
    "POST /api/v1/gmail/drafts": {
        "type": "object",
        "required": ["to", "subject"],
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
    },
    # ==========================================================================
    # Debate advanced operations
    # ==========================================================================
    "POST /api/v1/debates/{id}/fork": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "from_round": {"type": "integer"},
            "modifications": {"type": "object"},
        },
    },
    "POST /api/debates/{id}/fork": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "from_round": {"type": "integer"},
            "modifications": {"type": "object"},
        },
    },
    "POST /api/v1/debates/{id}/graph": {
        "type": "object",
        "properties": {
            "layout": {"type": "string", "enum": ["force", "tree", "radial"]},
            "include_evidence": {"type": "boolean", "default": True},
        },
    },
    "POST /api/debates/{id}/graph": {
        "type": "object",
        "properties": {
            "layout": {"type": "string", "enum": ["force", "tree", "radial"]},
            "include_evidence": {"type": "boolean", "default": True},
        },
    },
    "POST /api/v1/debates/{id}/hybrid": {
        "type": "object",
        "properties": {
            "human_input": {"type": "string"},
            "agent_override": {"type": "string"},
        },
    },
    "POST /api/debates/{id}/hybrid": {
        "type": "object",
        "properties": {
            "human_input": {"type": "string"},
            "agent_override": {"type": "string"},
        },
    },
    # ==========================================================================
    # Audit export endpoints
    # ==========================================================================
    "POST /api/v1/audit/export": {
        "type": "object",
        "properties": {
            "format": {"type": "string", "enum": ["json", "csv", "pdf"]},
            "start_date": {"type": "string", "format": "date-time"},
            "end_date": {"type": "string", "format": "date-time"},
            "filters": {"type": "object"},
        },
    },
    "POST /api/audit/export": {
        "type": "object",
        "properties": {
            "format": {"type": "string", "enum": ["json", "csv", "pdf"]},
            "start_date": {"type": "string", "format": "date-time"},
            "end_date": {"type": "string", "format": "date-time"},
            "filters": {"type": "object"},
        },
    },
    # ==========================================================================
    # Workflow execution endpoints
    # ==========================================================================
    "POST /api/v1/workflows/{id}/execute": {
        "type": "object",
        "properties": {
            "inputs": {"type": "object"},
            "async": {"type": "boolean", "default": False},
        },
    },
    "POST /api/workflows/{id}/execute": {
        "type": "object",
        "properties": {
            "inputs": {"type": "object"},
            "async": {"type": "boolean", "default": False},
        },
    },
    "POST /api/v1/workflows/{id}/cancel": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
    "POST /api/workflows/{id}/cancel": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
}

# HTTP methods that can have request bodies
BODY_METHODS = {"post", "put", "patch"}


def _infer_schema_from_handler(handler_dir: Path, operation_id: str) -> dict | None:
    """Try to infer request body schema from handler source code."""
    if not handler_dir.exists():
        return None

    # Search handler files for validate_body or request_body patterns
    for py_file in handler_dir.rglob("*.py"):
        try:
            source = py_file.read_text()
        except Exception:
            continue

        # Look for @validate_body(required_fields=[...]) decorator
        validate_match = re.search(r"@validate_body\(\s*required_fields\s*=\s*\[([^\]]+)\]", source)
        if validate_match and operation_id and operation_id.lower() in source.lower():
            fields_str = validate_match.group(1)
            fields = [f.strip().strip("'\"") for f in fields_str.split(",")]
            properties = {}
            for field in fields:
                if field:
                    properties[field] = {"type": "string"}
            if properties:
                return {
                    "type": "object",
                    "required": [f for f in fields if f],
                    "properties": properties,
                }

    return None


def add_request_schemas(
    spec: dict,
    handler_dir: Path | None = None,
    verbose: bool = False,
) -> tuple[dict, int, int, int]:
    """Add request body schemas to operations missing them.

    Returns: (updated_spec, added_known, added_inferred, added_generic)
    """
    added_known = 0
    added_inferred = 0
    added_generic = 0

    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue
            if method.lower() not in BODY_METHODS:
                continue

            # Check existing requestBody
            request_body = details.get("requestBody", {})
            has_proper_schema = False
            if isinstance(request_body, dict):
                content = request_body.get("content", {})
                if isinstance(content, dict):
                    json_content = content.get("application/json", {})
                    if isinstance(json_content, dict):
                        schema = json_content.get("schema", {})
                        # Check if schema is "proper" (has properties or $ref)
                        # vs generic (only additionalProperties: true)
                        if schema.get("properties") or schema.get("$ref"):
                            has_proper_schema = True
                        elif schema.get("additionalProperties") is True:
                            # This is a generic placeholder - can be upgraded
                            pass
                        elif schema.get("type") and schema.get("type") != "object":
                            # Has specific type definition
                            has_proper_schema = True

            if has_proper_schema:
                continue  # Already has proper schema

            # Try known schemas first
            key = f"{method.upper()} {path}"
            if key in KNOWN_SCHEMAS:
                details["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": KNOWN_SCHEMAS[key],
                        }
                    },
                }
                added_known += 1
                if verbose:
                    print(f"  [known] {key}")
                continue

            # Try to infer from handler source
            if handler_dir:
                operation_id = details.get("operationId", "")
                inferred = _infer_schema_from_handler(handler_dir, operation_id)
                if inferred:
                    details["requestBody"] = {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": inferred,
                            }
                        },
                    }
                    added_inferred += 1
                    if verbose:
                        print(f"  [inferred] {key}")
                    continue

            # Add generic schema as placeholder
            details["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            # Empty properties keeps this schema explicit and
                            # analyzable while still allowing arbitrary fields.
                            "properties": {},
                            "additionalProperties": True,
                            "description": f"Request body for {method.upper()} {path}",
                        }
                    }
                },
            }
            added_generic += 1
            if verbose:
                print(f"  [generic] {key}")

    return spec, added_known, added_inferred, added_generic


def main() -> None:
    parser = argparse.ArgumentParser(description="Add request body schemas to OpenAPI spec")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("docs/api/openapi.json"),
        help="Path to OpenAPI JSON spec",
    )
    parser.add_argument(
        "--handlers",
        type=Path,
        default=Path("aragora/server/handlers"),
        help="Path to handler source directory for schema inference",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    parser.add_argument("--verbose", action="store_true", help="Show each operation")
    args = parser.parse_args()

    spec_path = args.spec
    if not spec_path.exists():
        print(f"Error: {spec_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {spec_path}...")
    with open(spec_path) as f:
        spec = json.load(f)

    handler_dir = args.handlers if args.handlers.exists() else None

    spec, known, inferred, generic = add_request_schemas(
        spec, handler_dir=handler_dir, verbose=args.verbose
    )

    total = known + inferred + generic
    print(f"\nRequest schemas added: {total}")
    print(f"  Known schemas:    {known}")
    print(f"  Inferred schemas: {inferred}")
    print(f"  Generic schemas:  {generic}")

    if args.dry_run:
        print("\n(dry-run: no changes written)")
    else:
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"\nWrote updated spec to {spec_path}")


if __name__ == "__main__":
    main()
