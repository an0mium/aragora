"""Auto-generated missing SDK endpoints.

This module aggregates SDK endpoint definitions from category-specific modules.
The endpoints are split into logical categories for maintainability:

- sdk_missing_core.py: Helper functions and shared utilities
- sdk_missing_costs.py: Costs, Payments, and Accounting endpoints
- sdk_missing_compliance.py: Compliance, Policies, Audit, and Privacy endpoints
- sdk_missing_analytics.py: Analytics endpoints
- sdk_missing_integration.py: Integrations, Webhooks, and Connectors endpoints

For backward compatibility, all endpoints are re-exported from this module.
"""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

# Import categorized endpoints from submodules
from aragora.server.openapi.endpoints.sdk_missing_core import _method_stub
from aragora.server.openapi.endpoints.sdk_missing_costs import SDK_MISSING_COSTS_ENDPOINTS
from aragora.server.openapi.endpoints.sdk_missing_compliance import SDK_MISSING_COMPLIANCE_ENDPOINTS
from aragora.server.openapi.endpoints.sdk_missing_analytics import SDK_MISSING_ANALYTICS_ENDPOINTS
from aragora.server.openapi.endpoints.sdk_missing_integration import (
    SDK_MISSING_INTEGRATION_ENDPOINTS,
)


# Build the main endpoint dictionary by combining all categorized endpoints
SDK_MISSING_ENDPOINTS: dict = {}

# Merge categorized endpoints
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_COSTS_ENDPOINTS)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_COMPLIANCE_ENDPOINTS)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_ANALYTICS_ENDPOINTS)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_INTEGRATION_ENDPOINTS)


# =============================================================================
# Response Schemas for _OTHER_ENDPOINTS
# =============================================================================

# Debates - Replay schemas
_DEBATE_REPLAY_SCHEMA = {
    "replay_id": {"type": "string", "description": "Unique replay identifier"},
    "debate_id": {"type": "string", "description": "Source debate identifier"},
    "events": {
        "type": "array",
        "description": "Ordered list of debate events",
        "items": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "type": {
                    "type": "string",
                    "enum": [
                        "debate_start",
                        "round_start",
                        "agent_message",
                        "critique",
                        "vote",
                        "consensus",
                        "debate_end",
                    ],
                },
                "timestamp": {"type": "string", "format": "date-time"},
                "agent_id": {"type": "string"},
                "content": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
    },
    "duration_ms": {"type": "integer", "description": "Total replay duration in milliseconds"},
    "total_rounds": {"type": "integer", "description": "Number of debate rounds"},
    "participants": {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of participating agent IDs",
    },
    "created_at": {"type": "string", "format": "date-time"},
    "status": {"type": "string", "enum": ["ready", "processing", "error"]},
}

# Keys schemas
_API_KEY_SCHEMA = {
    "id": {"type": "string", "description": "Key identifier"},
    "name": {"type": "string", "description": "Human-readable key name"},
    "prefix": {"type": "string", "description": "Key prefix (e.g., 'sk_live_')"},
    "created_at": {"type": "string", "format": "date-time"},
    "expires_at": {"type": "string", "format": "date-time"},
    "last_used_at": {"type": "string", "format": "date-time"},
    "scopes": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Authorized permission scopes",
    },
    "status": {"type": "string", "enum": ["active", "expired", "revoked"]},
}

_API_KEYS_LIST_SCHEMA = {
    "keys": {"type": "array", "items": {"type": "object", "properties": _API_KEY_SCHEMA}},
    "total": {"type": "integer"},
}

# Knowledge Mound schemas
_KNOWLEDGE_NODE_SCHEMA = {
    "id": {"type": "string", "description": "Node identifier"},
    "type": {
        "type": "string",
        "enum": ["fact", "claim", "evidence", "concept", "entity"],
        "description": "Node classification",
    },
    "content": {"type": "string", "description": "Node content text"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "source_debate_id": {"type": "string"},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"},
    "metadata": {"type": "object"},
    "tags": {"type": "array", "items": {"type": "string"}},
}

_NODE_RELATIONSHIPS_SCHEMA = {
    "node_id": {"type": "string"},
    "relationships": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "relationship_id": {"type": "string"},
                "type": {
                    "type": "string",
                    "enum": [
                        "supports",
                        "contradicts",
                        "related_to",
                        "derives_from",
                        "part_of",
                    ],
                },
                "target_node_id": {"type": "string"},
                "strength": {"type": "number", "minimum": 0, "maximum": 1},
                "created_at": {"type": "string", "format": "date-time"},
            },
        },
    },
    "total_relationships": {"type": "integer"},
}

# Leaderboard schemas
_AGENT_STATS_SCHEMA = {
    "agent_id": {"type": "string"},
    "name": {"type": "string"},
    "elo_rating": {"type": "number"},
    "total_debates": {"type": "integer"},
    "wins": {"type": "integer"},
    "losses": {"type": "integer"},
    "draws": {"type": "integer"},
    "win_rate": {"type": "number", "minimum": 0, "maximum": 1},
    "average_score": {"type": "number"},
    "domains": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Expertise domains",
    },
    "rank": {"type": "integer"},
    "tier": {"type": "string", "enum": ["bronze", "silver", "gold", "platinum", "diamond"]},
}

_ELO_HISTORY_SCHEMA = {
    "agent_id": {"type": "string"},
    "history": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string", "format": "date-time"},
                "rating": {"type": "number"},
                "change": {"type": "number"},
                "debate_id": {"type": "string"},
                "opponent_id": {"type": "string"},
                "result": {"type": "string", "enum": ["win", "loss", "draw"]},
            },
        },
    },
    "current_rating": {"type": "number"},
    "peak_rating": {"type": "number"},
    "lowest_rating": {"type": "number"},
}

_AGENT_COMPARISON_SCHEMA = {
    "agents": {
        "type": "array",
        "items": {"type": "object", "properties": _AGENT_STATS_SCHEMA},
    },
    "head_to_head": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "agent_a": {"type": "string"},
                "agent_b": {"type": "string"},
                "a_wins": {"type": "integer"},
                "b_wins": {"type": "integer"},
                "draws": {"type": "integer"},
            },
        },
    },
}

_DOMAIN_LEADERBOARD_SCHEMA = {
    "domain": {"type": "string"},
    "agents": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "name": {"type": "string"},
                "domain_elo": {"type": "number"},
                "domain_debates": {"type": "integer"},
                "domain_win_rate": {"type": "number"},
                "rank": {"type": "integer"},
            },
        },
    },
    "total_agents": {"type": "integer"},
}

_DOMAINS_LIST_SCHEMA = {
    "domains": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "total_debates": {"type": "integer"},
                "active_agents": {"type": "integer"},
            },
        },
    },
}

_TOP_MOVERS_SCHEMA = {
    "period": {"type": "string", "enum": ["day", "week", "month"]},
    "gainers": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "name": {"type": "string"},
                "rating_change": {"type": "number"},
                "new_rating": {"type": "number"},
                "debates_in_period": {"type": "integer"},
            },
        },
    },
    "losers": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "name": {"type": "string"},
                "rating_change": {"type": "number"},
                "new_rating": {"type": "number"},
                "debates_in_period": {"type": "integer"},
            },
        },
    },
}

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

# Personas schemas
_PERSONA_SCHEMA = {
    "id": {"type": "string"},
    "name": {"type": "string"},
    "description": {"type": "string"},
    "traits": {"type": "array", "items": {"type": "string"}},
    "communication_style": {
        "type": "string",
        "enum": ["formal", "casual", "technical", "friendly"],
    },
    "expertise_domains": {"type": "array", "items": {"type": "string"}},
    "tone": {"type": "string"},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"},
}

_PERSONA_OPTIONS_SCHEMA = {
    "communication_styles": {"type": "array", "items": {"type": "string"}},
    "available_traits": {"type": "array", "items": {"type": "string"}},
    "expertise_domains": {"type": "array", "items": {"type": "string"}},
    "tone_options": {"type": "array", "items": {"type": "string"}},
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

# Replays schemas
_REPLAY_LIST_SCHEMA = {
    "replays": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "debate_id": {"type": "string"},
                "title": {"type": "string"},
                "duration_ms": {"type": "integer"},
                "created_at": {"type": "string", "format": "date-time"},
                "view_count": {"type": "integer"},
                "is_public": {"type": "boolean"},
            },
        },
    },
    "total": {"type": "integer"},
    "page": {"type": "integer"},
    "page_size": {"type": "integer"},
}

_REPLAY_DETAIL_SCHEMA = {
    "id": {"type": "string"},
    "debate_id": {"type": "string"},
    "title": {"type": "string"},
    "description": {"type": "string"},
    "events": {"type": "array", "items": {"type": "object"}},
    "duration_ms": {"type": "integer"},
    "total_rounds": {"type": "integer"},
    "participants": {"type": "array", "items": {"type": "string"}},
    "created_at": {"type": "string", "format": "date-time"},
    "view_count": {"type": "integer"},
    "is_public": {"type": "boolean"},
    "bookmarked": {"type": "boolean"},
}

_CREATE_REPLAY_RESPONSE = {
    "replay_id": {"type": "string"},
    "debate_id": {"type": "string"},
    "status": {"type": "string", "enum": ["processing", "ready", "error"]},
    "estimated_ready_at": {"type": "string", "format": "date-time"},
}

_SHARE_LINK_SCHEMA = {
    "share_id": {"type": "string"},
    "replay_id": {"type": "string"},
    "url": {"type": "string", "format": "uri"},
    "expires_at": {"type": "string", "format": "date-time"},
    "access_count": {"type": "integer"},
    "max_accesses": {"type": "integer"},
}

_BOOKMARK_RESPONSE = {
    "success": {"type": "boolean"},
    "replay_id": {"type": "string"},
    "bookmarked": {"type": "boolean"},
    "bookmarked_at": {"type": "string", "format": "date-time"},
}

_REPLAY_COMMENTS_SCHEMA = {
    "comments": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "user_id": {"type": "string"},
                "user_name": {"type": "string"},
                "content": {"type": "string"},
                "timestamp_ms": {"type": "integer", "description": "Position in replay"},
                "created_at": {"type": "string", "format": "date-time"},
                "edited": {"type": "boolean"},
            },
        },
    },
    "total": {"type": "integer"},
}

_ADD_COMMENT_RESPONSE = {
    "comment_id": {"type": "string"},
    "replay_id": {"type": "string"},
    "created_at": {"type": "string", "format": "date-time"},
}

_EXPORT_REPLAY_SCHEMA = {
    "export_id": {"type": "string"},
    "replay_id": {"type": "string"},
    "format": {"type": "string", "enum": ["json", "markdown", "pdf", "html"]},
    "download_url": {"type": "string", "format": "uri"},
    "expires_at": {"type": "string", "format": "date-time"},
    "size_bytes": {"type": "integer"},
}

_REPLAY_SUMMARY_SCHEMA = {
    "replay_id": {"type": "string"},
    "title": {"type": "string"},
    "summary": {"type": "string"},
    "key_points": {"type": "array", "items": {"type": "string"}},
    "outcome": {"type": "string"},
    "consensus_reached": {"type": "boolean"},
    "winning_agent": {"type": "string"},
}

_REPLAY_TRANSCRIPT_SCHEMA = {
    "replay_id": {"type": "string"},
    "format": {"type": "string", "enum": ["plain", "markdown", "annotated"]},
    "transcript": {"type": "string"},
    "word_count": {"type": "integer"},
    "sections": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "round": {"type": "integer"},
                "speaker": {"type": "string"},
                "content": {"type": "string"},
                "timestamp_ms": {"type": "integer"},
            },
        },
    },
}

# Routing schemas
_DOMAIN_LEADERBOARD_ROUTING_SCHEMA = {
    "domains": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "top_agents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string"},
                            "score": {"type": "number"},
                            "debates": {"type": "integer"},
                        },
                    },
                },
                "routing_weight": {"type": "number"},
            },
        },
    },
}

# Skills schemas
_SKILL_SCHEMA = {
    "id": {"type": "string"},
    "name": {"type": "string"},
    "description": {"type": "string"},
    "version": {"type": "string"},
    "author": {"type": "string"},
    "category": {"type": "string"},
    "capabilities": {"type": "array", "items": {"type": "string"}},
    "parameters": {"type": "object"},
    "installed": {"type": "boolean"},
    "enabled": {"type": "boolean"},
    "created_at": {"type": "string", "format": "date-time"},
}

# Users schemas
_USER_ME_SCHEMA = {
    "id": {"type": "string"},
    "email": {"type": "string", "format": "email"},
    "name": {"type": "string"},
    "avatar_url": {"type": "string", "format": "uri"},
    "created_at": {"type": "string", "format": "date-time"},
    "last_login_at": {"type": "string", "format": "date-time"},
    "roles": {"type": "array", "items": {"type": "string"}},
    "workspace_id": {"type": "string"},
    "tenant_id": {"type": "string"},
}

_USER_PREFERENCES_SCHEMA = {
    "theme": {"type": "string", "enum": ["light", "dark", "system"]},
    "language": {"type": "string"},
    "timezone": {"type": "string"},
    "notification_settings": {
        "type": "object",
        "properties": {
            "email": {"type": "boolean"},
            "push": {"type": "boolean"},
            "in_app": {"type": "boolean"},
        },
    },
    "default_debate_settings": {
        "type": "object",
        "properties": {
            "rounds": {"type": "integer"},
            "consensus_threshold": {"type": "number"},
        },
    },
}

_USER_PROFILE_SCHEMA = {
    "id": {"type": "string"},
    "email": {"type": "string", "format": "email"},
    "name": {"type": "string"},
    "display_name": {"type": "string"},
    "bio": {"type": "string"},
    "avatar_url": {"type": "string", "format": "uri"},
    "company": {"type": "string"},
    "location": {"type": "string"},
    "website": {"type": "string", "format": "uri"},
    "social_links": {"type": "object"},
    "updated_at": {"type": "string", "format": "date-time"},
}

_USER_DEBATES_SCHEMA = {
    "debates": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "status": {"type": "string", "enum": ["active", "completed", "cancelled"]},
                "role": {"type": "string", "enum": ["creator", "participant", "observer"]},
                "created_at": {"type": "string", "format": "date-time"},
                "completed_at": {"type": "string", "format": "date-time"},
            },
        },
    },
    "total": {"type": "integer"},
    "page": {"type": "integer"},
    "page_size": {"type": "integer"},
}

# SCIM schemas
_SCIM_GROUPS_SCHEMA = {
    "schemas": {"type": "array", "items": {"type": "string"}},
    "totalResults": {"type": "integer"},
    "startIndex": {"type": "integer"},
    "itemsPerPage": {"type": "integer"},
    "Resources": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "schemas": {"type": "array", "items": {"type": "string"}},
                "id": {"type": "string"},
                "displayName": {"type": "string"},
                "members": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "display": {"type": "string"},
                        },
                    },
                },
                "meta": {
                    "type": "object",
                    "properties": {
                        "resourceType": {"type": "string"},
                        "created": {"type": "string", "format": "date-time"},
                        "lastModified": {"type": "string", "format": "date-time"},
                        "location": {"type": "string", "format": "uri"},
                    },
                },
            },
        },
    },
}

_SCIM_USERS_SCHEMA = {
    "schemas": {"type": "array", "items": {"type": "string"}},
    "totalResults": {"type": "integer"},
    "startIndex": {"type": "integer"},
    "itemsPerPage": {"type": "integer"},
    "Resources": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "schemas": {"type": "array", "items": {"type": "string"}},
                "id": {"type": "string"},
                "userName": {"type": "string"},
                "name": {
                    "type": "object",
                    "properties": {
                        "givenName": {"type": "string"},
                        "familyName": {"type": "string"},
                        "formatted": {"type": "string"},
                    },
                },
                "emails": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string", "format": "email"},
                            "type": {"type": "string"},
                            "primary": {"type": "boolean"},
                        },
                    },
                },
                "active": {"type": "boolean"},
                "meta": {
                    "type": "object",
                    "properties": {
                        "resourceType": {"type": "string"},
                        "created": {"type": "string", "format": "date-time"},
                        "lastModified": {"type": "string", "format": "date-time"},
                        "location": {"type": "string", "format": "uri"},
                    },
                },
            },
        },
    },
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

_CREATE_PERSONA_REQUEST = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "traits": {"type": "array", "items": {"type": "string"}},
        "communication_style": {"type": "string"},
        "expertise_domains": {"type": "array", "items": {"type": "string"}},
        "tone": {"type": "string"},
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

_CREATE_REPLAY_REQUEST = {
    "type": "object",
    "required": ["debate_id"],
    "properties": {
        "debate_id": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "is_public": {"type": "boolean"},
    },
}

_BOOKMARK_REQUEST = {
    "type": "object",
    "properties": {
        "note": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}

_ADD_COMMENT_REQUEST = {
    "type": "object",
    "required": ["content"],
    "properties": {
        "content": {"type": "string"},
        "timestamp_ms": {"type": "integer"},
    },
}

_SHARE_REPLAY_REQUEST = {
    "type": "object",
    "properties": {
        "expires_in_hours": {"type": "integer"},
        "max_accesses": {"type": "integer"},
        "password": {"type": "string"},
    },
}

_UPDATE_PROFILE_REQUEST = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "display_name": {"type": "string"},
        "bio": {"type": "string"},
        "avatar_url": {"type": "string", "format": "uri"},
        "company": {"type": "string"},
        "location": {"type": "string"},
        "website": {"type": "string", "format": "uri"},
        "social_links": {"type": "object"},
    },
}


# Additional endpoints that don't fit into the main categories
_OTHER_ENDPOINTS: dict = {
    "/api/debates/{id}/replay": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get debate replay",
            "description": "Retrieve replay data for a completed debate including all events and timing",
            "operationId": "getDebatesReplay",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Debate ID",
                }
            ],
            "responses": {
                "200": _ok_response("Debate replay data", _DEBATE_REPLAY_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/keys": {
        "get": {
            "tags": ["Keys"],
            "summary": "List API keys",
            "description": "List all API keys for the current user or workspace",
            "operationId": "getKeys",
            "responses": {
                "200": _ok_response("List of API keys", _API_KEYS_LIST_SCHEMA),
            },
        },
    },
    "/api/keys/{id}": {
        "delete": {
            "tags": ["Keys"],
            "summary": "Delete API key",
            "description": "Revoke and delete an API key",
            "operationId": "deleteKeys",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Key ID",
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Key deleted", {"deleted": {"type": "boolean"}, "id": {"type": "string"}}
                ),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/knowledge/mound/nodes/{id}": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "Get knowledge node",
            "description": "Retrieve a specific knowledge node from the Knowledge Mound",
            "operationId": "getMoundNodes",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Node ID",
                }
            ],
            "responses": {
                "200": _ok_response("Knowledge node details", _KNOWLEDGE_NODE_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/knowledge/mound/nodes/{id}/relationships": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "Get node relationships",
            "description": "Retrieve all relationships for a knowledge node",
            "operationId": "getNodesRelationships",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Node ID",
                }
            ],
            "responses": {
                "200": _ok_response("Node relationships", _NODE_RELATIONSHIPS_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/leaderboard/agent/{id}": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "Get agent statistics",
            "description": "Retrieve detailed statistics for a specific agent",
            "operationId": "getLeaderboardAgent",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Agent ID",
                }
            ],
            "responses": {
                "200": _ok_response("Agent statistics", _AGENT_STATS_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/leaderboard/agent/{id}/elo-history": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "Get agent ELO history",
            "description": "Retrieve historical ELO rating changes for an agent",
            "operationId": "getAgentEloHistory",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Agent ID",
                }
            ],
            "responses": {
                "200": _ok_response("ELO rating history", _ELO_HISTORY_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/leaderboard/compare": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "Compare agents",
            "description": "Compare statistics between multiple agents including head-to-head records",
            "operationId": "getLeaderboardCompare",
            "parameters": [
                {
                    "name": "agent_ids",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "array", "items": {"type": "string"}},
                    "description": "Agent IDs to compare",
                }
            ],
            "responses": {
                "200": _ok_response("Agent comparison", _AGENT_COMPARISON_SCHEMA),
            },
        },
    },
    "/api/leaderboard/domain/{id}": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "Get domain leaderboard",
            "description": "Retrieve the leaderboard for a specific expertise domain",
            "operationId": "getLeaderboardDomain",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Domain ID",
                }
            ],
            "responses": {
                "200": _ok_response("Domain leaderboard", _DOMAIN_LEADERBOARD_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/leaderboard/domains": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "List domains",
            "description": "List all available expertise domains with statistics",
            "operationId": "getLeaderboardDomains",
            "responses": {
                "200": _ok_response("List of domains", _DOMAINS_LIST_SCHEMA),
            },
        },
    },
    "/api/leaderboard/movers": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "Get top movers",
            "description": "Get agents with the biggest rating changes in a period",
            "operationId": "getLeaderboardMovers",
            "parameters": [
                {
                    "name": "period",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["day", "week", "month"]},
                    "description": "Time period",
                }
            ],
            "responses": {
                "200": _ok_response("Top movers", _TOP_MOVERS_SCHEMA),
            },
        },
    },
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
    "/api/personas": {
        "post": {
            "tags": ["Personas"],
            "summary": "Create persona",
            "description": "Create a new agent persona with specified traits and style",
            "operationId": "postPersonas",
            "requestBody": {
                "content": {"application/json": {"schema": _CREATE_PERSONA_REQUEST}},
                "required": True,
            },
            "responses": {
                "200": _ok_response("Created persona", _PERSONA_SCHEMA),
            },
        },
    },
    "/api/personas/options": {
        "get": {
            "tags": ["Personas"],
            "summary": "Get persona options",
            "description": "Get available options for creating personas (traits, styles, domains)",
            "operationId": "getPersonasOptions",
            "responses": {
                "200": _ok_response("Persona configuration options", _PERSONA_OPTIONS_SCHEMA),
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
    "/api/replays": {
        "get": {
            "tags": ["Replays"],
            "summary": "List replays",
            "description": "List all available debate replays",
            "operationId": "getReplays",
            "parameters": [
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
                {
                    "name": "is_public",
                    "in": "query",
                    "schema": {"type": "boolean"},
                    "description": "Filter by public status",
                },
            ],
            "responses": {
                "200": _ok_response("List of replays", _REPLAY_LIST_SCHEMA),
            },
        },
    },
    "/api/replays/create": {
        "post": {
            "tags": ["Replays"],
            "summary": "Create replay",
            "description": "Create a new replay from a completed debate",
            "operationId": "postReplaysCreate",
            "requestBody": {
                "content": {"application/json": {"schema": _CREATE_REPLAY_REQUEST}},
                "required": True,
            },
            "responses": {
                "200": _ok_response("Replay creation started", _CREATE_REPLAY_RESPONSE),
            },
        },
    },
    "/api/replays/share/{id}": {
        "get": {
            "tags": ["Replays"],
            "summary": "Get shared replay",
            "description": "Access a replay via share link",
            "operationId": "getReplaysShare",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Share ID",
                }
            ],
            "responses": {
                "200": _ok_response("Shared replay data", _REPLAY_DETAIL_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/replays/{id}": {
        "get": {
            "tags": ["Replays"],
            "summary": "Get replay",
            "description": "Get detailed replay data including all events",
            "operationId": "getReplayById",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Replay ID",
                }
            ],
            "responses": {
                "200": _ok_response("Replay details", _REPLAY_DETAIL_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/replays/{id}/bookmark": {
        "delete": {
            "tags": ["Replays"],
            "summary": "Remove bookmark",
            "description": "Remove a bookmark from a replay",
            "operationId": "deleteReplaysBookmark",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Replay ID",
                }
            ],
            "responses": {
                "200": _ok_response("Bookmark removed", _BOOKMARK_RESPONSE),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Replays"],
            "summary": "Add bookmark",
            "description": "Bookmark a replay for easy access",
            "operationId": "postReplaysBookmark",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Replay ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _BOOKMARK_REQUEST}}},
            "responses": {
                "200": _ok_response("Bookmark added", _BOOKMARK_RESPONSE),
            },
        },
    },
    "/api/replays/{id}/comments": {
        "get": {
            "tags": ["Replays"],
            "summary": "Get comments",
            "description": "Get all comments on a replay",
            "operationId": "getReplaysComments",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Replay ID",
                }
            ],
            "responses": {
                "200": _ok_response("Replay comments", _REPLAY_COMMENTS_SCHEMA),
            },
        },
        "post": {
            "tags": ["Replays"],
            "summary": "Add comment",
            "description": "Add a comment to a replay at a specific timestamp",
            "operationId": "postReplaysComments",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Replay ID",
                }
            ],
            "requestBody": {
                "content": {"application/json": {"schema": _ADD_COMMENT_REQUEST}},
                "required": True,
            },
            "responses": {
                "200": _ok_response("Comment added", _ADD_COMMENT_RESPONSE),
            },
        },
    },
    "/api/replays/{id}/export": {
        "get": {
            "tags": ["Replays"],
            "summary": "Export replay",
            "description": "Export replay in various formats (JSON, Markdown, PDF)",
            "operationId": "getReplaysExport",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Replay ID",
                },
                {
                    "name": "format",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["json", "markdown", "pdf", "html"]},
                    "description": "Export format",
                },
            ],
            "responses": {
                "200": _ok_response("Export details", _EXPORT_REPLAY_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/replays/{id}/share": {
        "post": {
            "tags": ["Replays"],
            "summary": "Create share link",
            "description": "Create a shareable link for a replay",
            "operationId": "postReplaysShare",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Replay ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _SHARE_REPLAY_REQUEST}}},
            "responses": {
                "200": _ok_response("Share link created", _SHARE_LINK_SCHEMA),
            },
        },
    },
    "/api/replays/{id}/summary": {
        "get": {
            "tags": ["Replays"],
            "summary": "Get replay summary",
            "description": "Get an AI-generated summary of the replay",
            "operationId": "getReplaysSummary",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Replay ID",
                }
            ],
            "responses": {
                "200": _ok_response("Replay summary", _REPLAY_SUMMARY_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/replays/{id}/transcript": {
        "get": {
            "tags": ["Replays"],
            "summary": "Get replay transcript",
            "description": "Get full text transcript of the replay",
            "operationId": "getReplaysTranscript",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Replay ID",
                },
                {
                    "name": "format",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["plain", "markdown", "annotated"]},
                    "description": "Transcript format",
                },
            ],
            "responses": {
                "200": _ok_response("Replay transcript", _REPLAY_TRANSCRIPT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/routing/domain-leaderboard": {
        "get": {
            "tags": ["Routing"],
            "summary": "Get domain leaderboard for routing",
            "description": "Get agent rankings per domain used for intelligent routing",
            "operationId": "getRoutingDomainLeaderboard",
            "responses": {
                "200": _ok_response(
                    "Domain leaderboard for routing", _DOMAIN_LEADERBOARD_ROUTING_SCHEMA
                ),
            },
        },
    },
    "/api/skills/{id}": {
        "get": {
            "tags": ["Skills"],
            "summary": "Get skill details",
            "description": "Get detailed information about a specific skill",
            "operationId": "getSkills",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Skill ID",
                }
            ],
            "responses": {
                "200": _ok_response("Skill details", _SKILL_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/users/me": {
        "get": {
            "tags": ["Users"],
            "summary": "Get current user",
            "description": "Get information about the currently authenticated user",
            "operationId": "getUsersMe",
            "responses": {
                "200": _ok_response("Current user information", _USER_ME_SCHEMA),
            },
        },
    },
    "/api/users/me/preferences": {
        "get": {
            "tags": ["Users"],
            "summary": "Get user preferences",
            "description": "Get preferences and settings for the current user",
            "operationId": "getMePreferences",
            "responses": {
                "200": _ok_response("User preferences", _USER_PREFERENCES_SCHEMA),
            },
        },
    },
    "/api/users/me/profile": {
        "get": {
            "tags": ["Users"],
            "summary": "Get user profile",
            "description": "Get the profile of the current user",
            "operationId": "getMeProfile",
            "responses": {
                "200": _ok_response("User profile", _USER_PROFILE_SCHEMA),
            },
        },
        "patch": {
            "tags": ["Users"],
            "summary": "Update user profile",
            "description": "Update the profile of the current user",
            "operationId": "patchMeProfile",
            "requestBody": {
                "content": {"application/json": {"schema": _UPDATE_PROFILE_REQUEST}},
                "required": True,
            },
            "responses": {
                "200": _ok_response("Updated profile", _USER_PROFILE_SCHEMA),
            },
        },
    },
    "/api/users/{id}/debates": {
        "get": {
            "tags": ["Users"],
            "summary": "Get user debates",
            "description": "Get debates associated with a specific user",
            "operationId": "getUsersDebates",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "User ID",
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
                {
                    "name": "status",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["active", "completed", "cancelled"]},
                    "description": "Filter by status",
                },
            ],
            "responses": {
                "200": _ok_response("User debates", _USER_DEBATES_SCHEMA),
            },
        },
    },
    "/scim/v2/Groups": {
        "get": {
            "tags": ["SCIM"],
            "summary": "List SCIM groups",
            "description": "List groups according to SCIM 2.0 protocol",
            "operationId": "getV2Groups",
            "parameters": [
                {
                    "name": "filter",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "SCIM filter expression",
                },
                {
                    "name": "startIndex",
                    "in": "query",
                    "schema": {"type": "integer"},
                    "description": "Start index for pagination",
                },
                {
                    "name": "count",
                    "in": "query",
                    "schema": {"type": "integer"},
                    "description": "Number of results",
                },
            ],
            "responses": {
                "200": _ok_response("SCIM groups list", _SCIM_GROUPS_SCHEMA),
            },
        },
    },
    "/scim/v2/Users": {
        "get": {
            "tags": ["SCIM"],
            "summary": "List SCIM users",
            "description": "List users according to SCIM 2.0 protocol",
            "operationId": "getV2Users",
            "parameters": [
                {
                    "name": "filter",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "SCIM filter expression",
                },
                {
                    "name": "startIndex",
                    "in": "query",
                    "schema": {"type": "integer"},
                    "description": "Start index for pagination",
                },
                {
                    "name": "count",
                    "in": "query",
                    "schema": {"type": "integer"},
                    "description": "Number of results",
                },
            ],
            "responses": {
                "200": _ok_response("SCIM users list", _SCIM_USERS_SCHEMA),
            },
        },
    },
}

# Merge other endpoints
SDK_MISSING_ENDPOINTS.update(_OTHER_ENDPOINTS)


# Additional methods needed on existing or new paths to satisfy SDK contracts
_ADDITIONAL_METHODS: dict = {
    "/api/v1/connectors": {
        "get": _method_stub("Connectors", "GET", "List connectors", op_id="listConnectorsV1"),
    },
    "/api/v1/connectors/{id}": {
        "get": _method_stub(
            "Connectors", "GET", "Get connector", op_id="getConnectorV1", has_path_param=True
        ),
    },
    "/api/v1/connectors/{id}/sync": {
        "get": _method_stub(
            "Connectors",
            "GET",
            "Get connector sync status",
            op_id="getConnectorSyncV1",
            has_path_param=True,
        ),
    },
    "/api/v1/debates/{id}/red-team": {
        "get": _method_stub(
            "Debates",
            "GET",
            "Get red-team results",
            op_id="getDebateRedTeamV1",
            has_path_param=True,
        ),
    },
    "/api/v1/debates/{id}/followup": {
        "post": _method_stub(
            "Debates",
            "POST",
            "Submit followup",
            op_id="postDebateFollowupV1",
            has_path_param=True,
            has_body=True,
        ),
    },
    "/api/v1/debates/hybrid": {
        "post": _method_stub(
            "Debates", "POST", "Start hybrid debate", op_id="postDebatesHybridV1", has_body=True
        ),
    },
    "/api/v1/evolution/ab-tests": {
        "post": _method_stub(
            "Evolution", "POST", "Create AB test", op_id="postEvolutionAbTestsV1", has_body=True
        ),
        "delete": _method_stub(
            "Evolution", "DELETE", "Delete AB tests", op_id="deleteEvolutionAbTestsV1"
        ),
    },
    "/api/v1/knowledge/mound/curation/policy": {
        "get": _method_stub(
            "Knowledge", "GET", "Get curation policy", op_id="getKmCurationPolicyV1"
        ),
    },
    "/api/v1/knowledge/mound/analytics/quality/snapshot": {
        "get": _method_stub(
            "Knowledge", "GET", "Get quality snapshot", op_id="getKmQualitySnapshotV1"
        ),
    },
    "/api/v1/knowledge/mound/analytics/usage/record": {
        "get": _method_stub("Knowledge", "GET", "Get usage records", op_id="getKmUsageRecordV1"),
    },
    "/api/v1/knowledge/mound/confidence/decay": {
        "get": _method_stub(
            "Knowledge", "GET", "Get confidence decay", op_id="getKmConfidenceDecayV1"
        ),
    },
    "/api/v1/knowledge/mound/confidence/event": {
        "get": _method_stub(
            "Knowledge", "GET", "Get confidence events", op_id="getKmConfidenceEventV1"
        ),
    },
    "/api/v1/knowledge/mound/contradictions/detect": {
        "get": _method_stub(
            "Knowledge", "GET", "Get contradictions", op_id="getKmContradictionsDetectV1"
        ),
    },
    "/api/v1/knowledge/mound/contradictions/{id}/resolve": {
        "get": _method_stub(
            "Knowledge",
            "GET",
            "Get contradiction resolution",
            op_id="getKmContradictionResolveV1",
            has_path_param=True,
        ),
    },
    "/api/v1/knowledge/mound/curation/run": {
        "get": _method_stub(
            "Knowledge", "GET", "Get curation run status", op_id="getKmCurationRunV1"
        ),
    },
    "/api/v1/knowledge/mound/dashboard/metrics/reset": {
        "get": _method_stub(
            "Knowledge", "GET", "Get metrics reset status", op_id="getKmDashboardResetV1"
        ),
    },
    "/api/v1/knowledge/mound/dedup/auto-merge": {
        "get": _method_stub(
            "Knowledge", "GET", "Get auto-merge status", op_id="getKmDedupAutoMergeV1"
        ),
    },
    "/api/v1/knowledge/mound/dedup/merge": {
        "get": _method_stub("Knowledge", "GET", "Get merge status", op_id="getKmDedupMergeV1"),
    },
    "/api/v1/knowledge/mound/extraction/debate": {
        "get": _method_stub(
            "Knowledge", "GET", "Get extraction debates", op_id="getKmExtractionDebateV1"
        ),
    },
    "/api/v1/knowledge/mound/extraction/promote": {
        "get": _method_stub(
            "Knowledge", "GET", "Get promotion status", op_id="getKmExtractionPromoteV1"
        ),
    },
    "/api/v1/knowledge/mound/pruning/auto": {
        "get": _method_stub(
            "Knowledge", "GET", "Get auto-pruning config", op_id="getKmPruningAutoV1"
        ),
    },
    "/api/v1/knowledge/mound/pruning/execute": {
        "get": _method_stub(
            "Knowledge", "GET", "Get pruning execution status", op_id="getKmPruningExecuteV1"
        ),
    },
    "/api/v1/knowledge/mound/pruning/restore": {
        "get": _method_stub(
            "Knowledge", "GET", "Get pruning restore status", op_id="getKmPruningRestoreV1"
        ),
    },
    "/api/v2/integrations/wizard": {
        "get": _method_stub(
            "Integrations", "GET", "Get integration wizard", op_id="getIntegrationsWizardV2"
        ),
    },
    "/api/v1/personas/options": {
        "delete": _method_stub(
            "Personas", "DELETE", "Delete persona options", op_id="deletePersonaOptionsV1"
        ),
        "put": _method_stub(
            "Personas", "PUT", "Update persona options", op_id="putPersonaOptionsV1", has_body=True
        ),
        "post": _method_stub(
            "Personas",
            "POST",
            "Create persona options",
            op_id="postPersonaOptionsV1",
            has_body=True,
        ),
    },
    "/api/v1/rlm/contexts": {
        "delete": _method_stub("RLM", "DELETE", "Clear RLM contexts", op_id="deleteRlmContextsV1"),
        "post": _method_stub(
            "RLM", "POST", "Create RLM context", op_id="postRlmContextsV1", has_body=True
        ),
    },
    "/api/v1/rlm/stats": {
        "delete": _method_stub("RLM", "DELETE", "Clear RLM stats", op_id="deleteRlmStatsV1"),
        "post": _method_stub(
            "RLM", "POST", "Record RLM stats", op_id="postRlmStatsV1", has_body=True
        ),
    },
    "/api/v1/rlm/strategies": {
        "delete": _method_stub(
            "RLM", "DELETE", "Clear RLM strategies", op_id="deleteRlmStrategiesV1"
        ),
        "post": _method_stub(
            "RLM", "POST", "Create RLM strategy", op_id="postRlmStrategiesV1", has_body=True
        ),
    },
    "/api/v1/rlm/stream/modes": {
        "delete": _method_stub(
            "RLM", "DELETE", "Clear stream modes", op_id="deleteRlmStreamModesV1"
        ),
        "post": _method_stub(
            "RLM", "POST", "Set stream mode", op_id="postRlmStreamModesV1", has_body=True
        ),
    },
    "/api/v1/analytics/connect": {
        "get": _method_stub(
            "Analytics", "GET", "Get analytics connection", op_id="getAnalyticsConnectV1"
        ),
    },
    "/api/v1/analytics/query": {
        "get": _method_stub("Analytics", "GET", "Query analytics", op_id="getAnalyticsQueryV1"),
    },
    "/api/v1/analytics/reports/generate": {
        "get": _method_stub(
            "Analytics", "GET", "Generate analytics report", op_id="getAnalyticsReportsGenerateV1"
        ),
    },
    "/api/v1/analytics/{id}": {
        "get": _method_stub(
            "Analytics",
            "GET",
            "Get analytics by ID",
            op_id="getAnalyticsByIdV1",
            has_path_param=True,
        ),
    },
    "/api/v1/cross-pollination/km/staleness-check": {
        "get": _method_stub(
            "Cross-Pollination", "GET", "Check KM staleness", op_id="getCrossPollinationStalenessV1"
        ),
    },
    "/api/v1/personas": {
        "get": _method_stub("Personas", "GET", "List personas", op_id="listPersonasV1"),
        "post": _method_stub(
            "Personas", "POST", "Create persona", op_id="createPersonaV1", has_body=True
        ),
    },
    "/api/v1/policies/{id}/toggle": {
        "get": _method_stub(
            "Policies",
            "GET",
            "Get policy toggle state",
            op_id="getPolicyToggleV1",
            has_path_param=True,
        ),
    },
    "/api/v1/privacy/account": {
        "get": _method_stub(
            "Privacy", "GET", "Get account privacy settings", op_id="getPrivacyAccountV1"
        ),
    },
    "/api/v1/privacy/preferences": {
        "get": _method_stub(
            "Privacy", "GET", "Get privacy preferences", op_id="getPrivacyPreferencesV1"
        ),
    },
    "/api/v1/replays/{id}": {
        "get": _method_stub(
            "Replays", "GET", "Get replay", op_id="getReplayV1", has_path_param=True
        ),
    },
    "/api/v1/routing-rules": {
        "get": _method_stub("Routing", "GET", "List routing rules", op_id="listRoutingRulesV1"),
    },
    "/api/v1/routing-rules/evaluate": {
        "get": _method_stub(
            "Routing", "GET", "Evaluate routing rules", op_id="evaluateRoutingRulesV1"
        ),
    },
    "/api/v1/routing-rules/{id}": {
        "get": _method_stub(
            "Routing", "GET", "Get routing rule", op_id="getRoutingRuleV1", has_path_param=True
        ),
    },
    "/api/v1/routing-rules/{id}/toggle": {
        "get": _method_stub(
            "Routing",
            "GET",
            "Get routing rule toggle state",
            op_id="getRoutingRuleToggleV1",
            has_path_param=True,
        ),
    },
    "/api/v1/training/export/dpo": {
        "get": _method_stub("Training", "GET", "Export DPO data", op_id="getTrainingExportDpoV1"),
    },
    "/api/v1/training/export/gauntlet": {
        "get": _method_stub(
            "Training", "GET", "Export gauntlet data", op_id="getTrainingExportGauntletV1"
        ),
    },
    "/api/v1/training/export/sft": {
        "get": _method_stub("Training", "GET", "Export SFT data", op_id="getTrainingExportSftV1"),
    },
    "/api/v1/verticals/{id}/agent": {
        "get": _method_stub(
            "Verticals",
            "GET",
            "Get vertical agent",
            op_id="getVerticalAgentV1",
            has_path_param=True,
        ),
    },
    "/api/v1/verticals/{id}/config": {
        "get": _method_stub(
            "Verticals",
            "GET",
            "Get vertical config",
            op_id="getVerticalConfigV1",
            has_path_param=True,
        ),
    },
    "/api/v1/verticals/{id}/debate": {
        "get": _method_stub(
            "Verticals",
            "GET",
            "Get vertical debate",
            op_id="getVerticalDebateV1",
            has_path_param=True,
        ),
    },
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
    "/api/v1/ml/models": {
        "post": _method_stub(
            "ML", "POST", "Register ML model", op_id="postMlModelsV1", has_body=True
        ),
    },
    "/api/v1/ml/stats": {
        "post": _method_stub("ML", "POST", "Record ML stats", op_id="postMlStatsV1", has_body=True),
    },
    "/api/v1/probes/reports": {
        "post": _method_stub(
            "Probes", "POST", "Submit probe report", op_id="postProbesReportsV1", has_body=True
        ),
    },
    "/api/v1/routing/domain-leaderboard": {
        "post": _method_stub(
            "Routing",
            "POST",
            "Submit domain leaderboard",
            op_id="postRoutingDomainLeaderboardV1",
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
    "/api/replays/{id}": {
        "get": _method_stub("Replays", "GET", "Get replay", op_id="getReplay", has_path_param=True),
        "delete": _method_stub(
            "Replays", "DELETE", "Delete replay", op_id="deleteReplay", has_path_param=True
        ),
    },
    "/scim/v2/Groups": {
        "post": _method_stub("SCIM", "POST", "Create group", op_id="postScimGroups", has_body=True),
    },
    "/scim/v2/Users": {
        "post": _method_stub("SCIM", "POST", "Create user", op_id="postScimUsers", has_body=True),
    },
    "/api/integrations/teams/install": {
        "get": _method_stub("Teams", "GET", "Get Teams install link", op_id="getTeamsInstall"),
    },
    "/api/v1/uncertainty/estimate": {
        "post": _method_stub(
            "Uncertainty",
            "POST",
            "Estimate uncertainty",
            op_id="postUncertaintyEstimateV1",
            has_body=True,
        ),
    },
    "/api/v1/uncertainty/followups": {
        "post": _method_stub(
            "Uncertainty",
            "POST",
            "Get followup questions",
            op_id="postUncertaintyFollowupsV1",
            has_body=True,
        ),
    },
    "/api/workspaces/{id}": {
        "delete": _method_stub(
            "Workspace",
            "DELETE",
            "Delete workspace",
            op_id="deleteWorkspaceById",
            has_path_param=True,
        ),
        "get": _method_stub(
            "Workspace", "GET", "Get workspace", op_id="getWorkspaceById", has_path_param=True
        ),
    },
    "/api/workspaces/{id}/members": {
        "post": _method_stub(
            "Workspace",
            "POST",
            "Add workspace member",
            op_id="postWorkspaceMembers",
            has_path_param=True,
            has_body=True,
        ),
    },
    "/api/workspaces/{id}/members/{member_id}": {
        "put": _method_stub(
            "Workspace",
            "PUT",
            "Update workspace member",
            op_id="putWorkspaceMember",
            has_path_param=True,
            has_body=True,
        ),
    },
    # Non-versioned persona/pulse/routing/verticals/replays/policies endpoints
    "/api/personas/options": {
        "get": _method_stub("Personas", "GET", "List persona options", op_id="listPersonaOptions"),
    },
    "/api/personas": {
        "get": _method_stub("Personas", "GET", "List personas", op_id="listPersonas"),
        "post": _method_stub(
            "Personas", "POST", "Create persona", op_id="createPersona", has_body=True
        ),
    },
    "/api/policies/{id}/toggle": {
        "post": _method_stub(
            "Policies",
            "POST",
            "Toggle policy",
            op_id="togglePolicy",
            has_path_param=True,
            has_body=True,
        ),
    },
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
    "/api/routing/domain-leaderboard": {
        "get": _method_stub(
            "Routing", "GET", "Get domain leaderboard", op_id="getDomainLeaderboard"
        ),
    },
    "/api/verticals/{id}/agent": {
        "post": _method_stub(
            "Verticals",
            "POST",
            "Create vertical agent",
            op_id="postVerticalAgent",
            has_path_param=True,
            has_body=True,
        ),
    },
    "/api/verticals/{id}/debate": {
        "post": _method_stub(
            "Verticals",
            "POST",
            "Start vertical debate",
            op_id="postVerticalDebate",
            has_path_param=True,
            has_body=True,
        ),
    },
    "/api/verticals/{id}/config": {
        "put": _method_stub(
            "Verticals",
            "PUT",
            "Update vertical config",
            op_id="putVerticalConfig",
            has_path_param=True,
            has_body=True,
        ),
    },
}

# Merge additional methods into SDK_MISSING_ENDPOINTS
for path, methods in _ADDITIONAL_METHODS.items():
    if path in SDK_MISSING_ENDPOINTS:
        SDK_MISSING_ENDPOINTS[path].update(methods)  # type: ignore[attr-defined]
    else:
        SDK_MISSING_ENDPOINTS[path] = methods


# Re-export submodule dictionaries for direct access
__all__ = [
    "SDK_MISSING_ENDPOINTS",
    "SDK_MISSING_COSTS_ENDPOINTS",
    "SDK_MISSING_COMPLIANCE_ENDPOINTS",
    "SDK_MISSING_ANALYTICS_ENDPOINTS",
    "SDK_MISSING_INTEGRATION_ENDPOINTS",
    "_method_stub",
]
