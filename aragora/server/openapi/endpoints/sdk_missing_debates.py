"""SDK missing endpoints: Debates, Replays, Leaderboard, and Routing.

Contains OpenAPI schema definitions for:
- Debate replay schemas and endpoints
- Agent leaderboard/ELO schemas and endpoints
- Routing domain leaderboard schemas and endpoints
- Replay management (bookmarks, comments, export, share, summary, transcript)
"""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS
from aragora.server.openapi.endpoints.sdk_missing_core import _method_stub


# =============================================================================
# Response Schemas
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


# =============================================================================
# Request Body Schemas
# =============================================================================

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


# =============================================================================
# Endpoint Definitions
# =============================================================================

SDK_MISSING_DEBATES_ENDPOINTS: dict = {
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
}


# =============================================================================
# Additional Method Stubs (debates, replays, routing)
# =============================================================================

SDK_MISSING_DEBATES_ADDITIONAL: dict = {
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
    "/api/v1/replays/{id}": {
        "get": _method_stub(
            "Replays", "GET", "Get replay", op_id="getReplayV1", has_path_param=True
        ),
    },
    "/api/replays/{id}": {
        "get": _method_stub("Replays", "GET", "Get replay", op_id="getReplay", has_path_param=True),
        "delete": _method_stub(
            "Replays", "DELETE", "Delete replay", op_id="deleteReplay", has_path_param=True
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
    "/api/v1/routing/domain-leaderboard": {
        "post": _method_stub(
            "Routing",
            "POST",
            "Submit domain leaderboard",
            op_id="postRoutingDomainLeaderboardV1",
            has_body=True,
        ),
    },
    "/api/routing/domain-leaderboard": {
        "get": _method_stub(
            "Routing", "GET", "Get domain leaderboard", op_id="getDomainLeaderboard"
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
}
