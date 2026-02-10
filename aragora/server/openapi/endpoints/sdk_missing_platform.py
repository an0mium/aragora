"""SDK missing endpoints: Platform services (keys, knowledge, personas, skills, users, SCIM, etc).

Contains OpenAPI schema definitions for:
- API key management
- Knowledge Mound nodes and relationships
- Personas configuration
- Skills management
- User profiles, preferences, and debates
- SCIM 2.0 provisioning (Groups, Users)
- Connectors, workspaces, verticals
- Evolution, privacy, policies, RLM, ML, probes, analytics, uncertainty
"""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS
from aragora.server.openapi.endpoints.sdk_missing_core import _method_stub


# =============================================================================
# Response Schemas
# =============================================================================

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


# =============================================================================
# Endpoint Definitions
# =============================================================================

SDK_MISSING_PLATFORM_ENDPOINTS: dict = {
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


# =============================================================================
# Additional Method Stubs (platform services)
# =============================================================================

SDK_MISSING_PLATFORM_ADDITIONAL: dict = {
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
    # Non-versioned persona/policies/verticals endpoints
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
