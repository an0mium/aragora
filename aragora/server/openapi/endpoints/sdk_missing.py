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

# Additional endpoints that don't fit into the main categories
_OTHER_ENDPOINTS: dict = {
    "/api/agent/{name}/persona": {
        "delete": {
            "tags": ["Agent"],
            "summary": "DELETE persona",
            "operationId": "deleteAgentPersona",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "put": {
            "tags": ["Agent"],
            "summary": "PUT persona",
            "operationId": "putAgentPersona",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/debates/{id}/replay": {
        "get": {
            "tags": ["Debates"],
            "summary": "GET replay",
            "operationId": "getDebatesReplay",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/keys": {
        "get": {
            "tags": ["Keys"],
            "summary": "GET keys",
            "operationId": "getKeys",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/keys/{id}": {
        "delete": {
            "tags": ["Keys"],
            "summary": "DELETE {id}",
            "operationId": "deleteKeys",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/knowledge/mound/nodes/{id}": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET {id}",
            "operationId": "getMoundNodes",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/knowledge/mound/nodes/{id}/relationships": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET relationships",
            "operationId": "getNodesRelationships",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/leaderboard/agent/{id}": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "GET {id}",
            "operationId": "getLeaderboardAgent",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/leaderboard/agent/{id}/elo-history": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "GET elo-history",
            "operationId": "getAgentElo-History",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/leaderboard/compare": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "GET compare",
            "operationId": "getLeaderboardCompare",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/leaderboard/domain/{id}": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "GET {id}",
            "operationId": "getLeaderboardDomain",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/leaderboard/domains": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "GET domains",
            "operationId": "getLeaderboardDomains",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/leaderboard/movers": {
        "get": {
            "tags": ["Leaderboard"],
            "summary": "GET movers",
            "operationId": "getLeaderboardMovers",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/notifications/email/config": {
        "post": {
            "tags": ["Notifications"],
            "summary": "POST config",
            "operationId": "postEmailConfig",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/notifications/email/recipient": {
        "delete": {
            "tags": ["Notifications"],
            "summary": "DELETE recipient",
            "operationId": "deleteEmailRecipient",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Notifications"],
            "summary": "POST recipient",
            "operationId": "postEmailRecipient",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/notifications/email/recipients": {
        "get": {
            "tags": ["Notifications"],
            "summary": "GET recipients",
            "operationId": "getEmailRecipients",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/notifications/history": {
        "get": {
            "tags": ["Notifications"],
            "summary": "GET history",
            "operationId": "getNotificationsHistory",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/notifications/send": {
        "post": {
            "tags": ["Notifications"],
            "summary": "POST send",
            "operationId": "postNotificationsSend",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/notifications/status": {
        "get": {
            "tags": ["Notifications"],
            "summary": "GET status",
            "operationId": "getNotificationsStatus",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/notifications/telegram/config": {
        "post": {
            "tags": ["Notifications"],
            "summary": "POST config",
            "operationId": "postTelegramConfig",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/notifications/test": {
        "post": {
            "tags": ["Notifications"],
            "summary": "POST test",
            "operationId": "postNotificationsTest",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/personas": {
        "post": {
            "tags": ["Personas"],
            "summary": "POST personas",
            "operationId": "postPersonas",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/personas/options": {
        "get": {
            "tags": ["Personas"],
            "summary": "GET options",
            "operationId": "getPersonasOptions",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/pulse/analytics": {
        "get": {
            "tags": ["Pulse"],
            "summary": "GET analytics",
            "operationId": "getPulseAnalytics",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/pulse/debate-topic": {
        "post": {
            "tags": ["Pulse"],
            "summary": "POST debate-topic",
            "operationId": "postPulseDebate-Topic",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/pulse/scheduler/config": {
        "patch": {
            "tags": ["Pulse"],
            "summary": "PATCH config",
            "operationId": "patchSchedulerConfig",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/pulse/scheduler/history": {
        "get": {
            "tags": ["Pulse"],
            "summary": "GET history",
            "operationId": "getSchedulerHistory",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/pulse/scheduler/pause": {
        "post": {
            "tags": ["Pulse"],
            "summary": "POST pause",
            "operationId": "postSchedulerPause",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/pulse/scheduler/resume": {
        "post": {
            "tags": ["Pulse"],
            "summary": "POST resume",
            "operationId": "postSchedulerResume",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/pulse/scheduler/start": {
        "post": {
            "tags": ["Pulse"],
            "summary": "POST start",
            "operationId": "postSchedulerStart",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/pulse/scheduler/status": {
        "get": {
            "tags": ["Pulse"],
            "summary": "GET status",
            "operationId": "getSchedulerStatus",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/pulse/scheduler/stop": {
        "post": {
            "tags": ["Pulse"],
            "summary": "POST stop",
            "operationId": "postSchedulerStop",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET replays",
            "operationId": "getReplays",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/create": {
        "post": {
            "tags": ["Replays"],
            "summary": "POST create",
            "operationId": "postReplaysCreate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/share/{id}": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET {id}",
            "operationId": "getReplaysShare",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/{id}": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET {id}",
            "operationId": "getReplays",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/{id}/bookmark": {
        "delete": {
            "tags": ["Replays"],
            "summary": "DELETE bookmark",
            "operationId": "deleteReplaysBookmark",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Replays"],
            "summary": "POST bookmark",
            "operationId": "postReplaysBookmark",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/{id}/comments": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET comments",
            "operationId": "getReplaysComments",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Replays"],
            "summary": "POST comments",
            "operationId": "postReplaysComments",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/{id}/export": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET export",
            "operationId": "getReplaysExport",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/{id}/share": {
        "post": {
            "tags": ["Replays"],
            "summary": "POST share",
            "operationId": "postReplaysShare",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/{id}/summary": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET summary",
            "operationId": "getReplaysSummary",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/{id}/transcript": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET transcript",
            "operationId": "getReplaysTranscript",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/routing/domain-leaderboard": {
        "get": {
            "tags": ["Routing"],
            "summary": "GET domain-leaderboard",
            "operationId": "getRoutingDomain-Leaderboard",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/skills/{id}": {
        "get": {
            "tags": ["Skills"],
            "summary": "GET {id}",
            "operationId": "getSkills",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/users/me": {
        "get": {
            "tags": ["Users"],
            "summary": "GET me",
            "operationId": "getUsersMe",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/users/me/preferences": {
        "get": {
            "tags": ["Users"],
            "summary": "GET preferences",
            "operationId": "getMePreferences",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/users/me/profile": {
        "get": {
            "tags": ["Users"],
            "summary": "GET profile",
            "operationId": "getMeProfile",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "patch": {
            "tags": ["Users"],
            "summary": "PATCH profile",
            "operationId": "patchMeProfile",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/users/{id}/debates": {
        "get": {
            "tags": ["Users"],
            "summary": "GET debates",
            "operationId": "getUsersDebates",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/scim/v2/Groups": {
        "get": {
            "tags": ["SCIM"],
            "summary": "GET Groups",
            "operationId": "getV2Groups",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/scim/v2/Users": {
        "get": {
            "tags": ["SCIM"],
            "summary": "GET Users",
            "operationId": "getV2Users",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
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
    "/api/v1/memory/critiques": {
        "delete": _method_stub(
            "Memory", "DELETE", "Clear critiques", op_id="deleteMemoryCritiquesV1"
        ),
        "post": _method_stub(
            "Memory", "POST", "Store critique", op_id="postMemoryCritiquesV1", has_body=True
        ),
    },
    "/api/v1/memory/pressure": {
        "delete": _method_stub(
            "Memory", "DELETE", "Clear pressure data", op_id="deleteMemoryPressureV1"
        ),
        "post": _method_stub(
            "Memory", "POST", "Record memory pressure", op_id="postMemoryPressureV1", has_body=True
        ),
    },
    "/api/v1/memory/tiers": {
        "delete": _method_stub("Memory", "DELETE", "Clear tier data", op_id="deleteMemoryTiersV1"),
        "post": _method_stub(
            "Memory", "POST", "Configure memory tiers", op_id="postMemoryTiersV1", has_body=True
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
    "/api/agent/{name}/persona": {
        "get": _method_stub(
            "Agent", "GET", "Get agent persona", op_id="getAgentPersona", has_path_param=True
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
