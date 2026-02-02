"""Auto-generated missing SDK endpoints."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

SDK_MISSING_ENDPOINTS = {
    "/api/agent/{id}/persona": {
        "delete": {
            "tags": ["Agent"],
            "summary": "DELETE persona",
            "operationId": "deleteAgentPersona",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
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
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/analytics/agents/{id}/performance": {
        "get": {
            "tags": ["Analytics"],
            "summary": "GET performance",
            "operationId": "getAgentsPerformance",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/compliance/summary": {
        "get": {
            "tags": ["Compliance"],
            "summary": "GET summary",
            "operationId": "getComplianceSummary",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET costs",
            "operationId": "getCosts",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/alerts": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET alerts",
            "operationId": "getCostsAlerts",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/alerts/{id}/dismiss": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST dismiss",
            "operationId": "postAlertsDismiss",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/breakdown": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET breakdown",
            "operationId": "getCostsBreakdown",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/budget": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST budget",
            "operationId": "postCostsBudget",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/constraints/check": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST check",
            "operationId": "postConstraintsCheck",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/efficiency": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET efficiency",
            "operationId": "getCostsEfficiency",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/estimate": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST estimate",
            "operationId": "postCostsEstimate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/forecast": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET forecast",
            "operationId": "getCostsForecast",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/forecast/detailed": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET detailed",
            "operationId": "getForecastDetailed",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/forecast/simulate": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST simulate",
            "operationId": "postForecastSimulate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/recommendations": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET recommendations",
            "operationId": "getCostsRecommendations",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/recommendations/detailed": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET detailed",
            "operationId": "getRecommendationsDetailed",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/recommendations/{id}": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET {id}",
            "operationId": "getCostsRecommendations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/recommendations/{id}/apply": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST apply",
            "operationId": "postRecommendationsApply",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/recommendations/{id}/dismiss": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST dismiss",
            "operationId": "postRecommendationsDismiss",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/timeline": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET timeline",
            "operationId": "getCostsTimeline",
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
    "/api/integrations": {
        "get": {
            "tags": ["Integrations"],
            "summary": "GET integrations",
            "operationId": "getIntegrations",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Integrations"],
            "summary": "POST integrations",
            "operationId": "postIntegrations",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/available": {
        "get": {
            "tags": ["Integrations"],
            "summary": "GET available",
            "operationId": "getIntegrationsAvailable",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/config/{id}": {
        "get": {
            "tags": ["Integrations"],
            "summary": "GET {id}",
            "operationId": "getIntegrationsConfig",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/discord/callback": {
        "post": {
            "tags": ["Integrations"],
            "summary": "POST callback",
            "operationId": "postDiscordCallback",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/discord/install": {
        "post": {
            "tags": ["Integrations"],
            "summary": "POST install",
            "operationId": "postDiscordInstall",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/teams/callback": {
        "post": {
            "tags": ["Integrations"],
            "summary": "POST callback",
            "operationId": "postTeamsCallback",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/teams/install": {
        "post": {
            "tags": ["Integrations"],
            "summary": "POST install",
            "operationId": "postTeamsInstall",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/{id}": {
        "delete": {
            "tags": ["Integrations"],
            "summary": "DELETE {id}",
            "operationId": "deleteIntegrations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Integrations"],
            "summary": "GET {id}",
            "operationId": "getIntegrations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Integrations"],
            "summary": "PUT {id}",
            "operationId": "putIntegrations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/{id}/sync": {
        "get": {
            "tags": ["Integrations"],
            "summary": "GET sync",
            "operationId": "getIntegrationsSync",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Integrations"],
            "summary": "POST sync",
            "operationId": "postIntegrationsSync",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/{id}/test": {
        "post": {
            "tags": ["Integrations"],
            "summary": "POST test",
            "operationId": "postIntegrationsTest",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
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
    "/api/payments/authorize": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST authorize",
            "operationId": "postPaymentsAuthorize",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/capture": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST capture",
            "operationId": "postPaymentsCapture",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/charge": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST charge",
            "operationId": "postPaymentsCharge",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/customer": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST customer",
            "operationId": "postPaymentsCustomer",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/customer/{id}": {
        "delete": {
            "tags": ["Payments"],
            "summary": "DELETE {id}",
            "operationId": "deletePaymentsCustomer",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Payments"],
            "summary": "GET {id}",
            "operationId": "getPaymentsCustomer",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Payments"],
            "summary": "PUT {id}",
            "operationId": "putPaymentsCustomer",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/refund": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST refund",
            "operationId": "postPaymentsRefund",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/subscription": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST subscription",
            "operationId": "postPaymentsSubscription",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/subscription/{id}": {
        "delete": {
            "tags": ["Payments"],
            "summary": "DELETE {id}",
            "operationId": "deletePaymentsSubscription",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Payments"],
            "summary": "GET {id}",
            "operationId": "getPaymentsSubscription",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Payments"],
            "summary": "PUT {id}",
            "operationId": "putPaymentsSubscription",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/transaction/{id}": {
        "get": {
            "tags": ["Payments"],
            "summary": "GET {id}",
            "operationId": "getPaymentsTransaction",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/void": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST void",
            "operationId": "postPaymentsVoid",
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
    "/api/policies": {
        "get": {
            "tags": ["Policies"],
            "summary": "GET policies",
            "operationId": "getPolicies",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Policies"],
            "summary": "POST policies",
            "operationId": "postPolicies",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/validate": {
        "post": {
            "tags": ["Policies"],
            "summary": "POST validate",
            "operationId": "postPoliciesValidate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/violations": {
        "get": {
            "tags": ["Policies"],
            "summary": "GET violations",
            "operationId": "getPoliciesViolations",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/violations/{id}/resolve": {
        "post": {
            "tags": ["Policies"],
            "summary": "POST resolve",
            "operationId": "postViolationsResolve",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/{id}": {
        "delete": {
            "tags": ["Policies"],
            "summary": "DELETE {id}",
            "operationId": "deletePolicies",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Policies"],
            "summary": "GET {id}",
            "operationId": "getPolicies",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "patch": {
            "tags": ["Policies"],
            "summary": "PATCH {id}",
            "operationId": "patchPolicies",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/{id}/disable": {
        "post": {
            "tags": ["Policies"],
            "summary": "POST disable",
            "operationId": "postPoliciesDisable",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/{id}/enable": {
        "post": {
            "tags": ["Policies"],
            "summary": "POST enable",
            "operationId": "postPoliciesEnable",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/{id}/toggle": {
        "post": {
            "tags": ["Policies"],
            "summary": "POST toggle",
            "operationId": "postPoliciesToggle",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/{id}/violations": {
        "get": {
            "tags": ["Policies"],
            "summary": "GET violations",
            "operationId": "getPoliciesViolations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
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
    "/api/replays/compare": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET compare",
            "operationId": "getReplaysCompare",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/search": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET search",
            "operationId": "getReplaysSearch",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/stats": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET stats",
            "operationId": "getReplaysStats",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/{id}": {
        "delete": {
            "tags": ["Replays"],
            "summary": "DELETE {id}",
            "operationId": "deleteReplays",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/replays/{id}/events": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET events",
            "operationId": "getReplaysEvents",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/{id}/evolution": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET evolution",
            "operationId": "getReplaysEvolution",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
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
    "/api/replays/{id}/fork": {
        "post": {
            "tags": ["Replays"],
            "summary": "POST fork",
            "operationId": "postReplaysFork",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/{id}/forks": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET forks",
            "operationId": "getReplaysForks",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/replays/{id}/html": {
        "get": {
            "tags": ["Replays"],
            "summary": "GET html",
            "operationId": "getReplaysHtml",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
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
    "/api/users": {
        "get": {
            "tags": ["Users"],
            "summary": "GET users",
            "operationId": "getUsers",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/users/invite": {
        "post": {
            "tags": ["Users"],
            "summary": "POST invite",
            "operationId": "postUsersInvite",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/users/{id}": {
        "delete": {
            "tags": ["Users"],
            "summary": "DELETE {id}",
            "operationId": "deleteUsers",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/users/{id}/role": {
        "put": {
            "tags": ["Users"],
            "summary": "PUT role",
            "operationId": "putUsersRole",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/batch": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST batch",
            "operationId": "postApBatch",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/discounts": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET discounts",
            "operationId": "getApDiscounts",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/forecast": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET forecast",
            "operationId": "getApForecast",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/invoices": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET invoices",
            "operationId": "getApInvoices",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Accounting"],
            "summary": "POST invoices",
            "operationId": "postApInvoices",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/invoices/{id}": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET {id}",
            "operationId": "getApInvoices",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/invoices/{id}/payment": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST payment",
            "operationId": "postInvoicesPayment",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/optimize": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST optimize",
            "operationId": "postApOptimize",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/aging": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET aging",
            "operationId": "getArAging",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/collections": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET collections",
            "operationId": "getArCollections",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/customers": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST customers",
            "operationId": "postArCustomers",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/customers/{id}/balance": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET balance",
            "operationId": "getCustomersBalance",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/invoices": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET invoices",
            "operationId": "getArInvoices",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Accounting"],
            "summary": "POST invoices",
            "operationId": "postArInvoices",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET {id}",
            "operationId": "getArInvoices",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}/payment": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST payment",
            "operationId": "postInvoicesPayment",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}/reminder": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST reminder",
            "operationId": "postInvoicesReminder",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}/send": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST send",
            "operationId": "postInvoicesSend",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/connect": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST connect",
            "operationId": "postAccountingConnect",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/customers": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET customers",
            "operationId": "getAccountingCustomers",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/disconnect": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST disconnect",
            "operationId": "postAccountingDisconnect",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/expenses/{id}": {
        "delete": {
            "tags": ["Accounting"],
            "summary": "DELETE {id}",
            "operationId": "deleteAccountingExpenses",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Accounting"],
            "summary": "GET {id}",
            "operationId": "getAccountingExpenses",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Accounting"],
            "summary": "PUT {id}",
            "operationId": "putAccountingExpenses",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/expenses/{id}/approve": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST approve",
            "operationId": "postExpensesApprove",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/expenses/{id}/reject": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST reject",
            "operationId": "postExpensesReject",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET {id}",
            "operationId": "getAccountingInvoices",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/anomalies": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET anomalies",
            "operationId": "getInvoicesAnomalies",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/approve": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST approve",
            "operationId": "postInvoicesApprove",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/match": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST match",
            "operationId": "postInvoicesMatch",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/reject": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST reject",
            "operationId": "postInvoicesReject",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/schedule": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST schedule",
            "operationId": "postInvoicesSchedule",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/reports": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST reports",
            "operationId": "postAccountingReports",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/status": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET status",
            "operationId": "getAccountingStatus",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/transactions": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET transactions",
            "operationId": "getAccountingTransactions",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/analytics/connect": {
        "post": {
            "tags": ["Analytics"],
            "summary": "POST connect",
            "operationId": "postAnalyticsConnect",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/analytics/query": {
        "post": {
            "tags": ["Analytics"],
            "summary": "POST query",
            "operationId": "postAnalyticsQuery",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/analytics/reports/generate": {
        "post": {
            "tags": ["Analytics"],
            "summary": "POST generate",
            "operationId": "postReportsGenerate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/analytics/{id}": {
        "delete": {
            "tags": ["Analytics"],
            "summary": "DELETE {id}",
            "operationId": "deleteAnalytics",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/audit/denied": {
        "delete": {
            "tags": ["Audit"],
            "summary": "DELETE denied",
            "operationId": "deleteAuditDenied",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Audit"],
            "summary": "POST denied",
            "operationId": "postAuditDenied",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Audit"],
            "summary": "PUT denied",
            "operationId": "putAuditDenied",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/audit/resource/{id}/{id}/history": {
        "get": {
            "tags": ["Audit"],
            "summary": "GET history",
            "operationId": "getResourceHistory",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/batch": {
        "get": {
            "tags": ["Batch"],
            "summary": "GET batch",
            "operationId": "getBatch",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Batch"],
            "summary": "POST batch",
            "operationId": "postBatch",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/batch/queue/status": {
        "get": {
            "tags": ["Batch"],
            "summary": "GET status",
            "operationId": "getQueueStatus",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/batch/{id}": {
        "get": {
            "tags": ["Batch"],
            "summary": "GET {id}",
            "operationId": "getBatch",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/bindings": {
        "delete": {
            "tags": ["Bindings"],
            "summary": "DELETE bindings",
            "operationId": "deleteBindings",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/bots/email/status": {
        "get": {
            "tags": ["Bots"],
            "summary": "GET status",
            "operationId": "getEmailStatus",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/bots/slack/status": {
        "get": {
            "tags": ["Bots"],
            "summary": "GET status",
            "operationId": "getSlackStatus",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/checkpoints/{id}": {
        "delete": {
            "tags": ["Checkpoints"],
            "summary": "DELETE {id}",
            "operationId": "deleteCheckpoints",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Checkpoints"],
            "summary": "GET {id}",
            "operationId": "getCheckpoints",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/checkpoints/{id}/intervention": {
        "post": {
            "tags": ["Checkpoints"],
            "summary": "POST intervention",
            "operationId": "postCheckpointsIntervention",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/checkpoints/{id}/resume": {
        "post": {
            "tags": ["Checkpoints"],
            "summary": "POST resume",
            "operationId": "postCheckpointsResume",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/code-review/diff": {
        "post": {
            "tags": ["Code Review"],
            "summary": "POST diff",
            "operationId": "postCode-ReviewDiff",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/code-review/history": {
        "get": {
            "tags": ["Code Review"],
            "summary": "GET history",
            "operationId": "getCode-ReviewHistory",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/code-review/pr": {
        "post": {
            "tags": ["Code Review"],
            "summary": "POST pr",
            "operationId": "postCode-ReviewPr",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/code-review/results/{id}": {
        "get": {
            "tags": ["Code Review"],
            "summary": "GET {id}",
            "operationId": "getCode-ReviewResults",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/code-review/review": {
        "post": {
            "tags": ["Code Review"],
            "summary": "POST review",
            "operationId": "postCode-ReviewReview",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/code-review/security-scan": {
        "post": {
            "tags": ["Code Review"],
            "summary": "POST security-scan",
            "operationId": "postCode-ReviewSecurity-Scan",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/computer-use/actions/stats": {
        "get": {
            "tags": ["Computer Use"],
            "summary": "GET stats",
            "operationId": "getActionsStats",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/computer-use/tasks/{id}/cancel": {
        "post": {
            "tags": ["Computer Use"],
            "summary": "POST cancel",
            "operationId": "postTasksCancel",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors": {
        "post": {
            "tags": ["Connectors"],
            "summary": "POST connectors",
            "operationId": "postConnectors",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}": {
        "delete": {
            "tags": ["Connectors"],
            "summary": "DELETE {id}",
            "operationId": "deleteConnectors",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "patch": {
            "tags": ["Connectors"],
            "summary": "PATCH {id}",
            "operationId": "patchConnectors",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/health": {
        "get": {
            "tags": ["Connectors"],
            "summary": "GET health",
            "operationId": "getConnectorsHealth",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/sync": {
        "post": {
            "tags": ["Connectors"],
            "summary": "POST sync",
            "operationId": "postConnectorsSync",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/syncs": {
        "get": {
            "tags": ["Connectors"],
            "summary": "GET syncs",
            "operationId": "getConnectorsSyncs",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/syncs/{id}": {
        "get": {
            "tags": ["Connectors"],
            "summary": "GET {id}",
            "operationId": "getConnectorsSyncs",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/syncs/{id}/cancel": {
        "post": {
            "tags": ["Connectors"],
            "summary": "POST cancel",
            "operationId": "postSyncsCancel",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/test": {
        "post": {
            "tags": ["Connectors"],
            "summary": "POST test",
            "operationId": "postConnectorsTest",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/cross-pollination/km/staleness-check": {
        "post": {
            "tags": ["Cross Pollination"],
            "summary": "POST staleness-check",
            "operationId": "postKmStaleness-Check",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/batch": {
        "patch": {
            "tags": ["Debates"],
            "summary": "PATCH batch",
            "operationId": "patchDebatesBatch",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/batch/{id}/status": {
        "patch": {
            "tags": ["Debates"],
            "summary": "PATCH status",
            "operationId": "patchBatchStatus",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Debates"],
            "summary": "POST status",
            "operationId": "postBatchStatus",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/export/batch": {
        "patch": {
            "tags": ["Debates"],
            "summary": "PATCH batch",
            "operationId": "patchExportBatch",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/export/batch/{id}/results": {
        "patch": {
            "tags": ["Debates"],
            "summary": "PATCH results",
            "operationId": "patchBatchResults",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Debates"],
            "summary": "POST results",
            "operationId": "postBatchResults",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/export/batch/{id}/status": {
        "patch": {
            "tags": ["Debates"],
            "summary": "PATCH status",
            "operationId": "patchBatchStatus",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Debates"],
            "summary": "POST status",
            "operationId": "postBatchStatus",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/graph": {
        "get": {
            "tags": ["Debates"],
            "summary": "GET graph",
            "operationId": "getDebatesGraph",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/hybrid": {
        "get": {
            "tags": ["Debates"],
            "summary": "GET hybrid",
            "operationId": "getDebatesHybrid",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/hybrid/{id}": {
        "get": {
            "tags": ["Debates"],
            "summary": "GET {id}",
            "operationId": "getDebatesHybrid",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/matrix": {
        "get": {
            "tags": ["Debates"],
            "summary": "GET matrix",
            "operationId": "getDebatesMatrix",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/queue/status": {
        "patch": {
            "tags": ["Debates"],
            "summary": "PATCH status",
            "operationId": "patchQueueStatus",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Debates"],
            "summary": "POST status",
            "operationId": "postQueueStatus",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/{id}/cancel": {
        "patch": {
            "tags": ["Debates"],
            "summary": "PATCH cancel",
            "operationId": "patchDebatesCancel",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/{id}/checkpoint": {
        "post": {
            "tags": ["Debates"],
            "summary": "POST checkpoint",
            "operationId": "postDebatesCheckpoint",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/{id}/checkpoints": {
        "get": {
            "tags": ["Debates"],
            "summary": "GET checkpoints",
            "operationId": "getDebatesCheckpoints",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/{id}/followup": {
        "patch": {
            "tags": ["Debates"],
            "summary": "PATCH followup",
            "operationId": "patchDebatesFollowup",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/{id}/followups": {
        "patch": {
            "tags": ["Debates"],
            "summary": "PATCH followups",
            "operationId": "patchDebatesFollowups",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Debates"],
            "summary": "POST followups",
            "operationId": "postDebatesFollowups",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/{id}/forks": {
        "patch": {
            "tags": ["Debates"],
            "summary": "PATCH forks",
            "operationId": "patchDebatesForks",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Debates"],
            "summary": "POST forks",
            "operationId": "postDebatesForks",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/{id}/red-team": {
        "post": {
            "tags": ["Debates"],
            "summary": "POST red-team",
            "operationId": "postDebatesRed-Team",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/debates/{id}/verification-report": {
        "patch": {
            "tags": ["Debates"],
            "summary": "PATCH verification-report",
            "operationId": "patchDebatesVerification-Report",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Debates"],
            "summary": "POST verification-report",
            "operationId": "postDebatesVerification-Report",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/devices/user/{id}": {
        "get": {
            "tags": ["Devices"],
            "summary": "GET {id}",
            "operationId": "getDevicesUser",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/devices/user/{id}/notify": {
        "post": {
            "tags": ["Devices"],
            "summary": "POST notify",
            "operationId": "postUserNotify",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/devices/{id}/notify": {
        "post": {
            "tags": ["Devices"],
            "summary": "POST notify",
            "operationId": "postDevicesNotify",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/documents{id}": {
        "get": {
            "tags": ["Documents{Param}"],
            "summary": "GET documents{id}",
            "operationId": "getDocuments{Param}",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/categorize": {
        "get": {
            "tags": ["Email"],
            "summary": "GET categorize",
            "operationId": "getEmailCategorize",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/categorize/apply-label": {
        "get": {
            "tags": ["Email"],
            "summary": "GET apply-label",
            "operationId": "getCategorizeApply-Label",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/categorize/batch": {
        "get": {
            "tags": ["Email"],
            "summary": "GET batch",
            "operationId": "getCategorizeBatch",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/context/boost": {
        "get": {
            "tags": ["Email"],
            "summary": "GET boost",
            "operationId": "getContextBoost",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/feedback": {
        "get": {
            "tags": ["Email"],
            "summary": "GET feedback",
            "operationId": "getEmailFeedback",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/feedback/batch": {
        "get": {
            "tags": ["Email"],
            "summary": "GET batch",
            "operationId": "getFeedbackBatch",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/followups/{id}/resolve": {
        "post": {
            "tags": ["Email"],
            "summary": "POST resolve",
            "operationId": "postFollowupsResolve",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/gmail/oauth/callback": {
        "get": {
            "tags": ["Email"],
            "summary": "GET callback",
            "operationId": "getOauthCallback",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/gmail/oauth/url": {
        "get": {
            "tags": ["Email"],
            "summary": "GET url",
            "operationId": "getOauthUrl",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/prioritize": {
        "get": {
            "tags": ["Email"],
            "summary": "GET prioritize",
            "operationId": "getEmailPrioritize",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/rank-inbox": {
        "get": {
            "tags": ["Email"],
            "summary": "GET rank-inbox",
            "operationId": "getEmailRank-Inbox",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/vip": {
        "get": {
            "tags": ["Email"],
            "summary": "GET vip",
            "operationId": "getEmailVip",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/{id}/snooze": {
        "delete": {
            "tags": ["Email"],
            "summary": "DELETE snooze",
            "operationId": "deleteEmailSnooze",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Email"],
            "summary": "POST snooze",
            "operationId": "postEmailSnooze",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/email/{id}/snooze-suggestions": {
        "get": {
            "tags": ["Email"],
            "summary": "GET snooze-suggestions",
            "operationId": "getEmailSnooze-Suggestions",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/evolution/ab-tests": {
        "get": {
            "tags": ["Evolution"],
            "summary": "GET ab-tests",
            "operationId": "getEvolutionAb-Tests",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/evolution/ab-tests/{id}": {
        "get": {
            "tags": ["Evolution"],
            "summary": "GET {id}",
            "operationId": "getEvolutionAb-Tests",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/evolution/ab-tests/{id}/active": {
        "get": {
            "tags": ["Evolution"],
            "summary": "GET active",
            "operationId": "getAb-TestsActive",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/facts": {
        "get": {
            "tags": ["Facts"],
            "summary": "GET facts",
            "operationId": "getFacts",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Facts"],
            "summary": "POST facts",
            "operationId": "postFacts",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/facts/batch": {
        "post": {
            "tags": ["Facts"],
            "summary": "POST batch",
            "operationId": "postFactsBatch",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/facts/batch/delete": {
        "post": {
            "tags": ["Facts"],
            "summary": "POST delete",
            "operationId": "postBatchDelete",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/facts/merge": {
        "post": {
            "tags": ["Facts"],
            "summary": "POST merge",
            "operationId": "postFactsMerge",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/facts/relationships": {
        "post": {
            "tags": ["Facts"],
            "summary": "POST relationships",
            "operationId": "postFactsRelationships",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/facts/relationships/{id}": {
        "delete": {
            "tags": ["Facts"],
            "summary": "DELETE {id}",
            "operationId": "deleteFactsRelationships",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Facts"],
            "summary": "GET {id}",
            "operationId": "getFactsRelationships",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "patch": {
            "tags": ["Facts"],
            "summary": "PATCH {id}",
            "operationId": "patchFactsRelationships",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/facts/stats": {
        "get": {
            "tags": ["Facts"],
            "summary": "GET stats",
            "operationId": "getFactsStats",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/facts/validate": {
        "post": {
            "tags": ["Facts"],
            "summary": "POST validate",
            "operationId": "postFactsValidate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/facts/{id}": {
        "delete": {
            "tags": ["Facts"],
            "summary": "DELETE {id}",
            "operationId": "deleteFacts",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Facts"],
            "summary": "GET {id}",
            "operationId": "getFacts",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "patch": {
            "tags": ["Facts"],
            "summary": "PATCH {id}",
            "operationId": "patchFacts",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/feedback/general": {
        "post": {
            "tags": ["Feedback"],
            "summary": "POST general",
            "operationId": "postFeedbackGeneral",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/feedback/nps": {
        "post": {
            "tags": ["Feedback"],
            "summary": "POST nps",
            "operationId": "postFeedbackNps",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/feedback/nps/summary": {
        "get": {
            "tags": ["Feedback"],
            "summary": "GET summary",
            "operationId": "getNpsSummary",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/feedback/prompts": {
        "get": {
            "tags": ["Feedback"],
            "summary": "GET prompts",
            "operationId": "getFeedbackPrompts",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gateway/devices/{id}/heartbeat": {
        "post": {
            "tags": ["Gateway"],
            "summary": "POST heartbeat",
            "operationId": "postDevicesHeartbeat",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gateway/messages/route": {
        "post": {
            "tags": ["Gateway"],
            "summary": "POST route",
            "operationId": "postMessagesRoute",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gateway/routing/rules": {
        "get": {
            "tags": ["Gateway"],
            "summary": "GET rules",
            "operationId": "getRoutingRules",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gateway/routing/stats": {
        "get": {
            "tags": ["Gateway"],
            "summary": "GET stats",
            "operationId": "getRoutingStats",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gauntlet/run": {
        "get": {
            "tags": ["Gauntlet"],
            "summary": "GET run",
            "operationId": "getGauntletRun",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gauntlet/{id}/receipt/verify": {
        "get": {
            "tags": ["Gauntlet"],
            "summary": "GET verify",
            "operationId": "getReceiptVerify",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/genesis/debates/{id}/tree": {
        "get": {
            "tags": ["Genesis"],
            "summary": "GET tree",
            "operationId": "getDebatesTree",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/genesis/genomes/{id}/descendants": {
        "get": {
            "tags": ["Genesis"],
            "summary": "GET descendants",
            "operationId": "getGenomesDescendants",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/genesis/genomes/{id}/lineage": {
        "get": {
            "tags": ["Genesis"],
            "summary": "GET lineage",
            "operationId": "getGenomesLineage",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gmail/callback": {
        "post": {
            "tags": ["Gmail"],
            "summary": "POST callback",
            "operationId": "postGmailCallback",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gmail/connection": {
        "get": {
            "tags": ["Gmail"],
            "summary": "GET connection",
            "operationId": "getGmailConnection",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gmail/debate-configs": {
        "post": {
            "tags": ["Gmail"],
            "summary": "POST debate-configs",
            "operationId": "postGmailDebate-Configs",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gmail/processed": {
        "get": {
            "tags": ["Gmail"],
            "summary": "GET processed",
            "operationId": "getGmailProcessed",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gmail/stats": {
        "get": {
            "tags": ["Gmail"],
            "summary": "GET stats",
            "operationId": "getGmailStats",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gmail/triage-rules": {
        "post": {
            "tags": ["Gmail"],
            "summary": "POST triage-rules",
            "operationId": "postGmailTriage-Rules",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gmail/triage-rules/{id}": {
        "delete": {
            "tags": ["Gmail"],
            "summary": "DELETE {id}",
            "operationId": "deleteGmailTriage-Rules",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "patch": {
            "tags": ["Gmail"],
            "summary": "PATCH {id}",
            "operationId": "patchGmailTriage-Rules",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gusto/connect": {
        "post": {
            "tags": ["Gusto"],
            "summary": "POST connect",
            "operationId": "postGustoConnect",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gusto/disconnect": {
        "post": {
            "tags": ["Gusto"],
            "summary": "POST disconnect",
            "operationId": "postGustoDisconnect",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gusto/employees": {
        "get": {
            "tags": ["Gusto"],
            "summary": "GET employees",
            "operationId": "getGustoEmployees",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gusto/payrolls": {
        "get": {
            "tags": ["Gusto"],
            "summary": "GET payrolls",
            "operationId": "getGustoPayrolls",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gusto/payrolls/{id}": {
        "get": {
            "tags": ["Gusto"],
            "summary": "GET {id}",
            "operationId": "getGustoPayrolls",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gusto/payrolls/{id}/journal-entry": {
        "post": {
            "tags": ["Gusto"],
            "summary": "POST journal-entry",
            "operationId": "postPayrollsJournal-Entry",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/gusto/status": {
        "get": {
            "tags": ["Gusto"],
            "summary": "GET status",
            "operationId": "getGustoStatus",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/inbox/actions": {
        "post": {
            "tags": ["Inbox"],
            "summary": "POST actions",
            "operationId": "postInboxActions",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/inbox/bulk-actions": {
        "post": {
            "tags": ["Inbox"],
            "summary": "POST bulk-actions",
            "operationId": "postInboxBulk-Actions",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/inbox/command": {
        "get": {
            "tags": ["Inbox"],
            "summary": "GET command",
            "operationId": "getInboxCommand",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/inbox/daily-digest": {
        "get": {
            "tags": ["Inbox"],
            "summary": "GET daily-digest",
            "operationId": "getInboxDaily-Digest",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/inbox/reprioritize": {
        "post": {
            "tags": ["Inbox"],
            "summary": "POST reprioritize",
            "operationId": "postInboxReprioritize",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/inbox/sender-profile": {
        "get": {
            "tags": ["Inbox"],
            "summary": "GET sender-profile",
            "operationId": "getInboxSender-Profile",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/introspection/leaderboard{id}": {
        "get": {
            "tags": ["Introspection"],
            "summary": "GET leaderboard{id}",
            "operationId": "getIntrospectionLeaderboard{Param}",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/km/checkpoints": {
        "get": {
            "tags": ["Km"],
            "summary": "GET checkpoints",
            "operationId": "getKmCheckpoints",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Km"],
            "summary": "POST checkpoints",
            "operationId": "postKmCheckpoints",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/km/checkpoints/{id}": {
        "get": {
            "tags": ["Km"],
            "summary": "GET {id}",
            "operationId": "getKmCheckpoints",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/km/checkpoints/{id}/compare": {
        "get": {
            "tags": ["Km"],
            "summary": "GET compare",
            "operationId": "getCheckpointsCompare",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/km/checkpoints/{id}/restore": {
        "post": {
            "tags": ["Km"],
            "summary": "POST restore",
            "operationId": "postCheckpointsRestore",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/embeddings": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST embeddings",
            "operationId": "postKnowledgeEmbeddings",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/entries/{id}/embeddings": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET embeddings",
            "operationId": "getEntriesEmbeddings",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/entries/{id}/sources": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET sources",
            "operationId": "getEntriesSources",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/export": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET export",
            "operationId": "getKnowledgeExport",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/analytics/quality/snapshot": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST snapshot",
            "operationId": "postQualitySnapshot",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/analytics/usage/record": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST record",
            "operationId": "postUsageRecord",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/confidence/decay": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST decay",
            "operationId": "postConfidenceDecay",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/confidence/event": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST event",
            "operationId": "postConfidenceEvent",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/contradictions/detect": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST detect",
            "operationId": "postContradictionsDetect",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/contradictions/{id}/resolve": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST resolve",
            "operationId": "postContradictionsResolve",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/culture/documents": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST documents",
            "operationId": "postCultureDocuments",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/culture/promote": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST promote",
            "operationId": "postCulturePromote",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/curation/policy": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST policy",
            "operationId": "postCurationPolicy",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/curation/run": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST run",
            "operationId": "postCurationRun",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/dashboard/metrics/reset": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST reset",
            "operationId": "postMetricsReset",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/dedup/auto-merge": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST auto-merge",
            "operationId": "postDedupAuto-Merge",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/dedup/merge": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST merge",
            "operationId": "postDedupMerge",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/extraction/debate": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST debate",
            "operationId": "postExtractionDebate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/extraction/promote": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST promote",
            "operationId": "postExtractionPromote",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/federation/regions": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET regions",
            "operationId": "getFederationRegions",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST regions",
            "operationId": "postFederationRegions",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/federation/regions/{id}": {
        "delete": {
            "tags": ["Knowledge"],
            "summary": "DELETE {id}",
            "operationId": "deleteFederationRegions",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/knowledge/mound/federation/status": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET status",
            "operationId": "getFederationStatus",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/federation/sync/all": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST all",
            "operationId": "postSyncAll",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/federation/sync/pull": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST pull",
            "operationId": "postSyncPull",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/federation/sync/push": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST push",
            "operationId": "postSyncPush",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/global": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET global",
            "operationId": "getMoundGlobal",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST global",
            "operationId": "postMoundGlobal",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/global/facts": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET facts",
            "operationId": "getGlobalFacts",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/global/promote": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST promote",
            "operationId": "postGlobalPromote",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/graph/{id}": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET {id}",
            "operationId": "getMoundGraph",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/index/repository": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST repository",
            "operationId": "postIndexRepository",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/my-shares": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET my-shares",
            "operationId": "getMoundMy-Shares",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/nodes/{id}/access": {
        "delete": {
            "tags": ["Knowledge"],
            "summary": "DELETE access",
            "operationId": "deleteNodesAccess",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET access",
            "operationId": "getNodesAccess",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST access",
            "operationId": "postNodesAccess",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/nodes/{id}/visibility": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET visibility",
            "operationId": "getNodesVisibility",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Knowledge"],
            "summary": "PUT visibility",
            "operationId": "putNodesVisibility",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/pruning/auto": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST auto",
            "operationId": "postPruningAuto",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/pruning/execute": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST execute",
            "operationId": "postPruningExecute",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/pruning/restore": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST restore",
            "operationId": "postPruningRestore",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/share": {
        "delete": {
            "tags": ["Knowledge"],
            "summary": "DELETE share",
            "operationId": "deleteMoundShare",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "patch": {
            "tags": ["Knowledge"],
            "summary": "PATCH share",
            "operationId": "patchMoundShare",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST share",
            "operationId": "postMoundShare",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/shared-with-me": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "GET shared-with-me",
            "operationId": "getMoundShared-With-Me",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/sync/consensus": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST consensus",
            "operationId": "postSyncConsensus",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/sync/continuum": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST continuum",
            "operationId": "postSyncContinuum",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/mound/sync/facts": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST facts",
            "operationId": "postSyncFacts",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/refresh": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST refresh",
            "operationId": "postKnowledgeRefresh",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/knowledge/validate": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "POST validate",
            "operationId": "postKnowledgeValidate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/media/audio/{id}": {
        "get": {
            "tags": ["Media"],
            "summary": "GET {id}",
            "operationId": "getMediaAudio",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/memory/compact": {
        "post": {
            "tags": ["Memory"],
            "summary": "POST compact",
            "operationId": "postMemoryCompact",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/memory/context": {
        "get": {
            "tags": ["Memory"],
            "summary": "GET context",
            "operationId": "getMemoryContext",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Memory"],
            "summary": "POST context",
            "operationId": "postMemoryContext",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/memory/critiques": {
        "get": {
            "tags": ["Memory"],
            "summary": "GET critiques",
            "operationId": "getMemoryCritiques",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/memory/pressure": {
        "get": {
            "tags": ["Memory"],
            "summary": "GET pressure",
            "operationId": "getMemoryPressure",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/memory/prune": {
        "post": {
            "tags": ["Memory"],
            "summary": "POST prune",
            "operationId": "postMemoryPrune",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/memory/query": {
        "post": {
            "tags": ["Memory"],
            "summary": "POST query",
            "operationId": "postMemoryQuery",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/memory/sync": {
        "post": {
            "tags": ["Memory"],
            "summary": "POST sync",
            "operationId": "postMemorySync",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/memory/tier/{id}": {
        "get": {
            "tags": ["Memory"],
            "summary": "GET {id}",
            "operationId": "getMemoryTier",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/memory/tiers": {
        "get": {
            "tags": ["Memory"],
            "summary": "GET tiers",
            "operationId": "getMemoryTiers",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/memory/{id}": {
        "put": {
            "tags": ["Memory"],
            "summary": "PUT {id}",
            "operationId": "putMemory",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/memory/{id}/move": {
        "post": {
            "tags": ["Memory"],
            "summary": "POST move",
            "operationId": "postMemoryMove",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/ml/models": {
        "get": {
            "tags": ["Ml"],
            "summary": "GET models",
            "operationId": "getMlModels",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/ml/stats": {
        "get": {
            "tags": ["Ml"],
            "summary": "GET stats",
            "operationId": "getMlStats",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/moments/by-type/{id}{id}": {
        "get": {
            "tags": ["Moments"],
            "summary": "GET {id}{id}",
            "operationId": "getMomentsBy-Type",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/moments/summary{id}": {
        "get": {
            "tags": ["Moments"],
            "summary": "GET summary{id}",
            "operationId": "getMomentsSummary{Param}",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/moments/timeline{id}": {
        "get": {
            "tags": ["Moments"],
            "summary": "GET timeline{id}",
            "operationId": "getMomentsTimeline{Param}",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/moments/trending{id}": {
        "get": {
            "tags": ["Moments"],
            "summary": "GET trending{id}",
            "operationId": "getMomentsTrending{Param}",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/onboarding/flow/step": {
        "put": {
            "tags": ["Onboarding"],
            "summary": "PUT step",
            "operationId": "putFlowStep",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/orchestration/deliberate": {
        "post": {
            "tags": ["Orchestration"],
            "summary": "POST deliberate",
            "operationId": "postOrchestrationDeliberate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/orchestration/deliberate/sync": {
        "post": {
            "tags": ["Orchestration"],
            "summary": "POST sync",
            "operationId": "postDeliberateSync",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/orchestration/status/{id}": {
        "get": {
            "tags": ["Orchestration"],
            "summary": "GET {id}",
            "operationId": "getOrchestrationStatus",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/orchestration/templates": {
        "get": {
            "tags": ["Orchestration"],
            "summary": "GET templates",
            "operationId": "getOrchestrationTemplates",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/outlook/conversations/{id}": {
        "get": {
            "tags": ["Outlook"],
            "summary": "GET {id}",
            "operationId": "getOutlookConversations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/outlook/messages": {
        "get": {
            "tags": ["Outlook"],
            "summary": "GET messages",
            "operationId": "getOutlookMessages",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/outlook/messages/reply": {
        "post": {
            "tags": ["Outlook"],
            "summary": "POST reply",
            "operationId": "postMessagesReply",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/outlook/messages/search": {
        "get": {
            "tags": ["Outlook"],
            "summary": "GET search",
            "operationId": "getMessagesSearch",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/outlook/messages/send": {
        "post": {
            "tags": ["Outlook"],
            "summary": "POST send",
            "operationId": "postMessagesSend",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/outlook/messages/{id}": {
        "delete": {
            "tags": ["Outlook"],
            "summary": "DELETE {id}",
            "operationId": "deleteOutlookMessages",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Outlook"],
            "summary": "GET {id}",
            "operationId": "getOutlookMessages",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/outlook/messages/{id}/move": {
        "post": {
            "tags": ["Outlook"],
            "summary": "POST move",
            "operationId": "postMessagesMove",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/outlook/messages/{id}/read": {
        "patch": {
            "tags": ["Outlook"],
            "summary": "PATCH read",
            "operationId": "patchMessagesRead",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/outlook/oauth/callback": {
        "post": {
            "tags": ["Outlook"],
            "summary": "POST callback",
            "operationId": "postOauthCallback",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/outlook/oauth/url": {
        "post": {
            "tags": ["Outlook"],
            "summary": "POST url",
            "operationId": "postOauthUrl",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/outlook/status": {
        "get": {
            "tags": ["Outlook"],
            "summary": "GET status",
            "operationId": "getOutlookStatus",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/partners/keys": {
        "get": {
            "tags": ["Partners"],
            "summary": "GET keys",
            "operationId": "getPartnersKeys",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Partners"],
            "summary": "POST keys",
            "operationId": "postPartnersKeys",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/partners/keys/{id}": {
        "delete": {
            "tags": ["Partners"],
            "summary": "DELETE {id}",
            "operationId": "deletePartnersKeys",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/partners/limits": {
        "get": {
            "tags": ["Partners"],
            "summary": "GET limits",
            "operationId": "getPartnersLimits",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/partners/me": {
        "get": {
            "tags": ["Partners"],
            "summary": "GET me",
            "operationId": "getPartnersMe",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/partners/register": {
        "post": {
            "tags": ["Partners"],
            "summary": "POST register",
            "operationId": "postPartnersRegister",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/partners/usage": {
        "get": {
            "tags": ["Partners"],
            "summary": "GET usage",
            "operationId": "getPartnersUsage",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/partners/webhooks": {
        "post": {
            "tags": ["Partners"],
            "summary": "POST webhooks",
            "operationId": "postPartnersWebhooks",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/plugins/query": {
        "post": {
            "tags": ["Plugins"],
            "summary": "POST query",
            "operationId": "postPluginsQuery",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/plugins/validate": {
        "post": {
            "tags": ["Plugins"],
            "summary": "POST validate",
            "operationId": "postPluginsValidate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/plugins/{id}": {
        "delete": {
            "tags": ["Plugins"],
            "summary": "DELETE {id}",
            "operationId": "deletePlugins",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/privacy/account": {
        "delete": {
            "tags": ["Privacy"],
            "summary": "DELETE account",
            "operationId": "deletePrivacyAccount",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/privacy/preferences": {
        "post": {
            "tags": ["Privacy"],
            "summary": "POST preferences",
            "operationId": "postPrivacyPreferences",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/probes/reports": {
        "get": {
            "tags": ["Probes"],
            "summary": "GET reports",
            "operationId": "getProbesReports",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/probes/reports/{id}": {
        "get": {
            "tags": ["Probes"],
            "summary": "GET {id}",
            "operationId": "getProbesReports",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/repository": {
        "get": {
            "tags": ["Repository"],
            "summary": "GET repository",
            "operationId": "getRepository",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/repository/batch": {
        "get": {
            "tags": ["Repository"],
            "summary": "GET batch",
            "operationId": "getRepositoryBatch",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/repository/incremental": {
        "get": {
            "tags": ["Repository"],
            "summary": "GET incremental",
            "operationId": "getRepositoryIncremental",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/repository/index": {
        "get": {
            "tags": ["Repository"],
            "summary": "GET index",
            "operationId": "getRepositoryIndex",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/repository/{id}/entities": {
        "get": {
            "tags": ["Repository"],
            "summary": "GET entities",
            "operationId": "getRepositoryEntities",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/repository/{id}/graph": {
        "get": {
            "tags": ["Repository"],
            "summary": "GET graph",
            "operationId": "getRepositoryGraph",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/repository/{id}/status": {
        "get": {
            "tags": ["Repository"],
            "summary": "GET status",
            "operationId": "getRepositoryStatus",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/retention/expiring{id}": {
        "get": {
            "tags": ["Retention"],
            "summary": "GET expiring{id}",
            "operationId": "getRetentionExpiring{Param}",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/rlm/context/{id}": {
        "delete": {
            "tags": ["Rlm"],
            "summary": "DELETE {id}",
            "operationId": "deleteRlmContext",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Rlm"],
            "summary": "GET {id}",
            "operationId": "getRlmContext",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/rlm/contexts": {
        "get": {
            "tags": ["Rlm"],
            "summary": "GET contexts",
            "operationId": "getRlmContexts",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/rlm/stats": {
        "get": {
            "tags": ["Rlm"],
            "summary": "GET stats",
            "operationId": "getRlmStats",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/rlm/strategies": {
        "get": {
            "tags": ["Rlm"],
            "summary": "GET strategies",
            "operationId": "getRlmStrategies",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/rlm/stream/modes": {
        "get": {
            "tags": ["Rlm"],
            "summary": "GET modes",
            "operationId": "getStreamModes",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/routing-rules": {
        "post": {
            "tags": ["Routing Rules"],
            "summary": "POST routing-rules",
            "operationId": "postRouting-Rules",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/routing-rules/evaluate": {
        "post": {
            "tags": ["Routing Rules"],
            "summary": "POST evaluate",
            "operationId": "postRouting-RulesEvaluate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/routing-rules/{id}": {
        "delete": {
            "tags": ["Routing Rules"],
            "summary": "DELETE {id}",
            "operationId": "deleteRouting-Rules",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "put": {
            "tags": ["Routing Rules"],
            "summary": "PUT {id}",
            "operationId": "putRouting-Rules",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/routing-rules/{id}/toggle": {
        "post": {
            "tags": ["Routing Rules"],
            "summary": "POST toggle",
            "operationId": "postRouting-RulesToggle",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/teams/debates/send": {
        "post": {
            "tags": ["Teams"],
            "summary": "POST send",
            "operationId": "postDebatesSend",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/teams/tenants/{id}": {
        "get": {
            "tags": ["Teams"],
            "summary": "GET {id}",
            "operationId": "getTeamsTenants",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "patch": {
            "tags": ["Teams"],
            "summary": "PATCH {id}",
            "operationId": "patchTeamsTenants",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/teams/tenants/{id}/channels/{id}/notifications": {
        "get": {
            "tags": ["Teams"],
            "summary": "GET notifications",
            "operationId": "getChannelsNotifications",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "patch": {
            "tags": ["Teams"],
            "summary": "PATCH notifications",
            "operationId": "patchChannelsNotifications",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/training/export/dpo": {
        "post": {
            "tags": ["Training"],
            "summary": "POST dpo",
            "operationId": "postExportDpo",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/training/export/gauntlet": {
        "post": {
            "tags": ["Training"],
            "summary": "POST gauntlet",
            "operationId": "postExportGauntlet",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/training/export/sft": {
        "post": {
            "tags": ["Training"],
            "summary": "POST sft",
            "operationId": "postExportSft",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/training/jobs/{id}": {
        "delete": {
            "tags": ["Training"],
            "summary": "DELETE {id}",
            "operationId": "deleteTrainingJobs",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Training"],
            "summary": "GET {id}",
            "operationId": "getTrainingJobs",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/training/jobs/{id}/artifacts": {
        "get": {
            "tags": ["Training"],
            "summary": "GET artifacts",
            "operationId": "getJobsArtifacts",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/training/jobs/{id}/complete": {
        "post": {
            "tags": ["Training"],
            "summary": "POST complete",
            "operationId": "postJobsComplete",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/training/jobs/{id}/export": {
        "post": {
            "tags": ["Training"],
            "summary": "POST export",
            "operationId": "postJobsExport",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/training/jobs/{id}/metrics": {
        "get": {
            "tags": ["Training"],
            "summary": "GET metrics",
            "operationId": "getJobsMetrics",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/training/jobs/{id}/start": {
        "post": {
            "tags": ["Training"],
            "summary": "POST start",
            "operationId": "postJobsStart",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/transcription/config": {
        "get": {
            "tags": ["Transcription"],
            "summary": "GET config",
            "operationId": "getTranscriptionConfig",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/transcription/formats": {
        "get": {
            "tags": ["Transcription"],
            "summary": "GET formats",
            "operationId": "getTranscriptionFormats",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/transcription/upload": {
        "post": {
            "tags": ["Transcription"],
            "summary": "POST upload",
            "operationId": "postTranscriptionUpload",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/transcription/{id}": {
        "delete": {
            "tags": ["Transcription"],
            "summary": "DELETE {id}",
            "operationId": "deleteTranscription",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Transcription"],
            "summary": "GET {id}",
            "operationId": "getTranscription",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/transcription/{id}/segments": {
        "get": {
            "tags": ["Transcription"],
            "summary": "GET segments",
            "operationId": "getTranscriptionSegments",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/uncertainty/agent": {
        "get": {
            "tags": ["Uncertainty"],
            "summary": "GET agent",
            "operationId": "getUncertaintyAgent",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/uncertainty/debate": {
        "get": {
            "tags": ["Uncertainty"],
            "summary": "GET debate",
            "operationId": "getUncertaintyDebate",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/uncertainty/estimate": {
        "get": {
            "tags": ["Uncertainty"],
            "summary": "GET estimate",
            "operationId": "getUncertaintyEstimate",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/uncertainty/followups": {
        "get": {
            "tags": ["Uncertainty"],
            "summary": "GET followups",
            "operationId": "getUncertaintyFollowups",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/bulk": {
        "delete": {
            "tags": ["Webhooks"],
            "summary": "DELETE bulk",
            "operationId": "deleteWebhooksBulk",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/webhooks/events/categories": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET categories",
            "operationId": "getEventsCategories",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/pause-all": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "POST pause-all",
            "operationId": "postWebhooksPause-All",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/resume-all": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "POST resume-all",
            "operationId": "postWebhooksResume-All",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/deliveries": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET deliveries",
            "operationId": "getWebhooksDeliveries",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/deliveries/{id}": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET {id}",
            "operationId": "getWebhooksDeliveries",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/deliveries/{id}/retry": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "POST retry",
            "operationId": "postDeliveriesRetry",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/events": {
        "delete": {
            "tags": ["Webhooks"],
            "summary": "DELETE events",
            "operationId": "deleteWebhooksEvents",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Webhooks"],
            "summary": "POST events",
            "operationId": "postWebhooksEvents",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/retry-policy": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET retry-policy",
            "operationId": "getWebhooksRetry-Policy",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Webhooks"],
            "summary": "PUT retry-policy",
            "operationId": "putWebhooksRetry-Policy",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/rotate-secret": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "POST rotate-secret",
            "operationId": "postWebhooksRotate-Secret",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/signing": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET signing",
            "operationId": "getWebhooksSigning",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/stats": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET stats",
            "operationId": "getWebhooksStats",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/workflows/{id}": {
        "put": {
            "tags": ["Workflows"],
            "summary": "PUT {id}",
            "operationId": "putWorkflows",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/backups/cleanup": {
        "post": {
            "tags": ["V2"],
            "summary": "POST cleanup",
            "operationId": "postBackupsCleanup",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/backups/stats": {
        "get": {
            "tags": ["V2"],
            "summary": "GET stats",
            "operationId": "getBackupsStats",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/backups/{id}/restore-test": {
        "post": {
            "tags": ["V2"],
            "summary": "POST restore-test",
            "operationId": "postBackupsRestore-Test",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/backups/{id}/verify": {
        "post": {
            "tags": ["V2"],
            "summary": "POST verify",
            "operationId": "postBackupsVerify",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/backups/{id}/verify-comprehensive": {
        "post": {
            "tags": ["V2"],
            "summary": "POST verify-comprehensive",
            "operationId": "postBackupsVerify-Comprehensive",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/compliance/audit-events": {
        "get": {
            "tags": ["V2"],
            "summary": "GET audit-events",
            "operationId": "getComplianceAudit-Events",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/compliance/audit-verify": {
        "post": {
            "tags": ["V2"],
            "summary": "POST audit-verify",
            "operationId": "postComplianceAudit-Verify",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/compliance/gdpr-export": {
        "get": {
            "tags": ["V2"],
            "summary": "GET gdpr-export",
            "operationId": "getComplianceGdpr-Export",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/compliance/gdpr/right-to-be-forgotten": {
        "post": {
            "tags": ["V2"],
            "summary": "POST right-to-be-forgotten",
            "operationId": "postGdprRight-To-Be-Forgotten",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/compliance/soc2-report": {
        "get": {
            "tags": ["V2"],
            "summary": "GET soc2-report",
            "operationId": "getComplianceSoc2-Report",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/compliance/status": {
        "get": {
            "tags": ["V2"],
            "summary": "GET status",
            "operationId": "getComplianceStatus",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/integrations/wizard": {
        "post": {
            "tags": ["V2"],
            "summary": "POST wizard",
            "operationId": "postIntegrationsWizard",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/integrations/wizard/preflight": {
        "post": {
            "tags": ["V2"],
            "summary": "POST preflight",
            "operationId": "postWizardPreflight",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/integrations/wizard/providers/{id}": {
        "delete": {
            "tags": ["V2"],
            "summary": "DELETE {id}",
            "operationId": "deleteWizardProviders",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["V2"],
            "summary": "GET {id}",
            "operationId": "getWizardProviders",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/integrations/wizard/providers/{id}/install": {
        "post": {
            "tags": ["V2"],
            "summary": "POST install",
            "operationId": "postProvidersInstall",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/integrations/wizard/providers/{id}/refresh": {
        "post": {
            "tags": ["V2"],
            "summary": "POST refresh",
            "operationId": "postProvidersRefresh",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/integrations/wizard/recommendations": {
        "get": {
            "tags": ["V2"],
            "summary": "GET recommendations",
            "operationId": "getWizardRecommendations",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v2/integrations/wizard/status/{id}": {
        "get": {
            "tags": ["V2"],
            "summary": "GET {id}",
            "operationId": "getWizardStatus",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/verticals/{id}/agent": {
        "post": {
            "tags": ["Verticals"],
            "summary": "POST agent",
            "operationId": "postVerticalsAgent",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/verticals/{id}/config": {
        "put": {
            "tags": ["Verticals"],
            "summary": "PUT config",
            "operationId": "putVerticalsConfig",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/verticals/{id}/debate": {
        "post": {
            "tags": ["Verticals"],
            "summary": "POST debate",
            "operationId": "postVerticalsDebate",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/workspaces/{id}": {
        "put": {
            "tags": ["Workspaces"],
            "summary": "PUT {id}",
            "operationId": "putWorkspaces",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/workspaces/{id}/members": {
        "get": {
            "tags": ["Workspaces"],
            "summary": "GET members",
            "operationId": "getWorkspacesMembers",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/workspaces/{id}/members/{id}": {
        "delete": {
            "tags": ["Workspaces"],
            "summary": "DELETE {id}",
            "operationId": "deleteWorkspacesMembers",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "put": {
            "tags": ["Workspaces"],
            "summary": "PUT {id}",
            "operationId": "putWorkspacesMembers",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/workspaces/{id}/profiles": {
        "get": {
            "tags": ["Workspaces"],
            "summary": "GET profiles",
            "operationId": "getWorkspacesProfiles",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/accounts": {
        "get": {
            "tags": ["Inbox"],
            "summary": "GET accounts",
            "operationId": "getInboxAccounts",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/accounts/{id}": {
        "delete": {
            "tags": ["Inbox"],
            "summary": "DELETE {id}",
            "operationId": "deleteInboxAccounts",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/inbox/bulk-action": {
        "post": {
            "tags": ["Inbox"],
            "summary": "POST bulk-action",
            "operationId": "postInboxBulk-Action",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/connect": {
        "post": {
            "tags": ["Inbox"],
            "summary": "POST connect",
            "operationId": "postInboxConnect",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/messages": {
        "get": {
            "tags": ["Inbox"],
            "summary": "GET messages",
            "operationId": "getInboxMessages",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/messages/send": {
        "post": {
            "tags": ["Inbox"],
            "summary": "POST send",
            "operationId": "postMessagesSend",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/messages/{id}": {
        "get": {
            "tags": ["Inbox"],
            "summary": "GET {id}",
            "operationId": "getInboxMessages",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/messages/{id}/reply": {
        "post": {
            "tags": ["Inbox"],
            "summary": "POST reply",
            "operationId": "postMessagesReply",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/oauth/gmail": {
        "get": {
            "tags": ["Inbox"],
            "summary": "GET gmail",
            "operationId": "getOauthGmail",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/oauth/outlook": {
        "get": {
            "tags": ["Inbox"],
            "summary": "GET outlook",
            "operationId": "getOauthOutlook",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/stats": {
        "get": {
            "tags": ["Inbox"],
            "summary": "GET stats",
            "operationId": "getInboxStats",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/trends": {
        "get": {
            "tags": ["Inbox"],
            "summary": "GET trends",
            "operationId": "getInboxTrends",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/inbox/triage": {
        "post": {
            "tags": ["Inbox"],
            "summary": "POST triage",
            "operationId": "postInboxTriage",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/scim/v2/Groups": {
        "get": {
            "tags": ["Scim"],
            "summary": "GET Groups",
            "operationId": "getV2Groups",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/scim/v2/Users": {
        "get": {
            "tags": ["Scim"],
            "summary": "GET Users",
            "operationId": "getV2Users",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
}


def _method_stub(
    tag: str,
    method: str,
    summary: str,
    *,
    op_id: str,
    has_path_param: bool = False,
    has_body: bool = False,
):
    """Build a minimal endpoint operation dict."""
    op: dict = {
        "tags": [tag],
        "summary": summary,
        "operationId": op_id,
        "responses": {
            "200": _ok_response("Success", {"success": {"type": "boolean"}}),
        },
    }
    if has_path_param:
        op["parameters"] = [
            {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
        ]
    if has_body:
        op["requestBody"] = {"content": {"application/json": {"schema": {"type": "object"}}}}
    return op


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
    "/api/agent/{id}/persona": {
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
        "post": _method_stub("Scim", "POST", "Create group", op_id="postScimGroups", has_body=True),
    },
    "/scim/v2/Users": {
        "post": _method_stub("Scim", "POST", "Create user", op_id="postScimUsers", has_body=True),
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
            "Workspace", "DELETE", "Delete workspace", op_id="deleteWorkspace", has_path_param=True
        ),
        "get": _method_stub(
            "Workspace", "GET", "Get workspace", op_id="getWorkspace", has_path_param=True
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
        SDK_MISSING_ENDPOINTS[path].update(methods)
    else:
        SDK_MISSING_ENDPOINTS[path] = methods
