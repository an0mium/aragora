"""
OpenAPI endpoint definitions for Advertising Platform API.

Unified API for managing advertising campaigns across platforms:
- Google Ads (Search, Display, Shopping, YouTube)
- Meta Ads (Facebook, Instagram)
- LinkedIn Ads (B2B)
- Microsoft Ads (Bing)
"""

from aragora.server.openapi.helpers import (
    STANDARD_ERRORS,
)

ADVERTISING_ENDPOINTS = {
    "/api/v1/advertising/platforms": {
        "get": {
            "tags": ["Advertising"],
            "summary": "List advertising platforms",
            "description": """List all supported advertising platforms and their connection status.

**Supported platforms:**
- `google_ads`: Google Ads (Search, Display, Shopping, YouTube)
- `meta_ads`: Meta Ads (Facebook, Instagram)
- `linkedin_ads`: LinkedIn Ads (B2B professional targeting)
- `microsoft_ads`: Microsoft Ads (Bing search)

**Response includes:**
- Platform ID, name, and description
- Available features per platform
- Connection status and timestamp""",
            "operationId": "listAdvertisingPlatforms",
            "responses": {
                "200": {
                    "description": "List of advertising platforms",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "platforms": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "description": {"type": "string"},
                                                "features": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "connected": {"type": "boolean"},
                                                "connected_at": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                    "nullable": True,
                                                },
                                            },
                                        },
                                    },
                                    "connected_count": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/advertising/connect": {
        "post": {
            "tags": ["Advertising"],
            "summary": "Connect advertising platform",
            "description": """Connect an advertising platform with API credentials.

**Requires:** `advertising:configure` permission

**Required credentials by platform:**

**Google Ads:**
- `developer_token`, `client_id`, `client_secret`
- `refresh_token`, `customer_id`

**Meta Ads:**
- `access_token`, `ad_account_id`

**LinkedIn Ads:**
- `access_token`, `ad_account_id`

**Microsoft Ads:**
- `developer_token`, `client_id`, `client_secret`
- `refresh_token`, `account_id`, `customer_id`""",
            "operationId": "connectAdvertisingPlatform",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["platform", "credentials"],
                            "properties": {
                                "platform": {
                                    "type": "string",
                                    "enum": [
                                        "google_ads",
                                        "meta_ads",
                                        "linkedin_ads",
                                        "microsoft_ads",
                                    ],
                                    "description": "Platform to connect",
                                },
                                "credentials": {
                                    "type": "object",
                                    "description": "Platform-specific credentials",
                                },
                            },
                        },
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Platform connected",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                    "platform": {"type": "string"},
                                    "connected_at": {"type": "string", "format": "date-time"},
                                },
                            }
                        }
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/advertising/{platform}": {
        "delete": {
            "tags": ["Advertising"],
            "summary": "Disconnect advertising platform",
            "description": """Disconnect an advertising platform.

**Requires:** `advertising:configure` permission

Closes the connector and removes stored credentials.""",
            "operationId": "disconnectAdvertisingPlatform",
            "parameters": [
                {
                    "name": "platform",
                    "in": "path",
                    "required": True,
                    "description": "Platform identifier",
                    "schema": {
                        "type": "string",
                        "enum": ["google_ads", "meta_ads", "linkedin_ads", "microsoft_ads"],
                    },
                },
            ],
            "responses": {
                "200": {
                    "description": "Platform disconnected",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                    "platform": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/advertising/campaigns": {
        "get": {
            "tags": ["Advertising"],
            "summary": "List all campaigns (cross-platform)",
            "description": """List campaigns from all connected advertising platforms.

**Requires:** `advertising:read` permission

Aggregates campaigns across Google Ads, Meta Ads, LinkedIn Ads,
and Microsoft Ads into a unified format.""",
            "operationId": "listAllCampaigns",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by campaign status",
                    "schema": {
                        "type": "string",
                        "enum": ["ENABLED", "PAUSED", "REMOVED"],
                    },
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of campaigns to return",
                    "schema": {"type": "integer", "default": 100, "maximum": 500},
                },
            ],
            "responses": {
                "200": {
                    "description": "List of campaigns",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "campaigns": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/UnifiedCampaign"},
                                    },
                                    "total": {"type": "integer"},
                                    "platforms_queried": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/advertising/{platform}/campaigns": {
        "get": {
            "tags": ["Advertising"],
            "summary": "List platform campaigns",
            "description": """List campaigns from a specific advertising platform.

**Requires:** `advertising:read` permission""",
            "operationId": "listPlatformCampaigns",
            "parameters": [
                {
                    "name": "platform",
                    "in": "path",
                    "required": True,
                    "description": "Platform identifier",
                    "schema": {
                        "type": "string",
                        "enum": ["google_ads", "meta_ads", "linkedin_ads", "microsoft_ads"],
                    },
                },
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by campaign status",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "List of campaigns",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "campaigns": {"type": "array", "items": {"type": "object"}},
                                    "total": {"type": "integer"},
                                    "platform": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Advertising"],
            "summary": "Create campaign",
            "description": """Create a new campaign on a specific platform.

**Requires:** `advertising:write` permission

**Platform-specific requirements:**
- LinkedIn Ads requires `campaign_group_id`""",
            "operationId": "createCampaign",
            "parameters": [
                {
                    "name": "platform",
                    "in": "path",
                    "required": True,
                    "description": "Platform identifier",
                    "schema": {
                        "type": "string",
                        "enum": ["google_ads", "meta_ads", "linkedin_ads", "microsoft_ads"],
                    },
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {
                                "name": {"type": "string", "description": "Campaign name"},
                                "daily_budget": {"type": "number", "description": "Daily budget"},
                                "campaign_type": {"type": "string", "description": "Campaign type"},
                                "objective": {
                                    "type": "string",
                                    "description": "Campaign objective",
                                },
                                "campaign_group_id": {
                                    "type": "string",
                                    "description": "Campaign group ID (LinkedIn required)",
                                },
                            },
                        },
                    }
                },
            },
            "responses": {
                "201": {
                    "description": "Campaign created",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "campaign_id": {"type": "string"},
                                    "platform": {"type": "string"},
                                    "name": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/advertising/{platform}/campaigns/{campaign_id}": {
        "get": {
            "tags": ["Advertising"],
            "summary": "Get campaign details",
            "description": """Get details for a specific campaign.

**Requires:** `advertising:read` permission""",
            "operationId": "getCampaign",
            "parameters": [
                {
                    "name": "platform",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "campaign_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Campaign details",
                    "content": {"application/json": {"schema": {"type": "object"}}},
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "put": {
            "tags": ["Advertising"],
            "summary": "Update campaign",
            "description": """Update a campaign's status or budget.

**Requires:** `advertising:write` permission

**Updatable fields:**
- `status`: Campaign status (ENABLED, PAUSED)
- `daily_budget`: Daily budget amount""",
            "operationId": "updateCampaign",
            "parameters": [
                {
                    "name": "platform",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "campaign_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": ["ENABLED", "PAUSED"],
                                },
                                "daily_budget": {"type": "number"},
                            },
                        },
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Campaign updated",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                    "campaign_id": {"type": "string"},
                                    "platform": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/advertising/performance": {
        "get": {
            "tags": ["Advertising"],
            "summary": "Cross-platform performance",
            "description": """Get performance metrics across all connected platforms.

**Requires:** `advertising:read` permission

**Metrics included:**
- Impressions, clicks, cost
- Conversions and conversion value
- CTR, CPC, CPM, ROAS

**Aggregation:** Totals across all platforms with per-platform breakdown.""",
            "operationId": "getCrossplatformPerformance",
            "parameters": [
                {
                    "name": "days",
                    "in": "query",
                    "description": "Number of days to analyze",
                    "schema": {"type": "integer", "default": 30, "maximum": 90},
                },
            ],
            "responses": {
                "200": {
                    "description": "Performance metrics",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "date_range": {
                                        "type": "object",
                                        "properties": {
                                            "start": {"type": "string", "format": "date"},
                                            "end": {"type": "string", "format": "date"},
                                        },
                                    },
                                    "platforms": {
                                        "type": "array",
                                        "items": {"type": "object"},
                                    },
                                    "totals": {
                                        "type": "object",
                                        "properties": {
                                            "impressions": {"type": "integer"},
                                            "clicks": {"type": "integer"},
                                            "cost": {"type": "number"},
                                            "conversions": {"type": "integer"},
                                            "conversion_value": {"type": "number"},
                                            "ctr": {"type": "number"},
                                            "cpc": {"type": "number"},
                                            "cpm": {"type": "number"},
                                            "roas": {"type": "number"},
                                        },
                                    },
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/advertising/{platform}/performance": {
        "get": {
            "tags": ["Advertising"],
            "summary": "Platform performance",
            "description": """Get performance metrics for a specific platform.

**Requires:** `advertising:read` permission""",
            "operationId": "getPlatformPerformance",
            "parameters": [
                {
                    "name": "platform",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "days",
                    "in": "query",
                    "description": "Number of days to analyze",
                    "schema": {"type": "integer", "default": 30},
                },
            ],
            "responses": {
                "200": {
                    "description": "Performance metrics",
                    "content": {"application/json": {"schema": {"type": "object"}}},
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/advertising/analyze": {
        "post": {
            "tags": ["Advertising"],
            "summary": "Analyze performance (multi-agent)",
            "description": """Run multi-agent analysis on advertising performance.

**Requires:** `advertising:analyze` permission

**Analysis types:**
- `performance_review`: Overall performance assessment
- `budget_optimization`: Budget allocation recommendations
- `creative_analysis`: Creative performance insights

Returns AI-generated summary, recommendations, and insights.""",
            "operationId": "analyzeAdvertisingPerformance",
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "platforms": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Platforms to analyze (default: all connected)",
                                },
                                "type": {
                                    "type": "string",
                                    "default": "performance_review",
                                    "description": "Analysis type",
                                },
                                "days": {
                                    "type": "integer",
                                    "default": 30,
                                    "description": "Days to analyze",
                                },
                            },
                        },
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Analysis results",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "analysis_id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "status": {"type": "string"},
                                    "date_range": {"type": "object"},
                                    "platforms_analyzed": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "summary": {"type": "object"},
                                    "recommendations": {
                                        "type": "array",
                                        "items": {"type": "object"},
                                    },
                                    "insights": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            }
                        }
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/advertising/budget-recommendations": {
        "get": {
            "tags": ["Advertising"],
            "summary": "Get budget recommendations",
            "description": """Get AI-powered budget allocation recommendations.

**Requires:** `advertising:read` permission

**Objectives:**
- `balanced`: Equal weight to reach and conversions
- `awareness`: Maximize reach and impressions
- `conversions`: Maximize conversion volume

Recommendations based on historical ROAS performance.""",
            "operationId": "getBudgetRecommendations",
            "parameters": [
                {
                    "name": "budget",
                    "in": "query",
                    "description": "Total budget to allocate",
                    "schema": {"type": "number", "default": 10000},
                },
                {
                    "name": "objective",
                    "in": "query",
                    "description": "Optimization objective",
                    "schema": {
                        "type": "string",
                        "enum": ["balanced", "awareness", "conversions"],
                        "default": "balanced",
                    },
                },
            ],
            "responses": {
                "200": {
                    "description": "Budget recommendations",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "total_budget": {"type": "number"},
                                    "objective": {"type": "string"},
                                    "recommendations": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "platform": {"type": "string"},
                                                "recommended_budget": {"type": "number"},
                                                "share_percentage": {"type": "number"},
                                                "expected_roas": {"type": "number"},
                                            },
                                        },
                                    },
                                    "rationale": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
}
