"""Agent endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, _array_response, STANDARD_ERRORS

AGENT_ENDPOINTS = {
    "/api/agents": {
        "get": {
            "tags": ["Agents"],
            "summary": "List all agents",
            "description": """Get list of all known AI agents with optional performance statistics.

**Rate Limit:** 60 requests per minute (free), 300/min (pro), 1000/min (enterprise)

**Response:** Returns agent metadata including name, model, provider, and ELO rating.
Optionally includes detailed statistics when `include_stats=true`.""",
            "operationId": "listAgents",
            "parameters": [
                {
                    "name": "include_stats",
                    "in": "query",
                    "description": "Include detailed performance statistics for each agent",
                    "schema": {"type": "boolean", "default": False},
                },
            ],
            "responses": {"200": _array_response("List of agents", "Agent")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/leaderboard": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent leaderboard",
            "description": """Get agents ranked by ELO rating.

**Rate Limit:** 60 requests per minute

**Sorting:** Agents are sorted by ELO rating in descending order.
Use `domain` parameter to filter by expertise area (e.g., "coding", "science", "writing").""",
            "operationId": "getLeaderboard",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of agents to return",
                    "schema": {"type": "integer", "default": 20, "maximum": 100, "minimum": 1},
                },
                {
                    "name": "domain",
                    "in": "query",
                    "description": "Filter by expertise domain (e.g., coding, science, writing)",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Agent rankings",
                    {
                        "agents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "elo": {"type": "number"},
                                    "rank": {"type": "integer"},
                                },
                            },
                        }
                    },
                )
            },
        },
    },
    "/api/rankings": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent rankings",
            "description": """Alternative endpoint for agent rankings. Returns same data as /api/leaderboard.

**Deprecated:** Consider using /api/leaderboard instead.""",
            "operationId": "getAgentRankings",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of agents to return",
                    "schema": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Agent rankings",
                    {
                        "agents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "elo": {"type": "number"},
                                    "rank": {"type": "integer"},
                                },
                            },
                        }
                    },
                )
            },
        },
    },
    "/api/leaderboard-view": {
        "get": {
            "tags": ["Agents"],
            "summary": "Leaderboard view data",
            "description": """Get pre-formatted leaderboard data optimized for frontend display.

Includes display-ready fields like rank badges, formatted ratings, and trend indicators.""",
            "operationId": "getLeaderboardView",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of agents to return",
                    "schema": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                },
                {
                    "name": "domain",
                    "in": "query",
                    "description": "Filter by expertise domain",
                    "schema": {"type": "string"},
                },
                {
                    "name": "loop_id",
                    "in": "query",
                    "description": "Filter by nomic loop ID",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Leaderboard view",
                    {
                        "agents": {"type": "array", "items": {"type": "object"}},
                        "total": {"type": "integer"},
                    },
                )
            },
        },
    },
    "/api/agent/{name}/profile": {
        "get": {
            "tags": ["Agents"],
            "summary": "Get agent profile",
            "description": """Get detailed profile for a specific agent.

**Response includes:**
- Basic info (name, model, provider)
- Current ELO rating and rank
- Win/loss record
- Expertise domains
- Recent performance trends""",
            "operationId": "getAgentProfile",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name (e.g., 'claude', 'gpt-4', 'gemini')",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Agent profile", "Agent"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/agent/{name}/history": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent match history",
            "description": """Get recent debate matches for an agent.

**Response includes:** debate topic, opponents, outcome, ELO change per match.""",
            "operationId": "getAgentHistory",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of matches to return",
                    "schema": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Match history",
                    {
                        "matches": {"type": "array", "items": {"type": "object"}},
                        "total": {"type": "integer"},
                    },
                )
            },
        },
    },
    "/api/agent/{name}/calibration": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent calibration data",
            "description": """Get calibration metrics showing how well an agent's confidence matches actual accuracy.

**Calibration:** A well-calibrated agent's stated confidence (e.g., 80%) should match actual correctness rate.""",
            "operationId": "getAgentCalibration",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {"200": _ok_response("Calibration data", "Calibration")},
        },
    },
    "/api/agent/{name}/calibration-curve": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent calibration curve",
            "description": """Get calibration curve data for visualization.

Returns confidence buckets (0-10%, 10-20%, etc.) with actual accuracy for each bucket.""",
            "operationId": "getAgentCalibrationCurve",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Calibration curve data",
                    {
                        "buckets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "range": {"type": "string"},
                                    "predicted": {"type": "number"},
                                    "actual": {"type": "number"},
                                },
                            },
                        }
                    },
                )
            },
        },
    },
    "/api/agent/{name}/calibration-summary": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent calibration summary",
            "description": """Get summarized calibration metrics.

Returns Brier score, calibration error, and overconfidence/underconfidence indicators.""",
            "operationId": "getAgentCalibrationSummary",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Calibration summary",
                    {
                        "brier_score": {"type": "number"},
                        "calibration_error": {"type": "number"},
                        "overconfidence": {"type": "boolean"},
                    },
                )
            },
        },
    },
    "/api/agent/{name}/consistency": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent consistency score",
            "description": """Get consistency metrics measuring how stable an agent's positions are over time.

**High consistency:** Agent maintains coherent positions across similar debates.
**Low consistency:** Agent frequently contradicts previous positions.""",
            "operationId": "getAgentConsistency",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Consistency metrics",
                    {"score": {"type": "number"}, "total_debates": {"type": "integer"}},
                )
            },
        },
    },
    "/api/agent/{name}/flips": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent position flips",
            "description": """Get instances where the agent changed positions during or between debates.

Useful for understanding agent reasoning and persuadability.""",
            "operationId": "getAgentFlips",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of flips to return",
                    "schema": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Position flips",
                    {
                        "flips": {"type": "array", "items": {"type": "object"}},
                        "total": {"type": "integer"},
                    },
                )
            },
        },
    },
    "/api/agent/{name}/network": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent relationship network",
            "description": """Get the agent's relationship network showing connections to other agents.

Includes agreement rates, collaboration frequency, and influence scores.""",
            "operationId": "getAgentNetwork",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Relationship network",
                    {
                        "nodes": {"type": "array", "items": {"type": "object"}},
                        "edges": {"type": "array", "items": {"type": "object"}},
                    },
                )
            },
        },
    },
    "/api/agent/{name}/rivals": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent rivals",
            "description": """Get agents that frequently disagree with this agent.

Rivals are determined by low agreement rates and competitive debate history.""",
            "operationId": "getAgentRivals",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Rival agents",
                    {
                        "rivals": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "agreement_rate": {"type": "number"},
                                },
                            },
                        }
                    },
                )
            },
        },
    },
    "/api/agent/{name}/allies": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent allies",
            "description": """Get agents that frequently agree with this agent.

Allies are determined by high agreement rates and collaborative debate history.""",
            "operationId": "getAgentAllies",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Allied agents",
                    {
                        "allies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "agreement_rate": {"type": "number"},
                                },
                            },
                        }
                    },
                )
            },
        },
    },
    "/api/agent/{name}/moments": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent moments",
            "description": """Get significant moments from this agent's debate history.

**Moment types:** upset wins, consensus shifts, notable arguments, calibration milestones.""",
            "operationId": "getAgentMoments",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Agent moments",
                    {
                        "moments": {"type": "array", "items": {"type": "object"}},
                        "total": {"type": "integer"},
                    },
                )
            },
        },
    },
    "/api/agent/{name}/reputation": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent reputation",
            "description": """Get reputation metrics derived from debate performance.

Includes trustworthiness, expertise recognition, and community standing.""",
            "operationId": "getAgentReputation",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Reputation data",
                    {
                        "trustworthiness": {"type": "number"},
                        "expertise": {"type": "number"},
                        "standing": {"type": "string"},
                    },
                )
            },
        },
    },
    "/api/agent/{name}/persona": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent persona",
            "description": """Get the agent's persona profile including communication style and reasoning patterns.""",
            "operationId": "getAgentPersona",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Persona data",
                    {
                        "name": {"type": "string"},
                        "style": {"type": "string"},
                        "traits": {"type": "array", "items": {"type": "string"}},
                    },
                )
            },
        },
        "delete": {
            "tags": ["Agents"],
            "summary": "Delete persona",
            "description": "Delete an agent's custom persona configuration.",
            "operationId": "deleteAgentPersona",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": {"description": "Agent not found"},
            },
        },
        "put": {
            "tags": ["Agents"],
            "summary": "Update persona",
            "description": "Update an agent's persona configuration.",
            "operationId": "putAgentPersona",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/agent/{name}/grounded-persona": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent grounded persona",
            "description": """Get persona derived from actual debate performance rather than training data.

The grounded persona reflects observed behavior patterns across debates.""",
            "operationId": "getAgentGroundedPersona",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Grounded persona",
                    {
                        "name": {"type": "string"},
                        "observed_traits": {"type": "array", "items": {"type": "string"}},
                        "debate_count": {"type": "integer"},
                    },
                )
            },
        },
    },
    "/api/agent/{name}/identity-prompt": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent identity prompt",
            "description": """Get the system prompt used to establish this agent's identity in debates.""",
            "operationId": "getAgentIdentityPrompt",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Identity prompt", {"prompt": {"type": "string"}, "agent": {"type": "string"}}
                )
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/agent/{name}/performance": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent performance",
            "description": """Get detailed performance metrics for the agent.

**Metrics include:** win rate, average ELO delta, debate participation, consensus contribution.""",
            "operationId": "getAgentPerformance",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Performance metrics",
                    {
                        "win_rate": {"type": "number"},
                        "avg_elo_delta": {"type": "number"},
                        "debates_participated": {"type": "integer"},
                    },
                )
            },
        },
    },
    "/api/agent/{name}/domains": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent expertise domains",
            "description": """Get domains where this agent has demonstrated expertise.

Expertise is determined by debate performance in specific topic areas.""",
            "operationId": "getAgentDomains",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Domain expertise",
                    {
                        "domains": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "domain": {"type": "string"},
                                    "score": {"type": "number"},
                                },
                            },
                        }
                    },
                )
            },
        },
    },
    "/api/agent/{name}/accuracy": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent accuracy",
            "description": """Get accuracy metrics measuring how often the agent's positions align with eventual consensus.""",
            "operationId": "getAgentAccuracy",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Accuracy metrics",
                    {"accuracy": {"type": "number"}, "total_predictions": {"type": "integer"}},
                )
            },
        },
    },
    "/api/v1/agent/{agent_id}/head-to-head/{opponent_id}": {
        "get": {
            "tags": ["Agents"],
            "summary": "Head-to-head comparison",
            "description": """Get head-to-head comparison between two agents.

**Response includes:**
- Win/loss/draw record between the two agents
- Per-domain performance comparison
- ELO delta history
- Recent matchup results""",
            "operationId": "getAgentHeadToHead",
            "parameters": [
                {
                    "name": "agent_id",
                    "in": "path",
                    "required": True,
                    "description": "ID of the primary agent",
                    "schema": {"type": "string"},
                },
                {
                    "name": "opponent_id",
                    "in": "path",
                    "required": True,
                    "description": "ID of the opponent agent",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Head-to-head comparison data",
                    {
                        "wins": {"type": "integer"},
                        "losses": {"type": "integer"},
                        "draws": {"type": "integer"},
                        "elo_delta": {"type": "number"},
                    },
                ),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/agent/{agent_id}/opponent-briefing/{opponent_id}": {
        "get": {
            "tags": ["Agents"],
            "summary": "Opponent briefing",
            "description": """Get a strategic briefing about an opponent agent.

**Response includes:**
- Opponent strengths and weaknesses
- Preferred argumentation styles
- Historical vulnerability patterns
- Recommended counter-strategies""",
            "operationId": "getAgentOpponentBriefing",
            "parameters": [
                {
                    "name": "agent_id",
                    "in": "path",
                    "required": True,
                    "description": "ID of the agent requesting the briefing",
                    "schema": {"type": "string"},
                },
                {
                    "name": "opponent_id",
                    "in": "path",
                    "required": True,
                    "description": "ID of the opponent agent",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Opponent briefing data",
                    {
                        "strengths": {"type": "array", "items": {"type": "string"}},
                        "weaknesses": {"type": "array", "items": {"type": "string"}},
                        "strategies": {"type": "array", "items": {"type": "string"}},
                    },
                ),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/agent/compare": {
        "get": {
            "tags": ["Agents"],
            "summary": "Compare agents",
            "description": """Compare two agents side-by-side.

Returns comparative statistics including head-to-head record, relative strengths, and domain performance.""",
            "operationId": "compareAgents",
            "parameters": [
                {
                    "name": "agent_a",
                    "in": "query",
                    "required": True,
                    "description": "First agent name",
                    "schema": {"type": "string"},
                },
                {
                    "name": "agent_b",
                    "in": "query",
                    "required": True,
                    "description": "Second agent name",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Comparison data",
                    {
                        "agent_a": {"type": "object"},
                        "agent_b": {"type": "object"},
                        "comparison": {"type": "object"},
                    },
                )
            },
        },
    },
    "/api/matches/recent": {
        "get": {
            "tags": ["Agents"],
            "summary": "Recent matches",
            "description": """Get recent debate matches across all agents.

Returns match summaries with participants, outcomes, and ELO changes.""",
            "operationId": "getRecentMatches",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of matches to return",
                    "schema": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Recent matches",
                    {
                        "matches": {"type": "array", "items": {"type": "object"}},
                        "total": {"type": "integer"},
                    },
                )
            },
        },
    },
    "/api/calibration/leaderboard": {
        "get": {
            "tags": ["Agents"],
            "summary": "Calibration leaderboard",
            "description": """Get agents ranked by calibration score (how well confidence matches accuracy).

Lower calibration error = better calibrated agent.""",
            "operationId": "getCalibrationLeaderboard",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of agents to return",
                    "schema": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Calibration rankings",
                    {
                        "agents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "calibration_error": {"type": "number"},
                                },
                            },
                        }
                    },
                )
            },
        },
    },
    "/api/personas": {
        "get": {
            "tags": ["Agents"],
            "summary": "List all personas",
            "description": """Get all agent personas including their communication styles and reasoning patterns.""",
            "operationId": "listPersonas",
            "responses": {
                "200": _ok_response(
                    "All personas",
                    {
                        "personas": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "style": {"type": "string"},
                                },
                            },
                        }
                    },
                )
            },
        },
    },
    "/api/v1/agents/health": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent health check",
            "description": """Check health and availability of all configured agents.

**Response includes:**
- Per-agent availability status
- Response times and latency
- Error rates and failure counts
- API key validity indicators""",
            "operationId": "getAgentHealth",
            "responses": {
                "200": _ok_response(
                    "Agent health status",
                    {
                        "agents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["healthy", "degraded", "unavailable"],
                                    },
                                    "latency_ms": {"type": "number"},
                                    "last_check": {"type": "string", "format": "date-time"},
                                },
                            },
                        },
                        "healthy_count": {"type": "integer"},
                        "total_count": {"type": "integer"},
                    },
                ),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/agents/availability": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent availability matrix",
            "description": """Get detailed availability information for all agents.

**Response includes:**
- API key configuration status
- Model availability per provider
- Fallback options via OpenRouter
- Required environment variables""",
            "operationId": "getAgentAvailability",
            "responses": {
                "200": _ok_response(
                    "Agent availability",
                    {
                        "agents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "available": {"type": "boolean"},
                                    "provider": {"type": "string"},
                                    "model": {"type": "string"},
                                    "has_api_key": {"type": "boolean"},
                                    "fallback_available": {"type": "boolean"},
                                },
                            },
                        },
                    },
                ),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/agents/local": {
        "get": {
            "tags": ["Agents"],
            "summary": "List local LLM servers",
            "description": """Detect and list locally running LLM servers.

**Supported local servers:**
- Ollama (default port 11434)
- LM Studio (default port 1234)
- LocalAI (default port 8080)
- vLLM (default port 8000)

**Response includes:**
- Server type and URL
- Available models
- Connection status""",
            "operationId": "listLocalAgents",
            "responses": {
                "200": _ok_response(
                    "Local LLM servers",
                    {
                        "servers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "url": {"type": "string"},
                                    "models": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "status": {"type": "string"},
                                },
                            },
                        },
                    },
                ),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/agents/local/status": {
        "get": {
            "tags": ["Agents"],
            "summary": "Local LLM status",
            "description": """Get status of local LLM servers.

Returns quick health check for all detected local servers.""",
            "operationId": "getLocalAgentStatus",
            "responses": {
                "200": _ok_response(
                    "Local server status",
                    {
                        "available": {"type": "boolean"},
                        "servers": {"type": "integer"},
                        "total_models": {"type": "integer"},
                    },
                ),
            },
            "security": [{"bearerAuth": []}],
        },
    },
}
