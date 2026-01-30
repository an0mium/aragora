"""Additional endpoint definitions (tournaments, genesis, evolution, etc.)."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

ADDITIONAL_ENDPOINTS = {
    "/api/tournaments": {
        "get": {
            "tags": ["Tournaments"],
            "summary": "List tournaments",
            "description": "Get list of all agent tournaments and their status.",
            "operationId": "listTournaments",
            "responses": {"200": _ok_response("Tournament list")},
        },
    },
    "/api/tournaments/{id}/standings": {
        "get": {
            "tags": ["Tournaments"],
            "summary": "Tournament standings",
            "description": "Get current standings and rankings for a tournament.",
            "operationId": "getTournamentsStanding",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Standings")},
        },
    },
    "/api/genesis/stats": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Genesis statistics",
            "description": "Get statistics about agent genesis events and population.",
            "operationId": "listGenesisStats",
            "responses": {"200": _ok_response("Genesis stats")},
        },
    },
    "/api/genesis/events": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Genesis events",
            "description": "Get timeline of agent creation and evolution events.",
            "operationId": "listGenesisEvents",
            "responses": {"200": _ok_response("Genesis events")},
        },
    },
    "/api/genesis/lineage/{agent}": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Agent lineage",
            "description": "Get lineage tree showing agent ancestry and descendants.",
            "operationId": "getGenesisLineage",
            "parameters": [
                {"name": "agent", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Lineage data")},
        },
    },
    "/api/genesis/tree/{agent}": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Agent tree",
            "description": "Get hierarchical tree view of agent relationships.",
            "operationId": "getGenesisTree",
            "parameters": [
                {"name": "agent", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Tree data")},
        },
    },
    "/api/v1/tournaments/{tournament_id}": {
        "get": {
            "tags": ["Tournaments"],
            "summary": "Get tournament",
            "description": "Get details of a specific tournament by ID, including configuration, participants, and current status.",
            "operationId": "getTournament",
            "parameters": [
                {
                    "name": "tournament_id",
                    "in": "path",
                    "required": True,
                    "description": "Tournament ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Tournament details"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Tournaments"],
            "summary": "Delete tournament",
            "description": "Delete a specific tournament and its associated data.",
            "operationId": "deleteTournament",
            "parameters": [
                {
                    "name": "tournament_id",
                    "in": "path",
                    "required": True,
                    "description": "Tournament ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Tournament deleted"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/genesis/descendants/{genome_id}": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Get genome descendants",
            "description": """Get all descendant genomes derived from a specific genome.

**Response includes:**
- Direct child genomes
- Full descendant tree with depth levels
- Mutation history for each descendant
- Performance comparison across generations""",
            "operationId": "getGenesisDescendants",
            "parameters": [
                {
                    "name": "genome_id",
                    "in": "path",
                    "required": True,
                    "description": "ID of the parent genome",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Descendant genomes"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/genesis/genomes/{genome_id}": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Get genome",
            "description": """Get details of a specific genome by ID.

**Response includes:**
- Genome configuration and parameters
- Parent genome reference
- Mutation history
- Performance metrics and fitness score""",
            "operationId": "getGenesisGenome",
            "parameters": [
                {
                    "name": "genome_id",
                    "in": "path",
                    "required": True,
                    "description": "Genome ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Genome details"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/reviews/{review_id}": {
        "get": {
            "tags": ["Reviews"],
            "summary": "Get review",
            "description": "Get a specific code review by ID, including findings, agent critiques, and agreement scores.",
            "operationId": "getReview",
            "parameters": [
                {
                    "name": "review_id",
                    "in": "path",
                    "required": True,
                    "description": "Review ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Review details"),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "put": {
            "tags": ["Reviews"],
            "summary": "Update review",
            "description": "Update a specific code review, such as marking findings as resolved or adding annotations.",
            "operationId": "updateReview",
            "parameters": [
                {
                    "name": "review_id",
                    "in": "path",
                    "required": True,
                    "description": "Review ID",
                    "schema": {"type": "string"},
                }
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
                                    "enum": ["open", "resolved", "dismissed"],
                                    "description": "Updated review status",
                                },
                                "annotations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Additional annotations",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _ok_response("Review updated"),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Reviews"],
            "summary": "Delete review",
            "description": "Delete a specific code review and its associated data.",
            "operationId": "deleteReview",
            "parameters": [
                {
                    "name": "review_id",
                    "in": "path",
                    "required": True,
                    "description": "Review ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Review deleted"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/evolution/{agent}/history": {
        "get": {
            "tags": ["Evolution"],
            "summary": "Agent evolution history",
            "description": "Get history of agent evolution including version changes and improvements.",
            "operationId": "getEvolutionHistory",
            "parameters": [
                {"name": "agent", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Evolution history")},
        },
    },
    "/api/replays": {
        "get": {
            "tags": ["Replays"],
            "summary": "List replays",
            "description": "Get list of available debate replays.",
            "operationId": "listReplays",
            "responses": {"200": _ok_response("Replay list")},
        },
    },
    "/api/replays/{id}": {
        "get": {
            "tags": ["Replays"],
            "summary": "Get replay",
            "description": "Get detailed replay data for a specific debate.",
            "operationId": "getReplay",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Replay data")},
        },
    },
    "/api/learning/evolution": {
        "get": {
            "tags": ["Learning"],
            "summary": "Learning evolution",
            "description": "Get learning evolution data showing system improvements over time.",
            "operationId": "listLearningEvolution",
            "responses": {"200": _ok_response("Evolution data")},
        },
    },
    "/api/meta-learning/stats": {
        "get": {
            "tags": ["Learning"],
            "summary": "Meta-learning statistics",
            "description": "Get statistics from meta-learning processes.",
            "operationId": "listMetaLearningStats",
            "responses": {"200": _ok_response("Meta-learning stats")},
        },
    },
    "/api/critiques/patterns": {
        "get": {
            "tags": ["Critiques"],
            "summary": "Critique patterns",
            "description": "Get common critique patterns across debates.",
            "operationId": "listCritiquesPatterns",
            "responses": {"200": _ok_response("Patterns")},
        },
    },
    "/api/critiques/archive": {
        "get": {
            "tags": ["Critiques"],
            "summary": "Critique archive",
            "description": "Get archived critiques for historical analysis.",
            "operationId": "listCritiquesArchive",
            "responses": {"200": _ok_response("Archive")},
        },
    },
    "/api/reputation/all": {
        "get": {
            "tags": ["Critiques"],
            "summary": "All reputations",
            "description": "Get reputation data for all agents.",
            "operationId": "listReputationAll",
            "responses": {"200": _ok_response("Reputations")},
        },
    },
    "/api/routing/best-teams": {
        "get": {
            "tags": ["Routing"],
            "summary": "Best team combinations",
            "description": "Get recommended team combinations based on historical performance.",
            "operationId": "listRoutingBestTeams",
            "parameters": [
                {"name": "min_debates", "in": "query", "schema": {"type": "integer", "default": 3}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
            ],
            "responses": {"200": _ok_response("Best teams")},
        },
    },
    "/api/routing/recommendations": {
        "post": {
            "tags": ["Routing"],
            "summary": "Agent recommendations",
            "description": "Get agent recommendations based on domain requirements and traits.",
            "operationId": "createRoutingRecommendations",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "primary_domain": {"type": "string"},
                                "secondary_domains": {"type": "array", "items": {"type": "string"}},
                                "required_traits": {"type": "array", "items": {"type": "string"}},
                            },
                        }
                    }
                }
            },
            "responses": {"200": _ok_response("Recommendations")},
        },
    },
    "/api/introspection/all": {
        "get": {
            "tags": ["Introspection"],
            "summary": "All introspection data",
            "description": "Get comprehensive introspection data across all agents.",
            "operationId": "listIntrospectionAll",
            "responses": {"200": _ok_response("Introspection data")},
        },
    },
    "/api/introspection/leaderboard": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Introspection leaderboard",
            "description": "Get leaderboard rankings based on introspection metrics.",
            "operationId": "listIntrospectionLeaderboard",
            "responses": {"200": _ok_response("Leaderboard")},
        },
    },
    "/api/introspection/agents": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Agent introspection list",
            "description": "Get list of agents with introspection data available.",
            "operationId": "listIntrospectionAgents",
            "responses": {"200": _ok_response("Agent list")},
        },
    },
    "/api/introspection/agents/{name}": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Agent introspection",
            "description": "Get detailed introspection data for a specific agent.",
            "operationId": "getIntrospectionAgent",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Agent introspection")},
        },
    },
    # =========================================================================
    # Uncertainty Quantification
    # =========================================================================
    "/api/v1/uncertainty/agent/{agent_id}": {
        "get": {
            "tags": ["Uncertainty"],
            "summary": "Get agent uncertainty metrics",
            "operationId": "getAgentUncertainty",
            "description": "Get uncertainty quantification metrics for a specific agent, including calibration scores, confidence distributions, and epistemic vs aleatoric uncertainty breakdowns.",
            "parameters": [
                {
                    "name": "agent_id",
                    "in": "path",
                    "required": True,
                    "description": "Agent ID to retrieve uncertainty metrics for",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Agent uncertainty metrics",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "agent_id": {"type": "string"},
                                    "calibration_score": {"type": "number", "description": "How well-calibrated the agent's confidence is (0-1)"},
                                    "mean_confidence": {"type": "number"},
                                    "epistemic_uncertainty": {"type": "number", "description": "Uncertainty due to lack of knowledge"},
                                    "aleatoric_uncertainty": {"type": "number", "description": "Irreducible uncertainty in data"},
                                    "confidence_distribution": {
                                        "type": "object",
                                        "additionalProperties": {"type": "number"},
                                    },
                                    "total_predictions": {"type": "integer"},
                                    "computed_at": {"type": "string", "format": "date-time"},
                                },
                            },
                        },
                    },
                },
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/uncertainty/debate/{debate_id}": {
        "get": {
            "tags": ["Uncertainty"],
            "summary": "Get debate uncertainty metrics",
            "operationId": "getDebateUncertainty",
            "description": "Get uncertainty quantification metrics for a specific debate, including aggregate confidence levels, disagreement measures, and per-agent uncertainty breakdowns.",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "description": "Debate ID to retrieve uncertainty metrics for",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Debate uncertainty metrics",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "debate_id": {"type": "string"},
                                    "aggregate_confidence": {"type": "number"},
                                    "disagreement_score": {"type": "number", "description": "Degree of disagreement between agents (0-1)"},
                                    "convergence_rate": {"type": "number", "description": "Rate at which agents converge on consensus"},
                                    "per_agent_uncertainty": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "agent_id": {"type": "string"},
                                                "confidence": {"type": "number"},
                                                "uncertainty": {"type": "number"},
                                            },
                                        },
                                    },
                                    "rounds_analyzed": {"type": "integer"},
                                    "computed_at": {"type": "string", "format": "date-time"},
                                },
                            },
                        },
                    },
                },
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # Verticals
    # =========================================================================
    "/api/verticals/{vertical_id}": {
        "get": {
            "tags": ["Verticals"],
            "summary": "Get vertical configuration",
            "operationId": "getVertical",
            "description": "Get configuration details for a specific industry vertical, including domain settings, agent preferences, and compliance requirements.",
            "parameters": [
                {
                    "name": "vertical_id",
                    "in": "path",
                    "required": True,
                    "description": "Vertical configuration ID",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Vertical configuration",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "domain": {"type": "string"},
                                    "agent_preferences": {
                                        "type": "object",
                                        "description": "Preferred agent configuration for this vertical",
                                    },
                                    "compliance_requirements": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "enabled": {"type": "boolean"},
                                    "created_at": {"type": "string", "format": "date-time"},
                                    "updated_at": {"type": "string", "format": "date-time"},
                                },
                            },
                        },
                    },
                },
                "404": STANDARD_ERRORS["404"],
            },
        },
        "put": {
            "tags": ["Verticals"],
            "summary": "Update vertical configuration",
            "operationId": "updateVertical",
            "description": "Update the configuration for a specific industry vertical.",
            "parameters": [
                {
                    "name": "vertical_id",
                    "in": "path",
                    "required": True,
                    "description": "Vertical configuration ID",
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
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "domain": {"type": "string"},
                                "agent_preferences": {"type": "object"},
                                "compliance_requirements": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "enabled": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _ok_response("Vertical configuration updated"),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # Audio
    # =========================================================================
    "/audio/{audio_id}": {
        "get": {
            "tags": ["Audio"],
            "summary": "Get audio file",
            "operationId": "getAudioFile",
            "description": "Retrieve an audio file by ID. Returns the audio binary data with appropriate content type headers.",
            "parameters": [
                {
                    "name": "audio_id",
                    "in": "path",
                    "required": True,
                    "description": "Audio file ID",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Audio file binary data",
                    "content": {
                        "audio/mpeg": {
                            "schema": {"type": "string", "format": "binary"},
                        },
                        "audio/wav": {
                            "schema": {"type": "string", "format": "binary"},
                        },
                        "audio/ogg": {
                            "schema": {"type": "string", "format": "binary"},
                        },
                    },
                },
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
}
