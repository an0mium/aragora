"""Explainability API endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

EXPLAINABILITY_ENDPOINTS = {
    "/api/debates/{debate_id}/explainability": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get decision explanation",
            "operationId": "getDebatesExplainability",
            "description": "Get a full explanation of how the debate decision was reached, including narrative, factors, counterfactuals, and provenance.",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "description": "ID of the debate to explain",
                    "schema": {"type": "string"},
                },
                {
                    "name": "include_factors",
                    "in": "query",
                    "description": "Include factor decomposition",
                    "schema": {"type": "boolean", "default": True},
                },
                {
                    "name": "include_counterfactuals",
                    "in": "query",
                    "description": "Include counterfactual scenarios",
                    "schema": {"type": "boolean", "default": True},
                },
                {
                    "name": "include_provenance",
                    "in": "query",
                    "description": "Include decision provenance chain",
                    "schema": {"type": "boolean", "default": True},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Full decision explanation",
                    "DecisionExplanation",
                ),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/debates/{debate_id}/explainability/factors": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get contributing factors",
            "operationId": "getDebatesExplainabilityFactor",
            "description": "Get the factors that contributed to the debate decision with their relative contributions.",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "min_contribution",
                    "in": "query",
                    "description": "Minimum contribution threshold (0-1)",
                    "schema": {"type": "number", "minimum": 0, "maximum": 1},
                },
                {
                    "name": "sort_by",
                    "in": "query",
                    "description": "Sort factors by",
                    "schema": {
                        "type": "string",
                        "enum": ["contribution", "name", "type"],
                        "default": "contribution",
                    },
                },
            ],
            "responses": {
                "200": {
                    "description": "List of contributing factors",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "debate_id": {"type": "string"},
                                    "factors": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "contribution": {"type": "number"},
                                                "description": {"type": "string"},
                                                "type": {"type": "string"},
                                                "evidence": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                        },
                                    },
                                    "total_factors": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/debates/{debate_id}/explainability/counterfactual": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get counterfactual scenarios",
            "operationId": "getDebatesExplainabilityCounterfactual",
            "description": "Generate what-if scenarios showing how different inputs might have changed the outcome.",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "max_scenarios",
                    "in": "query",
                    "description": "Maximum number of scenarios to generate",
                    "schema": {"type": "integer", "default": 5, "maximum": 20},
                },
                {
                    "name": "min_probability",
                    "in": "query",
                    "description": "Minimum probability threshold for scenarios",
                    "schema": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.3},
                },
            ],
            "responses": {
                "200": {
                    "description": "Counterfactual scenarios",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "debate_id": {"type": "string"},
                                    "counterfactuals": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "scenario": {"type": "string"},
                                                "outcome": {"type": "string"},
                                                "probability": {"type": "number"},
                                                "affected_factors": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Explainability"],
            "summary": "Generate custom counterfactual",
            "operationId": "createDebatesExplainabilityCounterfactual",
            "description": "Generate a counterfactual scenario based on custom hypothetical changes.",
            "parameters": [
                {
                    "name": "debate_id",
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
                            "required": ["hypothesis"],
                            "properties": {
                                "hypothesis": {
                                    "type": "string",
                                    "description": "The hypothetical change to evaluate",
                                },
                                "affected_agents": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Agents affected by the change",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _ok_response("Custom counterfactual analysis"),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/debates/{debate_id}/explainability/provenance": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get decision provenance",
            "operationId": "getDebatesExplainabilityProvenance",
            "description": "Get the provenance chain showing how the decision was reached step by step.",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "include_timestamps",
                    "in": "query",
                    "description": "Include timestamps for each step",
                    "schema": {"type": "boolean", "default": True},
                },
                {
                    "name": "include_agents",
                    "in": "query",
                    "description": "Include agent information for each step",
                    "schema": {"type": "boolean", "default": True},
                },
                {
                    "name": "include_confidence",
                    "in": "query",
                    "description": "Include confidence levels for each step",
                    "schema": {"type": "boolean", "default": True},
                },
            ],
            "responses": {
                "200": {
                    "description": "Decision provenance chain",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "debate_id": {"type": "string"},
                                    "provenance": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "step": {"type": "integer"},
                                                "action": {"type": "string"},
                                                "timestamp": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                                "agent": {"type": "string"},
                                                "confidence": {"type": "number"},
                                                "evidence": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                        },
                                    },
                                    "total_steps": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/debates/{debate_id}/explainability/narrative": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get decision narrative",
            "operationId": "getDebatesExplainabilityNarrative",
            "description": "Get a natural language explanation of the decision.",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "format",
                    "in": "query",
                    "description": "Narrative format",
                    "schema": {
                        "type": "string",
                        "enum": ["brief", "detailed", "executive_summary"],
                        "default": "detailed",
                    },
                },
                {
                    "name": "language",
                    "in": "query",
                    "description": "Output language (ISO 639-1)",
                    "schema": {"type": "string", "default": "en"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Decision narrative",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "debate_id": {"type": "string"},
                                    "narrative": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "format": {"type": "string"},
                                    "word_count": {"type": "integer"},
                                    "generated_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                },
                            },
                        },
                    },
                },
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
}
