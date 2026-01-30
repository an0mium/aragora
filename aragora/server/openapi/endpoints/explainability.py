"""Explainability API endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

EXPLAINABILITY_ENDPOINTS = {
    "/api/v1/debates/{debate_id}/explanation": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get decision explanation",
            "operationId": "getDebateExplanation",
            "description": "Get a full explanation of how the debate decision was reached.",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "description": "ID of the debate to explain",
                    "schema": {"type": "string"},
                },
                {
                    "name": "format",
                    "in": "query",
                    "description": "Response format: json or summary",
                    "schema": {"type": "string", "default": "json"},
                },
            ],
            "responses": {
                "200": _ok_response("Decision explanation", "DecisionExplanation"),
                "404": STANDARD_ERRORS["404"],
            },
        }
    },
    "/api/v1/debates/{debate_id}/evidence": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get evidence chain",
            "operationId": "getDebateEvidence",
            "description": "Get evidence chain for a debate.",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 20},
                },
                {
                    "name": "min_relevance",
                    "in": "query",
                    "schema": {"type": "number", "default": 0.0},
                },
            ],
            "responses": {
                "200": _ok_response("Evidence chain", "EvidenceChain"),
                "404": STANDARD_ERRORS["404"],
            },
        }
    },
    "/api/v1/debates/{debate_id}/votes/pivots": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get vote pivots",
            "operationId": "getDebateVotePivots",
            "description": "Get vote influence pivots for a debate.",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "min_influence",
                    "in": "query",
                    "schema": {"type": "number", "default": 0.0},
                },
            ],
            "responses": {
                "200": _ok_response("Vote pivots", "VotePivots"),
                "404": STANDARD_ERRORS["404"],
            },
        }
    },
    "/api/v1/debates/{debate_id}/counterfactuals": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get counterfactuals",
            "operationId": "getDebateCounterfactuals",
            "description": "Get counterfactual scenarios for a debate.",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "min_sensitivity",
                    "in": "query",
                    "schema": {"type": "number", "default": 0.0},
                },
            ],
            "responses": {
                "200": _ok_response("Counterfactuals", "Counterfactuals"),
                "404": STANDARD_ERRORS["404"],
            },
        }
    },
    "/api/v1/debates/{debate_id}/summary": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get debate summary",
            "operationId": "getDebateSummary",
            "description": "Get a human-readable summary for a debate.",
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
                    "schema": {"type": "string", "default": "markdown"},
                },
            ],
            "responses": {
                "200": _ok_response("Summary", "DebateSummary"),
                "404": STANDARD_ERRORS["404"],
            },
        }
    },
    "/api/v1/explain/{decision_id}": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get decision explanation",
            "operationId": "getDecisionExplanation",
            "description": """Get a detailed explanation for a specific decision.

**Response includes:**
- Natural language summary of the decision rationale
- Key contributing factors with weights
- Agent positions and voting breakdown
- Evidence citations used in reaching the decision
- Confidence assessment and uncertainty indicators""",
            "parameters": [
                {
                    "name": "decision_id",
                    "in": "path",
                    "required": True,
                    "description": "ID of the decision to explain",
                    "schema": {"type": "string"},
                },
                {
                    "name": "format",
                    "in": "query",
                    "description": "Response format: json, markdown, or summary",
                    "schema": {
                        "type": "string",
                        "default": "json",
                        "enum": ["json", "markdown", "summary"],
                    },
                },
                {
                    "name": "include_factors",
                    "in": "query",
                    "description": "Include detailed factor decomposition",
                    "schema": {"type": "boolean", "default": True},
                },
            ],
            "responses": {
                "200": _ok_response("Decision explanation", "DecisionExplanation"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/explainability/batch": {
        "post": {
            "tags": ["Explainability"],
            "summary": "Create explainability batch",
            "operationId": "createExplainabilityBatch",
            "description": "Generate explainability output for multiple debates.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_ids": {"type": "array", "items": {"type": "string"}},
                                "include_evidence": {"type": "boolean", "default": True},
                                "include_counterfactuals": {"type": "boolean", "default": False},
                            },
                            "required": ["debate_ids"],
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Batch created", "ExplainabilityBatch"),
                "404": STANDARD_ERRORS["404"],
            },
        }
    },
    "/api/v1/explainability/batch/{batch_id}/status": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get explainability batch status",
            "description": """Check the processing status of an explainability batch job.

**Status values:**
- pending: Batch queued for processing
- processing: Batch is being generated
- completed: All explanations ready
- failed: Batch processing failed

**Use cases:**
- Poll for batch completion
- Monitor processing progress
- Handle async explanation generation""",
            "operationId": "getExplainabilityBatchStatus",
            "parameters": [
                {
                    "name": "batch_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Batch status", "ExplainabilityBatchStatus"),
                "404": STANDARD_ERRORS["404"],
            },
        }
    },
    "/api/v1/explainability/batch/{batch_id}/results": {
        "get": {
            "tags": ["Explainability"],
            "summary": "Get explainability batch results",
            "description": """Retrieve the generated explanations for a completed batch.

**Response includes:**
- Individual explanation for each debate in the batch
- Evidence chains (if requested during batch creation)
- Counterfactual scenarios (if requested)

**Note:** Only available after batch status is 'completed'.""",
            "operationId": "getExplainabilityBatchResults",
            "parameters": [
                {
                    "name": "batch_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Batch results", "ExplainabilityBatchResults"),
                "404": STANDARD_ERRORS["404"],
            },
        }
    },
}
