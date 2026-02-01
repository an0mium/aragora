"""Knowledge Base endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS

KNOWLEDGE_BASE_ENDPOINTS = {
    "/api/v1/knowledge/query": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "Query knowledge base",
            "operationId": "queryKnowledgeBase",
            "description": "Run a natural-language query against the knowledge base.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "workspace_id": {"type": "string"},
                                "options": {"type": "object"},
                            },
                            "required": ["question"],
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Knowledge query response", "KnowledgeQueryResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/search": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "Search knowledge",
            "operationId": "searchKnowledgeBase",
            "description": "Search knowledge chunks via embeddings.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "q", "in": "query", "schema": {"type": "string"}},
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
            ],
            "responses": {
                "200": _ok_response("Knowledge search response", "KnowledgeSearchResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/stats": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "Knowledge stats",
            "operationId": "getKnowledgeStats",
            "description": "Get knowledge base statistics.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Knowledge stats", "KnowledgeStats"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/facts": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "List facts",
            "operationId": "listKnowledgeFacts",
            "description": "List knowledge facts with optional filtering.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {"name": "topic", "in": "query", "schema": {"type": "string"}},
                {"name": "min_confidence", "in": "query", "schema": {"type": "number"}},
                {"name": "status", "in": "query", "schema": {"type": "string"}},
                {"name": "include_superseded", "in": "query", "schema": {"type": "boolean"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}},
                {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
            ],
            "responses": {
                "200": _ok_response("Knowledge facts", "KnowledgeFactList"),
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Knowledge"],
            "summary": "Create fact",
            "operationId": "createKnowledgeFact",
            "description": "Create a new knowledge fact.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "statement": {"type": "string"},
                                "workspace_id": {"type": "string"},
                                "confidence": {"type": "number"},
                                "topics": {"type": "array", "items": {"type": "string"}},
                                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                                "source_documents": {"type": "array", "items": {"type": "string"}},
                                "metadata": {"type": "object"},
                            },
                            "required": ["statement"],
                        }
                    }
                },
            },
            "responses": {
                "201": _ok_response("Created fact", "KnowledgeFact"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/knowledge/facts/{fact_id}": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "Get fact",
            "description": """Retrieve a specific knowledge fact by ID.

**Response includes:**
- Fact statement and confidence score
- Associated topics and evidence
- Creation and update timestamps
- Supersession history (if applicable)""",
            "operationId": "getKnowledgeFact",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "fact_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Knowledge fact", "KnowledgeFact"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "put": {
            "tags": ["Knowledge"],
            "summary": "Update fact",
            "description": """Update an existing knowledge fact.

**Updatable fields:**
- statement: The fact text
- confidence: Confidence score (0.0-1.0)
- topics: Associated topic tags
- metadata: Custom metadata

**Note:** Updates create an audit trail. Consider superseding instead of updating for significant changes.""",
            "operationId": "updateKnowledgeFact",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "fact_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Updated fact content",
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence score (0.0-1.0)",
                                },
                                "topics": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Associated topic tags",
                                },
                                "metadata": {"type": "object", "description": "Custom metadata"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Updated fact", "KnowledgeFact"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Knowledge"],
            "summary": "Delete fact",
            "description": """Delete a knowledge fact from the knowledge base.

**Behavior:**
- Soft delete by default (fact marked as deleted)
- Related evidence links are preserved for audit
- Supersession chains are maintained

**Note:** Deleted facts can be restored by admin if needed.""",
            "operationId": "deleteKnowledgeFact",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "fact_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Deleted fact"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/knowledge/facts/{fact_id}/verify": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "Verify fact",
            "description": """Trigger verification of a knowledge fact against evidence.

**Verification process:**
- Cross-references fact against source documents
- Checks for conflicting facts in the knowledge base
- Updates confidence score based on evidence strength

**Async:** Returns immediately; verification runs in background.""",
            "operationId": "verifyKnowledgeFact",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "fact_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Fact verification started"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/facts/{fact_id}/contradictions": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "List contradictions",
            "description": """Find facts that contradict the specified fact.

**Detection methods:**
- Semantic similarity with opposite polarity
- Explicit contradiction links
- Temporal inconsistencies

**Use cases:**
- Identify knowledge conflicts
- Resolve contradictory information
- Audit knowledge quality""",
            "operationId": "listKnowledgeContradictions",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "fact_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Contradicting facts", "KnowledgeFactList"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/facts/{fact_id}/relations": {
        "get": {
            "tags": ["Knowledge"],
            "summary": "List relations",
            "description": """Get all facts related to the specified fact.

**Relation types:**
- supports: Evidence supporting this fact
- contradicts: Conflicting facts
- supersedes: Facts that replace this one
- derived_from: Source facts used to derive this one""",
            "operationId": "listKnowledgeRelations",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "fact_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Fact relations", "KnowledgeQueryResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Knowledge"],
            "summary": "Add relation",
            "description": """Create a relation between this fact and another fact.

**Required fields:**
- target_fact_id: ID of the related fact
- relation_type: Type of relation (supports, contradicts, etc.)

**Optional fields:**
- confidence: Confidence in the relation (0.0-1.0)
- evidence: Supporting evidence for the relation""",
            "operationId": "addKnowledgeRelation",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "fact_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "target_fact_id": {
                                    "type": "string",
                                    "description": "ID of the related fact",
                                },
                                "relation_type": {
                                    "type": "string",
                                    "enum": [
                                        "supports",
                                        "contradicts",
                                        "supersedes",
                                        "derived_from",
                                    ],
                                    "description": "Type of relation",
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence in the relation",
                                },
                                "evidence": {
                                    "type": "string",
                                    "description": "Supporting evidence",
                                },
                            },
                            "required": ["target_fact_id", "relation_type"],
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Relation added"),
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/knowledge/facts/relations": {
        "post": {
            "tags": ["Knowledge"],
            "summary": "Add relation between facts",
            "description": """Create a relation between two facts by specifying both IDs.

**Required fields:**
- source_fact_id: ID of the source fact
- target_fact_id: ID of the target fact
- relation_type: Type of relation

**Alternative to:** POST /api/v1/knowledge/facts/{fact_id}/relations""",
            "operationId": "addKnowledgeRelationBetweenFacts",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "source_fact_id": {
                                    "type": "string",
                                    "description": "ID of the source fact",
                                },
                                "target_fact_id": {
                                    "type": "string",
                                    "description": "ID of the target fact",
                                },
                                "relation_type": {
                                    "type": "string",
                                    "enum": [
                                        "supports",
                                        "contradicts",
                                        "supersedes",
                                        "derived_from",
                                    ],
                                    "description": "Type of relation",
                                },
                            },
                            "required": ["source_fact_id", "target_fact_id", "relation_type"],
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Relation added"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}
