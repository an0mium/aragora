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
            "operationId": "updateKnowledgeFact",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "fact_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
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
            "operationId": "addKnowledgeRelation",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {"name": "fact_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
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
            "operationId": "addKnowledgeRelationBetweenFacts",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Relation added"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}
