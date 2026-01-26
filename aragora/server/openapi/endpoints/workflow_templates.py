"""Workflow Templates API endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

WORKFLOW_TEMPLATES_ENDPOINTS = {
    "/api/workflow/templates": {
        "get": {
            "tags": ["Workflow Templates"],
            "summary": "List workflow templates",
            "operationId": "listWorkflowTemplates",
            "description": "Get list of available workflow templates with optional filtering.",
            "parameters": [
                {
                    "name": "category",
                    "in": "query",
                    "description": "Filter by category",
                    "schema": {"type": "string"},
                },
                {
                    "name": "pattern",
                    "in": "query",
                    "description": "Filter by pattern type",
                    "schema": {
                        "type": "string",
                        "enum": ["hive_mind", "map_reduce", "review_cycle", "pipeline", "parallel"],
                    },
                },
                {
                    "name": "search",
                    "in": "query",
                    "description": "Search templates by name or description",
                    "schema": {"type": "string"},
                },
                {
                    "name": "tags",
                    "in": "query",
                    "description": "Filter by tags (comma-separated)",
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50, "maximum": 100},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "schema": {"type": "integer", "default": 0},
                },
            ],
            "responses": {
                "200": {
                    "description": "List of workflow templates",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "templates": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/WorkflowTemplate"},
                                    },
                                    "total": {"type": "integer"},
                                    "limit": {"type": "integer"},
                                    "offset": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
    "/api/workflow/templates/{template_id}": {
        "get": {
            "tags": ["Workflow Templates"],
            "summary": "Get template details",
            "operationId": "getWorkflowTemplate",
            "description": "Get detailed information about a specific workflow template.",
            "parameters": [
                {
                    "name": "template_id",
                    "in": "path",
                    "required": True,
                    "description": "Template ID (category/name format)",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Workflow template details", "WorkflowTemplate"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/workflow/templates/{template_id}/package": {
        "get": {
            "tags": ["Workflow Templates"],
            "summary": "Get template package",
            "operationId": "getWorkflowTemplatesPackage",
            "description": "Get the full template package including workflow definition, metadata, and documentation.",
            "parameters": [
                {
                    "name": "template_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "include_examples",
                    "in": "query",
                    "description": "Include usage examples",
                    "schema": {"type": "boolean", "default": True},
                },
            ],
            "responses": {
                "200": {
                    "description": "Template package",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "category": {"type": "string"},
                                    "pattern": {"type": "string"},
                                    "workflow_definition": {"type": "object"},
                                    "input_schema": {"type": "object"},
                                    "output_schema": {"type": "object"},
                                    "documentation": {"type": "string"},
                                    "examples": {
                                        "type": "array",
                                        "items": {"type": "object"},
                                    },
                                    "author": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "email": {"type": "string"},
                                        },
                                    },
                                    "version": {"type": "string"},
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
    },
    "/api/workflow/templates/{template_id}/run": {
        "post": {
            "tags": ["Workflow Templates"],
            "summary": "Run workflow template",
            "operationId": "createWorkflowTemplatesRun",
            "description": "Execute a workflow template with the provided inputs.",
            "parameters": [
                {
                    "name": "template_id",
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
                                "inputs": {
                                    "type": "object",
                                    "description": "Template input values",
                                },
                                "config": {
                                    "type": "object",
                                    "description": "Execution configuration overrides",
                                    "properties": {
                                        "timeout": {"type": "integer"},
                                        "priority": {
                                            "type": "string",
                                            "enum": ["low", "normal", "high"],
                                        },
                                        "async": {"type": "boolean", "default": False},
                                    },
                                },
                                "workspace_id": {
                                    "type": "string",
                                    "description": "Workspace to run in",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "Workflow execution result (sync)",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "execution_id": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "running", "completed", "failed"],
                                    },
                                    "result": {"type": "object"},
                                    "duration_ms": {"type": "number"},
                                },
                            },
                        },
                    },
                },
                "202": {
                    "description": "Workflow execution started (async)",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "execution_id": {"type": "string"},
                                    "status": {"type": "string"},
                                    "status_url": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/workflow/categories": {
        "get": {
            "tags": ["Workflow Templates"],
            "summary": "List template categories",
            "operationId": "listWorkflowCategories",
            "description": "Get list of available template categories with counts.",
            "responses": {
                "200": {
                    "description": "List of categories",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "categories": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "description": {"type": "string"},
                                                "template_count": {"type": "integer"},
                                                "icon": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
    "/api/workflow/patterns": {
        "get": {
            "tags": ["Workflow Templates"],
            "summary": "List workflow patterns",
            "operationId": "listWorkflowPatterns",
            "description": "Get list of available workflow patterns.",
            "responses": {
                "200": {
                    "description": "List of patterns",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "patterns": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "description": {"type": "string"},
                                                "available": {"type": "boolean"},
                                                "use_cases": {
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
            },
        },
    },
    "/api/workflow/patterns/{pattern_id}/instantiate": {
        "post": {
            "tags": ["Workflow Templates"],
            "summary": "Instantiate pattern as template",
            "operationId": "createWorkflowPatternsInstantiate",
            "description": "Create a new template instance from a workflow pattern.",
            "parameters": [
                {
                    "name": "pattern_id",
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
                            "required": ["name", "description"],
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "category": {"type": "string"},
                                "config": {
                                    "type": "object",
                                    "description": "Pattern-specific configuration",
                                },
                                "agents": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Agents to use in the workflow",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _ok_response("Created template instance", "WorkflowTemplate"),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
}
