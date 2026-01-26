"""Workflow endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

WORKFLOW_ENDPOINTS = {
    "/api/workflows": {
        "get": {
            "tags": ["Workflows"],
            "summary": "List workflows",
            "operationId": "listWorkflows",
            "description": "Get list of workflows with optional filtering by category, tags, or search term.",
            "parameters": [
                {
                    "name": "category",
                    "in": "query",
                    "description": "Filter by workflow category",
                    "schema": {
                        "type": "string",
                        "enum": ["debate", "analysis", "integration", "custom"],
                    },
                },
                {
                    "name": "tags",
                    "in": "query",
                    "description": "Filter by tags (comma-separated)",
                    "schema": {"type": "string"},
                },
                {
                    "name": "search",
                    "in": "query",
                    "description": "Search in workflow name and description",
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50, "maximum": 200},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "schema": {"type": "integer", "default": 0},
                },
            ],
            "responses": {
                "200": _ok_response("List of workflows with pagination", "WorkflowList"),
            },
        },
        "post": {
            "tags": ["Workflows"],
            "summary": "Create workflow",
            "operationId": "createWorkflows",
            "description": "Create a new workflow definition with steps and transitions.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name", "steps"],
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "category": {
                                    "type": "string",
                                    "enum": ["debate", "analysis", "integration", "custom"],
                                },
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "steps": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/StepDefinition"},
                                },
                                "transitions": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/TransitionRule"},
                                },
                                "input_schema": {"type": "object"},
                                "output_schema": {"type": "object"},
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _ok_response("Workflow created", "Workflow"),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/workflows/{workflow_id}": {
        "get": {
            "tags": ["Workflows"],
            "summary": "Get workflow",
            "operationId": "getWorkflow",
            "description": "Get detailed workflow definition by ID.",
            "parameters": [
                {
                    "name": "workflow_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Workflow definition", "Workflow"),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "put": {
            "tags": ["Workflows"],
            "summary": "Update workflow",
            "operationId": "updateWorkflow",
            "description": "Update an existing workflow definition. Creates a new version.",
            "parameters": [
                {
                    "name": "workflow_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/WorkflowUpdate"},
                    },
                },
            },
            "responses": {
                "200": _ok_response("Workflow updated", "Workflow"),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
        "delete": {
            "tags": ["Workflows"],
            "summary": "Delete workflow",
            "operationId": "deleteWorkflow",
            "description": "Delete a workflow definition.",
            "parameters": [
                {
                    "name": "workflow_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Workflow deleted"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/workflows/{workflow_id}/execute": {
        "post": {
            "tags": ["Workflows"],
            "summary": "Execute workflow",
            "operationId": "createWorkflowsExecute",
            "description": "Start execution of a workflow with provided inputs.",
            "parameters": [
                {
                    "name": "workflow_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "inputs": {
                                    "type": "object",
                                    "description": "Input parameters for the workflow",
                                },
                                "async": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Run asynchronously and return execution ID",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _ok_response("Workflow execution result or execution ID"),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/workflows/{workflow_id}/versions": {
        "get": {
            "tags": ["Workflows"],
            "summary": "Get workflow versions",
            "operationId": "getWorkflowsVersion",
            "description": "Get version history of a workflow.",
            "parameters": [
                {
                    "name": "workflow_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 20},
                },
            ],
            "responses": {
                "200": _ok_response("List of workflow versions"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/workflow-templates": {
        "get": {
            "tags": ["Workflows"],
            "summary": "List workflow templates",
            "operationId": "listWorkflowTemplates",
            "description": "Get gallery of workflow templates for quick start.",
            "parameters": [
                {
                    "name": "category",
                    "in": "query",
                    "description": "Filter templates by category",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("List of workflow templates", "WorkflowTemplateList"),
            },
        },
    },
    "/api/workflow-templates/{template_id}": {
        "get": {
            "tags": ["Workflows"],
            "summary": "Get workflow template",
            "operationId": "getWorkflowTemplate",
            "description": "Get a specific workflow template for use as starting point.",
            "parameters": [
                {
                    "name": "template_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Workflow template", "WorkflowTemplate"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/workflow-executions": {
        "get": {
            "tags": ["Workflows"],
            "summary": "List workflow executions",
            "operationId": "listWorkflowExecutions",
            "description": "Get list of all workflow executions for the runtime dashboard.",
            "parameters": [
                {
                    "name": "workflow_id",
                    "in": "query",
                    "description": "Filter by workflow ID",
                    "schema": {"type": "string"},
                },
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by execution status",
                    "schema": {
                        "type": "string",
                        "enum": ["pending", "running", "completed", "failed", "cancelled"],
                    },
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50},
                },
            ],
            "responses": {
                "200": _ok_response("List of workflow executions", "ExecutionList"),
            },
        },
    },
    "/api/workflow-executions/{execution_id}": {
        "get": {
            "tags": ["Workflows"],
            "summary": "Get execution status",
            "operationId": "getWorkflowExecution",
            "description": "Get detailed status of a workflow execution.",
            "parameters": [
                {
                    "name": "execution_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Execution details with step progress"),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "delete": {
            "tags": ["Workflows"],
            "summary": "Cancel execution",
            "operationId": "deleteWorkflowExecution",
            "description": "Cancel a running workflow execution.",
            "parameters": [
                {
                    "name": "execution_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Execution cancelled"),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/workflow-approvals": {
        "get": {
            "tags": ["Workflows"],
            "summary": "List pending approvals",
            "operationId": "listWorkflowApprovals",
            "description": "Get list of workflow steps awaiting human approval.",
            "security": [{"bearerAuth": []}],
            "responses": {
                "200": _ok_response("List of pending approvals"),
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/workflow-approvals/{approval_id}": {
        "post": {
            "tags": ["Workflows"],
            "summary": "Submit approval decision",
            "operationId": "createWorkflowApproval",
            "description": "Approve or reject a workflow step requiring human approval.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "approval_id",
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
                            "required": ["decision"],
                            "properties": {
                                "decision": {
                                    "type": "string",
                                    "enum": ["approve", "reject"],
                                },
                                "comment": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _ok_response("Approval submitted"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
}
