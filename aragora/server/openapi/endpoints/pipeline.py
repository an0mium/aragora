"""Pipeline endpoint definitions for the Idea-to-Execution canvas pipeline."""

from aragora.server.openapi.helpers import STANDARD_ERRORS, AUTH_REQUIREMENTS

_ID_PARAM = {
    "name": "id",
    "in": "path",
    "required": True,
    "schema": {"type": "string"},
    "description": "Pipeline ID",
}

_STAGE_PARAM = {
    "name": "stage",
    "in": "path",
    "required": True,
    "schema": {"type": "string", "enum": ["ideas", "goals", "actions", "orchestration"]},
    "description": "Pipeline stage name",
}

_GRAPH_BODY = {
    "required": True,
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "summary": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "string"},
                                "target_id": {"type": "string"},
                                "relation": {"type": "string"},
                            },
                        },
                    },
                },
            }
        }
    },
}


def _json_response(description: str) -> dict:
    """Inline 200 response with generic JSON object body."""
    return {
        "description": description,
        "content": {"application/json": {"schema": {"type": "object"}}},
    }


PIPELINE_ENDPOINTS = {
    "/api/v1/canvas/pipeline/from-ideas": {
        "post": {
            "tags": ["Pipeline"],
            "summary": "Create pipeline from ideas",
            "operationId": "createPipelineFromIdeas",
            "description": "Create a full 4-stage pipeline from a list of raw idea strings.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["ideas"],
                            "properties": {
                                "ideas": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of idea strings",
                                }
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _json_response("Pipeline result"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/canvas/pipeline/from-debate": {
        "post": {
            "tags": ["Pipeline"],
            "summary": "Create pipeline from debate",
            "operationId": "createPipelineFromDebate",
            "description": "Create a pipeline from an ArgumentCartographer debate graph (nodes + edges).",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": _GRAPH_BODY,
            "responses": {
                "200": _json_response("Pipeline result"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/canvas/pipeline/from-template": {
        "post": {
            "tags": ["Pipeline"],
            "summary": "Create pipeline from template",
            "operationId": "createPipelineFromTemplate",
            "description": "Create a pipeline from a named template.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["template_id"],
                            "properties": {
                                "template_id": {"type": "string"},
                                "parameters": {"type": "object"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _json_response("Pipeline result"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/canvas/pipeline/advance": {
        "post": {
            "tags": ["Pipeline"],
            "summary": "Advance pipeline stage",
            "operationId": "advancePipelineStage",
            "description": "Advance a pipeline to the next stage (e.g. ideas -> goals).",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["pipeline_id", "target_stage"],
                            "properties": {
                                "pipeline_id": {"type": "string"},
                                "target_stage": {
                                    "type": "string",
                                    "enum": ["ideas", "goals", "actions", "orchestration"],
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _json_response("Advance result"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/canvas/pipeline/run": {
        "post": {
            "tags": ["Pipeline"],
            "summary": "Run async pipeline",
            "operationId": "runPipeline",
            "description": "Start an asynchronous pipeline execution from ideas through all stages.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["ideas"],
                            "properties": {
                                "ideas": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "202": {
                    "description": "Pipeline started",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "pipeline_id": {"type": "string"},
                                    "status": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/canvas/pipeline/extract-goals": {
        "post": {
            "tags": ["Pipeline"],
            "summary": "Extract goals from ideas",
            "operationId": "extractGoals",
            "description": "Use AI to extract structured goals from an ideas canvas.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": _GRAPH_BODY,
            "responses": {
                "200": _json_response("Extracted goals"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/canvas/convert/debate": {
        "post": {
            "tags": ["Pipeline"],
            "summary": "Convert debate to ideas canvas",
            "operationId": "convertDebateToCanvas",
            "description": "Convert a debate graph into an ideas-stage canvas.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": _GRAPH_BODY,
            "responses": {
                "200": _json_response("Ideas canvas"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/canvas/convert/workflow": {
        "post": {
            "tags": ["Pipeline"],
            "summary": "Convert workflow to actions canvas",
            "operationId": "convertWorkflowToCanvas",
            "description": "Convert a workflow definition into an actions-stage canvas.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"type": "object"},
                    }
                },
            },
            "responses": {
                "200": _json_response("Actions canvas"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/canvas/pipeline/templates": {
        "get": {
            "tags": ["Pipeline"],
            "summary": "List pipeline templates",
            "operationId": "listPipelineTemplates",
            "description": "List available pipeline templates.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _json_response("Template list"),
            },
        },
    },
    "/api/v1/canvas/pipeline/{id}": {
        "get": {
            "tags": ["Pipeline"],
            "summary": "Get pipeline",
            "operationId": "getPipeline",
            "description": "Get a pipeline result by ID.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [_ID_PARAM],
            "responses": {
                "200": _json_response("Pipeline result"),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "put": {
            "tags": ["Pipeline"],
            "summary": "Save pipeline canvas state",
            "operationId": "savePipeline",
            "description": "Save the current canvas state for a pipeline.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [_ID_PARAM],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": {"type": "object"}},
                },
            },
            "responses": {
                "200": _json_response("Saved"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/canvas/pipeline/{id}/status": {
        "get": {
            "tags": ["Pipeline"],
            "summary": "Get pipeline stage status",
            "operationId": "getPipelineStatus",
            "description": "Get per-stage completion status for a pipeline.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [_ID_PARAM],
            "responses": {
                "200": _json_response("Stage status"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/canvas/pipeline/{id}/stage/{stage}": {
        "get": {
            "tags": ["Pipeline"],
            "summary": "Get pipeline stage canvas",
            "operationId": "getPipelineStage",
            "description": "Get the canvas data for a specific pipeline stage.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [_ID_PARAM, _STAGE_PARAM],
            "responses": {
                "200": _json_response("Stage canvas"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/canvas/pipeline/{id}/graph": {
        "get": {
            "tags": ["Pipeline"],
            "summary": "Get pipeline React Flow graph",
            "operationId": "getPipelineGraph",
            "description": "Get the React Flow compatible graph JSON for a pipeline.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                _ID_PARAM,
                {
                    "name": "stage",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter to a specific stage",
                },
            ],
            "responses": {
                "200": _json_response("React Flow graph"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/canvas/pipeline/{id}/receipt": {
        "get": {
            "tags": ["Pipeline"],
            "summary": "Get pipeline decision receipt",
            "operationId": "getPipelineReceipt",
            "description": "Get the cryptographic decision receipt for a completed pipeline.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [_ID_PARAM],
            "responses": {
                "200": _json_response("Decision receipt"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/canvas/pipeline/{id}/approve-transition": {
        "post": {
            "tags": ["Pipeline"],
            "summary": "Approve stage transition",
            "operationId": "approvePipelineTransition",
            "description": "Approve or reject a pending stage transition.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [_ID_PARAM],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["transition_id", "approved"],
                            "properties": {
                                "transition_id": {"type": "string"},
                                "approved": {"type": "boolean"},
                                "reason": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _json_response("Transition result"),
                "404": STANDARD_ERRORS["404"],
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
}
