"""Knowledge Mound checkpoint endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

CHECKPOINT_ENDPOINTS = {
    "/api/km/checkpoints": {
        "get": {
            "tags": ["Knowledge Mound"],
            "summary": "List checkpoints",
            "description": "Get list of Knowledge Mound checkpoints.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of checkpoints to return",
                    "schema": {"type": "integer", "default": 20, "maximum": 100},
                },
            ],
            "responses": {
                "200": _ok_response("List of checkpoints", "CheckpointList"),
                "401": STANDARD_ERRORS["401"],
            },
        },
        "post": {
            "tags": ["Knowledge Mound"],
            "summary": "Create checkpoint",
            "description": "Create a new Knowledge Mound checkpoint.",
            "security": [{"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Checkpoint name (must be unique)",
                                    "minLength": 1,
                                    "maxLength": 256,
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Optional checkpoint description",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional tags for organization",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _ok_response("Checkpoint created", "CheckpointMetadata"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "409": {
                    "description": "Checkpoint with this name already exists",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "error": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
    "/api/km/checkpoints/{checkpoint_name}": {
        "get": {
            "tags": ["Knowledge Mound"],
            "summary": "Get checkpoint",
            "description": "Get checkpoint metadata by name.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "checkpoint_name",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Checkpoint metadata", "CheckpointMetadata"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
            },
        },
        "delete": {
            "tags": ["Knowledge Mound"],
            "summary": "Delete checkpoint",
            "description": "Delete a checkpoint by name.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "checkpoint_name",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Checkpoint deleted"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/km/checkpoints/{checkpoint_name}/restore": {
        "post": {
            "tags": ["Knowledge Mound"],
            "summary": "Restore checkpoint",
            "description": "Restore Knowledge Mound state from a checkpoint.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "checkpoint_name",
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
                                "strategy": {
                                    "type": "string",
                                    "enum": ["merge", "replace"],
                                    "default": "merge",
                                    "description": "Restore strategy: merge (keep existing, add missing) or replace (clear then restore)",
                                },
                                "skip_duplicates": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Skip nodes that already exist",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _ok_response("Restore result", "RestoreResult"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/km/checkpoints/{checkpoint_name}/compare": {
        "get": {
            "tags": ["Knowledge Mound"],
            "summary": "Compare checkpoint",
            "description": "Compare checkpoint with current Knowledge Mound state.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "checkpoint_name",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Comparison result",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "checkpoint_name": {"type": "string"},
                                    "checkpoint_node_count": {"type": "integer"},
                                    "current_node_count": {"type": "integer"},
                                    "nodes_added_since": {"type": "integer"},
                                    "nodes_removed_since": {"type": "integer"},
                                    "nodes_modified_since": {"type": "integer"},
                                    "drift_percentage": {"type": "number"},
                                },
                            },
                        },
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/km/checkpoints/compare": {
        "post": {
            "tags": ["Knowledge Mound"],
            "summary": "Compare two checkpoints",
            "description": "Compare two checkpoints to see differences.",
            "security": [{"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["checkpoint_a", "checkpoint_b"],
                            "properties": {
                                "checkpoint_a": {
                                    "type": "string",
                                    "description": "First checkpoint name",
                                },
                                "checkpoint_b": {
                                    "type": "string",
                                    "description": "Second checkpoint name",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "Comparison between two checkpoints",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "checkpoint_a": {"type": "string"},
                                    "checkpoint_b": {"type": "string"},
                                    "a_node_count": {"type": "integer"},
                                    "b_node_count": {"type": "integer"},
                                    "nodes_only_in_a": {"type": "integer"},
                                    "nodes_only_in_b": {"type": "integer"},
                                    "nodes_in_both": {"type": "integer"},
                                    "nodes_modified": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/km/checkpoints/{checkpoint_name}/download": {
        "get": {
            "tags": ["Knowledge Mound"],
            "summary": "Download checkpoint",
            "description": "Download checkpoint file for backup or transfer.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "checkpoint_name",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Checkpoint file download",
                    "content": {
                        "application/octet-stream": {
                            "schema": {
                                "type": "string",
                                "format": "binary",
                            },
                        },
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
}
