"""Pattern template endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

PATTERN_ENDPOINTS = {
    "/api/patterns": {
        "get": {
            "tags": ["Patterns"],
            "summary": "List pattern templates",
            "description": "Get list of available workflow pattern templates.",
            "parameters": [
                {
                    "name": "category",
                    "in": "query",
                    "description": "Filter by pattern category",
                    "schema": {"type": "string"},
                },
                {
                    "name": "tags",
                    "in": "query",
                    "description": "Filter by tags (comma-separated)",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("List of pattern templates", "PatternTemplateList"),
            },
        },
    },
    "/api/patterns/{pattern_id}": {
        "get": {
            "tags": ["Patterns"],
            "summary": "Get pattern template",
            "description": "Get a specific pattern template by ID.",
            "parameters": [
                {
                    "name": "pattern_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Pattern template details", "PatternTemplate"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/patterns/hive-mind": {
        "post": {
            "tags": ["Patterns"],
            "summary": "Create Hive Mind workflow",
            "description": "Create a workflow from the Hive Mind pattern template.",
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Custom workflow name",
                                },
                                "task": {
                                    "type": "string",
                                    "description": "Task to analyze",
                                },
                                "agents": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Agent names to include",
                                },
                                "consensus_mode": {
                                    "type": "string",
                                    "enum": ["majority", "weighted", "unanimous"],
                                    "default": "majority",
                                },
                                "consensus_threshold": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "default": 0.7,
                                },
                                "include_dissent": {
                                    "type": "boolean",
                                    "default": True,
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _ok_response("Workflow created from Hive Mind pattern", "Workflow"),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/patterns/map-reduce": {
        "post": {
            "tags": ["Patterns"],
            "summary": "Create MapReduce workflow",
            "description": "Create a workflow from the MapReduce pattern template.",
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Custom workflow name",
                                },
                                "split_strategy": {
                                    "type": "string",
                                    "enum": ["chunks", "lines", "sentences", "paragraphs"],
                                    "default": "chunks",
                                },
                                "chunk_size": {
                                    "type": "integer",
                                    "minimum": 100,
                                    "maximum": 100000,
                                    "default": 4000,
                                },
                                "map_agent": {
                                    "type": "string",
                                    "description": "Agent for map phase",
                                },
                                "reduce_agent": {
                                    "type": "string",
                                    "description": "Agent for reduce phase",
                                },
                                "map_prompt": {
                                    "type": "string",
                                    "description": "Custom map prompt template",
                                },
                                "reduce_prompt": {
                                    "type": "string",
                                    "description": "Custom reduce prompt template",
                                },
                                "parallel_limit": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 20,
                                    "default": 5,
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _ok_response("Workflow created from MapReduce pattern", "Workflow"),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/patterns/review-cycle": {
        "post": {
            "tags": ["Patterns"],
            "summary": "Create Review Cycle workflow",
            "description": "Create a workflow from the Review Cycle pattern template.",
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Custom workflow name",
                                },
                                "task": {
                                    "type": "string",
                                    "description": "Task to accomplish",
                                },
                                "draft_agent": {
                                    "type": "string",
                                    "description": "Agent for drafting phase",
                                },
                                "review_agent": {
                                    "type": "string",
                                    "description": "Agent for review phase",
                                },
                                "max_iterations": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "default": 3,
                                },
                                "convergence_threshold": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "default": 0.85,
                                },
                                "review_criteria": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Criteria for review",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _ok_response("Workflow created from Review Cycle pattern", "Workflow"),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
}
