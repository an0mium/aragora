"""OpenAPI endpoint definitions for Prompt Engine."""

from __future__ import annotations

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS, _ok_response

_obj: dict[str, Any] = {"type": "object", "additionalProperties": True}
_str: dict[str, Any] = {"type": "string"}
_bool: dict[str, Any] = {"type": "boolean"}
_arr_obj: dict[str, Any] = {"type": "array", "items": _obj}


def _json_request(
    *,
    required: list[str] | None = None,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": properties or {},
    }
    if required:
        schema["required"] = required
    return {
        "required": True,
        "content": {"application/json": {"schema": schema}},
    }


_TIMING = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "total_duration_ms": {"type": "number"},
        "stage_durations_ms": _obj,
        "operation_timings": _arr_obj,
    },
}

_PROMPT_BODY = {
    "prompt": {
        "type": "string",
        "description": "Natural-language operator prompt to transform into a structured specification.",
    },
    "context": {
        "description": "Optional contextual data to pass into the prompt-engine stage.",
    },
}

_INTENT_BODY = {
    "intent": {
        "type": "object",
        "description": "PromptIntent payload returned by decompose or by the full pipeline.",
        "additionalProperties": True,
    }
}

_RUN_RESPONSE = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "specification": _obj,
        "spec_bundle": _obj,
        "intent": _obj,
        "questions": _arr_obj,
        "research": {"anyOf": [_obj, {"type": "null"}]},
        "auto_approved": _bool,
        "stages_completed": {"type": "array", "items": _str},
        "validation": _obj,
        "timing": _TIMING,
        "run": _obj,
        "decision_plan": _obj,
        "execution": _obj,
    },
}


PROMPT_ENGINE_ENDPOINTS: dict[str, dict[str, Any]] = {
    "/api/prompt-engine/runs": {
        "get": {
            "tags": ["Prompt Engine"],
            "summary": "List prompt-engine runs",
            "description": "List persisted prompt-engine backbone runs with optional status and artifact filters.",
            "operationId": "listPromptEngineRuns",
            "parameters": [
                {"name": "status", "in": "query", "schema": _str},
                {"name": "plan_id", "in": "query", "schema": _str},
                {"name": "debate_id", "in": "query", "schema": _str},
                {"name": "execution_id", "in": "query", "schema": _str},
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 0, "default": 0},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Prompt-engine runs",
                    {"runs": {"type": "array", "items": _obj}},
                )
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/prompt-engine/runs/{run_id}": {
        "get": {
            "tags": ["Prompt Engine"],
            "summary": "Get prompt-engine run",
            "description": "Fetch a persisted prompt-engine backbone run by identifier.",
            "operationId": "getPromptEngineRun",
            "parameters": [
                {
                    "name": "run_id",
                    "in": "path",
                    "required": True,
                    "description": "Prompt-engine run identifier.",
                    "schema": _str,
                }
            ],
            "responses": {
                "200": _ok_response("Prompt-engine run", {"run": _obj}),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/prompt-engine/run": {
        "post": {
            "tags": ["Prompt Engine"],
            "summary": "Run prompt-engine pipeline",
            "description": "Run the full prompt-to-specification pipeline and optionally create or schedule a decision plan.",
            "operationId": "runPromptEnginePipeline",
            "requestBody": _json_request(
                required=["prompt"],
                properties={
                    **_PROMPT_BODY,
                    "profile": _str,
                    "autonomy": _str,
                    "skip_research": _bool,
                    "skip_interrogation": _bool,
                    "decision_plan": _obj,
                },
            ),
            "responses": {
                "200": _ok_response("Prompt-engine pipeline result", _RUN_RESPONSE),
                "400": STANDARD_ERRORS["400"],
                "422": _ok_response("Specification is not execution-grade", _obj),
                "503": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/prompt-engine/decompose": {
        "post": {
            "tags": ["Prompt Engine"],
            "summary": "Decompose prompt",
            "description": "Transform a natural-language prompt into a structured PromptIntent.",
            "operationId": "decomposePromptEnginePrompt",
            "requestBody": _json_request(required=["prompt"], properties=_PROMPT_BODY),
            "responses": {
                "200": _ok_response("Prompt intent", {"intent": _obj, "timing": _TIMING}),
                "400": STANDARD_ERRORS["400"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/prompt-engine/interrogate": {
        "post": {
            "tags": ["Prompt Engine"],
            "summary": "Interrogate prompt intent",
            "description": "Generate clarifying questions for a prompt intent.",
            "operationId": "interrogatePromptEngineIntent",
            "requestBody": _json_request(
                required=["intent"],
                properties={**_INTENT_BODY, "depth": _str},
            ),
            "responses": {
                "200": _ok_response(
                    "Clarifying questions",
                    {"questions": _arr_obj, "timing": _TIMING},
                ),
                "400": STANDARD_ERRORS["400"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/prompt-engine/research": {
        "post": {
            "tags": ["Prompt Engine"],
            "summary": "Research prompt intent",
            "description": "Research supporting context for a prompt intent.",
            "operationId": "researchPromptEngineIntent",
            "requestBody": _json_request(
                required=["intent"],
                properties={**_INTENT_BODY, "context": {}},
            ),
            "responses": {
                "200": _ok_response("Research report", {"research": _obj, "timing": _TIMING}),
                "400": STANDARD_ERRORS["400"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/prompt-engine/specify": {
        "post": {
            "tags": ["Prompt Engine"],
            "summary": "Build prompt specification",
            "description": "Build an execution specification from intent, questions, research, and context.",
            "operationId": "specifyPromptEngineIntent",
            "requestBody": _json_request(
                required=["intent"],
                properties={
                    **_INTENT_BODY,
                    "questions": _arr_obj,
                    "research": _obj,
                    "context": {},
                },
            ),
            "responses": {
                "200": _ok_response(
                    "Prompt specification",
                    {"specification": _obj, "spec_bundle": _obj, "timing": _TIMING},
                ),
                "400": STANDARD_ERRORS["400"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/prompt-engine/validate": {
        "post": {
            "tags": ["Prompt Engine"],
            "summary": "Validate prompt specification",
            "description": "Validate a prompt-engine specification and return an execution-grade spec bundle.",
            "operationId": "validatePromptEngineSpecification",
            "requestBody": _json_request(
                required=["specification"],
                properties={"specification": _obj},
            ),
            "responses": {
                "200": _ok_response(
                    "Validation result",
                    {"validation": _obj, "spec_bundle": _obj, "timing": _TIMING},
                ),
                "400": STANDARD_ERRORS["400"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
}
