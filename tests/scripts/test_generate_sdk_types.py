from __future__ import annotations

import json

import pytest

import scripts.generate_sdk_types as generate_sdk_types


def test_find_duplicate_operation_ids_returns_collisions() -> None:
    spec = {
        "paths": {
            "/api/v1/agent/{name}/positions": {
                "get": {"operationId": "_get_positions"},
            },
            "/api/v1/debates/{debate_id}/positions": {
                "get": {"operationId": "_get_positions"},
            },
            "/api/v1/ok": {
                "post": {"operationId": "createOk"},
            },
        }
    }

    assert generate_sdk_types.find_duplicate_operation_ids(spec) == {
        "_get_positions": [
            "GET /api/v1/agent/{name}/positions",
            "GET /api/v1/debates/{debate_id}/positions",
        ]
    }


def test_ensure_unique_operation_ids_rejects_duplicate_json_spec(tmp_path) -> None:
    openapi_path = tmp_path / "openapi.json"
    openapi_path.write_text(
        json.dumps(
            {
                "paths": {
                    "/api/v1/coordination/workspaces": {
                        "get": {"operationId": "_handle_list_workspaces"},
                    },
                    "/api/v1/federation/workspaces": {
                        "get": {"operationId": "_handle_list_workspaces"},
                    },
                }
            }
        )
    )

    with pytest.raises(ValueError, match="_handle_list_workspaces"):
        generate_sdk_types.ensure_unique_operation_ids(openapi_path)
