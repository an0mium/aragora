"""Tests for aragora.server.openapi.schemas.helpers."""

from aragora.server.openapi.schemas.helpers import (
    STANDARD_ERRORS,
    array_response,
    error_response,
    ok_response,
)


class TestOkResponse:
    def test_description_only(self):
        result = ok_response("Success")
        assert result == {"description": "Success"}
        assert "content" not in result

    def test_with_schema_ref(self):
        result = ok_response("Created", schema_ref="Debate")
        assert result["description"] == "Created"
        assert result["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/Debate"
        }

    def test_none_schema_ref(self):
        result = ok_response("OK", schema_ref=None)
        assert "content" not in result


class TestArrayResponse:
    def test_structure(self):
        result = array_response("List of agents", "Agent")
        assert result["description"] == "List of agents"
        schema = result["content"]["application/json"]["schema"]
        assert schema["type"] == "object"
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"] == {"$ref": "#/components/schemas/Agent"}
        assert schema["properties"]["total"]["type"] == "integer"


class TestErrorResponse:
    def test_structure(self):
        result = error_response("404", "Not found")
        assert result["description"] == "Not found"
        assert result["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/Error"
        }


class TestStandardErrors:
    def test_expected_codes(self):
        assert set(STANDARD_ERRORS.keys()) == {"400", "401", "402", "404", "429", "500"}

    def test_all_reference_error_schema(self):
        for code, resp in STANDARD_ERRORS.items():
            assert resp["content"]["application/json"]["schema"] == {
                "$ref": "#/components/schemas/Error"
            }, f"Code {code} missing Error schema ref"
