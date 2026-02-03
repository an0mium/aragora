"""Tests for codebase OpenAPI schema definitions."""

import pytest

from aragora.server.openapi.schemas.codebase import CODEBASE_SCHEMAS


class TestCodebaseSchemasStructure:
    """Verify all expected schemas exist and have correct structure."""

    def test_schemas_dict_exists(self):
        assert isinstance(CODEBASE_SCHEMAS, dict)
        assert len(CODEBASE_SCHEMAS) > 0

    def test_all_schemas_are_objects(self):
        for name, schema in CODEBASE_SCHEMAS.items():
            assert isinstance(schema, dict), f"{name} is not a dict"
            assert "type" in schema or "$ref" in schema or "allOf" in schema, (
                f"{name} missing 'type', '$ref', or 'allOf'"
            )

    def test_all_object_schemas_have_properties(self):
        for name, schema in CODEBASE_SCHEMAS.items():
            if schema.get("type") == "object":
                assert "properties" in schema, f"{name} object missing 'properties'"


class TestVulnerabilitySchemas:
    def test_vulnerability_reference_exists(self):
        assert "VulnerabilityReference" in CODEBASE_SCHEMAS
        schema = CODEBASE_SCHEMAS["VulnerabilityReference"]
        props = schema["properties"]
        assert "url" in props
        assert "source" in props
        assert "tags" in props

    def test_vulnerability_finding_exists(self):
        assert "VulnerabilityFinding" in CODEBASE_SCHEMAS
        schema = CODEBASE_SCHEMAS["VulnerabilityFinding"]
        props = schema["properties"]
        assert "id" in props
        assert "severity" in props
        assert "cvss_score" in props
        assert "package_name" in props
        assert "fix_available" in props

    def test_vulnerability_finding_has_references_array(self):
        schema = CODEBASE_SCHEMAS["VulnerabilityFinding"]
        refs = schema["properties"]["references"]
        assert refs["type"] == "array"


class TestDependencySchemas:
    def test_dependency_info_exists(self):
        assert "DependencyInfo" in CODEBASE_SCHEMAS
        props = CODEBASE_SCHEMAS["DependencyInfo"]["properties"]
        assert "name" in props
        assert "version" in props
        assert "ecosystem" in props
        assert "direct" in props
        assert "dev_dependency" in props

    def test_dependency_info_boolean_fields(self):
        props = CODEBASE_SCHEMAS["DependencyInfo"]["properties"]
        assert props["direct"]["type"] == "boolean"
        assert props["dev_dependency"]["type"] == "boolean"
        assert props["has_vulnerabilities"]["type"] == "boolean"


class TestScanSchemas:
    def test_codebase_scan_summary_exists(self):
        assert "CodebaseScanSummary" in CODEBASE_SCHEMAS
        props = CODEBASE_SCHEMAS["CodebaseScanSummary"]["properties"]
        assert "total_dependencies" in props
        assert "critical_count" in props
        assert "high_count" in props
        assert "medium_count" in props
        assert "low_count" in props

    def test_scan_summary_integer_fields(self):
        props = CODEBASE_SCHEMAS["CodebaseScanSummary"]["properties"]
        for field_name in ["total_dependencies", "critical_count", "high_count"]:
            assert props[field_name]["type"] == "integer"

    def test_codebase_scan_result_exists(self):
        assert "CodebaseScanResult" in CODEBASE_SCHEMAS
        props = CODEBASE_SCHEMAS["CodebaseScanResult"]["properties"]
        assert "scan_id" in props
        assert "repository" in props
        assert "status" in props
        assert "started_at" in props


class TestSchemaPropertyTypes:
    """Validate that property types are valid OpenAPI types."""

    VALID_TYPES = {"string", "integer", "number", "boolean", "array", "object"}

    def test_all_property_types_valid(self):
        for schema_name, schema in CODEBASE_SCHEMAS.items():
            if schema.get("type") != "object":
                continue
            for prop_name, prop in schema.get("properties", {}).items():
                if "type" in prop:
                    assert prop["type"] in self.VALID_TYPES, (
                        f"{schema_name}.{prop_name} has invalid type: {prop['type']}"
                    )

    def test_array_properties_have_items(self):
        for schema_name, schema in CODEBASE_SCHEMAS.items():
            if schema.get("type") != "object":
                continue
            for prop_name, prop in schema.get("properties", {}).items():
                if prop.get("type") == "array":
                    assert "items" in prop, f"{schema_name}.{prop_name} array missing 'items'"

    def test_datetime_fields_have_format(self):
        for schema_name, schema in CODEBASE_SCHEMAS.items():
            if schema.get("type") != "object":
                continue
            for prop_name, prop in schema.get("properties", {}).items():
                if "date" in prop_name.lower() or prop_name.endswith("_at"):
                    if prop.get("type") == "string" and not prop.get("nullable"):
                        assert prop.get("format") == "date-time", (
                            f"{schema_name}.{prop_name} should have format 'date-time'"
                        )
