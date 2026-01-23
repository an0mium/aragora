"""
Tests for Enterprise Connectors Handler.

Tests cover connector types configuration and basic handler creation.
"""

import pytest

from aragora.server.handlers.features.connectors import (
    ConnectorsHandler,
    CONNECTOR_TYPES,
)


class TestConnectorTypes:
    """Tests for connector type configuration."""

    def test_connector_types_defined(self):
        """Test that connector types are configured."""
        assert len(CONNECTOR_TYPES) > 0

    def test_expected_connectors_exist(self):
        """Test that expected connectors are defined."""
        expected = [
            "github",
            "s3",
            "sharepoint",
            "postgresql",
            "mongodb",
            "confluence",
            "notion",
            "slack",
        ]
        for connector in expected:
            assert connector in CONNECTOR_TYPES, f"{connector} should be defined"

    def test_connector_has_required_fields(self):
        """Test that all connectors have required configuration."""
        for connector_id, config in CONNECTOR_TYPES.items():
            assert "name" in config, f"{connector_id} missing name"
            assert "description" in config, f"{connector_id} missing description"
            assert "category" in config, f"{connector_id} missing category"

    def test_connector_categories(self):
        """Test that connectors have valid categories."""
        valid_categories = {
            "accounting",
            "collaboration",
            "database",
            "devops",
            "documents",
            "git",
            "healthcare",
            "legal",
        }
        for connector_id, config in CONNECTOR_TYPES.items():
            category = config.get("category")
            assert category in valid_categories, f"{connector_id} has invalid category: {category}"

    def test_github_connector(self):
        """Test GitHub connector configuration."""
        assert "github" in CONNECTOR_TYPES
        github = CONNECTOR_TYPES["github"]
        assert github["name"] == "GitHub Enterprise"
        assert github["category"] == "git"

    def test_s3_connector(self):
        """Test S3 connector configuration."""
        assert "s3" in CONNECTOR_TYPES
        s3 = CONNECTOR_TYPES["s3"]
        assert s3["name"] == "Amazon S3"
        assert s3["category"] == "documents"

    def test_sharepoint_connector(self):
        """Test SharePoint connector configuration."""
        assert "sharepoint" in CONNECTOR_TYPES
        sharepoint = CONNECTOR_TYPES["sharepoint"]
        assert sharepoint["name"] == "Microsoft SharePoint"
        assert sharepoint["category"] == "documents"

    def test_database_connectors(self):
        """Test database connector configurations."""
        assert "postgresql" in CONNECTOR_TYPES
        assert "mongodb" in CONNECTOR_TYPES

        assert CONNECTOR_TYPES["postgresql"]["category"] == "database"
        assert CONNECTOR_TYPES["mongodb"]["category"] == "database"

    def test_collaboration_connectors(self):
        """Test collaboration connector configurations."""
        assert "confluence" in CONNECTOR_TYPES
        assert "notion" in CONNECTOR_TYPES
        assert "slack" in CONNECTOR_TYPES

        assert CONNECTOR_TYPES["confluence"]["category"] == "collaboration"
        assert CONNECTOR_TYPES["notion"]["category"] == "collaboration"
        assert CONNECTOR_TYPES["slack"]["category"] == "collaboration"

    def test_fhir_healthcare_connector(self):
        """Test FHIR healthcare connector configuration."""
        assert "fhir" in CONNECTOR_TYPES
        fhir = CONNECTOR_TYPES["fhir"]
        assert fhir["category"] == "healthcare"
        assert "FHIR" in fhir["name"]


class TestConnectorsHandler:
    """Tests for ConnectorsHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = ConnectorsHandler(server_context={})
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(ConnectorsHandler, "ROUTES")
        routes = ConnectorsHandler.ROUTES
        assert "/api/v1/connectors" in routes
        assert "/api/v1/connectors/sync-history" in routes
        assert "/api/v1/connectors/stats" in routes
        assert "/api/v1/connectors/health" in routes
        assert "/api/v1/connectors/types" in routes

    def test_handler_resource_type(self):
        """Test that handler has resource type defined."""
        assert hasattr(ConnectorsHandler, "RESOURCE_TYPE")
        assert ConnectorsHandler.RESOURCE_TYPE == "connector"
