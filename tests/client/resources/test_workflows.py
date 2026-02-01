"""Tests for WorkflowsAPI resource."""

import pytest

from aragora.client import AragoraClient
from aragora.client.resources.workflows import WorkflowsAPI


class TestWorkflowsAPI:
    """Tests for WorkflowsAPI resource."""

    def test_workflows_api_exists(self):
        """Test that WorkflowsAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.workflows, WorkflowsAPI)

    def test_workflows_api_has_basic_methods(self):
        """Test that WorkflowsAPI has basic methods."""
        client = AragoraClient()
        api = client.workflows
        # Check API is properly instantiated
        assert api is not None
        assert api._client is not None

    def test_workflows_api_has_list_methods(self):
        """Test that WorkflowsAPI has list methods."""
        client = AragoraClient()
        assert hasattr(client.workflows, "list")
        assert hasattr(client.workflows, "list_async")

    def test_workflows_api_has_get_methods(self):
        """Test that WorkflowsAPI has get methods."""
        client = AragoraClient()
        assert hasattr(client.workflows, "get")
        assert hasattr(client.workflows, "get_async")
