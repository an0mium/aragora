"""Tests for GraphDebatesAPI resource."""

import pytest

from aragora.client import AragoraClient
from aragora.client.resources.graph_debates import GraphDebatesAPI


class TestGraphDebatesAPI:
    """Tests for GraphDebatesAPI resource."""

    def test_graph_debates_api_exists(self):
        """Test that GraphDebatesAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.graph_debates, GraphDebatesAPI)

    def test_graph_debates_api_has_create_methods(self):
        """Test that GraphDebatesAPI has create methods."""
        client = AragoraClient()
        assert hasattr(client.graph_debates, "create")
        assert hasattr(client.graph_debates, "create_async")
        assert callable(client.graph_debates.create)

    def test_graph_debates_api_has_get_methods(self):
        """Test that GraphDebatesAPI has get methods."""
        client = AragoraClient()
        assert hasattr(client.graph_debates, "get")
        assert hasattr(client.graph_debates, "get_async")

    def test_graph_debates_api_has_get_branches_methods(self):
        """Test that GraphDebatesAPI has get_branches methods."""
        client = AragoraClient()
        assert hasattr(client.graph_debates, "get_branches")
        assert hasattr(client.graph_debates, "get_branches_async")


class TestGraphDebateModels:
    """Tests for GraphDebate model classes."""

    def test_graph_debate_create_request_import(self):
        """Test GraphDebateCreateRequest model can be imported."""
        from aragora.client.models import GraphDebateCreateRequest

        request = GraphDebateCreateRequest(
            task="Should we use microservices?",
        )
        assert request.task == "Should we use microservices?"

    def test_graph_debate_create_response_import(self):
        """Test GraphDebateCreateResponse model can be imported."""
        from aragora.client.models import GraphDebateCreateResponse

        response = GraphDebateCreateResponse(
            debate_id="graph_debate_001",
            status="running",
            task="Test task",
        )
        assert response.debate_id == "graph_debate_001"
        assert response.status == "running"

    def test_graph_debate_branch_import(self):
        """Test GraphDebateBranch model can be imported."""
        from aragora.client.models import GraphDebateBranch

        branch = GraphDebateBranch(
            branch_id="branch_001",
            name="Main branch",
        )
        assert branch.branch_id == "branch_001"
        assert branch.name == "Main branch"
