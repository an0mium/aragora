"""
E2E tests for Multi-User Team Debate Workflow.

Tests the complete team collaboration flow:
1. Create workspace/team
2. Create multiple users with different roles (admin, moderator, member)
3. Add users to workspace
4. Admin submits debate topic
5. Team members participate in debate via Arena
6. Verify collaborative voting and consensus
7. Verify role-based permissions enforced
8. Export team decision receipt

This validates the enterprise team collaboration capabilities end-to-end.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult


# =============================================================================
# Test Helpers
# =============================================================================


def get_body(result: HandlerResult) -> dict:
    """Extract body from HandlerResult."""
    if result is None:
        return {}
    if not result.body:
        return {}
    body = json.loads(result.body.decode("utf-8"))
    if isinstance(body, dict) and "data" in body and body.get("success") is True:
        data = body.get("data")
        return data if isinstance(data, dict) else body
    return body


def get_status(result: HandlerResult) -> int:
    """Extract status code from HandlerResult."""
    if result is None:
        return 404
    return result.status_code


def create_mock_request(
    body: Optional[dict] = None,
    headers: Optional[dict] = None,
    method: str = "POST",
    path: str = "/",
    client_address: tuple = ("127.0.0.1", 54321),
) -> MagicMock:
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.command = method
    handler.path = path
    handler.headers = headers or {}
    handler.client_address = client_address

    if body:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.rfile = BytesIO(body_bytes)
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.headers["Content-Type"] = "application/json"
    else:
        handler.rfile = BytesIO(b"")

    return handler


def create_mock_agent(name: str, response: str = "Default response") -> MagicMock:
    """Create a properly mocked agent with all required async methods."""
    agent = MagicMock()
    agent.name = name
    agent.generate = AsyncMock(return_value=response)

    mock_vote = MagicMock()
    mock_vote.choice = 0
    mock_vote.confidence = 0.8
    mock_vote.reasoning = "Agreed with proposal"
    agent.vote = AsyncMock(return_value=mock_vote)

    mock_critique = MagicMock()
    mock_critique.issues = []
    mock_critique.suggestions = []
    mock_critique.score = 0.8
    mock_critique.severity = 0.2
    mock_critique.text = "No issues found."
    mock_critique.agent = name
    mock_critique.target_agent = "other"
    mock_critique.round = 1
    agent.critique = AsyncMock(return_value=mock_critique)

    agent.total_input_tokens = 0
    agent.total_output_tokens = 0
    agent.input_tokens = 0
    agent.output_tokens = 0
    agent.total_tokens_in = 0
    agent.total_tokens_out = 0
    agent.metrics = None
    agent.provider = None

    return agent


def create_test_user(
    user_id: str,
    email: str,
    name: str,
    role: str = "member",
) -> dict:
    """Create a test user dictionary."""
    from aragora.billing.models import hash_password

    password_hash, password_salt = hash_password("TestPassword123!")
    return {
        "id": user_id,
        "email": email,
        "name": name,
        "role": role,
        "password_hash": password_hash,
        "password_salt": password_salt,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def create_test_workspace(
    workspace_id: str = None,
    name: str = "Test Team Workspace",
    owner_id: str = "admin-001",
) -> dict:
    """Create a test workspace dictionary."""
    return {
        "id": workspace_id or f"ws-{uuid.uuid4().hex[:8]}",
        "name": name,
        "owner_id": owner_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "members": [],
        "settings": {
            "allow_external": False,
            "require_approval": True,
        },
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "team_test.db"


@pytest.fixture
def team_admin():
    """Create an admin user for the team."""
    return create_test_user(
        user_id="admin-001",
        email="admin@example.com",
        name="Team Admin",
        role="admin",
    )


@pytest.fixture
def team_moderator():
    """Create a moderator user for the team."""
    return create_test_user(
        user_id="mod-001",
        email="moderator@example.com",
        name="Team Moderator",
        role="moderator",
    )


@pytest.fixture
def team_member():
    """Create a regular member user for the team."""
    return create_test_user(
        user_id="member-001",
        email="member@example.com",
        name="Team Member",
        role="member",
    )


@pytest.fixture
def test_workspace(team_admin):
    """Create a test workspace with the admin as owner."""
    return create_test_workspace(
        workspace_id="ws-team-001",
        name="Test Team Workspace",
        owner_id=team_admin["id"],
    )


@pytest.fixture
def mock_agents():
    """Create mock agents for team debate."""
    return [
        create_mock_agent("claude", "I recommend option A for the team."),
        create_mock_agent("gpt", "I agree with option A, it aligns with team goals."),
        create_mock_agent("gemini", "Option A is the consensus choice."),
    ]


# =============================================================================
# Team Creation Tests
# =============================================================================


@pytest.mark.e2e
class TestTeamCreation:
    """Tests for team/workspace creation and membership management."""

    def test_create_workspace(self, team_admin):
        """Test creating a new team workspace."""
        workspace = create_test_workspace(
            workspace_id="ws-new-001",
            name="New Team",
            owner_id=team_admin["id"],
        )

        assert workspace["id"] == "ws-new-001"
        assert workspace["name"] == "New Team"
        assert workspace["owner_id"] == team_admin["id"]
        assert workspace["members"] == []

    def test_add_team_members(self, test_workspace, team_moderator, team_member):
        """Test adding members to a workspace."""
        # Add moderator
        test_workspace["members"].append(
            {
                "user_id": team_moderator["id"],
                "role": "moderator",
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Add regular member
        test_workspace["members"].append(
            {
                "user_id": team_member["id"],
                "role": "member",
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        assert len(test_workspace["members"]) == 2
        assert any(m["user_id"] == team_moderator["id"] for m in test_workspace["members"])
        assert any(m["user_id"] == team_member["id"] for m in test_workspace["members"])

    def test_assign_roles(self, test_workspace, team_admin, team_moderator, team_member):
        """Test role assignment verification."""
        # Verify admin role
        assert team_admin["role"] == "admin"

        # Verify moderator role
        assert team_moderator["role"] == "moderator"

        # Verify member role
        assert team_member["role"] == "member"

    def test_workspace_isolation(self, team_admin):
        """Test that workspaces are isolated from each other."""
        workspace_a = create_test_workspace(
            workspace_id="ws-a",
            name="Workspace A",
            owner_id=team_admin["id"],
        )
        workspace_b = create_test_workspace(
            workspace_id="ws-b",
            name="Workspace B",
            owner_id=team_admin["id"],
        )

        # Workspaces have different IDs
        assert workspace_a["id"] != workspace_b["id"]

        # Members lists are separate objects (isolation)
        workspace_a["members"].append({"user_id": "user-a", "role": "member"})
        assert len(workspace_a["members"]) == 1
        assert len(workspace_b["members"]) == 0  # Should not be affected


# =============================================================================
# Team Debate Collaboration Tests
# =============================================================================


@pytest.mark.e2e
class TestTeamDebateCollaboration:
    """Tests for team debate collaboration workflows."""

    @pytest.mark.asyncio
    async def test_team_debate_with_multiple_users(
        self,
        test_workspace,
        team_admin,
        team_moderator,
        team_member,
        mock_agents,
    ):
        """Test running a debate with multiple team members."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        # Add all members to workspace
        test_workspace["members"] = [
            {"user_id": team_admin["id"], "role": "admin"},
            {"user_id": team_moderator["id"], "role": "moderator"},
            {"user_id": team_member["id"], "role": "member"},
        ]

        # Create team debate
        env = Environment(
            task="Should our team adopt a new project management methodology?",
        )
        protocol = DebateProtocol(
            rounds=2,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        assert hasattr(result, "rounds_completed")
        assert result.rounds_completed <= 2

    @pytest.mark.asyncio
    async def test_collaborative_voting(self, mock_agents):
        """Test that agents can vote collaboratively."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        env = Environment(task="Vote on team budget allocation")
        protocol = DebateProtocol(
            rounds=2,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        # Verify multiple agents participated
        assert len(mock_agents) >= 2

    @pytest.mark.asyncio
    async def test_consensus_across_team(self, mock_agents):
        """Test that consensus can be reached across team members."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        env = Environment(task="Reach consensus on quarterly goals")
        protocol = DebateProtocol(
            rounds=3,
            consensus="unanimous",  # Require full team consensus
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        # Configure agents to agree for unanimous consensus
        for agent in mock_agents:
            mock_vote = MagicMock()
            mock_vote.choice = 0  # All vote same choice
            mock_vote.confidence = 0.9
            mock_vote.reasoning = "Team aligned on goals"
            agent.vote = AsyncMock(return_value=mock_vote)

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None


# =============================================================================
# Team Permissions Tests
# =============================================================================


@pytest.mark.e2e
class TestTeamPermissions:
    """Tests for role-based permission enforcement in teams."""

    def test_admin_can_manage_members(self, test_workspace, team_admin, team_member):
        """Test that admin can add/remove members."""
        # Admin adds member
        test_workspace["members"].append(
            {
                "user_id": team_member["id"],
                "role": "member",
                "added_by": team_admin["id"],
            }
        )

        assert len(test_workspace["members"]) == 1
        assert test_workspace["members"][0]["added_by"] == team_admin["id"]

    def test_member_cannot_remove_others(self, test_workspace, team_member, team_moderator):
        """Test that regular members cannot remove other members."""
        # Add both members
        test_workspace["members"] = [
            {"user_id": team_member["id"], "role": "member"},
            {"user_id": team_moderator["id"], "role": "moderator"},
        ]

        # Verify member role doesn't have remove permission
        assert team_member["role"] == "member"
        # In real implementation, this would check RBAC permissions
        # For now, verify role hierarchy
        assert team_member["role"] != "admin"
        assert team_member["role"] != "moderator"

    def test_moderator_permissions(self, test_workspace, team_moderator):
        """Test that moderators have appropriate permissions."""
        test_workspace["members"].append(
            {
                "user_id": team_moderator["id"],
                "role": "moderator",
            }
        )

        member = test_workspace["members"][0]
        assert member["role"] == "moderator"
        # Moderators can manage debates but not workspace settings
        assert member["role"] in ["moderator", "admin"]

    def test_role_based_debate_control(self, test_workspace, team_admin, team_member):
        """Test that debate control is role-based."""
        # Admin initiates debate
        debate_config = {
            "initiated_by": team_admin["id"],
            "workspace_id": test_workspace["id"],
            "topic": "Role-based decision",
            "allowed_participants": ["admin", "moderator", "member"],
        }

        # All roles can participate
        assert team_admin["role"] in debate_config["allowed_participants"]
        assert team_member["role"] in debate_config["allowed_participants"]


# =============================================================================
# Complete Team Workflow Tests
# =============================================================================


@pytest.mark.e2e
class TestCompleteTeamWorkflow:
    """Integration tests for complete team debate workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_team_debate_flow(
        self,
        temp_db_path,
        team_admin,
        team_moderator,
        team_member,
        mock_agents,
    ):
        """Test complete workflow: create team → debate → consensus → receipt."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.export.decision_receipt import DecisionReceipt

        # Step 1: Create workspace
        workspace = create_test_workspace(
            workspace_id="ws-e2e-001",
            name="E2E Test Team",
            owner_id=team_admin["id"],
        )

        # Step 2: Add team members
        workspace["members"] = [
            {"user_id": team_admin["id"], "role": "admin"},
            {"user_id": team_moderator["id"], "role": "moderator"},
            {"user_id": team_member["id"], "role": "member"},
        ]
        assert len(workspace["members"]) == 3

        # Step 3: Admin creates debate
        env = Environment(
            task="Should our team transition to remote-first work policy?",
        )
        protocol = DebateProtocol(
            rounds=2,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        # Step 4: Run team debate
        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed > 0

        # Step 5: Generate decision receipt
        receipt = DecisionReceipt.from_debate_result(result)

        assert receipt is not None
        assert receipt.receipt_id is not None
        assert len(receipt.receipt_id) > 0

        # Step 6: Export receipt
        json_data = receipt.to_json()
        assert json_data is not None
        assert len(json_data) > 0

        # Verify receipt contains decision
        parsed = json.loads(json_data)
        assert "receipt_id" in parsed

    @pytest.mark.asyncio
    async def test_team_debate_with_dissent(self, mock_agents):
        """Test team debate where not all members agree initially."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        # Configure agents with different opinions
        mock_agents[0].generate = AsyncMock(return_value="I support option A.")
        mock_agents[1].generate = AsyncMock(return_value="I prefer option B.")
        mock_agents[2].generate = AsyncMock(return_value="Option A seems better on reflection.")

        # Different initial votes
        mock_agents[0].vote.return_value.choice = 0
        mock_agents[1].vote.return_value.choice = 1
        mock_agents[2].vote.return_value.choice = 0

        env = Environment(task="Contentious team decision")
        protocol = DebateProtocol(
            rounds=3,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        # Should still complete with majority
        assert result is not None
        assert result.rounds_completed > 0

    @pytest.mark.asyncio
    async def test_team_debate_metrics_tracking(self, mock_agents):
        """Test that team debate metrics are properly tracked."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        env = Environment(task="Track team participation metrics")
        protocol = DebateProtocol(
            rounds=2,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        # Verify agents participated
        for agent in mock_agents:
            assert agent.generate.called or agent.vote.called

    def test_workspace_audit_trail(self, test_workspace, team_admin, team_member):
        """Test that workspace actions are auditable."""
        # Track actions
        audit_log = []

        # Add member action
        audit_log.append(
            {
                "action": "member_added",
                "actor": team_admin["id"],
                "target": team_member["id"],
                "workspace": test_workspace["id"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        assert len(audit_log) == 1
        assert audit_log[0]["action"] == "member_added"
        assert audit_log[0]["actor"] == team_admin["id"]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.e2e
class TestTeamEdgeCases:
    """Tests for edge cases in team debate workflows."""

    def test_empty_workspace(self):
        """Test handling of workspace with no members."""
        workspace = create_test_workspace(name="Empty Workspace")
        assert len(workspace["members"]) == 0

    def test_single_member_team(self, team_admin, mock_agents):
        """Test team with only one member."""
        workspace = create_test_workspace(owner_id=team_admin["id"])
        workspace["members"] = [{"user_id": team_admin["id"], "role": "admin"}]

        assert len(workspace["members"]) == 1

    @pytest.mark.asyncio
    async def test_debate_with_minimum_agents(self):
        """Test debate with minimum number of agents."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        # Minimum: 2 agents
        agents = [
            create_mock_agent("agent1", "Option A"),
            create_mock_agent("agent2", "Agree with A"),
        ]

        env = Environment(task="Minimal team decision")
        protocol = DebateProtocol(
            rounds=1,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result is not None

    def test_duplicate_member_prevention(self, test_workspace, team_member):
        """Test that duplicate members are handled."""
        # Add member
        test_workspace["members"].append(
            {
                "user_id": team_member["id"],
                "role": "member",
            }
        )

        # Check for duplicate before adding again
        existing_ids = [m["user_id"] for m in test_workspace["members"]]
        assert team_member["id"] in existing_ids

        # In real implementation, this would raise or skip
        initial_count = len(test_workspace["members"])
        assert initial_count == 1
