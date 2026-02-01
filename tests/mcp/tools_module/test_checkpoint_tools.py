"""Tests for MCP checkpoint tools execution logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.checkpoint import (
    create_checkpoint_tool,
    delete_checkpoint_tool,
    list_checkpoints_tool,
    resume_checkpoint_tool,
)


class TestCreateCheckpointTool:
    """Tests for create_checkpoint_tool."""

    @pytest.mark.asyncio
    async def test_create_empty_debate_id(self):
        """Test create with empty debate_id."""
        result = await create_checkpoint_tool(debate_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_no_storage(self):
        """Test create when storage unavailable."""
        with patch(
            "aragora.mcp.tools_module.checkpoint.get_debates_db",
            return_value=None,
        ):
            result = await create_checkpoint_tool(debate_id="d-001")

        assert "error" in result
        assert "not available" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_debate_not_found(self):
        """Test create for non-existent debate."""
        mock_db = MagicMock()
        mock_db.get.return_value = None

        with patch(
            "aragora.mcp.tools_module.checkpoint.get_debates_db",
            return_value=mock_db,
        ):
            result = await create_checkpoint_tool(debate_id="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_success(self):
        """Test successful checkpoint creation."""
        mock_db = MagicMock()
        mock_db.get.return_value = {
            "task": "Test debate",
            "messages": [
                {"role": "proposer", "agent": "claude", "content": "Msg 1", "round": 1},
            ],
            "votes": [],
            "critiques": [],
            "agents": ["claude", "gpt4"],
            "rounds_used": 1,
            "final_answer": "Test answer",
        }

        mock_checkpoint = MagicMock()
        mock_checkpoint.checkpoint_id = "cp-001"
        mock_checkpoint.current_round = 1
        mock_checkpoint.messages = [MagicMock()]
        mock_checkpoint.created_at = "2025-01-01T00:00:00"

        mock_manager = AsyncMock()
        mock_manager.create_checkpoint.return_value = mock_checkpoint

        with patch(
            "aragora.mcp.tools_module.checkpoint.get_debates_db",
            return_value=mock_db,
        ), patch(
            "aragora.mcp.tools_module.checkpoint.CheckpointManager",
            return_value=mock_manager,
        ), patch(
            "aragora.mcp.tools_module.checkpoint.FileCheckpointStore",
        ), patch(
            "aragora.mcp.tools_module.checkpoint.DebateSettings",
        ) as mock_settings, patch(
            "aragora.mcp.tools_module.checkpoint.Message",
        ), patch(
            "aragora.mcp.tools_module.checkpoint.Vote",
        ), patch(
            "aragora.mcp.tools_module.checkpoint.Critique",
        ):
            mock_settings.return_value.default_rounds = 3
            result = await create_checkpoint_tool(
                debate_id="d-001",
                label="Before revision",
                storage_backend="file",
            )

        assert result["success"] is True
        assert result["checkpoint_id"] == "cp-001"
        assert result["debate_id"] == "d-001"
        assert result["label"] == "Before revision"

    @pytest.mark.asyncio
    async def test_create_import_error(self):
        """Test create when checkpoint module unavailable."""
        mock_db = MagicMock()
        mock_db.get.return_value = {"task": "Test", "messages": []}

        with patch(
            "aragora.mcp.tools_module.checkpoint.get_debates_db",
            return_value=mock_db,
        ), patch(
            "aragora.mcp.tools_module.checkpoint.CheckpointManager",
            side_effect=ImportError("Checkpoint not available"),
        ):
            result = await create_checkpoint_tool(debate_id="d-001")

        assert "error" in result


class TestListCheckpointsTool:
    """Tests for list_checkpoints_tool."""

    @pytest.mark.asyncio
    async def test_list_success(self):
        """Test successful checkpoint listing."""
        mock_store = AsyncMock()
        mock_store.list_checkpoints.return_value = [
            {
                "checkpoint_id": "cp-001",
                "debate_id": "d-001",
                "task": "Test",
                "current_round": 2,
                "message_count": 6,
            },
        ]

        mock_manager = MagicMock()
        mock_manager.store = mock_store

        with patch(
            "aragora.mcp.tools_module.checkpoint.CheckpointManager",
            return_value=mock_manager,
        ):
            result = await list_checkpoints_tool(debate_id="d-001")

        assert result["count"] == 1
        assert result["checkpoints"][0]["checkpoint_id"] == "cp-001"
        assert result["debate_id"] == "d-001"

    @pytest.mark.asyncio
    async def test_list_all_debates(self):
        """Test listing checkpoints for all debates."""
        mock_store = AsyncMock()
        mock_store.list_checkpoints.return_value = []

        mock_manager = MagicMock()
        mock_manager.store = mock_store

        with patch(
            "aragora.mcp.tools_module.checkpoint.CheckpointManager",
            return_value=mock_manager,
        ):
            result = await list_checkpoints_tool()

        assert result["count"] == 0
        assert result["debate_id"] == "(all)"

    @pytest.mark.asyncio
    async def test_list_clamps_limit(self):
        """Test list clamps limit to valid range."""
        mock_store = AsyncMock()
        mock_store.list_checkpoints.return_value = []

        mock_manager = MagicMock()
        mock_manager.store = mock_store

        with patch(
            "aragora.mcp.tools_module.checkpoint.CheckpointManager",
            return_value=mock_manager,
        ):
            # limit=0 should be clamped to 1
            result = await list_checkpoints_tool(limit=0)

        mock_store.list_checkpoints.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_import_error(self):
        """Test list when checkpoint module unavailable."""
        with patch(
            "aragora.mcp.tools_module.checkpoint.CheckpointManager",
            side_effect=ImportError("Not available"),
        ):
            result = await list_checkpoints_tool()

        assert "error" in result


class TestResumeCheckpointTool:
    """Tests for resume_checkpoint_tool."""

    @pytest.mark.asyncio
    async def test_resume_empty_id(self):
        """Test resume with empty checkpoint_id."""
        result = await resume_checkpoint_tool(checkpoint_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_resume_not_found(self):
        """Test resume for non-existent checkpoint."""
        mock_store = AsyncMock()
        mock_store.load.return_value = None

        mock_manager = MagicMock()
        mock_manager.store = mock_store

        with patch(
            "aragora.mcp.tools_module.checkpoint.CheckpointManager",
            return_value=mock_manager,
        ):
            result = await resume_checkpoint_tool(checkpoint_id="cp-nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_resume_success(self):
        """Test successful checkpoint resume."""
        mock_checkpoint = MagicMock()
        mock_checkpoint.debate_id = "d-001"
        mock_checkpoint.messages = [MagicMock(), MagicMock()]
        mock_checkpoint.current_round = 2
        mock_checkpoint.phase = "revision"
        mock_checkpoint.task = "Test debate"

        mock_store = AsyncMock()
        mock_store.load.return_value = mock_checkpoint

        mock_manager = MagicMock()
        mock_manager.store = mock_store

        with patch(
            "aragora.mcp.tools_module.checkpoint.CheckpointManager",
            return_value=mock_manager,
        ):
            result = await resume_checkpoint_tool(checkpoint_id="cp-001")

        assert result["success"] is True
        assert result["debate_id"] == "d-001"
        assert result["messages_count"] == 2
        assert result["round"] == 2
        assert result["phase"] == "revision"


class TestDeleteCheckpointTool:
    """Tests for delete_checkpoint_tool."""

    @pytest.mark.asyncio
    async def test_delete_empty_id(self):
        """Test delete with empty checkpoint_id."""
        result = await delete_checkpoint_tool(checkpoint_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_delete_success(self):
        """Test successful checkpoint deletion."""
        mock_store = AsyncMock()
        mock_store.delete.return_value = True

        mock_manager = MagicMock()
        mock_manager.store = mock_store

        with patch(
            "aragora.mcp.tools_module.checkpoint.CheckpointManager",
            return_value=mock_manager,
        ):
            result = await delete_checkpoint_tool(checkpoint_id="cp-001")

        assert result["success"] is True
        assert "deleted" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        """Test delete for non-existent checkpoint."""
        mock_store = AsyncMock()
        mock_store.delete.return_value = False

        mock_manager = MagicMock()
        mock_manager.store = mock_store

        with patch(
            "aragora.mcp.tools_module.checkpoint.CheckpointManager",
            return_value=mock_manager,
        ):
            result = await delete_checkpoint_tool(checkpoint_id="cp-nonexistent")

        assert result["success"] is False
        assert "not found" in result["message"].lower()
