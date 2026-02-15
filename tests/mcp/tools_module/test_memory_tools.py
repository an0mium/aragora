"""Tests for MCP memory tools execution logic."""

from unittest.mock import MagicMock, patch

import pytest

from aragora.mcp.tools_module.memory import (
    get_memory_pressure_tool,
    query_memory_tool,
    store_memory_tool,
)



class TestQueryMemoryTool:
    """Tests for query_memory_tool."""

    @pytest.mark.asyncio
    async def test_query_success(self):
        """Test successful memory query."""
        mock_memory = MagicMock()
        mock_memory.id = "mem-123"
        mock_memory.tier = MagicMock(name="medium")
        mock_memory.tier.name = "medium"
        mock_memory.content = "Test memory content"
        mock_memory.importance = 0.75
        mock_memory.created_at = "2024-01-01"

        mock_continuum = MagicMock()
        mock_continuum.retrieve.return_value = [mock_memory]

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            return_value=mock_continuum,
        ):
            result = await query_memory_tool(query="test query")

        assert result["count"] == 1
        assert result["query"] == "test query"
        assert len(result["memories"]) == 1
        assert result["memories"][0]["id"] == "mem-123"
        assert result["memories"][0]["importance"] == 0.75

    @pytest.mark.asyncio
    async def test_query_with_tier_filter(self):
        """Test query with specific tier filter."""
        mock_continuum = MagicMock()
        mock_continuum.retrieve.return_value = []

        mock_memory_tier = MagicMock()
        mock_memory_tier.FAST = "fast"
        mock_memory_tier.MEDIUM = "medium"
        mock_memory_tier.SLOW = "slow"
        mock_memory_tier.GLACIAL = "glacial"

        with (
            patch(
                "aragora.memory.continuum.ContinuumMemory",
                return_value=mock_continuum,
            ),
            patch(
                "aragora.memory.continuum.MemoryTier",
                mock_memory_tier,
            ),
        ):
            result = await query_memory_tool(query="test", tier="fast")

        assert result["tier"] == "fast"

    @pytest.mark.asyncio
    async def test_query_truncates_long_content(self):
        """Test that long content is truncated."""
        long_content = "x" * 600

        mock_memory = MagicMock()
        mock_memory.id = "mem-1"
        mock_memory.tier = MagicMock(name="medium")
        mock_memory.tier.name = "medium"
        mock_memory.content = long_content
        mock_memory.importance = 0.5
        mock_memory.created_at = None

        mock_continuum = MagicMock()
        mock_continuum.retrieve.return_value = [mock_memory]

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            return_value=mock_continuum,
        ):
            result = await query_memory_tool(query="test")

        assert len(result["memories"][0]["content"]) == 503  # 500 + "..."

    @pytest.mark.asyncio
    async def test_query_limit_clamped(self):
        """Test that limit is clamped to valid range."""
        mock_continuum = MagicMock()
        mock_continuum.retrieve.return_value = []

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            return_value=mock_continuum,
        ):
            # Limit below minimum
            await query_memory_tool(query="test", limit=0)
            call_args = mock_continuum.retrieve.call_args
            assert call_args.kwargs["limit"] == 1

            # Limit above maximum
            await query_memory_tool(query="test", limit=200)
            call_args = mock_continuum.retrieve.call_args
            assert call_args.kwargs["limit"] == 100

    @pytest.mark.asyncio
    async def test_query_import_error(self):
        """Test graceful handling when continuum not available."""
        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("Not installed"),
        ):
            result = await query_memory_tool(query="test")

        assert result["count"] == 0
        assert result["memories"] == []


class TestStoreMemoryTool:
    """Tests for store_memory_tool."""

    @pytest.mark.asyncio
    async def test_store_empty_content(self):
        """Test store with empty content."""
        result = await store_memory_tool(content="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_store_success(self):
        """Test successful memory storage."""
        mock_continuum = MagicMock()

        mock_tier_medium = MagicMock()
        mock_tier_medium.name = "medium"

        mock_memory_tier = MagicMock()
        mock_memory_tier.MEDIUM = mock_tier_medium
        mock_memory_tier.__getitem__ = MagicMock(return_value=mock_tier_medium)

        with (
            patch(
                "aragora.memory.continuum.ContinuumMemory",
                return_value=mock_continuum,
            ),
            patch(
                "aragora.memory.continuum.MemoryTier",
                mock_memory_tier,
            ),
        ):
            result = await store_memory_tool(
                content="Test memory content",
                tier="medium",
                importance=0.7,
            )

        assert result["success"] is True
        assert "memory_id" in result
        assert result["memory_id"].startswith("mcp_")
        mock_continuum.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_invalid_tier_defaults(self):
        """Test that invalid tier defaults to medium."""
        mock_continuum = MagicMock()

        mock_tier_medium = MagicMock()
        mock_tier_medium.name = "medium"

        mock_memory_tier = MagicMock()
        mock_memory_tier.MEDIUM = mock_tier_medium
        mock_memory_tier.__getitem__ = MagicMock(side_effect=KeyError)

        with (
            patch(
                "aragora.memory.continuum.ContinuumMemory",
                return_value=mock_continuum,
            ),
            patch(
                "aragora.memory.continuum.MemoryTier",
                mock_memory_tier,
            ),
        ):
            result = await store_memory_tool(
                content="Test",
                tier="invalid_tier",
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_store_importance_clamped(self):
        """Test that importance is clamped to 0-1."""
        mock_continuum = MagicMock()

        mock_tier_medium = MagicMock()
        mock_tier_medium.name = "medium"

        mock_memory_tier = MagicMock()
        mock_memory_tier.MEDIUM = mock_tier_medium
        mock_memory_tier.__getitem__ = MagicMock(return_value=mock_tier_medium)

        with (
            patch(
                "aragora.memory.continuum.ContinuumMemory",
                return_value=mock_continuum,
            ),
            patch(
                "aragora.memory.continuum.MemoryTier",
                mock_memory_tier,
            ),
        ):
            await store_memory_tool(content="Test", importance=1.5)
            call_args = mock_continuum.add.call_args.kwargs
            assert call_args["importance"] == 1.0

            await store_memory_tool(content="Test", importance=-0.5)
            call_args = mock_continuum.add.call_args.kwargs
            assert call_args["importance"] == 0.0

    @pytest.mark.asyncio
    async def test_store_import_error(self):
        """Test graceful handling when continuum not available."""
        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("Not installed"),
        ):
            result = await store_memory_tool(content="Test")

        assert "error" in result
        assert "not available" in result["error"].lower()


class TestGetMemoryPressureTool:
    """Tests for get_memory_pressure_tool."""

    @pytest.mark.asyncio
    async def test_pressure_normal(self):
        """Test normal memory pressure."""
        mock_continuum = MagicMock()
        mock_continuum.get_memory_pressure.return_value = 0.3
        mock_continuum.get_stats.return_value = {
            "total_memories": 100,
            "by_tier": {"fast": 20, "medium": 60, "slow": 20},
        }

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            return_value=mock_continuum,
        ):
            result = await get_memory_pressure_tool()

        assert result["status"] == "normal"
        assert result["pressure"] == 0.3
        assert result["cleanup_recommended"] is False

    @pytest.mark.asyncio
    async def test_pressure_elevated(self):
        """Test elevated memory pressure."""
        mock_continuum = MagicMock()
        mock_continuum.get_memory_pressure.return_value = 0.65
        mock_continuum.get_stats.return_value = {"total_memories": 500}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            return_value=mock_continuum,
        ):
            result = await get_memory_pressure_tool()

        assert result["status"] == "elevated"

    @pytest.mark.asyncio
    async def test_pressure_high(self):
        """Test high memory pressure."""
        mock_continuum = MagicMock()
        mock_continuum.get_memory_pressure.return_value = 0.85
        mock_continuum.get_stats.return_value = {"total_memories": 800}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            return_value=mock_continuum,
        ):
            result = await get_memory_pressure_tool()

        assert result["status"] == "high"

    @pytest.mark.asyncio
    async def test_pressure_critical(self):
        """Test critical memory pressure."""
        mock_continuum = MagicMock()
        mock_continuum.get_memory_pressure.return_value = 0.95
        mock_continuum.get_stats.return_value = {"total_memories": 1000}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            return_value=mock_continuum,
        ):
            result = await get_memory_pressure_tool()

        assert result["status"] == "critical"
        assert result["cleanup_recommended"] is True

    @pytest.mark.asyncio
    async def test_pressure_import_error(self):
        """Test graceful handling when continuum not available."""
        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("Not installed"),
        ):
            result = await get_memory_pressure_tool()

        assert "error" in result
        assert "not available" in result["error"].lower()
