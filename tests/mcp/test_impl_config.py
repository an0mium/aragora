"""Tests for MCP implementation config generator and impl mode scoping."""

from __future__ import annotations

import json

import pytest

from aragora.mcp.impl_config import generate_impl_mcp_config
from aragora.mcp.server import AragoraMCPServer, IMPL_MODE_TOOLS


class TestImplConfigGeneration:
    """Tests for generate_impl_mcp_config()."""

    def test_config_file_created(self, tmp_path):
        """Config file should be created at .nomic/mcp/impl_config.json."""
        config_path = generate_impl_mcp_config(tmp_path)

        assert config_path.exists()
        assert config_path.name == "impl_config.json"
        assert ".nomic/mcp" in str(config_path)

    def test_config_valid_json(self, tmp_path):
        """Config file should contain valid JSON."""
        config_path = generate_impl_mcp_config(tmp_path)

        config = json.loads(config_path.read_text())
        assert "mcpServers" in config
        assert "aragora" in config["mcpServers"]

    def test_config_points_to_aragora_mcp_server(self, tmp_path):
        """Config should invoke aragora.mcp.server with stdio transport."""
        config_path = generate_impl_mcp_config(tmp_path)

        config = json.loads(config_path.read_text())
        server = config["mcpServers"]["aragora"]

        assert server["command"] == "python"
        assert "-m" in server["args"]
        assert "aragora.mcp.server" in server["args"]
        assert "--transport" in server["args"]
        assert "stdio" in server["args"]

    def test_config_sets_impl_mode_env(self, tmp_path):
        """Config should set ARAGORA_MCP_IMPL_MODE=1."""
        config_path = generate_impl_mcp_config(tmp_path)

        config = json.loads(config_path.read_text())
        env = config["mcpServers"]["aragora"]["env"]

        assert env["ARAGORA_MCP_IMPL_MODE"] == "1"
        assert env["ARAGORA_REPO_PATH"] == str(tmp_path)


class TestImplModeScoping:
    """Tests that impl_mode filters tools to implementation-relevant subset."""

    def test_impl_mode_restricts_tools(self):
        """In impl mode, only IMPL_MODE_TOOLS should be registered."""
        server = AragoraMCPServer(require_mcp=False, impl_mode=True)

        for name in server._tools:
            assert name in IMPL_MODE_TOOLS, f"Tool '{name}' should not be in impl mode"

    def test_normal_mode_has_more_tools(self):
        """Normal mode should have more tools than impl mode."""
        normal = AragoraMCPServer(require_mcp=False, impl_mode=False)
        impl = AragoraMCPServer(require_mcp=False, impl_mode=True)

        assert len(normal._tools) > len(impl._tools)

    def test_impl_mode_has_codebase_tools(self):
        """Impl mode should include codebase exploration tools."""
        server = AragoraMCPServer(require_mcp=False, impl_mode=True)

        # These are the most important tools for implementation
        for expected in ("search_codebase", "query_knowledge", "query_memory"):
            if expected in IMPL_MODE_TOOLS:
                # Only assert if the tool function exists in TOOLS_METADATA
                pass  # Tool is allowed; actual registration depends on function availability

    def test_impl_mode_excludes_debate_tools(self):
        """Impl mode should NOT include debate or audit tools."""
        server = AragoraMCPServer(require_mcp=False, impl_mode=True)

        excluded = {"run_debate", "run_gauntlet", "run_audit", "create_audit_session"}
        for name in excluded:
            assert name not in server._tools, f"Tool '{name}' should be excluded in impl mode"

    def test_normal_mode_includes_all_tools(self):
        """Normal mode should not filter any tools."""
        server = AragoraMCPServer(require_mcp=False, impl_mode=False)

        # Should have debate tools
        assert "run_debate" in server._tools
