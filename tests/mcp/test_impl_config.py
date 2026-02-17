"""Tests for MCP implementation config generator."""

from __future__ import annotations

import json

import pytest

from aragora.mcp.impl_config import generate_impl_mcp_config


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
