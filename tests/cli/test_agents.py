"""
Tests for aragora.cli.agents module.

Tests agent listing CLI commands.
"""

from __future__ import annotations

import argparse
import os
from unittest.mock import patch

import pytest

from aragora.cli.agents import (
    AGENT_REGISTRY,
    get_agents_by_category,
    main,
    print_api_agents,
    print_cli_agents,
    print_demo_agents,
    print_local_agents,
    print_usage_hints,
)


# ===========================================================================
# Tests: AGENT_REGISTRY
# ===========================================================================


class TestAgentRegistry:
    """Tests for AGENT_REGISTRY constant."""

    def test_registry_is_list(self):
        """Test registry is a list."""
        assert isinstance(AGENT_REGISTRY, list)

    def test_registry_not_empty(self):
        """Test registry is not empty."""
        assert len(AGENT_REGISTRY) > 0

    def test_registry_has_required_fields(self):
        """Test each agent has required fields."""
        required_fields = {"type", "name", "category", "description"}
        for agent in AGENT_REGISTRY:
            assert required_fields.issubset(agent.keys()), f"Agent {agent} missing fields"

    def test_registry_has_demo_agent(self):
        """Test registry includes demo agent."""
        demo_agents = [a for a in AGENT_REGISTRY if a["type"] == "demo"]
        assert len(demo_agents) == 1
        assert demo_agents[0]["category"] == "demo"

    def test_registry_has_anthropic_agent(self):
        """Test registry includes Anthropic agent."""
        anthropic_agents = [a for a in AGENT_REGISTRY if a["type"] == "anthropic-api"]
        assert len(anthropic_agents) == 1
        assert anthropic_agents[0]["env_var"] == "ANTHROPIC_API_KEY"

    def test_registry_has_openai_agent(self):
        """Test registry includes OpenAI agent."""
        openai_agents = [a for a in AGENT_REGISTRY if a["type"] == "openai-api"]
        assert len(openai_agents) == 1
        assert openai_agents[0]["env_var"] == "OPENAI_API_KEY"


# ===========================================================================
# Tests: get_agents_by_category
# ===========================================================================


class TestGetAgentsByCategory:
    """Tests for get_agents_by_category function."""

    def test_returns_dict(self):
        """Test returns dictionary."""
        result = get_agents_by_category()
        assert isinstance(result, dict)

    def test_has_all_categories(self):
        """Test has all expected categories."""
        result = get_agents_by_category()
        assert "demo" in result
        assert "api" in result
        assert "local" in result
        assert "cli" in result

    def test_demo_category_contents(self):
        """Test demo category contains demo agents."""
        result = get_agents_by_category()
        assert len(result["demo"]) >= 1
        for agent in result["demo"]:
            assert agent["category"] == "demo"

    def test_api_category_contents(self):
        """Test API category contains API agents."""
        result = get_agents_by_category()
        assert len(result["api"]) >= 1
        for agent in result["api"]:
            assert agent["category"] == "api"

    def test_local_category_contents(self):
        """Test local category contains local agents."""
        result = get_agents_by_category()
        for agent in result["local"]:
            assert agent["category"] == "local"

    def test_cli_category_contents(self):
        """Test CLI category contains CLI agents."""
        result = get_agents_by_category()
        for agent in result["cli"]:
            assert agent["category"] == "cli"


# ===========================================================================
# Tests: print_demo_agents
# ===========================================================================


class TestPrintDemoAgents:
    """Tests for print_demo_agents function."""

    def test_prints_demo_agents(self, capsys):
        """Test printing demo agents."""
        agents = [
            {"type": "demo", "name": "Demo Agent", "requirement": "None", "description": "Test"}
        ]
        print_demo_agents(agents)

        captured = capsys.readouterr()
        assert "Demo Agents" in captured.out
        assert "demo" in captured.out

    def test_empty_list(self, capsys):
        """Test printing empty list does nothing."""
        print_demo_agents([])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_mode(self, capsys):
        """Test verbose mode includes description."""
        agents = [
            {
                "type": "demo",
                "name": "Demo Agent",
                "requirement": "None",
                "description": "Test desc",
            }
        ]
        print_demo_agents(agents, verbose=True)

        captured = capsys.readouterr()
        assert "Description: Test desc" in captured.out


# ===========================================================================
# Tests: print_api_agents
# ===========================================================================


class TestPrintApiAgents:
    """Tests for print_api_agents function."""

    def test_prints_api_agents_without_key(self, capsys, monkeypatch):
        """Test printing API agents without configured key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        agents = [
            {
                "type": "anthropic-api",
                "name": "Anthropic",
                "env_var": "ANTHROPIC_API_KEY",
                "description": "Claude",
                "fallback": "openrouter",
            }
        ]
        print_api_agents(agents)

        captured = capsys.readouterr()
        assert "API Agents" in captured.out
        assert "anthropic-api" in captured.out
        assert "needs API key" in captured.out

    def test_prints_api_agents_with_key(self, capsys, monkeypatch):
        """Test printing API agents with configured key."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        agents = [
            {
                "type": "anthropic-api",
                "name": "Anthropic",
                "env_var": "ANTHROPIC_API_KEY",
                "description": "Claude",
            }
        ]
        print_api_agents(agents)

        captured = capsys.readouterr()
        assert "configured" in captured.out

    def test_prints_fallback_available(self, capsys, monkeypatch):
        """Test printing fallback status when OpenRouter is available."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        agents = [
            {
                "type": "anthropic-api",
                "name": "Anthropic",
                "env_var": "ANTHROPIC_API_KEY",
                "description": "Claude",
                "fallback": "openrouter",
            }
        ]
        print_api_agents(agents)

        captured = capsys.readouterr()
        assert "via OpenRouter" in captured.out
        assert "fallback available" in captured.out

    def test_verbose_mode(self, capsys, monkeypatch):
        """Test verbose mode includes description."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        agents = [
            {
                "type": "anthropic-api",
                "name": "Anthropic",
                "env_var": "ANTHROPIC_API_KEY",
                "description": "Claude models",
            }
        ]
        print_api_agents(agents, verbose=True)

        captured = capsys.readouterr()
        assert "Description: Claude models" in captured.out


# ===========================================================================
# Tests: print_local_agents
# ===========================================================================


class TestPrintLocalAgents:
    """Tests for print_local_agents function."""

    def test_prints_local_agents(self, capsys):
        """Test printing local agents."""
        agents = [
            {
                "type": "ollama",
                "name": "Ollama",
                "requirement": "Ollama running",
                "description": "Local",
            }
        ]
        print_local_agents(agents)

        captured = capsys.readouterr()
        assert "Local Agents" in captured.out
        assert "ollama" in captured.out
        assert "Ollama running" in captured.out

    def test_verbose_mode(self, capsys):
        """Test verbose mode includes description."""
        agents = [
            {
                "type": "ollama",
                "name": "Ollama",
                "requirement": "None",
                "description": "Local models",
            }
        ]
        print_local_agents(agents, verbose=True)

        captured = capsys.readouterr()
        assert "Description: Local models" in captured.out


# ===========================================================================
# Tests: print_cli_agents
# ===========================================================================


class TestPrintCliAgents:
    """Tests for print_cli_agents function."""

    def test_prints_cli_agents_not_installed(self, capsys):
        """Test printing CLI agents when not installed."""
        with patch("shutil.which", return_value=None):
            agents = [{"type": "claude", "name": "Claude CLI", "description": "Claude CLI"}]
            print_cli_agents(agents)

        captured = capsys.readouterr()
        assert "CLI Agents" in captured.out
        assert "claude" in captured.out
        assert "not installed" in captured.out

    def test_prints_cli_agents_installed(self, capsys):
        """Test printing CLI agents when installed."""
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            agents = [{"type": "claude", "name": "Claude CLI", "description": "Claude CLI"}]
            print_cli_agents(agents)

        captured = capsys.readouterr()
        assert "/usr/local/bin/claude" in captured.out

    def test_verbose_mode(self, capsys):
        """Test verbose mode includes description."""
        with patch("shutil.which", return_value=None):
            agents = [{"type": "claude", "name": "Claude CLI", "description": "Claude via CLI"}]
            print_cli_agents(agents, verbose=True)

        captured = capsys.readouterr()
        assert "Description: Claude via CLI" in captured.out


# ===========================================================================
# Tests: print_usage_hints
# ===========================================================================


class TestPrintUsageHints:
    """Tests for print_usage_hints function."""

    def test_prints_usage(self, capsys):
        """Test printing usage hints."""
        print_usage_hints()

        captured = capsys.readouterr()
        assert "Usage:" in captured.out
        assert "aragora ask" in captured.out
        assert "ANTHROPIC_API_KEY" in captured.out
        assert "aragora doctor" in captured.out


# ===========================================================================
# Tests: main
# ===========================================================================


class TestMain:
    """Tests for main function."""

    def test_main_runs_without_error(self, capsys, monkeypatch):
        """Test main runs without error."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        args = argparse.Namespace()
        args.verbose = False

        with patch("shutil.which", return_value=None):
            main(args)

        captured = capsys.readouterr()
        assert "AVAILABLE AGENTS" in captured.out
        assert "API Agents" in captured.out
        assert "Local Agents" in captured.out
        assert "CLI Agents" in captured.out

    def test_main_verbose_mode(self, capsys, monkeypatch):
        """Test main with verbose flag."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        args = argparse.Namespace()
        args.verbose = True

        with patch("shutil.which", return_value=None):
            main(args)

        captured = capsys.readouterr()
        assert "Description:" in captured.out

    def test_main_shows_configured_agents(self, capsys, monkeypatch):
        """Test main shows configured status for agents with keys."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        args = argparse.Namespace()
        args.verbose = False

        with patch("shutil.which", return_value=None):
            main(args)

        captured = capsys.readouterr()
        assert "configured" in captured.out

    def test_main_default_verbose_false(self, capsys, monkeypatch):
        """Test main handles missing verbose attribute."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        args = argparse.Namespace()  # No verbose attribute

        with patch("shutil.which", return_value=None):
            main(args)

        captured = capsys.readouterr()
        assert "AVAILABLE AGENTS" in captured.out
