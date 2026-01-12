"""
CLI agents command - list available agents and their configuration.

Extracted from main.py for modularity.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Any


# Agent registry with metadata for CLI display
AGENT_REGISTRY: list[dict[str, Any]] = [
    # Built-in demo agent
    {
        "type": "demo",
        "name": "Demo Agent",
        "env_var": None,
        "category": "demo",
        "description": "Built-in deterministic agent for demos and CI",
        "requirement": "None",
    },
    # API-based agents (require API keys)
    {
        "type": "anthropic-api",
        "name": "Anthropic (Claude)",
        "env_var": "ANTHROPIC_API_KEY",
        "category": "api",
        "description": "Claude models via Anthropic API",
        "fallback": "openrouter",
    },
    {
        "type": "openai-api",
        "name": "OpenAI (GPT)",
        "env_var": "OPENAI_API_KEY",
        "category": "api",
        "description": "GPT-4 and GPT-3.5 models",
        "fallback": "openrouter",
    },
    {
        "type": "gemini",
        "name": "Google (Gemini)",
        "env_var": "GEMINI_API_KEY",
        "category": "api",
        "description": "Gemini Pro and Ultra models",
        "fallback": None,
    },
    {
        "type": "grok",
        "name": "xAI (Grok)",
        "env_var": "XAI_API_KEY",
        "category": "api",
        "description": "Grok models from xAI",
        "fallback": None,
    },
    {
        "type": "mistral",
        "name": "Mistral AI",
        "env_var": "MISTRAL_API_KEY",
        "category": "api",
        "description": "Mistral Large and Codestral",
        "fallback": "openrouter",
    },
    {
        "type": "deepseek",
        "name": "DeepSeek",
        "env_var": "DEEPSEEK_API_KEY",
        "category": "api",
        "description": "DeepSeek Coder and Chat",
        "fallback": "openrouter",
    },
    # Fallback/aggregator
    {
        "type": "openrouter",
        "name": "OpenRouter",
        "env_var": "OPENROUTER_API_KEY",
        "category": "api",
        "description": "Aggregator - access many models via single API",
        "fallback": None,
    },
    # Local agents (no API key needed)
    {
        "type": "ollama",
        "name": "Ollama (Local)",
        "env_var": None,
        "category": "local",
        "description": "Local models via Ollama",
        "requirement": "Ollama running locally",
    },
    {
        "type": "lm-studio",
        "name": "LM Studio (Local)",
        "env_var": None,
        "category": "local",
        "description": "Local models via LM Studio",
        "requirement": "LM Studio running locally",
    },
    # CLI-based agents
    {
        "type": "claude",
        "name": "Claude CLI",
        "env_var": None,
        "category": "cli",
        "description": "Claude via Claude Code CLI",
        "requirement": "claude CLI installed",
    },
    {
        "type": "codex",
        "name": "Codex CLI",
        "env_var": None,
        "category": "cli",
        "description": "OpenAI Codex via CLI",
        "requirement": "codex CLI installed",
    },
]


def get_agents_by_category() -> dict[str, list[dict]]:
    """Group agents by category."""
    return {
        "demo": [a for a in AGENT_REGISTRY if a["category"] == "demo"],
        "api": [a for a in AGENT_REGISTRY if a["category"] == "api"],
        "local": [a for a in AGENT_REGISTRY if a["category"] == "local"],
        "cli": [a for a in AGENT_REGISTRY if a["category"] == "cli"],
    }


def print_demo_agents(agents: list[dict], verbose: bool = False) -> None:
    """Print demo agents section."""
    if not agents:
        return
    print("\n🧪 Demo Agents (no API keys):")
    print("-" * 40)
    for agent in agents:
        print(f"\n  {agent['type']}")
        print(f"    Name: {agent['name']}")
        print(f"    Requirement: {agent.get('requirement', 'None')}")
        if verbose:
            print(f"    Description: {agent['description']}")


def print_api_agents(agents: list[dict], verbose: bool = False) -> None:
    """Print API agents section with status."""
    print("\n📡 API Agents (require API keys):")
    print("-" * 40)
    for agent in agents:
        env_var = agent["env_var"]
        has_key = bool(os.environ.get(env_var, "")) if env_var else False

        status = "✓ configured" if has_key else "✗ needs API key"
        fallback_note = ""
        if not has_key and agent.get("fallback"):
            fallback_key = "OPENROUTER_API_KEY"
            if os.environ.get(fallback_key, ""):
                status = "○ via OpenRouter"
                fallback_note = " (fallback available)"

        print(f"\n  {agent['type']}")
        print(f"    Name: {agent['name']}")
        print(f"    Status: {status}{fallback_note}")
        print(f"    Env: {env_var}")
        if verbose:
            print(f"    Description: {agent['description']}")


def print_local_agents(agents: list[dict], verbose: bool = False) -> None:
    """Print local agents section."""
    print("\n\n🖥️  Local Agents (no API key needed):")
    print("-" * 40)
    for agent in agents:
        print(f"\n  {agent['type']}")
        print(f"    Name: {agent['name']}")
        print(f"    Requirement: {agent.get('requirement', 'None')}")
        if verbose:
            print(f"    Description: {agent['description']}")


def print_cli_agents(agents: list[dict], verbose: bool = False) -> None:
    """Print CLI agents section with installation status."""
    print("\n\n🔧 CLI Agents (require CLI tools):")
    print("-" * 40)
    for agent in agents:
        cli_name = agent["type"]
        cli_path = shutil.which(cli_name)
        status = f"✓ {cli_path}" if cli_path else "✗ not installed"

        print(f"\n  {agent['type']}")
        print(f"    Name: {agent['name']}")
        print(f"    Status: {status}")
        if verbose:
            print(f"    Description: {agent['description']}")


def print_usage_hints() -> None:
    """Print usage examples."""
    print("\n" + "=" * 60)
    print("Usage:")
    print("  aragora ask \"Your question\" --demo")
    print("  aragora ask \"Your question\" --agents anthropic-api,openai-api")
    print("  aragora ask \"Your question\" --agents ollama,anthropic-api")
    print("\nTo configure API keys:")
    print("  export ANTHROPIC_API_KEY='your-key'")
    print("  export OPENAI_API_KEY='your-key'")
    print("\nRun 'aragora doctor' for full diagnostic.")


def main(args: argparse.Namespace) -> None:
    """Handle 'agents' command - list available agents and their configuration."""
    print("\n" + "=" * 60)
    print("AVAILABLE AGENTS")
    print("=" * 60)

    categories = get_agents_by_category()
    verbose = getattr(args, "verbose", False)

    print_demo_agents(categories["demo"], verbose)
    print_api_agents(categories["api"], verbose)
    print_local_agents(categories["local"], verbose)
    print_cli_agents(categories["cli"], verbose)
    print_usage_hints()


if __name__ == "__main__":
    # Allow running directly for testing
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    main(parser.parse_args())
