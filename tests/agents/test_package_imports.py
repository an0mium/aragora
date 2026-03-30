"""Regression coverage for package-level agent imports."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_aragora_agents_import_survives_missing_api_agent_deps() -> None:
    """Importing aragora.agents should degrade gracefully when API agents are unavailable."""
    project_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import builtins
        import importlib

        real_import = builtins.__import__

        def selective_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "aiohttp":
                raise ImportError("aiohttp unavailable")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = selective_import

        agents = importlib.import_module("aragora.agents")
        assert agents.ClaudeAgent is not None
        assert agents.OpenAIAPIAgent is None
        assert agents.AnthropicAPIAgent is None
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
