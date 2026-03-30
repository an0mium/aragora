"""Regression coverage for optional package import boundaries."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_protocols_a2a_import_survives_missing_httpx() -> None:
    """Importing A2A protocol types should not require the optional httpx client."""
    project_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib
        import sys

        class BlockHTTPX(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "httpx" or fullname.startswith("httpx."):
                    raise ImportError("httpx unavailable")
                return None

        sys.meta_path.insert(0, BlockHTTPX())

        module = importlib.import_module("aragora.protocols.a2a")
        assert module.TaskRequest is not None
        assert module.A2AServer is not None
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


def test_memory_import_survives_missing_aiohttp() -> None:
    """Importing the memory package should not require transport-backed embeddings."""
    project_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib
        import sys

        class BlockAIOHTTP(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "aiohttp" or fullname.startswith("aiohttp."):
                    raise ImportError("aiohttp unavailable")
                return None

        sys.meta_path.insert(0, BlockAIOHTTP())

        module = importlib.import_module("aragora.memory")
        assert module.CritiqueStore is not None
        assert module.SemanticRetriever is None
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
