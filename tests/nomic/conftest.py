"""
Pytest fixtures for Nomic loop tests.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_aragora_path(tmp_path: Path) -> Path:
    """Create a mock aragora project structure."""
    # Create basic directory structure
    (tmp_path / "aragora").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()

    # Create a minimal pyproject.toml
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "aragora"
version = "1.0.0"
"""
    )

    # Create a sample Python file
    (tmp_path / "aragora" / "__init__.py").write_text('"""Aragora package."""')
    (tmp_path / "aragora" / "core.py").write_text(
        '''
"""Core module."""

def example_function():
    """Example function."""
    return "hello"
'''
    )

    return tmp_path


@pytest.fixture
def mock_claude_agent() -> MagicMock:
    """Create a mock Claude agent."""
    agent = MagicMock()
    agent.name = "claude"
    agent.generate = AsyncMock(return_value="Mock Claude response")
    return agent


@pytest.fixture
def mock_codex_agent() -> MagicMock:
    """Create a mock Codex agent."""
    agent = MagicMock()
    agent.name = "codex"
    agent.generate = AsyncMock(return_value="Mock Codex response")
    return agent


@pytest.fixture
def mock_log_fn() -> MagicMock:
    """Create a mock logging function."""
    return MagicMock()


@pytest.fixture
def mock_stream_emit_fn() -> MagicMock:
    """Create a mock stream emit function."""
    return MagicMock()


@pytest.fixture
def mock_harness() -> MagicMock:
    """Create a mock agent harness."""
    harness = MagicMock()
    harness.explore_codebase = AsyncMock(
        return_value={
            "files": ["aragora/core.py"],
            "summary": "Mock codebase exploration",
            "features": ["feature1", "feature2"],
        }
    )
    harness.run_tests = AsyncMock(
        return_value={
            "passed": True,
            "failures": [],
            "output": "All tests passed",
        }
    )
    harness.generate_code = AsyncMock(
        return_value={
            "code": "def new_feature(): pass",
            "files_modified": ["aragora/new.py"],
        }
    )
    return harness


@pytest.fixture
def mock_debate_result() -> Dict[str, Any]:
    """Create a mock debate result."""
    return {
        "consensus": True,
        "confidence": 0.85,
        "final_claim": "We should implement feature X",
        "proposals": [
            {
                "agent": "claude",
                "proposal": "Add error handling",
                "votes": 3,
            },
            {
                "agent": "codex",
                "proposal": "Optimize performance",
                "votes": 2,
            },
        ],
        "votes": {
            "claude": "Add error handling",
            "codex": "Add error handling",
            "gemini": "Add error handling",
        },
    }


@pytest.fixture
def mock_design_result() -> Dict[str, Any]:
    """Create a mock design result."""
    return {
        "approved": True,
        "design": {
            "components": ["ErrorHandler", "RetryLogic"],
            "files_to_modify": ["aragora/errors.py", "aragora/retry.py"],
            "tests_required": ["test_error_handling.py"],
        },
        "safety_review": {
            "safe": True,
            "concerns": [],
        },
    }


@pytest.fixture
def mock_implementation_result() -> Dict[str, Any]:
    """Create a mock implementation result."""
    return {
        "success": True,
        "files_created": ["aragora/errors.py"],
        "files_modified": ["aragora/core.py"],
        "lines_added": 50,
        "lines_removed": 5,
        "code_changes": {
            "aragora/errors.py": "class ErrorHandler: pass",
        },
    }


@pytest.fixture
def mock_verification_result() -> Dict[str, Any]:
    """Create a mock verification result."""
    return {
        "passed": True,
        "test_results": {
            "total": 10,
            "passed": 10,
            "failed": 0,
            "skipped": 0,
        },
        "coverage": 85.5,
        "mypy_clean": True,
        "lint_clean": True,
    }


class MockNomicState:
    """Mock nomic state for testing."""

    def __init__(self):
        self.cycle_count = 1
        self.phase = "context"
        self.context = {}
        self.proposals = []
        self.design = {}
        self.implementation = {}
        self.verification = {}
        self.errors = []
        self.checkpoints = []


@pytest.fixture
def mock_nomic_state() -> MockNomicState:
    """Create a mock nomic state."""
    return MockNomicState()
