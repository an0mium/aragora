"""Temporary file and directory fixtures (pytest plugin)."""

import json
import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Create a temporary SQLite database file.

    Yields the path to a temporary .db file that is automatically
    cleaned up after the test completes.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory.

    Yields a Path to a temporary directory that is automatically
    cleaned up after the test completes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_nomic_dir() -> Generator[Path, None, None]:
    """Create a temporary nomic directory with state files.

    Creates a directory structure mimicking the nomic system:
    - nomic_state.json: Current nomic state
    - nomic_loop.log: Recent log entries

    Yields a Path to the directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        nomic_dir = Path(tmpdir)

        # Create nomic state file
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(
            json.dumps(
                {
                    "phase": "implement",
                    "stage": "executing",
                    "cycle": 1,
                    "total_tasks": 5,
                    "completed_tasks": 2,
                }
            )
        )

        # Create nomic log file
        log_file = nomic_dir / "nomic_loop.log"
        log_file.write_text(
            "\n".join(
                [
                    "2026-01-05 00:00:01 Starting cycle 1",
                    "2026-01-05 00:00:02 Phase: context",
                    "2026-01-05 00:00:03 Phase: debate",
                    "2026-01-05 00:00:04 Phase: design",
                    "2026-01-05 00:00:05 Phase: implement",
                ]
            )
        )

        yield nomic_dir
