"""Pytest configuration and fixtures for aragora tests."""

import sys
from pathlib import Path

import pytest

# Add aragora to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Project root for test files
PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def sample_python_file(project_root: Path) -> Path:
    """Return a sample Python file from the project."""
    return project_root / "aragora" / "core.py"


@pytest.fixture
def sample_markdown_file(project_root: Path) -> Path:
    """Return a sample Markdown file from the project."""
    return project_root / "CLAUDE.md"


@pytest.fixture
def sample_files(project_root: Path) -> list[Path]:
    """Return a list of sample files for batch testing."""
    files = [
        project_root / "aragora" / "core.py",
        project_root / "CLAUDE.md",
        project_root / "aragora" / "debate" / "orchestrator.py",
        project_root / "aragora" / "agents" / "cli_agents.py",
    ]
    return [f for f in files if f.exists()]


@pytest.fixture
def temp_upload_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for uploads."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@pytest.fixture
def sample_text_content() -> str:
    """Return sample text content for testing."""
    return """
    # Sample Document

    This is a sample document for testing the document auditing system.

    ## Section 1: Introduction

    The effective date is 01/15/2024. This document expires on 12/31/2025.

    ## Section 2: Pricing

    The price for the service is $1,500 per month.
    The maximum limit is 10,000 requests.

    ## Section 3: Definitions

    "Service" means the cloud computing platform provided by the Company.
    "User" refers to any individual who accesses the Service.

    ## Section 4: SLA

    Uptime: 99.9%
    Response timeout: 500 ms

    Version: 2.1.0
    Last updated: 01/10/2024
    """


@pytest.fixture
def inconsistent_documents() -> list[dict]:
    """Return documents with intentional inconsistencies for testing."""
    return [
        {
            "id": "doc1",
            "content": """
            Contract Terms v1.0

            Effective date: 01/15/2024
            Price: $1,500 per month
            Maximum users: 100
            Uptime SLA: 99.9%
            """,
        },
        {
            "id": "doc2",
            "content": """
            Service Agreement v1.0

            Start date: 02/01/2024
            Cost: $2,000 per month
            User limit: 50
            Availability: 99.5%
            """,
        },
    ]
