"""Regression tests for the standalone playground endpoint catalog.

These checks keep the frontend REST playground aligned with the actual FastAPI
route table so the public demo does not advertise 404/422 examples.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from aragora.server.fastapi import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAYGROUND_CATALOG_FILES = (
    PROJECT_ROOT / "aragora/live/src/components/playground/EndpointSelector.tsx",
    PROJECT_ROOT / "aragora/live/src/components/playground/ApiPlayground.tsx",
)

_PATH_PATTERN = re.compile(r"path:\s*'([^']+)'")
_CREATE_DEBATE_BODY_PATTERN = re.compile(
    r"method:\s*'POST',\s*path:\s*'/api/v2/debates',.*?body:\s*\{(?P<body>.*?)\n\s*\},",
    re.DOTALL,
)


def _read_catalog(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app():
    """Create the FastAPI app once for route-table inspection."""
    return create_app()


def test_playground_catalog_paths_exist_in_fastapi_app(app):
    """Every advertised REST playground path should resolve to a real route."""
    registered_paths = {route.path for route in app.routes if getattr(route, "path", None)}
    missing_by_file: dict[str, list[str]] = {}

    for catalog_path in PLAYGROUND_CATALOG_FILES:
        advertised_paths = _PATH_PATTERN.findall(_read_catalog(catalog_path))
        missing = sorted(path for path in advertised_paths if path not in registered_paths)
        if missing:
            missing_by_file[catalog_path.name] = missing

    assert missing_by_file == {}


def test_playground_create_debate_examples_use_question_field():
    """POST /api/v2/debates examples should match FastAPI's request schema."""
    for catalog_path in PLAYGROUND_CATALOG_FILES:
        content = _read_catalog(catalog_path)
        match = _CREATE_DEBATE_BODY_PATTERN.search(content)
        assert match is not None, (
            f"Could not find POST /api/v2/debates example in {catalog_path.name}"
        )

        body = match.group("body")
        assert "question:" in body, (
            f"{catalog_path.name} should send `question` for debate creation"
        )
        assert "task:" not in body, f"{catalog_path.name} should not send deprecated `task`"
