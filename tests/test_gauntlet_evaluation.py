"""
Tests for the Gauntlet evaluation harness.
"""

from pathlib import Path

import pytest

from benchmarks.gauntlet_evaluation import load_fixtures, evaluate_fixture


@pytest.mark.asyncio
async def test_gauntlet_evaluation_fixture_counts():
    fixtures = load_fixtures(Path("benchmarks/fixtures/gauntlet"))
    fixture = next(f for f in fixtures if f.fixture_id == "auth-spec")
    result = await evaluate_fixture(fixture)

    assert result["found"]["critical"] == fixture.expected["critical"]
    assert result["found"]["total"] == fixture.expected["critical"]
    assert result["quality_score"] == pytest.approx(1.0)
