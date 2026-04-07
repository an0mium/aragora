"""Tests for essay rubric scoring and YAML loading — written FIRST (TDD)."""

from __future__ import annotations

import json
import os
import tempfile
import textwrap

import pytest
import yaml

from aragora.essay.rubric import (
    EssayScore,
    _DEFAULT_WEIGHTS,
    load_rubric,
    parse_score_response,
)


# ---------------------------------------------------------------------------
# 1. EssayScore: weighted composite
# ---------------------------------------------------------------------------


def test_essay_score_overall_is_weighted_composite():
    """Create a score with known values, verify overall is the weighted sum."""
    score = EssayScore(
        thesis_clarity=0.8,
        argument_coherence=0.7,
        evidence_grounding=0.6,
        rhetorical_force=0.9,
        concision=0.5,
        factual_accuracy=1.0,
        originality=0.4,
    )
    # Compute expected from default weights
    expected = (
        0.8 * 0.20 + 0.7 * 0.20 + 0.6 * 0.15 + 0.9 * 0.15 + 0.5 * 0.10 + 1.0 * 0.10 + 0.4 * 0.10
    )
    assert abs(score.overall - expected) < 1e-9


# ---------------------------------------------------------------------------
# 2. EssayScore: defaults
# ---------------------------------------------------------------------------


def test_essay_score_defaults():
    """An empty score has 0.0 overall and empty lists."""
    score = EssayScore()
    assert score.overall == 0.0
    assert score.severity_notes == []
    assert score.suggestions == []
    assert score.factual_claims_to_verify == []
    assert score.weakest_paragraph == ""
    assert score.strongest_paragraph == ""


# ---------------------------------------------------------------------------
# 3. parse_score_response: extracts JSON
# ---------------------------------------------------------------------------


def test_parse_score_response_extracts_json():
    """JSON embedded in free-form LLM text is parsed correctly."""
    payload = {
        "thesis_clarity": 0.9,
        "argument_coherence": 0.85,
        "evidence_grounding": 0.7,
        "rhetorical_force": 0.8,
        "concision": 0.6,
        "factual_accuracy": 0.95,
        "originality": 0.75,
        "severity_notes": ["Weak transition in para 3"],
        "suggestions": ["Add a counter-argument"],
        "weakest_paragraph": "Paragraph 3",
        "strongest_paragraph": "Paragraph 1",
        "factual_claims_to_verify": ["GDP grew 5% in 2024"],
    }
    text = f"Here is my evaluation:\n```json\n{json.dumps(payload)}\n```\nDone."
    score = parse_score_response(text)

    assert score.thesis_clarity == 0.9
    assert score.argument_coherence == 0.85
    assert score.severity_notes == ["Weak transition in para 3"]
    assert score.suggestions == ["Add a counter-argument"]
    assert score.weakest_paragraph == "Paragraph 3"
    assert score.factual_claims_to_verify == ["GDP grew 5% in 2024"]
    # overall should be computed with default weights
    assert score.overall > 0


# ---------------------------------------------------------------------------
# 4. parse_score_response: handles missing fields
# ---------------------------------------------------------------------------


def test_parse_score_response_handles_missing_fields():
    """Partial JSON fills defaults for missing dimensions."""
    text = '{"thesis_clarity": 0.5, "originality": 0.3}'
    score = parse_score_response(text)

    assert score.thesis_clarity == 0.5
    assert score.originality == 0.3
    # Missing dimensions default to 0.0
    assert score.argument_coherence == 0.0
    assert score.evidence_grounding == 0.0
    assert score.severity_notes == []
    assert score.overall > 0  # at least thesis_clarity and originality contribute


# ---------------------------------------------------------------------------
# 5. parse_score_response: normalizes malformed numeric fields
# ---------------------------------------------------------------------------


def test_parse_score_response_normalizes_malformed_numeric_fields():
    """Malformed numeric fields fail closed to 0.0 instead of raising."""
    text = json.dumps(
        {
            "thesis_clarity": "not-a-number",
            "argument_coherence": None,
            "evidence_grounding": {"bad": "shape"},
            "rhetorical_force": [],
            "concision": "NaN",
            "factual_accuracy": "Infinity",
            "originality": 0.75,
        }
    )

    score = parse_score_response(text)

    assert score.thesis_clarity == 0.0
    assert score.argument_coherence == 0.0
    assert score.evidence_grounding == 0.0
    assert score.rhetorical_force == 0.0
    assert score.concision == 0.0
    assert score.factual_accuracy == 0.0
    assert score.originality == 0.75
    assert score.overall == pytest.approx(0.075)


# ---------------------------------------------------------------------------
# 6. parse_score_response: scalar strings stay whole in list fields
# ---------------------------------------------------------------------------


def test_parse_score_response_keeps_scalar_strings_for_list_fields():
    """Scalar strings for list-like fields are wrapped, not split into chars."""
    text = json.dumps(
        {
            "severity_notes": "Needs stronger evidence in the middle section",
            "suggestions": "Add a counter-argument before the conclusion",
            "factual_claims_to_verify": "The 2024 GDP figure needs a citation",
        }
    )

    score = parse_score_response(text)

    assert score.severity_notes == ["Needs stronger evidence in the middle section"]
    assert score.suggestions == ["Add a counter-argument before the conclusion"]
    assert score.factual_claims_to_verify == ["The 2024 GDP figure needs a citation"]


# ---------------------------------------------------------------------------
# 7. load_rubric: custom YAML
# ---------------------------------------------------------------------------


def test_load_rubric_from_yaml(tmp_path):
    """A custom YAML file loads correctly and weights sum to 1.0."""
    rubric_data = {
        "name": "test_rubric",
        "quality_threshold": 0.7,
        "weights": {
            "thesis_clarity": 0.15,
            "argument_coherence": 0.15,
            "evidence_grounding": 0.15,
            "rhetorical_force": 0.15,
            "concision": 0.15,
            "factual_accuracy": 0.15,
            "originality": 0.10,
        },
    }
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.dump(rubric_data))

    rubric = load_rubric(str(path))

    assert rubric["name"] == "test_rubric"
    assert rubric["quality_threshold"] == 0.7
    assert abs(sum(rubric["weights"].values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 8. load_default_rubric
# ---------------------------------------------------------------------------


def test_load_default_rubric():
    """No-arg load_rubric returns the default rubric with weights and name."""
    rubric = load_rubric()

    assert "name" in rubric
    assert "weights" in rubric
    assert "quality_threshold" in rubric
    assert isinstance(rubric["weights"], dict)
    assert len(rubric["weights"]) == 7
    assert abs(sum(rubric["weights"].values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 9. EssayScore.to_dict
# ---------------------------------------------------------------------------


def test_essay_score_to_dict():
    """to_dict returns a JSON-serializable dictionary."""
    score = EssayScore(
        thesis_clarity=0.8,
        argument_coherence=0.7,
        severity_notes=["note1"],
        suggestions=["suggestion1"],
    )
    d = score.to_dict()

    # Should be JSON-serializable
    serialized = json.dumps(d)
    assert isinstance(serialized, str)

    assert d["thesis_clarity"] == 0.8
    assert d["argument_coherence"] == 0.7
    assert d["severity_notes"] == ["note1"]
    assert d["overall"] == score.overall


# ---------------------------------------------------------------------------
# 10. EssayScore.compute_overall with custom weights
# ---------------------------------------------------------------------------


def test_essay_score_compute_overall_with_custom_weights():
    """compute_overall recalculates using provided weights."""
    score = EssayScore(
        thesis_clarity=1.0,
        argument_coherence=0.0,
        evidence_grounding=0.0,
        rhetorical_force=0.0,
        concision=0.0,
        factual_accuracy=0.0,
        originality=0.0,
    )
    custom_weights = {
        "thesis_clarity": 1.0,
        "argument_coherence": 0.0,
        "evidence_grounding": 0.0,
        "rhetorical_force": 0.0,
        "concision": 0.0,
        "factual_accuracy": 0.0,
        "originality": 0.0,
    }
    score.compute_overall(custom_weights)
    assert score.overall == 1.0


# ---------------------------------------------------------------------------
# 11. parse_score_response: returns empty on garbage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "This is not JSON at all!",
        "Model output: {not valid json}",
        "[]",
    ],
)
def test_parse_score_response_returns_empty_on_garbage(text):
    """Garbage input returns the existing empty EssayScore fallback."""
    assert parse_score_response(text) == EssayScore()
