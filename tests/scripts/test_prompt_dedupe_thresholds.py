"""Regression tests for prompt extraction dedupe similarity thresholds."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(name: str):
    script_path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULES = [
    ("extract_and_rank_prompts", "threshold"),
    ("extract_intent_prompts", "threshold"),
    ("extract_user_prompts", "similarity_threshold"),
]


def _prompts() -> list[dict[str, str]]:
    return [
        {"id": "first", "text": "build a durable prompt ranking similarity surface"},
        {"id": "duplicate", "text": "build a durable prompt ranking similarity surface"},
        {"id": "distinct", "text": "measure proof receipt freshness without queue authority"},
    ]


@pytest.mark.parametrize(("module_name", "threshold_kw"), MODULES)
@pytest.mark.parametrize("invalid_threshold", [-0.01, 1.0, True, "0.5"])
def test_deduplicate_rejects_invalid_similarity_thresholds(
    module_name: str,
    threshold_kw: str,
    invalid_threshold: object,
) -> None:
    module = _load_script_module(module_name)

    with pytest.raises(ValueError, match="between 0.0 inclusive and 1.0 exclusive"):
        module.deduplicate(_prompts(), **{threshold_kw: invalid_threshold})


@pytest.mark.parametrize(("module_name", "threshold_kw"), MODULES)
def test_deduplicate_removes_exact_duplicates_with_valid_threshold(
    module_name: str,
    threshold_kw: str,
) -> None:
    module = _load_script_module(module_name)

    result = module.deduplicate(_prompts(), **{threshold_kw: 0.99})

    assert [item["id"] for item in result] == ["first", "distinct"]
