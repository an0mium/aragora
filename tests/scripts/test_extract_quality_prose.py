"""Tests for scripts/extract_quality_prose.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "extract_quality_prose.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("extract_quality_prose", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _required_args() -> list[str]:
    return ["--input", "conversations.json", "--output", "quality.json"]


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--min-words", "0"),
        ("--min-words", "-1"),
        ("--top-n", "0"),
        ("--top-n", "-1"),
        ("--min-quality", "-0.1"),
    ],
)
def test_parse_args_rejects_invalid_quality_filter_values(flag: str, value: str) -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args([*_required_args(), flag, value])


def test_parse_args_rejects_conflicting_role_filters() -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args([*_required_args(), "--user-only", "--assistant-only"])


def test_parse_args_accepts_valid_quality_filters() -> None:
    module = _load_module()

    args = module.parse_args(
        [
            *_required_args(),
            "--min-quality",
            "0",
            "--min-words",
            "50",
            "--top-n",
            "3",
            "--user-only",
        ]
    )

    assert args.min_quality == 0.0
    assert args.min_words == 50
    assert args.top_n == 3
    assert args.user_only is True
    assert args.assistant_only is False
