import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "debate_quality_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("debate_quality_benchmark", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_benchmark_rejects_empty_prompt_collection() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="requires at least one prompt"):
        asyncio.run(module.run_benchmark([], dry_run=True))


def test_select_prompts_zero_preserves_all_prompts_sentinel() -> None:
    module = _load_module()

    assert module.parse_args(["--prompts", "0"]).prompts == 0
    assert module.select_prompts(0) == module.PROMPTS


def test_prompt_limit_rejects_negative_values() -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--prompts", "-1"])
    with pytest.raises(ValueError, match="non-negative integer"):
        module.select_prompts(-1)


def test_select_prompts_positive_limit_returns_prefix() -> None:
    module = _load_module()

    assert module.select_prompts(2) == module.PROMPTS[:2]
