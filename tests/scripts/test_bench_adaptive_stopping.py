import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "benchmarks" / "bench_adaptive_stopping.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bench_adaptive_stopping", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_benchmark_rejects_empty_iteration_count(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_get_detector", lambda: pytest.fail("detector should not load"))

    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        module.run_benchmark(0, 7)


def test_run_benchmark_rejects_empty_vote_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_get_detector", lambda: pytest.fail("detector should not load"))

    with pytest.raises(ValueError, match="votes_per_round must be a positive integer"):
        module.run_benchmark(3, 0)


def test_parse_args_rejects_non_positive_values() -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--iterations", "0"])
    with pytest.raises(SystemExit):
        module.parse_args(["--votes-per-round", "-1"])


def test_run_benchmark_records_one_score_per_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_get_detector", lambda: None)

    result = module.run_benchmark(3, 2)

    assert result.iterations == 3
    assert len(result.stability_scores) == 3
    assert len(result.times_ms) == 3
