import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "benchmarks" / "bench_lara_routing.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bench_lara_routing", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_benchmark_rejects_empty_iteration_count(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "LaRARouter", lambda: pytest.fail("router should not load"))

    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        module.run_benchmark(0, 0, 5)


def test_run_benchmark_rejects_invalid_node_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "LaRARouter", lambda: pytest.fail("router should not load"))

    with pytest.raises(ValueError, match="min_nodes must be a non-negative integer"):
        module.run_benchmark(3, -1, 5)
    with pytest.raises(ValueError, match="min_nodes must be less than or equal to max_nodes"):
        module.run_benchmark(3, 10, 5)


def test_parse_args_rejects_invalid_cli_ranges() -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--iterations", "0"])
    with pytest.raises(SystemExit):
        module.parse_args(["--min-nodes", "-1"])
    with pytest.raises(SystemExit):
        module.parse_args(["--min-nodes", "10", "--max-nodes", "5"])


def test_run_benchmark_records_one_route_per_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()

    class _Router:
        def route(self, *_args, **_kwargs):
            return SimpleNamespace(route="mock-route")

    monkeypatch.setattr(module, "LaRARouter", _Router)

    result = module.run_benchmark(3, 0, 0)

    assert result.iterations == 3
    assert len(result.times_ms) == 3
    assert result.route_counts == {"mock-route": 3}
