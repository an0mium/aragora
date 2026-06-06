from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_benchmark_regression.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("check_benchmark_regression", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["check_benchmark_regression"] = module
    spec.loader.exec_module(module)
    return module


bench_mod = _load_module()


def _write_benchmarks(path: Path, means: dict[str, float]) -> Path:
    payload = {
        "benchmarks": [
            {
                "name": name,
                "stats": {
                    "mean": mean,
                },
            }
            for name, mean in means.items()
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compare_fails_when_current_and_baseline_share_no_benchmark_names(
    tmp_path: Path,
    capsys,
) -> None:
    current = _write_benchmarks(tmp_path / "current.json", {"bench_new": 0.001})
    baseline = _write_benchmarks(tmp_path / "baseline.json", {"bench_old": 0.001})

    rc = bench_mod.compare_benchmarks(current, baseline, threshold_pct=20.0)

    captured = capsys.readouterr()
    assert rc == 1
    assert "No shared benchmark names" in captured.out
    assert "bench_new" in captured.out
    assert "bench_old" in captured.out


def test_compare_allows_partial_overlap_without_regression(tmp_path: Path, capsys) -> None:
    current = _write_benchmarks(
        tmp_path / "current.json",
        {
            "bench_shared": 0.0011,
            "bench_current_only": 0.004,
        },
    )
    baseline = _write_benchmarks(
        tmp_path / "baseline.json",
        {
            "bench_shared": 0.0010,
            "bench_baseline_only": 0.004,
        },
    )

    rc = bench_mod.compare_benchmarks(current, baseline, threshold_pct=20.0)

    captured = capsys.readouterr()
    assert rc == 0
    assert "PASSED: 1 benchmark(s)" in captured.out
    assert "No shared benchmark names" not in captured.out
