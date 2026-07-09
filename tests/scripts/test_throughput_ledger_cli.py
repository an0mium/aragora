"""Snapshot locking + dedupe for scripts/throughput_ledger.py (#9048 openai [P2])."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "throughput_ledger.py"


@pytest.fixture()
def mod():
    spec = importlib.util.spec_from_file_location("throughput_ledger_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_prs(mod, monkeypatch):
    monkeypatch.setattr(
        mod,
        "_gh_merged_prs",
        lambda limit, *, repo_root=".": [
            {
                "number": 42,
                "title": "feat: x",
                "mergedAt": "2026-07-09T00:00:00Z",
                "labels": [],
                "files": [{"path": "aragora/debate/a.py", "additions": 5, "deletions": 1}],
            }
        ],
    )


def test_snapshot_dedupes_across_runs_and_takes_lock(mod, monkeypatch, tmp_path):
    _fake_prs(mod, monkeypatch)
    for _ in range(2):
        rc = mod.main(["--repo-root", str(tmp_path), "snapshot", "--limit", "5"])
        assert rc == 0
    from aragora.nomic.throughput import ThroughputLedger

    ledger = ThroughputLedger(tmp_path)
    merges = [r for r in ledger.records() if r.kind == "merge"]
    assert len(merges) == 1  # second run deduped under the lock
    assert ledger.path.with_suffix(ledger.path.suffix + ".lock").exists()
