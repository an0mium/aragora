"""TET T1 example call-site wiring: auto_evidence_cycle → intent chain.

Verifies the ONE production call site added in phase T1: a successfully
posted evidence packet records an ``agent-app``/``settle_pr`` intent, gated
behind ``ARAGORA_TRAIL=1`` and non-fatal in every failure mode.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from aragora.trail.intent_chain import read_records, verify_chain


def _load_cycle() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "auto_evidence_cycle.py"
    spec = importlib.util.spec_from_file_location("auto_evidence_cycle_trail_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cycle = _load_cycle()


def _posted_run(record_trail: Any) -> dict[str, Any]:
    return cycle.run_cycle(
        list_prs=lambda: [{"number": 7, "isDraft": False, "statusCheckRollup": []}],
        fetch_packet=lambda pr: {
            "pr_number": pr,
            "status": "needs_model_review_quorum",
            "tier": 2,
            "counted_model_families": [],
        },
        run_collect=lambda pr, apply: {
            "ok": True,
            "counting_families": ["grok", "mistral"],
            "posted_families": ["grok", "mistral"],
        },
        run_reconciler=lambda: 0,
        record_trail=record_trail,
        apply=True,
        max_prs=1,
        max_scan=5,
        budget_seconds=60.0,
        breaker_threshold=3,
        log=lambda _line: None,
    )


def test_posted_evidence_records_trail_intent() -> None:
    seen: list[tuple[int, list[str]]] = []
    summary = _posted_run(lambda pr, families: seen.append((pr, families)))
    assert summary["posted_prs"] == [7]
    assert seen == [(7, ["grok", "mistral"])]


def test_default_record_trail_appends_when_enabled(tmp_path: Path, monkeypatch: Any) -> None:
    chain = tmp_path / "intent-chain.jsonl"
    monkeypatch.setenv("ARAGORA_TRAIL", "1")
    monkeypatch.setenv("ARAGORA_TRAIL_CHAIN", str(chain))
    cycle.default_record_trail("synaptent/aragora", 42, ["grok", "mistral"])
    records = read_records(chain)
    assert len(records) == 1
    assert records[0]["actor_class"] == "agent-app"
    assert records[0]["intent_type"] == "settle_pr"
    assert records[0]["target"] == {"repo": "synaptent/aragora", "pr": 42}
    assert records[0]["payload"]["posted_families"] == ["grok", "mistral"]
    ok, _ = verify_chain(chain)
    assert ok


def test_default_record_trail_is_noop_when_disabled(tmp_path: Path, monkeypatch: Any) -> None:
    chain = tmp_path / "intent-chain.jsonl"
    monkeypatch.delenv("ARAGORA_TRAIL", raising=False)
    monkeypatch.setenv("ARAGORA_TRAIL_CHAIN", str(chain))
    cycle.default_record_trail("synaptent/aragora", 42, ["grok"])
    assert not chain.exists()
