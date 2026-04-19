from __future__ import annotations

import builtins
import sys
import types

from benchmarks.bench_readiness.tier1 import judge, runner
from benchmarks.bench_readiness.tier1.systems import SystemOutput
from benchmarks.bench_readiness.tier1.tasks import aragora_custom, legal, mmlu_pro, swebench
from benchmarks.bench_readiness.tier1.tasks.base import TaskItem


def test_legal_loader_respects_limit() -> None:
    items = list(legal.load(limit=2))

    assert len(items) == 2
    assert all(item.domain == "legal" for item in items)
    assert items[0].task_id.startswith("legal-")


def test_aragora_custom_loader_respects_limit() -> None:
    items = list(aragora_custom.load(limit=3))

    assert len(items) == 3
    assert all(item.domain == "aragora_custom" for item in items)
    assert items[0].task_id.startswith("aragora-")


def test_extract_final_letter_and_exact_match() -> None:
    task = TaskItem(
        task_id="mmlu-1",
        domain="mmlu_pro",
        prompt="Question",
        reference_answer="B",
        eval_strategy="exact_match",
    )
    solo = SystemOutput(system="solo", answer="Reasoning\nAnswer: B", latency_sec=1.0)
    debate = SystemOutput(system="debate", answer="Reasoning\nAnswer: C", latency_sec=1.0)

    verdict = judge.judge(task, solo, debate, api_key="unused", seed=0)

    assert verdict.exact_match_used is True
    assert verdict.winner_system == "solo"
    assert verdict.scores_a["correctness"] == 10
    assert verdict.scores_b["correctness"] == 0


def test_runner_summarize_groups_rows_by_domain() -> None:
    summary = runner._summarize(
        [
            {
                "domain": "legal",
                "solo_system": "solo",
                "debate_system": "debate",
                "winner_system": "solo",
                "judge_error": "",
                "solo_error": "",
                "debate_error": "",
                "solo_latency_sec": 1.0,
                "debate_latency_sec": 2.0,
            },
            {
                "domain": "legal",
                "solo_system": "solo",
                "debate_system": "debate",
                "winner_system": "TIE",
                "judge_error": "",
                "solo_error": "",
                "debate_error": "",
                "solo_latency_sec": 1.5,
                "debate_latency_sec": 2.5,
            },
        ]
    )

    assert "# Tier-1 Benchmark Summary" in summary
    assert "## legal  (n=2)" in summary
    assert "Solo (solo) wins: **1**" in summary
    assert "Ties: **1**" in summary


def test_systems_module_keeps_provider_imports_lazy(monkeypatch) -> None:
    for name in (
        "benchmarks.bench_readiness.tier1.systems",
        "benchmarks.bench_readiness.tier1.systems.solo_opus",
        "benchmarks.bench_readiness.tier1.systems.aragora_debate",
    ):
        sys.modules.pop(name, None)

    imported: list[str] = []
    real_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {
            "benchmarks.bench_readiness.tier1.systems.solo_opus",
            "benchmarks.bench_readiness.tier1.systems.aragora_debate",
        }:
            imported.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    import benchmarks.bench_readiness.tier1.systems as systems

    assert imported == []
    assert hasattr(systems, "run_solo_opus")
    assert hasattr(systems, "run_aragora_debate")


def test_mmlu_loader_uses_dataset_rows(monkeypatch) -> None:
    class FakeDataset:
        def __init__(self, rows):
            self._rows = list(rows)

        def filter(self, predicate):
            return FakeDataset([row for row in self._rows if predicate(row)])

        def shuffle(self, seed):
            assert seed == 7
            return self

        def __iter__(self):
            return iter(self._rows)

    fake_module = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: FakeDataset(
            [
                {
                    "question_id": "q1",
                    "category": "law",
                    "question": "What is law?",
                    "options": ["A1", "B1", "C1"],
                    "answer_index": 1,
                    "src": "test",
                }
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    items = list(mmlu_pro.load(limit=1, seed=7))

    assert len(items) == 1
    assert items[0].domain == "mmlu_pro"
    assert items[0].reference_answer == "B"


def test_swebench_loader_uses_dataset_rows(monkeypatch) -> None:
    class FakeDataset:
        def __init__(self, rows):
            self._rows = list(rows)

        def shuffle(self, seed):
            assert seed == 11
            return self

        def __iter__(self):
            return iter(self._rows)

    fake_module = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: FakeDataset(
            [
                {
                    "instance_id": "inst1",
                    "repo": "owner/repo",
                    "problem_statement": "Bug details",
                    "patch": "diff --git a/x b/x",
                    "base_commit": "abc123",
                }
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    items = list(swebench.load(limit=1, seed=11))

    assert len(items) == 1
    assert items[0].domain == "swebench_lite"
    assert items[0].metadata["repo"] == "owner/repo"
