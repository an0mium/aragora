"""MMLU-Pro hard subset loader.

Uses the public TIGER-Lab/MMLU-Pro HuggingFace dataset. Samples only from
the harder categories where frontier models still disagree (law,
formal_logic, math, physics, economics). Exact-match primary + LLM-judge
secondary for reasoning quality.
"""

from __future__ import annotations

from collections.abc import Iterable

from benchmarks.bench_readiness.tier1.tasks.base import TaskItem

_HARD_CATEGORIES = ("law", "math", "physics", "economics", "philosophy")


def load(limit: int, seed: int = 42) -> Iterable[TaskItem]:
    """Yield up to ``limit`` MMLU-Pro items from hard categories.

    The dataset is streamed, filtered, shuffled deterministically, then
    truncated to ``limit``.
    """
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds = ds.filter(lambda r: r["category"] in _HARD_CATEGORIES)
    ds = ds.shuffle(seed=seed)

    for i, row in enumerate(ds):
        if i >= limit:
            break

        options = row["options"]
        lettered = "\n".join(f"({chr(ord('A') + j)}) {opt}" for j, opt in enumerate(options))
        prompt = (
            f"{row['question']}\n\n"
            f"Options:\n{lettered}\n\n"
            "Think step-by-step, then give your final answer on its own line "
            "as 'Answer: X' where X is a single letter."
        )

        answer_letter = chr(ord("A") + int(row["answer_index"]))

        yield TaskItem(
            task_id=f"mmlu_pro-{row['question_id']}",
            domain="mmlu_pro",
            prompt=prompt,
            context="",
            reference_answer=answer_letter,
            eval_strategy="exact_match",
            metadata={
                "category": row["category"],
                "src": row.get("src", ""),
                "option_count": len(options),
            },
        )
