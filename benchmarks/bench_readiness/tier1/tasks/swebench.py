"""SWE-bench Lite loader (problem-statement only).

Tier-1 does NOT execute patches — that requires Docker image builds per
repo. Instead we present the problem statement and let each system propose
a patch as text. The LLM-judge then compares the proposed patch to the
reference patch for semantic equivalence. This is a weaker signal than
actual test execution but viable for a first-pass benchmark.

A follow-up Tier-2 run should swap this for the full SWE-bench harness.
"""

from __future__ import annotations

from collections.abc import Iterable

from benchmarks.bench_readiness.tier1.tasks.base import TaskItem


def load(limit: int, seed: int = 42) -> Iterable[TaskItem]:
    """Yield up to ``limit`` SWE-bench Lite problems."""
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    ds = ds.shuffle(seed=seed)

    for i, row in enumerate(ds):
        if i >= limit:
            break

        prompt = (
            f"Repository: {row['repo']}\n"
            f"Issue:\n{row['problem_statement']}\n\n"
            "Propose a minimal patch in unified diff format that would "
            "resolve this issue. Explain your reasoning briefly before the "
            "diff."
        )

        yield TaskItem(
            task_id=f"swebench-{row['instance_id']}",
            domain="swebench_lite",
            prompt=prompt,
            context="",
            reference_answer=row["patch"],
            eval_strategy="llm_judge",
            metadata={
                "repo": row["repo"],
                "instance_id": row["instance_id"],
                "base_commit": row.get("base_commit", ""),
            },
        )
