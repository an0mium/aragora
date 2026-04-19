"""TaskItem — the uniform interface across all four benchmark domains.

A TaskItem carries everything a system-under-test needs to answer a question
plus everything a judge needs to score the answer. Keep this surface small
so new domains only have to implement a loader that yields TaskItems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskItem:
    """A single benchmarked item.

    Attributes
    ----------
    task_id:
        Stable identifier for this item. Used as the CSV row key.
    domain:
        One of ``mmlu_pro``, ``swebench_lite``, ``legal``, ``aragora_custom``.
    prompt:
        The question handed verbatim to both the solo and debate systems.
    context:
        Optional background text. Legal contracts go here; reasoning tasks
        typically leave it empty.
    reference_answer:
        The ground-truth answer when one exists (MMLU-Pro, SWE-bench). Empty
        string when the task is subjective (legal, custom).
    eval_strategy:
        ``exact_match`` | ``test_based`` | ``llm_judge``. Determines whether
        primary scoring uses the reference answer, runs tests, or defers to
        the LLM judge.
    metadata:
        Free-form per-domain metadata (e.g. category for MMLU, repo/issue for
        SWE-bench). Persisted to CSV under a single ``metadata_json`` column.
    """

    task_id: str
    domain: str
    prompt: str
    context: str = ""
    reference_answer: str = ""
    eval_strategy: str = "llm_judge"
    metadata: dict[str, Any] = field(default_factory=dict)
