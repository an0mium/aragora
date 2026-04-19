"""Solo Opus 4.7 baseline — single model call, no tools, no agent loop."""

from __future__ import annotations

import time

from benchmarks.bench_readiness.tier1.systems.base import SystemOutput
from benchmarks.bench_readiness.tier1.tasks.base import TaskItem

SYSTEM_PROMPT = (
    "You are a careful expert analyst. Think step-by-step. When the question "
    "has a clear correct answer, give it. When the question is open-ended, "
    "take a defensible position and justify it concisely."
)


def run_solo_opus(
    task: TaskItem,
    *,
    api_key: str,
    model: str = "claude-opus-4-7",
    max_tokens: int = 2048,
) -> SystemOutput:
    """Run a single Opus 4.7 completion for the given task.

    Returns a :class:`SystemOutput` whose ``answer`` is the model's raw text.
    On any exception, ``error`` is populated and ``answer`` is empty.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    user_content = task.prompt
    if task.context:
        user_content = f"Context:\n{task.context}\n\nQuestion:\n{task.prompt}"

    t0 = time.perf_counter()
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        latency = time.perf_counter() - t0

        # Best-effort text extraction
        parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        answer = "\n".join(parts).strip()

        usage = getattr(msg, "usage", None)
        tokens_in = getattr(usage, "input_tokens", 0) or 0
        tokens_out = getattr(usage, "output_tokens", 0) or 0

        # Opus 4.7 pricing (as of 2026-04): $15/M input, $75/M output
        cost = (tokens_in * 15 + tokens_out * 75) / 1_000_000

        return SystemOutput(
            system=f"solo_opus_{model}",
            answer=answer,
            latency_sec=latency,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            raw={"stop_reason": getattr(msg, "stop_reason", "")},
        )
    except Exception as e:  # noqa: BLE001 - provider SDKs can fail in many ways at runtime
        latency = time.perf_counter() - t0
        return SystemOutput(
            system=f"solo_opus_{model}",
            answer="",
            latency_sec=latency,
            error=f"{type(e).__name__}: {e}",
        )
