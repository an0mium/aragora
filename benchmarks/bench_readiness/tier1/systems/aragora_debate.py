"""Aragora-debate system — 3 x Opus 4.7 agents with affirmative/negative/neutral stances.

Uses the standalone ``aragora_debate`` package (MIT, zero required deps)
rather than the full ``aragora.Arena`` because Phase B established that
the full Arena pulls in pipeline stages, research, and network hooks even
with demo flags. For a clean A/B, we want the debate engine proper —
nothing else.
"""

from __future__ import annotations

import asyncio
import time

from benchmarks.bench_readiness.tier1.systems.base import SystemOutput
from benchmarks.bench_readiness.tier1.tasks.base import TaskItem

_STANCES = ("affirmative", "negative", "neutral")

SYSTEM_PROMPT_TEMPLATE = (
    "You are a careful expert analyst participating in an adversarial debate. "
    "Your assigned stance for this debate is '{stance}'. Take that stance "
    "seriously but adjust your view if the counter-argument is stronger. "
    "Be specific, defensible, and concise."
)


def _synthesize_answer(result) -> str:
    """Produce a single text answer from the debate result.

    The standalone package's DebateResult carries a receipt (with final
    consensus) plus per-agent proposals. We return the consensus verdict
    text if present, otherwise the final-round proposals concatenated.
    """
    receipt = getattr(result, "receipt", None)
    if receipt is not None:
        verdict = getattr(receipt, "verdict", None)
        if verdict is not None:
            text = getattr(verdict, "text", "") or getattr(verdict, "summary", "")
            if text:
                return text.strip()

    # Fallback: concatenate final proposals
    proposals = getattr(result, "final_proposals", None) or []
    if proposals:
        parts = []
        for p in proposals:
            who = getattr(p, "author", "agent")
            txt = getattr(p, "text", "") or getattr(p, "content", "")
            if txt:
                parts.append(f"[{who}]\n{txt}")
        if parts:
            return "\n\n---\n\n".join(parts)

    # Last-ditch: stringify the result
    return str(result)


async def _run(task: TaskItem, api_key: str, model: str, rounds: int) -> SystemOutput:
    from aragora_debate import Debate, create_agent

    topic = task.prompt
    context = task.context

    debate = Debate(
        topic=topic,
        context=context,
        rounds=rounds,
        consensus="majority",
        early_stopping=True,
        enable_trickster=False,
        enable_convergence=True,
        convergence_threshold=0.85,
    )

    for stance in _STANCES:
        agent = create_agent(
            "anthropic",
            name=f"opus-{stance}",
            model=model,
            api_key=api_key,
            system_prompt=SYSTEM_PROMPT_TEMPLATE.format(stance=stance),
            stance=stance,
        )
        debate.add_agent(agent)

    t0 = time.perf_counter()
    try:
        result = await debate.run()
        latency = time.perf_counter() - t0
        answer = _synthesize_answer(result)

        # Token counting across agents is best-effort; the standalone package
        # does not aggregate it by default. We approximate by counting what's
        # in the final answer and the topic/context.
        tokens_in_approx = (len(topic) + len(context)) // 4  # 1 tok ~ 4 chars
        tokens_out_approx = len(answer) // 4

        return SystemOutput(
            system=f"aragora_debate_3x_{model}",
            answer=answer,
            latency_sec=latency,
            tokens_in=tokens_in_approx,
            tokens_out=tokens_out_approx,
            cost_usd=0.0,  # per-agent billing not exposed; tracked via raw
            raw={
                "rounds": rounds,
                "stances": list(_STANCES),
                "convergence_reached": getattr(result, "converged", None),
            },
        )
    except Exception as e:  # noqa: BLE001 - debate providers raise non-uniform runtime errors
        latency = time.perf_counter() - t0
        return SystemOutput(
            system=f"aragora_debate_3x_{model}",
            answer="",
            latency_sec=latency,
            error=f"{type(e).__name__}: {e}",
        )


def run_aragora_debate(
    task: TaskItem,
    *,
    api_key: str,
    model: str = "claude-opus-4-7",
    rounds: int = 2,
) -> SystemOutput:
    """Synchronous wrapper around the async :class:`Debate`."""
    return asyncio.run(_run(task, api_key, model, rounds))
