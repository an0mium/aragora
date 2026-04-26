"""Tier-1 benchmark runner — iterates tasks, runs both systems, judges, writes CSV."""

from __future__ import annotations

import csv
import json
import pathlib
import time
from collections.abc import Iterable, Iterator

from benchmarks.bench_readiness.tier1.judge import JudgeVerdict, judge
from benchmarks.bench_readiness.tier1.systems import (
    SystemOutput,
    run_aragora_debate,
    run_solo_opus,
)
from benchmarks.bench_readiness.tier1.tasks import legal as legal_loader
from benchmarks.bench_readiness.tier1.tasks import aragora_custom as custom_loader
from benchmarks.bench_readiness.tier1.tasks.base import TaskItem

CSV_FIELDS = [
    "task_id",
    "domain",
    "eval_strategy",
    "solo_system",
    "debate_system",
    "solo_latency_sec",
    "debate_latency_sec",
    "solo_tokens_in",
    "solo_tokens_out",
    "solo_cost_usd",
    "debate_tokens_in",
    "debate_tokens_out",
    "solo_error",
    "debate_error",
    "solo_correctness",
    "solo_completeness",
    "solo_reasoning_quality",
    "solo_usefulness",
    "debate_correctness",
    "debate_completeness",
    "debate_reasoning_quality",
    "debate_usefulness",
    "winner_system",
    "exact_match_used",
    "judge_error",
    "judge_rationale",
    "metadata_json",
    "solo_answer",
    "debate_answer",
]


def _load_tasks(domains: Iterable[str], limit: int, seed: int) -> Iterator[TaskItem]:
    """Yield up to ``limit`` items per domain, lazily."""
    for d in domains:
        if d == "legal":
            yield from legal_loader.load(limit, seed)
        elif d == "aragora_custom":
            yield from custom_loader.load(limit, seed)
        elif d == "mmlu_pro":
            # Import lazily so harness still works if datasets is absent
            from benchmarks.bench_readiness.tier1.tasks import mmlu_pro

            yield from mmlu_pro.load(limit, seed)
        elif d == "swebench_lite":
            from benchmarks.bench_readiness.tier1.tasks import swebench

            yield from swebench.load(limit, seed)
        else:
            raise ValueError(f"unknown domain: {d}")


def _row(
    task: TaskItem,
    solo: SystemOutput,
    dbt: SystemOutput,
    verdict: JudgeVerdict,
) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "eval_strategy": task.eval_strategy,
        "solo_system": solo.system,
        "debate_system": dbt.system,
        "solo_latency_sec": round(solo.latency_sec, 3),
        "debate_latency_sec": round(dbt.latency_sec, 3),
        "solo_tokens_in": solo.tokens_in,
        "solo_tokens_out": solo.tokens_out,
        "solo_cost_usd": round(solo.cost_usd, 4),
        "debate_tokens_in": dbt.tokens_in,
        "debate_tokens_out": dbt.tokens_out,
        "solo_error": solo.error,
        "debate_error": dbt.error,
        "solo_correctness": verdict.scores_a.get("correctness", ""),
        "solo_completeness": verdict.scores_a.get("completeness", ""),
        "solo_reasoning_quality": verdict.scores_a.get("reasoning_quality", ""),
        "solo_usefulness": verdict.scores_a.get("usefulness", ""),
        "debate_correctness": verdict.scores_b.get("correctness", ""),
        "debate_completeness": verdict.scores_b.get("completeness", ""),
        "debate_reasoning_quality": verdict.scores_b.get("reasoning_quality", ""),
        "debate_usefulness": verdict.scores_b.get("usefulness", ""),
        "winner_system": verdict.winner_system,
        "exact_match_used": verdict.exact_match_used,
        "judge_error": verdict.error,
        "judge_rationale": verdict.rationale[:1500],
        "metadata_json": json.dumps(task.metadata, separators=(",", ":")),
        "solo_answer": solo.answer[:4000],
        "debate_answer": dbt.answer[:4000],
    }


def _summarize(rows: list[dict[str, object]]) -> str:
    """Produce a human-readable Markdown summary of the run."""
    by_domain: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by_domain.setdefault(str(r["domain"]), []).append(r)

    lines = ["# Tier-1 Benchmark Summary", ""]
    lines.append(f"Total items: **{len(rows)}**  ")
    lines.append(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for domain, drs in sorted(by_domain.items()):
        lines.append(f"## {domain}  (n={len(drs)})")
        if not drs:
            lines.append("")
            continue

        solo_sys = drs[0]["solo_system"]
        dbt_sys = drs[0]["debate_system"]
        solo_wins = sum(1 for r in drs if r["winner_system"] == solo_sys)
        dbt_wins = sum(1 for r in drs if r["winner_system"] == dbt_sys)
        ties = sum(1 for r in drs if r["winner_system"] == "TIE")
        errors = sum(1 for r in drs if r["judge_error"] or r["solo_error"] or r["debate_error"])

        lines.append(f"- Solo ({solo_sys}) wins: **{solo_wins}**")
        lines.append(f"- Debate ({dbt_sys}) wins: **{dbt_wins}**")
        lines.append(f"- Ties: **{ties}**")
        lines.append(f"- Errors / skipped: **{errors}**")

        # Average latency
        solo_avg = sum(float(r["solo_latency_sec"]) for r in drs) / len(drs)
        dbt_avg = sum(float(r["debate_latency_sec"]) for r in drs) / len(drs)
        lines.append(f"- Avg latency — solo: {solo_avg:.1f}s | debate: {dbt_avg:.1f}s")
        lines.append("")

    return "\n".join(lines)


def run(
    *,
    api_key: str,
    domains: list[str],
    limit: int,
    seed: int,
    out_dir: pathlib.Path,
    debate_rounds: int = 2,
    model: str = "claude-opus-4-7",
    log=print,  # noqa: A002
) -> dict[str, object]:
    """Execute the benchmark end-to-end.

    Returns a manifest dict with CSV path, summary path, and counts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"tier1_results_{ts}.csv"
    summary_path = out_dir / f"tier1_summary_{ts}.md"

    rows: list[dict[str, object]] = []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for i, task in enumerate(_load_tasks(domains, limit, seed), start=1):
            log(f"[{i}] {task.task_id}  ({task.domain})")

            solo = run_solo_opus(task, api_key=api_key, model=model)
            log(f"    solo   : {solo.latency_sec:.1f}s  err={solo.error!r}")

            dbt = run_aragora_debate(task, api_key=api_key, model=model, rounds=debate_rounds)
            log(f"    debate : {dbt.latency_sec:.1f}s  err={dbt.error!r}")

            verdict = judge(task, solo, dbt, api_key=api_key, seed=seed + i)
            log(f"    judge  : winner={verdict.winner_system!r}  err={verdict.error!r}")

            row = _row(task, solo, dbt, verdict)
            writer.writerow(row)
            f.flush()
            rows.append(row)

    summary_md = _summarize(rows)
    summary_path.write_text(summary_md, encoding="utf-8")

    return {
        "csv": str(csv_path),
        "summary": str(summary_path),
        "items": len(rows),
        "domains": domains,
    }
