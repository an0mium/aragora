#!/usr/bin/env python3
"""Calibration-only multi-model issue triage CLI.

V1 contract (per #7170 review by Codex):
    - Estimate cost.
    - Pick a stratified sample of issues (default 30).
    - Gather evidence from GitHub + repo state BEFORE invoking models.
    - Run heterogeneous panel; persist receipt-equivalent artifacts.
    - Write JSONL + markdown.
    - No comments. No labels. No closures. No automation pause.

Examples
--------

Show the cost projection without invoking any model::

    python scripts/triage_issues_via_debate.py \\
        --repo synaptent/aragora \\
        --sample 30 \\
        --estimate

Run the 30-issue calibration sample (writes artifacts only)::

    python scripts/triage_issues_via_debate.py \\
        --repo synaptent/aragora \\
        --sample 30 \\
        --output-dir .aragora/triage/runs/$(date +%Y%m%dT%H%M%S) \\
        --budget-usd 10

Re-evaluate specific issues (calibration follow-up)::

    python scripts/triage_issues_via_debate.py \\
        --issues 7172,7171,7169 \\
        --output-dir .aragora/triage/calibration

Closing remains a founder action. Founder reviews artifacts and acts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.triage import (  # noqa: E402
    DEFAULT_PANEL,
    IssueDebateReceipt,
    IssueEvidence,
    IssueRecord,
    PanelMember,
    build_panel,
    estimate_cost_usd,
    evaluate_issue,
    gather_evidence,
    write_jsonl_receipt,
    write_markdown_report,
)

logger = logging.getLogger(__name__)


@dataclass
class StratumKey:
    label_bucket: str
    author_bucket: str
    is_automation: bool

    def key(self) -> str:
        return f"{self.author_bucket}::{self.label_bucket}::auto={self.is_automation}"


def _classify(issue: IssueRecord) -> StratumKey:
    labels_lower = {label.lower() for label in issue.labels}
    if "boss-stuck" in labels_lower:
        label_bucket = "boss-stuck"
    elif "stage-gate-drift" in labels_lower:
        label_bucket = "stage-gate-drift"
    elif "boss-ready" in labels_lower:
        label_bucket = "boss-ready"
    elif "automation" in labels_lower:
        label_bucket = "automation-label"
    elif not issue.labels:
        label_bucket = "no-label"
    else:
        label_bucket = "other"
    author_bucket = (
        "automation-author"
        if "[bot]" in issue.author or issue.author in {"an0mium"}
        else "human-author"
    )
    from aragora.triage.evidence import is_automation_generated

    return StratumKey(
        label_bucket=label_bucket,
        author_bucket=author_bucket,
        is_automation=is_automation_generated(
            author=issue.author, labels=issue.labels, body=issue.body
        ),
    )


def stratified_sample(
    issues: Sequence[IssueRecord],
    *,
    sample_size: int,
    seed: int = 0,
) -> list[IssueRecord]:
    """Return a stratified sample across (label_bucket, author_bucket, automation).

    Aims for proportional coverage so the calibration set isn't dominated
    by the single largest bucket (currently boss-stuck).
    """
    if sample_size <= 0:
        return []
    rng = random.Random(seed)
    buckets: dict[str, list[IssueRecord]] = defaultdict(list)
    for issue in issues:
        buckets[_classify(issue).key()].append(issue)

    if not buckets:
        return []

    total = len(issues)
    quotas: dict[str, int] = {}
    for key, bucket in buckets.items():
        proportional = max(1, round(sample_size * len(bucket) / total))
        quotas[key] = min(proportional, len(bucket))

    while sum(quotas.values()) > sample_size:
        largest = max(quotas, key=lambda k: quotas[k])
        if quotas[largest] <= 1:
            break
        quotas[largest] -= 1
    while sum(quotas.values()) < sample_size:
        candidates = [k for k, v in quotas.items() if v < len(buckets[k])]
        if not candidates:
            break
        pick = rng.choice(candidates)
        quotas[pick] += 1

    selected: list[IssueRecord] = []
    for key, count in quotas.items():
        rng.shuffle(buckets[key])
        selected.extend(buckets[key][:count])
    rng.shuffle(selected)
    return selected[:sample_size]


def _gh_json(args: Sequence[str]) -> Any:
    try:
        proc = subprocess.run(
            ["gh", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"gh invocation failed: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gh did not return JSON: {exc}") from exc


def fetch_open_issues(
    repo: str,
    *,
    limit: int = 1000,
    fetcher: Any = None,
) -> list[IssueRecord]:
    """Fetch open issues via gh. Returns ``IssueRecord`` objects."""
    runner = fetcher or _gh_json
    raw = runner(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            str(limit),
            "--json",
            "number,title,body,author,labels,state,url,createdAt,updatedAt,comments",
        ]
    )
    records: list[IssueRecord] = []
    for item in raw:
        author_obj = item.get("author") or {}
        author = author_obj.get("login") or author_obj.get("name") or "unknown"
        labels = tuple(label.get("name", "") for label in item.get("labels") or [])
        comments = item.get("comments") or []
        records.append(
            IssueRecord(
                number=int(item["number"]),
                title=item.get("title") or "",
                body=item.get("body") or "",
                author=author,
                labels=labels,
                state=item.get("state") or "open",
                url=item.get("url") or "",
                created_at=item.get("createdAt") or "",
                updated_at=item.get("updatedAt") or "",
                comments_count=len(comments) if isinstance(comments, list) else int(comments or 0),
            )
        )
    return records


def fetch_specific_issues(
    repo: str,
    numbers: Sequence[int],
    *,
    fetcher: Any = None,
) -> list[IssueRecord]:
    """Fetch a specific subset of issues by number."""
    runner = fetcher or _gh_json
    records: list[IssueRecord] = []
    for num in numbers:
        item = runner(
            [
                "issue",
                "view",
                str(num),
                "--repo",
                repo,
                "--json",
                "number,title,body,author,labels,state,url,createdAt,updatedAt,comments",
            ]
        )
        if not item:
            continue
        author_obj = item.get("author") or {}
        author = author_obj.get("login") or author_obj.get("name") or "unknown"
        labels = tuple(label.get("name", "") for label in item.get("labels") or [])
        comments = item.get("comments") or []
        records.append(
            IssueRecord(
                number=int(item["number"]),
                title=item.get("title") or "",
                body=item.get("body") or "",
                author=author,
                labels=labels,
                state=item.get("state") or "open",
                url=item.get("url") or "",
                created_at=item.get("createdAt") or "",
                updated_at=item.get("updatedAt") or "",
                comments_count=len(comments) if isinstance(comments, list) else int(comments or 0),
            )
        )
    return records


async def _agent_generator_factory(
    panel: Sequence[PanelMember],
) -> Any:
    """Return an async callable wrapping ``aragora.agents.create_agent``.

    Creates and caches one agent per panel member. Each call invokes
    ``agent.generate(prompt)``.
    """
    from aragora.agents import create_agent

    cache: dict[str, Any] = {}
    for member in panel:
        cache[member.agent_type] = create_agent(
            member.agent_type,
            name=f"triage-panel-{member.nickname or member.agent_type}",
            role=member.role,
            model=member.model_id,
        )

    async def generator(member: PanelMember, prompt: str) -> str:
        agent = cache[member.agent_type]
        return await agent.generate(prompt)

    return generator


async def _run_triage(
    *,
    issues: Sequence[IssueRecord],
    open_index: Sequence[IssueRecord],
    panel: Sequence[PanelMember],
    output_dir: Path,
    repo: str,
    budget_usd: float,
    max_concurrent: int,
    generator: Any,
) -> tuple[list[IssueDebateReceipt], dict[str, Any]]:
    jsonl_path = output_dir / "receipts.jsonl"
    md_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"

    already_done: set[int] = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                already_done.add(int(json.loads(line).get("issue_number", -1)))
            except (json.JSONDecodeError, ValueError):
                continue

    receipts: list[IssueDebateReceipt] = []
    spent_usd = 0.0
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_one(issue: IssueRecord) -> IssueDebateReceipt | None:
        nonlocal spent_usd
        async with semaphore:
            if spent_usd >= budget_usd:
                logger.warning(
                    "budget_cap_reached spent=%.4f cap=%.4f issue=#%s",
                    spent_usd,
                    budget_usd,
                    issue.number,
                )
                return None
            evidence = gather_evidence(
                issue,
                repo=repo,
                repo_root=REPO_ROOT,
                open_issue_index=open_index,
                now_iso=datetime.now(timezone.utc).isoformat(),
            )
            receipt = await evaluate_issue(
                evidence,
                panel=panel,
                generator=generator,
            )
            spent_usd += receipt.cost_usd
            write_jsonl_receipt(jsonl_path, receipt)
            print(
                f"#{receipt.issue_number} -> {receipt.aggregate_verdict} "
                f"({receipt.aggregate_consensus}, conf {receipt.aggregate_confidence:.2f}, "
                f"${receipt.cost_usd:.4f})",
                flush=True,
            )
            return receipt

    pending = [iss for iss in issues if iss.number not in already_done]
    if not pending:
        print("All issues already triaged in this run directory (resume noop).", flush=True)
    results = await asyncio.gather(*(evaluate_one(iss) for iss in pending))
    receipts = [r for r in results if r is not None]

    summary = {
        "repo": repo,
        "panel": [m.model_id for m in panel],
        "issues_targeted": len(issues),
        "issues_evaluated_this_run": len(receipts),
        "issues_skipped_resume": len(already_done),
        "spent_usd": round(spent_usd, 4),
        "budget_usd": budget_usd,
        "verdict_counts": _verdict_counts(receipts),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "jsonl": str(jsonl_path),
        "markdown": str(md_path),
    }
    all_receipts = _load_all_receipts(jsonl_path)
    write_markdown_report(
        md_path,
        all_receipts,
        summary_header=(
            f"Repo: {repo} | Panel: {', '.join(m.model_id for m in panel)} | "
            f"Budget used: ${spent_usd:.4f}/${budget_usd:.2f}"
        ),
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return receipts, summary


def _verdict_counts(receipts: Sequence[IssueDebateReceipt]) -> dict[str, int]:
    counter: dict[str, int] = {}
    for receipt in receipts:
        counter[receipt.aggregate_verdict] = counter.get(receipt.aggregate_verdict, 0) + 1
    return counter


def _load_all_receipts(path: Path) -> list[IssueDebateReceipt]:
    if not path.exists():
        return []
    loaded: list[IssueDebateReceipt] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        loaded.append(
            IssueDebateReceipt(
                issue_number=int(data.get("issue_number", 0)),
                issue_title=data.get("issue_title", ""),
                issue_url=data.get("issue_url", ""),
                issue_author=data.get("issue_author", ""),
                is_automation_generated=bool(data.get("is_automation_generated", False)),
                panel=list(data.get("panel", []) or []),
                prompt=data.get("prompt", ""),
                per_model=list(data.get("per_model", []) or []),
                aggregate_verdict=data.get("aggregate_verdict", "flag-for-human"),
                aggregate_confidence=float(data.get("aggregate_confidence", 0.0)),
                aggregate_consensus=data.get("aggregate_consensus", "unclear"),
                aggregation_rationale=data.get("aggregation_rationale", ""),
                confidence_class=data.get("confidence_class", "needs-spot-check"),
                recommendation=data.get("recommendation"),
                automation_value=data.get("automation_value", "n/a"),
                suggested_action=data.get("suggested_action", ""),
                evidence=dict(data.get("evidence", {}) or {}),
                started_at=data.get("started_at", ""),
                finished_at=data.get("finished_at", ""),
                cost_usd=float(data.get("cost_usd", 0.0)),
                latency_seconds=float(data.get("latency_seconds", 0.0)),
                notes=list(data.get("notes", []) or []),
                schema_version=data.get("schema_version", "triage-receipt/1.1"),
            )
        )
    return loaded


def _render_sample_card() -> str:
    """Build a synthetic receipt with mock per-model verdicts and render it.

    Used by ``--sample-card`` so the founder can see exactly what a single
    triage card looks like BEFORE paying for model calls. No network, no
    gh, no model API.
    """
    from aragora.triage.issue_evaluator import (
        PerModelVerdict,
        aggregate_verdicts,
    )
    from aragora.triage.receipts import write_markdown_report

    sample_issue = IssueRecord(
        number=9999,
        title="Narrow broad except Exception in aragora/billing/cost_tracker.py",
        body=(
            "Auto-generated by stage-gate-conductor. The handler in "
            "`aragora/billing/cost_tracker.py` catches `Exception:` at line 142 "
            "which masks real failures.\n\nProposed: replace with a narrower "
            "exception or re-raise after logging. See similar #6371."
        ),
        author="an0mium",
        labels=("boss-stuck", "automation", "narrow-except"),
        state="open",
        url="https://github.com/synaptent/aragora/issues/9999",
        created_at="2026-05-10T03:00:00Z",
        updated_at="2026-05-10T03:00:00Z",
        comments_count=0,
    )
    sample_evidence = IssueEvidence(
        issue=sample_issue,
        is_automation_generated=True,
        referenced_files=[
            {"path": "aragora/billing/cost_tracker.py", "exists_in_head": True},
        ],
        referenced_issues=[
            {
                "number": 6371,
                "title": "Narrow broad except Exception in fabric/runtime.py",
                "state": "OPEN",
                "url": "https://github.com/synaptent/aragora/issues/6371",
            },
        ],
        duplicate_candidates=[
            {
                "number": 6371,
                "title": "Narrow broad except Exception in fabric/runtime.py",
                "similarity": 0.62,
                "url": "https://github.com/synaptent/aragora/issues/6371",
            },
        ],
        repo_head_sha="dbd25030c",
        gathered_at="2026-05-15T04:00:00Z",
        notes=[],
    )

    mock_responses = {
        DEFAULT_PANEL[0]: PerModelVerdict(
            panel_member=DEFAULT_PANEL[0],
            verdict="refine",
            confidence=0.82,
            confidence_class="needs-spot-check",
            automation_value="valuable",
            rationale=(
                "Real bug in a live file; automation correctly identified an actionable "
                "narrow-except site. Title and body lack repro steps and don't propose a "
                "specific catch target, so refining the issue would unblock a human owner."
            ),
            suggested_action=(
                "Rewrite the title with the file/line and add a 3-line repro / proposed catch list."
            ),
            evidence_used=[
                "body para 1 cites aragora/billing/cost_tracker.py line 142",
                "referenced file exists in HEAD",
                "duplicate candidate #6371 similarity 0.62 (related but different file)",
            ],
            what_to_inspect=(
                "Open aragora/billing/cost_tracker.py around line 142 and confirm the "
                "broad except is still there. Compare to #6371 to see how the same kind of "
                "fix was scoped previously."
            ),
            safety_note=(
                "Safe to refine because the file exists and the issue describes a real "
                "anti-pattern."
            ),
            refined_title=(
                "billing/cost_tracker: narrow `except Exception` at line ~142 to specific errors"
            ),
            refined_body_outline=(
                "- Current code (link to line)\n"
                "- Why it masks real failures (one example)\n"
                "- Proposed narrower catch list (ConnectionError, ValueError, ...)\n"
                "- How to test (unit test pattern)"
            ),
            consolidate_with=None,
            raw_response="(mock)",
            prompt_chars=4500,
            response_chars=900,
            cost_usd=0.045,
            latency_seconds=4.2,
            error=None,
        ),
        DEFAULT_PANEL[1]: PerModelVerdict(
            panel_member=DEFAULT_PANEL[1],
            verdict="refine",
            confidence=0.74,
            confidence_class="needs-spot-check",
            automation_value="valuable",
            rationale=(
                "Concrete file reference resolves to HEAD; the broad-except anti-pattern is "
                "real and consistent with house style enforcement. The issue would benefit "
                "from cleaner scope and an explicit owner."
            ),
            suggested_action="Refine with repro and proposed catch list before acting.",
            evidence_used=[
                "file ref exists",
                "label `narrow-except` consistent with body",
            ],
            what_to_inspect="Inspect cost_tracker.py line range 130-160 and the test for it.",
            safety_note="Refining a real-bug ticket is reversible.",
            refined_title="narrow except in cost_tracker.py line 142",
            refined_body_outline=("- Reproducer\n- Proposed exception list\n- Test plan"),
            consolidate_with=None,
            raw_response="(mock)",
            prompt_chars=4500,
            response_chars=600,
            cost_usd=0.013,
            latency_seconds=3.4,
            error=None,
        ),
        DEFAULT_PANEL[2]: PerModelVerdict(
            panel_member=DEFAULT_PANEL[2],
            verdict="consolidate",
            confidence=0.58,
            confidence_class="do-not-act-without-human",
            automation_value="neutral",
            rationale=(
                "Title is nearly identical to #6371; duplicate similarity at 0.62 suggests "
                "this should merge into the broader narrow-except track rather than stand "
                "alone. Not fully certain because the files are different."
            ),
            suggested_action="Consolidate into #6371 as a sub-bullet for cost_tracker.py.",
            evidence_used=[
                "duplicate candidate #6371",
                "title pattern identical to several other broad-except issues",
            ],
            what_to_inspect=(
                "Open #6371 and decide whether it should be one tracker issue covering all "
                "narrow-except sites or per-file tickets."
            ),
            safety_note=(
                "Risk: consolidating prematurely loses the per-file specificity if the "
                "tracker doesn't enumerate sub-sites."
            ),
            refined_title="",
            refined_body_outline="",
            consolidate_with=6371,
            raw_response="(mock)",
            prompt_chars=4500,
            response_chars=550,
            cost_usd=0.008,
            latency_seconds=2.8,
            error=None,
        ),
    }
    per_model_objs = list(mock_responses.values())
    aggregate = aggregate_verdicts(per_model_objs)
    recommendation_payload = (
        aggregate.recommendation.to_dict() if aggregate.recommendation else None
    )
    receipt = IssueDebateReceipt(
        issue_number=sample_issue.number,
        issue_title=sample_issue.title,
        issue_url=sample_issue.url,
        issue_author=sample_issue.author,
        is_automation_generated=True,
        panel=[m.model_id for m in DEFAULT_PANEL],
        prompt="(mock prompt)",
        per_model=[pm.to_dict() for pm in per_model_objs],
        aggregate_verdict=aggregate.verdict,
        aggregate_confidence=aggregate.confidence,
        aggregate_consensus=aggregate.consensus,
        aggregation_rationale=aggregate.rationale,
        confidence_class=aggregate.confidence_class,
        recommendation=recommendation_payload,
        automation_value=aggregate.automation_value,
        suggested_action=aggregate.suggested_action,
        evidence=sample_evidence.to_dict(),
        started_at="2026-05-15T04:00:00Z",
        finished_at="2026-05-15T04:00:05Z",
        cost_usd=round(sum(pm.cost_usd for pm in per_model_objs), 6),
        latency_seconds=4.5,
        notes=list(aggregate.notes),
    )
    import tempfile

    with tempfile.NamedTemporaryFile(
        prefix="aragora-sample-card-", suffix=".md", delete=False
    ) as fh:
        out_path = Path(fh.name)
    write_markdown_report(
        out_path,
        [receipt],
        summary_header=("Repo: synaptent/aragora (SAMPLE CARD ONLY -- no real models invoked)"),
    )
    return out_path.read_text(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibration-only multi-model GitHub issue triage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "V1 is calibration-only: no comments, no labels, no closures.\n"
            "Founder reviews the JSONL + markdown artifacts and decides next steps."
        ),
    )
    parser.add_argument("--repo", default="synaptent/aragora")
    parser.add_argument("--sample", type=int, default=30, help="Stratified sample size.")
    parser.add_argument(
        "--issues",
        default="",
        help="Comma-separated issue numbers (overrides --sample).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("ARAGORA_TRIAGE_SEED", "1337")),
    )
    parser.add_argument(
        "--panel",
        default="",
        help="Comma-separated agent_types to use; default = anthropic-api,openai-api,gemini.",
    )
    parser.add_argument(
        "--output-dir",
        default=f".aragora/triage/runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
    )
    parser.add_argument("--budget-usd", type=float, default=10.0)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print cost projection and exit without invoking any model.",
    )
    parser.add_argument(
        "--dry-run-prompt",
        action="store_true",
        help="Print the first issue's full prompt and exit without invoking any model.",
    )
    parser.add_argument(
        "--sample-card",
        action="store_true",
        help=(
            "Generate a synthetic receipt with mock per-model verdicts and "
            "render the founder-facing card to stdout. No model calls, no gh "
            "calls. Use to inspect report shape before running real calibration."
        ),
    )
    parser.add_argument(
        "--limit-pool",
        type=int,
        default=1000,
        help="Max issues to fetch from gh before sampling.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("ARAGORA_TRIAGE_LOG_LEVEL", "INFO"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if getattr(args, "sample_card", False):
        print(_render_sample_card())
        return 0

    panel = build_panel([item.strip() for item in args.panel.split(",")] if args.panel else None)

    if args.issues:
        numbers = [int(part) for part in args.issues.split(",") if part.strip()]
        target_issues = fetch_specific_issues(args.repo, numbers)
        open_index = fetch_open_issues(args.repo, limit=args.limit_pool)
    else:
        open_index = fetch_open_issues(args.repo, limit=args.limit_pool)
        target_issues = stratified_sample(open_index, sample_size=args.sample, seed=args.seed)

    if not target_issues:
        print("No target issues resolved; nothing to do.", file=sys.stderr)
        return 1

    projection = estimate_cost_usd(panel=panel, issue_count=len(target_issues))
    print("Cost projection (4-chars-per-token heuristic):")
    print(json.dumps(projection, indent=2))
    print()
    print(f"Target issues: {[iss.number for iss in target_issues]}")
    print(f"Output dir: {args.output_dir}")

    if args.estimate:
        return 0

    if projection["total_usd"] > args.budget_usd:
        print(
            f"Projected cost ${projection['total_usd']:.2f} exceeds budget "
            f"${args.budget_usd:.2f}; aborting (rerun with higher --budget-usd or smaller --sample).",
            file=sys.stderr,
        )
        return 2

    if args.dry_run_prompt:
        evidence = gather_evidence(
            target_issues[0],
            repo=args.repo,
            repo_root=REPO_ROOT,
            open_issue_index=open_index,
            now_iso=datetime.now(timezone.utc).isoformat(),
        )
        from aragora.triage.issue_evaluator import build_panel_prompt

        prompt = build_panel_prompt(evidence)
        print("---- DRY RUN PROMPT (first issue) ----")
        print(prompt)
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = asyncio.run(_agent_generator_factory(panel))
    receipts, summary = asyncio.run(
        _run_triage(
            issues=target_issues,
            open_index=open_index,
            panel=panel,
            output_dir=output_dir,
            repo=args.repo,
            budget_usd=args.budget_usd,
            max_concurrent=args.max_concurrent,
            generator=generator,
        )
    )

    print()
    print("Run summary:")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Receipts: {output_dir / 'receipts.jsonl'}")
    print(f"Report:   {output_dir / 'report.md'}")
    print(f"Summary:  {output_dir / 'summary.json'}")
    print()
    print(
        "Calibration v1 wrote artifacts only. Closing remains a founder action; "
        "review precision against the rubric before scaling."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
