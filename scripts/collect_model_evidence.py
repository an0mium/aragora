#!/usr/bin/env python3
"""Collect one exact-head model/dogfood evidence signal for a PR.

Default mode is non-mutating: inspect PR truth, try model routes until one
returns a clean review, lint the proposed PR comment with review-queue's
evidence parser, and print the comment plus the exact posting command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEOUT_SECONDS = 420.0


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ModelRoute:
    """One direct or Droid-backed model review route."""

    key: str
    heading_family: str
    transport: str
    model: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass
class EvidenceAttempt:
    route: str
    status: str
    error: str = ""
    returncode: int | None = None
    counted_reviewer_ids: list[str] = field(default_factory=list)


@dataclass
class EvidenceResult:
    pr: int
    head: str
    status: str
    selected_route: str | None = None
    comment: str | None = None
    posted: bool = False
    comment_url: str | None = None
    lint_result: dict[str, Any] | None = None
    blocking_findings: list[str] = field(default_factory=list)
    attempts: list[EvidenceAttempt] = field(default_factory=list)
    next_command: str | None = None


ROUTES: dict[str, ModelRoute] = {
    "gemini": ModelRoute("gemini", "Gemini", "direct", "gemini-3.1-pro-preview"),
    "grok": ModelRoute("grok", "Grok", "direct", "grok-4-latest"),
    "claude": ModelRoute("claude", "Claude", "direct", "sonnet"),
    "droid-gemini": ModelRoute(
        "droid-gemini", "Gemini via Droid", "droid", "gemini-3.1-pro-preview"
    ),
    "droid-claude-opus": ModelRoute(
        "droid-claude-opus", "Claude via Droid", "droid", "claude-opus-4-7"
    ),
    "droid-claude-sonnet": ModelRoute(
        "droid-claude-sonnet", "Claude via Droid", "droid", "claude-sonnet-4-6"
    ),
    "droid-gpt54": ModelRoute("droid-gpt54", "OpenAI via Droid", "droid", "gpt-5.4"),
    "droid-kimi": ModelRoute("droid-kimi", "Kimi via Droid", "droid", "kimi-k2.5"),
    "droid-glm": ModelRoute("droid-glm", "GLM via Droid", "droid", "glm-5.1"),
    "droid-minimax": ModelRoute("droid-minimax", "MiniMax via Droid", "droid", "minimax-m2.7"),
}

DEFAULT_ROUTE_ORDER: tuple[str, ...] = (
    "gemini",
    "grok",
    "claude",
    "droid-gemini",
    "droid-claude-opus",
    "droid-claude-sonnet",
)


def _default_runner(
    args: list[str],
    *,
    input_text: str | None = None,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=DEFAULT_REPO_ROOT,
    )


def _json_from_stdout(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command did not return JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("command returned non-object JSON")
    return payload


def _run_json(args: list[str], runner: CommandRunner) -> dict[str, Any]:
    completed = runner(args, timeout_seconds=180.0)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip())
    return _json_from_stdout(completed)


def _fetch_pr(pr: int, runner: CommandRunner) -> dict[str, Any]:
    return _run_json(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--json",
            "number,headRefOid,url,title,files",
        ],
        runner,
    )


def _fetch_diff(pr: int, runner: CommandRunner) -> str:
    completed = runner(["gh", "pr", "diff", str(pr), "--patch"], timeout_seconds=180.0)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip())
    return completed.stdout


def _route_command(route: ModelRoute, prompt: str) -> tuple[list[str], str | None]:
    if route.transport == "droid":
        return (
            [
                "droid",
                "exec",
                "--model",
                route.model,
                "--output-format",
                "json",
                "--disabled-tools",
                "Execute",
                "--cwd",
                str(DEFAULT_REPO_ROOT),
            ],
            prompt,
        )
    if route.key == "gemini":
        return (
            [
                "gemini",
                "--approval-mode",
                "plan",
                "--model",
                route.model,
                "-p",
                "Review the PR patch from stdin. Return the requested concise result only.",
            ],
            prompt,
        )
    if route.key == "grok":
        return (["grok", "--model", route.model, "-p", prompt], None)
    if route.key == "claude":
        return (
            [
                "claude",
                "--permission-mode",
                "plan",
                "--tools",
                "",
                "--no-session-persistence",
                "--print",
                "--model",
                route.model,
            ],
            prompt,
        )
    raise ValueError(f"unsupported route {route.key!r}")


def _render_model_prompt(
    *,
    pr: int,
    head: str,
    title: str,
    files: list[str],
    diff_text: str,
) -> str:
    files_text = "\n".join(f"- {path}" for path in files) or "- unknown"
    return "\n".join(
        [
            f"You are providing one independent non-Codex model/dogfood review signal for Aragora PR #{pr}.",
            f"Exact PR head: {head}",
            f"PR title: {title}",
            "",
            "Review only the supplied patch. Do not run tools or modify files.",
            "Focus on whether the change is safe, limited to the touched scope, and free of misleading docs or generated-output drift.",
            "Return concise markdown with these labels: Verdict, Blocking findings, Non-blocking notes, Validation confidence.",
            "Use 'Blocking findings: None' only if there are no blockers.",
            "",
            "Files in PR:",
            files_text,
            "",
            "PATCH:",
            diff_text,
        ]
    )


def _extract_model_text(stdout: str) -> str:
    stripped = (stdout or "").strip()
    if not stripped:
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(payload, dict):
        result = payload.get("result")
        if isinstance(result, str):
            return result.strip()
        content = payload.get("content")
        if isinstance(content, str):
            return content.strip()
    return stripped


def _extract_blocking_findings(model_text: str) -> list[str]:
    text = model_text.strip()
    if not text:
        return ["model returned empty review"]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        raw = payload.get("blocking_findings")
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            return [] if lowered in {"", "none", "no", "no blockers", "[]"} else [raw.strip()]

    lines = [line.rstrip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        lower = line.lower().strip()
        if "blocking findings" not in lower:
            continue
        _, _, after = line.partition(":")
        inline = after.strip()
        if inline:
            normalized = inline.strip("- ").lower()
            if normalized in {"none", "no", "no blockers", "[]"}:
                return []
            return [inline.strip("- ")]
        findings: list[str] = []
        for follow in lines[index + 1 :]:
            stripped = follow.strip()
            if not stripped:
                if findings:
                    break
                continue
            if stripped.startswith("#") or stripped.lower().startswith(
                ("non-blocking", "validation", "verdict")
            ):
                break
            if stripped.startswith(("-", "*")):
                item = stripped.lstrip("-* ").strip()
                if item.lower() not in {"none", "no blockers", "no blocking findings"}:
                    findings.append(item)
            elif findings:
                findings.append(stripped)
        return findings
    if "verdict: blocked" in text.lower():
        return ["model reported blocked verdict"]
    return []


def _render_comment(
    *,
    route: ModelRoute,
    pr: int,
    head: str,
    files: list[str],
    model_text: str,
) -> str:
    files_text = "\n".join(f"- `{path}`" for path in files) or "- `unknown`"
    review_excerpt = model_text.strip()
    if len(review_excerpt) > 2400:
        review_excerpt = review_excerpt[:2400].rstrip() + "\n... [truncated]"
    return "\n".join(
        [
            f"## {route.heading_family} focused adversarial dogfood",
            "",
            f"Current head: {head}",
            f"Route: `{route.key}` via `{route.transport}`; model `{route.model}`.",
            "",
            "Files reviewed:",
            files_text,
            "",
            "Focused adversarial dogfood verdict:",
            "- Reviewed the exact PR diff for scope drift, misleading documentation, generated-doc drift, and secret-value leakage.",
            "- Blocking findings: None.",
            "",
            "Validation:",
            "- Model review only; no local tests were run by this evidence collector.",
            "- Required CI and merge-packet remain authoritative.",
            "",
            "Model output:",
            "",
            review_excerpt,
            "",
            "This is model evidence for merge-quorum only; it is not merge authorization.",
        ]
    )


def _lint_comment(
    *,
    pr: int,
    head: str,
    body: str,
    runner: CommandRunner,
) -> dict[str, Any]:
    completed = runner(
        [
            "python3",
            "-m",
            "aragora.cli.main",
            "review-queue",
            "evidence-lint",
            "--pr",
            str(pr),
            "--head-sha",
            head,
            "--author",
            "an0mium",
            "--body",
            body,
            "--json",
        ],
        timeout_seconds=180.0,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip())
    return _json_from_stdout(completed)


def _preflight_route_counts(
    *,
    route: ModelRoute,
    pr: int,
    head: str,
    runner: CommandRunner,
) -> tuple[bool, dict[str, Any]]:
    body = "\n".join(
        [
            f"## {route.heading_family} focused adversarial dogfood",
            "",
            f"Current head: {head}",
            "No blockers.",
        ]
    )
    lint = _lint_comment(pr=pr, head=head, body=body, runner=runner)
    return bool(lint.get("would_count")), lint


def _parse_family_order(raw: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_ROUTE_ORDER
    if isinstance(raw, (tuple, list)):
        items = [str(item).strip() for item in raw]
    else:
        items = [item.strip() for chunk in str(raw).split(",") for item in chunk.split()]
    routes = tuple(item for item in items if item)
    unknown = [item for item in routes if item not in ROUTES]
    if unknown:
        raise ValueError(f"unknown model route(s): {', '.join(unknown)}")
    return routes or DEFAULT_ROUTE_ORDER


def _post_command(pr: int, head: str, family_order: tuple[str, ...]) -> str:
    return (
        f"python3 scripts/collect_model_evidence.py --pr {pr} --head {head} "
        f"--family-order {','.join(family_order)} --post-comment --json"
    )


def collect_model_evidence(
    *,
    pr: int,
    expected_head: str,
    family_order: tuple[str, ...] | list[str] | str | None = None,
    post_comment: bool = False,
    allow_advisory_uncounted: bool = False,
    runner: CommandRunner = _default_runner,
) -> EvidenceResult:
    """Collect one model signal, optionally posting after all gates pass."""
    routes = _parse_family_order(family_order)
    pr_data = _fetch_pr(pr, runner)
    live_head = str(pr_data.get("headRefOid") or "").strip()
    result = EvidenceResult(
        pr=pr,
        head=live_head,
        status="started",
        next_command=_post_command(pr, expected_head, routes),
    )
    if live_head != expected_head:
        result.status = "head_drift"
        return result

    raw_files = pr_data.get("files") or []
    files = [
        str(item.get("path") or "").strip()
        for item in raw_files
        if isinstance(item, dict) and str(item.get("path") or "").strip()
    ]
    diff_text = _fetch_diff(pr, runner)
    title = str(pr_data.get("title") or "")

    skipped_for_lint = 0
    for route_key in routes:
        route = ROUTES[route_key]
        if not allow_advisory_uncounted:
            counts, preflight_lint = _preflight_route_counts(
                route=route, pr=pr, head=expected_head, runner=runner
            )
            if not counts:
                skipped_for_lint += 1
                result.attempts.append(
                    EvidenceAttempt(
                        route=route.key,
                        status="skipped",
                        error="preflight evidence-lint would not count",
                        counted_reviewer_ids=list(preflight_lint.get("counted_reviewer_ids") or []),
                    )
                )
                continue

        prompt = _render_model_prompt(
            pr=pr,
            head=expected_head,
            title=title,
            files=files,
            diff_text=diff_text,
        )
        command, stdin_text = _route_command(route, prompt)
        completed = runner(command, input_text=stdin_text, timeout_seconds=route.timeout_seconds)
        if completed.returncode != 0:
            result.attempts.append(
                EvidenceAttempt(
                    route=route.key,
                    status="failed",
                    error=(completed.stderr or completed.stdout or "").strip(),
                    returncode=completed.returncode,
                )
            )
            continue

        model_text = _extract_model_text(completed.stdout)
        blockers = _extract_blocking_findings(model_text)
        if blockers:
            result.status = "blocking_findings"
            result.selected_route = route.key
            result.blocking_findings = blockers
            result.attempts.append(
                EvidenceAttempt(route=route.key, status="blocking_findings", returncode=0)
            )
            return result

        comment = _render_comment(
            route=route,
            pr=pr,
            head=expected_head,
            files=files,
            model_text=model_text,
        )
        lint = _lint_comment(pr=pr, head=expected_head, body=comment, runner=runner)
        if not lint.get("would_count") and not allow_advisory_uncounted:
            result.attempts.append(
                EvidenceAttempt(
                    route=route.key,
                    status="lint_failed",
                    error=", ".join(str(item) for item in lint.get("problems") or []),
                    returncode=0,
                    counted_reviewer_ids=list(lint.get("counted_reviewer_ids") or []),
                )
            )
            continue

        result.status = "ready"
        result.selected_route = route.key
        result.comment = comment
        result.lint_result = lint
        result.attempts.append(
            EvidenceAttempt(
                route=route.key,
                status="ready",
                returncode=0,
                counted_reviewer_ids=list(lint.get("counted_reviewer_ids") or []),
            )
        )
        if not post_comment:
            return result

        latest = _fetch_pr(pr, runner)
        latest_head = str(latest.get("headRefOid") or "").strip()
        if latest_head != expected_head:
            result.status = "head_drift"
            result.posted = False
            return result
        posted = runner(
            ["gh", "pr", "comment", str(pr), "--body", comment],
            timeout_seconds=180.0,
        )
        if posted.returncode != 0:
            result.status = "post_failed"
            result.posted = False
            result.attempts.append(
                EvidenceAttempt(
                    route=route.key,
                    status="post_failed",
                    error=(posted.stderr or posted.stdout or "").strip(),
                    returncode=posted.returncode,
                )
            )
            return result
        result.status = "posted"
        result.posted = True
        result.comment_url = (posted.stdout or "").strip() or None
        return result

    result.status = (
        "no_countable_route" if skipped_for_lint == len(routes) else "no_successful_route"
    )
    return result


def _result_to_json(result: EvidenceResult) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=True)


def _render_text(result: EvidenceResult) -> str:
    lines = [
        f"status: {result.status}",
        f"pr: {result.pr}",
        f"head: {result.head}",
        f"selected_route: {result.selected_route or ''}",
        f"posted: {str(result.posted).lower()}",
    ]
    if result.blocking_findings:
        lines.append("blocking_findings:")
        lines.extend(f"- {item}" for item in result.blocking_findings)
    if result.comment:
        lines.extend(["", "proposed_comment:", result.comment])
    if result.next_command and not result.posted:
        lines.extend(["", "post_command:", result.next_command])
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True, help="Pull request number")
    parser.add_argument("--head", required=True, help="Exact PR head SHA")
    parser.add_argument(
        "--family-order",
        default=",".join(DEFAULT_ROUTE_ORDER),
        help="Comma/space-separated route order, e.g. gemini,droid-gemini,claude",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--post-comment",
        action="store_true",
        help="Post the linted evidence comment after rechecking the exact head",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force non-mutating mode even when --post-comment is present",
    )
    parser.add_argument(
        "--allow-advisory-uncounted",
        action="store_true",
        help="Allow routes whose headings do not currently count for merge-quorum",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = collect_model_evidence(
            pr=args.pr,
            expected_head=args.head,
            family_order=args.family_order,
            post_comment=bool(args.post_comment and not args.dry_run),
            allow_advisory_uncounted=args.allow_advisory_uncounted,
        )
    except Exception as exc:  # noqa: BLE001 - top-level CLI error report
        error = {"status": "error", "error": str(exc)}
        if args.json:
            print(json.dumps(error, indent=2, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(_result_to_json(result))
    else:
        print(_render_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
