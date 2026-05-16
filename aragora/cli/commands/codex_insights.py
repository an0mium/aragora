"""CLI handlers for ``aragora codex insights {summary, anomalies, crossref, digest}``.

Analysis layer over the read-only Codex Desktop inspector. Read-only with one
opt-in write: ``digest --emit-receipt`` writes a signed JSON document under
``.aragora/codex_insights/``.

Heavy imports deferred to invocation time (matches ``aragora/cli/backup.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parse_since(value: str):  # type: ignore[no-untyped-def]
    from aragora.codex.duration import parse_duration

    return parse_duration(value)


def _resolve_paths(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from aragora.codex.desktop_paths import resolve

    return resolve(getattr(args, "codex_home", None))


def _emit_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


def _format_kv_block(title: str, items: dict[str, Any]) -> str:
    if not items:
        return f"{title}: (empty)"
    lines = [title + ":"]
    for key in sorted(
        items, key=lambda k: -float(items[k]) if isinstance(items[k], (int, float)) else 0
    ):
        lines.append(f"    {items[key]!s:>10}  {key}")
    return "\n".join(lines)


def cmd_codex_insights_summary(args: argparse.Namespace) -> int:
    """Aggregate session patterns across the analyzed window."""
    from aragora.codex.insights import summarize_patterns

    try:
        since = _parse_since(args.since)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    paths = _resolve_paths(args)
    if not paths.sqlite_path.exists():
        print(
            f"error: Codex Desktop state DB not found at {paths.sqlite_path}",
            file=sys.stderr,
        )
        return 1
    pattern, pairs = summarize_patterns(
        since=since,
        include_archived=args.include_archived,
        paths=paths,
    )
    if args.json:
        _emit_json(
            {
                "patterns": pattern.to_dict(),
                "thread_count": len(pairs),
            }
        )
        return 0
    print(f"Window:    {since} ({len(pairs)} threads scanned)")
    print(f"Tokens:    {pattern.total_tokens_used:,}")
    print(f"Distinct cwds: {pattern.distinct_cwds}")
    print(
        f"Duration:  p50={pattern.duration_seconds_p50:.1f}s  p95={pattern.duration_seconds_p95:.1f}s"
    )
    print(f"Abandoned: {pattern.abandoned_count} (silent rollouts > 10m)")
    print(_format_kv_block("Models", pattern.model_distribution))
    print(_format_kv_block("Tool calls", pattern.tool_call_distribution))
    return 0


def cmd_codex_insights_anomalies(args: argparse.Namespace) -> int:
    """Detect stuck / runaway / over-budget sessions."""
    from aragora.codex.insights import detect_anomalies, summarize_patterns

    try:
        since = _parse_since(args.since)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    paths = _resolve_paths(args)
    if not paths.sqlite_path.exists():
        print(
            f"error: Codex Desktop state DB not found at {paths.sqlite_path}",
            file=sys.stderr,
        )
        return 1
    _, pairs = summarize_patterns(
        since=since,
        include_archived=args.include_archived,
        paths=paths,
    )
    anomalies = detect_anomalies(
        pairs,
        token_cap=args.token_cap,
        runaway_tool_calls=args.runaway_tool_calls,
        stuck_turn_minutes=args.stuck_turn_minutes,
    )
    if args.json:
        _emit_json([a.to_dict() for a in anomalies])
        return 0
    if not anomalies:
        print(f"(no anomalies in last {since})")
        return 0
    print(f"{len(anomalies)} anomalies in last {since}:\n")
    for a in anomalies:
        print(f"  [{a.severity.upper():6}] {a.kind:24}  {a.thread_id[:12]}")
        print(f"           {a.detail}")
    return 0


def cmd_codex_insights_crossref(args: argparse.Namespace) -> int:
    """Cross-reference active sessions to PR/issue references found in metadata."""
    from aragora.codex.insights import crossref_work_board, summarize_patterns

    try:
        since = _parse_since(args.since)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    paths = _resolve_paths(args)
    if not paths.sqlite_path.exists():
        print(
            f"error: Codex Desktop state DB not found at {paths.sqlite_path}",
            file=sys.stderr,
        )
        return 1
    _, pairs = summarize_patterns(
        since=since,
        include_archived=args.include_archived,
        paths=paths,
    )
    crossref = crossref_work_board(pairs)
    if args.json:
        _emit_json([c.to_dict() for c in crossref])
        return 0
    print(f"Cross-references for last {since}:\n")
    for c in crossref:
        if not c.pr_references and not c.issue_references and not c.git_branch:
            continue
        refs = sorted(set(c.pr_references) | set(c.issue_references))
        print(f"  {c.thread_id[:12]}  branch={c.git_branch or '-'}  refs={','.join(refs) or '-'}")
    return 0


def cmd_codex_insights_digest(args: argparse.Namespace) -> int:
    """Build a complete digest. With --emit-receipt, persist to disk."""
    from aragora.codex.insights import (
        build_digest,
        ingest_digest_into_km,
        persist_digest,
    )

    try:
        since = _parse_since(args.since)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    paths = _resolve_paths(args)
    if not paths.sqlite_path.exists():
        print(
            f"error: Codex Desktop state DB not found at {paths.sqlite_path}",
            file=sys.stderr,
        )
        return 1

    digest = build_digest(
        since=since,
        include_archived=args.include_archived,
        paths=paths,
    )

    if args.emit_receipt:
        target = persist_digest(
            digest,
            root=Path(args.receipt_dir) if args.receipt_dir else None,
        )
        signing_note = (
            f"signed (hmac_sha256={digest.hmac_sha256[:12]}...)"
            if digest.hmac_sha256
            else "unsigned (ARAGORA_CONTEXT_SIGNING_KEY not set)"
        )
        km_result: dict[str, Any] | None = None
        if args.ingest_km:
            ok, detail = ingest_digest_into_km(target)
            km_result = {"ok": ok, "detail": detail}
        if args.json:
            payload = digest.to_dict()
            payload["receipt_path"] = str(target)
            payload["signing_note"] = signing_note
            if km_result is not None:
                payload["km_ingest"] = km_result
            _emit_json(payload)
        else:
            print(f"wrote {target} (sha256={digest.sha256[:12]}..., {signing_note})")
            if km_result is not None:
                if km_result["ok"]:
                    print(f"km: ingested ({km_result['detail']})")
                else:
                    print(f"km: skipped — {km_result['detail']}", file=sys.stderr)
    else:
        if args.json:
            _emit_json(digest.to_dict())
        else:
            print(json.dumps(digest.to_dict(), indent=2, sort_keys=True, default=str))
    return 0
