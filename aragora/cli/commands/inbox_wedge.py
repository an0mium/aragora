"""
CLI for the inbox trust wedge.

Provides a founder-usable approval loop for receipt-gated inbox actions:
- create a persisted signed receipt from a debated triage decision
- review it via approve/reject/edit/skip
- inspect stored state
- execute only after approval
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from aragora.inbox.trust_wedge import (
    ActionIntent,
    InboxWedgeAction,
    ReceiptState,
    TriageDecision,
    get_inbox_trust_wedge_service,
    get_inbox_trust_wedge_store,
)


def _coerce_local_datetime(value: Any) -> datetime | None:
    """Convert stored wedge timestamps into the operator's local time."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone() if value.tzinfo is not None else value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone() if parsed.tzinfo is not None else parsed
    return None


def add_inbox_wedge_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "inbox-wedge",
        help="Receipt-gated inbox trust wedge commands",
        description=(
            "Create, review, inspect, and execute inbox trust wedge receipts "
            "for narrow Gmail triage actions."
        ),
    )
    sub = parser.add_subparsers(dest="inbox_wedge_command")

    create_parser = sub.add_parser("create", help="Create a persisted signed wedge receipt")
    create_parser.add_argument(
        "--provider", default="gmail", help="Email provider (default: gmail)"
    )
    create_parser.add_argument("--user-id", required=True, help="Email connector user/account id")
    create_parser.add_argument("--message-id", required=True, help="Provider message id")
    create_parser.add_argument(
        "--action",
        required=True,
        choices=[action.value for action in InboxWedgeAction],
        help="Final triage action",
    )
    create_parser.add_argument(
        "--confidence",
        type=float,
        required=True,
        help="Synthesizer confidence for the final decision",
    )
    create_parser.add_argument(
        "--rationale",
        required=True,
        help="Synthesized rationale for the final action",
    )
    create_parser.add_argument(
        "--provider-route",
        default="openrouter-fallback",
        help="Provider route metadata to persist",
    )
    create_parser.add_argument("--debate-id", help="Optional debate id")
    create_parser.add_argument("--dissent-summary", default="", help="Optional dissent summary")
    create_parser.add_argument("--label-id", help="Required when action=label")
    create_parser.add_argument("--content-hash", help="Precomputed content hash")
    create_parser.add_argument("--content-text", help="Inline message content to hash")
    create_parser.add_argument("--content-file", help="File containing message content to hash")
    create_parser.add_argument(
        "--expires-in-hours",
        type=float,
        default=24.0,
        help="Receipt lifetime in hours (default: 24)",
    )
    create_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve only if the wedge policy marks the decision eligible",
    )
    create_parser.set_defaults(func=cmd_inbox_wedge_create)

    review_parser = sub.add_parser("review", help="Approve, reject, edit, or skip a receipt")
    review_parser.add_argument("receipt_id", help="Receipt id")
    review_parser.add_argument(
        "--choice",
        choices=["approve", "reject", "edit", "skip"],
        help="Review choice. If omitted, prompt interactively.",
    )
    review_parser.add_argument(
        "--action",
        choices=[action.value for action in InboxWedgeAction],
        help="Edited action when --choice edit",
    )
    review_parser.add_argument("--rationale", help="Edited rationale when --choice edit")
    review_parser.add_argument("--label-id", help="Edited label id when --choice edit")
    review_parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    review_parser.set_defaults(func=cmd_inbox_wedge_review)

    show_parser = sub.add_parser("show", help="Show a receipt envelope")
    show_parser.add_argument("receipt_id", help="Receipt id")
    show_parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    show_parser.set_defaults(func=cmd_inbox_wedge_show)

    list_parser = sub.add_parser("list", help="List recent wedge receipts")
    list_parser.add_argument(
        "--state",
        choices=[state.value for state in ReceiptState],
        help="Optional state filter",
    )
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum results")
    list_parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    list_parser.set_defaults(func=cmd_inbox_wedge_list)

    execute_parser = sub.add_parser("execute", help="Execute an approved wedge receipt")
    execute_parser.add_argument("receipt_id", help="Receipt id")
    execute_parser.set_defaults(func=cmd_inbox_wedge_execute)

    report_parser = sub.add_parser("report", help="Summarize recent wedge receipt telemetry")
    report_parser.add_argument(
        "--state",
        choices=[state.value for state in ReceiptState],
        help="Optional state filter",
    )
    report_parser.add_argument("--limit", type=int, default=200, help="Maximum results")
    report_parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    report_parser.set_defaults(func=cmd_inbox_wedge_report)

    export_parser = sub.add_parser("export", help="Export recent wedge receipts for labeling")
    export_parser.add_argument("output_path", help="Destination file path")
    export_parser.add_argument(
        "--state",
        choices=[state.value for state in ReceiptState],
        help="Optional state filter",
    )
    export_parser.add_argument("--limit", type=int, default=200, help="Maximum results")
    export_parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Export format (default: jsonl)",
    )
    export_parser.set_defaults(func=cmd_inbox_wedge_export)

    parser.set_defaults(func=cmd_inbox_wedge, _parser=parser)


def _dump(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _load_content_hash(args: argparse.Namespace) -> str:
    if args.content_hash:
        return args.content_hash
    if args.content_text:
        return ActionIntent.compute_content_hash(args.content_text)
    if args.content_file:
        text = Path(args.content_file).read_text(encoding="utf-8")
        return ActionIntent.compute_content_hash(text)
    raise ValueError("One of --content-hash, --content-text, or --content-file is required")


def _prompt(prompt: str) -> str:
    print(prompt, end="", file=sys.stderr, flush=True)
    return input().strip()


def _format_timestamp(value: Any) -> str:
    parsed = _coerce_local_datetime(value)
    if parsed is not None:
        return parsed.strftime("%Y-%m-%d %H:%M")
    if value in (None, ""):
        return "N/A"
    return str(value)


def _short(value: str | None, width: int = 14) -> str:
    text = value or ""
    return text[: width - 2] + ".." if len(text) > width else text


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _print_envelope_summary(envelope: Any) -> None:
    receipt = envelope.receipt
    decision = envelope.decision
    intent = envelope.intent

    print("Inbox Trust Wedge Receipt")
    print("=" * 60)
    print(f"Receipt ID:    {receipt.receipt_id}")
    print(f"Debate ID:     {envelope.debate_id or 'N/A'}")
    print(f"State:         {receipt.state.value}")
    print(f"Review:        {envelope.review_choice or 'pending'}")
    print(f"Created:       {_format_timestamp(receipt.created_at)}")
    print(f"Expires:       {_format_timestamp(receipt.expires_at)}")
    print(f"Provider:      {intent.provider}")
    print(f"User:          {intent.user_id or 'N/A'}")
    print(f"Message ID:    {intent.message_id}")
    print(f"Route:         {envelope.provider_route}")
    print("")
    print("--- Decision ---")
    print(f"Action:        {intent.action.value}")
    print(f"Confidence:    {float(decision.confidence):.0%}")
    print(f"Blocked:       {_yes_no(bool(decision.blocked_by_policy))}")
    print(f"Auto-Eligible: {_yes_no(bool(decision.auto_approval_eligible))}")
    print(f"Tier:          {decision.execution_tier}")
    if decision.label_id:
        print(f"Label ID:      {decision.label_id}")
    if intent.synthesized_rationale:
        print(f"Rationale:     {intent.synthesized_rationale}")
    if decision.dissent_summary:
        print(f"Dissent:       {decision.dissent_summary}")
    if decision.escalation_reasons:
        print(f"Escalations:   {', '.join(decision.escalation_reasons)}")
    if decision.suppressed_diagnostics_count:
        print(f"Suppressed:    {decision.suppressed_diagnostics_count}")
    print("")
    print("--- Execution ---")
    print(f"Execution Cnt: {receipt.execution_count}")
    print(f"Executed At:   {_format_timestamp(receipt.executed_at)}")
    if receipt.last_error:
        print(f"Last Error:    {receipt.last_error}")
    execution_result = getattr(envelope, "execution_result", None)
    if execution_result:
        success = execution_result.get("success")
        print(f"Result:        {'success' if success else 'failure'}")


def _print_receipt_list(receipts: list[Any]) -> None:
    if not receipts:
        print("No inbox trust wedge receipts found.")
        return
    print(
        f"{'ID':<14} {'STATE':<9} {'ACTION':<8} {'CONF':>6} "
        f"{'BLOCK':<6} {'ROUTE':<10} {'MSG':<14} {'CREATED':<16}"
    )
    print("-" * 92)
    for envelope in receipts:
        receipt = envelope.receipt
        decision = envelope.decision
        intent = envelope.intent
        print(
            f"{_short(receipt.receipt_id, 14):<14} "
            f"{receipt.state.value:<9} "
            f"{intent.action.value:<8} "
            f"{float(decision.confidence):>5.0%} "
            f"{_yes_no(bool(decision.blocked_by_policy)):<6} "
            f"{_short(envelope.provider_route, 10):<10} "
            f"{_short(intent.message_id, 14):<14} "
            f"{_format_timestamp(receipt.created_at):<16}"
        )
    print(f"\n{len(receipts)} receipt(s) shown.")


def _print_report_summary(summary: dict[str, Any]) -> None:
    print("Inbox Trust Wedge Report")
    print("=" * 60)
    print(f"Total Receipts:           {summary['total_receipts']}")
    print(f"Executed:                 {summary['executed_count']}")
    print(f"Eligible Auto-Approve:    {summary['eligible_auto_approve_count']}")
    print(f"Auto-Approved:            {summary['auto_approved_count']}")

    avg_conf = summary["average_confidence"]
    avg_lat = summary["average_latency_seconds"]
    avg_cost = summary["average_cost_usd"]
    print(
        f"Average Confidence:       {f'{float(avg_conf):.0%}' if avg_conf is not None else 'N/A'}"
    )
    print(f"Average Latency Seconds:  {avg_lat if avg_lat is not None else 'N/A'}")
    print(f"Average Cost USD:         {avg_cost if avg_cost is not None else 'N/A'}")
    print("")
    print(f"States:                   {summary['states']}")
    print(f"Actions:                  {summary['actions']}")
    print(f"Review Choices:           {summary['review_choices']}")
    print(f"Provider Routes:          {summary['provider_routes']}")


def cmd_inbox_wedge(args: argparse.Namespace) -> None:
    parser = getattr(args, "_parser", None)
    if parser is not None:
        parser.print_help()


def cmd_inbox_wedge_create(args: argparse.Namespace) -> None:
    service = get_inbox_trust_wedge_service()
    content_hash = _load_content_hash(args)
    intent = ActionIntent.create(
        provider=args.provider,
        user_id=args.user_id,
        message_id=args.message_id,
        action=args.action,
        content_hash=content_hash,
        synthesized_rationale=args.rationale,
        confidence=args.confidence,
        provider_route=args.provider_route,
        debate_id=args.debate_id,
        label_id=args.label_id,
    )
    decision = TriageDecision.create(
        final_action=args.action,
        confidence=args.confidence,
        dissent_summary=args.dissent_summary,
        label_id=args.label_id,
    )
    envelope = service.create_receipt(
        intent,
        decision,
        expires_in_hours=args.expires_in_hours,
        auto_approve=args.auto_approve,
    )
    _dump(envelope.to_dict())


def cmd_inbox_wedge_review(args: argparse.Namespace) -> None:
    service = get_inbox_trust_wedge_service()
    choice = args.choice or _prompt("Choice [approve/reject/edit/skip]: ")
    normalized_choice = choice.strip().lower()
    edited_action = args.action
    edited_rationale = args.rationale
    label_id = args.label_id

    if normalized_choice == "edit":
        if not edited_action and not edited_rationale and not label_id:
            edited_action = _prompt("New action (blank to keep current): ") or None
            edited_rationale = _prompt("New rationale (blank to keep current): ") or None
            label_id = _prompt("New label id (blank to keep current): ") or None

    envelope = service.review_receipt(
        args.receipt_id,
        choice=normalized_choice,
        edited_action=edited_action,
        edited_rationale=edited_rationale,
        label_id=label_id,
    )
    if getattr(args, "json", False):
        _dump(envelope.to_dict())
        return

    _print_envelope_summary(envelope)
    print("")
    print("--- Review Command ---")
    print(f"Selected:      {normalized_choice}")
    if normalized_choice == "skip":
        print("Effect:        no state change")
    elif envelope.receipt.state is ReceiptState.APPROVED:
        print(f"Next:          aragora inbox-wedge execute {envelope.receipt.receipt_id}")


def cmd_inbox_wedge_show(args: argparse.Namespace) -> None:
    store = get_inbox_trust_wedge_store()
    envelope = store.get_receipt(args.receipt_id)
    if envelope is None:
        raise SystemExit(f"Receipt not found: {args.receipt_id}")
    if getattr(args, "json", False):
        _dump(envelope.to_dict())
    else:
        _print_envelope_summary(envelope)


def cmd_inbox_wedge_list(args: argparse.Namespace) -> None:
    store = get_inbox_trust_wedge_store()
    state = ReceiptState(args.state) if args.state else None
    receipts = store.list_receipts(state=state, limit=args.limit)
    if getattr(args, "json", False):
        _dump({"receipts": [receipt.to_dict() for receipt in receipts]})
    else:
        _print_receipt_list(receipts)


def cmd_inbox_wedge_execute(args: argparse.Namespace) -> None:
    service = get_inbox_trust_wedge_service()
    result = asyncio.run(service.execute_receipt(args.receipt_id))
    _dump(result.to_dict())


def _load_receipts_for_cli(
    *,
    state_value: str | None,
    limit: int,
) -> list[Any]:
    store = get_inbox_trust_wedge_store()
    state = ReceiptState(state_value) if state_value else None
    return store.list_receipts(state=state, limit=limit)


def _rounded_average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def cmd_inbox_wedge_report(args: argparse.Namespace) -> None:
    receipts = _load_receipts_for_cli(state_value=args.state, limit=args.limit)

    state_counts = Counter(receipt.receipt.state.value for receipt in receipts)
    action_counts = Counter(receipt.intent.action.value for receipt in receipts)
    review_counts = Counter(receipt.review_choice or "pending" for receipt in receipts)
    route_counts = Counter(receipt.provider_route for receipt in receipts)

    confidences = [float(receipt.decision.confidence) for receipt in receipts]
    latencies = [
        float(receipt.decision.latency_seconds)
        for receipt in receipts
        if receipt.decision.latency_seconds is not None
    ]
    costs = [
        float(receipt.decision.cost_usd)
        for receipt in receipts
        if receipt.decision.cost_usd is not None
    ]

    executed_count = sum(
        1 for receipt in receipts if receipt.receipt.state is ReceiptState.EXECUTED
    )
    auto_approved_count = sum(1 for receipt in receipts if receipt.review_choice == "auto_approve")
    eligible_auto_approve_count = sum(
        1 for receipt in receipts if receipt.decision.auto_approval_eligible
    )

    summary = {
        "total_receipts": len(receipts),
        "states": dict(state_counts),
        "actions": dict(action_counts),
        "review_choices": dict(review_counts),
        "provider_routes": dict(route_counts),
        "executed_count": executed_count,
        "auto_approved_count": auto_approved_count,
        "eligible_auto_approve_count": eligible_auto_approve_count,
        "average_confidence": _rounded_average(confidences),
        "average_latency_seconds": _rounded_average(latencies),
        "average_cost_usd": _rounded_average(costs),
    }
    if getattr(args, "json", False):
        _dump(summary)
    else:
        _print_report_summary(summary)


def cmd_inbox_wedge_export(args: argparse.Namespace) -> None:
    receipts = _load_receipts_for_cli(state_value=args.state, limit=args.limit)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        output_path.write_text(
            json.dumps([receipt.to_dict() for receipt in receipts], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        lines = [json.dumps(receipt.to_dict(), sort_keys=True) for receipt in receipts]
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    _dump(
        {
            "output_path": str(output_path),
            "format": args.format,
            "count": len(receipts),
        }
    )


__all__ = [
    "add_inbox_wedge_parser",
    "cmd_inbox_wedge",
    "cmd_inbox_wedge_create",
    "cmd_inbox_wedge_execute",
    "cmd_inbox_wedge_export",
    "cmd_inbox_wedge_list",
    "cmd_inbox_wedge_report",
    "cmd_inbox_wedge_review",
    "cmd_inbox_wedge_show",
]
