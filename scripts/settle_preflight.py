#!/usr/bin/env python3
"""Read-only settlement gate preflight for conductor queue selection."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.settle_one_pr as settle_one_pr

RECHECK_RULE = "recheck on next origin/main push; never poll in a loop."

MAIN_RED_HALT = "MAIN_RED_HALT"
DRAFT_SKIP = "DRAFT_SKIP"
HUMAN_GATED = "HUMAN_GATED"
HEAD_BLOCKED = "HEAD_BLOCKED"
GITHUB_UNSTABLE = "GITHUB_UNSTABLE"
READY = "READY"


@dataclass(frozen=True)
class PreflightResult:
    pr_number: int
    verdict: str
    action: str
    recheck_rule: str
    title: str = ""
    head_sha: str = ""
    tier: int | None = None
    mergeable: str = ""
    merge_state: str = ""
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pr_number(*payloads: dict[str, Any]) -> int:
    for payload in payloads:
        value = settle_one_pr._coerce_int(payload.get("pr_number") or payload.get("number"))
        if value is not None:
            return value
    return 0


def _title(entry: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(metadata.get("title") or entry.get("title") or "")


def _head_sha(entry: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(metadata.get("headRefOid") or entry.get("head_sha") or "")


def _mergeable(metadata: dict[str, Any], entry: dict[str, Any]) -> str:
    return str(metadata.get("mergeable") or entry.get("mergeable") or "").upper()


def _merge_state(metadata: dict[str, Any], entry: dict[str, Any]) -> str:
    return str(metadata.get("mergeStateStatus") or entry.get("mergeStateStatus") or "").upper()


def _tier(entry: dict[str, Any]) -> int | None:
    return settle_one_pr._coerce_int(entry.get("tier"))


def _human_preapproval_recorded(entry: dict[str, Any]) -> bool:
    if bool(entry.get("human_preapproval_recorded")):
        return True
    settlement = entry.get("settlement_creator_pin")
    if isinstance(settlement, dict):
        return bool(settlement.get("verified") and settlement.get("trusted_creator"))
    return False


def _model_authorized(entry: dict[str, Any]) -> bool:
    return (
        bool(entry.get("admin_squash_allowed"))
        and str(entry.get("status") or "") == "satisfied"
        and str(entry.get("verdict") or "") == "admin_squash_allowed"
    )


def _head_drift_reason(entry: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    packet_head = str(entry.get("head_sha") or "")
    live_head = str(metadata.get("headRefOid") or "")
    if packet_head and live_head and packet_head != live_head:
        return f"head drift: packet {packet_head} live {live_head}"
    return None


def classify_pr(
    *,
    entry: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    main_red: bool = False,
) -> PreflightResult:
    """Classify one PR using read-only metadata and merge-packet fields."""
    entry = dict(entry or {})
    metadata = dict(metadata or {})
    pr_number = _pr_number(entry, metadata)
    tier = _tier(entry)
    mergeable = _mergeable(metadata, entry)
    merge_state = _merge_state(metadata, entry)
    title = _title(entry, metadata)
    head_sha = _head_sha(entry, metadata)

    def result(verdict: str, action: str, reasons: tuple[str, ...]) -> PreflightResult:
        return PreflightResult(
            pr_number=pr_number,
            verdict=verdict,
            action=action,
            recheck_rule=RECHECK_RULE,
            title=title,
            head_sha=head_sha,
            tier=tier,
            mergeable=mergeable,
            merge_state=merge_state,
            reasons=reasons,
        )

    if main_red:
        return result(
            MAIN_RED_HALT,
            "halt conductor work and enter main-red incident mode",
            ("origin/main required checks are not green",),
        )

    if bool(metadata.get("isDraft") or entry.get("isDraft")):
        return result(
            DRAFT_SKIP,
            "skip this PR until it is marked ready for review",
            ("PR is draft",),
        )

    head_drift = _head_drift_reason(entry, metadata)
    if head_drift:
        return result(
            HEAD_BLOCKED,
            "park this head until the merge packet is regenerated for the live head",
            (head_drift,),
        )

    policy_reasons = settle_one_pr.policy_exclusion_reasons(
        entry, policy_metadata={pr_number: metadata}
    )
    tier_human = tier is not None and tier > 2
    requires_human_risk = bool(
        entry.get("requires_human_risk_settlement")
    ) and not _human_preapproval_recorded(entry)

    if merge_state in {"DIRTY", "BEHIND"} or mergeable == "CONFLICTING":
        return result(
            HEAD_BLOCKED,
            "park this head until conflicts, behind-base state, or current-head blocker clears",
            (f"mergeable={mergeable or 'unknown'} mergeStateStatus={merge_state or 'unknown'}",),
        )

    requires_human_preapproval = bool(
        entry.get("requires_human_preapproval")
    ) and not _human_preapproval_recorded(entry)
    recorded_human_settlement = _human_preapproval_recorded(entry)
    policy_gate_reasons = [
        reason
        for reason in policy_reasons
        if reason != "dirty/conflicting PR"
        and not (recorded_human_settlement and reason == "requires_human_risk_settlement=true")
    ]
    if tier_human or requires_human_risk or requires_human_preapproval or policy_gate_reasons:
        reasons = []
        if tier_human:
            reasons.append(f"Tier {tier}")
        if requires_human_risk:
            reasons.append("requires_human_risk_settlement=true without recorded preapproval")
        if requires_human_preapproval:
            reasons.append("requires_human_preapproval=true without recorded preapproval")
        for reason in policy_gate_reasons:
            if reason not in reasons:
                reasons.append(reason)
        return result(
            HUMAN_GATED,
            "stop and request exact-head human settlement or operator decision before evidence or merge",
            tuple(reasons),
        )

    entry_blockers = settle_one_pr.entry_blockers(entry) if entry else []
    if recorded_human_settlement:
        entry_blockers = [
            blocker
            for blocker in entry_blockers
            if blocker != "requires_human_risk_settlement=true"
        ]
    if entry_blockers:
        return result(
            HEAD_BLOCKED,
            "park this head until the merge-packet blockers are resolved",
            tuple(entry_blockers),
        )

    model_authorized = _model_authorized(entry)
    if model_authorized and (merge_state not in {"CLEAN", "BLOCKED"} or mergeable != "MERGEABLE"):
        return result(
            GITHUB_UNSTABLE,
            "do not merge; wait for GitHub merge state to become settlement-stable",
            (
                f"model-authorized but mergeable={mergeable or 'unknown'} mergeStateStatus={merge_state or 'unknown'}",
            ),
        )

    if mergeable == "MERGEABLE" and merge_state in {"CLEAN", "BLOCKED"} and model_authorized:
        return result(
            READY,
            "run exact-head normal protected squash merge after one final live-state check",
            ("model-authorized and settlement-stable",),
        )

    return result(
        HEAD_BLOCKED,
        "park this head until it has a satisfied model packet and stable GitHub merge state",
        (f"status={entry.get('status') or 'unknown'} verdict={entry.get('verdict') or 'unknown'}",),
    )


def _packet_entry(packet: dict[str, Any], pr_number: int) -> dict[str, Any]:
    entries = packet.get("entries")
    if not isinstance(entries, list):
        return {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if _pr_number(entry) == pr_number:
            return entry
    return {}


def _load_single(
    cwd: Path, pr_number: int, repo: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = settle_one_pr._load_single_pr_packet(cwd=cwd, pr=pr_number, repo=repo)
    entry = _packet_entry(packet, pr_number)
    metadata, _command = settle_one_pr.load_pr_policy_metadata(cwd, pr_number, repo=repo)
    live_payload, _live_command = settle_one_pr._run_json(
        settle_one_pr._with_repo(
            [
                "gh",
                "pr",
                "view",
                str(pr_number),
                "--json",
                "number,title,headRefName,headRefOid,isDraft,mergeable,mergeStateStatus,files",
            ],
            repo,
        ),
        cwd=cwd,
        timeout=settle_one_pr.GH_METADATA_TIMEOUT_SECONDS,
    )
    if isinstance(live_payload, dict):
        metadata.update(live_payload)
    return entry, metadata


def _classify_queue(cwd: Path, repo: str | None, limit: int) -> list[PreflightResult]:
    metadata_by_pr, _command = settle_one_pr.load_open_pr_metadata(cwd, limit=limit, repo=repo)
    results: list[PreflightResult] = []
    for pr_number, metadata in sorted(metadata_by_pr.items()):
        entry: dict[str, Any] = {}
        if not metadata.get("isDraft"):
            try:
                packet = settle_one_pr._load_single_pr_packet(cwd=cwd, pr=pr_number, repo=repo)
                entry = _packet_entry(packet, pr_number)
            except RuntimeError as exc:
                entry = {
                    "pr_number": pr_number,
                    "title": metadata.get("title"),
                    "head_sha": metadata.get("headRefOid"),
                    "status": "packet_unavailable",
                    "verdict": "packet_unavailable",
                    "reasons": [str(exc)],
                }
        results.append(classify_pr(entry=entry, metadata=metadata))
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pr", type=int, help="Classify one pull request")
    target.add_argument("--queue", action="store_true", help="Classify open pull requests")
    parser.add_argument("--repo", default=None, help="GitHub repo owner/name")
    parser.add_argument("--limit", type=int, default=50, help="Open-PR limit for --queue")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument(
        "--main-red",
        action="store_true",
        help="Classify all targets as MAIN_RED_HALT after an external main-health check",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cwd = Path.cwd()
    if args.pr is not None:
        entry, metadata = _load_single(cwd, args.pr, args.repo)
        results = [classify_pr(entry=entry, metadata=metadata, main_red=args.main_red)]
    else:
        results = _classify_queue(cwd, args.repo, args.limit)
        if args.main_red:
            results = [
                classify_pr(
                    entry={
                        "pr_number": item.pr_number,
                        "title": item.title,
                        "head_sha": item.head_sha,
                    },
                    metadata={},
                    main_red=True,
                )
                for item in results
            ]

    payload = {"results": [result.to_dict() for result in results]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for result in results:
            print(f"#{result.pr_number} {result.verdict}: {result.action} ({result.recheck_rule})")
            for reason in result.reasons:
                print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
