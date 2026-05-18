#!/usr/bin/env python3
"""Apply a downloaded operator-decisions JSON to GitHub via ``gh``.

Closes the loop on the ``/review-queue/packets/[receiptId]`` sign-off
pipeline:

    drop receipt → keyboard sign-off → download
        → ``python3 scripts/apply_operator_decisions.py operator-decisions-*.json --apply``
        → queue clears

The downloaded JSON has the shape ``aragora-operator-decisions/1.0`` and
is produced by ``PacketsClient`` in ``aragora/live/src/app/(app)/
review-queue/packets/[receiptId]/PacketsClient.tsx``. Each entry carries
a per-PR decision plus timing fields. This script verifies the
payload's ``payload_sha256`` binding, then walks ``decisions[]`` and
maps each entry to the corresponding ``gh`` action:

    approve_tier       -> gh pr review N --approve         --body <body>
    approve_downgrade  -> gh pr review N --approve         --body "DOWNGRADED: <body>"
    request_changes    -> gh pr review N --request-changes --body <body>
    reject             -> gh pr close  N                   --comment <body>
    hold_operator      -> no-op (operator-only action)
    null               -> no-op (no decision recorded)

Every applied comment body ends with a binding footer so the audit
trail is recoverable from any PR thread:

    ---
    Applied from operator-decisions <payload[:10]> bound to packet <receipt[:10]>

Defaults to ``--dry-run`` so the first invocation never mutates state.
``--apply`` is required to actually call ``gh``.
Apply mode also refuses draft/closed/merged PR targets and skips a
decision when the same payload/receipt binding footer is already present
on the PR, making replay idempotent.

Hard-coded hold list: PRs explicitly marked as operator-only / held
elsewhere in this repo are hard-skipped regardless of the receipt's
decision for them (the operator may have legitimately recorded a
hypothetical decision, but the CLI refuses to advance held PRs by
even one byte).

Pure stdlib. No ``aragora.*`` imports. No third-party dependencies.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Hard-coded hold list (PR numbers only). The "#7209 lane" and "BC-12
# soak" holds from the standard repo discipline are not PR numbers, so
# they do not appear here. Keep this list in sync with the holds
# enumerated in the operator-decisions ingestion PR body.
HELD_PR_NUMBERS: frozenset[int] = frozenset({4990, 7173, 7215, 7240, 7243, 7245, 7249, 7252})
EXPECTED_SCHEMA_VERSION = "aragora-operator-decisions/1.0"

# The five decision IDs ``PacketDecisionCard`` emits. Mirrors the
# ``PacketDecisionId`` union in
# ``aragora/live/src/components/review-queue/PacketDecisionCard.tsx``.
_APPLY_DECISIONS: frozenset[str] = frozenset(
    {"approve_tier", "approve_downgrade", "request_changes", "reject"}
)

# Status codes used in ``EntryResult.status`` — stable strings so
# ``--json`` consumers can switch on them.
STATUS_APPLIED = "applied"
STATUS_WOULD_APPLY = "would-apply"
STATUS_HELD = "held"
STATUS_SKIPPED = "skipped"
STATUS_DRIFTED = "drifted"
STATUS_FAILED = "failed"
STATUS_NO_DECISION = "no-decision"

_HUMAN_PREFIX: dict[str, str] = {
    STATUS_APPLIED: "APPLIED",
    STATUS_WOULD_APPLY: "WOULD APPLY",
    STATUS_HELD: "HELD",
    STATUS_SKIPPED: "SKIP",
    STATUS_DRIFTED: "DRIFT",
    STATUS_FAILED: "FAIL",
    STATUS_NO_DECISION: "SKIP",
}

_REPO_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GITHUB_PR_URL_RE = re.compile(
    r"^https://github\.com/(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/[0-9]+/?$"
)
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_MISSING = object()


@dataclasses.dataclass(frozen=True)
class DecisionEntry:
    """One row in ``decisions[]`` from the downloaded JSON."""

    pr_number: int
    head_sha: str
    tier: str | None
    decision: str | None
    comment: str
    first_focused_at_utc: str | None
    decided_at_utc: str | None
    decision_seconds: float | None


@dataclasses.dataclass(frozen=True)
class OperatorDecisionsPayload:
    """The top-level downloaded JSON."""

    schema_version: str
    generated_at_utc: str
    receipt_id_hint: str
    receipt_repo: str
    receipt_sha256: str
    receipt_sha256_verified: bool
    decisions: tuple[DecisionEntry, ...]
    payload_sha256: str


@dataclasses.dataclass
class EntryResult:
    """Outcome of processing one ``decisions[]`` entry."""

    pr_number: int
    decision: str | None
    status: str
    reason: str
    gh_command: list[str] | None = None


@dataclasses.dataclass(frozen=True)
class LivePrState:
    """Minimal live PR state required before mutating GitHub."""

    head_sha: str
    is_draft: bool
    state: str
    binding_footer_present: bool


class PayloadValidationError(ValueError):
    """Raised when the downloaded payload is malformed or unsafe to apply."""


def canonical_json(value: Any) -> str:
    """Canonical JSON serialization that matches the TS side.

    The TS-side ``canonicalJson()`` in
    ``aragora/live/src/hooks/useReviewQueueFromPacket.ts`` writes
    sorted-keys, comma+colon separators, no extra whitespace, and
    preserves non-ASCII characters as-is (because ``TextEncoder``
    encodes UTF-8). We replicate that here with ``sort_keys=True``,
    the most-compact separators, and ``ensure_ascii=False`` so
    Unicode survives without ``\\u`` escaping.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def verify_payload_sha256(raw: dict[str, Any]) -> tuple[str, str, bool]:
    """Re-derive the payload SHA-256 and compare against the claimed value.

    Returns ``(claimed, recomputed, matches)``. The recomputation hashes
    the canonical JSON of every top-level field except ``payload_sha256``
    itself — same shape the browser used to compute it on download.
    """

    claimed = str(raw.get("payload_sha256", ""))
    verify_copy: dict[str, Any] = {k: v for k, v in raw.items() if k != "payload_sha256"}
    canonical = canonical_json(verify_copy)
    recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return claimed, recomputed, claimed == recomputed


def compute_file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_receipt_repo(path: Path) -> tuple[str | None, str | None]:
    """Extract the single GitHub ``OWNER/REPO`` from a settlement receipt JSON."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"receipt file is not valid JSON: {exc}"
    repos: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            pr_url = value.get("pr_url")
            if isinstance(pr_url, str):
                match = _GITHUB_PR_URL_RE.fullmatch(pr_url)
                if match:
                    repos.add(match.group("repo"))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(raw)
    if not repos:
        return None, "receipt file does not contain a GitHub pr_url"
    if len(repos) != 1:
        return None, "receipt file contains PR URLs from multiple repositories"
    return next(iter(repos)), None


def _optional_str(value: Any, *, field: str, index: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PayloadValidationError(f"decisions[{index}].{field} must be a string or null")
    return value


def _required_payload_str(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str):
        raise PayloadValidationError(f"{field} must be a string")
    return value


def _required_payload_bool(raw: dict[str, Any], field: str) -> bool:
    value = raw.get(field)
    if not isinstance(value, bool):
        raise PayloadValidationError(f"{field} must be a boolean")
    return value


def _parse_decision_entry(raw: Any, *, index: int) -> DecisionEntry:
    if not isinstance(raw, dict):
        raise PayloadValidationError(f"decisions[{index}] must be a JSON object")
    raw_pr_number = raw.get("pr_number")
    if not isinstance(raw_pr_number, int) or isinstance(raw_pr_number, bool) or raw_pr_number <= 0:
        raise PayloadValidationError(f"decisions[{index}].pr_number must be a positive integer")

    head_sha = raw.get("head_sha")
    if not isinstance(head_sha, str) or not head_sha:
        raise PayloadValidationError(f"decisions[{index}].head_sha must be a non-empty string")

    raw_decision = raw.get("decision", _MISSING)
    if raw_decision is _MISSING:
        raise PayloadValidationError(f"decisions[{index}].decision must be a string or null")
    decision = None if raw_decision is None else raw_decision
    allowed_decisions = _APPLY_DECISIONS | {"hold_operator"}
    if decision is not None:
        if not isinstance(decision, str):
            raise PayloadValidationError(f"decisions[{index}].decision must be a string or null")
        if not decision:
            raise PayloadValidationError(f"decisions[{index}].decision must be a string or null")
        if decision not in allowed_decisions:
            raise PayloadValidationError(
                f"decisions[{index}].decision has unsupported value {decision!r}"
            )

    comment = raw.get("comment", "")
    if not isinstance(comment, str):
        raise PayloadValidationError(f"decisions[{index}].comment must be a string")

    raw_decision_seconds = raw.get("decision_seconds")
    if raw_decision_seconds is None:
        decision_seconds = None
    elif isinstance(raw_decision_seconds, bool) or not isinstance(
        raw_decision_seconds, (int, float)
    ):
        raise PayloadValidationError(
            f"decisions[{index}].decision_seconds must be a number or null"
        )
    else:
        decision_seconds = float(raw_decision_seconds)

    return DecisionEntry(
        pr_number=raw_pr_number,
        head_sha=head_sha,
        tier=_optional_str(raw.get("tier"), field="tier", index=index),
        decision=decision,
        comment=comment,
        first_focused_at_utc=_optional_str(
            raw.get("first_focused_at_utc"), field="first_focused_at_utc", index=index
        ),
        decided_at_utc=_optional_str(
            raw.get("decided_at_utc"), field="decided_at_utc", index=index
        ),
        decision_seconds=decision_seconds,
    )


def parse_payload(raw: dict[str, Any]) -> OperatorDecisionsPayload:
    """Lift the raw dict into the typed payload + entries."""

    decisions_raw = raw.get("decisions")
    if not isinstance(decisions_raw, list):
        raise PayloadValidationError("decisions must be a JSON array")
    decisions = [
        _parse_decision_entry(decision_raw, index=index)
        for index, decision_raw in enumerate(decisions_raw)
    ]
    return OperatorDecisionsPayload(
        schema_version=_required_payload_str(raw, "schema_version"),
        generated_at_utc=_required_payload_str(raw, "generated_at_utc"),
        receipt_id_hint=_required_payload_str(raw, "receipt_id_hint"),
        receipt_repo=_required_payload_str(raw, "receipt_repo"),
        receipt_sha256=_required_payload_str(raw, "receipt_sha256"),
        receipt_sha256_verified=_required_payload_bool(raw, "receipt_sha256_verified"),
        decisions=tuple(decisions),
        payload_sha256=_required_payload_str(raw, "payload_sha256"),
    )


def validate_payload_envelope(payload: OperatorDecisionsPayload) -> str | None:
    """Return a refusal reason when signed payload metadata is not trusted."""

    if payload.schema_version != EXPECTED_SCHEMA_VERSION:
        return (
            "unsupported schema_version "
            f"{payload.schema_version!r}; expected {EXPECTED_SCHEMA_VERSION!r}"
        )
    if not payload.receipt_sha256_verified:
        return "receipt_sha256_verified must be true"
    if not _SHA256_HEX_RE.fullmatch(payload.receipt_sha256):
        return "receipt_sha256 must be a lowercase 64-character SHA-256 hex digest"
    if not _SHA256_HEX_RE.fullmatch(payload.payload_sha256):
        return "payload_sha256 must be a lowercase 64-character SHA-256 hex digest"
    if not _REPO_NAME_RE.fullmatch(payload.receipt_repo):
        return (
            "receipt_repo must be a GitHub repository in OWNER/REPO form "
            "using only letters, numbers, '.', '_' or '-'"
        )
    return None


def _short_sha(sha: str) -> str:
    return sha[:10] if sha else "(none)"


def build_comment_body(entry: DecisionEntry, payload: OperatorDecisionsPayload) -> str:
    """Compose the final body for ``gh pr review/close``.

    For ``approve_downgrade``, prepend ``"DOWNGRADED: "`` so the audit
    trail makes the downgrade obvious without the reader having to
    cross-reference the original packet.

    Every body ends with the binding footer carrying both 10-char SHA
    prefixes.
    """

    main = entry.comment.strip()
    if entry.decision == "approve_downgrade":
        main = ("DOWNGRADED: " + main).rstrip()
    footer = (
        "\n\n---\n"
        f"Applied from operator-decisions {_short_sha(payload.payload_sha256)} "
        f"bound to packet {_short_sha(payload.receipt_sha256)}"
    )
    return (main + footer) if main else footer.lstrip("\n")


def _binding_footer_marker(payload: OperatorDecisionsPayload) -> str:
    return (
        f"Applied from operator-decisions {_short_sha(payload.payload_sha256)} "
        f"bound to packet {_short_sha(payload.receipt_sha256)}"
    )


def _body_collection_contains_marker(value: Any, marker: str) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, dict):
            continue
        body = item.get("body")
        if isinstance(body, str) and marker in body:
            return True
    return False


def _gh_view_pr_state(
    pr_number: int,
    *,
    repo: str,
    binding_footer_marker: str,
) -> tuple[LivePrState | None, str]:
    """Return live PR state required before mutation, or ``(None, err)``."""

    cmd = [
        "gh",
        "pr",
        "view",
        str(pr_number),
        "--repo",
        repo,
        "--json",
        "headRefOid,isDraft,state,comments,reviews",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        return None, f"gh pr view #{pr_number} failed: {stderr[:200]}"
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return None, f"gh pr view #{pr_number} returned non-JSON: {exc}"
    head = data.get("headRefOid")
    if not head or not isinstance(head, str):
        return None, f"gh pr view #{pr_number} returned no headRefOid"
    is_draft = data.get("isDraft")
    if not isinstance(is_draft, bool):
        return None, f"gh pr view #{pr_number} returned no isDraft boolean"
    state = data.get("state")
    if not isinstance(state, str) or not state:
        return None, f"gh pr view #{pr_number} returned no state"
    binding_present = _body_collection_contains_marker(
        data.get("comments"), binding_footer_marker
    ) or _body_collection_contains_marker(data.get("reviews"), binding_footer_marker)
    return LivePrState(
        head_sha=head,
        is_draft=is_draft,
        state=state,
        binding_footer_present=binding_present,
    ), ""


def _plan_action(entry: DecisionEntry, body: str, *, repo: str) -> list[str] | None:
    """Return the ``gh`` argv to apply ``entry``, or ``None`` for no-op."""

    pr = str(entry.pr_number)
    if entry.decision in ("approve_tier", "approve_downgrade"):
        return ["gh", "pr", "review", pr, "--repo", repo, "--approve", "--body", body]
    if entry.decision == "request_changes":
        return ["gh", "pr", "review", pr, "--repo", repo, "--request-changes", "--body", body]
    if entry.decision == "reject":
        return ["gh", "pr", "close", pr, "--repo", repo, "--comment", body]
    return None


def process_entry(
    entry: DecisionEntry,
    payload: OperatorDecisionsPayload,
    *,
    apply: bool,
    only_prs: frozenset[int],
) -> EntryResult:
    """Decide what to do with a single entry — pure logic over fakes-friendly IO."""

    if only_prs and entry.pr_number not in only_prs:
        return EntryResult(
            pr_number=entry.pr_number,
            decision=entry.decision,
            status=STATUS_SKIPPED,
            reason="not in --only-pr filter",
        )

    if entry.pr_number in HELD_PR_NUMBERS:
        return EntryResult(
            pr_number=entry.pr_number,
            decision=entry.decision,
            status=STATUS_HELD,
            reason=f"#{entry.pr_number} is on the hard-coded hold list",
        )

    if entry.decision is None:
        return EntryResult(
            pr_number=entry.pr_number,
            decision=None,
            status=STATUS_NO_DECISION,
            reason="no decision recorded",
        )

    if entry.decision == "hold_operator":
        return EntryResult(
            pr_number=entry.pr_number,
            decision=entry.decision,
            status=STATUS_SKIPPED,
            reason="hold_operator (operator-only action)",
        )

    body = build_comment_body(entry, payload)
    cmd = _plan_action(entry, body, repo=payload.receipt_repo)
    if cmd is None:
        # Defensive — _APPLY_DECISIONS membership already implies this
        # branch is unreachable, but keep the safety net for future
        # decision IDs that get added without a mapping.
        return EntryResult(
            pr_number=entry.pr_number,
            decision=entry.decision,
            status=STATUS_SKIPPED,
            reason=f"no gh action mapped for {entry.decision!r}",
        )

    if not apply:
        return EntryResult(
            pr_number=entry.pr_number,
            decision=entry.decision,
            status=STATUS_WOULD_APPLY,
            reason="dry-run",
            gh_command=cmd,
        )

    live_state, err = _gh_view_pr_state(
        entry.pr_number,
        repo=payload.receipt_repo,
        binding_footer_marker=_binding_footer_marker(payload),
    )
    if live_state is None:
        return EntryResult(
            pr_number=entry.pr_number,
            decision=entry.decision,
            status=STATUS_FAILED,
            reason=err or "gh pr view failed",
        )
    if live_state.binding_footer_present:
        return EntryResult(
            pr_number=entry.pr_number,
            decision=entry.decision,
            status=STATUS_SKIPPED,
            reason="binding footer already present; refusing duplicate apply",
        )
    if live_state.state != "OPEN":
        return EntryResult(
            pr_number=entry.pr_number,
            decision=entry.decision,
            status=STATUS_FAILED,
            reason=f"PR state is {live_state.state}; refusing mutation",
        )
    if live_state.is_draft:
        return EntryResult(
            pr_number=entry.pr_number,
            decision=entry.decision,
            status=STATUS_FAILED,
            reason="PR is draft; refusing mutation",
        )
    if live_state.head_sha != entry.head_sha:
        return EntryResult(
            pr_number=entry.pr_number,
            decision=entry.decision,
            status=STATUS_DRIFTED,
            reason=(f"HEAD DRIFT: expected {entry.head_sha[:10]}, got {live_state.head_sha[:10]}"),
        )

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        return EntryResult(
            pr_number=entry.pr_number,
            decision=entry.decision,
            status=STATUS_FAILED,
            reason=f"gh failed: {stderr[:200]}",
            gh_command=cmd,
        )

    return EntryResult(
        pr_number=entry.pr_number,
        decision=entry.decision,
        status=STATUS_APPLIED,
        reason="ok",
        gh_command=cmd,
    )


def _print_human(results: Sequence[EntryResult], applied: bool) -> None:
    for r in results:
        prefix = _HUMAN_PREFIX.get(r.status, r.status.upper())
        decision_label = r.decision if r.decision else "(none)"
        print(f"{prefix:11s} #{r.pr_number:>5d}  {decision_label:20s}  — {r.reason}")
    if not applied:
        print("\nDRY RUN — no PRs were touched. Re-run with --apply to commit.")


def _print_json(
    payload: OperatorDecisionsPayload,
    results: Sequence[EntryResult],
    applied: bool,
) -> None:
    print(
        json.dumps(
            {
                "payload_sha256": payload.payload_sha256,
                "receipt_sha256": payload.receipt_sha256,
                "applied": applied,
                "results": [dataclasses.asdict(r) for r in results],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apply_operator_decisions.py",
        description=(
            "Apply a downloaded aragora-operator-decisions/1.0 JSON to "
            "GitHub via `gh`. Defaults to --dry-run."
        ),
    )
    parser.add_argument(
        "decisions_path",
        type=Path,
        help="Path to a downloaded operator-decisions-*.json file.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Mutate GitHub state. Without this flag, nothing is sent.",
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        help=(
            "Original settlement receipt JSON used by the packet UI. "
            "Required with --apply so the CLI independently verifies "
            "receipt_sha256 before mutating GitHub."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Explicit dry-run (this is the default behaviour when --apply is omitted)."),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit per-entry results as JSON to stdout.",
    )
    parser.add_argument(
        "--only-pr",
        action="append",
        type=int,
        default=[],
        metavar="N",
        help="Apply only to the listed PR numbers (repeatable).",
    )
    parser.add_argument(
        "--skip-hold-decisions",
        action="store_true",
        default=True,
        help=(
            "Always honoured; the hold list is hard-coded and held PRs "
            "are skipped regardless of this flag."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run are mutually exclusive.", file=sys.stderr)
        return 2

    if shutil.which("gh") is None:
        print(
            "ERROR: `gh` CLI not found on PATH — install gh (https://cli.github.com) and retry.",
            file=sys.stderr,
        )
        return 2

    decisions_path: Path = args.decisions_path
    if not decisions_path.exists():
        print(f"ERROR: file not found: {decisions_path}", file=sys.stderr)
        return 2

    try:
        raw = json.loads(decisions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON ({exc}): {decisions_path}", file=sys.stderr)
        return 2

    if not isinstance(raw, dict):
        print("ERROR: receipt root must be a JSON object", file=sys.stderr)
        return 2

    claimed, recomputed, matches = verify_payload_sha256(raw)
    if not matches:
        print(
            "ERROR: payload_sha256 mismatch — claimed "
            f"{_short_sha(claimed)} vs recomputed {_short_sha(recomputed)}. "
            "Refusing to apply.",
            file=sys.stderr,
        )
        return 2

    try:
        payload = parse_payload(raw)
    except PayloadValidationError as exc:
        print(f"ERROR: malformed operator-decisions payload: {exc}", file=sys.stderr)
        return 2
    if reason := validate_payload_envelope(payload):
        print(f"ERROR: refusing operator-decisions payload: {reason}", file=sys.stderr)
        return 2

    apply = bool(args.apply)
    if apply:
        receipt_path: Path | None = args.receipt_path
        if receipt_path is None:
            print(
                "ERROR: --apply requires --receipt-path for receipt SHA verification.",
                file=sys.stderr,
            )
            return 2
        if not receipt_path.exists():
            print(f"ERROR: receipt file not found: {receipt_path}", file=sys.stderr)
            return 2
        recomputed_receipt_sha256 = compute_file_sha256(receipt_path)
        if recomputed_receipt_sha256 != payload.receipt_sha256:
            print(
                "ERROR: receipt_sha256 mismatch — payload claims "
                f"{_short_sha(payload.receipt_sha256)} but {receipt_path} hashes to "
                f"{_short_sha(recomputed_receipt_sha256)}. Refusing to apply.",
                file=sys.stderr,
            )
            return 2
        receipt_repo, repo_error = extract_receipt_repo(receipt_path)
        if repo_error:
            print(f"ERROR: cannot verify receipt repo: {repo_error}", file=sys.stderr)
            return 2
        if receipt_repo != payload.receipt_repo:
            print(
                "ERROR: receipt_repo mismatch — payload claims "
                f"{payload.receipt_repo!r} but {receipt_path} is for {receipt_repo!r}. "
                "Refusing to apply.",
                file=sys.stderr,
            )
            return 2
    only_prs: frozenset[int] = frozenset(args.only_pr)

    results = [
        process_entry(entry, payload, apply=apply, only_prs=only_prs) for entry in payload.decisions
    ]

    if args.json:
        _print_json(payload, results, applied=apply)
    else:
        _print_human(results, applied=apply)

    any_failed = any(r.status == STATUS_FAILED for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
