"""Benchmark baseline vs staged inbox-triage profiles on a fixed fixture set."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aragora.inbox.auto_approval import AutoApprovalPolicy
from aragora.inbox.triage_diagnostics import TriageRunDiagnostics
from aragora.inbox.triage_runner import InboxTriageRunner
from aragora.inbox.trust_wedge import ReceiptState, TriageDecision


DEFAULT_MIN_AGREEMENT_RATE = 0.95
DEFAULT_MIN_LATENCY_IMPROVEMENT_PCT = 40.0
DEFAULT_MAX_BLOCKED_RATE_DELTA_PP = 5.0


@dataclass
class ProfileRunResult:
    profile: str
    fixture_path: str
    message_count: int
    total_duration_seconds: float
    diagnostics_artifact_dir: str
    meta: dict[str, Any]
    decisions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FixtureGmailConnector:
    """Minimal connector that serves a fixed list of messages."""

    connector_id = "fixture-gmail"
    user_id = "fixture-user"

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = [dict(message) for message in messages]
        self._by_id = {str(message["id"]): dict(message) for message in self._messages}

    async def list_messages(self, *, query: str, max_results: int):
        del query
        return [str(message["id"]) for message in self._messages[:max_results]], None

    async def get_message(self, message_id: str) -> dict[str, Any]:
        return dict(self._by_id[message_id])


class BenchmarkWedgeService:
    """Receipt stub for offline triage benchmarking."""

    def __init__(self, policy: AutoApprovalPolicy | None = None) -> None:
        self._policy = policy or AutoApprovalPolicy()

    def create_receipt(
        self,
        intent: Any,
        decision: TriageDecision,
        *,
        auto_approve: bool = False,
    ) -> Any:
        eligible = self._policy.can_auto_approve(decision)
        state = ReceiptState.APPROVED if auto_approve and eligible else ReceiptState.CREATED
        updated = replace(
            decision,
            receipt_id=f"fixture-{intent.message_id}",
            auto_approval_eligible=eligible,
            receipt_state=state.value,
            intent=intent,
        )
        return SimpleNamespace(
            intent=intent,
            decision=updated,
            receipt=SimpleNamespace(receipt_id=updated.receipt_id, state=state),
            provider_route=updated.provider_route,
        )

    async def execute_receipt(self, receipt_id: str) -> None:
        del receipt_id
        return None


def load_fixture_messages(path: str | Path) -> list[dict[str, Any]]:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected non-empty message list in {fixture_path}")

    messages: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Fixture item {index} in {fixture_path} is not an object")
        message_id = str(item.get("id", "")).strip()
        if not message_id:
            raise ValueError(f"Fixture item {index} in {fixture_path} is missing id")
        subject = str(item.get("subject", "")).strip()
        sender = str(item.get("from_address", item.get("sender", ""))).strip()
        body_text = str(item.get("body_text", item.get("body", item.get("snippet", ""))))
        messages.append(
            {
                "id": message_id,
                "subject": subject or "(no subject)",
                "from_address": sender or "(unknown)",
                "snippet": str(item.get("snippet", body_text[:120])),
                "body_text": body_text,
            }
        )
    return messages


def _summarize_decision(decision: TriageDecision, policy: AutoApprovalPolicy) -> dict[str, Any]:
    message_id = ""
    subject = "(unknown)"
    if decision.intent is not None:
        message_id = decision.intent.message_id
        subject = getattr(decision.intent, "_subject", subject)
    return {
        "message_id": message_id,
        "subject": subject,
        "final_action": str(decision.final_action.value),
        "confidence": float(decision.confidence),
        "blocked_by_policy": bool(decision.blocked_by_policy),
        "execution_tier": str(decision.execution_tier),
        "escalation_reasons": list(decision.escalation_reasons),
        "suppressed_diagnostics_count": int(decision.suppressed_diagnostics_count),
        "latency_seconds": float(decision.latency_seconds or 0.0),
        "auto_approval_candidate": bool(policy.can_auto_approve(decision)),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 4)


def compare_profile_runs(
    baseline: ProfileRunResult,
    staged: ProfileRunResult,
    *,
    min_agreement_rate: float = DEFAULT_MIN_AGREEMENT_RATE,
    min_latency_improvement_pct: float = DEFAULT_MIN_LATENCY_IMPROVEMENT_PCT,
    max_blocked_rate_delta_pp: float = DEFAULT_MAX_BLOCKED_RATE_DELTA_PP,
) -> dict[str, Any]:
    baseline_by_id = {item["message_id"]: item for item in baseline.decisions}
    staged_by_id = {item["message_id"]: item for item in staged.decisions}
    all_ids = sorted(set(baseline_by_id) | set(staged_by_id))

    agreement_count = 0
    disagreements: list[dict[str, Any]] = []
    unsafe_auto_approval_ids: list[str] = []

    for message_id in all_ids:
        left = baseline_by_id.get(message_id)
        right = staged_by_id.get(message_id)
        agreed = bool(
            left
            and right
            and left["final_action"] == right["final_action"]
            and left["blocked_by_policy"] == right["blocked_by_policy"]
        )
        if agreed:
            agreement_count += 1
        else:
            disagreements.append(
                {
                    "message_id": message_id,
                    "baseline": left,
                    "staged_v1": right,
                }
            )

        if left and right and left["blocked_by_policy"] and right["auto_approval_candidate"]:
            unsafe_auto_approval_ids.append(message_id)

    baseline_blocked_rate = sum(
        1 for item in baseline.decisions if item["blocked_by_policy"]
    ) / max(len(baseline.decisions), 1)
    staged_blocked_rate = sum(1 for item in staged.decisions if item["blocked_by_policy"]) / max(
        len(staged.decisions), 1
    )
    blocked_rate_delta_pp = round((staged_blocked_rate - baseline_blocked_rate) * 100.0, 2)

    non_escalated_ids = [
        item["message_id"] for item in staged.decisions if item["execution_tier"] == "fast"
    ]
    baseline_fast_latencies = [
        float(baseline_by_id[item_id]["latency_seconds"])
        for item_id in non_escalated_ids
        if item_id in baseline_by_id
    ]
    staged_fast_latencies = [
        float(staged_by_id[item_id]["latency_seconds"])
        for item_id in non_escalated_ids
        if item_id in staged_by_id
    ]
    baseline_p50 = _median(baseline_fast_latencies)
    staged_p50 = _median(staged_fast_latencies)
    latency_improvement_pct = None
    if baseline_p50 and staged_p50 is not None and baseline_p50 > 0:
        latency_improvement_pct = round(((baseline_p50 - staged_p50) / baseline_p50) * 100.0, 2)

    agreement_rate = round(agreement_count / max(len(all_ids), 1), 4)
    acceptance = {
        "latency_improvement": bool(
            latency_improvement_pct is not None
            and latency_improvement_pct >= min_latency_improvement_pct
        ),
        "decision_agreement": agreement_rate >= min_agreement_rate,
        "unsafe_auto_approval": len(unsafe_auto_approval_ids) == 0,
        "blocked_rate_delta": blocked_rate_delta_pp <= max_blocked_rate_delta_pp,
    }

    return {
        "message_count": len(all_ids),
        "agreement_rate": agreement_rate,
        "agreement_count": agreement_count,
        "disagreements": disagreements,
        "unsafe_auto_approval_ids": unsafe_auto_approval_ids,
        "baseline_blocked_rate": round(baseline_blocked_rate, 4),
        "staged_blocked_rate": round(staged_blocked_rate, 4),
        "blocked_rate_delta_pp": blocked_rate_delta_pp,
        "non_escalated_message_ids": non_escalated_ids,
        "baseline_non_escalated_p50_latency_seconds": baseline_p50,
        "staged_non_escalated_p50_latency_seconds": staged_p50,
        "latency_improvement_pct": latency_improvement_pct,
        "thresholds": {
            "min_agreement_rate": min_agreement_rate,
            "min_latency_improvement_pct": min_latency_improvement_pct,
            "max_blocked_rate_delta_pp": max_blocked_rate_delta_pp,
        },
        "acceptance": acceptance,
        "passes_all_thresholds": all(acceptance.values()),
    }


async def run_triage_profile(
    fixture_path: str | Path,
    *,
    profile: str,
    diagnostics_root: str | Path | None = None,
    verbose: bool = False,
) -> ProfileRunResult:
    messages = load_fixture_messages(fixture_path)
    policy = AutoApprovalPolicy()
    diagnostics_dir = Path(diagnostics_root) / profile if diagnostics_root is not None else None
    diagnostics = TriageRunDiagnostics(
        profile=profile,
        batch_size=len(messages),
        auto_approve=False,
        dry_run=True,
        verbose=verbose,
        diagnostics_dir=diagnostics_dir,
    )
    runner = InboxTriageRunner(
        gmail_connector=FixtureGmailConnector(messages),
        wedge_service=BenchmarkWedgeService(policy=policy),
        diagnostics=diagnostics,
        profile=profile,
    )

    started = time.perf_counter()
    with diagnostics.activate(), diagnostics.capture_logging():
        decisions = await runner.run_triage(batch_size=len(messages), auto_approve=False)
    elapsed = round(time.perf_counter() - started, 4)
    meta = diagnostics.finalize(decisions)
    summarized = [_summarize_decision(decision, policy) for decision in decisions]

    return ProfileRunResult(
        profile=profile,
        fixture_path=str(Path(fixture_path)),
        message_count=len(messages),
        total_duration_seconds=elapsed,
        diagnostics_artifact_dir=str(meta["artifact_dir"]),
        meta=meta,
        decisions=summarized,
    )


async def run_fixture_benchmark(
    fixture_path: str | Path,
    *,
    diagnostics_root: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    baseline = await run_triage_profile(
        fixture_path,
        profile="baseline",
        diagnostics_root=diagnostics_root,
        verbose=verbose,
    )
    staged = await run_triage_profile(
        fixture_path,
        profile="staged_v1",
        diagnostics_root=diagnostics_root,
        verbose=verbose,
    )
    comparison = compare_profile_runs(baseline, staged)
    return {
        "fixture_path": str(Path(fixture_path)),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profiles": {
            baseline.profile: baseline.to_dict(),
            staged.profile: staged.to_dict(),
        },
        "comparison": comparison,
    }


def render_benchmark_report(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    baseline = report["profiles"]["baseline"]
    staged = report["profiles"]["staged_v1"]
    acceptance = comparison["acceptance"]
    lines = [
        "Triage Profile Benchmark",
        f"Fixture: {report['fixture_path']}",
        f"Messages: {comparison['message_count']}",
        "",
        "Profile runs:",
        (
            f"  baseline   duration={baseline['total_duration_seconds']:.2f}s "
            f"blocked={baseline['meta']['blocked_count']} "
            f"diag={baseline['meta']['suppressed_diagnostics_count']} "
            f"artifact={baseline['diagnostics_artifact_dir']}"
        ),
        (
            f"  staged_v1  duration={staged['total_duration_seconds']:.2f}s "
            f"fast={staged['meta']['fast_tier_count']} "
            f"escalated={staged['meta']['escalated_count']} "
            f"blocked={staged['meta']['blocked_count']} "
            f"diag={staged['meta']['suppressed_diagnostics_count']} "
            f"artifact={staged['diagnostics_artifact_dir']}"
        ),
        "",
        "Acceptance checks:",
        f"  agreement_rate={comparison['agreement_rate']:.2%} pass={acceptance['decision_agreement']}",
        (
            "  latency_improvement_pct="
            f"{comparison['latency_improvement_pct']} "
            f"pass={acceptance['latency_improvement']}"
        ),
        (
            f"  blocked_rate_delta_pp={comparison['blocked_rate_delta_pp']} "
            f"pass={acceptance['blocked_rate_delta']}"
        ),
        (
            f"  unsafe_auto_approval_ids={len(comparison['unsafe_auto_approval_ids'])} "
            f"pass={acceptance['unsafe_auto_approval']}"
        ),
        f"Overall: {'PASS' if comparison['passes_all_thresholds'] else 'FAIL'}",
    ]
    return "\n".join(lines)


__all__ = [
    "BenchmarkWedgeService",
    "DEFAULT_MAX_BLOCKED_RATE_DELTA_PP",
    "DEFAULT_MIN_AGREEMENT_RATE",
    "DEFAULT_MIN_LATENCY_IMPROVEMENT_PCT",
    "FixtureGmailConnector",
    "ProfileRunResult",
    "compare_profile_runs",
    "load_fixture_messages",
    "render_benchmark_report",
    "run_fixture_benchmark",
    "run_triage_profile",
]
