"""Triage instrumentation: skeptical audit and calibration.

Provides adversarial audit of triage decisions (Claude reviews them)
and calibration metrics (Brier scores by confidence bucket) to close
the active-learning loop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.inbox.trust_wedge import InboxTrustWedgeStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skeptical audit
# ---------------------------------------------------------------------------

SKEPTICAL_AUDIT_SYSTEM = """\
You are an adversarial auditor reviewing email triage decisions.
For each decision, assess whether the action (archive/star/label/ignore) was
correct given the email subject, sender, snippet, confidence, and rationale.

Respond ONLY with a JSON array. Each element must have exactly these keys:
  "receipt_id": the receipt ID string,
  "label": one of "good", "bad", or "skip",
  "rationale": a short explanation (1-2 sentences)

Be skeptical. Flag "bad" when:
- A personal or important email was archived
- A financial/legal/medical email was auto-processed
- Low confidence but auto-approved
- The rationale doesn't match the action
Flag "skip" when you genuinely can't tell.
Flag "good" for clear spam, newsletters, and promotions correctly archived.
"""


@dataclass(frozen=True)
class AuditVerdict:
    receipt_id: str
    action: str
    confidence: float
    subject: str
    label: str  # "good" | "bad" | "skip"
    rationale: str


def build_audit_prompt(items: list[dict[str, Any]]) -> str:
    """Format triage decisions into an audit prompt."""
    lines = [SKEPTICAL_AUDIT_SYSTEM, "", "Decisions to audit:", ""]
    for item in items:
        rid = item.get("receipt_id", "?")
        action = item.get("action", "?")
        conf = item.get("confidence", 0.0) or 0.0
        subject = item.get("subject", "")
        sender = item.get("sender", "")
        blocked = item.get("blocked", False)
        lines.append(
            f"- receipt_id={rid} action={action} confidence={conf:.0%} "
            f"blocked={blocked} sender={sender} subject={subject}"
        )
    return "\n".join(lines)


async def run_skeptical_audit(
    items: list[dict[str, Any]],
    *,
    store: InboxTrustWedgeStore | None = None,
    max_batch: int = 20,
) -> list[AuditVerdict]:
    """Run Claude as adversarial reviewer over recent triage decisions."""
    if not items:
        return []

    items = items[:max_batch]
    prompt = build_audit_prompt(items)

    # Build a lookup for display info
    item_map = {item.get("receipt_id", ""): item for item in items}

    # Try to call Anthropic agent
    response_text = ""
    try:
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent(
            name="audit-agent",
            model="claude-haiku-4-5-20251001",
            role="critic",
        )
        response = await agent.generate(prompt)
        response_text = str(response) if response else ""
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.debug("Anthropic agent unavailable for audit: %s", exc)
        # Fallback: try OpenRouter
        try:
            from aragora.agents.api_agents.openrouter import OpenRouterAPIAgent

            agent = OpenRouterAPIAgent(
                name="audit-agent",
                model="anthropic/claude-haiku-4-5-20251001",
                role="critic",
            )
            response = await agent.generate(prompt)
            response_text = str(response) if response else ""
        except (ImportError, RuntimeError, ValueError) as exc2:
            logger.warning("No agent available for audit: %s", exc2)
            return []

    # Parse JSON response
    verdicts: list[AuditVerdict] = []
    try:
        # Extract JSON array from response (may have markdown fencing)
        text = response_text.strip()
        if "```" in text:
            # Strip markdown code fences
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                text = text[start:end]
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            parsed = [parsed]

        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            rid = str(entry.get("receipt_id", "")).strip()
            label = str(entry.get("label", "skip")).strip().lower()
            if label not in ("good", "bad", "skip"):
                label = "skip"
            rationale = str(entry.get("rationale", "")).strip()

            source_item = item_map.get(rid, {})
            verdict = AuditVerdict(
                receipt_id=rid,
                action=source_item.get("action", "?"),
                confidence=source_item.get("confidence", 0.0) or 0.0,
                subject=source_item.get("subject", ""),
                label=label,
                rationale=rationale,
            )
            verdicts.append(verdict)

            # Record feedback in store
            if store and rid:
                try:
                    store.record_feedback(
                        rid,
                        label=label,
                        source="audit",
                        notes=rationale[:500],
                    )
                except (ValueError, Exception):
                    logger.debug("Failed to record audit feedback for %s", rid)

    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to parse audit response: %s", exc)

    return verdicts


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationBucket:
    bucket_key: str
    total: int
    good: int
    bad: int
    skip: int
    accuracy: float
    avg_confidence: float
    brier_score: float


def compute_triage_calibration(store: InboxTrustWedgeStore) -> dict[str, Any]:
    """Compute calibration metrics by joining feedback with receipt confidence."""

    # Query joined data
    with store._cursor() as cursor:
        rows = cursor.execute(
            """
            SELECT
                f.label,
                json_extract(r.decision_json, '$.confidence') as confidence
            FROM triage_feedback f
            JOIN inbox_trust_receipts r ON f.receipt_id = r.receipt_id
            WHERE f.label IN ('good', 'bad')
            """
        ).fetchall()

    if not rows:
        return {
            "total_labeled": 0,
            "overall_accuracy": 0.0,
            "overall_brier": 0.0,
            "ece": 0.0,
            "buckets": [],
            "recommended_threshold": 0.85,
            "current_threshold": 0.85,
        }

    # Bucket by confidence (0.1-width)
    buckets_data: dict[str, dict[str, Any]] = {}
    total_good = 0
    total_bad = 0
    total_brier = 0.0

    for row in rows:
        label = row["label"]
        conf = float(row["confidence"] or 0.0)
        is_correct = 1.0 if label == "good" else 0.0
        brier = (conf - is_correct) ** 2

        # Bucket key
        bucket_start = int(conf * 10) / 10.0
        bucket_end = min(bucket_start + 0.1, 1.0)
        key = f"{bucket_start:.1f}-{bucket_end:.1f}"

        if key not in buckets_data:
            buckets_data[key] = {
                "good": 0,
                "bad": 0,
                "skip": 0,
                "confidences": [],
                "brier_sum": 0.0,
            }
        bd = buckets_data[key]
        if label == "good":
            bd["good"] += 1
            total_good += 1
        else:
            bd["bad"] += 1
            total_bad += 1
        bd["confidences"].append(conf)
        bd["brier_sum"] += brier
        total_brier += brier

    total = total_good + total_bad
    overall_accuracy = total_good / total if total > 0 else 0.0
    overall_brier = total_brier / total if total > 0 else 0.0

    # Build bucket list
    buckets: list[dict[str, Any]] = []
    ece = 0.0
    for key in sorted(buckets_data.keys()):
        bd = buckets_data[key]
        bt = bd["good"] + bd["bad"]
        acc = bd["good"] / bt if bt > 0 else 0.0
        avg_conf = sum(bd["confidences"]) / len(bd["confidences"]) if bd["confidences"] else 0.0
        brier = bd["brier_sum"] / bt if bt > 0 else 0.0
        ece += (bt / total) * abs(acc - avg_conf) if total > 0 else 0.0
        buckets.append(
            {
                "bucket_key": key,
                "total": bt,
                "good": bd["good"],
                "bad": bd["bad"],
                "skip": bd["skip"],
                "accuracy": acc,
                "avg_confidence": avg_conf,
                "brier_score": brier,
            }
        )

    # Recommend threshold: lowest bucket where accuracy >= 0.9
    recommended = 0.85
    for b in buckets:
        if b["accuracy"] >= 0.9 and b["total"] >= 3:
            parts = b["bucket_key"].split("-")
            recommended = float(parts[0])
            break

    # Get current threshold
    current = 0.85
    try:
        from aragora.inbox.auto_approval import AutoApprovalPolicy

        current = AutoApprovalPolicy().confidence_threshold
    except (ImportError, AttributeError):
        pass

    return {
        "total_labeled": total,
        "overall_accuracy": overall_accuracy,
        "overall_brier": overall_brier,
        "ece": ece,
        "buckets": buckets,
        "recommended_threshold": recommended,
        "current_threshold": current,
    }


def suggest_threshold_adjustment(calibration: dict[str, Any]) -> str | None:
    """Return a suggestion if the auto-approval threshold should change."""
    recommended = calibration.get("recommended_threshold", 0.85)
    current = calibration.get("current_threshold", 0.85)
    total = calibration.get("total_labeled", 0)

    if total < 10:
        return None

    diff = recommended - current
    if abs(diff) < 0.05:
        return None

    if diff > 0:
        return (
            f"Consider raising auto-approval threshold from {current:.2f} to "
            f"{recommended:.2f} — accuracy in lower buckets is below 90%."
        )
    else:
        return (
            f"Consider lowering auto-approval threshold from {current:.2f} to "
            f"{recommended:.2f} — accuracy is strong at lower confidence levels."
        )
