#!/usr/bin/env python3
"""Compare a single-family baseline receipt against a heterogeneous panel receipt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aragora.heterogeneity.probe import ACCEPTANCE_GATES, SEEDED_CLASSES
from aragora.heterogeneity.receipt import (
    HETEROGENEITY_RECEIPT_SCHEMA_VERSION,
    canonical_json,
    source_artifact_status,
)

COMPARISON_RECEIPT_SCHEMA_VERSION = "heterogeneity_comparison_receipt.v1"
STRICT_CI_SEPARATION = "strict-ci-separation"
ADDITIVE_MARGIN = "additive-margin"
ADDITIVE_MARGIN_VALUE = 0.10
BASELINE_SATURATION_CI_UPPER = 0.999
MIN_INDEPENDENT_FLAG_TRIALS = 18
DIRECT_PROVIDER_OPERATORS = frozenset({"anthropic", "openai", "gemini", "xai", "mistral"})
PROVIDER_ALIASES = {
    "google": "gemini",
    "google-ai": "gemini",
    "google-vertex": "gemini",
    "x-ai": "xai",
}


class ComparisonError(ValueError):
    """Raised when a receipt cannot be parsed or validated."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparisonError(f"{path}: expected top-level JSON object")
    return payload


def load_probe_receipt(path: str | Path) -> dict[str, Any]:
    """Load and validate a heterogeneity probe receipt."""
    receipt_path = Path(path)
    receipt = _read_json(receipt_path)
    schema_version = receipt.get("schema_version")
    if schema_version != HETEROGENEITY_RECEIPT_SCHEMA_VERSION:
        raise ComparisonError(
            f"{receipt_path}: expected schema_version "
            f"{HETEROGENEITY_RECEIPT_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        raise ComparisonError(f"{receipt_path}: missing metrics object")
    n_per_class = metrics.get("n_per_class", receipt.get("n_per_class"))
    if not isinstance(n_per_class, dict):
        raise ComparisonError(f"{receipt_path}: missing metrics.n_per_class object")
    return receipt


def _receipt_id(receipt: Mapping[str, Any]) -> str | None:
    receipt_id = receipt.get("receipt_id")
    return receipt_id if isinstance(receipt_id, str) and receipt_id else None


def _metric(receipt: Mapping[str, Any], key: str, *, default: Any = None) -> Any:
    metrics = receipt.get("metrics")
    if not isinstance(metrics, Mapping):
        return default
    return metrics.get(key, default)


def _float_metric(receipt: Mapping[str, Any], key: str) -> float:
    value = _metric(receipt, key, default=0.0)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _int_metric(receipt: Mapping[str, Any], key: str) -> int:
    value = _metric(receipt, key, default=0)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return 0


def _ci(receipt: Mapping[str, Any], key: str) -> tuple[float, float]:
    value = _metric(receipt, key, default=None)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and len(value) == 2
        and all(isinstance(item, int | float) for item in value)
    ):
        return (float(value[0]), float(value[1]))
    raise ComparisonError(f"receipt missing numeric two-point CI metric {key!r}")


def _n_per_class(receipt: Mapping[str, Any]) -> dict[str, int]:
    n_per_class = _metric(receipt, "n_per_class", default=None)
    if n_per_class is None:
        n_per_class = receipt.get("n_per_class", {})
    if not isinstance(n_per_class, Mapping):
        return {}
    return {
        str(prompt_class): int(count)
        for prompt_class, count in n_per_class.items()
        if isinstance(count, int | float)
    }


def _seeded_class_counts(receipt: Mapping[str, Any]) -> dict[str, int]:
    n_per_class = _n_per_class(receipt)
    return {
        prompt_class: int(n_per_class.get(prompt_class, 0))
        for prompt_class in sorted(SEEDED_CLASSES)
    }


def _composition_match(
    baseline_receipt: Mapping[str, Any], panel_receipt: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    baseline_counts = _seeded_class_counts(baseline_receipt)
    panel_counts = _seeded_class_counts(panel_receipt)
    mismatches = {
        prompt_class: {
            "baseline": baseline_counts.get(prompt_class, 0),
            "panel": panel_counts.get(prompt_class, 0),
        }
        for prompt_class in sorted(SEEDED_CLASSES)
        if baseline_counts.get(prompt_class, 0) != panel_counts.get(prompt_class, 0)
    }
    return (
        not mismatches,
        {
            "baseline_seeded_class_counts": baseline_counts,
            "panel_seeded_class_counts": panel_counts,
            "mismatches": mismatches,
        },
    )


def _provider_operator(agent: str) -> str:
    value = agent.lower()
    if "openrouter" in value:
        return "openrouter"
    if "claude" in value or "anthropic" in value:
        return "anthropic"
    if "gemini" in value or "google" in value:
        return "gemini"
    if "grok" in value or "xai" in value:
        return "xai"
    if "mistral" in value or "codestral" in value:
        return "mistral"
    if "openai" in value or "gpt" in value or "codex" in value:
        return "openai"
    for separator in ("/", ":", "@"):
        if separator in value:
            value = value.split(separator, 1)[0]
    return value or "unknown"


def _prompt_class(prompt: Mapping[str, Any]) -> str:
    value = prompt.get("class", prompt.get("prompt_class", ""))
    return value if isinstance(value, str) else ""


def _string_field(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _normalize_provider(value: str) -> str:
    normalized = value.lower()
    base = normalized.split("/", 1)[0]
    return PROVIDER_ALIASES.get(base, base)


def _classification_transport_provider(classification: Mapping[str, Any]) -> str:
    transport_provider = _string_field(classification.get("transport_provider"))
    if transport_provider:
        return _normalize_provider(transport_provider)
    if classification.get("fallback_used") is True:
        return "openrouter"
    agent = _string_field(classification.get("agent")) or "unknown"
    return _provider_operator(agent)


def _classification_requested_provider(classification: Mapping[str, Any]) -> str:
    requested_provider = _string_field(classification.get("requested_provider"))
    if requested_provider:
        return _normalize_provider(requested_provider)
    requested_model = _string_field(classification.get("requested_model"))
    if requested_model and "/" in requested_model:
        return _normalize_provider(requested_model)
    agent = _string_field(classification.get("agent")) or "unknown"
    if agent.startswith("openrouter:") and "/" in agent:
        model = agent.split(":", 1)[1]
        return _normalize_provider(model)
    return _provider_operator(agent)


def _is_fallback_success(classification: Mapping[str, Any]) -> bool:
    return classification.get("fallback_used") is True


def _successful_panel_classifications(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    breakdown = receipt.get("per_prompt_breakdown", [])
    if not isinstance(breakdown, list):
        return []
    successful: list[dict[str, Any]] = []
    for prompt in breakdown:
        if not isinstance(prompt, Mapping):
            continue
        classifications = prompt.get("panelist_classifications", [])
        if not isinstance(classifications, list):
            continue
        for classification in classifications:
            if not isinstance(classification, dict):
                continue
            if classification.get("verdict") == "dispatch_failed":
                continue
            successful.append(classification)
    return successful


def _successful_seeded_trials(receipt: Mapping[str, Any]) -> int:
    breakdown = receipt.get("per_prompt_breakdown", [])
    if not isinstance(breakdown, list):
        return 0
    total = 0
    for prompt in breakdown:
        if not isinstance(prompt, Mapping) or _prompt_class(prompt) not in SEEDED_CLASSES:
            continue
        classifications = prompt.get("panelist_classifications", [])
        if not isinstance(classifications, list):
            continue
        total += sum(
            1
            for classification in classifications
            if isinstance(classification, Mapping)
            and classification.get("verdict") != "dispatch_failed"
        )
    return total


def provider_summary(panel_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize direct, OpenRouter-mediated, and model-family successes."""
    classifications = _successful_panel_classifications(panel_receipt)
    transport_distribution: dict[str, int] = {}
    model_family_distribution: dict[str, int] = {}
    fallback_successes = 0
    direct_successes = 0
    for classification in classifications:
        transport = _classification_transport_provider(classification)
        model_family = _classification_requested_provider(classification)
        transport_distribution[transport] = transport_distribution.get(transport, 0) + 1
        model_family_distribution[model_family] = model_family_distribution.get(model_family, 0) + 1
        if transport in DIRECT_PROVIDER_OPERATORS:
            direct_successes += 1
        if _is_fallback_success(classification):
            fallback_successes += 1

    total = len(classifications)
    openrouter_mediated = transport_distribution.get("openrouter", 0)
    return {
        "successful_trials": total,
        "direct_successes": direct_successes,
        "openrouter_mediated_successes": openrouter_mediated,
        "fallback_successes": fallback_successes,
        "fallback_share": fallback_successes / total if total else 0.0,
        "fallback_augmented": fallback_successes > 0,
        "transport_provider_distribution": dict(sorted(transport_distribution.items())),
        "model_family_distribution": dict(sorted(model_family_distribution.items())),
    }


def evaluate_provider_rule(panel_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the Round 31 provider heterogeneity rule from successful trials."""
    classifications = _successful_panel_classifications(panel_receipt)
    distribution: dict[str, int] = {}
    for classification in classifications:
        provider = _classification_transport_provider(classification)
        distribution[provider] = distribution.get(provider, 0) + 1

    total = sum(distribution.values())
    reasons: list[str] = []
    provider_set = set(distribution)
    direct_present = bool(provider_set & DIRECT_PROVIDER_OPERATORS)

    if len(provider_set) < 2:
        reasons.append("fewer_than_two_provider_operators")
    if not direct_present:
        reasons.append("no_direct_provider_operator")
    if provider_set == {"openrouter"}:
        reasons.append("openrouter_only_panel")
    if total <= 0:
        reasons.append("no_successful_panel_trials")
    else:
        dominant_provider, dominant_count = max(distribution.items(), key=lambda item: item[1])
        dominant_share = dominant_count / total
        if dominant_share > 0.50:
            reasons.append(f"provider_dominance:{dominant_provider}:{dominant_share:.3f}")

    return {
        "satisfied": not reasons,
        "reasons": reasons,
        "provider_distribution": dict(sorted(distribution.items())),
        "successful_trials": total,
    }


def _fp_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "clean_neutral": {
            "false_positives": _int_metric(receipt, "clean_neutral_false_positives"),
            "trials": _int_metric(receipt, "clean_neutral_false_positive_trials"),
            "rate": _float_metric(receipt, "false_positive_rate_on_clean_neutral"),
        },
        "null_negative": {
            "false_positives": _int_metric(receipt, "null_negative_false_positives"),
            "trials": _int_metric(receipt, "null_negative_false_positive_trials"),
            "rate": _float_metric(receipt, "false_positive_rate_on_null_negative"),
        },
    }


def _metrics_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    ci_low, ci_high = _ci(receipt, "independent_flag_rate_ci_95_wilson")
    return {
        "receipt_id": _receipt_id(receipt),
        "n_panelists": _metric(receipt, "n_panelists", default=receipt.get("n_panelists")),
        "n_per_class": _n_per_class(receipt),
        "independent_flag_successes": _int_metric(receipt, "independent_flag_successes"),
        "independent_flag_trials": _int_metric(receipt, "independent_flag_trials"),
        "independent_flag_rate": _float_metric(receipt, "independent_flag_rate"),
        "independent_flag_rate_ci_95_wilson": [ci_low, ci_high],
        "false_positive_rates": _fp_summary(receipt),
        "successful_seeded_trials": _successful_seeded_trials(receipt),
    }


def _receipt_provenance(receipt: Mapping[str, Any]) -> dict[str, Any]:
    status = source_artifact_status(receipt, base_dir=ROOT)
    artifacts = receipt.get("source_artifacts", [])
    return {
        "canonical": status["canonical"],
        "status": status["status"],
        "problems": list(status.get("problems", [])),
        "source_artifact_count": status.get("source_artifact_count", 0),
        "source_artifacts": artifacts if isinstance(artifacts, list) else [],
    }


def _fp_flags(
    baseline_receipt: Mapping[str, Any], panel_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_clean = _float_metric(baseline_receipt, "false_positive_rate_on_clean_neutral")
    panel_clean = _float_metric(panel_receipt, "false_positive_rate_on_clean_neutral")
    baseline_null = _float_metric(baseline_receipt, "false_positive_rate_on_null_negative")
    panel_null = _float_metric(panel_receipt, "false_positive_rate_on_null_negative")
    return {
        "baseline_clean_neutral_exceeds_gate": baseline_clean
        > ACCEPTANCE_GATES["false_positive_rate_on_clean_neutral_max"],
        "baseline_null_negative_exceeds_gate": baseline_null
        > ACCEPTANCE_GATES["false_positive_rate_on_null_negative_max"],
        "panel_clean_neutral_exceeds_baseline": panel_clean > baseline_clean,
        "panel_clean_neutral_exceeds_gate": panel_clean
        > ACCEPTANCE_GATES["false_positive_rate_on_clean_neutral_max"],
        "panel_null_negative_exceeds_baseline": panel_null > baseline_null,
        "panel_null_negative_exceeds_gate": panel_null
        > ACCEPTANCE_GATES["false_positive_rate_on_null_negative_max"],
        "gate_thresholds": {
            "false_positive_rate_on_clean_neutral_max": ACCEPTANCE_GATES[
                "false_positive_rate_on_clean_neutral_max"
            ],
            "false_positive_rate_on_null_negative_max": ACCEPTANCE_GATES[
                "false_positive_rate_on_null_negative_max"
            ],
        },
    }


def _verdict(
    baseline_receipt: Mapping[str, Any],
    panel_receipt: Mapping[str, Any],
    *,
    verdict_rule: str,
    composition_match: bool,
    provider_rule: Mapping[str, Any],
) -> tuple[str, str]:
    if not composition_match:
        return ("INSUFFICIENT", "composition_mismatch")

    baseline_trials = _int_metric(baseline_receipt, "independent_flag_trials")
    panel_trials = _int_metric(panel_receipt, "independent_flag_trials")
    baseline_successful_seeded = _successful_seeded_trials(baseline_receipt)
    panel_successful_seeded = _successful_seeded_trials(panel_receipt)
    if (
        baseline_trials < MIN_INDEPENDENT_FLAG_TRIALS
        or panel_trials < MIN_INDEPENDENT_FLAG_TRIALS
        or baseline_successful_seeded < MIN_INDEPENDENT_FLAG_TRIALS
        or panel_successful_seeded < MIN_INDEPENDENT_FLAG_TRIALS
    ):
        return (
            "INSUFFICIENT",
            "insufficient_n:"
            f"baseline_attempted={baseline_trials},panel_attempted={panel_trials},"
            f"baseline_successful={baseline_successful_seeded},"
            f"panel_successful={panel_successful_seeded},"
            f"minimum={MIN_INDEPENDENT_FLAG_TRIALS}",
        )

    if not provider_rule.get("satisfied", False):
        reasons = provider_rule.get("reasons", [])
        if isinstance(reasons, list):
            return ("INSUFFICIENT", "provider_rule_failed:" + ",".join(map(str, reasons)))
        return ("INSUFFICIENT", "provider_rule_failed")

    baseline_ci_low, baseline_ci_high = _ci(baseline_receipt, "independent_flag_rate_ci_95_wilson")
    panel_ci_low, panel_ci_high = _ci(panel_receipt, "independent_flag_rate_ci_95_wilson")

    if baseline_ci_high >= BASELINE_SATURATION_CI_UPPER:
        return ("INSUFFICIENT-WITH-DATA", "baseline_saturation")

    required_gap = ADDITIVE_MARGIN_VALUE if verdict_rule == ADDITIVE_MARGIN else 0.0
    if panel_ci_low > baseline_ci_high + required_gap:
        return ("GO", verdict_rule)

    if panel_ci_high <= baseline_ci_high:
        return ("NO-GO", "panel_ci_within_or_below_baseline_ci")

    return (
        "INSUFFICIENT",
        f"ci_overlap:baseline=[{baseline_ci_low:.6f},{baseline_ci_high:.6f}],"
        f"panel=[{panel_ci_low:.6f},{panel_ci_high:.6f}]",
    )


def compute_comparison_receipt_id(receipt: Mapping[str, Any]) -> str:
    body = copy.deepcopy(dict(receipt))
    body.pop("receipt_id", None)
    body.pop("produced_at", None)
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def build_comparison_receipt(
    baseline_receipt: Mapping[str, Any],
    panel_receipt: Mapping[str, Any],
    *,
    baseline_receipt_path: str,
    panel_receipt_path: str,
    verdict_rule: str = STRICT_CI_SEPARATION,
    produced_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic comparison receipt from two probe receipts."""
    if verdict_rule not in {STRICT_CI_SEPARATION, ADDITIVE_MARGIN}:
        raise ComparisonError(f"unsupported verdict rule: {verdict_rule}")

    composition_match, composition = _composition_match(baseline_receipt, panel_receipt)
    provider_rule = evaluate_provider_rule(panel_receipt)
    fallback_summary = provider_summary(panel_receipt)
    baseline_provenance = _receipt_provenance(baseline_receipt)
    panel_provenance = _receipt_provenance(panel_receipt)
    comparison_canonical = bool(baseline_provenance["canonical"] and panel_provenance["canonical"])
    non_canonical_inputs = [
        name
        for name, provenance in (
            ("baseline_receipt", baseline_provenance),
            ("panel_receipt", panel_provenance),
        )
        if provenance["canonical"] is not True
    ]
    verdict, verdict_reason = _verdict(
        baseline_receipt,
        panel_receipt,
        verdict_rule=verdict_rule,
        composition_match=composition_match,
        provider_rule=provider_rule,
    )
    baseline_ci_low, baseline_ci_high = _ci(baseline_receipt, "independent_flag_rate_ci_95_wilson")
    panel_ci_low, panel_ci_high = _ci(panel_receipt, "independent_flag_rate_ci_95_wilson")

    receipt: dict[str, Any] = {
        "schema_version": COMPARISON_RECEIPT_SCHEMA_VERSION,
        "produced_at": produced_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline_receipt_path": baseline_receipt_path,
        "baseline_receipt_id": _receipt_id(baseline_receipt),
        "panel_receipt_path": panel_receipt_path,
        "panel_receipt_id": _receipt_id(panel_receipt),
        "comparison_canonical": comparison_canonical,
        "non_canonical_inputs": non_canonical_inputs,
        "composition_match": composition_match,
        "fallback_augmented": fallback_summary["fallback_augmented"],
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "verdict_rule_applied": verdict_rule,
        "baseline_metrics": _metrics_summary(baseline_receipt),
        "panel_metrics": _metrics_summary(panel_receipt),
        "comparison": {
            "composition": composition,
            "input_provenance": {
                "baseline_receipt": baseline_provenance,
                "panel_receipt": panel_provenance,
            },
            "provider_rule": provider_rule,
            "provider_summary": fallback_summary,
            "independent_flag_rate_ci_gap": {
                "panel_ci_lower_minus_baseline_ci_upper": panel_ci_low - baseline_ci_high,
                "panel_ci_upper_minus_baseline_ci_upper": panel_ci_high - baseline_ci_high,
                "baseline_ci": [baseline_ci_low, baseline_ci_high],
                "panel_ci": [panel_ci_low, panel_ci_high],
            },
            "false_positive_flags": _fp_flags(baseline_receipt, panel_receipt),
        },
    }
    receipt["receipt_id"] = compute_comparison_receipt_id(receipt)
    return receipt


def _write_output_receipt(receipt: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    comparison = receipt.get("comparison")
    provider_rule: Mapping[str, Any] = {}
    if isinstance(comparison, Mapping):
        raw_provider_rule = comparison.get("provider_rule")
        if isinstance(raw_provider_rule, Mapping):
            provider_rule = raw_provider_rule
    return {
        "receipt_id": receipt.get("receipt_id"),
        "verdict": receipt.get("verdict"),
        "verdict_reason": receipt.get("verdict_reason"),
        "composition_match": receipt.get("composition_match"),
        "comparison_canonical": receipt.get("comparison_canonical"),
        "provider_rule_satisfied": provider_rule.get("satisfied"),
        "provider_distribution": provider_rule.get("provider_distribution", {}),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare heterogeneity baseline and panel receipts.",
    )
    parser.add_argument("--baseline-receipt", required=True, type=Path)
    parser.add_argument("--panel-receipt", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    parser.add_argument(
        "--verdict-rule",
        choices=(STRICT_CI_SEPARATION, ADDITIVE_MARGIN),
        default=STRICT_CI_SEPARATION,
    )
    parser.add_argument(
        "--require-canonical-inputs",
        action="store_true",
        help="Fail closed unless both input receipts are bound to source artifacts.",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON summary to stdout.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        baseline_receipt = load_probe_receipt(args.baseline_receipt)
        panel_receipt = load_probe_receipt(args.panel_receipt)
        comparison_receipt = build_comparison_receipt(
            baseline_receipt,
            panel_receipt,
            baseline_receipt_path=str(args.baseline_receipt),
            panel_receipt_path=str(args.panel_receipt),
            verdict_rule=args.verdict_rule,
        )
        if (
            args.require_canonical_inputs
            and comparison_receipt.get("comparison_canonical") is not True
        ):
            raise ComparisonError(
                "comparison inputs are not canonical: "
                + ", ".join(map(str, comparison_receipt.get("non_canonical_inputs", [])))
            )
        _write_output_receipt(comparison_receipt, args.output_receipt)
    except ComparisonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = _summary(comparison_receipt)
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(
            "verdict={verdict} reason={verdict_reason} "
            "composition_match={composition_match} provider_rule={provider_rule_satisfied}".format(
                **summary
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
