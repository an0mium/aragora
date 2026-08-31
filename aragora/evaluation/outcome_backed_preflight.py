"""Fail-closed readiness checks for outcome-backed development inference."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from aragora.config.secrets import get_secret_presence_report

from aragora.evaluation.outcome_backed_budget import (
    DAILY_BUDGET_CAP_USD,
    BudgetLedgerError,
    OutcomeBackedBudgetLedger,
)
from aragora.evaluation.outcome_backed_conditions import (
    ConditionRosterError,
    preflight_condition_roster,
)
from aragora.evaluation.outcome_backed_corpus import (
    BENCHMARK_ID,
    VisibleCorpusError,
    canonical_json_sha256,
    load_visible_cases,
    validate_corpus_directory,
)
from aragora.evaluation.outcome_backed_prompt import (
    OutcomeBackedPromptError,
    render_outcome_backed_prompt,
)


DEVELOPMENT_PREFLIGHT_SCHEMA = "outcome-backed-development-preflight/1.0"
EXPECTED_DEVELOPMENT_CASES = 16
EXPECTED_CONDITIONS = 4
EXPECTED_PROMPTS = EXPECTED_DEVELOPMENT_CASES * EXPECTED_CONDITIONS
_IMPLEMENTATION_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CREDENTIAL_ENV_VARS = {
    "claude": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


class OutcomeBackedPreflightError(ValueError):
    """Raised when preflight inputs cannot be represented safely."""


@dataclass(frozen=True)
class PreflightBlocker:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class DevelopmentPreflightReport:
    implementation_sha: str
    corpus_sha256: str | None
    packet_set_sha256: str | None
    roster_sha256: str | None
    case_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    prompt_set_sha256: str | None
    credential_readiness: tuple[Mapping[str, object], ...]
    budget: Mapping[str, object] | None
    blockers: tuple[PreflightBlocker, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": DEVELOPMENT_PREFLIGHT_SCHEMA,
            "benchmark_id": BENCHMARK_ID,
            "implementation_sha": self.implementation_sha,
            "corpus_sha256": self.corpus_sha256,
            "packet_set_sha256": self.packet_set_sha256,
            "roster_sha256": self.roster_sha256,
            "development_case_ids": list(self.case_ids),
            "case_count": len(self.case_ids),
            "condition_ids": list(self.condition_ids),
            "condition_count": len(self.condition_ids),
            "prompt_count": EXPECTED_PROMPTS if self.prompt_set_sha256 else 0,
            "prompt_set_sha256": self.prompt_set_sha256,
            "credential_readiness": [dict(item) for item in self.credential_readiness],
            "budget": dict(self.budget) if self.budget is not None else None,
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "ready": self.ready,
        }


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OutcomeBackedPreflightError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise OutcomeBackedPreflightError(f"non-finite JSON number: {value}")


def _load_json_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_key,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OutcomeBackedPreflightError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise OutcomeBackedPreflightError(f"{path} must contain one JSON object")
    return value


def _tree_sha256(root: Path) -> str:
    paths = sorted(root.glob("*.json"))
    if not paths:
        raise OutcomeBackedPreflightError(f"no corpus JSON files found in {root}")
    entries = [
        {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths
    ]
    return canonical_json_sha256(entries)


def _packet_content_bytes(source: Mapping[str, object], *, case_id: str) -> bytes:
    content = source.get("content")
    encoding = source.get("content_encoding")
    media_type = source.get("media_type")
    if not isinstance(content, str):
        raise OutcomeBackedPreflightError(f"case {case_id} packet source content must be text")
    if encoding == "utf-8" and media_type == "text/plain; charset=utf-8":
        return content.encode("utf-8")
    if encoding == "base64" and media_type == "application/pdf":
        try:
            decoded = base64.b64decode(content, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise OutcomeBackedPreflightError(
                f"case {case_id} packet source contains invalid base64"
            ) from exc
        if not decoded.startswith(b"%PDF-"):
            raise OutcomeBackedPreflightError(
                f"case {case_id} packet PDF source lacks a PDF signature"
            )
        return decoded
    raise OutcomeBackedPreflightError(
        f"case {case_id} packet source has an unsupported media type or encoding"
    )


def _validate_packet_against_case(
    packet: Mapping[str, object],
    case: Mapping[str, object],
) -> None:
    case_id = str(case.get("case_id", ""))
    expected_case = dict(case)
    expected_sources = expected_case.pop("sources", None)
    if packet.get("case") != expected_case:
        raise OutcomeBackedPreflightError(
            f"case {case_id} packet is not bound to the visible corpus case"
        )
    if not isinstance(expected_sources, list):
        raise OutcomeBackedPreflightError(f"case {case_id} has invalid source metadata")
    packet_sources = packet.get("sources")
    if not isinstance(packet_sources, list) or len(packet_sources) != len(expected_sources):
        raise OutcomeBackedPreflightError(f"case {case_id} packet source set mismatch")

    metadata_fields = ("source_id", "title", "url", "published_at", "content_sha256")
    expected_by_id: dict[str, Mapping[str, object]] = {}
    for source in expected_sources:
        if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str):
            raise OutcomeBackedPreflightError(f"case {case_id} has invalid source metadata")
        source_id = str(source["source_id"])
        if source_id in expected_by_id:
            raise OutcomeBackedPreflightError(f"case {case_id} has duplicate source IDs")
        expected_by_id[source_id] = source

    seen: set[str] = set()
    for packet_source in packet_sources:
        if not isinstance(packet_source, Mapping) or not isinstance(
            packet_source.get("source_id"), str
        ):
            raise OutcomeBackedPreflightError(f"case {case_id} packet has invalid source metadata")
        source_id = str(packet_source["source_id"])
        expected = expected_by_id.get(source_id)
        if expected is None or source_id in seen:
            raise OutcomeBackedPreflightError(f"case {case_id} packet source set mismatch")
        seen.add(source_id)
        if any(packet_source.get(field) != expected.get(field) for field in metadata_fields):
            raise OutcomeBackedPreflightError(
                f"case {case_id} packet source {source_id} metadata mismatch"
            )
        digest = hashlib.sha256(_packet_content_bytes(packet_source, case_id=case_id)).hexdigest()
        if digest != packet_source.get("content_sha256"):
            raise OutcomeBackedPreflightError(
                f"case {case_id} packet source {source_id} content hash mismatch"
            )


def _credential_readiness(
    roster_members: tuple[Mapping[str, object], ...],
    environ: Mapping[str, str] | None,
) -> tuple[tuple[Mapping[str, object], ...], tuple[PreflightBlocker, ...]]:
    readiness: list[Mapping[str, object]] = []
    blockers: list[PreflightBlocker] = []
    env_names = tuple(
        name for member in roster_members for name in _CREDENTIAL_ENV_VARS[str(member["family"])]
    )
    if environ is None:
        try:
            credential_sources = {
                status.name: status.source for status in get_secret_presence_report(env_names)
            }
        except (OSError, RuntimeError, ValueError) as exc:
            credential_sources = {}
            blockers.append(
                PreflightBlocker(
                    "credential_discovery_failed",
                    f"credential presence discovery failed ({type(exc).__name__})",
                )
            )
    else:
        credential_sources = {
            name: "provided_environment" if environ.get(name, "").strip() else "missing"
            for name in env_names
        }

    for member in roster_members:
        family = str(member["family"])
        accepted = _CREDENTIAL_ENV_VARS[family]
        available_name = next(
            (
                name
                for name in accepted
                if credential_sources.get(name) in {"aws", "env", "provided_environment"}
            ),
            None,
        )
        available = available_name is not None
        readiness.append(
            {
                "family": family,
                "agent_type": member["agent_type"],
                "requested_model": member["requested_model"],
                "expected_resolved_model": member["expected_resolved_model"],
                "transport": member["transport"],
                "allow_fallback": member["allow_fallback"],
                "accepted_environment_variables": list(accepted),
                "credential_available": available,
                "credential_source": (
                    credential_sources.get(available_name, "missing")
                    if available_name
                    else "missing"
                ),
            }
        )
        if not available:
            blockers.append(
                PreflightBlocker(
                    "missing_provider_credential",
                    f"{family} direct-api credential is unavailable",
                )
            )
    return tuple(readiness), tuple(blockers)


def preflight_development_run(
    corpus_dir: Path | str,
    packet_dir: Path | str,
    budget_ledger_path: Path | str,
    *,
    implementation_sha: str,
    environ: Mapping[str, str] | None = None,
    utc_date: date | None = None,
) -> DevelopmentPreflightReport:
    """Attest development-run readiness without model calls or ledger writes."""

    if not _IMPLEMENTATION_SHA_RE.fullmatch(implementation_sha):
        raise OutcomeBackedPreflightError("implementation_sha must be a lowercase 40-hex SHA")

    blockers: list[PreflightBlocker] = []
    corpus_sha256: str | None = None
    packet_set_sha256: str | None = None
    roster_sha256: str | None = None
    prompt_set_sha256: str | None = None
    case_ids: tuple[str, ...] = ()
    condition_ids: tuple[str, ...] = ()
    credential_readiness: tuple[Mapping[str, object], ...] = ()
    budget: Mapping[str, object] | None = None
    root = Path(corpus_dir)

    report = validate_corpus_directory(root)
    if not report.valid:
        blockers.append(
            PreflightBlocker(
                "invalid_corpus",
                f"frozen corpus has {len(report.issues)} integrity issue(s)",
            )
        )
    else:
        corpus_sha256 = _tree_sha256(root)

    cases: tuple[Mapping[str, Any], ...] = ()
    try:
        visible = load_visible_cases(root)
        cases = tuple(case for case in visible if case.get("split") == "development")
        case_ids = tuple(str(case.get("case_id", "")) for case in cases)
        if len(cases) != EXPECTED_DEVELOPMENT_CASES:
            blockers.append(
                PreflightBlocker(
                    "development_case_count",
                    f"expected {EXPECTED_DEVELOPMENT_CASES} development cases, found {len(cases)}",
                )
            )
    except VisibleCorpusError as exc:
        blockers.append(PreflightBlocker("visible_corpus_invalid", str(exc)))

    roster = None
    try:
        roster = preflight_condition_roster()
        roster_sha256 = roster.roster_sha256
        condition_ids = tuple(condition.condition_id for condition in roster.conditions)
        if len(condition_ids) != EXPECTED_CONDITIONS:
            blockers.append(
                PreflightBlocker(
                    "condition_count",
                    f"expected {EXPECTED_CONDITIONS} conditions, found {len(condition_ids)}",
                )
            )
    except ConditionRosterError as exc:
        blockers.append(PreflightBlocker("invalid_condition_roster", str(exc)))

    if roster is not None:
        unique_members: dict[str, Mapping[str, object]] = {}
        for condition in roster.conditions:
            for member in condition.members:
                unique_members.setdefault(member.family, member.to_dict())
        credential_readiness, credential_blockers = _credential_readiness(
            tuple(unique_members.values()), environ
        )
        blockers.extend(credential_blockers)

    packet_root = Path(packet_dir)
    prompt_entries: list[dict[str, str]] = []
    if len(cases) == EXPECTED_DEVELOPMENT_CASES and len(condition_ids) == EXPECTED_CONDITIONS:
        try:
            packet_set = _load_json_object(packet_root / "packet-set.json")
            expected_case_ids = set(case_ids)
            packet_entries = packet_set.get("packets")
            if not isinstance(packet_entries, list):
                raise OutcomeBackedPreflightError("packet-set packets must be a list")
            manifest_case_ids = {
                str(entry.get("case_id"))
                for entry in packet_entries
                if isinstance(entry, Mapping) and isinstance(entry.get("case_id"), str)
            }
            if manifest_case_ids != expected_case_ids:
                raise OutcomeBackedPreflightError(
                    "packet-set case IDs do not match the frozen development corpus"
                )
            cases_by_id = {str(case["case_id"]): case for case in cases}
            rendered_packet_set_sha: str | None = None
            for case_id in case_ids:
                packet = _load_json_object(packet_root / f"{case_id}.packet.json")
                _validate_packet_against_case(packet, cases_by_id[case_id])
                for condition_id in condition_ids:
                    rendered = render_outcome_backed_prompt(
                        packet,
                        packet_set=packet_set,
                        condition_id=condition_id,
                    )
                    if rendered_packet_set_sha is None:
                        rendered_packet_set_sha = rendered.packet_set_sha256
                    elif rendered.packet_set_sha256 != rendered_packet_set_sha:
                        raise OutcomeBackedPreflightError(
                            "rendered prompts disagree on packet-set identity"
                        )
                    prompt_entries.append(
                        {
                            "case_id": case_id,
                            "condition_id": condition_id,
                            "prompt_sha256": rendered.prompt_sha256,
                        }
                    )
            if len(prompt_entries) != EXPECTED_PROMPTS:
                raise OutcomeBackedPreflightError(
                    f"expected {EXPECTED_PROMPTS} prompts, rendered {len(prompt_entries)}"
                )
            if len({entry["prompt_sha256"] for entry in prompt_entries}) != EXPECTED_PROMPTS:
                raise OutcomeBackedPreflightError("development prompts are not all unique")
            packet_set_sha256 = rendered_packet_set_sha
            prompt_set_sha256 = canonical_json_sha256(prompt_entries)
        except (OSError, OutcomeBackedPreflightError, OutcomeBackedPromptError) as exc:
            blockers.append(PreflightBlocker("development_packets_not_ready", str(exc)))

    try:
        snapshot = OutcomeBackedBudgetLedger(budget_ledger_path).snapshot(utc_date=utc_date)
        budget = snapshot.to_dict()
        if snapshot.cap_usd != DAILY_BUDGET_CAP_USD:
            blockers.append(
                PreflightBlocker(
                    "budget_cap_mismatch",
                    f"expected daily cap ${DAILY_BUDGET_CAP_USD}, found ${snapshot.cap_usd}",
                )
            )
        if snapshot.exceeded:
            blockers.append(PreflightBlocker("budget_exceeded", "daily budget is exceeded"))
        if snapshot.open_reservations:
            blockers.append(
                PreflightBlocker(
                    "open_budget_reservations",
                    f"{snapshot.open_reservations} budget reservation(s) remain open",
                )
            )
        if snapshot.remaining_usd <= 0:
            blockers.append(PreflightBlocker("budget_exhausted", "no paid-call budget remains"))
    except (BudgetLedgerError, OSError, ValueError) as exc:
        blockers.append(PreflightBlocker("budget_ledger_invalid", str(exc)))

    unique_blockers = tuple(
        PreflightBlocker(code, message)
        for code, message in dict.fromkeys((item.code, item.message) for item in blockers)
    )
    return DevelopmentPreflightReport(
        implementation_sha=implementation_sha,
        corpus_sha256=corpus_sha256,
        packet_set_sha256=packet_set_sha256,
        roster_sha256=roster_sha256,
        case_ids=case_ids,
        condition_ids=condition_ids,
        prompt_set_sha256=prompt_set_sha256,
        credential_readiness=credential_readiness,
        budget=budget,
        blockers=unique_blockers,
    )


__all__ = [
    "DEVELOPMENT_PREFLIGHT_SCHEMA",
    "DevelopmentPreflightReport",
    "OutcomeBackedPreflightError",
    "PreflightBlocker",
    "preflight_development_run",
]
