"""
Quickstart CLI command: truthful first-run onboarding in one command.

Guides new users through a short debate:
1. Checks for supported API keys (loads .env if present)
2. Accepts a question via --question or interactive prompt
3. Runs a live debate when keys are available, otherwise falls back to demo
4. Can alternatively generate a first-pass execution specification
5. Saves one deterministic result artifact
6. Optionally opens an HTML view in the browser
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import ssl
import sys
import tempfile
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO, cast

logger = logging.getLogger(__name__)

_DEFAULT_QUESTION = "Should we adopt microservices or keep our monolith?"
_DEMO_DEFAULT_ROUNDS = 2
_LIVE_DEFAULT_ROUNDS = 1
_LIVE_PRECHECK_TIMEOUT_SECONDS = 3.0
_LIVE_DEBATE_TIMEOUT_SECONDS = 120.0
_LIVE_ROLES: tuple[tuple[str, str], ...] = (
    ("proposer", "proposer"),
    ("critic", "critic"),
    ("synthesizer", "synthesizer"),
)
_LIVE_PROVIDER_PRIORITY: tuple[str, ...] = (
    "gemini",
    "openai-api",
    "anthropic-api",
    "grok",
    "mistral",
    "deepseek",
)
_LIVE_ARENA_KWARGS: dict[str, Any] = {
    "disable_post_debate_pipeline": True,
}
_PROVIDER_TLS_HOSTS: dict[str, str] = {
    "anthropic-api": "api.anthropic.com",
    "openai-api": "api.openai.com",
    "gemini": "generativelanguage.googleapis.com",
    "mistral": "api.mistral.ai",
    "grok": "api.x.ai",
    "deepseek": "openrouter.ai",
}
_PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "agent_type": "anthropic-api",
        "model": "claude-haiku-4-5-20251001",
        "env_vars": ("ANTHROPIC_API_KEY",),
    },
    "openai": {
        "agent_type": "openai-api",
        "model": "gpt-4o-mini",
        "env_vars": ("OPENAI_API_KEY",),
    },
    "gemini": {
        "agent_type": "gemini",
        "model": "gemini-2.0-flash",
        "env_vars": ("GEMINI_API_KEY",),
    },
    "mistral": {
        "agent_type": "mistral",
        "model": None,
        "env_vars": ("MISTRAL_API_KEY",),
    },
    "grok": {
        "agent_type": "grok",
        "model": None,
        "env_vars": ("XAI_API_KEY", "GROK_API_KEY"),
    },
    "openrouter": {
        "agent_type": "deepseek",
        "model": None,
        "env_vars": ("OPENROUTER_API_KEY",),
    },
}
_PROVIDER_ALIASES = {
    "anthropic-api": "anthropic",
    "openai-api": "openai",
    "xai": "grok",
    "deepseek": "openrouter",
}
_TLS_VERIFICATION_ERROR_MARKERS: tuple[str, ...] = (
    "CERTIFICATE_VERIFY_FAILED",
    "certificate verify failed",
    "unable to get local issuer certificate",
)
_QUICKSTART_SETTLEMENT_REVIEW_DAYS = 30
_QUICKSTART_STRICT_FALSIFIER_THRESHOLD = 0.8
_QUICKSTART_SETTLEMENT_CONFIDENCE_CAP = 0.79
_QUICKSTART_SETTLEMENT_REVIEW_NOTE = (
    "Quickstart settlement metadata did not include explicit falsifiers; "
    "settlement confidence was capped below the strict-review threshold."
)


def _quickstart_loop_factory() -> asyncio.AbstractEventLoop:
    """Create a private loop for sync CLI entrypoints without inheriting test policy."""
    selector_loop = getattr(asyncio, "SelectorEventLoop", None)
    if selector_loop is not None:
        return selector_loop()
    return asyncio.new_event_loop()


def _run_sync(coro: Any) -> Any:
    """Run quickstart coroutines on an isolated event loop."""
    with asyncio.Runner(loop_factory=_quickstart_loop_factory) as runner:
        return runner.run(coro)


def add_quickstart_parser(subparsers: Any) -> None:
    """Register the 'quickstart' subcommand."""
    qs_parser = subparsers.add_parser(
        "quickstart",
        help="Guided zero-to-receipt first debate (new user onboarding)",
        description="""
Run your first adversarial debate with a bounded live onboarding path.

Automatically detects available API keys, picks agents, runs a fast
live debate, and opens the decision receipt in your browser.
No configuration needed.

Examples:
  aragora quickstart --demo                              # Zero-config demo
  aragora quickstart --question "Should we use Kubernetes?"
  aragora quickstart --topic "Migrate to TypeScript?" --rounds 1 --json
  aragora quickstart --provider openai --api-key sk-... --save-key
  aragora quickstart --question "Migrate to TypeScript?" --output receipt.json
  aragora quickstart --question "Migrate to TypeScript?" --spec-first
  aragora quickstart --demo --no-browser                 # CI/headless mode
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    qs_parser.add_argument(
        "--question",
        "--topic",
        "-q",
        help="The question to debate (uses a default if omitted with --demo)",
    )
    qs_parser.add_argument(
        "--output",
        "-o",
        help="Save receipt to file (supports .json, .md, .html)",
    )
    qs_parser.add_argument(
        "--demo",
        action="store_true",
        help="Use mock agents (no API keys required)",
    )
    qs_parser.add_argument(
        "--spec-first",
        action="store_true",
        help="Generate a prompt-to-spec artifact instead of starting with debate",
    )
    qs_parser.add_argument(
        "--provider",
        help=(
            "Live provider to use for quickstart (anthropic, openai, gemini, "
            "mistral, grok, openrouter). Required with --api-key."
        ),
    )
    qs_parser.add_argument(
        "--api-key",
        help="Provider API key to use for this run without pre-configuring env vars",
    )
    qs_parser.add_argument(
        "--save-key",
        action="store_true",
        help="Persist --api-key into the Aragora secure key store",
    )
    qs_parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=None,
        help="Number of debate rounds (default: 2 for demo, 1 for live quickstart)",
    )
    qs_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "md", "html"],
        default="json",
        help="Receipt output format (default: json)",
    )
    qs_parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured debate result JSON to stdout",
    )
    qs_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open receipt in browser (for CI/headless environments)",
    )
    qs_parser.set_defaults(func=cmd_quickstart)


def _normalize_provider(provider: str | None) -> str | None:
    """Normalize provider names from CLI input into quickstart keys."""
    if not provider:
        return None
    normalized = provider.strip().lower()
    if not normalized:
        return None
    normalized = _PROVIDER_ALIASES.get(normalized, normalized)
    return normalized if normalized in _PROVIDER_SPECS else None


def _is_tls_verification_failure(detail: object) -> bool:
    """Best-effort detection for wrapped certificate verification failures."""
    if isinstance(detail, ssl.SSLCertVerificationError):
        return True

    detail_text = str(detail).strip()
    if not detail_text:
        return False

    return any(marker in detail_text for marker in _TLS_VERIFICATION_ERROR_MARKERS)


def _load_dotenv() -> bool:
    """Try to load .env file from cwd or parent. Returns True if loaded."""
    for candidate in [Path.cwd() / ".env", Path.cwd().parent / ".env"]:
        if candidate.is_file():
            try:
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        if key and key not in os.environ:
                            os.environ[key] = value
                return True
            except OSError:
                pass
    return False


def _detect_agents(preferred_provider: str | None = None) -> list[tuple[str, str | None]]:
    """Detect available agents based on API keys.

    Returns list of (provider, model) tuples.
    """
    agents: list[tuple[str, str | None]] = []

    requested = _normalize_provider(preferred_provider)
    if preferred_provider and requested is None:
        raise ValueError(
            "Unsupported provider. Choose from: anthropic, openai, gemini, "
            "mistral, grok, openrouter."
        )

    for provider_name, spec in _PROVIDER_SPECS.items():
        if requested and provider_name != requested:
            continue
        if any(os.environ.get(env_var) for env_var in spec["env_vars"]):
            agents.append((str(spec["agent_type"]), cast(str | None, spec["model"])))

    return agents


def _configure_inline_api_key(
    provider: str | None,
    api_key: str | None,
    *,
    save_key: bool = False,
) -> tuple[str | None, dict[str, str] | None]:
    """Inject an inline API key into the current process and optionally persist it."""
    normalized_provider = _normalize_provider(provider)
    if not api_key:
        return normalized_provider, None
    if normalized_provider is None:
        raise ValueError(
            "--api-key requires --provider (anthropic, openai, gemini, mistral, grok, openrouter)"
        )

    spec = _PROVIDER_SPECS[normalized_provider]
    primary_env_var = str(spec["env_vars"][0])
    os.environ[primary_env_var] = api_key

    if not save_key:
        return normalized_provider, None

    from aragora.cli.api_keys import set_provider_key

    stored = set_provider_key(normalized_provider, api_key)
    return normalized_provider, {
        "provider": stored.provider,
        "env_var": stored.env_var,
        "backend": stored.backend,
        "masked_value": stored.masked_value,
    }


def _get_question(args: argparse.Namespace, *, stream: TextIO | None = None) -> str | None:
    """Get the debate question from args, default, or interactive prompt."""
    if args.question:
        return args.question

    # In demo mode, use the default question instead of prompting
    if getattr(args, "demo", False):
        return _DEFAULT_QUESTION

    # Interactive prompt
    try:
        console = stream or sys.stdout
        print("\nWhat question should the agents debate?", file=console)
        print("(Example: 'Should we migrate from REST to GraphQL?')\n", file=console)
        print("> ", end="", file=console, flush=True)
        question = input().strip()
        return question if question else None
    except (EOFError, KeyboardInterrupt):
        return None


def _resolve_rounds(requested_rounds: int | None, *, use_demo: bool) -> int:
    """Resolve quickstart rounds with mode-specific defaults."""
    if requested_rounds is not None:
        return requested_rounds
    return _DEMO_DEFAULT_ROUNDS if use_demo else _LIVE_DEFAULT_ROUNDS


def _default_receipt_path(mode: str, fmt: str) -> Path:
    """Return the default saved artifact path for quickstart results."""
    receipts_dir = Path.cwd() / ".aragora" / "receipts"
    suffix = {
        "json": ".json",
        "md": ".md",
        "html": ".html",
    }.get(fmt, ".json")
    normalized_mode = (mode or "demo").strip().lower()
    return receipts_dir / f"quickstart-{normalized_mode}-receipt{suffix}"


def _default_spec_path() -> Path:
    """Return the default saved artifact path for quickstart spec-first output."""
    return Path.cwd() / ".aragora" / "specs" / "quickstart-spec.json"


def _clamp_confidence(raw_confidence: Any) -> float:
    """Normalize confidence values into the expected [0.0, 1.0] range."""
    try:
        confidence = float(raw_confidence or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _derive_receipt_id(
    *,
    mode: str,
    question: str,
    rounds: int,
    existing: str | None = None,
) -> str:
    """Provide a stable receipt id when the debate engine does not supply one."""
    if existing:
        return existing
    basis = f"{mode}:{question}:{rounds}"
    return f"quickstart-{mode}-{hashlib.sha256(basis.encode('utf-8')).hexdigest()[:12]}"


def _normalize_json_result(result: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    """Return a stdout-safe JSON payload with stable top-level quickstart fields."""
    payload = _build_quickstart_receipt_payload(result)
    payload["artifact_path"] = str(artifact_path)
    return payload


def _normalize_spec_json_result(result: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    """Return a stdout-safe JSON payload for spec-first quickstart runs."""
    payload = dict(result)
    payload["artifact_path"] = str(artifact_path)
    return payload


async def _run_quickstart_spec_first(question: str) -> dict[str, Any]:
    """Generate a first-pass specification for quickstart onboarding.

    Prefer the orchestrator-backed path for canonical backbone tracking, but
    fall back to the lighter prompt-engine conductor so onboarding still
    produces a usable artifact when the wider stack is unavailable.
    """
    from aragora.cli.commands.spec import _run_spec_pipeline

    try:
        result = await _run_spec_pipeline(
            question,
            depth="quick",
            profile="founder",
            output_format="json",
            use_orchestrator=True,
        )
        result["pipeline"] = "orchestrator"
        return result
    except Exception:
        logger.debug("quickstart_spec_first_orchestrator_failed", exc_info=True)

    result = await _run_spec_pipeline(
        question,
        depth="quick",
        profile="founder",
        output_format="json",
        use_orchestrator=False,
    )
    result["pipeline"] = "prompt_engine"
    return result


def _build_quickstart_spec_payload(question: str, result: dict[str, Any]) -> dict[str, Any]:
    """Normalize quickstart spec-first output into a saved JSON artifact."""
    payload = dict(result)
    payload["question"] = question
    payload["mode"] = "quickstart-spec"
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    return payload


def _save_quickstart_spec_payload(payload: dict[str, Any], output_path: str | None = None) -> Path:
    """Persist the quickstart spec-first payload to JSON."""
    path = Path(output_path) if output_path else _default_spec_path()
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _coerce_string_list(values: Any) -> list[str]:
    """Return a clean string list from optional mixed values."""
    if not isinstance(values, (list, tuple, set)):
        return []
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_dissent_records(
    payload: dict[str, Any], participants: list[str]
) -> tuple[list[dict[str, str]], list[str]]:
    """Normalize dissent entries into receipt-friendly records and reasons."""
    dissent = payload.get("dissent")
    if isinstance(dissent, list) and dissent:
        records: list[dict[str, str]] = []
        reasons: list[str] = []
        for index, entry in enumerate(dissent):
            if isinstance(entry, dict):
                agent = (
                    str(
                        entry.get("agent")
                        or (participants[index % len(participants)] if participants else "agent")
                    ).strip()
                    or "agent"
                )
                reason = str(entry.get("reason") or entry.get("description") or "").strip()
            else:
                agent = participants[index % len(participants)] if participants else "agent"
                reason = str(entry).strip()
            if reason:
                reasons.append(reason)
            records.append({"agent": agent, "reason": reason or "N/A"})
        return records, reasons

    reasons = _coerce_string_list(payload.get("dissenting_views"))
    return _summarize_dissenting_views(reasons, participants), reasons


def _settlement_value_is_blank(value: Any) -> bool:
    """Identify settlement fields that still need a canonical default."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _normalize_quickstart_settlement_confidence(settlement_metadata: dict[str, Any]) -> None:
    """Fail closed for sparse quickstart receipts instead of inventing falsifiers."""
    falsifiers = settlement_metadata.get("falsifiers")
    if isinstance(falsifiers, list) and falsifiers:
        return

    try:
        confidence = float(settlement_metadata.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence >= _QUICKSTART_STRICT_FALSIFIER_THRESHOLD:
        settlement_metadata["confidence"] = min(
            confidence,
            _QUICKSTART_SETTLEMENT_CONFIDENCE_CAP,
        )

    review_notes = settlement_metadata.get("review_notes")
    if not isinstance(review_notes, list):
        review_notes = []
        settlement_metadata["review_notes"] = review_notes
    if _QUICKSTART_SETTLEMENT_REVIEW_NOTE not in review_notes:
        review_notes.append(_QUICKSTART_SETTLEMENT_REVIEW_NOTE)


def _build_quickstart_settlement_metadata(
    *,
    settlement_metadata: Any,
    debate_result: Any,
    receipt_context: Any | None,
    timestamp: str,
) -> dict[str, Any]:
    """Capture settlement metadata via the canonical settlement tracker."""
    from aragora.debate.settlement import EpistemicSettlementTracker

    tracker = EpistemicSettlementTracker()
    captured = tracker.capture_settlement(
        debate_result,
        receipt_context,
        review_horizon_days=_QUICKSTART_SETTLEMENT_REVIEW_DAYS,
        settled_at=timestamp,
    ).to_dict()

    existing = dict(settlement_metadata) if isinstance(settlement_metadata, dict) else {}
    normalized = dict(existing)
    for key, value in captured.items():
        if _settlement_value_is_blank(normalized.get(key)):
            normalized[key] = value
    if _settlement_value_is_blank(normalized.get("falsifiers")):
        normalized["falsifiers"] = []
        _normalize_quickstart_settlement_confidence(normalized)
    return normalized


def _build_quickstart_receipt_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize quickstart debate results into a receipt-compatible artifact payload."""
    from aragora.gauntlet.receipt_models import ConsensusProof, DecisionReceipt, ProvenanceRecord

    payload = dict(result)
    receipt_info = payload.get("receipt", {})
    if not isinstance(receipt_info, dict):
        receipt_info = {}

    rounds = int(payload.get("rounds", 0) or 0)
    question = str(payload.get("question") or payload.get("input_summary") or "")
    mode = str(payload.get("mode", "demo") or "demo").strip().lower() or "demo"
    participants = _coerce_string_list(payload.get("agents") or receipt_info.get("participants"))
    summary = str(payload.get("summary") or payload.get("verdict_reasoning") or "")
    confidence = _clamp_confidence(payload.get("confidence", 0.0))
    receipt_id = _derive_receipt_id(
        mode=mode,
        question=question,
        rounds=rounds,
        existing=str(payload.get("receipt_id") or receipt_info.get("id") or ""),
    )
    timestamp = str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat())
    input_hash = str(
        payload.get("input_hash") or hashlib.sha256(question.encode("utf-8")).hexdigest()
    )

    dissent_records, dissenting_views = _normalize_dissent_records(payload, participants)
    consensus = payload.get("consensus_reached")
    if consensus is None and isinstance(payload.get("consensus_proof"), dict):
        consensus = bool(payload["consensus_proof"].get("reached"))
    if consensus is None:
        consensus = str(payload.get("verdict", "")).strip().lower() in {
            "consensus",
            "pass",
            "approve",
            "approved",
        }

    votes = payload.get("votes")
    if not isinstance(votes, list):
        votes = payload.get("agent_votes")
    if not isinstance(votes, list):
        votes = []

    has_receipt_contract = bool(
        str(payload.get("receipt_id") or "").strip()
        and str(payload.get("timestamp") or "").strip()
        and str(payload.get("artifact_hash") or "").strip()
        and str(payload.get("input_hash") or "").strip()
        and isinstance(payload.get("risk_summary"), dict)
    )

    existing_consensus = payload.get("consensus_proof")
    supporting_agents = (
        _coerce_string_list(existing_consensus.get("supporting_agents"))
        if isinstance(existing_consensus, dict)
        else []
    )
    dissenting_agents = (
        _coerce_string_list(existing_consensus.get("dissenting_agents"))
        if isinstance(existing_consensus, dict)
        else []
    )
    if not supporting_agents and consensus:
        supporting_agents = participants[:]
    if not dissenting_agents and dissent_records:
        dissenting_agents = _coerce_string_list([record["agent"] for record in dissent_records])

    risk_summary = payload.get("risk_summary")
    if not isinstance(risk_summary, dict):
        risk_summary = {}
    if not risk_summary:
        risk_summary = {
            "critical": 0 if consensus else int(bool(dissenting_views)),
            "high": len(dissenting_agents) if not consensus else 0,
            "medium": len(dissenting_views) if not consensus else 0,
            "low": 0,
        }
    risk_summary = dict(risk_summary)
    risk_summary.setdefault(
        "total",
        sum(
            int(risk_summary.get(bucket, 0) or 0)
            for bucket in ("critical", "high", "medium", "low")
        ),
    )

    settlement_result = SimpleNamespace(
        debate_id=str(payload.get("debate_id") or receipt_id),
        confidence=confidence,
        consensus_reached=bool(consensus),
        winner=str(payload.get("winner") or "") or None,
        participants=participants,
        dissenting_views=dissenting_views,
        final_answer=str(payload.get("verdict_reasoning") or summary),
        unresolved_tensions=list(payload.get("unresolved_tensions", []) or []),
        convergence_similarity=float(payload.get("convergence_similarity", 1.0) or 1.0),
        claims_kernel=payload.get("claims_kernel"),
        epistemic_hygiene=payload.get("epistemic_hygiene"),
        verification_criteria=_coerce_string_list(payload.get("verification_criteria")),
    )
    settlement_receipt_context = SimpleNamespace(
        consensus_proof=SimpleNamespace(dissenting_agents=dissenting_agents)
    )
    settlement_metadata = _build_quickstart_settlement_metadata(
        settlement_metadata=payload.get("settlement_metadata"),
        debate_result=settlement_result,
        receipt_context=settlement_receipt_context,
        timestamp=timestamp,
    )

    if has_receipt_contract:
        canonical = dict(payload)
    else:
        receipt = DecisionReceipt(
            receipt_id=receipt_id,
            gauntlet_id=str(payload.get("gauntlet_id") or receipt_id),
            timestamp=timestamp,
            input_summary=str(payload.get("input_summary") or question),
            input_hash=input_hash,
            risk_summary=risk_summary,
            attacks_attempted=int(
                payload.get("attacks_attempted", 0) or rounds * max(1, len(participants))
            ),
            attacks_successful=int(
                payload.get("attacks_successful", 0)
                or (0 if consensus else max(1, len(dissenting_views)) if dissenting_views else 0)
            ),
            probes_run=int(payload.get("probes_run", 0) or len(votes)),
            vulnerabilities_found=int(
                payload.get("vulnerabilities_found", 0) or len(dissenting_views)
            ),
            verdict=str(payload.get("verdict", "")),
            confidence=confidence,
            robustness_score=_clamp_confidence(payload.get("robustness_score", confidence)),
            verdict_reasoning=str(payload.get("verdict_reasoning") or summary),
            dissenting_views=dissenting_views,
            consensus_proof=ConsensusProof(
                reached=bool(consensus),
                confidence=confidence,
                supporting_agents=supporting_agents,
                dissenting_agents=dissenting_agents,
                method=(
                    str(existing_consensus.get("method") or "majority")
                    if isinstance(existing_consensus, dict)
                    else "majority"
                ),
                evidence_hash=input_hash,
                tainted_proposals=(
                    _coerce_string_list(existing_consensus.get("tainted_proposals"))
                    if isinstance(existing_consensus, dict)
                    else []
                ),
                trust_score=(
                    float(existing_consensus.get("trust_score", 1.0))
                    if isinstance(existing_consensus, dict)
                    else 1.0
                ),
            ),
            provenance_chain=[
                ProvenanceRecord(
                    timestamp=timestamp,
                    event_type="task",
                    description=question,
                ),
                ProvenanceRecord(
                    timestamp=timestamp,
                    event_type="verdict",
                    description=str(
                        payload.get("verdict_reasoning") or summary or payload.get("verdict", "")
                    ),
                ),
            ],
            cost_summary=payload.get("cost_summary"),
            thinking_traces=payload.get("thinking_traces"),
            settlement_metadata=settlement_metadata,
            config_used=(
                payload.get("config_used", {})
                if isinstance(payload.get("config_used"), dict)
                else {}
            ),
            artifact_hash=str(payload.get("artifact_hash") or ""),
        )
        canonical = dict(payload)
        canonical.update(receipt.to_dict())

    canonical["question"] = question
    canonical["rounds"] = rounds
    canonical["agents"] = participants
    canonical["summary"] = summary
    canonical["dissent"] = dissent_records
    canonical["mode"] = mode
    canonical["votes"] = votes
    canonical["agent_votes"] = votes
    canonical["consensus"] = bool(consensus)
    canonical["consensus_reached"] = bool(consensus)
    canonical["settlement_metadata"] = settlement_metadata
    canonical["receipt"] = {
        "id": str(canonical.get("receipt_id") or receipt_id),
        "artifact_hash": str(canonical.get("artifact_hash") or ""),
        "consensus_reached": bool(consensus),
        "confidence": confidence,
        "participants": participants,
    }
    return canonical


def _save_receipt(receipt_data: dict[str, Any], path: str | Path, fmt: str) -> Path:
    """Save receipt to file in the specified format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    fallback_json = json.dumps(receipt_data, indent=2, default=str)

    if fmt == "json" or suffix == ".json":
        output_path.write_text(fallback_json)
    elif fmt == "md" or suffix == ".md":
        try:
            from aragora.cli.receipt_formatter import receipt_to_markdown

            output_path.write_text(receipt_to_markdown(receipt_data))
        except ImportError as e:
            logger.debug("Receipt markdown formatter unavailable, writing JSON fallback: %s", e)
            output_path.write_text(fallback_json)
    elif fmt == "html" or suffix == ".html":
        try:
            from aragora.cli.receipt_formatter import receipt_to_html

            output_path.write_text(receipt_to_html(receipt_data))
        except ImportError as e:
            logger.debug("Receipt HTML formatter unavailable, writing JSON fallback: %s", e)
            output_path.write_text(fallback_json)
    else:
        output_path.write_text(fallback_json)

    return output_path.resolve()


def _open_receipt_in_browser(
    receipt_data: dict[str, Any], html_path: str | Path | None = None
) -> str | None:
    """Generate HTML receipt and open in browser.

    Returns the path to the saved HTML file, or None on failure.
    """
    try:
        if html_path is not None:
            resolved_path = str(Path(html_path).resolve())
            webbrowser.open(f"file://{resolved_path}")
            return resolved_path

        from aragora.cli.receipt_formatter import receipt_to_html

        html = receipt_to_html(receipt_data)
        # Create a persistent temp file (not auto-deleted)
        fd, path = tempfile.mkstemp(suffix=".html", prefix="aragora-receipt-")
        with os.fdopen(fd, "w") as f:
            f.write(html)
        webbrowser.open(f"file://{path}")
        return path
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.debug("Failed to open receipt in browser: %s", e)
        return None


async def _run_demo_debate(question: str, rounds: int) -> dict[str, Any]:
    """Run a debate with mock agents (no API keys needed)."""
    agent_names = ["analyst", "critic", "synthesizer"]
    receipt_id = _derive_receipt_id(mode="demo", question=question, rounds=rounds)
    summary = (
        f"Demo synthesis for: {question}\n\n"
        "- Combine a minimal baseline with explicit risk guardrails.\n"
        "- Make metrics and rollback criteria first-class requirements.\n"
        "- Ship in phases and revisit assumptions after initial data.\n\n"
        "Decision: Proceed with a phased rollout and explicit success metrics."
    )

    return {
        "question": question,
        "receipt_id": receipt_id,
        "verdict": "consensus",
        "confidence": 0.85,
        "rounds": rounds,
        "agents": agent_names,
        "summary": summary,
        "dissent": [],
        "votes": [],
        "consensus": True,
        "consensus_reached": True,
        "verification_criteria": [
            "Guardrail metrics remain within agreed thresholds during the initial rollout phase.",
            "Rollback indicators stay clear after the first production release.",
        ],
        "receipt": {
            "id": receipt_id,
            "confidence": 0.85,
            "participants": agent_names,
            "consensus_reached": True,
        },
        "mode": "demo",
    }


async def _run_live_debate(
    question: str,
    agents_list: list[tuple[str, str | None]],
    rounds: int,
    *,
    provider: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run a debate with live API agents."""
    # Quickstart is a bounded onboarding lane that relies on already-detected
    # env vars or inline keys. Avoid boot-time AWS Secrets Manager hydration
    # when importing the full debate stack for this path.
    os.environ.setdefault("ARAGORA_USE_SECRETS_MANAGER", "false")
    try:
        from aragora.config.secrets import reset_secret_manager

        reset_secret_manager()
    except ImportError:
        pass

    from aragora.agents.base import AgentType, create_agent
    from aragora.core import Environment
    from aragora.debate.orchestrator import Arena, DebateProtocol
    from aragora.insights.store import InsightStore

    agents_list = await _filter_reachable_live_agents(agents_list)

    team = _build_live_team(agents_list, provider=provider, api_key=api_key)
    if not team:
        raise RuntimeError("No live debate team could be assembled for quickstart")

    env = Environment(task=question)
    protocol = DebateProtocol(
        rounds=rounds,
        consensus="majority",
        convergence_detection=False,
        vote_grouping=False,
        enable_trickster=False,
        enable_research=False,
        enable_trending_injection=False,
        enable_llm_question_classification=False,
        enable_llm_synthesis=False,
    )
    store = InsightStore()

    agents = []
    agent_names = []
    for member in team:
        provider_name = str(member["provider"])
        agent = create_agent(
            cast(AgentType, provider_name),
            name=str(member["name"]),
            role=str(member["role"]),
            model=cast(str | None, member.get("model")),
            api_key=cast(str | None, member.get("api_key")),
        )
        agents.append(agent)
        agent_names.append(agent.name)

    # Quickstart is a bounded onboarding lane, not the full debate control plane.
    # Use config objects to avoid deprecation warnings from individual kwargs.
    from aragora.debate.arena_primary_configs import KnowledgeConfig, MemoryConfig, MLConfig

    memory_config = MemoryConfig(
        enable_knowledge_retrieval=True,
        enable_knowledge_ingestion=False,
        auto_create_knowledge_mound=True,
        enable_belief_guidance=False,
        enable_cross_debate_memory=False,
        use_rlm_limiter=False,
    )
    knowledge_config = KnowledgeConfig(
        auto_create_knowledge_mound=True,
        enable_knowledge_retrieval=True,
        enable_knowledge_ingestion=False,
        enable_belief_guidance=False,
    )
    ml_config = MLConfig(
        enable_ml_delegation=False,
        enable_quality_gates=False,
        enable_consensus_estimation=False,
    )
    arena = Arena(
        env,
        agents,
        protocol,
        insight_store=store,
        memory_config=memory_config,
        knowledge_config=knowledge_config,
        ml_config=ml_config,
        **_LIVE_ARENA_KWARGS,
    )
    arena.enable_introspection = False
    run_task = asyncio.create_task(arena.run())
    try:
        result = await asyncio.wait_for(run_task, timeout=_LIVE_DEBATE_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, RuntimeError):
            await run_task
        raise RuntimeError(
            f"Live debate timed out after {_LIVE_DEBATE_TIMEOUT_SECONDS:.0f}s"
        ) from exc
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise RuntimeError(f"Live debate failed before producing a result: {detail}") from exc

    if result is None:
        raise RuntimeError("Live debate returned no result")

    live_receipt = _build_live_receipt(result, question, rounds, team)
    live_receipt["agents"] = agent_names
    live_receipt["km_ingested"] = bool(
        getattr(arena, "knowledge_mound", None)
        and getattr(arena, "enable_knowledge_ingestion", False)
    )
    return live_receipt


def _build_live_team(
    agents_list: list[tuple[str, str | None]],
    *,
    provider: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Build a quickstart debate team, guaranteeing a real multi-role debate."""
    if not agents_list:
        return []

    normalized_provider = _normalize_provider(provider)
    provider_configs: list[dict[str, Any]] = []
    if normalized_provider:
        provider_configs.append(
            {
                "provider": agents_list[0][0],
                "model": agents_list[0][1],
                "api_key": api_key,
            }
        )
    else:
        selected_provider = next(
            (
                (agent_type, model)
                for preferred in _LIVE_PROVIDER_PRIORITY
                for agent_type, model in agents_list[:4]
                if agent_type == preferred
            ),
            agents_list[0],
        )
        provider_configs.append(
            {
                "provider": selected_provider[0],
                "model": selected_provider[1],
                "api_key": None,
            }
        )

    team: list[dict[str, Any]] = []
    for index, (role, role_label) in enumerate(_LIVE_ROLES):
        provider_cfg = provider_configs[index % len(provider_configs)]
        provider_name = str(provider_cfg["provider"])
        team.append(
            {
                "provider": provider_name,
                "model": provider_cfg.get("model"),
                "api_key": provider_cfg.get("api_key"),
                "role": role,
                "name": f"{provider_name}-{role_label}",
            }
        )

    return team


def _extract_thinking_traces(result: Any) -> dict[str, str] | None:
    """Extract extended thinking traces from debate result metadata."""
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        traces = metadata.get("thinking_traces")
        if isinstance(traces, dict) and traces:
            return traces
    return None


def _clean_summary(text: str) -> str:
    """Strip LLM chain-of-thought preamble from the summary for clean CLI output."""
    import re

    # Strip common LLM preamble patterns (both at start and after markdown headers)
    preamble_patterns = [
        r"Okay,\s+I\s+will\s+.*?\n+",
        r"Here'?s?\s+(?:the|my)\s+(?:revised\s+)?.*?\n+",
        r"Let\s+me\s+.*?\n+",
        r"Sure[,!.].*?\n+",
        r"I'?ll\s+.*?\n+",
    ]
    cleaned = text.strip()
    for pattern in preamble_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # Collapse multiple blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip() or text.strip()


def _summarize_dissenting_views(
    dissenting_views: list[str], participants: list[str]
) -> list[dict[str, str]]:
    """Convert dissenting views into CLI-friendly agent/reason records."""
    dissent: list[dict[str, str]] = []
    fallback_agents = participants or ["agent"]
    for index, view in enumerate(dissenting_views):
        dissent.append(
            {
                "agent": fallback_agents[index % len(fallback_agents)],
                "reason": str(view),
            }
        )
    return dissent


def _build_live_receipt(
    result: Any,
    question: str,
    rounds: int,
    team: list[dict[str, Any]],
) -> dict[str, Any]:
    """Shape a live debate result into one deterministic receipt payload."""
    from aragora.gauntlet.receipt_models import ConsensusProof, DecisionReceipt, ProvenanceRecord

    participants = list(getattr(result, "participants", []) or [])
    if not participants:
        participants = [str(agent["name"]) for agent in team]

    final_answer = str(getattr(result, "final_answer", "") or "")
    confidence = _clamp_confidence(getattr(result, "confidence", 0.0))
    consensus_reached = bool(getattr(result, "consensus_reached", False))
    verdict = (
        "PASS"
        if consensus_reached and confidence >= 0.75
        else "CONDITIONAL"
        if confidence >= 0.45
        else "FAIL"
    )
    dissenting_views = [str(view) for view in list(getattr(result, "dissenting_views", []) or [])]
    dissent = _summarize_dissenting_views(dissenting_views, participants)
    receipt_id = str(
        getattr(result, "debate_id", "")
        or getattr(result, "id", "")
        or f"quickstart-{uuid.uuid4().hex[:12]}"
    )
    proposals = dict(getattr(result, "proposals", {}) or {})
    timestamp = datetime.now(timezone.utc).isoformat()
    input_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()

    supporting_agents: list[str] = []
    dissenting_agents: list[str] = []
    vote_records: list[dict[str, Any]] = []
    for vote in list(getattr(result, "votes", []) or []):
        vote_agent = str(getattr(vote, "agent", "") or "")
        vote_choice = str(getattr(vote, "choice", "") or "")
        vote_reasoning = str(getattr(vote, "reasoning", "") or "")
        if vote_agent or vote_choice or vote_reasoning:
            vote_records.append(
                {
                    "agent": vote_agent,
                    "choice": vote_choice,
                    "reasoning": vote_reasoning,
                }
            )
        if vote_choice == final_answer and vote_agent:
            supporting_agents.append(vote_agent)
        elif vote_agent:
            dissenting_agents.append(vote_agent)

    if not supporting_agents and consensus_reached:
        supporting_agents = participants[:]
        dissenting_agents = []

    rounds_used = int(getattr(result, "rounds_used", 0) or rounds)
    settlement_receipt_context = SimpleNamespace(
        consensus_proof=SimpleNamespace(dissenting_agents=dissenting_agents)
    )
    settlement_metadata = _build_quickstart_settlement_metadata(
        settlement_metadata=getattr(result, "settlement_metadata", None),
        debate_result=result,
        receipt_context=settlement_receipt_context,
        timestamp=timestamp,
    )
    receipt = DecisionReceipt(
        receipt_id=receipt_id,
        gauntlet_id=receipt_id,
        timestamp=timestamp,
        input_summary=question,
        input_hash=input_hash,
        risk_summary={
            "critical": 0 if consensus_reached else int(bool(dissenting_views)),
            "high": len(dissenting_agents),
            "medium": len(dissenting_views),
            "low": max(0, len(participants) - len(dissenting_views)),
        },
        attacks_attempted=rounds_used * max(1, len(participants)),
        attacks_successful=0 if consensus_reached else max(1, len(dissenting_views)),
        probes_run=len(vote_records),
        vulnerabilities_found=len(dissenting_views),
        verdict=verdict,
        confidence=confidence,
        robustness_score=confidence,
        verdict_reasoning=final_answer,
        dissenting_views=dissenting_views,
        consensus_proof=ConsensusProof(
            reached=consensus_reached,
            confidence=confidence,
            supporting_agents=supporting_agents,
            dissenting_agents=dissenting_agents,
            method="majority",
            evidence_hash=input_hash,
        ),
        provenance_chain=[
            ProvenanceRecord(
                timestamp=timestamp,
                event_type="task",
                description=question,
            ),
            ProvenanceRecord(
                timestamp=timestamp,
                event_type="verdict",
                description=final_answer or verdict,
            ),
        ],
        cost_summary={
            "cost_usd": float(getattr(result, "total_cost_usd", 0.0) or 0.0),
            "tokens_used": int(getattr(result, "total_tokens", 0) or 0),
        },
        thinking_traces=_extract_thinking_traces(result),
        settlement_metadata=settlement_metadata,
        config_used={
            "mode": "quickstart-live",
            "rounds": rounds_used,
            "participants": participants,
        },
    )

    payload = receipt.to_dict()
    payload.update(
        {
            "question": question,
            "rounds": rounds_used,
            "agents": participants,
            "summary": final_answer,
            "dissent": dissent,
            "mode": "live",
            "settlement_metadata": settlement_metadata,
            "receipt": {
                "id": receipt.receipt_id,
                "artifact_hash": receipt.artifact_hash,
                "consensus_reached": consensus_reached,
                "confidence": confidence,
                "participants": participants,
            },
            "proposals": proposals,
            "votes": vote_records,
            "consensus_reached": consensus_reached,
        }
    )
    return payload


async def _can_reach_provider_tls(provider: str) -> tuple[bool, str | None]:
    """Return whether the provider host passes a basic TLS handshake."""
    host = _PROVIDER_TLS_HOSTS.get(provider)
    if not host:
        return True, None

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                host,
                443,
                ssl=ssl.create_default_context(),
                server_hostname=host,
            ),
            timeout=_LIVE_PRECHECK_TIMEOUT_SECONDS,
        )
    except ssl.SSLCertVerificationError:
        return False, "CERTIFICATE_VERIFY_FAILED"
    except ssl.SSLError as exc:
        if _is_tls_verification_failure(exc):
            return False, "CERTIFICATE_VERIFY_FAILED"
        detail = str(exc).strip() or type(exc).__name__
        return False, detail
    except Exception as exc:
        if _is_tls_verification_failure(exc):
            return False, "CERTIFICATE_VERIFY_FAILED"
        detail = str(exc).strip() or type(exc).__name__
        return False, detail

    writer.close()
    try:
        await writer.wait_closed()
    except (ConnectionError, OSError, ssl.SSLError):
        # The verified handshake already succeeded; some transports raise while closing.
        pass
    return True, None


async def _filter_reachable_live_agents(
    agents_list: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    """Keep only providers that pass a fast TLS preflight.

    Fail closed before entering the debate engine when no detected live providers
    can establish a verified TLS connection.
    """
    limited_agents = agents_list[:4]
    probe_results = await asyncio.gather(
        *(_can_reach_provider_tls(provider) for provider, _ in limited_agents)
    )

    reachable: list[tuple[str, str | None]] = []
    failures: list[str] = []
    certificate_failure = False
    for agent_spec, (ok, detail) in zip(limited_agents, probe_results, strict=False):
        if ok:
            reachable.append(agent_spec)
            continue
        provider = agent_spec[0]
        if _is_tls_verification_failure(detail):
            certificate_failure = True
        failures.append(f"{provider}: {detail}")

    if reachable:
        return reachable

    # All providers failed TLS — return empty list instead of crashing.
    # The caller falls back to demo mode when no live agents are available.
    if certificate_failure:
        providers = ", ".join(provider for provider, _ in limited_agents)
        logger.warning(
            "Provider TLS verification failed for %s. Check the local CA trust store.", providers
        )
    else:
        failure_summary = "; ".join(failures) if failures else "no providers available"
        logger.warning("No live providers passed connectivity preflight: %s", failure_summary)

    return []


def cmd_quickstart(args: argparse.Namespace) -> None:
    """Handle the 'quickstart' command."""
    output_json = bool(getattr(args, "json", False))
    console: TextIO = sys.stderr if output_json else sys.stdout

    def emit(message: str = "") -> None:
        print(message, file=console)

    emit("\n" + "=" * 60)
    emit("  ARAGORA QUICKSTART")
    emit("  Zero-to-receipt adversarial debate")
    emit("=" * 60)

    # Step 1: Load .env
    loaded = _load_dotenv()
    if loaded:
        emit("\n[+] Loaded .env configuration")

    # Step 2: Get question
    question = _get_question(args, stream=console)
    if not question:
        emit("\nNo question provided. Exiting.")
        sys.exit(1)

    emit(f"\nQuestion: {question}")

    if getattr(args, "spec_first", False):
        requested_format = str(getattr(args, "format", "json") or "json").lower()
        if requested_format != "json":
            emit("\n[*] Spec-first quickstart always saves JSON artifacts; ignoring --format.")

        emit("\n[*] Run mode: spec-first")
        emit("    Pipeline: prompt -> spec -> saved artifact")

        start_time = time.monotonic()
        try:
            spec_result = _run_sync(_run_quickstart_spec_first(question))
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            emit(f"\n[!] Spec-first pipeline failed: {exc}")
            sys.exit(1)

        elapsed = time.monotonic() - start_time
        spec_payload = _build_quickstart_spec_payload(question, spec_result)
        saved_artifact = _save_quickstart_spec_payload(
            spec_payload,
            getattr(args, "output", None),
        )

        bundle = spec_payload.get("spec_bundle") if isinstance(spec_payload, dict) else None
        spec = spec_payload.get("specification") if isinstance(spec_payload, dict) else None
        title = ""
        problem = ""
        criteria: list[Any] = []
        risks: list[Any] = []
        run_id = str(spec_payload.get("run_id", "") or "")

        if isinstance(bundle, dict):
            title = str(bundle.get("title", "")).strip()
            problem = str(bundle.get("problem_statement", "")).strip()
            criteria = list(bundle.get("acceptance_criteria", []) or [])
            risks = list(bundle.get("rollback_plan", []) or [])
        elif isinstance(spec, dict):
            title = str(spec.get("title", "")).strip()
            problem = str(spec.get("problem_statement", "")).strip()
            criteria = list(spec.get("success_criteria", []) or [])
            risks = list(spec.get("risks", []) or [])

        emit("=" * 60)
        emit("  SPEC RESULT")
        emit("=" * 60)
        if title:
            emit(f"\n  Title:      {title}")
        if problem:
            emit(f"  Problem:    {problem[:300]}")
        emit(f"  Criteria:   {len(criteria)}")
        emit(f"  Risks:      {len(risks)}")
        emit(f"  Pipeline:   {spec_payload.get('pipeline', 'unknown')}")
        emit(f"  Elapsed:    {elapsed:.1f}s")
        if run_id:
            emit(f"  Run:        {run_id}")
        emit(f"\nSpec artifact (json): {saved_artifact}")

        if output_json:
            json.dump(
                _normalize_spec_json_result(spec_payload, saved_artifact), sys.stdout, indent=2
            )
            sys.stdout.write("\n")
            return

        emit("\nNext steps:")
        emit(f"  aragora decide '{question}' --spec {saved_artifact}")
        emit(f"  aragora spec '{question}' --output {saved_artifact}")
        emit("  aragora ask 'Your question'                    # Debate the approach")
        return

    # Step 3: Detect agents
    use_demo = getattr(args, "demo", False)
    rounds = _resolve_rounds(getattr(args, "rounds", None), use_demo=use_demo)
    provider = getattr(args, "provider", None)
    inline_api_key = getattr(args, "api_key", None)
    try:
        normalized_provider, saved_key = _configure_inline_api_key(
            provider,
            inline_api_key,
            save_key=getattr(args, "save_key", False),
        )
    except (RuntimeError, ValueError) as exc:
        emit(f"\n[!] Quickstart configuration failed: {exc}")
        sys.exit(1)

    if saved_key:
        emit(
            "\n[+] Saved "
            f"{saved_key['env_var']} to secure store ({saved_key['backend']}) as {saved_key['masked_value']}"
        )

    if use_demo:
        emit("\n[*] Run mode: demo (requested with --demo)")
        emit("    Agents: analyst (supportive), critic (critical), synthesizer (balanced)")
    else:
        try:
            detected = _detect_agents(normalized_provider)
        except ValueError as exc:
            emit(f"\n[!] Quickstart configuration failed: {exc}")
            sys.exit(1)
        if not detected:
            emit("\n[!] No supported API keys detected. Falling back to demo mode.")
            emit("    This run will use local mock agents, not live model calls.")
            emit("    Set ANTHROPIC_API_KEY or OPENAI_API_KEY for live debates.")
            emit("    Agents: analyst (supportive), critic (critical), synthesizer (balanced)")
            use_demo = True
        else:
            preview_team = _build_live_team(
                detected[:4],
                provider=normalized_provider,
                api_key=inline_api_key,
            )
            providers = list(dict.fromkeys(str(member["provider"]) for member in preview_team))
            emit("\n[+] Run mode: live")
            emit(f"    Agents: {', '.join(providers)}")

    emit(f"[*] Running {rounds}-round debate...\n")

    # Step 4: Run debate
    start_time = time.monotonic()
    try:
        if use_demo:
            result = _run_sync(_run_demo_debate(question, rounds))
        else:
            result = _run_sync(
                _run_live_debate(
                    question,
                    detected[:4],
                    rounds,
                    provider=normalized_provider,
                    api_key=inline_api_key,
                )
            )
    except (OSError, ConnectionError, RuntimeError, ValueError, TypeError) as e:
        logger.debug("Live debate failed, falling back to demo: %s", e)
        if _is_tls_verification_failure(e):
            emit("\n[!] Provider TLS check failed. Falling back to demo mode.")
            emit("    Check the local CA trust store for live debates.")
        elif "No live" in str(e) or "no live" in str(e):
            emit("\n[!] No live providers available. Falling back to demo mode.")
        else:
            emit(f"\n[!] Live debate failed: {e}")
            emit("    Falling back to demo mode.")
        # Fall back to demo — the user should always get a result
        try:
            result = _run_sync(_run_demo_debate(question, rounds))
        except (OSError, RuntimeError, ValueError) as demo_err:
            logger.debug("Demo debate also failed: %s", demo_err)
            emit(f"\n[!] Demo debate also failed: {demo_err}")
            sys.exit(1)

    elapsed = time.monotonic() - start_time
    result["elapsed_seconds"] = elapsed

    # Step 5: Display results
    emit("=" * 60)
    emit("  RESULT")
    emit("=" * 60)
    verdict_display = str(result["verdict"]).replace("_", " ").title()
    emit(f"\n  Verdict:    {verdict_display}")
    emit(f"  Confidence: {result['confidence']:.0%}")
    emit(f"  Mode:       {str(result.get('mode', 'demo')).title()}")
    emit(f"  Agents:     {', '.join(result['agents'])}")
    emit(f"  Rounds:     {result['rounds']}")
    emit(f"  Elapsed:    {elapsed:.1f}s")
    if "consensus_proof" in result:
        consensus_text = "Reached" if result["consensus_proof"].get("reached") else "Not reached"
        emit(f"  Consensus:  {consensus_text}")
    if result.get("receipt_id"):
        emit(f"  Receipt:    {result['receipt_id']}")
    if result.get("artifact_hash"):
        emit(f"  Artifact:   {str(result['artifact_hash'])[:16]}...")

    if result.get("summary"):
        summary_text = _clean_summary(result["summary"])
        if len(summary_text) > 800:
            cutoff = summary_text[:800].rfind(".")
            if cutoff > 200:
                summary_text = summary_text[: cutoff + 1] + "\n  [... truncated]"
            else:
                summary_text = summary_text[:800] + "..."
        emit(f"\n  Summary:\n  {summary_text}")

    if result.get("dissent"):
        emit("\n  Dissent:")
        for d in result["dissent"]:
            emit(f"    - {d.get('agent', '?')}: {d.get('reason', 'N/A')}")

    thinking = result.get("thinking_traces")
    if thinking:
        emit(f"\n  Thinking: {len(thinking)} agent(s) provided extended reasoning traces")

    emit("\n" + "=" * 60)

    # Step 6: Save receipt
    output_path = getattr(args, "output", None)
    fmt = getattr(args, "format", "json")
    canonical_result = _build_quickstart_receipt_payload(result)
    saved_artifact = _save_receipt(
        canonical_result,
        output_path or _default_receipt_path(str(result.get("mode", "demo")), fmt),
        fmt,
    )
    artifact_format = saved_artifact.suffix.lstrip(".") or fmt
    emit(f"\nResult artifact ({result.get('mode', 'demo')}/{artifact_format}): {saved_artifact}")

    # Persist to receipt store so API/dashboard/CLI-list can serve it
    try:
        from aragora.storage.receipt_store import get_receipt_store

        store = get_receipt_store()
        receipt_nested = canonical_result.get("receipt", {}) or {}
        store_payload = dict(canonical_result)
        store_payload.setdefault("receipt_id", receipt_nested.get("id", ""))
        store_payload.setdefault(
            "debate_id",
            str(canonical_result.get("debate_id") or canonical_result.get("receipt_id") or ""),
        )
        store_payload.setdefault("verdict", canonical_result.get("verdict", ""))
        store_payload.setdefault("checksum", canonical_result.get("artifact_hash", ""))
        store.save(store_payload)
        logger.info("receipt_persisted id=%s", store_payload.get("receipt_id", ""))
    except Exception:  # noqa: BLE001 - best-effort, local file is primary
        logger.debug("receipt_store_persist_skipped", exc_info=True)

    # Step 6b: Report KM ingestion status truthfully
    if result.get("mode") == "live":
        km_ingested = result.get("km_ingested", False)
        if km_ingested:
            emit("[+] Knowledge Mound: outcome ingested")
        else:
            emit("[*] Knowledge Mound: ingestion skipped (quickstart uses lightweight KM)")
            emit("    Use 'aragora ask' or 'aragora decide' for full KM writeback.")

    # Step 7: Open receipt in browser
    no_browser = getattr(args, "no_browser", False) or output_json
    if not no_browser:
        browser_path = _open_receipt_in_browser(
            result,
            saved_artifact if saved_artifact.suffix.lower() == ".html" else None,
        )
        if browser_path:
            if Path(browser_path) == saved_artifact:
                emit("\nOpened saved artifact in browser.")
            else:
                emit(f"\nOpened HTML preview in browser: {browser_path}")
        else:
            emit("\nCould not open browser. View the saved artifact directly.")

    if output_json:
        json.dump(
            _normalize_json_result(canonical_result, saved_artifact),
            sys.stdout,
            indent=2,
            default=str,
        )
        sys.stdout.write("\n")
        return

    emit("\nNext steps:")
    emit("  aragora ask 'Your question' --agents anthropic-api,openai-api  # Full debate")
    emit("  aragora decide 'Your question'                                  # Full pipeline")
    emit("  aragora doctor                                                  # System health")
