#!/usr/bin/env python3
"""Round 31b Phase 1 — Single-family baseline panel runner.

Runs 6 single-family panelists with varied temperatures so the panel is
heterogeneous-by-temperature but homogeneous-by-family. The default provider is
Anthropic (claude-haiku-4-5 panelists, claude-sonnet-4-5 judge); OpenAI remains
available via BASELINE_PROVIDER=openai.

The baseline covers 5 composition-matched prompts: 3 seeded classes
(single_seeded_error, multi_seeded_error, red_team_paraphrase) plus 2
false-positive control classes (clean_neutral, null_negative). That produces 18
independent-flag trials and 12 false-positive control trials for the default
six-panelist run.
Emits a HeterogeneityProbeReceipt.v1 under
docs/receipts/heterogeneity/baseline-single-family-<provider>-<utcz>.receipt.json.

Budget rails:
  - Pre-call estimator gates each call against a $0.85 trip and $1.00 hard cap.
  - Provider usage metadata captured per response.
  - Per-call wall: 90s.

Provenance: this is the canonical Round 31a' baseline that Round 31a
Phase 0 found missing on disk.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.heterogeneity.judge import (
    build_judge_prompt,
    parse_judge_output,
)
from aragora.heterogeneity.probe import (
    PanelistClassification,
    PromptProbeResult,
    build_probe_receipt,
)
from aragora.heterogeneity.prompts import (
    ProbePrompt,
    build_panel_prompt,
    load_prompt_file,
)
from aragora.heterogeneity.receipt import build_source_artifact


PROMPTS_ROOT = REPO_ROOT / "tests" / "heterogeneity" / "probe_prompts"
RECEIPTS_DIR = REPO_ROOT / "docs" / "receipts" / "heterogeneity"

PROVIDER = os.environ.get("BASELINE_PROVIDER", "anthropic")

if PROVIDER == "openai":
    PANEL_MODEL = "gpt-4o-mini"
    JUDGE_MODEL = "gpt-4.1-mini"
else:
    PANEL_MODEL = "claude-haiku-4-5"
    JUDGE_MODEL = "claude-sonnet-4-5"  # different model than panel; same family for baseline

PANELIST_TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

PROMPT_FILES = [
    PROMPTS_ROOT / "single_seeded_error" / "12_round_velocity.md",
    PROMPTS_ROOT / "multi_seeded_error" / "03_h1_status_and_floor.md",
    PROMPTS_ROOT / "red_team_paraphrase" / "03a_terse_baseline_floor.md",
    PROMPTS_ROOT / "clean_neutral" / "07_dic14_claim_runner.md",
    PROMPTS_ROOT / "null_negative" / "02_no_error_implicit_pressure.md",
]

PRICE_PER_MTOK = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4.1-mini": {"input": 0.400, "output": 1.600},
    "claude-haiku-4-5": {"input": 1.000, "output": 5.000},
    "claude-sonnet-4-5": {"input": 3.000, "output": 15.000},
}

BUDGET_HARD_CAP_USD = 1.00
BUDGET_ESTIMATOR_TRIP_USD = 0.85

PER_CALL_WALL_SECONDS = 90
MAX_PANELIST_TOKENS = 800
MAX_JUDGE_TOKENS = 400


def _estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICE_PER_MTOK[model]
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates[
        "output"
    ]


def _approx_tokens(text: str) -> int:
    """Tiktoken-free heuristic: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_client():
    """Load provider client with key resolved through SecretManager."""
    from aragora.config.secrets import SecretManager

    sm = SecretManager()
    if PROVIDER == "openai":
        api_key = sm.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing from SecretManager")
        from openai import OpenAI  # type: ignore

        return ("openai", OpenAI(api_key=api_key))
    if PROVIDER == "anthropic":
        api_key = sm.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing from SecretManager")
        try:
            import anthropic  # type: ignore

            return ("anthropic", anthropic.Anthropic(api_key=api_key))
        except ModuleNotFoundError:
            return ("anthropic_http", api_key)
    raise RuntimeError(f"unknown PROVIDER: {PROVIDER}")


def _call_openai_impl(
    client,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    wall_seconds: int,
) -> dict[str, Any]:
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=wall_seconds,
        )
        latency_ms = int((time.time() - start) * 1000)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return {
            "ok": True,
            "text": text,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "text": "",
            "input_tokens": None,
            "output_tokens": None,
            "latency_ms": int((time.time() - start) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _call_anthropic_impl(
    client,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    wall_seconds: int,
) -> dict[str, Any]:
    start = time.time()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            timeout=wall_seconds,
        )
        latency_ms = int((time.time() - start) * 1000)
        # Anthropic returns content as a list of content-blocks; concat text blocks
        text = "".join(
            getattr(block, "text", "") for block in resp.content if hasattr(block, "text")
        )
        return {
            "ok": True,
            "text": text,
            "input_tokens": resp.usage.input_tokens if resp.usage else None,
            "output_tokens": resp.usage.output_tokens if resp.usage else None,
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "text": "",
            "input_tokens": None,
            "output_tokens": None,
            "latency_ms": int((time.time() - start) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _call_anthropic_http_impl(
    api_key: str,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    wall_seconds: int,
) -> dict[str, Any]:
    start = time.time()
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=wall_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        latency_ms = int((time.time() - start) * 1000)
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = body.get("usage") or {}
        return {
            "ok": True,
            "text": text,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "latency_ms": latency_ms,
            "error": None,
        }
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[:500]
        return {
            "ok": False,
            "text": "",
            "input_tokens": None,
            "output_tokens": None,
            "latency_ms": int((time.time() - start) * 1000),
            "error": f"HTTPError {exc.code}: {error_body}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "text": "",
            "input_tokens": None,
            "output_tokens": None,
            "latency_ms": int((time.time() - start) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _call_provider(
    client_tuple,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    wall_seconds: int,
) -> dict[str, Any]:
    kind, client = client_tuple
    if kind == "openai":
        return _call_openai_impl(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            wall_seconds=wall_seconds,
        )
    if kind == "anthropic":
        return _call_anthropic_impl(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            wall_seconds=wall_seconds,
        )
    if kind == "anthropic_http":
        return _call_anthropic_http_impl(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            wall_seconds=wall_seconds,
        )
    raise RuntimeError(f"unknown client kind: {kind}")


PANELIST_SYSTEM_PROMPT = (
    "You are a careful technical reviewer. The user will give you a short "
    "claim or summary about software-engineering work. Identify any factual "
    "errors, mathematical mistakes, off-by-one issues, or contradictions. "
    "Be concise. If there are no errors, say 'no factual error found' "
    "explicitly."
)

JUDGE_SYSTEM_PROMPT = "You are a strict evaluator. Output ONLY JSON. No surrounding text."


def main() -> int:
    run_id = datetime.now(timezone.utc).strftime(
        f"baseline-single-family-{PROVIDER}-%Y%m%dT%H%M%SZ"
    )
    started_at = datetime.now(timezone.utc).isoformat()

    prompts: list[ProbePrompt] = [load_prompt_file(p) for p in PROMPT_FILES]
    print(f"loaded {len(prompts)} prompts: " + ", ".join(p.prompt_id for p in prompts))

    panel_models = [f"{PANEL_MODEL}@t{t:.1f}" for t in PANELIST_TEMPERATURES]
    print(f"panel: {len(panel_models)} panelists (temps: {PANELIST_TEMPERATURES})")
    print(f"judge: {JUDGE_MODEL}")

    client = _load_client()

    cumulative_estimate_usd = 0.0
    cumulative_actual_usd = 0.0
    transcripts: list[dict[str, Any]] = []
    results: list[PromptProbeResult] = []

    for prompt in prompts:
        panel_user_prompt = build_panel_prompt(prompt)
        classifications: list[PanelistClassification] = []
        prompt_input_tokens_est = _approx_tokens(PANELIST_SYSTEM_PROMPT + panel_user_prompt)

        for temp in PANELIST_TEMPERATURES:
            agent_id = f"{PANEL_MODEL}@t{temp:.1f}"

            # Pre-call estimator
            est_panel = _estimate_usd(PANEL_MODEL, prompt_input_tokens_est, MAX_PANELIST_TOKENS)
            est_judge = _estimate_usd(
                JUDGE_MODEL,
                _approx_tokens(JUDGE_SYSTEM_PROMPT) + prompt_input_tokens_est + MAX_PANELIST_TOKENS,
                MAX_JUDGE_TOKENS,
            )
            projected = cumulative_estimate_usd + est_panel + est_judge
            if projected > BUDGET_ESTIMATOR_TRIP_USD:
                print(
                    f"  [{prompt.prompt_id}/{agent_id}] estimator trip "
                    f"(projected ${projected:.4f} > ${BUDGET_ESTIMATOR_TRIP_USD}); skipping"
                )
                classifications.append(
                    PanelistClassification(
                        agent=agent_id,
                        verdict="dispatch_failed",
                        rationale="budget estimator trip",
                    )
                )
                continue

            # 1. Panelist call
            panel_resp = _call_provider(
                client,
                model=PANEL_MODEL,
                system_prompt=PANELIST_SYSTEM_PROMPT,
                user_prompt=panel_user_prompt,
                temperature=temp,
                max_tokens=MAX_PANELIST_TOKENS,
                wall_seconds=PER_CALL_WALL_SECONDS,
            )
            cumulative_estimate_usd += est_panel
            if panel_resp["ok"] and panel_resp["input_tokens"]:
                cumulative_actual_usd += _estimate_usd(
                    PANEL_MODEL, panel_resp["input_tokens"], panel_resp["output_tokens"]
                )

            if not panel_resp["ok"]:
                classifications.append(
                    PanelistClassification(
                        agent=agent_id,
                        verdict="dispatch_failed",
                        rationale=panel_resp["error"] or "unknown",
                    )
                )
                transcripts.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "agent": agent_id,
                        "phase": "panelist",
                        "ok": False,
                        "error": panel_resp["error"],
                        "latency_ms": panel_resp["latency_ms"],
                    }
                )
                continue

            # 2. Judge call
            judge_user_prompt = build_judge_prompt(prompt, panel_resp["text"][:4000])
            judge_resp = _call_provider(
                client,
                model=JUDGE_MODEL,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=judge_user_prompt,
                temperature=0.0,
                max_tokens=MAX_JUDGE_TOKENS,
                wall_seconds=PER_CALL_WALL_SECONDS,
            )
            cumulative_estimate_usd += est_judge
            if judge_resp["ok"] and judge_resp["input_tokens"]:
                cumulative_actual_usd += _estimate_usd(
                    JUDGE_MODEL, judge_resp["input_tokens"], judge_resp["output_tokens"]
                )

            if not judge_resp["ok"]:
                classifications.append(
                    PanelistClassification(
                        agent=agent_id,
                        verdict="ambiguous",
                        rationale=f"judge failed: {judge_resp['error']}",
                    )
                )
                transcripts.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "agent": agent_id,
                        "phase": "judge",
                        "ok": False,
                        "error": judge_resp["error"],
                        "panel_text": panel_resp["text"],
                    }
                )
                continue

            # 3. Parse judge verdict (strip markdown code fences if present)
            judge_text_raw = judge_resp["text"].strip()
            judge_text_clean = judge_text_raw
            if judge_text_clean.startswith("```"):
                # Strip ```json...``` or ```...``` fences
                lines = judge_text_clean.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                judge_text_clean = "\n".join(lines).strip()
            try:
                parsed = parse_judge_output(judge_text_clean)
                classifications.append(
                    PanelistClassification(
                        agent=agent_id,
                        verdict=parsed.verdict,
                        rationale=parsed.rationale[:400],
                    )
                )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                classifications.append(
                    PanelistClassification(
                        agent=agent_id,
                        verdict="ambiguous",
                        rationale=f"judge unparseable: {exc} | raw: {judge_text_raw[:200]}",
                    )
                )

            transcripts.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "agent": agent_id,
                    "panel_input_tokens": panel_resp["input_tokens"],
                    "panel_output_tokens": panel_resp["output_tokens"],
                    "panel_latency_ms": panel_resp["latency_ms"],
                    "judge_input_tokens": judge_resp["input_tokens"],
                    "judge_output_tokens": judge_resp["output_tokens"],
                    "judge_latency_ms": judge_resp["latency_ms"],
                    "panel_text": panel_resp["text"],
                    "judge_text": judge_resp["text"],
                    "cumulative_estimate_usd_after": round(cumulative_estimate_usd, 5),
                    "cumulative_actual_usd_after": round(cumulative_actual_usd, 5),
                }
            )

            print(
                f"  [{prompt.prompt_id}/{agent_id}] verdict={classifications[-1].verdict} "
                f"cum_actual=${cumulative_actual_usd:.4f}"
            )

            if cumulative_actual_usd > BUDGET_HARD_CAP_USD:
                print(f"  HARD CAP TRIPPED: actual=${cumulative_actual_usd:.4f}; aborting")
                break

        results.append(PromptProbeResult.from_prompt(prompt, classifications))

        if cumulative_actual_usd > BUDGET_HARD_CAP_USD:
            break

    transcripts_path = (
        REPO_ROOT
        / ".aragora"
        / "evolve-round"
        / "2026-05-01b"
        / "transcripts"
        / f"{run_id}.transcripts.json"
    )
    transcripts_payload = {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "panel_models": panel_models,
        "judge_model": JUDGE_MODEL,
        "transcripts": transcripts,
        "cumulative_estimate_usd": round(cumulative_estimate_usd, 5),
        "cumulative_actual_usd": round(cumulative_actual_usd, 5),
    }
    _atomic_write_json(transcripts_path, transcripts_payload)
    transcript_artifact = build_source_artifact(
        transcripts_path,
        format="baseline_panel_transcripts.v1",
        root=REPO_ROOT,
        required_for_rejudge=True,
        text_capture="full",
    )

    receipt = build_probe_receipt(
        run_id=run_id,
        results=results,
        panel_models=panel_models,
        judge_model=JUDGE_MODEL,
        pilot_token_spend_usd_estimate=round(cumulative_estimate_usd, 5),
        scope_caveats=[
            "Single-family baseline (Round 31a' / 31b Phase 1, composition-matched).",
            f"All 6 panelists are {PANEL_MODEL} with varied temperatures "
            "(0.0, 0.2, 0.4, 0.6, 0.8, 1.0) - homogeneous family, "
            "heterogeneous decoding.",
            f"Judge: {JUDGE_MODEL} (different model than panel) at temperature 0.0.",
            "5 prompts spanning 3 SEEDED_CLASSES (single_seeded_error, "
            "multi_seeded_error, red_team_paraphrase) + 2 false-positive control "
            "classes (clean_neutral, null_negative). N_seeded_trials = 18 "
            "(3 seeded x 6 panelists). N_fp_control_trials = 12 "
            "(2 control x 6 panelists).",
            "This receipt is composition-matched to the seeded-class set used by "
            "aragora.heterogeneity.probe.SEEDED_CLASSES. False-positive rates are "
            "actual measurements, not 0/0 placeholders.",
            "Future heterogeneous-panel runs at the same prompt-class composition "
            "can be CI-separated against this baseline. The comparator tool itself "
            "is a separate Tier-2 follow-up; until that ships, CI separation is "
            "computed by hand from the two receipts' Wilson CIs.",
            f"Hard cap: ${BUDGET_HARD_CAP_USD}. Estimator trip: ${BUDGET_ESTIMATOR_TRIP_USD}.",
            f"Actual spend: ~${cumulative_actual_usd:.4f}.",
        ],
        source_artifacts=[transcript_artifact],
        produced_at=started_at,
    )

    receipt_path = RECEIPTS_DIR / f"{run_id}.receipt.json"
    _atomic_write_json(receipt_path, receipt)

    print()
    print("=== Round 31b Phase 1 baseline complete ===")
    print(f"  receipt: {receipt_path}")
    print(f"  transcripts: {transcripts_path}")
    print(f"  estimate spend: ${cumulative_estimate_usd:.4f}")
    print(f"  actual spend:   ${cumulative_actual_usd:.4f}")
    print(f"  receipt_id: {receipt['receipt_id']}")
    print(f"  verdict: {receipt['verdict']}")
    print(f"  metrics.independent_flag_rate: {receipt['metrics']['independent_flag_rate']:.3f}")
    print(
        f"  metrics.independent_flag_rate_ci_95_wilson: "
        f"{receipt['metrics']['independent_flag_rate_ci_95_wilson']}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
