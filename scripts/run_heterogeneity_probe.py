#!/usr/bin/env python3
"""Run or validate the Round 30f heterogeneity contamination probe.

The default mode is dry-run validation: load the authored prompt set,
select the deterministic pilot subset, and print class counts. Passing
``--synthetic-fixture`` writes a deterministic fixture receipt for
tests and operator plumbing checks; it is explicitly marked as
synthetic and must not be interpreted as a live model result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aragora.heterogeneity.probe import (  # noqa: E402
    PanelistClassification,
    PromptProbeResult,
    build_probe_receipt,
)
from aragora.heterogeneity.prompts import (  # noqa: E402
    ProbePrompt,
    build_panel_prompt,
    load_prompt_set,
    select_pilot_prompts,
)
from aragora.heterogeneity.receipt import write_receipt  # noqa: E402
from aragora.swarm.multi_agent_dialog import (  # noqa: E402
    AgentSpec,
    DialogRound,
    run_round_and_persist,
)

DEFAULT_PROMPT_ROOT = Path("tests/heterogeneity/probe_prompts")
DEFAULT_OUTPUT_ROOT = Path(".aragora/heterogeneity/probes")
DEFAULT_PANEL_MODELS = (
    "claude-opus",
    "claude-sonnet",
    "codex",
    "droid-gpt5",
    "droid-gemini",
    "droid-kimi",
)
VALID_CLASSIFICATION_VERDICTS = frozenset(
    {
        "flagged_correctly",
        "partial_multi_seeded",
        "flagged_wrongly",
        "missed",
        "ambiguous",
        "dispatch_failed",
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-root", type=Path, default=DEFAULT_PROMPT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--all-prompts",
        action="store_true",
        help="Use all authored prompts instead of the deterministic pilot subset.",
    )
    parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help="Write a deterministic synthetic receipt for plumbing tests only.",
    )
    parser.add_argument(
        "--classifications-file",
        type=Path,
        default=None,
        help=(
            "JSON file with judged panelist classifications. Writes a real "
            "receipt from those classifications without dispatching models."
        ),
    )
    parser.add_argument(
        "--dispatch-live-transcripts",
        action="store_true",
        help=(
            "Dispatch selected prompts to the canonical heterogeneous panel and "
            "write transcripts. This does not judge responses or claim a verdict."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit selected prompts.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    return parser.parse_args(argv)


def _run_id(specified: str | None) -> str:
    if specified:
        return specified
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _select_prompts(args: argparse.Namespace) -> list[ProbePrompt]:
    prompts = load_prompt_set(args.prompt_root)
    selected = prompts if args.all_prompts else select_pilot_prompts(prompts)
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def _class_counts(prompts: Sequence[ProbePrompt]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for prompt in prompts:
        counts[prompt.prompt_class] = counts.get(prompt.prompt_class, 0) + 1
    return dict(sorted(counts.items()))


def _synthetic_results(prompts: Sequence[ProbePrompt]) -> list[PromptProbeResult]:
    results: list[PromptProbeResult] = []
    for prompt in prompts:
        classifications: list[PanelistClassification] = []
        expected_flags = prompt.expected_flags or 0
        for index, agent in enumerate(DEFAULT_PANEL_MODELS):
            if prompt.seeded_error is None:
                verdict = "missed"
            elif index < expected_flags:
                verdict = "flagged_correctly"
            else:
                verdict = "missed"
            classifications.append(
                PanelistClassification(
                    agent=agent,
                    verdict=verdict,  # type: ignore[arg-type]
                    rationale="synthetic fixture classification",
                )
            )
        results.append(PromptProbeResult.from_prompt(prompt, classifications))
    return results


def _results_from_classifications_file(
    prompts: Sequence[ProbePrompt],
    path: Path,
) -> tuple[list[PromptProbeResult], list[str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("classifications file must contain a JSON object")
    panel_models = payload.get("panel_models", DEFAULT_PANEL_MODELS)
    judge_model = payload.get("judge_model", "external-judge")
    raw_results = payload.get("results")
    if not isinstance(panel_models, list) or not all(
        isinstance(item, str) for item in panel_models
    ):
        raise ValueError("classifications file panel_models must be a list of strings")
    if not isinstance(judge_model, str) or not judge_model:
        raise ValueError("classifications file judge_model must be a non-empty string")
    if not isinstance(raw_results, list):
        raise ValueError("classifications file results must be a list")

    prompts_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    results: list[PromptProbeResult] = []
    seen_prompt_ids: set[str] = set()
    panel_model_set = set(panel_models)
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            raise ValueError("each classifications result must be an object")
        prompt_id = raw_result.get("prompt_id")
        if not isinstance(prompt_id, str) or prompt_id not in prompts_by_id:
            raise ValueError(f"unknown prompt_id in classifications file: {prompt_id!r}")
        if prompt_id in seen_prompt_ids:
            raise ValueError(f"duplicate prompt_id in classifications file: {prompt_id}")
        seen_prompt_ids.add(prompt_id)
        raw_classifications = raw_result.get("classifications")
        if not isinstance(raw_classifications, list):
            raise ValueError(f"{prompt_id}: classifications must be a list")
        classifications: list[PanelistClassification] = []
        seen_agents: set[str] = set()
        for raw_classification in raw_classifications:
            if not isinstance(raw_classification, dict):
                raise ValueError(f"{prompt_id}: classification entries must be objects")
            agent = raw_classification.get("agent")
            verdict = raw_classification.get("verdict")
            rationale = raw_classification.get("rationale", "")
            if not isinstance(agent, str) or not agent:
                raise ValueError(f"{prompt_id}: classification agent must be a non-empty string")
            if agent not in panel_model_set:
                raise ValueError(f"{prompt_id}: classification agent not in panel_models: {agent}")
            if agent in seen_agents:
                raise ValueError(f"{prompt_id}: duplicate classification agent: {agent}")
            seen_agents.add(agent)
            if not isinstance(verdict, str):
                raise ValueError(f"{prompt_id}: classification verdict must be a string")
            if verdict not in VALID_CLASSIFICATION_VERDICTS:
                raise ValueError(f"{prompt_id}: unknown classification verdict: {verdict}")
            if not isinstance(rationale, str):
                raise ValueError(f"{prompt_id}: classification rationale must be a string")
            classifications.append(
                PanelistClassification(
                    agent=agent,
                    verdict=verdict,  # type: ignore[arg-type]
                    rationale=rationale,
                )
            )
        missing_agents = [agent for agent in panel_models if agent not in seen_agents]
        if missing_agents:
            raise ValueError(
                f"{prompt_id}: missing classifications for panel_models: "
                + ", ".join(missing_agents)
            )
        results.append(PromptProbeResult.from_prompt(prompts_by_id[prompt_id], classifications))
    missing_prompt_ids = [
        prompt.prompt_id for prompt in prompts if prompt.prompt_id not in seen_prompt_ids
    ]
    if missing_prompt_ids:
        raise ValueError(
            "classifications file missing results for prompt_ids: " + ", ".join(missing_prompt_ids)
        )
    return results, panel_models, judge_model


async def _dispatch_live_transcripts(
    *,
    run_id: str,
    prompts: Sequence[ProbePrompt],
    output_root: Path,
) -> list[dict[str, object]]:
    agents = AgentSpec.heterogeneous_panel()
    transcript_dir = output_root / run_id / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    for prompt in prompts:
        round_ = DialogRound(
            round_id=f"{run_id}-{prompt.prompt_id}",
            prompt=build_panel_prompt(prompt),
            metadata={"prompt_id": prompt.prompt_id, "class": prompt.prompt_class},
        )
        jsonl_path, md_path, turns = await run_round_and_persist(round_, agents, transcript_dir)
        summary.append(
            {
                "prompt_id": prompt.prompt_id,
                "class": prompt.prompt_class,
                "jsonl": str(jsonl_path),
                "markdown": str(md_path),
                "successful": sum(1 for turn in turns if turn.succeeded()),
                "dispatched": len(turns),
            }
        )
    return summary


def _print_summary(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"run_id: {payload['run_id']}")
    print(f"prompt_count: {payload['prompt_count']}")
    print(f"class_counts: {payload['class_counts']}")
    if "receipt_path" in payload:
        print(f"receipt_path: {payload['receipt_path']}")
        print(f"receipt_verdict: {payload['receipt_verdict']}")
    if "transcripts" in payload:
        transcripts = payload["transcripts"]
        if not isinstance(transcripts, list):
            raise TypeError("transcripts payload must be a list")
        print(f"transcripts: {len(transcripts)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = _run_id(args.run_id)
    prompts = _select_prompts(args)
    payload: dict[str, object] = {
        "run_id": run_id,
        "prompt_count": len(prompts),
        "class_counts": _class_counts(prompts),
        "prompt_ids": [prompt.prompt_id for prompt in prompts],
    }

    if args.synthetic_fixture and args.classifications_file is not None:
        raise SystemExit("--synthetic-fixture and --classifications-file are mutually exclusive")

    if args.synthetic_fixture or args.classifications_file is not None:
        if args.classifications_file is not None:
            results, panel_models, judge_model = _results_from_classifications_file(
                prompts, args.classifications_file
            )
            caveats = ["receipt computed from external judged classifications file"]
        else:
            results = _synthetic_results(prompts)
            panel_models = list(DEFAULT_PANEL_MODELS)
            judge_model = "synthetic-fixture"
            caveats = [
                "synthetic fixture only; not a live model or judge result",
                "used for receipt plumbing and deterministic metric tests",
            ]
        receipt = build_probe_receipt(
            run_id=run_id,
            results=results,
            panel_models=panel_models,
            judge_model=judge_model,
            scope_caveats=caveats,
        )
        receipt_path = write_receipt(receipt, args.output_root / run_id)
        payload["receipt_path"] = str(receipt_path)
        payload["receipt_verdict"] = receipt["verdict"]
        payload["receipt_id"] = receipt["receipt_id"]

    if args.dispatch_live_transcripts:
        transcripts = asyncio.run(
            _dispatch_live_transcripts(run_id=run_id, prompts=prompts, output_root=args.output_root)
        )
        payload["transcripts"] = transcripts

    _print_summary(payload, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
