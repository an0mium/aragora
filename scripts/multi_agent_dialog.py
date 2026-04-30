#!/usr/bin/env python3
"""CLI entry point for the cross-agent dialog harness.

Dispatch a single prompt to multiple agents (claude, codex, droid) in
parallel, persist the round transcript as JSONL + markdown, and exit.

This is the operator-facing surface of
``aragora.swarm.multi_agent_dialog``. See that module's docstring for
the rationale behind each agent's invocation flags.

Usage::

    # Single-prompt round, all three agents:
    python3 scripts/multi_agent_dialog.py \\
        --round-id phase-d-pr6839 \\
        --prompt-file prompts/review.md \\
        --output-dir .aragora/evolve-round/2026-04-30d/dogfood/

    # With extra context (e.g. source file):
    python3 scripts/multi_agent_dialog.py \\
        --round-id sample \\
        --prompt 'Review this code for bugs.' \\
        --context-file aragora/swarm/dispatch_evidence.py \\
        --output-dir /tmp/dialog/

    # Subset of agents:
    python3 scripts/multi_agent_dialog.py \\
        --round-id sample \\
        --prompt 'Hello' \\
        --agents claude,codex \\
        --output-dir /tmp/dialog/
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Callable

# Allow running this script before the package is installed.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aragora.swarm.multi_agent_dialog import (  # noqa: E402
    AgentSpec,
    DEFAULT_CLAUDE_TIMEOUT,
    DEFAULT_CODEX_TIMEOUT,
    DEFAULT_DROID_TIMEOUT,
    DEFAULT_MODEL_TIMEOUT,
    DialogRound,
    run_round_and_persist,
)


AGENT_FACTORIES: dict[str, Callable[..., AgentSpec]] = {
    "claude": AgentSpec.claude,
    "codex": AgentSpec.codex,
    "droid": AgentSpec.droid,
    # Heterogeneous model factories (round 30e Phase B):
    "claude-opus": AgentSpec.claude_opus,
    "claude-sonnet": AgentSpec.claude_sonnet,
    "droid-gpt5": AgentSpec.droid_gpt5,
    "droid-gemini": AgentSpec.droid_gemini,
    "droid-kimi": AgentSpec.droid_kimi,
    "droid-glm": AgentSpec.droid_glm,
}

# Agent groups for the ``--agents-spec`` shorthand. Keys are validated
# at parse time; values are tuples of factory keys above.
AGENT_GROUPS: dict[str, tuple[str, ...]] = {
    "default": ("claude", "codex", "droid"),
    "heterogeneous": (
        "claude-opus",
        "claude-sonnet",
        "codex",
        "droid-gpt5",
        "droid-gemini",
        "droid-kimi",
    ),
    "anthropic-only": ("claude-opus", "claude-sonnet"),
    "frontier-chinese": ("droid-kimi", "droid-glm"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round-id",
        required=True,
        help="Short identifier used as the JSONL/markdown filename suffix",
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Inline prompt string")
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a file whose contents are used as the prompt",
    )

    parser.add_argument(
        "--context-file",
        type=Path,
        default=None,
        help="Optional file appended verbatim under an 'Extra context' "
        "section in the rendered prompt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write dialog-<round-id>.{jsonl,md}",
    )
    parser.add_argument(
        "--agents",
        default=None,
        help="Comma-separated agent names. Available: " + ",".join(sorted(AGENT_FACTORIES)),
    )
    parser.add_argument(
        "--agents-spec",
        default=None,
        choices=sorted(AGENT_GROUPS),
        help=(
            "Pre-built agent group: "
            + ", ".join(f"{k}=({','.join(v)})" for k, v in AGENT_GROUPS.items())
        ),
    )
    parser.add_argument(
        "--claude-timeout",
        type=int,
        default=DEFAULT_CLAUDE_TIMEOUT,
        help="Per-agent timeout (s)",
    )
    parser.add_argument(
        "--codex-timeout",
        type=int,
        default=DEFAULT_CODEX_TIMEOUT,
        help="Per-agent timeout (s)",
    )
    parser.add_argument(
        "--droid-timeout",
        type=int,
        default=DEFAULT_DROID_TIMEOUT,
        help="Per-agent timeout (s)",
    )
    parser.add_argument(
        "--model-timeout",
        type=int,
        default=DEFAULT_MODEL_TIMEOUT,
        help="Per-agent timeout (s) for heterogeneous-model factories.",
    )
    return parser.parse_args(argv)


def _build_agents(spec: argparse.Namespace) -> list[AgentSpec]:
    # Resolve agent name list. Precedence:
    #  1. Explicit --agents wins.
    #  2. --agents-spec resolves to its group.
    #  3. Default to the legacy 3-CLI panel.
    if spec.agents:
        names = [name.strip() for name in spec.agents.split(",") if name.strip()]
    elif spec.agents_spec:
        names = list(AGENT_GROUPS[spec.agents_spec])
    else:
        names = list(AGENT_GROUPS["default"])
    out: list[AgentSpec] = []
    legacy_timeouts = {
        "claude": spec.claude_timeout,
        "codex": spec.codex_timeout,
        "droid": spec.droid_timeout,
    }
    for name in names:
        factory = AGENT_FACTORIES.get(name)
        if factory is None:
            raise SystemExit(
                f"unknown agent: {name!r}. Available: " + ",".join(sorted(AGENT_FACTORIES))
            )
        # Legacy 3-CLI factories have explicit per-name timeouts; the
        # heterogeneous factories share ``--model-timeout``.
        timeout = legacy_timeouts.get(name, spec.model_timeout)
        out.append(factory(timeout_seconds=timeout))
    return out


def _resolve_prompt(spec: argparse.Namespace) -> str:
    if spec.prompt is not None:
        return spec.prompt
    return spec.prompt_file.read_text(encoding="utf-8")


def _resolve_context(spec: argparse.Namespace) -> str:
    if spec.context_file is None:
        return ""
    return spec.context_file.read_text(encoding="utf-8")


async def _run(spec: argparse.Namespace) -> int:
    agents = _build_agents(spec)
    round_ = DialogRound(
        round_id=spec.round_id,
        prompt=_resolve_prompt(spec),
        extra_context=_resolve_context(spec),
    )
    jsonl_path, md_path, turns = await run_round_and_persist(round_, agents, spec.output_dir)
    print(f"jsonl: {jsonl_path}")
    print(f"markdown: {md_path}")
    print(f"agents dispatched: {len(turns)}")
    print(f"successful: {sum(1 for t in turns if t.succeeded())}")
    return 0


def main(argv: list[str] | None = None) -> int:
    spec = parse_args(argv)
    return asyncio.run(_run(spec))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
